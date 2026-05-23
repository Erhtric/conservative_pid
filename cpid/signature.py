from orderly_set import OrderedSet, StableSet
import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from .io import CausalQuery, CausalExpression
import networkx as nx


def _compute_function_index(
    parent_vals: list[int], parent_names: list[str], domains: dict[str, int]
) -> int:
    """Converts a tuple of parent values into a flat list index.

    This utility function computes the index into a function lookup table given a
    tuple of parent values. Uses big-endian ordering (rightmost variable is least significant).

    Args:
        parent_vals: A list of parent variable values (e.g., [0, 1]).
        parent_names: A list of parent variable names in the same order as parent_vals (e.g., ['X', 'Z']).
        domains: Dict mapping variable names to their domain sizes.

    Returns:
        An integer index corresponding to the combination of parent values.

    Example:
        For parent_names = ['X', 'Z'] and domains = {'X': 2, 'Z': 2}:
        - parent_vals = [0, 0] -> index 0
        - parent_vals = [0, 1] -> index 1
        - parent_vals = [1, 0] -> index 2
        - parent_vals = [1, 1] -> index 3
    """
    idx = 0
    multiplier = 1
    for p_val, p_name in zip(reversed(parent_vals), reversed(parent_names)):
        idx += p_val * multiplier
        multiplier *= domains[p_name]
    return idx


class SignatureQueryEvaluator:
    """
    Evaluates whether signature rows satisfy causal queries and observational constraints.

    This class provides methods to:
    - Evaluate the state of all variables under a given signature row and interventions
    - Check if rows satisfy evidence constraints
    - Filter rows by observational value tuples (natural state matching)
    - Filter rows by causal query satisfaction (counterfactuals + evidence)
    """

    def __init__(
        self,
        domains: dict[str, int],
        signature_obj: "ResponseSignature",
        query: CausalQuery | CausalExpression | None = None,
    ):
        """Initialize the evaluator.

        Args:
            domains: Dict mapping variable names to their number of discrete values.
            signature_obj: A ResponseSignature object defining the signature space and structure.
            query: Optional CausalQuery or CausalExpression to evaluate. If None, evaluator
                   can still be used for state evaluation and evidence checking.

        Raises:
            ValueError: If the signature is not compatible with the query (when query is provided).
        """
        self.domains = domains
        self.signature = signature_obj
        self.query = query

        if query is not None:
            try:
                self.signature.is_compatible(self.query)
            except ValueError as e:
                raise ValueError(f"Signature is not compatible with query: {e}")

    def evaluate_state(
        self,
        row: tuple[tuple[int, ...], ...],
        interventions: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """Evaluate the state of all variables under a signature row and interventions.

        Flows through the topological order, computing each variable's value either
        from its function (if not intervened) or from the intervention value. It
        equals to the iterative application of the composition axiom for the given signature row.

        Args:
            row: A list of tuples representing the function outputs for each node.
            interventions: Optional dict of interventions to apply (e.g., {'X': 1}).
                If None, uses empty dict (pure observational state).

        Returns:
            dict[str, int]: The resulting state of all variables after applying interventions.
        """
        if interventions is None:
            interventions = {}

        state = {}
        for i, node in enumerate(self.signature.ordered_nodes):
            if node in interventions:
                state[node] = interventions[node]
            else:
                parents = self.signature.structure[node]
                parent_vals = [state[p] for p in parents]
                func_idx = _compute_function_index(parent_vals, parents, self.domains)
                state[node] = row[i][func_idx]
        return state

    def _check_evidence(self, sig_row: tuple[tuple[int, ...], ...]) -> bool:
        """Check if a row satisfies the evidence constraints of the query.

        Args:
            sig_row: A signature row.

        Returns:
            bool: True if there is no evidence, or if the natural state matches all evidence constraints.
        """
        if self.query is None or not hasattr(self.query, "evidence"):
            return True

        if not self.query.evidence:
            return True

        nat_state = self.evaluate_state(sig_row, interventions={})
        return all(
            nat_state[e_var] == e_val for e_var, e_val in self.query.evidence.items()
        )

    def _check_counterfactuals(self, sig_row: tuple[tuple[int, ...], ...]) -> bool:
        """Check if a row satisfies all counterfactual constraints of the query.

        Args:
            sig_row: A signature row.

        Returns:
            bool: True if all counterfactuals are satisfied.

        Raises:
            ValueError: If nested interventions are detected (query must be unnested first).
        """
        if self.query is None or not hasattr(self.query, "counterfactuals"):
            return True

        for atomic in self.query.counterfactuals:
            # Validate interventions are unnested (all ints, not nested structures)
            for iv in atomic.interventions.values():
                if not isinstance(iv, int):
                    raise ValueError(
                        "Nested interventions detected during evaluation. "
                        "Call `CausalQuery.unnest(domains)` before evaluating or solving LPs."
                    )

            int_state = self.evaluate_state(sig_row, interventions=atomic.interventions)
            if int_state[atomic.target_var] != atomic.target_val:
                return False

        return True

    def _filter_rows_by_predicate(
        self, predicate: Callable[[tuple[tuple[int, ...], ...]], bool]
    ) -> list[int]:
        """Filter signature rows using a custom predicate function.

        This is the unified filtering mechanism used by specific query methods.
        Works with both materialized and lazy signatures.

        Args:
            predicate: A function that takes a signature row and returns True if it should be included.

        Returns:
            list[int]: Indices of rows that satisfy the predicate.
        """
        matching_indices = []
        for i, sig in enumerate(self.signature._iterate_space()):
            if predicate(sig):
                matching_indices.append(i)
        return matching_indices

    def get_matching_row_indices(self, obs_vals: tuple[int]) -> list[int]:
        """Return indices of signature rows with matching natural state (observational constraint).

        Given observational values like (0, 0, 1) = P(X=0, W=0, Y=1), returns the indices
        of all signature rows whose natural state (without interventions) matches these values.

        Args:
            obs_vals: A tuple of observed values (e.g., (0, 1, 0)).

        Returns:
            list[int]: Indices of signature rows whose natural state matches obs_vals.
        """

        def matches_obs(sig: tuple[tuple[int, ...], ...]) -> bool:
            nat_state = self.evaluate_state(sig, interventions={})
            nat_tuple = tuple(nat_state[v] for v in self.signature.ordered_nodes)
            return nat_tuple == obs_vals

        return self._filter_rows_by_predicate(matches_obs)

    def get_satisfying_row_indices(self, query_term: CausalQuery) -> list[int]:
        """Return indices of signature rows that satisfy a CausalQuery.

        A row satisfies a query if:
        1. The natural state satisfies all evidence constraints (if any)
        2. All counterfactual constraints are satisfied

        Args:
            query_term: A CausalQuery to check against each row.

        Returns:
            list[int]: Indices of signature rows satisfying the query_term.

        Raises:
            ValueError: If nested interventions are detected (query must be unnested first).
        """

        def satisfies_query(sig: tuple[tuple[int, ...], ...]) -> bool:
            # Check evidence first (fail fast)
            if query_term.evidence:
                nat_state = self.evaluate_state(sig, interventions={})
                if not all(
                    nat_state[e_var] == e_val
                    for e_var, e_val in query_term.evidence.items()
                ):
                    return False

            # Check all counterfactuals (fail fast on first mismatch)
            for atomic in query_term.counterfactuals:
                # Validate interventions are unnested
                for iv in atomic.interventions.values():
                    if not isinstance(iv, int):
                        raise ValueError(
                            "Nested interventions detected during evaluation. "
                            "Call `CausalQuery.unnest(domains)` before evaluating or solving LPs."
                        )

                int_state = self.evaluate_state(sig, interventions=atomic.interventions)
                if int_state[atomic.target_var] != atomic.target_val:
                    return False

            return True

        return self._filter_rows_by_predicate(satisfies_query)

    def row_satisfies_query(self, sig_row: tuple[tuple[int, ...], ...]) -> bool:
        """Check if a specific signature row satisfies the entire CausalQuery.

        Returns True if:
        1. Natural state satisfies all evidence constraints (if any)
        2. All counterfactuals are satisfied

        Args:
            sig_row: A signature row.

        Returns:
            bool: True if the row satisfies the query, False otherwise.

        Raises:
            ValueError: If query is None or not a CausalQuery, or if nested interventions detected.
            TypeError: If query is not a CausalQuery.
        """
        if self.query is None:
            raise ValueError("No query provided to SignatureQueryEvaluator.")
        if not isinstance(self.query, CausalQuery):
            raise TypeError(
                "row_satisfies_query expects a CausalQuery, not CausalExpression."
            )

        if not self._check_evidence(sig_row):
            return False

        return self._check_counterfactuals(sig_row)


class ResponseSignature(ABC):
    """
    Abstract base class for all response signatures.

    Supports two modes of operation:
    - Materialized mode (lazy=False, default): Entire signature space is materialized into self.space list.
    - Lazy mode (lazy=True): Space is never materialized; only size is computed symbolically.
      WARNING: In lazy mode, direct access to self.space will be None; use iter_space() instead.
    """

    def __init__(self, domains: dict[str, int], lazy: bool = False):
        self.domains: dict[str, int] = domains
        self.ordered_nodes: list[str] = []
        self.structure: dict[str, list[str]] = {}
        self.space: list[tuple[int]] | None = [] if not lazy else None
        self.lazy: bool = lazy
        self._cached_size: int | None = None

    @abstractmethod
    def _build_structure(self):
        """Must be implemented by subclasses to define the topological parent mapping."""
        pass

    @staticmethod
    def _compute_space_size(
        domains: dict[str, int], structure: dict[str, list[str]]
    ) -> int:
        """Compute the total signature space size symbolically without materialization.

        For each node, the number of distinct functions from its parents to itself is:
            |D(node)|^(|D(parent1)| * |D(parent2)| * ... * |D(parentN)|)

        Total space size is the product over all nodes.

        Args:
            domains: Dict mapping variable names to their domain sizes.
            structure: Dict mapping each variable to its list of parent variables.

        Returns:
            int: Total number of signatures in the space.

        Example:
            For 3 binary variables in total order [X, M, Y]:
            - X has no parents: |D(X)|^1 = 2^1 = 2 functions
            - M has parent X: |D(M)|^|D(X)| = 2^2 = 4 functions
            - Y has parents X, M: |D(Y)|^(|D(X)|*|D(M)|) = 2^(2*2) = 2^4 = 16 functions
            - Total: 2 * 4 * 16 = 128
        """
        total_size = 1
        for node in structure.keys():
            parents = structure[node]
            if not parents:
                # No parents: one function per domain value
                num_functions = domains[node]
            else:
                # Number of input configurations (Cartesian product of parent domains)
                input_space_size = math.prod(domains[p] for p in parents)
                # Number of functions from inputs to this node's domain
                num_functions = domains[node] ** input_space_size
            total_size *= num_functions
        return total_size

    def _generate_space(self):
        """Generates the Cartesian product of functional mappings based on the structure.

        If lazy=True, skips materialization and caches size for symbolic computation.
        If lazy=False, materializes entire space into self.space list.
        """
        if not self.lazy:
            self.space = list(self.iter_space())
        # Cache size for later retrieval
        self._cached_size = self._compute_space_size(self.domains, self.structure)

    def iter_space(self):
        """Lazily iterate over the full signature space."""
        funcs_per_node = []
        for node in self.ordered_nodes:
            parents = self.structure[node]
            input_space_size = (
                math.prod([self.domains[p] for p in parents]) if parents else 1
            )
            node_funcs = itertools.product(
                range(self.domains[node]), repeat=input_space_size
            )
            funcs_per_node.append(node_funcs)

        yield from itertools.product(*funcs_per_node)

    def _iterate_space(self):
        """Internal helper that returns materialized space if available, else lazy iterator.

        Use this method internally when iterating over the space to support both lazy and materialized modes.

        Returns:
            Iterator over signature rows. In materialized mode, iterates over self.space list.
            In lazy mode, yields from iter_space().
        """
        if self.space is not None:
            # Materialized mode: iterate over cached list
            return iter(self.space)
        else:
            # Lazy mode: iterate lazily
            return self.iter_space()

    def get_equivalence_classes(
        self, query: CausalQuery | CausalExpression
    ) -> dict[tuple[tuple[int, ...], float, float], int]:
        """Retrieve query-relevant equivalence classes of signatures using an MDD."""
        from .mdd import ResponseSignatureMDD

        if isinstance(query, CausalExpression):
            flat_expr = query
            evidence_dict = list(query.terms.keys())[0].evidence if query.terms else {}
        else:
            flat_expr = CausalExpression({query: 1.0})
            evidence_dict = query.evidence

        # Context 0 is always the natural context {}
        contexts = [{}]
        context_map = {frozenset(): 0}

        for cq in flat_expr.terms.keys():
            for atomic in cq.counterfactuals:
                frozen_int = frozenset(atomic.interventions.items())
                if frozen_int not in context_map:
                    context_map[frozen_int] = len(contexts)
                    contexts.append(atomic.interventions)

        mdd = ResponseSignatureMDD(
            domains=self.domains,
            ordered_nodes=self.ordered_nodes,
            structure=self.structure,
            contexts=contexts,
        )
        mdd.build()
        return mdd.get_equivalence_classes(flat_expr, evidence_dict, context_map)

    @property
    def size(self) -> int:
        """Return the total number of signatures in the space.

        For materialized mode (lazy=False): returns len(self.space).
        For lazy mode (lazy=True): returns cached symbolic computation.

        Returns:
            int: Total signatures, computed once at initialization.
        """
        if self._cached_size is not None:
            return self._cached_size
        if self.space is not None:
            return len(self.space)
        # Fallback: compute size if not yet cached (should not occur in normal flow)
        return self._compute_space_size(self.domains, self.structure)

    def __str__(self):
        return f"ResponseSignature with {len(self.ordered_nodes)} nodes and {self.size:,} functions"

    def __repr__(self):
        return self.__str__()

    # ===============================================
    # Graph
    # ===============================================

    def endogenous_structure(self) -> nx.DiGraph:
        """Returns the endogenous structure as a networkx DiGraph."""
        return nx.subgraph(self.build_canonical_pscm(), self.ordered_nodes)

    def build_canonical_pscm(self) -> nx.DiGraph:
        """
        Constructs the canonical Partial SCM graph based on the structure.
        Each node has an edge from a single exogenous variable 'U' and edges from its
        parents.

        Returns:
            A networkx DiGraph representing the canonical PSCM.
        """

        G = nx.DiGraph()
        node_positions = {
            node: (index, 0) for index, node in enumerate(self.ordered_nodes)
        }
        for node, parents in self.structure.items():
            G.add_node(node, pos=node_positions[node])
            child_index = self.ordered_nodes.index(node)
            for parent in parents:
                parent_index = self.ordered_nodes.index(parent)
                span = abs(child_index - parent_index)
                bend = min(0.25, 0.08 * span)
                G.add_edge(
                    parent,
                    node,
                    connectionstyle=f"arc3,rad={bend}",
                    span=span,
                )

        # Add exogenous variable
        G.add_node("U", pos=((len(self.ordered_nodes) - 1) / 2, -0.5))
        for node in self.ordered_nodes:
            span = abs(
                self.ordered_nodes.index(node) - (len(self.ordered_nodes) - 1) / 2
            )
            bend = min(0.25, 0.05 * span)
            G.add_edge("U", node, connectionstyle=f"arc3,rad={bend}", span=span)

        return G

    def is_compatible(self, *queries: CausalQuery | CausalExpression) -> bool:
        """Checks if the signature structure is compatible with the given queries."""
        import networkx as nx

        sig_graph = nx.subgraph(self.build_canonical_pscm(), self.ordered_nodes)
        if not nx.is_directed_acyclic_graph(sig_graph):
            raise ValueError("Signature structure must be a DAG.")

        for query in queries:
            induced_order = query.induced_order()
            if not nx.is_directed_acyclic_graph(induced_order):
                raise ValueError("Induced order from query must be a DAG.")

            # All variables used by the query must exist in the signature
            missing_nodes = set(induced_order.nodes()) - set(sig_graph.nodes())
            if missing_nodes:
                raise ValueError(
                    f"Signature is missing variables required by query: {sorted(missing_nodes)}"
                )

            for u, v in induced_order.edges():
                if not nx.has_path(sig_graph, u, v):
                    raise ValueError(
                        f"Signature is not compatible: requires a path {u} -> {v} but none exists in signature structure"
                    )

        return True

    def canonical_edge_connectionstyles(self) -> list[str]:
        """Return edge connection styles in graph edge order for curved drawing."""
        graph = self.build_canonical_pscm()
        return [
            graph[u][v].get("connectionstyle", "arc3,rad=0.0") for u, v in graph.edges()
        ]

    @staticmethod
    def draw_canonical_pscm(signature_obj: ResponseSignature):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = signature_obj.build_canonical_pscm()
        pos = nx.get_node_attributes(G, "pos")

        nx.draw_networkx_nodes(G, pos, node_size=400)
        nx.draw_networkx_labels(G, pos)

        for u, v, data in G.edges(data=True):
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                arrows=True,
                arrowstyle="-|>",
                connectionstyle=data.get("connectionstyle", "arc3,rad=0.0"),
            )

        plt.axis("off")
        return plt


class TotalOrderSignature(ResponseSignature):
    """
    Constructs a complete canonical signature space based on a strict total order.

    Note:
    - This is the baseline signature and most expressive one. Mostly not used.

    Args:
        domains: Dict mapping variable names to domain sizes.
        total_order: List of variable names in topological order.
        lazy: If True, skips space materialization (default: False).
    """

    def __init__(
        self, domains: dict[str, int], total_order: list[str], lazy: bool = False
    ):
        super().__init__(domains, lazy=lazy)
        self.ordered_nodes: list[str] = total_order
        self._build_structure()
        self._generate_space()

    def _build_structure(self):
        """In a total order, each node's parents are all preceding nodes in the order."""
        for i, node in enumerate(self.ordered_nodes):
            self.structure[node] = self.ordered_nodes[:i]

    def __str__(self):
        return f"TotalOrderSignature with order {self.ordered_nodes} and {self.size:,} functions"


class PartialOrderSignature(ResponseSignature):
    """
    Constructs a reduced signature space enforcing parallel roots (interventions)
    and a targets ordered by domain size.

    Note:
    - This signature is designed to be compatible with any query. The signature structure
    is built from the query induced order.

    Args:
        domains: Dict mapping variable names to domain sizes.
        query: CausalQuery or CausalExpression to build structure from.
        lazy: If True, skips space materialization (default: False).
    """

    def __init__(
        self,
        domains: dict[str, int],
        query: CausalQuery | CausalExpression,
        lazy: bool = False,
    ):
        super().__init__(domains, lazy=lazy)
        self.query: CausalQuery | CausalExpression = query
        self._build_structure()
        self._generate_space()

    def _build_structure(self):
        """Builds the structure based on the query's counterfactuals and evidence."""
        induced_order = self.query.induced_order()
        topo_nodes = list(
            nx.lexicographical_topological_sort(
                induced_order, key=lambda node: (self.domains[node], node)
            )
        )

        roots = [node for node in topo_nodes if induced_order.in_degree(node) == 0]
        outcomes = [node for node in topo_nodes if induced_order.in_degree(node) > 0]

        self.ordered_nodes = roots + outcomes
        self.structure = {node: [] for node in roots}

        current_parents: list[str] = list(roots)
        for node in outcomes:
            self.structure[node] = list(current_parents)
            current_parents.append(node)

    def __str__(self):
        return f"PartialOrderSignature with order {self.ordered_nodes} and {self.size:,} functions"
