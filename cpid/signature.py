import itertools
import math
from abc import ABC, abstractmethod
from .io import CausalQuery, CausalExpression
import networkx as nx


class SignatureQueryEvaluator:
    """
    Evaluates whether a given signature row satisfies a CausalQuery.
    """

    def __init__(
        self,
        domains: dict[str, int],
        signature_obj: ResponseSignature | None = None,
        query: CausalQuery | CausalExpression | None = None,
    ):
        self.domains = domains
        self.signature = signature_obj
        self.query = query

        if self.signature is not None and self.query is not None:
            try:
                self.signature.is_compatible(self.query)
            except ValueError as e:
                raise ValueError(f"Signature is not compatible with query: {e}")

    def _get_function_index(
        self, parent_vals: list[int], parent_names: list[str]
    ) -> int:
        """Converts a tuple of parent values into a flat list index.

        Args:
            parent_vals: A list of parent variable values (e.g., [0, 1]).
            parent_names: A list of parent variable names in the same order as parent_vals (e.g., ['X', 'Z']).

        Returns:
            An integer index corresponding to the combination of parent values.

        Usage:
            For example, if parent_names = ['X', 'Z'] and domains = {'X': 2, 'Z': 2}, then:
            - parent_vals = [0, 0] -> index 0
            - parent_vals = [0, 1] -> index 1
            - parent_vals = [1, 0] -> index 2
            - parent_vals = [1, 1] -> index 3
        """
        idx = 0
        multiplier = 1
        for p_val, p_name in zip(reversed(parent_vals), reversed(parent_names)):
            idx += p_val * multiplier
            multiplier *= self.domains[p_name]
        return idx

    def evaluate_state(
        self,
        row: list[tuple[int]],
        interventions: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """
        Flows the topological cascade to compute the state of all variables
        under a specific set of interventions.

        Args:
            row: A list of tuples representing the function outputs for each node.
            signature_obj: The ResponseSignature object defining the structure.
            interventions: A dict of interventions to apply (e.g., {'X': 1}).

        Returns:
            dict[str, int]: The resulting state of all variables after applying interventions.
        """
        if interventions is None:
            interventions = {}

        state = {}
        for i, node in enumerate(self.signature.ordered_nodes):
            if node in interventions:
                # Override with intervention
                state[node] = interventions[node]
            else:
                # Calculate from parents
                parents = self.signature.structure[node]
                parent_vals = [state[p] for p in parents]
                func_idx = self._get_function_index(parent_vals, parents)
                state[node] = row[i][func_idx]
        return state

    def row_satisfies_query(
        self,
        sig_row: list[tuple[int]],
    ) -> bool:
        """
        Checks if a specific signature row satisfies the entire CausalQuery.
        Returns True if it matches evidence AND all counterfactuals.

        Args:
            row: A list of tuples representing the function outputs for each node.
            signature_obj: The ResponseSignature object defining the structure.

        Returns:
            bool: True if the row satisfies the query, False otherwise.
        """
        active_query = self.query
        if active_query is None:
            raise ValueError("No query provided to SignatureQueryEvaluator.")
        if not isinstance(active_query, CausalQuery):
            raise TypeError(
                "SignatureQueryEvaluator.row_satisfies_query expects a CausalQuery."
            )

        if active_query.evidence:
            nat_state = self.evaluate_state(sig_row, interventions={})
            for e_var, e_val in active_query.evidence.items():
                if nat_state[e_var] != e_val:
                    return False

        for atomic in active_query.counterfactuals:
            # Ensure the query has been un-nested: interventions should be ints
            for iv in atomic.interventions.values():
                if not isinstance(iv, int):
                    raise ValueError(
                        "Nested interventions detected during evaluation. Call `CausalQuery.unnest(domains)` before evaluating or solving LPs."
                    )

            int_state = self.evaluate_state(sig_row, interventions=atomic.interventions)
            if int_state[atomic.target_var] != atomic.target_val:
                return False

        return True


class ResponseSignature(ABC):
    """
    Abstract base class for all response signatures.
    """

    def __init__(self, domains: dict[str, int]):
        self.domains: dict[str, int] = domains
        self.ordered_nodes: list[str] = []
        self.structure: dict[str, list[str]] = {}
        self.space: list[tuple[int]] = []

    @abstractmethod
    def _build_structure(self):
        """Must be implemented by subclasses to define the topological parent mapping."""
        pass

    def _generate_space(self):
        """Generates the Cartesian product of functional mappings based on the structure."""
        self.space = list(self.iter_space())

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

    def get_equivalence_classes(
        self, query: CausalQuery | CausalExpression
    ) -> dict[tuple, int]:
        """
        Returns a dictionary of algebraic equivalence classes.
        Key: (natural_state_tuple, numerator_coefficient, denominator_coefficient)
        Value: Cardinality (number of response signatures belonging to this class)
        """
        equivalence_classes = {}
        state_evaluator = SignatureQueryEvaluator(self.domains, signature_obj=self)

        # Extract flat queries and evidence for coefficient evaluation
        if isinstance(query, CausalExpression):
            flat_expr = query
            evidence_dict = list(query.terms.keys())[0].evidence if query.terms else {}
        else:
            flat_expr = CausalExpression({query: 1.0})
            evidence_dict = query.evidence

        evidence_query = CausalQuery(counterfactuals=[], evidence=evidence_dict)
        query_evaluators = [
            (SignatureQueryEvaluator(self.domains, signature_obj=self, query=cq), w)
            for cq, w in flat_expr.terms.items()
        ]
        evidence_evaluator = SignatureQueryEvaluator(
            self.domains, signature_obj=self, query=evidence_query
        )

        # WARNING: Naive iteration. For large domains, this must be replaced
        # with a symbolic counting algorithm to bypass self.iter_space().
        for sig in self.iter_space():
            nat_state = state_evaluator.evaluate_state(sig)
            nat_tuple = tuple(nat_state[v] for v in self.ordered_nodes)

            num_coeff = 0.0
            for evaluator, w in query_evaluators:
                if evaluator.row_satisfies_query(sig):
                    num_coeff += w

            den_coeff = 1.0 if evidence_evaluator.row_satisfies_query(sig) else 0.0

            eq_key = (nat_tuple, num_coeff, den_coeff)
            equivalence_classes[eq_key] = equivalence_classes.get(eq_key, 0) + 1

        return equivalence_classes

    @property
    def size(self):
        return len(self.space)

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
    """

    def __init__(self, domains: dict[str, int], total_order: list[str]):
        super().__init__(domains)
        self.ordered_nodes: list[str] = total_order
        self._build_structure()
        self._generate_space()

    def _build_structure(self):
        """In a total order, each node's parents are all preceding nodes in the order."""
        for i, node in enumerate(self.ordered_nodes):
            self.structure[node] = self.ordered_nodes[:i]


class PartialOrderSignature(ResponseSignature):
    """
    Constructs a reduced signature space enforcing parallel roots (interventions)
    and a targets ordered by domain size.
    """

    def __init__(self, domains: dict[str, int], query: CausalQuery | CausalExpression):
        super().__init__(domains)
        self.query: CausalQuery | CausalExpression = query
        self._build_structure()
        self._generate_space()

    def _build_structure(self):
        """Builds the structure based on the query's counterfactuals and evidence."""
        if isinstance(self.query, CausalQuery):
            roots_set: set[str] = set()
            for cf in self.query.counterfactuals:
                roots_set.update(cf.interventions.keys())
            # NOTE: observational evidence variables are also roots (?)
            roots_set.update(self.query.evidence.keys())

            roots = list(roots_set)
            print(f"Identified roots: {roots}")

            outcomes = list({cf.target_var for cf in self.query.counterfactuals})
            outcomes.sort(key=lambda v: self.domains[v])

            self.ordered_nodes = roots + outcomes
            self.structure = {r: [] for r in roots}

            current_parents = list(roots)
            for o in outcomes:
                self.structure[o] = list(current_parents)
                current_parents.append(o)

        elif isinstance(self.query, CausalExpression):
            target_vars = self.query.target_variables
            intervention_vars = self.query.intervention_variables

            roots = sorted(intervention_vars)
            outcomes = sorted(target_vars, key=lambda v: self.domains[v])
            self.ordered_nodes = roots + outcomes
            self.structure = {r: [] for r in roots}
            current_parents = list(roots)
            for o in outcomes:
                self.structure[o] = list(current_parents)
                current_parents.append(o)
