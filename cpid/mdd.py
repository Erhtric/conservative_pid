import itertools
import math
from cpid.io import CausalExpression


class ResponseSignatureMDD:
    def __init__(
        self,
        domains: dict[str, int],
        ordered_nodes: list[str],
        structure: dict[str, list[str]],
        contexts: list[dict[str, int]],
    ) -> None:
        """
        Initializes the MDD with the given causal graph structure and contexts.
        Args:
            domains: Dict mapping variable names to their number of discrete values.
            ordered_nodes: List of variable names in topological order.
            structure: Dict mapping each variable to its list of parent variables.
            contexts: List of dicts, each representing a context with variable assignments.
        """
        self.domains = domains
        self.ordered_nodes = ordered_nodes
        self.structure = structure
        self.contexts = contexts
        self.paths: dict[tuple[tuple[int, ...], ...], int] = {(): 1}

        self._product_cache: dict[tuple[int, int], list[tuple[int, ...]]] = {}

    def build(self) -> None:
        """
        Constructs the MDD by iterating through nodes in topological order and
        expanding paths based on parent configurations and contexts.

        For more on the binary case: https://en.wikipedia.org/wiki/Binary_decision_diagram
        On the use for efficient representation of boolean functions (and more):
        https://arxiv.org/abs/1301.3880
        """
        for node in self.ordered_nodes:
            parents = self.structure[node]
            node_domain = self.domains[node]
            total_configs = (
                1 if not parents else math.prod(self.domains[p] for p in parents)
            )

            next_paths: dict[tuple[tuple[int, ...], ...], int] = {}
            parent_indices = [self.ordered_nodes.index(p) for p in parents]
            node_ctx_static = {
                ctx_idx: ctx[node]
                for ctx_idx, ctx in enumerate(self.contexts)
                if node in ctx
            }

            for path_tuple, mult in self.paths.items():
                context_parent_configs = []

                for ctx_idx, ctx in enumerate(self.contexts):
                    p_vals = tuple(
                        ctx[p] if p in ctx else path_tuple[p_idx][ctx_idx]
                        for p, p_idx in zip(parents, parent_indices)
                    )
                    context_parent_configs.append(p_vals)

                unique_configs = list(dict.fromkeys(context_parent_configs))
                k = len(unique_configs)

                inactive_configs = total_configs - k
                new_base_mult = mult * (node_domain**inactive_configs)

                # Retrieve or compute the permutations for 'k' unique configurations
                cache_key = (node_domain, k)
                if cache_key not in self._product_cache:
                    self._product_cache[cache_key] = list(
                        itertools.product(range(node_domain), repeat=k)
                    )
                combinations = self._product_cache[cache_key]

                for outputs in combinations:
                    config_to_val = dict(zip(unique_configs, outputs))

                    # Utilise the pre-resolved static contexts to avoid dictionary 'in' checks
                    node_ctx_vals = tuple(
                        node_ctx_static.get(
                            ctx_idx, config_to_val[context_parent_configs[ctx_idx]]
                        )
                        for ctx_idx in range(len(self.contexts))
                    )

                    new_path = path_tuple + (node_ctx_vals,)
                    next_paths[new_path] = next_paths.get(new_path, 0) + new_base_mult

            self.paths = next_paths

    # def get_equivalence_classes(
    #     self,
    #     flat_expr: CausalExpression,
    #     evidence_dict: dict[str, int],
    #     context_map: dict[frozenset, int],
    # ) -> dict[tuple[tuple[int, ...], float, float], int]:
    #     """Reduces the completed MDD paths to query-relevant equivalence classes.

    #     We proceed to evaluate each leaf against the query (represented as context_map). For each path,
    #     we determine the natural context variable values (nat_tuple) and
    #     check if the path satisfies the query's counterfactual conditions
    #     (for the numerator) and evidence conditions (for the denominator).
    #     We then aggregate counts of paths that share the same
    #     (nat_tuple, num_coeff, den_coeff) signature into equivalence classes.

    #     Args:
    #         flat_expr: Flattened causal expression query.
    #         evidence_dict: Observational evidence constraints.
    #         context_map: Mapping from normalized interventions to context multiplicities.

    #     Returns:
    #         Dict mapping (nat_tuple, num_coeff, den_coeff) -> count.
    #     """
    #     equivalence_classes: dict[tuple[tuple[int, ...], float, float], int] = {}

    #     for path_tuple, count in self.paths.items():
    #         print(
    #             f"Evaluating path with multiplicity {path_tuple, count}..."
    #         )  # Debug statement
    #         # Create a lookup mapping for variable values: values[(ctx_idx, node_name)] = val
    #         values = {
    #             (ctx_idx, node): path_tuple[n_idx][ctx_idx]
    #             for n_idx, node in enumerate(self.ordered_nodes)
    #             for ctx_idx in range(len(self.contexts))
    #         }

    #         # Observational state (natural context 0)
    #         nat_tuple = tuple(values[(0, node)] for node in self.ordered_nodes)

    #         # Evaluate numerator coefficient (gamma \land delta)
    #         num_coeff = 0.0
    #         for cq, w in flat_expr.terms.items():
    #             # Evidence check
    #             if not all(
    #                 values[(0, e_var)] == e_val for e_var, e_val in cq.evidence.items()
    #             ):
    #                 continue

    #             # Counterfactuals query check
    #             if all(
    #                 values[
    #                     (
    #                         context_map[frozenset(atomic.interventions.items())],
    #                         atomic.target_var,
    #                     )
    #                 ]
    #                 == atomic.target_val
    #                 for atomic in cq.counterfactuals
    #             ):
    #                 num_coeff += w

    #         # Evaluate denominator coefficient (delta)
    #         den_coeff = (
    #             1.0
    #             if all(
    #                 values[(0, e_var)] == e_val
    #                 for e_var, e_val in evidence_dict.items()
    #             )
    #             else 0.0
    #         )

    #         eq_key = (nat_tuple, num_coeff, den_coeff)
    #         equivalence_classes[eq_key] = equivalence_classes.get(eq_key, 0) + count

    #     return equivalence_classes

    def get_equivalence_classes(
        self,
        flat_expr: CausalExpression,
        evidence_dict: dict[str, int],
        context_map: dict[frozenset, int],
    ) -> dict[tuple[tuple[int, ...], float, float], int]:
        """
        Each MDD leaf is associated to a signature row. We tests that tuple against the query:
        1. extract the natural config (under empty intervention)
        2. check if the path satisfies the query evidence (denominator)
        3. check if the path satisfies the query counterfactual conditions (numerator)
        4. aggregate counts of paths that share the same (nat_tuple, num_coeff, den_coeff) signature into equivalence classes.
        """
        equivalence_classes: dict[tuple[tuple[int, ...], float, float], int] = {}

        node_to_idx = {node: i for i, node in enumerate(self.ordered_nodes)}
        num_nodes = len(self.ordered_nodes)

        query_ev = [(node_to_idx[v], val) for v, val in evidence_dict.items()]

        term_meta = []
        for cq, w in flat_expr.terms.items():
            ev_pairs = [(node_to_idx[v], val) for v, val in cq.evidence.items()]
            atomics = []
            for atomic in cq.counterfactuals:
                ctx_idx = context_map[frozenset(atomic.interventions.items())]

                atomics.append(
                    (ctx_idx, node_to_idx[atomic.target_var], atomic.target_val)
                )

            term_meta.append((ev_pairs, atomics, w))

        # Evaluate paths against query conditions to determine equivalence classes
        for path_tuple, count in self.paths.items():
            # Natural context
            nat_tuple = tuple(path_tuple[i][0] for i in range(num_nodes))

            # Denominator: evidence in the natural world (delta)
            den_coeff = (
                1.0 if all(path_tuple[idx][0] == val for idx, val in query_ev) else 0.0
            )

            # Numerator: gamma \land delta check across all terms
            num_coeff = 0.0
            for ev_pairs, atomics, w in term_meta:
                # Evidence check (context 0)
                if not all(path_tuple[idx][0] == val for idx, val in ev_pairs):
                    continue

                # Counterfactual check (possibly other contexts)
                if all(
                    path_tuple[idx][ctx_idx] == val for ctx_idx, idx, val in atomics
                ):
                    num_coeff += w

            eq_key = (nat_tuple, num_coeff, den_coeff)
            equivalence_classes[eq_key] = equivalence_classes.get(eq_key, 0) + count

        return equivalence_classes
