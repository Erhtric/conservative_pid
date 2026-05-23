from dataclasses import dataclass
import itertools
import math
from collections.abc import Callable
from cpid.io import CausalQuery, CausalExpression


class ResponseSignatureMDD:
    """A Multi-valued Decision Diagram representing the response signature space.

    This class builds the MDD topologically, merges isomorphic nodes to perform
    reduction, and computes the exact path counts (multiplicities). Avoids
    explicit enumeration of all input configurations by leveraging the structure of
    the underlying causal graph and the query contexts.
    """

    def __init__(
        self,
        domains: dict[str, int],
        ordered_nodes: list[str],
        structure: dict[str, list[str]],
        contexts: list[dict[str, int]],
    ) -> None:
        """Initialize the MDD.

        Args:
            domains: Dict mapping variable names to their domain sizes.
            ordered_nodes: Topological ordering of the variables.
            structure: Parent mapping for each variable.
            contexts: List of unique query contexts (intervention dicts).
        """
        self.domains = domains
        self.ordered_nodes = ordered_nodes
        self.structure = structure
        self.contexts = contexts

        # Maps a path tuple (tuple of context-value tuples) to its multiplicity count
        # Represents the active layers of the MDD during topological construction
        self.paths: dict[tuple[tuple[int, ...], ...], int] = {(): 1}

        print(
            f"Initialized MDD with {len(self.ordered_nodes)} nodes and {len(self.contexts)} contexts."
        )

    def build(self) -> None:
        """Constructs the reduced MDD layer-by-layer topologically.

        This method executes state-merging reduction at each layer of the diagram,
        ensuring that the number of active nodes remains minimal.
        """
        for node in self.ordered_nodes:
            parents = self.structure[node]
            node_domain = self.domains[node]

            # Total theoretical input configurations for this node's function
            total_configs = (
                1 if not parents else math.prod(self.domains[p] for p in parents)
            )

            next_paths: dict[tuple[tuple[int, ...], ...], int] = {}

            for path_tuple, mult in self.paths.items():
                # Compute parent configurations across all contexts
                context_parent_configs = []
                for ctx_idx, ctx in enumerate(self.contexts):
                    p_vals = []
                    for p in parents:
                        if p in ctx:
                            # Intervention on parent: use the intervened value
                            p_vals.append(ctx[p])
                        else:
                            # Natural parent: use the value from the path tuple
                            p_idx = self.ordered_nodes.index(p)
                            p_vals.append(path_tuple[p_idx][ctx_idx])
                    context_parent_configs.append(tuple(p_vals))

                # Identify unique parent configurations evaluated
                unique_configs = list(dict.fromkeys(context_parent_configs))
                k = len(unique_configs)

                # Count inactive configurations that can be assigned arbitrarily
                inactive_configs = total_configs - k
                state_multiplier = node_domain**inactive_configs

                # Branch over all output combinations for the k active configurations
                for outputs in itertools.product(range(node_domain), repeat=k):
                    config_to_val = dict(zip(unique_configs, outputs))

                    # Resolve node value in each context
                    node_ctx_vals = tuple(
                        ctx[node]
                        if node in ctx
                        else config_to_val[context_parent_configs[ctx_idx]]
                        for ctx_idx, ctx in enumerate(self.contexts)
                    )

                    new_path = path_tuple + (node_ctx_vals,)
                    new_mult = mult * state_multiplier

                    # Merge duplicate nodes by summing their multiplicities (ROBDD-style reduction)
                    next_paths[new_path] = next_paths.get(new_path, 0) + new_mult

            self.paths = next_paths

    def get_equivalence_classes(
        self,
        flat_expr: CausalExpression,
        evidence_dict: dict[str, int],
        context_map: dict[frozenset, int],
    ) -> dict[tuple[tuple[int, ...], float, float], int]:
        """Reduces the completed MDD paths to query-relevant equivalence classes.

        Args:
            flat_expr: Flattened causal expression query.
            evidence_dict: Observational evidence constraints.
            context_map: Mapping from normalized interventions to context index.

        Returns:
            Dict mapping (nat_tuple, num_coeff, den_coeff) -> count.
        """
        equivalence_classes: dict[tuple[tuple[int, ...], float, float], int] = {}

        for path_tuple, count in self.paths.items():
            # Create a lookup mapping for variable values: values[(ctx_idx, node_name)] = val
            values = {
                (ctx_idx, node): path_tuple[n_idx][ctx_idx]
                for n_idx, node in enumerate(self.ordered_nodes)
                for ctx_idx in range(len(self.contexts))
            }

            # Observational state (natural context 0)
            nat_tuple = tuple(values[(0, node)] for node in self.ordered_nodes)

            # Evaluate numerator coefficient (gamma \land delta)
            num_coeff = 0.0
            for cq, w in flat_expr.terms.items():
                # Evidence check
                if not all(
                    values[(0, e_var)] == e_val for e_var, e_val in cq.evidence.items()
                ):
                    continue

                # Counterfactuals query check
                if all(
                    values[
                        (
                            context_map[frozenset(atomic.interventions.items())],
                            atomic.target_var,
                        )
                    ]
                    == atomic.target_val
                    for atomic in cq.counterfactuals
                ):
                    num_coeff += w

            # Evaluate denominator coefficient (delta)
            den_coeff = (
                1.0
                if all(
                    values[(0, e_var)] == e_val
                    for e_var, e_val in evidence_dict.items()
                )
                else 0.0
            )

            eq_key = (nat_tuple, num_coeff, den_coeff)
            equivalence_classes[eq_key] = equivalence_classes.get(eq_key, 0) + count

        return equivalence_classes
