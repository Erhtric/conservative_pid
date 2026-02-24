import itertools
import numpy as np
from loguru import logger

from typing import List, Union
from .symbolic import CounterfactualTerm, Event, Variable, MonotonicityConstraint


class VectorizedCanonicalBasis:
    """
    A Numpy-based implementation of the canonical basis for counterfactuals.

    We store the basis as a list of matrices, where each matrix is of shape
    (N_worlds, N_functions).
    A stride array is stored for each variable, indicating the stride for each parent.
    """

    def __init__(self, variables: List[Variable]):
        """
        Initializes the basis.

        Args:
            variables: List of **ordered** variables in the causal model.
        """
        self.variables: list[Variable] = variables
        self.var_to_idx: dict[Variable, int] = {
            var: i for i, var in enumerate(variables)
        }

        self.domain_maps: dict[Variable, dict] = {}
        self.inverse_maps: dict[Variable, dict] = {}
        for var in variables:
            d_map = {val: i for i, val in enumerate(var.domain)}
            self.domain_maps[var] = d_map
            self.inverse_maps[var] = {i: val for val, i in d_map.items()}

        self.func_tables, self.parent_strides = self._generate_basis_matrices()

        self.n_worlds: int = self.func_tables[0].shape[0]
        logger.info(f"Generated Basis with {self.n_worlds} worlds.")

    def _generate_basis_matrices(self):
        """
        Generates the function tables using Cartesian product.

        Returns:
            func_tables: List of 2D arrays, where each array has shape (Num_Worlds, Num_Parent_Configs)
            parent_strides: List of arrays, where each array has shape (Num_Parent_Configs,)
        """
        var_func_blocks = []
        parent_strides_all = []

        for i, var in enumerate(self.variables):
            parents = self.variables[:i]

            parent_sizes = [len(p.domain) for p in parents]
            n_parent_configs = int(np.prod(parent_sizes)) if parents else 1

            if parents:
                strides = []
                current = 1
                for size in reversed(parent_sizes):
                    strides.append(current)
                    current *= size
                parent_strides_all.append(np.array(strides[::-1]))
            else:
                parent_strides_all.append(np.array([]))

            possible_outcomes = list(
                itertools.product(range(len(var.domain)), repeat=n_parent_configs)
            )
            func_block = np.array(possible_outcomes, dtype=np.int8)
            var_func_blocks.append(func_block)

        final_tables = []
        n_total_worlds = np.prod([len(b) for b in var_func_blocks])

        current_repeat = int(n_total_worlds)

        for block in var_func_blocks:
            n_funcs = len(block)
            current_repeat //= n_funcs

            expanded = np.repeat(block, current_repeat, axis=0)
            tiles_needed = n_total_worlds // len(expanded)
            full_column = np.tile(expanded, (tiles_needed, 1))

            final_tables.append(full_column)

        return final_tables, parent_strides_all

    def evaluate(self, term: CounterfactualTerm) -> np.ndarray:
        """
        Evaluates the term in every world.
        Returns an array of shape (N_worlds,) containing the value of `term` in every world.
        In a programmatic way, it returns the column of the matrix corresponding to 'term'.

        Args:
            term: The term to evaluate.

        Returns:
            np.ndarray: An array of shape (N_worlds,) containing the value of 'term' in every world.
        """
        response_var = term.variable
        var_idx = self.var_to_idx[response_var]

        if response_var in term.intervention:
            val = term.intervention[response_var]

            if isinstance(val, CounterfactualTerm):
                return self.evaluate(val)

            int_code = self.domain_maps[response_var][val]
            return np.full(self.n_worlds, int_code, dtype=np.int8)

        parents = self.variables[:var_idx]
        if not parents:
            return self.func_tables[var_idx][:, 0]

        col_indices = np.zeros(self.n_worlds, dtype=np.int64)
        strides = self.parent_strides[var_idx]

        for p_idx, parent in enumerate(parents):
            p_term = CounterfactualTerm(parent, term.intervention)
            p_values = self.evaluate(p_term)

            col_indices += p_values * strides[p_idx]

        row_indices = np.arange(self.n_worlds)
        return self.func_tables[var_idx][row_indices, col_indices]

    def get_mask(self, event: Union[Event]) -> np.ndarray:
        """
        Returns a boolean array (N_worlds,) where True indicates the world satisfies the event.
        Corresponds to an entailment for the event.

        Args:
            event: The event to evaluate.

        Returns:
            np.ndarray: A boolean array of shape (N_worlds,) where True indicates the world satisfies the event.
        """
        mask: np.ndarray = np.ones(self.n_worlds, dtype=bool)

        for term, val in event.assignments.items():
            term_values_coded = self.evaluate(term)
            req_code = self.domain_maps[term.variable][val]
            mask &= term_values_coded == req_code

        return mask

    def filter_worlds(
        self, constraints: List[MonotonicityConstraint], return_removed=False
    ) -> np.ndarray:
        """
        Filters the basis by removing worlds that violate the given monotonicity constraints.

        Args:
            constraints: A list of MonotonicityConstraint objects.
        """
        if not constraints:
            return np.array([], dtype=int)  # No constraints, no worlds removed

        valid_mask = np.ones(self.n_worlds, dtype=bool)

        for constraint in constraints:
            lhs_values = self.evaluate(constraint.lhs)
            rhs_values = self.evaluate(constraint.rhs)

            if constraint.operator == "ge":
                valid_mask &= lhs_values >= rhs_values
            elif constraint.operator == "le":
                valid_mask &= lhs_values <= rhs_values
            else:
                raise ValueError(f"Unknown operator: {constraint.operator}")

        # Apply the mask to all function tables
        for i in range(len(self.func_tables)):
            self.func_tables[i] = self.func_tables[i][valid_mask]

        self.n_worlds = self.func_tables[0].shape[0]
        logger.info(f"Filtered Basis. Remaining worlds: {self.n_worlds}")

        if return_removed:
            return np.where(~valid_mask)[0]
