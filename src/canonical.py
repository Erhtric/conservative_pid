import itertools
from typing import Any, Dict, List, Union

import numpy as np
from loguru import logger

from src.symbolic import CounterfactualTerm, Event, Variable


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
        self.variables = variables
        self.var_to_idx = {var: i for i, var in enumerate(variables)}

        # 1. Map domains to integers
        self.domain_maps = {}  # {Variable: {val: int_code}}
        self.inverse_maps = {}  # {Variable: {int_code: val}}
        for var in variables:
            d_map = {val: i for i, val in enumerate(var.domain)}
            self.domain_maps[var] = d_map
            self.inverse_maps[var] = {i: val for val, i in d_map.items()}

        # 2. Build the Basis matrix
        # func_tables[i] is a 2D array: (Num_Worlds, Num_Parent_Configs)
        self.func_tables, self.parent_strides = self._generate_basis_matrices()

        self.n_worlds = self.func_tables[0].shape[0]
        logger.info(f"Generated Basis with {self.n_worlds} worlds.")

    def _generate_basis_matrices(self):
        """
        Generates the function tables using Cartesian product.

        Returns:
            func_tables: List of 2D arrays, where each array has shape (Num_Worlds, Num_Parent_Configs)
            parent_strides: List of arrays, where each array has shape (Num_Parent_Configs,)
        """
        # Step A: Calculate number of functions for each variable
        # A "function" is just a specific set of outputs for all parent inputs.
        var_func_blocks = []  # Stores the raw columns for each variable
        parent_strides_all = []  # Stores the strides for each variable

        for i, var in enumerate(self.variables):
            parents = self.variables[:i]

            # Calculate parent configuration space size
            parent_sizes = [len(p.domain) for p in parents]
            n_parent_configs = int(np.prod(parent_sizes)) if parents else 1

            # Pre-calc strides for converting parent values -> linear index
            # This works like np.ravel_multi_index
            if parents:
                strides = []
                current = 1
                for size in reversed(parent_sizes):
                    strides.append(current)
                    current *= size
                parent_strides_all.append(np.array(strides[::-1]))
            else:
                parent_strides_all.append(np.array([]))

            # Generate all possible outcome vectors (functions) for this variable
            # Shape: (Num_Functions, Num_Parent_Configs)
            # Domain size ^ Parent Configs
            possible_outcomes = list(
                itertools.product(range(len(var.domain)), repeat=n_parent_configs)
            )
            func_block = np.array(possible_outcomes, dtype=np.int8)
            var_func_blocks.append(func_block)

        # Step B: Cartesian Product of all function blocks to form the worlds
        # Total worlds = Product(len(block) for block in var_func_blocks)

        final_tables = []
        n_total_worlds = np.prod([len(b) for b in var_func_blocks])

        # We construct the full world matrix column by column (variable by variable)
        # Using repeat/tile logic to simulate itertools.product
        current_repeat = int(n_total_worlds)

        for block in var_func_blocks:
            n_funcs = len(block)
            current_repeat //= n_funcs

            # 1. Repeat each row of the block 'current_repeat' times
            # 2. Tile the whole result to fill the total length

            # Example: Block has 2 funcs [A, B], total worlds 4.
            # repeat=2 -> [A, A, B, B]
            # If total was 8 and this was inner, tile -> [A,A,B,B, A,A,B,B]

            expanded = np.repeat(block, current_repeat, axis=0)
            tiles_needed = n_total_worlds // len(expanded)
            full_column = np.tile(expanded, (tiles_needed, 1))

            final_tables.append(full_column)

        return final_tables, parent_strides_all

    def evaluate(self, term: CounterfactualTerm) -> np.ndarray:
        """
        Evaluates the term in every world.
        Returns an array of shape (N_worlds,) containing the value of 'term' in every world.
        In other words, it returns the column of the basis matrix corresponding to 'term'.

        Args:
            term: The term to evaluate.

        Returns:
            np.ndarray: An array of shape (N_worlds,) containing the value of 'term' in every world.
        """
        response_var = term.variable
        var_idx = self.var_to_idx[response_var]

        # 1. Handle Interventions
        # If the variable is directly intervened on, return constant array
        if response_var in term.intervention:
            val = term.intervention[response_var]

            # Handle nested intervention recursively: X_{Z_w}
            if isinstance(val, CounterfactualTerm):
                return self.evaluate(val)

            # Constant intervention
            int_code = self.domain_maps[response_var][val]
            return np.full(self.n_worlds, int_code, dtype=np.int8)

        # 2. Evaluate Parents recursively
        parents = self.variables[:var_idx]
        if not parents:
            # No parents (root node), simply return column 0 of its table
            return self.func_tables[var_idx][:, 0]

        # Calculate indices for the function table columns
        # Column Index = P1_val * stride1 + P2_val * stride2 + ...
        col_indices = np.zeros(self.n_worlds, dtype=np.int64)
        strides = self.parent_strides[var_idx]

        for p_idx, parent in enumerate(parents):
            # Create term for parent with same intervention
            p_term = CounterfactualTerm(parent, term.intervention)
            p_values = self.evaluate(p_term)  # Recursive call

            col_indices += p_values * strides[p_idx]

        # 3. Vectorized Lookup
        # We want: result[w] = table[w, col_indices[w]]
        row_indices = np.arange(self.n_worlds)
        return self.func_tables[var_idx][row_indices, col_indices]

    def get_mask(self, event: Union[Event]) -> np.ndarray:
        """
        Returns a boolean array (N_worlds,) where True indicates the world satisfies the event.
        Logic corresponds to Theorems 3.2 (Observational) and 3.3 (Counterfactual) in the paper.
        omega |- gamma

        Args:
            event: The event to evaluate.

        Returns:
            np.ndarray: A boolean array of shape (N_worlds,) where True indicates the world satisfies the event.
        """
        mask = np.ones(self.n_worlds, dtype=bool)

        for term, val in event.assignments.items():
            # Get values for this term across all worlds
            term_values_coded = self.evaluate(term)

            # Get integer code for the required value
            req_code = self.domain_maps[term.variable][val]

            # Update mask
            mask &= term_values_coded == req_code

        return mask
