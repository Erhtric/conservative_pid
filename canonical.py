import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger

from symbolic import CounterfactualTerm, Event, Variable


class CanonicalConfiguration:
    """
    Represent a single deterministic world 'omega'.
    In this world, every counterfactual term Y_{X=x} is assigned a value.

    Example:
        X = Variable("X", domain=(0, 1))
        Y = Variable("Y", domain=(0, 1))
        omega = CanonicalConfiguration({
            X: 0,
            Y: 1,  // This can be removed since it is a deterministic function of the other two responses
            Y @ {X: 1}: 0,
            Y @ {X: 0}: 1,
        })

    """

    def __init__(
        self,
        functions: Dict[Variable, Dict[Tuple[Any, ...], Any]],
        order: List[Variable],
    ):
        self.functions = functions
        self.order = order  # The topological order defining parentage
        self.parents_map = self._build_parents_map(order)

    def _build_parents_map(
        self, order: List[Variable]
    ) -> Dict[Variable, List[Variable]]:
        """
        Constructs the parent set for each variable based on the total order.
        For a fully connected DAG (conservative), parents are all predecessors in 'order'.
        """
        parents = {}
        for i, var in enumerate(order):
            parents[var] = order[:i]
        return parents

    def evaluate(self, term: CounterfactualTerm) -> Any:
        """
        Recursively evaluate a counterfactual term, i.e., Y @ {X: x, Z: z}
        in this deterministic world.
        Ref: Theorem 3.3 in the paper.
        """
        target_var = term.variable
        intervention = term.intervention

        # 1. Check if the variable is directly intervened on
        if target_var in intervention:
            val = intervention[target_var]

            # Nested case
            if isinstance(val, CounterfactualTerm):
                return self.evaluate(val)

            # Base case: value is a constant
            return val

        # 2. If not intervened, then it is a response variable
        # It follows its structural equation

        # First evaluate its parents
        parents = self.parents_map[target_var]

        parent_values = []
        for p in parents:
            # Recursive step:
            # The value of a parent P is P_{assignment_of_current_term}
            # Effectively, the intervention propagates down.

            # Construct the term for the parent
            parent_term = CounterfactualTerm(p, intervention)
            parent_val = self.evaluate(parent_term)
            parent_values.append(parent_val)

        # 3. Apply the deterministic function for this variable
        func = self.functions[target_var]
        return func[tuple(parent_values)]

    def satisfies(self, event: Event) -> bool:
        """
        Check if this canonical configuration satisfies a given event.
        """
        for term, required_value in event.assignments.items():
            if self.evaluate(term) != required_value:
                return False
        return True

    def is_compatible(self, observation: Event) -> bool:
        """
        Checks if this world is compatible with a pure observation v.
        omega |= v
        """
        # An observation is just a special case of an event where intervention is empty
        for term, val in observation.assignments.items():
            # term is CounterfactualTerm(variable, {})
            if self.evaluate(term) != val:
                return False
        return True


class BasisGenerator:
    """
    Generate a basis of canonical configurations for a given set of variables.
    """

    def __init__(self, variables: List[Variable]):
        self.variables = variables
        self.basis = []

        self.check_domains()

    def check_domains(self):
        for var in self.variables:
            if var.domain is None:
                raise ValueError(f"Variable {var} has no domain.")

    def generate_basis(self) -> List[CanonicalConfiguration]:
        # Process variables in order
        # For each variable

        all_var_functions = []
        for i, var in enumerate(self.variables):
            parents = self.variables[:i]

            # 1. Generate all parent configurations
            parent_domains = [p.domain for p in parents]
            parent_configs = list(itertools.product(*parent_domains))

            # 2. Generate all possible functions for this variable
            # This is equivalent to generating all possible canonical configurations.
            pot_out = list(itertools.product(var.domain, repeat=len(parent_configs)))

            # 3. Create dicts for these functions
            var_funcs = []
            for out in pot_out:
                # Map (parent_config) -> outputG
                f_map = {cfg: res for cfg, res in zip(parent_configs, out)}
                var_funcs.append(f_map)

            all_var_functions.append(var_funcs)

        # 4. Generate all possible configurations
        basis = []
        for var_funcs in itertools.product(*all_var_functions):
            # var_funcs is tuple(func_v1, func_v2, ...)
            func_map = {var: func for var, func in zip(self.variables, var_funcs)}
            basis.append(CanonicalConfiguration(func_map, self.variables))

        return basis


class VectorizedCanonicalBasis:
    """
    A Numpy-based implementation of the canonical basis.
    """

    def __init__(self, variables: List[Variable]):
        """
        Initializes the basis with the given variables.
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

    def get_mask(self, event: Event) -> np.ndarray:
        """
        Returns a boolean array (N_worlds,) where True indicates the world satisfies the event.
        omega |- gamma
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

    def get_compatibility_mask(self, observation: Dict[Variable, Any]) -> np.ndarray:
        """
        Returns boolean mask for worlds compatible with observational data.
        omega |= v
        """
        # Convert observation dict to Event (intervention is empty)
        obs_event = Event(
            {CounterfactualTerm(v, {}): val for v, val in observation.items()}
        )
        return self.get_mask(obs_event)
