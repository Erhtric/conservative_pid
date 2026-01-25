"""
Translates the Canonical Representation into a Linear Program using PuLP.
Handles both unconditional queries (Standard LP) and conditional queries (Linear-Fractional).
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pulp

from canonical import VectorizedCanonicalBasis
from symbolic import Event, Query, Variable


class LPSolver:
    def __init__(
        self,
        basis: VectorizedCanonicalBasis,
        observational_data: Dict[Tuple[Any, ...], float],
        variables: List[Variable],
        verbose: bool = False,
    ):
        """
        Initializes the solver with the pre-computed canonical basis and observational data.

        :param basis: The list of all deterministic worlds (Omega).
        :param observational_data: A dictionary mapping variable value tuples to probabilities.
                                   e.g., {(0, 1): 0.4, ...} for variables (X, Y).
        :param variables: The list of variables corresponding to the data tuples.
        """
        self.basis = basis
        self.data = observational_data
        self.variables = variables
        self.verbose = verbose

        # Pre-compute the compatibility matrix to speed up multiple query solves.
        # This maps each data_row (observation) to the list of indices of compatible worlds.
        self.compatibility_map = self._build_compatibility_map()

    def _build_compatibility_map(self) -> Dict[Tuple[Any, ...], List[int]]:
        """
        Optimized version using NumPy masks.
        """
        compatibility = {}
        for row in self.data.keys():
            obs_dict = {var: val for var, val in zip(self.variables, row)}

            # Get boolean mask from VectorizedBasis
            mask = self.basis.get_compatibility_mask(obs_dict)

            # Convert to list of indices for PuLP
            # np.where returns a tuple, take [0]
            indices = np.where(mask)[0].tolist()
            compatibility[row] = indices
        return compatibility

    def solve(self, query: Query) -> Tuple[float, float]:
        """
        Computes the Lower and Upper bounds for the given query.
        Returns: (min_bound, max_bound)
        """
        if query.evidence:
            # Use Charnes-Cooper transformation for P(gamma | delta)
            lb = self._solve_linear_fractional(query, sense=pulp.LpMinimize)
            ub = self._solve_linear_fractional(query, sense=pulp.LpMaximize)
        else:
            lb = self._solve_standard_lp(query, sense=pulp.LpMinimize)
            ub = self._solve_standard_lp(query, sense=pulp.LpMaximize)

        return lb, ub

    def _solve_standard_lp(self, query: Query, sense) -> float:
        """
        Implements Algorithm 1 (Def 3.4).
        Min/Max sum(q_omega) subject to observational constraints.

        Args:
            query: The query to solve.
            sense: pulp.LpMinimize or pulp.LpMaximize.

        Returns:
            The lower and upper bounds for the query.
        """
        # 1. Define the Problem
        prob = pulp.LpProblem("Counterfactual_Bounding", sense)

        # 2. Decision Variables: q_omega >= 0
        # We use a dictionary or list to hold LpVariables
        q_vars = [pulp.LpVariable(f"q_{i}", lowBound=0) for i in range(len(self.basis))]

        # 3. Objective Function: P(gamma) = sum(q_omega where omega |- gamma)
        # Theorem 3.3
        objective_mask = self.basis.get_mask(query.target)
        objective_indices = np.where(objective_mask)[0].tolist()
        prob += pulp.lpSum([q_vars[i] for i in objective_indices])

        # 4. Constraints

        # A. Unit Sum Constraint: sum(q) = 1
        prob += pulp.lpSum(q_vars) == 1.0, "Normalisation_Probabilities"

        # B. Observational Consistency: sum(q where omega |= v) = P(v)
        # Theorem 3.2
        for row, prob_val in self.data.items():
            compatible_indices = self.compatibility_map[row]
            if not compatible_indices:
                # If data has probability > 0 but no world can generate it, the model is invalid.
                if prob_val > 0:
                    raise ValueError(
                        f"Data row {row} is impossible under the current causal order."
                    )
                continue

            prob += (
                pulp.lpSum([q_vars[i] for i in compatible_indices]) == prob_val,
                f"Obs_{row}",
            )

        # 5. Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=self.verbose))

        if prob.status != pulp.LpStatusOptimal:
            return float("nan")  # Or handle infeasibility appropriately

        return pulp.value(prob.objective)

    def _solve_linear_fractional(self, query: Query, sense) -> float:
        """
        Implements Charnes-Cooper transformation for Conditional Queries.
        Section 5.1 [cite: 231-235].
        """
        prob = pulp.LpProblem("Conditional_Bounding", sense)

        # 1. Transformed Variables
        # q_prime corresponds to q_omega * t
        q_prime_vars = [
            pulp.LpVariable(f"q_prime_{i}", lowBound=0) for i in range(len(self.basis))
        ]
        # t is the inverse of the probability of the evidence P(delta)
        t = pulp.LpVariable("t", lowBound=0)

        # 2. Objective: P(gamma & delta) * t
        # (Transformed numerator)
        joint_event = query.target & query.evidence  # Logical AND
        obj_indices = [i for i, w in enumerate(self.basis) if w.satisfies(joint_event)]
        prob += pulp.lpSum([q_prime_vars[i] for i in obj_indices])

        # 3. Constraints

        # A. Denominator Constraint: sum(q_prime where omega |- evidence) = 1
        # This ensures that we are normalizing by P(evidence)
        evidence_indices = [
            i for i, w in enumerate(self.basis) if w.satisfies(query.evidence)
        ]
        prob += (
            pulp.lpSum([q_prime_vars[i] for i in evidence_indices]) == 1.0,
            "Charnes_Cooper_Norm",
        )

        # B. Unit Sum Transformed: sum(q_prime) = t
        prob += pulp.lpSum(q_prime_vars) == t, "Sum_t"

        # C. Observational Constraints Transformed: sum(q_prime) = P(v) * t
        for row, prob_val in self.data.items():
            compatible_indices = self.compatibility_map[row]
            if compatible_indices:
                # LHS: sum of transformed vars
                lhs = pulp.lpSum([q_prime_vars[i] for i in compatible_indices])
                # RHS: prob_val * t
                prob += lhs == prob_val * t, f"Obs_Transformed_{row}"

        # 4. Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status != pulp.LpStatusOptimal:
            return float("nan")

        return pulp.value(prob.objective)
