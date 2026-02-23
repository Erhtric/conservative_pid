from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pulp

from canonical import VectorizedCanonicalBasis
from symbolic import CounterfactualTerm, Event, Expression, Query, Variable


class LPSolver:
    def __init__(
        self,
        basis: VectorizedCanonicalBasis,
        observational_probs: Dict[Tuple[Any, ...], float],
        variables: List[Variable],
        verbose: bool = False,
    ):
        """
        Initializes the solver with the pre-computed canonical basis and observational data.

        Args:
            basis: The list of all deterministic worlds (Omega).
            observational_data: A dictionary mapping variable value tuples to probabilities.
                                e.g., {(0, 1): 0.4, ...} for variables (X, Y).
            variables: The list of variables corresponding to the data tuples.
            verbose: Whether to print verbose output.

        Raises:
            ValueError: If the observational data is inconsistent with the causal order.
        """
        self.basis: VectorizedCanonicalBasis = basis
        self.obs_probs = observational_probs
        self.variables = variables
        self.verbose = verbose
        self.data_compatibility_map = self._build_data_compatibility_map()

    def _build_data_compatibility_map(self) -> Dict[Tuple[Any, ...], List[int]]:
        """
        Builds a map from each observation to the list of compatible worlds.
        It is the equivalent of checking the consistency of the observational data with the causal order.
        """
        compatibility = {}
        for joint_assignment in self.obs_probs.keys():
            # Convert tuple observation to Event object
            obs_event = Event(
                {
                    CounterfactualTerm(var, {}): val
                    for var, val in zip(self.variables, joint_assignment)
                }
            )

            compatibility[joint_assignment] = np.where(self.basis.get_mask(obs_event))[
                0
            ].tolist()
        return compatibility

    def solve(
        self,
        query: Union[Query, Expression],
        monotonic: Union[bool, List[Variable]] = False,
    ) -> Tuple[float, float]:
        """
        Computes the Lower and Upper bounds for the given query or expression.

        Args:
            query: The query or expression to solve.
            monotonic: If True, enforces monotonicity constraints (response functions must be non-decreasing) on all variables.
                       If a list of Variables, enforces monotonicity only on those variables.

        Returns:
            A tuple containing the lower and upper bounds (lb, ub).
        """
        # NB: all the queries become expressions after expansion, so we can handle them in a unified way.
        # TODO: could we write everything in terms of Expressions from the start and avoid this check? The main reason to keep Query is for user-friendly syntax, but maybe we can have Query be a thin wrapper around Expression that just handles the initial parsing and expansion.

        if isinstance(query, Query):
            expression = query.expand()
        elif isinstance(query, Expression):
            # If the user passes an Expression directly, we assume it might
            # still contain unexpanded queries, so we expand each term.
            new_terms = {}
            for q, w in query.terms.items():
                expanded_sub = q.expand()
                for sub_q, sub_w in expanded_sub.terms.items():
                    new_terms[sub_q] = new_terms.get(sub_q, 0.0) + sub_w * w
            expression = Expression(new_terms)
        else:
            raise TypeError("Input must be a Query or an Expression.")

        # Check for evidence consistency (all sub-queries must share the same evidence)
        if not expression.terms:
            return 0.0, 0.0

        first_term_query = next(iter(expression.terms))
        first_evidence = first_term_query.evidence

        for q in expression.terms:
            if q.evidence != first_evidence:
                raise ValueError(
                    f"All terms in an expression must share the same evidence.\n"
                    f"Found differing evidences: {first_evidence} vs {q.evidence}"
                )

        # If evidence is present, use Charnes-Cooper transformation for P(gamma | delta)
        if first_evidence:
            _, lb = self._solve_linear_fractional(
                expression, first_evidence, sense=pulp.LpMinimize, monotonic=monotonic
            )
            _, ub = self._solve_linear_fractional(
                expression, first_evidence, sense=pulp.LpMaximize, monotonic=monotonic
            )
        else:
            # If no evidence, solve standard LP
            _, lb = self._solve_standard_lp(
                expression, sense=pulp.LpMinimize, monotonic=monotonic
            )
            _, ub = self._solve_standard_lp(
                expression, sense=pulp.LpMaximize, monotonic=monotonic
            )

        return lb, ub

    def _get_monotonicity_mask(
        self, monotonic: Union[bool, List[Variable]]
    ) -> np.ndarray:
        """
        Identifies worlds where the response functions are monotonic (non-decreasing)
        with respect to their parents.

        Args:
            monotonic: Boolean or list of variables to check.

        Returns:
            A boolean array (N_worlds,) where True indicates the world satisfies monotonicity.
        """
        valid_worlds_mask = np.ones(self.basis.n_worlds, dtype=bool)

        # Determine which variables to check
        if monotonic is True:
            vars_to_check = set(self.basis.variables)
        elif monotonic is False or monotonic is None:
            return valid_worlds_mask  # All valid (no constraints)
        else:
            vars_to_check = set(monotonic)

        for i, var in enumerate(self.basis.variables):
            if var not in vars_to_check:
                continue

            parents = self.basis.variables[:i]
            if not parents:
                continue

            strides = self.basis.parent_strides[i]
            table = self.basis.func_tables[i]  # (N_worlds, N_configs)
            n_configs = table.shape[1]

            for p_idx, parent in enumerate(parents):
                stride = strides[p_idx]
                dom_size = len(parent.domain)

                for col in range(n_configs):
                    # Check if parent value can be incremented
                    val_p = (col // stride) % dom_size
                    if val_p < dom_size - 1:
                        col_next = col + stride
                        # Monotonicity: f(x) <= f(x') when x <= x'
                        # So we require table[col] <= table[col_next]
                        # Invalid if table[col] > table[col_next]
                        valid_worlds_mask &= table[:, col] <= table[:, col_next]

        return valid_worlds_mask

    def _solve_standard_lp(
        self,
        expression: Expression,
        sense: int,
        monotonic: Union[bool, List[Variable]] = False,
    ) -> Tuple[pulp.LpProblem, float]:
        """
        Implements standard LP over the sigma signature.

        Args:
            expression: The query or expression to solve.
            sense: pulp.LpMinimize or pulp.LpMaximize.
            monotonic: If True, enforce monotonicity.

        Returns:
            A tuple containing the solved LP problem and the optimal value.
        """
        # 1. Define the Problem
        prob = pulp.LpProblem("Counterfactual_Bounding", sense)

        # 2. Decision Variables
        q_vars = [
            pulp.LpVariable(f"q_{i}", lowBound=0) for i in range(self.basis.n_worlds)
        ]

        # 3. Objective Function: P(gamma) = sum(q_omega where omega |- gamma)
        # We handle weights from the expression.
        objective_terms = []
        for q, w in expression.terms.items():
            mask = self.basis.get_mask(q.target)
            indices = np.where(mask)[0]
            for i in indices:
                objective_terms.append(w * q_vars[i])

        prob += pulp.lpSum(objective_terms)

        # 4. Constraints

        # A. Unit Sum Constraint: sum(q) = 1
        prob += pulp.lpSum(q_vars) == 1.0, "Normalisation_Probability"

        # B. Observational Consistency: sum(q where omega |= v) = P(v)
        # Theorem 3.2
        for row, prob_val in self.obs_probs.items():
            compatible_indices = self.data_compatibility_map[row]
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

        # C. Monotonicity Constraints
        if monotonic:
            valid_mask = self._get_monotonicity_mask(monotonic)
            points_to_zero = np.where(~valid_mask)[0]
            for i in points_to_zero:
                # Enforce q_i = 0 for non-monotonic worlds
                prob += q_vars[i] == 0, f"Monotonicity_Exclusion_{i}"

        # 5. Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=self.verbose))

        if prob.status != pulp.LpStatusOptimal:
            return prob, float("nan")  # Or handle infeasibility appropriately

        return prob, pulp.value(prob.objective)

    def _solve_linear_fractional(
        self,
        expression: Expression,
        evidence: Event,
        sense: int,
        monotonic: Union[bool, List[Variable]] = False,
    ) -> Tuple[pulp.LpProblem, float]:
        """
        Implements Charnes-Cooper transformation for Conditional Queries.
        """
        prob = pulp.LpProblem("Conditional_Bounding", sense)

        # 1. Transformed Variables
        # q_prime corresponds to q_omega * t
        q_prime_vars: list[pulp.LpVariable] = [
            pulp.LpVariable(f"q_prime_{i}", lowBound=0)
            for i in range(self.basis.n_worlds)
        ]
        # t is the inverse of the probability of the evidence P(delta)
        t = pulp.LpVariable("t", lowBound=0)

        # 2. Objective: P(gamma & delta) * t
        # (Transformed numerator)
        objective_terms = []
        for q, w in expression.terms.items():
            # For conditional P(T|E), the numerator is P(T, E).
            event_to_measure = q.target & evidence

            mask = self.basis.get_mask(event_to_measure)
            indices = np.where(mask)[0]
            for i in indices:
                objective_terms.append(w * q_prime_vars[i])

        prob += pulp.lpSum(objective_terms)

        # 3. Constraints

        # A. Denominator Constraint: sum(q_prime where omega |- evidence) = 1
        # This ensures that we are normalizing by P(evidence)
        # Expand evidence to handle any nested counterfactuals properly
        expanded_evidence = evidence.expand()
        evidence_mask = self.basis.get_mask(expanded_evidence[0])
        for ev in expanded_evidence[1:]:
            evidence_mask = np.logical_or(evidence_mask, self.basis.get_mask(ev))

        evidence_indices = np.where(evidence_mask)[0].tolist()
        prob += (
            pulp.lpSum([q_prime_vars[i] for i in evidence_indices]) == 1.0,
            "Charnes_Cooper_Norm",
        )

        # B. Unit Sum Transformed: sum(q_prime) = t
        prob += pulp.lpSum(q_prime_vars) == t, "Sum_t"

        # C. Observational Constraints Transformed: sum(q_prime) = P(v) * t
        for row, prob_val in self.obs_probs.items():
            compatible_indices = self.data_compatibility_map[row]
            if compatible_indices:
                # LHS: sum of transformed vars
                lhs = pulp.lpSum([q_prime_vars[i] for i in compatible_indices])
                # RHS: prob_val * t
                prob += lhs == prob_val * t, f"Obs_Transformed_{row}"

        # D. Monotonicity Constraints
        if monotonic:
            valid_mask = self._get_monotonicity_mask(monotonic)
            points_to_zero = np.where(~valid_mask)[0]
            for i in points_to_zero:
                # In Charnes-Cooper, q_i = 0 implies q_prime_i = 0 (since t >= 0)
                prob += q_prime_vars[i] == 0, f"Monotonicity_Exclusion_Prime_{i}"

        # 4. Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status != pulp.LpStatusOptimal:
            return float("nan")

        return prob, pulp.value(prob.objective)
