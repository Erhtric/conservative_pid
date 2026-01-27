"""
Translates the Canonical Representation into a Linear Program using PuLP.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pulp

from canonical import VectorizedCanonicalBasis
from symbolic import CounterfactualTerm, Event, Expression, Query, Variable


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

        Args:
            basis: The list of all deterministic worlds (Omega).
            observational_data: A dictionary mapping variable value tuples to probabilities.
                                e.g., {(0, 1): 0.4, ...} for variables (X, Y).
            variables: The list of variables corresponding to the data tuples.
            verbose: Whether to print verbose output.

        Raises:
            ValueError: If the observational data is inconsistent with the causal order.
        """
        self.basis = basis
        self.data = observational_data
        self.variables = variables
        self.verbose = verbose
        self.data_compatibility_map = self._build_data_compatibility_map()

    def _build_data_compatibility_map(self) -> Dict[Tuple[Any, ...], List[int]]:
        """
        Builds a map from each observation to the list of compatible worlds.
        It is the equivalent of checking the consistency of the observational data with the causal order.
        """
        compatibility = {}
        for joint_assignment in self.data.keys():
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

    def solve(self, query: Union[Query, Expression]) -> Tuple[float, float]:
        """
        Computes the Lower and Upper bounds for the given query or expression.

        Args:
            query: The query or expression to solve.

        Returns:
            A tuple containing the lower and upper bounds.
        """
        # Ensure we are working with an Expression (a collection of atomic queries)
        if isinstance(query, Query):
            expression = query.expand()
        else:
            # If the user passes an Expression directly, we assume it might
            # still contain unexpanded queries, so we expand each term.
            new_terms = {}
            for q, w in query.terms.items():
                expanded_sub = q.expand()
                for sub_q, sub_w in expanded_sub.terms.items():
                    new_terms[sub_q] = new_terms.get(sub_q, 0.0) + sub_w * w
            expression = Expression(new_terms)

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
                expression, first_evidence, sense=pulp.LpMinimize
            )
            _, ub = self._solve_linear_fractional(
                expression, first_evidence, sense=pulp.LpMaximize
            )
        else:
            # If no evidence, solve standard LP
            _, lb = self._solve_standard_lp(expression, sense=pulp.LpMinimize)
            _, ub = self._solve_standard_lp(expression, sense=pulp.LpMaximize)

        return lb, ub

    def _solve_standard_lp(
        self, expression: Expression, sense
    ) -> Tuple[pulp.LpStatus, float]:
        """
        Implements Algorithm 1 (Def 3.4).
        Min/Max sum(q_omega) subject to observational constraints.
        Extended to handle Expressions.

        Args:
            query: The query or expression to solve.
            sense: pulp.LpMinimize or pulp.LpMaximize.

        Returns:
            The lower and upper bounds.
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
        for row, prob_val in self.data.items():
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

        # 5. Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=self.verbose))

        if prob.status != pulp.LpStatusOptimal:
            return prob, float("nan")  # Or handle infeasibility appropriately

        return prob, pulp.value(prob.objective)

    def _solve_linear_fractional(
        self, expression: Expression, evidence: Event, sense
    ) -> Tuple[pulp.LpStatus, float]:
        """
        Implements Charnes-Cooper transformation for Conditional Queries.
        Section 5.1 [cite: 231-235].
        """
        prob = pulp.LpProblem("Conditional_Bounding", sense)

        # 1. Transformed Variables
        # q_prime corresponds to q_omega * t
        q_prime_vars = [
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
        for row, prob_val in self.data.items():
            compatible_indices = self.data_compatibility_map[row]
            if compatible_indices:
                # LHS: sum of transformed vars
                lhs = pulp.lpSum([q_prime_vars[i] for i in compatible_indices])
                # RHS: prob_val * t
                prob += lhs == prob_val * t, f"Obs_Transformed_{row}"

        # 4. Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status != pulp.LpStatusOptimal:
            return float("nan")

        return prob, pulp.value(prob.objective)
