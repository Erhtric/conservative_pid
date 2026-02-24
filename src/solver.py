from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pulp
import pandas as pd

from .canonical import VectorizedCanonicalBasis
from .symbolic import CounterfactualTerm, Event, Expression, Query, Variable


class LPSolver:
    def __init__(
        self,
        basis: VectorizedCanonicalBasis,
        observational_probs: pd.DataFrame,
        variables: List[Variable],
        experimental_probs: Optional[pd.DataFrame] = None,
        verbose: bool = False,
    ):
        """
        Initializes the solver with the pre-computed canonical basis and observational/experimental data.

        Args:
            basis: The list of all deterministic worlds (Omega).
            observational_probs: A pandas DataFrame with columns corresponding to variable names and a 'probability' column.
                                e.g., DataFrame with columns ['X', 'Y', 'probability'].
            variables: The list of variables corresponding to the data tuples (ordered by causal order).
            experimental_probs: Optional DataFrame with interventional data. May have a subset of variables (marginalization performed).
            verbose: Whether to print verbose output.

        Raises:
            ValueError: If the observational data is inconsistent with the causal order.
        """
        self.basis: VectorizedCanonicalBasis = basis
        self.obs_probs = observational_probs
        self.exp_probs = experimental_probs
        self.variables = variables
        self.verbose = verbose
        self.obs_compatibility_map = self._build_compatibility_map(
            self.obs_probs, "observational"
        )
        self.exp_compatibility_map = (
            self._build_compatibility_map(self.exp_probs, "experimental")
            if self.exp_probs is not None
            else None
        )

    def _build_compatibility_map(
        self, data: pd.DataFrame, data_type: str = "observational"
    ) -> Dict[Tuple[Any, ...], List[int]]:
        """
        Builds a map from each data row to the list of compatible worlds.

        Handles BOTH observational and experimental data:
        - Observational: All variables in `self.variables` must be present.
        - Experimental: A subset of variables is allowed; marginalization is implicit in the LP.

        Args:
            data: The probability DataFrame to map.
            data_type: Either "observational" or "experimental" for logging context.

        Returns:
            Dictionary mapping tuples of observed values (for present variables) to world indices.

        Raises:
            ValueError: If observational data has missing variables or impossible assignments.
        """
        compatibility: Dict[Tuple[Any, ...], List[int]] = {}

        # Determine which variables are present in the data
        data_vars = [v for v in self.variables if v.name in data.columns]

        if data_type == "observational":
            # Strict: all variables must be in observational data
            missing_vars = [
                v.name for v in self.variables if v.name not in data.columns
            ]
            if missing_vars:
                raise ValueError(
                    f"Observational data is missing variables {missing_vars}. "
                    f"All variables in causal order must be present in observational data."
                )

        for _, row in data.iterrows():
            # Extract only the values for variables present in this data
            joint_assignment = tuple(row[v.name] for v in data_vars)

            # Build the event: only for variables in data_vars
            obs_event = Event(
                {
                    CounterfactualTerm(var, {}): val
                    for var, val in zip(data_vars, joint_assignment)
                }
            )

            # Get world indices where the event holds
            compatible_indices = np.where(self.basis.get_mask(obs_event))[0].tolist()

            if data_type == "observational" and not compatible_indices:
                prob_val = row.get("probability", None)
                if prob_val and prob_val > 0:
                    raise ValueError(
                        f"Data row {dict(row)} is impossible under the current causal order."
                    )
                continue
            elif data_type == "experimental" and not compatible_indices:
                # Log but do not error for experimental data (it may be strictly narrower than canonical basis)
                from loguru import logger

                prob_val = row.get("probability", None)
                logger.warning(
                    f"Experimental data row {dict(row)} has zero compatible worlds. "
                    f"Probability {prob_val} will contribute no constraint."
                )
                continue

            compatibility[joint_assignment] = compatible_indices

        return compatibility

    def solve(
        self,
        query: Union[Query, Expression],
        return_problems: bool = False,
    ) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, pulp.LpProblem]]]:
        """
        Computes the Lower and Upper bounds for the given query or expression.

        Args:
            query: The query or expression to solve.
            return_problems: Whether to return the pulp LP problem objects.

        Returns:
            If return_problems is False: (lb, ub)
            If return_problems is True: (lb, ub, {"lower": lower_prob, "upper": upper_prob})
        """
        if isinstance(query, Query):
            expression = query.expand()
        elif isinstance(query, Expression):
            new_terms = {}
            for q, w in query.terms.items():
                expanded_sub = q.expand()
                for sub_q, sub_w in expanded_sub.terms.items():
                    new_terms[sub_q] = new_terms.get(sub_q, 0.0) + sub_w * w
            expression = Expression(new_terms)
        else:
            raise TypeError("Input must be a Query or an Expression.")

        if not expression.terms:
            if return_problems:
                return 0.0, 0.0, {}
            return 0.0, 0.0

        first_term_query = next(iter(expression.terms))
        first_evidence = first_term_query.evidence

        for q in expression.terms:
            if q.evidence != first_evidence:
                raise ValueError(
                    f"All terms in an expression must share the same evidence.\n"
                    f"Found differing evidences: {first_evidence} vs {q.evidence}"
                )

        if first_evidence:
            prob_lb, lb = self._solve_linear_fractional(
                expression,
                first_evidence,
                sense=pulp.LpMinimize,
            )
            prob_ub, ub = self._solve_linear_fractional(
                expression, first_evidence, sense=pulp.LpMaximize
            )
        else:
            prob_lb, lb = self._solve_standard_lp(expression, sense=pulp.LpMinimize)
            prob_ub, ub = self._solve_standard_lp(expression, sense=pulp.LpMaximize)

        if return_problems:
            return lb, ub, {"lower": prob_lb, "upper": prob_ub}

        return lb, ub

    def _solve_standard_lp(
        self,
        expression: Expression,
        sense: int,
    ) -> Tuple[pulp.LpProblem, float]:
        """
        Implements standard LP over the sigma signature.

        Args:
            expression: The query or expression to solve.
            sense: pulp.LpMinimize or pulp.LpMaximize.

        Returns:
            A tuple containing the solved LP problem and the optimal value.
        """
        # 1. Define the Problem
        prob = pulp.LpProblem("Counterfactual_Bounding", sense)

        # 2. Decision Variables
        q_vars: list[pulp.LpVariable] = [
            pulp.LpVariable(f"q_{i}", lowBound=0) for i in range(self.basis.n_worlds)
        ]

        # 3. Objective Function: P(gamma) = sum(q_omega where omega |- gamma)
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
        obs_vars = [v for v in self.variables if v.name in self.obs_probs.columns]
        for _, row in self.obs_probs.iterrows():
            joint_assignment = tuple(row[v.name] for v in obs_vars)
            prob_val = row["probability"]
            compatible_indices = self.obs_compatibility_map[joint_assignment]
            if not compatible_indices:
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
        self,
        expression: Expression,
        evidence: Event,
        sense: int,
    ) -> Tuple[pulp.LpProblem, float]:
        """
        Implements Charnes-Cooper transformation for Conditional Queries.
        """
        prob = pulp.LpProblem("Conditional_Bounding", sense)

        # 1. Transformed Variables
        q_prime_vars: list[pulp.LpVariable] = [
            pulp.LpVariable(f"q_prime_{i}", lowBound=0)
            for i in range(self.basis.n_worlds)
        ]
        t = pulp.LpVariable("t", lowBound=0)

        # 2. Objective: P(gamma & delta) * t
        objective_terms = []
        for q, w in expression.terms.items():
            event_to_measure = q.target & evidence

            mask = self.basis.get_mask(event_to_measure)
            indices = np.where(mask)[0]
            for i in indices:
                objective_terms.append(w * q_prime_vars[i])

        prob += pulp.lpSum(objective_terms)

        # 3. Constraints

        # A. Denominator Constraint: sum(q_prime where omega |- evidence) = 1
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
        obs_vars = [v for v in self.variables if v.name in self.obs_probs.columns]
        for _, row in self.obs_probs.iterrows():
            joint_assignment = tuple(row[v.name] for v in obs_vars)
            prob_val = row["probability"]
            compatible_indices = self.obs_compatibility_map[joint_assignment]
            if compatible_indices:
                lhs = pulp.lpSum([q_prime_vars[i] for i in compatible_indices])
                prob += lhs == prob_val * t, f"Obs_Transformed_{row}"

        # 4. Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status != pulp.LpStatusOptimal:
            return float("nan")

        return prob, pulp.value(prob.objective)
