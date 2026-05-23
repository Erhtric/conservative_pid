import pandas as pd
import pulp

from .signature import (
    TotalOrderSignature,
    PartialOrderSignature,
    SignatureQueryEvaluator,
)
from .io import CausalExpression, CausalQuery


class OrderFunctionalLPSolver:
    """
    Formulates and solves the linear program. We require some data, a query to evaluate and optionally an
    order over variables. If the order is not provided we will infer the order from the query.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        query: CausalQuery | CausalExpression,
        order: list[str] | None = None,
        solver_verbose: bool = False,
        domains: dict[str, int] | None = None,
        lazy: bool = True,
    ):
        # Force integer type to prevent type mismatches with the evaluator's internal states
        try:
            data = data.astype(int)
        except ValueError:
            raise ValueError(
                "Data must be coercible to integers to match signature domains."
            )

        vars_in_data = set(data.columns)

        # Permit explicit domain definitions to avoid inference failures on missing states
        if domains is not None:
            self.domains = domains
        else:
            self.domains = {v: data[v].nunique() for v in vars_in_data}

        self.solver_verbose = solver_verbose

        query = self._flatten_query_or_expr(query)

        if order:
            if not set(order).issubset(vars_in_data):
                missing = set(order) - vars_in_data
                raise ValueError(
                    f"Provided order contains variables not in data: {missing}"
                )
            self.signature_obj = TotalOrderSignature(
                domains=self.domains, total_order=order, lazy=lazy
            )

            try:
                self.signature_obj.is_compatible(query)
            except ValueError as e:
                raise ValueError(f"Provided order is not compatible with query: {e}")
        else:
            query_order = query.induced_order()
            query_vars = set(query_order.nodes())
            if not query_vars.issubset(vars_in_data):
                missing = query_vars - vars_in_data
                raise ValueError(
                    f"Query-induced order contains variables not in data: {missing}"
                )

            self.signature_obj = PartialOrderSignature(
                domains=self.domains, query=query, lazy=lazy
            )

        self.data_dist = (
            data[
                self.signature_obj.ordered_nodes
            ]  # Order and marginalise to signature variables
            .value_counts(normalize=True)
            .to_dict()
        )

        self.query = query

    def _flatten_query_or_expr(
        self, qobj: CausalQuery | CausalExpression
    ) -> CausalExpression:
        """Return a CausalExpression representing the (possibly unnested) linear combination."""
        if isinstance(qobj, CausalQuery):
            print(f"Unnesting query {qobj}")
            u = qobj.unnest(self.domains)
            if isinstance(u, CausalQuery):
                return CausalExpression({u: 1.0})
            return u
        # CausalExpression: unnest each term and merge
        terms: dict[CausalQuery, float] = {}
        for cq, w in qobj.terms.items():
            print(f"Unnesting term {cq} with weight {w}")
            u = cq.unnest(self.domains)
            if isinstance(u, CausalQuery):
                terms[u] = terms.get(u, 0.0) + w
            else:  # CausalExpression
                for inner_cq, inner_w in u.terms.items():
                    terms[inner_cq] = terms.get(inner_cq, 0.0) + w * inner_w
        return CausalExpression(terms)

    def __call__(
        self,
        return_lp: bool = False,
    ) -> tuple[float, float] | tuple[tuple[float, float], list[pulp.LpProblem]]:
        """
        Solves the LP to compute bounds for the query.

        For conditional queries, applies the Charnes-Cooper transformation to convert the fractional program into a
        linear one. Constructs the LP constraints based on the equivalence classes derived from the MDD paths,
        ensuring that the solution space is correctly defined by the observed data distribution and the query structure.

        Args:
            return_lp: If True, also return the raw LP problem objects for inspection.

        Returns:
            A tuple of (lower_bound, upper_bound) for the query, and optionally the list of LP problem objects.
        """

        # Group signature rows into equivalence class:
        # Observational consistency
        # Numerator consistency (satisfies same query terms)
        # Denominator consistency (satisfies evidence if conditional)
        eq_classes = self.signature_obj.get_equivalence_classes(self.query)

        bounds = []
        pulp_probs = []

        # Create mapping from equivalence class keys to indices for LP variables
        eq_class_keys = list(eq_classes.keys())
        eq_class_to_idx = {key: idx for idx, key in enumerate(eq_class_keys)}

        for sense in [pulp.LpMinimize, pulp.LpMaximize]:
            prob: pulp.LpProblem = pulp.LpProblem("Partial_ID", sense)

            if self.query.is_conditional():
                print(
                    "Detected conditional query. Applying Charnes-Cooper transformation."
                )

                y_vars: dict[int, pulp.LpVariable] = pulp.LpVariable.dicts(
                    "y", range(len(eq_classes)), lowBound=0
                )
                t = pulp.LpVariable("t", lowBound=0)

                for obs_vals, obs_prob in self.data_dist.items():
                    matching_classes = [
                        eq_class_to_idx[key]
                        for key in eq_class_keys
                        if key[0] == obs_vals
                    ]
                    if matching_classes:
                        prob += (
                            pulp.lpSum([y_vars[i] for i in matching_classes])
                            - obs_prob * t
                            == 0
                        )

                prob += pulp.lpSum(y_vars.values()) - t == 0

                # Denominator constraint leverages the computed coefficients (key[2] is den_coeff)
                denom_terms = [
                    key[2] * y_vars[eq_class_to_idx[key]]
                    for key in eq_class_keys
                    if key[2] > 0
                ]
                prob += pulp.lpSum(denom_terms) == 1.0

                # Objective Function (key[1] is num_coeff)
                obj_terms = [
                    key[1] * y_vars[eq_class_to_idx[key]]
                    for key in eq_class_keys
                    if key[1] != 0
                ]
                prob += pulp.lpSum(obj_terms)

                prob.solve(pulp.PULP_CBC_CMD(msg=self.solver_verbose))
                bounds.append(pulp.value(prob.objective))
                pulp_probs.append(prob)

            else:
                q_vars: dict[int, pulp.LpVariable] = pulp.LpVariable.dicts(
                    "q", range(len(eq_classes)), lowBound=0, upBound=1
                )

                for obs_vals, obs_prob in self.data_dist.items():
                    matching_classes = [
                        eq_class_to_idx[key]
                        for key in eq_class_keys
                        if key[0] == obs_vals
                    ]
                    if matching_classes:
                        prob += (
                            pulp.lpSum([q_vars[i] for i in matching_classes])
                            == obs_prob
                        )

                prob += pulp.lpSum(q_vars.values()) == 1.0

                # Objective Function (key[1] is num_coeff)
                obj_terms = [
                    key[1] * q_vars[eq_class_to_idx[key]]
                    for key in eq_class_keys
                    if key[1] != 0
                ]

                prob += pulp.lpSum(obj_terms)
                prob.solve(pulp.PULP_CBC_CMD(msg=self.solver_verbose))
                bounds.append(pulp.value(prob.objective))
                pulp_probs.append(prob)

        if return_lp:
            return tuple(bounds), pulp_probs

        return tuple(bounds)

    def compute_bounds_explicit(
        self,
        return_lp: bool = False,
    ) -> tuple[float, float] | tuple[tuple[float, float], list[pulp.LpProblem]]:
        """
        Explicitly constructs the LP using the full path enumeration from the MDD, without pre-grouping into equivalence classes.

        Args:
            return_lp: If True, also return the raw LP problem objects for inspection.

        Returns:
            A tuple of (lower_bound, upper_bound) for the query, and optionally the list of LP problem objects.
        """
        bounds = []
        pulp_probs = []
        evaluator = SignatureQueryEvaluator(
            self.domains, self.signature_obj, self.query
        )

        matching_indices_per_obs = {
            obs_vals: evaluator.get_matching_row_indices(obs_vals)
            for obs_vals in self.data_dist.keys()
        }

        satisfying_indices_per_term = {
            cq: evaluator.get_satisfying_row_indices(cq)
            for cq in self.query.terms.keys()
        }

        denom_indices = None
        if self.query.is_conditional():
            evidence_dict = {}
            if isinstance(self.query, CausalExpression):
                # Extract evidence from first term (they should all have same evidence)
                for cq, _ in self.query.terms.items():
                    evidence_dict = cq.evidence if cq.evidence else {}
                    break

            if evidence_dict:
                evidence_query = CausalQuery(counterfactuals=[], evidence=evidence_dict)
                denom_indices = evaluator.get_satisfying_row_indices(evidence_query)
            else:
                denom_indices = list(range(self.signature_obj.size))

        for sense in [pulp.LpMinimize, pulp.LpMaximize]:
            prob: pulp.LpProblem = pulp.LpProblem("Partial_ID", sense)

            if self.query.is_conditional():
                print(
                    "Detected conditional query. Applying Charnes-Cooper transformation."
                )
                # Conditional query: apply Charnes-Cooper transformation.
                # Variables: y_i (for each world) and scalar t >= 0
                y_vars = pulp.LpVariable.dicts(
                    "y", range(self.signature_obj.size), lowBound=0
                )
                t = pulp.LpVariable("t", lowBound=0)

                # Observational constraints scaled by t: sum_{i in matching} y_i - obs_prob * t == 0
                for obs_vals, obs_prob in self.data_dist.items():
                    constraint_expr = [
                        y_vars[i] for i in matching_indices_per_obs[obs_vals]
                    ]
                    prob += pulp.lpSum(constraint_expr) - obs_prob * t == 0

                # Simplex constraint transformed: sum_i y_i - t == 0
                prob += pulp.lpSum(y_vars.values()) - t == 0

                # Denominator constraint: d^T y == 1
                prob += pulp.lpSum([y_vars[i] for i in denom_indices]) == 1.0

                # Objective: c^T y where c picks rows satisfying numerator (gamma ∧ delta)
                obj_terms = []
                for cq, w in self.query.terms.items():
                    for i in satisfying_indices_per_term[cq]:
                        obj_terms.append(w * y_vars[i])

                prob += pulp.lpSum(obj_terms)
                prob.solve(pulp.PULP_CBC_CMD(msg=self.solver_verbose))
                bounds.append(pulp.value(prob.objective))
                pulp_probs.append(prob)
            else:
                q_vars: dict[int, pulp.LpVariable] = pulp.LpVariable.dicts(
                    "q", range(self.signature_obj.size), lowBound=0, upBound=1
                )
                for obs_vals, obs_prob in self.data_dist.items():
                    constraint_expr = [
                        q_vars[i] for i in matching_indices_per_obs[obs_vals]
                    ]
                    prob += pulp.lpSum(constraint_expr) == obs_prob

                prob += pulp.lpSum(q_vars.values()) == 1.0

                # Objective Function
                obj_terms = []
                for cq, w in self.query.terms.items():
                    satisfying_indices = satisfying_indices_per_term[cq]
                    for i in satisfying_indices:
                        obj_terms.append(w * q_vars[i])

                prob += pulp.lpSum(obj_terms)
                prob.solve(pulp.PULP_CBC_CMD(msg=self.solver_verbose))
                bounds.append(pulp.value(prob.objective))
                pulp_probs.append(prob)

        if return_lp:
            return tuple(bounds), pulp_probs

        return tuple(bounds)

    def __str__(self):
        return f"""OrderFunctionalLPSolver:
    Signature: {self.signature_obj} and query: {self.query}
    Domains: {self.domains}
    Data distribution: {self.data_dist}"""

    def __repr__(self):
        return self.__str__()
