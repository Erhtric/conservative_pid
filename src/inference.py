"""
High-level entry point for inference.
Partial order extraction, permutation generation, bound aggregation
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pulp
from loguru import logger

from solver import LPSolver, VectorizedCanonicalBasis
from symbolic import CounterfactualTerm, Event, Expression, Query, Variable


class ConservativePID:
    """
    Conservative PID inference engine.
    """

    def __init__(
        self,
        variables: List[Variable],
        observational_probs: Dict[Tuple[Tuple[str, Any], ...], float],
    ):
        """
        Initializes the Conservative PID inference engine.

        Args:
            variables: List of Variable objects representing the SCM. Not necessarily ordered.
            observational_probs: Dictionary mapping tuples of variable (Name, value) to their observational probabilities.
        """
        self.variables: list[Variable] = variables
        self.observational_probs: dict[tuple[tuple[str, Any], ...], int | float] = (
            observational_probs
        )

    def infer(
        self,
        query: Union[Query, Expression],
        causal_order: Optional[List[str]] = None,
        monotonic: Union[bool, List[Variable]] = False,
    ) -> Tuple[float, float]:
        """
        Computes bounds for a query or expression.

        Args:
            query: A Query or Expression object.
            causal_order: (Optional) A list of variable names, e.g. ['X', 'Z', 'Y'], representing
                            the known topological order. If provided, skips permutation search.
                            NB: this only works with variables names for now.
            monotonic: If True, enforces monotonicity on all variables.
                       If a list of Variables, enforces monotonicity only on those variables.

        Returns:
            A tuple containing the lower and upper bounds (lb, ub).
        """
        self._validate_inputs(query)
        logger.success("Input validation passed.")

        if causal_order:
            logger.info(f"Using fixed causal order: {causal_order}")

            if set(causal_order) != set(v.name for v in self.variables):
                raise ValueError(
                    "Fixed order must contain exactly all defined variables."
                )

            # Convert names to Variable objects
            # TODO: add the possibility pass a list of Variable objects instead of names
            var_map = {v.name: v for v in self.variables}
            ordered_vars = [var_map[name] for name in causal_order]

            # TODO: check if the fixed order allows the query

            return self._solve_for_order(ordered_vars, query, monotonic=monotonic)

        # NO ORDER KNOWN
        # 1. Extract Partial Order from Query
        partial_order: nx.DiGraph = self._extract_partial_order(query)

        # 2. Generate Linear Extensions
        try:
            valid_orders = list(nx.all_topological_sorts(partial_order))
        except nx.NetworkXUnfeasible:
            raise ValueError(
                "Query implies a cyclic dependency, which is impossible in a recursive SCM."
            )

        logger.info(
            f"Found {len(valid_orders)} valid causal orders compatible with the query."
        )

        global_lb = float("inf")
        global_ub = float("-inf")

        # 3. Iterate and Aggregate
        for i, order_names in enumerate(valid_orders):
            var_map: dict[str, Variable] = {v.name: v for v in self.variables}
            ordered_vars: list[Variable] = [var_map[name] for name in order_names]

            lb, ub = self._solve_for_order(ordered_vars, query, monotonic=monotonic)

            # Update Global Bounds (Algorithm 2: min of LBs, max of UBs)
            if not np.isnan(lb):
                global_lb = min(global_lb, lb)
            if not np.isnan(ub):
                global_ub = max(global_ub, ub)

        return global_lb, global_ub

    def _solve_for_order(
        self,
        ordered_vars: List[Variable],
        query: Query | Expression,
        monotonic: Union[bool, List[Variable]] = False,
    ) -> Tuple[float, float]:
        """
        Helper to run the pipeline for a single specific order.
        """
        logger.debug(
            f"Solving for order: {[v.name for v in ordered_vars]} with monotonic={monotonic}"
        )
        basis = VectorizedCanonicalBasis(ordered_vars)
        solver = LPSolver(basis, self.observational_probs, ordered_vars)
        lb, ub = solver.solve(query, monotonic=monotonic)
        return lb, ub

    def _extract_partial_order(self, query: Union[Query, Expression]) -> nx.DiGraph:
        """
        Builds a DAG representing the strict partial order implied by the query.

        Args:
            query: The causal query P(gamma | delta) or an Expression.

        Returns:
            A directed acyclic graph (DAG) representing the strict partial order.
        """
        G = nx.DiGraph()
        G.add_nodes_from([v.name for v in self.variables])

        if isinstance(query, Query):
            queries = [query]
        else:
            queries = list(query.terms.keys())

        for q in queries:
            terms = list(q.target.assignments.keys())
            if q.evidence:
                terms.extend(q.evidence.assignments.keys())

            for term in terms:
                self._add_subscript_constraints(G, term)

        return G

    def _add_subscript_constraints(self, G: nx.DiGraph, term: CounterfactualTerm):
        """
        Recursively adds constraints to the graph based on the counterfactual term.
        Modify in-place the associated DAG.
        Ref: proposition 2.1 of the paper.

        Args:
            G: The graph to add constraints to.
            term: The counterfactual term to add constraints for.
        """
        target = term.variable.name
        for var, val in term.intervention.items():
            cause = var.name
            if not G.has_edge(cause, target):
                G.add_edge(cause, target)
            if isinstance(val, CounterfactualTerm):
                self._add_subscript_constraints(G, val)

    def _validate_inputs(self, query: Union[Query, Expression]):
        """
        Performs validation on variables, data, and the query.

        Args:
            query: The causal query P(gamma | delta).
        """
        # 1. Check Variables
        for var in self.variables:
            if var.domain is None or len(var.domain) == 0:
                raise ValueError(f"Variable '{var.name}' must have a non-empty domain.")

        # 2. Check Observational Data
        expected_len = len(self.variables)
        total_prob = 0.0

        for row, prob in self.observational_probs.items():
            if len(row) != expected_len:
                raise ValueError(
                    f"Data row {row} has length {len(row)}, expected {expected_len} (variables: {[v.name for v in self.variables]})."
                )
            # Check domain compatibility
            for i, val in enumerate(row):
                if val not in self.variables[i].domain:
                    raise ValueError(
                        f"Value '{val}' in data row {row} is not in domain of variable '{self.variables[i].name}'."
                    )

            if prob < 0 or prob > 1:
                raise ValueError(
                    f"Probability {prob} in data is invalid. Must be between 0 and 1."
                )

            total_prob += prob

        if not np.isclose(total_prob, 1.0, atol=1e-5):
            raise ValueError(
                f"Observational probabilities sum to {total_prob}, expected 1.0."
            )

        # 3. Check Query or Expression
        known_vars = set(self.variables)

        # Helper to check an event
        def check_event(event: Event, context: str):
            for term, val in event.assignments.items():
                if term.variable not in known_vars:
                    raise ValueError(
                        f"Unknown variable '{term.variable.name}' in query {context}."
                    )

                # Recursively check intervention variables
                self._check_intervention_vars(term, known_vars)

        def validate_single_query(q: Query):
            if not isinstance(q.target, Event):
                raise TypeError("Query target must be an Event object.")
            check_event(q.target, "target")

            if q.evidence:
                if not isinstance(q.evidence, Event):
                    raise TypeError("Query evidence must be an Event object.")
                check_event(q.evidence, "evidence")

        if isinstance(query, Expression):
            for sub_query in query.terms.keys():
                validate_single_query(sub_query)
        elif isinstance(query, Query):
            validate_single_query(query)
        else:
            raise TypeError("Input must be a Query or an Expression object.")

    def _check_intervention_vars(self, term: CounterfactualTerm, known_vars: set):
        for var, val in term.intervention.items():
            if var not in known_vars:
                raise ValueError(
                    f"Unknown intervention variable '{var.name}' in term '{term}'."
                )
            if isinstance(val, CounterfactualTerm):
                self._check_intervention_vars(val, known_vars)

    @staticmethod
    def marginalize_data(
        data: Dict[Tuple[Any, ...], float],
        all_variables: List[Variable],
        target_variables: List[Variable],
    ) -> Dict[Tuple[Any, ...], float]:
        """
        Marginalizes the observational data to the subset of target variables.

        Args:
            data: The full observational data dictionary.
            all_variables: The list of variables corresponding to the full data tuples.
            target_variables: The subset of variables to keep.

        Returns:
            A new data dictionary with keys corresponding to target_variables and aggregated probabilities.
        """
        if not set(target_variables).issubset(set(all_variables)):
            raise ValueError("Target variables must be a subset of all variables.")

        # Find indices of target variables in the original list
        indices = [all_variables.index(var) for var in target_variables]

        new_data = {}

        for full_tuple, prob in data.items():
            # Extract sub-tuple
            sub_tuple = tuple(full_tuple[i] for i in indices)
            new_data[sub_tuple] = new_data.get(sub_tuple, 0.0) + prob

        return new_data

    def __repr__(self):
        return f"ConservativePID(variables={self.variables}, data={self.observational_probs})"
