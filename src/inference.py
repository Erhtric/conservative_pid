"""
High-level entry point for inference.
Partial order extraction, permutation generation, bound aggregation
"""

from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from .canonical import VectorizedCanonicalBasis
from .solver import LPSolver
from .symbolic import (
    CounterfactualTerm,
    Expression,
    Query,
    Variable,
    MonotonicityConstraint,
)


class ConservativePID:
    """
    Conservative PID inference engine.
    """

    def __init__(
        self,
        observational_probs: pd.DataFrame,
    ):
        """
        Initializes the Conservative PID inference engine.

        Args:
            observational_probs: A pandas DataFrame containing the observational probabilities. Must have columns corresponding to variable names and a 'probability' column.
        """
        self.observational_probs: pd.DataFrame = observational_probs
        self.monotonicity_constraints: List[MonotonicityConstraint] = []

    def add_monotonicity(self, constraint: MonotonicityConstraint) -> None:
        """
        Adds a monotonicity constraint to the inference engine.

        Args:
            constraint: A MonotonicityConstraint object.
        """
        self.monotonicity_constraints.append(constraint)

    def infer(
        self,
        query: Union[Query, Expression],
        causal_order: Optional[List[Variable]] = None,
        return_problems: bool = False,
    ) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
        """
        Computes bounds for a query or expression.

        Args:
            query: A Query or Expression object.
            causal_order: (Optional) A list of Variable objects, e.g. [X, Z, Y], representing
                            the known topological order. If provided, skips permutation search.
            return_problems: Whether to return the pulp LP problem objects.

        Returns:
            If return_problems is False: (lb, ub)
            If return_problems is True: (lb, ub, {"lower": lower_prob, "upper": upper_prob})
        """
        self._validate_inputs(query)
        logger.success("Input validation passed.")

        var_names_in_data = [
            col for col in self.observational_probs.columns if col != "probability"
        ]

        if causal_order:
            logger.info(f"Using fixed causal order: {[v.name for v in causal_order]}")

            ordered_vars = []
            for item in causal_order:
                if isinstance(item, Variable):
                    if not item.domain and item.name in var_names_in_data:
                        domain = tuple(
                            sorted(self.observational_probs[item.name].unique())
                        )
                        ordered_vars.append(Variable(item.name, domain))
                    else:
                        ordered_vars.append(item)
                else:
                    raise TypeError(
                        f"Invalid type in causal_order: {type(item)}. Expected Variable."
                    )

            # Check if all variables in data are in the causal order
            ordered_var_names = [v.name for v in ordered_vars]
            for name in var_names_in_data:
                if name not in ordered_var_names:
                    raise ValueError(
                        f"Variable '{name}' is in observational data but not in causal_order."
                    )

            return self._solve_for_order(
                ordered_vars, query, return_problems=return_problems
            )

        # 1. Extract Partial Order from Query
        partial_order = self._extract_partial_order(query)

        var_map = {}
        for node in partial_order.nodes:
            var_map[node.name] = node

        for name in var_names_in_data:
            domain = tuple(sorted(self.observational_probs[name].unique()))
            if name in var_map:
                if not var_map[name].domain:
                    var_map[name] = Variable(name, domain)
            else:
                var_map[name] = Variable(name, domain)

        for name in var_names_in_data:
            if name not in [n.name for n in partial_order.nodes]:
                partial_order.add_node(var_map[name])

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
        global_problems: Dict[str, Any] = {"lower": None, "upper": None}

        # 3. Iterate and Aggregate
        for i, order_nodes in enumerate(valid_orders):
            ordered_vars = []
            for node in order_nodes:
                if node.name in var_map and var_map[node.name].domain:
                    ordered_vars.append(var_map[node.name])
                else:
                    raise ValueError(
                        f"Variable '{node.name}' is in the query but not in the observational data, "
                        f"and its domain could not be inferred."
                    )

            res = self._solve_for_order(
                ordered_vars, query, return_problems=return_problems
            )
            if return_problems:
                lb, ub, probs = res
            else:
                lb, ub = res

            if not np.isnan(lb):
                if lb < global_lb:
                    global_lb = lb
                    if return_problems:
                        global_problems["lower"] = probs["lower"]

            if not np.isnan(ub):
                if ub > global_ub:
                    global_ub = ub
                    if return_problems:
                        global_problems["upper"] = probs["upper"]

        if return_problems:
            return global_lb, global_ub, global_problems

        return global_lb, global_ub

    def _solve_for_order(
        self,
        ordered_vars: List[Variable],
        query: Query | Expression,
        return_problems: bool = False,
    ) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
        """
        Helper to run the pipeline for a single specific order.
        """
        logger.debug(f"Solving for order: {[v.name for v in ordered_vars]}")
        basis = VectorizedCanonicalBasis(ordered_vars)
        if self.monotonicity_constraints:
            logger.debug(
                f"Applying {len(self.monotonicity_constraints)} monotonicity constraints. Number of worlds before filtering: {basis.n_worlds}"
            )
            basis.filter_worlds(self.monotonicity_constraints)
            logger.debug(f"Number of worlds after filtering: {basis.n_worlds}")
        solver = LPSolver(basis, self.observational_probs, ordered_vars)
        return solver.solve(query, return_problems=return_problems)

    def _extract_partial_order(self, query: Union[Query, Expression]) -> nx.DiGraph:
        """
        Builds a DAG representing the strict partial order implied by the query.

        Args:
            query: The causal query P(gamma | delta) or an Expression.

        Returns:
            A directed acyclic graph (DAG) representing the strict partial order.
        """
        G = nx.DiGraph()

        variables_in_query = set()

        if isinstance(query, Query):
            queries = [query]
        else:
            queries = list(query.terms.keys())

        for q in queries:
            terms = list(q.target.assignments.keys())
            if q.evidence:
                terms.extend(q.evidence.assignments.keys())

            for term in terms:
                variables_in_query.add(term.variable)
                self._add_subscript_constraints(G, term)

        for var in variables_in_query:
            if not G.has_node(var):
                G.add_node(var)

        return G

    def _add_subscript_constraints(
        self,
        G: nx.DiGraph,
        term: CounterfactualTerm,
    ):
        """
        Recursively adds constraints to the graph based on the counterfactual term.
        Modify in-place the associated DAG.
        Ref: proposition 2.1 of the paper.

        Args:
            G: The graph to add constraints to.
            term: The counterfactual term to add constraints for.
        """
        target = term.variable
        for var, val in term.intervention.items():
            cause = var
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
        total_prob = 0.0

        for _, row in self.observational_probs.iterrows():
            prob: float = row["probability"]

            if prob < 0 or prob > 1:
                raise ValueError(
                    f"Probability {prob} in data is invalid. Must be between 0 and 1."
                )

            total_prob += prob

        if not np.isclose(total_prob, 1.0, atol=1e-5):
            raise ValueError(
                f"Observational probabilities sum to {total_prob}, expected 1.0."
            )

    def __repr__(self):
        return f"ConservativePID(data={self.observational_probs})"
