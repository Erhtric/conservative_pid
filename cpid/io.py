from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Union
import networkx as nx


@dataclass(frozen=True, repr=False)
class AtomicCounterfactual:
    """
    Represents a single atomic proposition: e.g., Y_{x=1} = 1. In this case
    target_var = 'Y', target_val = 1, interventions = {'X': 1}.

    We can also represent nested interventions, e.g. Y_{W_{X=0}}=1
    (Y in the submodel where W is intervened as to behaves as X has been intervened to be 0),
    as target_var = 'Y', target_val = 1, interventions = {'W': {'X': 0}}}.
    """

    target_var: str
    target_val: int
    # interventions may be either an integer or
    # a dict describing a nested intervention whose inner interventions
    # should be resolved by un-nesting before evaluation. For example
    # {'Z': 0} represents an inner intervention placeholder.
    interventions: Dict[str, Union[int, Dict[str, int]]] = field(default_factory=dict)

    def __str__(self):
        int_str = ", ".join(f"{k}={v}" for k, v in self.interventions.items())
        if int_str:
            return f"{self.target_var}_{{{int_str}}}={self.target_val}"
        return f"{self.target_var}={self.target_val}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        "If two CTF are identical in target, value, and interventions (regardless of order), they should hash the same."

        def _normalize_val(v):
            if isinstance(v, dict):
                return tuple(sorted(v.items()))
            return v

        items = tuple(
            sorted((k, _normalize_val(v)) for k, v in self.interventions.items())
        )
        return hash((self.target_var, self.target_val, items))

    def is_nested(self) -> bool:
        "Returns True if this atomic counterfactual contains any nested interventions."
        for val in self.interventions.values():
            if isinstance(val, dict):
                return True
        return False


@dataclass(frozen=True, repr=False)
class CausalQuery:
    """
    Represents the full conjunction: P(gamma | delta)
    - counterfactuals: list of AtomicCounterfactual objects
    - evidence: dict of observational evidence (natural state)

    Usage
    -----
    ```python
    #Example: P(Y_{X=1}=1 | Z=0)
    cq = CausalQuery(
        counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})],
        evidence={"Z": 0},
    )
    ```
    """

    counterfactuals: List[AtomicCounterfactual]
    evidence: Dict[str, int] = field(default_factory=dict)

    @property
    def all_variables(self) -> List[str]:
        vars_set: Set[str] = set()
        for cf in self.counterfactuals:
            vars_set.add(cf.target_var)
            vars_set.update(cf.interventions.keys())
        vars_set.update(self.evidence.keys())
        return sorted(list(vars_set))

    def __str__(self):
        gamma = ", ".join(str(cf) for cf in self.counterfactuals)
        delta = ", ".join(f"{k}={v}" for k, v in self.evidence.items())
        if delta:
            return f"P({gamma} | {delta})"
        return f"P({gamma})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        cf_tuple = tuple(self.counterfactuals)
        evidence_items = tuple(sorted(self.evidence.items()))
        return hash((cf_tuple, evidence_items))

    def unnest(self, domains: Dict[str, int]) -> Union[CausalQuery, CausalExpression]:
        """
        Expand any nested interventions into a sum of standard atomic counterfactual
        queries using the law of total probability. For example,

        P(Y_{X_{Z=0}}=y) -> sum_x P(Y_{X=x}=y, X_{Z=0}=x)

        This method returns either a single `CausalQuery` (if no nested
        interventions are present) or a `CausalExpression` representing the
        sum of expanded queries.

        Args:
            domains: A dict mapping variable names to their domain sizes. Required to
                determine how many terms to generate when expanding nested interventions.
                We assume that domain values are in the range [0, domain_size-1].

        Raises:
            KeyError: If a variable appearing in a nested intervention is not present in `domains`.

        Returns:
            A `CausalQuery` if no nested interventions are present, or a `CausalExpression` representing the sum of expanded queries if nested interventions were found.
        """
        # Worklist expansion: iteratively replace the first nested intervention
        # found in each query until none remain.
        work = [self]
        expanded: List[CausalQuery] = []

        while work:
            cq = work.pop()
            found_nested = False
            for cf_idx, atomic in enumerate(cq.counterfactuals):
                for int_var, int_val in atomic.interventions.items():
                    # A nested intervention is represented as a dict, e.g. {'Z': 0}.
                    if isinstance(int_val, dict):
                        found_nested = True
                        nested = int_val
                        inner_var = int_var
                        if inner_var not in domains:
                            raise KeyError(
                                f"Domain for variable '{inner_var}' not provided to unnest()."
                            )
                        domain_size = domains[inner_var]
                        for x in range(domain_size):
                            # copy and replace the atomic with a concrete intervention
                            new_cfs = list(cq.counterfactuals)
                            orig_atomic = atomic
                            new_interventions = dict(orig_atomic.interventions)
                            new_interventions[int_var] = x
                            new_atomic = AtomicCounterfactual(
                                target_var=orig_atomic.target_var,
                                target_val=orig_atomic.target_val,
                                interventions=new_interventions,
                            )
                            new_cfs[cf_idx] = new_atomic

                            # append the inner atomic that asserts the inner variable
                            # took value x under its inner interventions
                            inner_atomic = AtomicCounterfactual(
                                target_var=inner_var,
                                target_val=x,
                                interventions=dict(nested),
                            )
                            new_cfs.append(inner_atomic)

                            new_cq = CausalQuery(
                                counterfactuals=new_cfs, evidence=dict(cq.evidence)
                            )
                            work.append(new_cq)
                        break
                if found_nested:
                    break

            if not found_nested:
                expanded.append(cq)

        if len(expanded) == 1:
            return expanded[0]

        # combine into a CausalExpression with unit weights
        terms = {q: 1.0 for q in expanded}
        return CausalExpression(terms)

    def induced_order(self) -> nx.DiGraph:
        """
        Returns the induced partial order over variables implied by the counterfactuals in this query.
        For example, P(Y_{X=1}=1) implies X < Y, while P(Y_{X_{Z=0}}=1) implies Z < X < Y.

        Raises:
            ValueError: If the induced order contains a cycle, which would indicate an ill-formed query.

        Returns:
            A directed acyclic graph (DAG) represented as a `networkx.DiGraph`, where an edge from A to B indicates that A is an intervention variable for B in at least
        """
        order = nx.DiGraph()
        for cf in self.counterfactuals:
            target = cf.target_var
            order.add_node(target)
            for int_var, int_val in cf.interventions.items():
                if isinstance(int_val, dict):
                    inner_query = CausalQuery(
                        counterfactuals=[
                            AtomicCounterfactual(
                                target_var=int_var,
                                target_val=-1,  # value doesn't matter
                                interventions=int_val,
                            )
                        ],
                        evidence={},
                    )
                    inner_order = inner_query.induced_order()
                    order = nx.compose(order, inner_order)

                order.add_edge(int_var, target)

        for var in self.evidence.keys():
            order.add_node(var)

        try:
            # Check for cycles and raise an error if found, since that would indicate an ill-formed query.
            nx.find_cycle(order)
            raise ValueError("Induced order contains a cycle.")
        except nx.NetworkXNoCycle:
            return order

    def is_conditional(self) -> bool:
        "Returns True if this query has any evidence variables (i.e. is a conditional query)."
        return len(self.evidence) > 0

    def __add__(self, other: Union[CausalQuery, CausalExpression]):
        if isinstance(other, CausalQuery):
            return CausalExpression({self: 1.0, other: 1.0})
        if isinstance(other, CausalExpression):
            terms = other.terms.copy()
            terms[self] = terms.get(self, 0.0) + 1.0
            return CausalExpression(terms)
        return NotImplemented

    def __sub__(self, other: Union[CausalQuery, CausalExpression]):
        if isinstance(other, CausalQuery):
            return CausalExpression({self: 1.0, other: -1.0})
        if isinstance(other, CausalExpression):
            terms = {q: -w for q, w in other.terms.items()}
            terms[self] = terms.get(self, 0.0) + 1.0
            return CausalExpression(terms)
        return NotImplemented

    def __mul__(self, scalar: float):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return CausalExpression({self: float(scalar)})

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __neg__(self):
        return CausalExpression({self: -1.0})


@dataclass(frozen=True, repr=False)
class CausalExpression:
    """
    Represents a linear combination of `CausalQuery` objects.
    Useful to compose and manipulate CausalQuery objects.
    """

    terms: Dict[CausalQuery, float] = field(default_factory=dict)

    def __post_init__(self):
        # All the terms must concord in their evidence
        evidence_sets = [frozenset(cq.evidence.items()) for cq in self.terms.keys()]
        if len(set(evidence_sets)) > 1:
            raise ValueError(
                f"All CausalQuery terms in a CausalExpression must have the same evidence. Found evidence sets: {evidence_sets}"
            )

    @property
    def target_variables(self) -> List[str]:
        vars_set: Set[str] = set()
        for cq in self.terms.keys():
            for cf in cq.counterfactuals:
                vars_set.add(cf.target_var)
        return sorted(list(vars_set))

    @property
    def intervention_variables(self) -> List[str]:
        vars_set: Set[str] = set()
        for cq in self.terms.keys():
            for cf in cq.counterfactuals:
                vars_set.update(cf.interventions.keys())
        return sorted(list(vars_set))

    def __add__(self, other: Union[CausalQuery, CausalExpression]):
        new_terms = self.terms.copy()
        if isinstance(other, CausalQuery):
            new_terms[other] = new_terms.get(other, 0.0) + 1.0
        elif isinstance(other, CausalExpression):
            for q, w in other.terms.items():
                new_terms[q] = new_terms.get(q, 0.0) + w
        else:
            return NotImplemented
        # remove zeros
        new_terms = {q: w for q, w in new_terms.items() if w != 0}
        return CausalExpression(new_terms)

    def __sub__(self, other: Union[CausalQuery, CausalExpression]):
        if isinstance(other, CausalQuery):
            return self + CausalExpression({other: -1.0})
        if isinstance(other, CausalExpression):
            neg = {q: -w for q, w in other.terms.items()}
            return self + CausalExpression(neg)
        return NotImplemented

    def __mul__(self, scalar: float):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return CausalExpression({q: w * scalar for q, w in self.terms.items()})

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __neg__(self):
        return self * -1.0

    def __repr__(self):
        parts = []
        for q, w in self.terms.items():
            parts.append(f"{w:+g} * {q}")
        return " ".join(parts) or "0.0"

    def is_conditional(self) -> bool:
        "Returns True if this expression has any evidence variables (i.e. is a conditional query)."
        if not self.terms:
            return False
        # All terms must have the same evidence, so we can just check the first one.
        first_cq = next(iter(self.terms.keys()))
        return len(first_cq.evidence) > 0

    def induced_order(self) -> nx.DiGraph:
        """
        Returns the induced partial order over variables implied by the counterfactuals in all queries in this expression.
        For example, P(Y_{X=1}=1) + P(Z_{Y=0}=1) implies X < Y and Y < Z.

        Raises:
            ValueError: If the induced order contains a cycle, which would indicate an ill-formed expression.

        Returns:
            A directed acyclic graph (DAG) represented as a `networkx.DiGraph`, where an edge from A to B indicates that A is an intervention variable for B in at least
        """
        order = nx.DiGraph()
        for cq in self.terms.keys():
            order = nx.compose(order, cq.induced_order())

        try:
            # Check for cycles and raise an error if found, since that would indicate an ill-formed query.
            nx.find_cycle(order)
            raise ValueError("Induced order contains a cycle.")
        except nx.NetworkXNoCycle:
            return order
