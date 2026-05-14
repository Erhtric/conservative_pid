from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Union


@dataclass(frozen=True)
class AtomicCounterfactual:
    """
    Represents a single atomic proposition: e.g., Y_{x=1} = 1
    """

    target_var: str
    target_val: int
    interventions: Dict[str, int] = field(default_factory=dict)  # e.g., {'X': 1}

    def __str__(self):
        int_str = ", ".join(f"{k}={v}" for k, v in self.interventions.items())
        if int_str:
            return f"{self.target_var}_{{{int_str}}}={self.target_val}"
        return f"{self.target_var}={self.target_val}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        "If two CTF are identical in target, value, and interventions (regardless of order), they should hash the same."
        items = tuple(sorted(self.interventions.items()))
        return hash((self.target_var, self.target_val, items))


@dataclass(frozen=True, repr=False)
class CausalQuery:
    """
    Represents the full conjunction: P(gamma | delta)
    - counterfactuals: list of AtomicCounterfactual objects
    - evidence: dict of observational evidence (natural state)
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


@dataclass(frozen=True)
class CausalExpression:
    """
    Represents a linear combination of `CausalQuery` objects.
    """

    terms: Dict[CausalQuery, float] = field(default_factory=dict)

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
