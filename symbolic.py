"""
Defines the symbolic language for causal inference: Variables, Interventions,
Counterfactual Terms, and Events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union


@dataclass(frozen=True)
class Variable:
    """
    Represents a random variable with a finite domain
    """

    name: str
    domain: Tuple[Any, ...] = field(default_factory=tuple)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"{self.name}"

    def __matmul__(self, intervention: Dict[Variable, Any]) -> CounterfactualTerm:
        """
        Syntax sugar for creating a counterfactual term.
        Usage: Y @ {X: 1}  ->  Y_{X=1}
        """
        return CounterfactualTerm(self, intervention)

    def __eq__(self, other: Any) -> Union[bool, Event]:
        """
        If other is a Variable, compare names for equality.
        Otherwise, return an atomic event (Syntax sugar: Y == 0).
        """
        if isinstance(other, Variable):
            return self.name == other.name

        # Note: We return an Event, not a boolean. This overrides standard equality for values.
        return Event({CounterfactualTerm(self, {}): other})


@dataclass(frozen=True)
class CounterfactualTerm:
    """
    Represents a counterfactual term Y_{X=x}
    """

    variable: Variable
    intervention: Dict[Variable, Union[Any, CounterfactualTerm]] = field(
        default_factory=dict
    )

    def __hash__(self):
        # Convert mutable dictionary to hashable frozenset for hashing
        return hash((self.variable, frozenset(self.intervention.items())))

    def __repr__(self):
        if not self.intervention:
            return self.variable.name

        intervention_str = []
        for var, val in self.intervention.items():
            val_str = f"{val}" if isinstance(val, CounterfactualTerm) else f"{val}"
            intervention_str.append(f"{var.name}={val_str}")

        return f"{self.variable.name}_{{{', '.join(intervention_str)}}}"

    def __eq__(self, other: Any) -> Union[bool, Event]:
        """
        If other is a CounterfactualTerm, compare for structural equality.
        Otherwise, return an atomic event (Syntax sugar: Y == 0).
        """
        if isinstance(other, CounterfactualTerm):
            return (
                self.variable.name == other.variable.name
                and self.intervention == other.intervention
            )

        # Note: We return an Event, not a boolean. This overrides standard equality.
        return Event({self: other})


@dataclass(frozen=True)
class Event:
    """
    Represents an event, i.e. a conjunction of atomic propositions:
    e.g., {Y_{X=1}: 0, Z_{X=1, Y=0}: 1}
    Stored as a dictionary mapping counterfactual terms to their values.
    """

    assignments: Dict[CounterfactualTerm, Any] = field(default_factory=dict)

    def __and__(self, other: Event) -> Event:
        """
        Syntax sugar for creating a conjunction of events.
        Usage: event1 & event2
        """
        new_assigments = self.assignments.copy()

        # Check for contradictions
        for term, value in other.assignments.items():
            if term in new_assigments and new_assigments[term] != value:
                raise ValueError(
                    f"Contradictory assignments. {term} cannot be both {new_assigments[term]} and {value}"
                )
            new_assigments[term] = value

        return Event(new_assigments)

    def __or__(self, other: Event) -> Event:
        """
        Syntax sugar for creating a disjunction of events.
        Usage: event1 | event2
        """
        return Event({**self.assignments, **other.assignments})

    def __repr__(self):
        return " & ".join(
            [f"{term} = {value}" for term, value in self.assignments.items()]
        )

    def __bool__(self):
        return bool(self.assignments)


@dataclass
class Query:
    """
    Represents a probability query P(target | evidence)
    """

    target: Event
    evidence: Event

    def __repr__(self):
        if self.evidence:
            return f"P({self.target} | {self.evidence})"
        else:
            return f"P({self.target})"


# Helper functions for creating queries
def P(target: Event, evidence: Event = None) -> Query:
    """
    Syntax sugar for creating a query.
    Usage: P(Y == 1, X == 0)

    Args:
        target: The target event
        evidence: The evidence event

    Returns:
        A query
    """
    if evidence is None:
        evidence = Event()
    return Query(target, evidence)
