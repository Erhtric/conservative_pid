"""
Defines the symbolic language for causal inference: Variables, Interventions,
Counterfactual Terms, Events, Queries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union


@dataclass(frozen=True)
class Variable:
    """
    Represents a random variable with a finite domain. In this simplified
    representation we allow a variable to have an identificative name and a
    domain of values.

    In this sense, it is an immutable object with an hashable name. This means that we
    allow two variables with different domains to have the same name.
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
    Represents a counterfactual term $Y_{\mathbf{X}=\mathbf{x}}$, or in short $Y_{\mathbf{x}}$.
    This is the equivalent of the Pearl's notation $Y$ in a world where ${do(X=x)}$.

    Usage:
    ```python
    Y @ {X: 1} # Y_{X=1}
    Y @ {X:1, W: W @ {X: 0}} # Y_{X=1, W_{X=0}}
    ```
    """

    variable: Variable
    intervention: Dict[Variable, Union[Any, CounterfactualTerm]] = field(
        default_factory=dict
    )

    def __hash__(self):
        # Convert mutable dictionary to hashable frozenset for hashing
        return hash((self.variable, frozenset(self.intervention.items())))

    def __repr__(self):
        # TODO: nested counterfactuals break this
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
    An atomic proposition in this context is a counterfactual term with a value.
    Stored as a dictionary mapping counterfactual terms to their values.
    """

    assignments: Dict[CounterfactualTerm, Any] = field(default_factory=dict)

    def expand(self) -> List[Event]:
        """
        Applies the Counterfactual Unnesting Theorem (CUT) once.

        TODO: Make this recursive. We can at the end call expand when extending the list of events.

        Example:
        Input: (Y_{X_z}=y)
        Output: [{Y_{X=0}=y & X_z=0}, {Y_{X=1}=y & X_z=1}, ...]

        Args:
            None

        Returns:
            A list of events

        Raises:
            ValueError: If there are contradictory assignments.
        """
        # Identify which terms are nested if there are any
        nested_term_loc = None

        for term in self.assignments.keys():
            for int_var, int_val in term.intervention.items():
                if isinstance(int_val, CounterfactualTerm):
                    # Nested: term has intervention {int_var: int_val}
                    nested_term_loc = (term, int_var, int_val)
                    break
                if nested_term_loc:
                    break

        # Base case: no nesting found, return self as a single item list
        if not nested_term_loc:
            return [self]

        # Recursive step: unnest the first nesting found
        outer_term: CounterfactualTerm = nested_term_loc[0]
        inner_var: Variable = nested_term_loc[1]
        inner_term: CounterfactualTerm = nested_term_loc[2]

        expanded_events = []
        # Iterate over the domain of the inner variable
        for val in inner_var.domain:
            # Create a new event with two atomic propositions
            # 1. The outer term with the intervention set to the current value
            # Note this modification of the subscript holds only for the inner_var variable, the others are left untouched
            new_outer_term = CounterfactualTerm(
                outer_term.variable,
                {**outer_term.intervention, inner_var: val},
            )

            # 2. The inner term with the intervention set to the current value while keeping the rest of the intervention
            new_inner_term = CounterfactualTerm(
                inner_term.variable,
                {**inner_term.intervention},
            )

            # Create a new event with the two atomic propositions
            conjunction = Event(
                {new_outer_term: self.assignments[outer_term], new_inner_term: val}
            )

            # Check for conflict, if the inner term is already assigned a different value, raise an error
            if inner_term in self.assignments and self.assignments[inner_term] != val:
                raise ValueError(
                    f"Contradictory assignments. {inner_term} cannot be both {self.assignments[inner_term]} and {val}"
                )

            # Add the new event to the list of expanded events
            expanded_events.extend(conjunction)

        return expanded_events

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

    def __hash__(self):
        return hash(frozenset(self.assignments.items()))


@dataclass(frozen=True)
class Query:
    """
    Represents a probability query P(target | evidence)

    Args:
        target: The target event
        evidence: The evidence event
    """

    target: Event
    evidence: Event = field(default_factory=Event)

    def __repr__(self):
        if self.evidence:
            return f"P({self.target} | {self.evidence})"
        else:
            return f"P({self.target})"

    def __add__(self, other: Union[Query, Expression]) -> Expression:
        return Expression({self: 1.0}) + other

    def __sub__(self, other: Union[Query, Expression]) -> Expression:
        return Expression({self: 1.0}) - other

    def __mul__(self, other: Union[float, int]) -> Expression:
        return Expression({self: 1.0}) * other

    def __rmul__(self, other: Union[float, int]) -> Expression:
        return Expression({self: 1.0}) * other

    def __neg__(self) -> Expression:
        return Expression({self: -1.0})


@dataclass(frozen=True)
class Expression:
    """
    Represents a linear combination of queries (an Effect).
    Form: sum(weight_i * query_i)
    """

    terms: Dict[Query, float] = field(default_factory=dict)

    def __add__(self, other: Union[Expression, Query]) -> Expression:
        new_terms = self.terms.copy()

        if isinstance(other, Query):
            # We assume equal contribution
            new_terms[other] = new_terms.get(other, 0.0) + 1.0
        elif isinstance(other, Expression):
            for q, w in other.terms.items():
                new_terms[q] = new_terms.get(q, 0.0) + w
        else:
            return NotImplemented

        # Remove terms with zero weight to keep it clean
        new_terms = {q: w for q, w in new_terms.items() if w != 0}
        return Expression(terms=new_terms)

    def __sub__(self, other: Union[Expression, Query]) -> Expression:
        return self + (-other)

    def __neg__(self) -> Expression:
        return self * -1.0

    def __mul__(self, other: Union[float, int]) -> Expression:
        if not isinstance(other, (float, int)):
            return NotImplemented
        new_terms = {q: w * other for q, w in self.terms.items()}
        return Expression(terms=new_terms)

    def __rmul__(self, other: Union[float, int]) -> Expression:
        return self.__mul__(other)

    def __repr__(self):
        parts = []
        for q, w in self.terms.items():
            sign = "+" if w >= 0 else "-"
            # scalar 1 is implicit in printing if we want cleaner output
            weight = f"{abs(w)} * " if abs(w) != 1.0 else ""
            parts.append(f"{sign} {weight}{q}")

        res = " ".join(parts).strip()
        if res.startswith("+ "):
            res = res[2:]
        return res or "0.0"


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
