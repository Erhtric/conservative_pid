import pytest

from symbolic import CounterfactualTerm, Event, P, Query, Variable


def test_variable_equality():
    X1 = Variable("X")
    X2 = Variable("X")
    Y = Variable("Y")

    assert X1 == X2
    assert X1 != Y

    d = {X1: 100}
    assert d[X2] == 100


def test_variable_domain():
    X = Variable("X", domain=(0, 1))
    assert X.domain == (0, 1)


def test_variable_dsl_event():
    X = Variable("X")

    # X == 0 should return an Event, not a boolean
    event = X == 0
    assert isinstance(event, Event)

    # Check internal structure
    # Key should be CounterfactualTerm(X, {})
    # The dictionary should have 1 item
    assert len(event.assignments) == 1
    for term, value in event.assignments.items():
        assert term.variable == X
        assert term.intervention == {}
        assert value == 0


def test_matmul_syntax():
    X = Variable("X")
    Y = Variable("Y")

    # Y @ {X: 1} -> Y_{X=1}
    cf = Y @ {X: 1}
    assert isinstance(cf, CounterfactualTerm)
    assert cf.variable == Y
    assert cf.intervention == {X: 1}


def test_counterfactual_repr():
    X = Variable("X")
    Y = Variable("Y")

    cf = Y @ {X: 1}
    assert str(cf) == "Y_{X=1}"


def test_event_operations():
    X = Variable("X")
    Y = Variable("Y")

    e1 = X == 1
    e2 = Y == 0

    # Conjunction
    e3 = e1 & e2
    assert isinstance(e3, Event)
    assert len(e3.assignments) == 2

    # Contradiction
    e4 = X == 2
    with pytest.raises(ValueError):
        _ = e1 & e4

    # Disjunction (merge)
    e5 = e1 | e2
    assert len(e5.assignments) == 2


def test_query():
    X = Variable("X")
    Y = Variable("Y")

    q = P(Y == 1, X == 0)
    assert isinstance(q, Query)
    assert str(q) == "P(Y = 1 | X = 0)"

    q2 = P(Y == 1)
    assert str(q2) == "P(Y = 1)"

    joint_event = (X == 1) & (Y == 0)
    assert isinstance(joint_event, Event)
    assert len(joint_event.assignments) == 2

    q3 = P(joint_event)
    assert str(q3) == "P(X = 1 & Y = 0)"
