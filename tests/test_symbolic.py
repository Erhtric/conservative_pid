import pytest


from symbolic import CounterfactualTerm, Event, Expression, P, Query, Variable


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
    event = X == 0
    assert isinstance(event, Event)
    assert len(event.assignments) == 1
    for term, value in event.assignments.items():
        assert term.variable == X
        assert term.intervention == {}
        assert value == 0


def test_matmul_syntax():
    "Subscript functional specifications as syntax sugar"
    X = Variable("X")
    Y = Variable("Y")
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

    e3 = e1 & e2
    assert isinstance(e3, Event)
    assert len(e3.assignments) == 2
    e4 = X == 2
    with pytest.raises(ValueError):
        _ = e1 & e4


def test_query():
    X = Variable("X")
    Y = Variable("Y")

    q = P(Y == 1, X == 0)
    assert isinstance(q, Query)

    joint_event = (X == 1) & (Y == 0)
    assert isinstance(joint_event, Event)
    assert len(joint_event.assignments) == 2


def test_expression_operations():
    X = Variable("X")
    Y = Variable("Y")

    q1 = P(X == 1)
    q2 = P(Y == 1)

    expr = q1 + q2
    assert isinstance(expr, Expression)
    assert len(expr.terms) == 2
    assert expr.terms[q1] == 1.0
    assert expr.terms[q2] == 1.0
    expr_diff = q1 - q2
    assert expr_diff.terms[q1] == 1.0
    assert expr_diff.terms[q2] == -1.0
    expr_mul = q1 * 2.0
    assert expr_mul.terms[q1] == 2.0

    expr_rmul = 3.0 * q2
    assert expr_rmul.terms[q2] == 3.0
    expr_neg = -q1
    assert expr_neg.terms[q1] == -1.0
    expr_comb = expr + q1
    assert expr_comb.terms[q1] == 2.0
    assert expr_comb.terms[q2] == 1.0
    expr_sub = expr - q1
    assert q1 not in expr_sub.terms
    assert expr_sub.terms[q2] == 1.0


def test_nested_counterfactuals():
    X = Variable("X")
    Y = Variable("Y")
    Z = Variable("Z")
    inner = X @ {Z: 1}
    outer = Y @ {inner: 0}

    assert isinstance(outer, CounterfactualTerm)
    assert list(outer.intervention.keys())[0] == inner


def test_event_expand():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y")
    Z = Variable("Z", domain=(0, 1))
    X_z = X @ {Z: 0}
    Y_nested = Y @ {X: X_z}
    e = Y_nested == 1
    expanded = e.expand()
    assert len(expanded) == 2

    term_Y_0 = Y @ {X: 0}
    term_X_0 = X_z
    e_0 = Event({term_Y_0: 1, term_X_0: 0})
    term_Y_1 = Y @ {X: 1}
    term_X_1 = X_z
    e_1 = Event({term_Y_1: 1, term_X_1: 1})
    assert e_0 in expanded
    assert e_1 in expanded


def test_query_expand():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y")
    Z = Variable("Z", domain=(0, 1))
    X_z = X @ {Z: 0}
    Y_nested = Y @ {X: X_z}
    e = Y_nested == 1

    q = P(e)

    expr = q.expand()

    assert isinstance(expr, Expression)
    term_Y_0 = Y @ {X: 0}
    term_X_0 = X_z
    e_0 = Event({term_Y_0: 1, term_X_0: 0})
    q_0 = P(e_0)

    term_Y_1 = Y @ {X: 1}
    term_X_1 = X_z
    e_1 = Event({term_Y_1: 1, term_X_1: 1})
    q_1 = P(e_1)

    assert len(expr.terms) == 2
    assert expr.terms[q_0] == 1.0
    assert expr.terms[q_1] == 1.0
