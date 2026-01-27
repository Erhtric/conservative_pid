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

    # # Disjunction (merge)
    # e5 = e1 | e2
    # assert len(e5.assignments) == 2


def test_query():
    X = Variable("X")
    Y = Variable("Y")

    q = P(Y == 1, X == 0)
    assert isinstance(q, Query)
    # assert str(q) == "P(Y = 1 | X = 0)"

    # q2 = P(Y == 1)
    # assert str(q2) == "P(Y = 1)"

    joint_event = (X == 1) & (Y == 0)
    assert isinstance(joint_event, Event)
    assert len(joint_event.assignments) == 2

    # q3 = P(joint_event)
    # assert str(q3) == "P(X = 1 & Y = 0)"


def test_expression_operations():
    X = Variable("X")
    Y = Variable("Y")

    q1 = P(X == 1)
    q2 = P(Y == 1)

    # 1. Query + Query -> Expression
    expr = q1 + q2
    assert isinstance(expr, Expression)
    assert len(expr.terms) == 2
    assert expr.terms[q1] == 1.0
    assert expr.terms[q2] == 1.0

    # 2. Query - Query
    expr_diff = q1 - q2
    assert expr_diff.terms[q1] == 1.0
    assert expr_diff.terms[q2] == -1.0

    # 3. Scalar Multiplication
    expr_mul = q1 * 2.0
    assert expr_mul.terms[q1] == 2.0

    expr_rmul = 3.0 * q2
    assert expr_rmul.terms[q2] == 3.0

    # 4. Negation
    expr_neg = -q1
    assert expr_neg.terms[q1] == -1.0

    # 5. Expression Arithmetic
    # (q1 + q2) + q1 -> 2*q1 + q2
    expr_comb = expr + q1
    assert expr_comb.terms[q1] == 2.0
    assert expr_comb.terms[q2] == 1.0

    # Expression - Query
    expr_sub = expr - q1
    assert q1 not in expr_sub.terms
    assert expr_sub.terms[q2] == 1.0


def test_nested_counterfactuals():
    X = Variable("X")
    Y = Variable("Y")
    Z = Variable("Z")

    # Y_{X_{Z}}
    # Inner: X_Z
    inner = X @ {Z: 1}
    # Outer: Y_{...}
    outer = Y @ {inner: 0}

    assert isinstance(outer, CounterfactualTerm)
    # The key in the dictionary is the inner term
    assert list(outer.intervention.keys())[0] == inner


def test_event_expand():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y")
    Z = Variable("Z", domain=(0, 1))

    # Event: Y_{X_Z} = y
    # We want to expand this based on Z.
    # The term is Y_{X_Z=0} (implicit value? No, X_Z is the variable in the intervention)
    # The structure is Y @ {X_Z: val} ? No.
    # The structure is Y @ {X: X @ {Z: ...}} ?
    # Let's look at how expand works.
    # It looks for nested counterfactuals in the intervention values.
    # query: P(Y_{X=Z} = y)
    # target event: {Y_{X=Z}: y}
    # term: Y_{X=Z} -> Y @ {X: Z} (where Z is a Variable, not a value)
    # Wait, Variable as value in intervention?
    # symbolic.py line 67: intervention: Dict[Variable, Union[Any, CounterfactualTerm]]
    # It seems it expects CounterfactualTerm as value for nesting.
    # So Y_{X_Z} corresponds to Y @ {X: X @ {Z: ...}} ?
    # Let's check the docstring example in symbolic.py:
    # Input: (Y_{X_z}=y)
    # Output: [{Y_{X=0}=y & X_z=0}, {Y_{X=1}=y & X_z=1}, ...]
    # This implies we are expanding the value of X.
    # The nested term is X_z.
    # So the intervention is {X: X_z}.

    # Construct X_z
    X_z = X @ {Z: 0}  # X_{Z=0}

    # Construct Y_{X=X_z}
    # The grammar allows intervention values to be CounterfactualTerms.
    # So we map X -> X_z
    Y_nested = Y @ {X: X_z}

    # Event: Y_{X=X_z} = 1
    e = Y_nested == 1

    # Expand
    # Expected:
    # For val in domain of X (0, 1):
    #  (Y_{X=val, ...} = 1) & (X_z = val)

    expanded = e.expand()
    assert len(expanded) == 2

    # Check first expansion (val=0)
    # e1: {Y_{X=0}: 1, X_{Z=0}: 0}
    # Note: expand() returns a list of Events. Each event is a conjunction.

    # We don't know the order, so we check existence

    # Case val=0
    term_Y_0 = Y @ {X: 0}
    term_X_0 = X_z  # X_{Z=0}
    e_0 = Event({term_Y_0: 1, term_X_0: 0})

    # Case val=1
    term_Y_1 = Y @ {X: 1}
    term_X_1 = X_z  # X_{Z=0}
    e_1 = Event({term_Y_1: 1, term_X_1: 1})

    # Verify both are present (using hash/eq)
    assert e_0 in expanded
    assert e_1 in expanded


def test_query_expand():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y")
    Z = Variable("Z", domain=(0, 1))

    # Query: P(Y_{X=X_z} = 1)
    X_z = X @ {Z: 0}
    Y_nested = Y @ {X: X_z}
    e = Y_nested == 1

    q = P(e)

    # Expand
    expr = q.expand()

    assert isinstance(expr, Expression)
    # P(Y_{X=X_z} = 1) = P(Y_{X=0}=1, X_z=0) + P(Y_{X=1}=1, X_z=1)

    term_Y_0 = Y @ {X: 0}
    term_X_0 = X_z  # X_{Z=0}
    e_0 = Event({term_Y_0: 1, term_X_0: 0})
    q_0 = P(e_0)

    term_Y_1 = Y @ {X: 1}
    term_X_1 = X_z  # X_{Z=0}
    e_1 = Event({term_Y_1: 1, term_X_1: 1})
    q_1 = P(e_1)

    assert len(expr.terms) == 2
    assert expr.terms[q_0] == 1.0
    assert expr.terms[q_1] == 1.0
