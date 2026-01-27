import numpy as np
import pytest

from inference import ConservativePID
from symbolic import CounterfactualTerm, Event, P, Query, Variable


def test_inference_validation_variables():
    # TEST 1: Variable without domain
    X = Variable("X", domain=None)
    data = {}

    with pytest.raises(ValueError, match="must have a non-empty domain"):
        ConservativePID([X], data).infer(P(X == 0))

    # TEST 2: Empty domain
    Y = Variable("Y", domain=())
    with pytest.raises(ValueError, match="must have a non-empty domain"):
        ConservativePID([Y], data).infer(P(Y == 0))


def test_inference_validation_data():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    # TEST 3: Data row length mismatch
    data_bad_len = {(0,): 0.5}  # Missing Y
    with pytest.raises(ValueError, match="has length 1, expected 2"):
        ConservativePID([X, Y], data_bad_len).infer(P(X == 0))

    # TEST 4: Value not in domain
    data_bad_domain = {(0, 2): 0.5}  # Y=2 invalid
    with pytest.raises(ValueError, match="Value '2' in data row .* is not in domain"):
        ConservativePID([X, Y], data_bad_domain).infer(P(X == 0))

    # TEST 5: Invalid probability
    data_bad_prob = {(0, 0): 1.5}
    with pytest.raises(ValueError, match="Probability 1.5 in data is invalid"):
        ConservativePID([X, Y], data_bad_prob).infer(P(X == 0))

    # TEST 6: Sum distinct from 1
    data_bad_sum = {(0, 0): 0.5}  # Sums to 0.5
    with pytest.raises(ValueError, match="Observational probabilities sum to 0.5"):
        ConservativePID([X, Y], data_bad_sum).infer(P(X == 0))


def test_inference_validation_query():
    X = Variable("X", domain=(0, 1))
    data = {(0,): 1.0}

    pid = ConservativePID([X], data)

    # TEST 7: Unknown variable in target
    Z = Variable("Z", domain=(0, 1))
    with pytest.raises(ValueError, match="Unknown variable 'Z'"):
        pid.infer(P(Z == 0))

    # TEST 8: Unknown variable in evidence
    with pytest.raises(ValueError, match="Unknown variable 'Z'"):
        pid.infer(P(X == 0, Z == 0))

    # TEST 9: Unknown variable in intervention
    # X_{Z=0}
    term = X @ {Z: 0}
    e = term == 0
    # The Event itself doesn't check vars, but inference should.
    with pytest.raises(ValueError, match="Unknown intervention variable 'Z'"):
        pid.infer(P(e))

    # TEST 10: Invalid target type
    # We pass a real Query object so it passes the top-level type check,
    # but with an invalid target to fail the inner check.
    q_invalid = Query(target="NotAnEvent")  # type: ignore
    with pytest.raises(TypeError, match="Query target must be an Event object"):
        pid.infer(q_invalid)


def test_inference_validation_expression():
    X = Variable("X", domain=(0, 1))
    data = {(0,): 1.0}
    pid = ConservativePID([X], data)

    # Valid expression
    q1 = P(X == 0)
    q2 = P(X == 1)
    expr = q1 + q2
    # Should pass validation
    pid._validate_inputs(expr)

    # Expression with invalid variable
    Z = Variable("Z", domain=(0, 1))
    q_bad = P(Z == 0)
    expr_bad = q1 + q_bad
    with pytest.raises(ValueError, match="Unknown variable 'Z'"):
        pid._validate_inputs(expr_bad)

    # Input neither Query nor Expression
    with pytest.raises(
        TypeError, match="Input must be a Query or an Expression object"
    ):
        pid._validate_inputs("NotAQuery")  # type: ignore
