import numpy as np
import pytest

from src.canonical import VectorizedCanonicalBasis
from src.solver import LPSolver
from src.symbolic import Event, P, Variable


def test_solver_expression_basic():
    # Variables X -> Y
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    # Let's say P(X=0, Y=0)=0.5, P(X=1, Y=1)=0.5
    data = {(0, 0): 0.5, (1, 1): 0.5}

    # Setup Basis and Solver
    # Order X, Y
    basis = VectorizedCanonicalBasis([X, Y])
    solver = LPSolver(basis, data, [X, Y])

    # Expression: P(Y=0) + P(Y=1). Should be 1.0
    ev1 = Event({Y: 0})
    ev2 = Event({Y: 1})
    q1 = P(ev1)
    q2 = P(ev2)
    expr = q1 + q2

    lb, ub = solver.solve(expr)

    assert np.isclose(lb, 1.0)
    assert np.isclose(ub, 1.0)


def test_solver_expression_ate_bounds():
    # ATE: P(Y_{X=1} = 1) - P(Y_{X=0} = 1)
    # Bounds should be within [-1, 1]
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    # Data: perfect correlation
    data = {(0, 0): 0.5, (1, 1): 0.5}

    basis = VectorizedCanonicalBasis([X, Y])
    solver = LPSolver(basis, data, [X, Y])

    t1 = Y @ {X: 1}
    t0 = Y @ {X: 0}

    q1 = P(t1 == 1)
    q0 = P(t0 == 1)

    # Expression
    ate = q1 - q0

    lb, ub = solver.solve(ate)

    assert lb >= -1.0
    assert ub <= 1.0
    # In this simple case, P(Y_{X=1}=1) can be [0.5, 1] and P(Y_{X=0}=1) can be [0, 0.5] ?
    # Actually with perfect correlation X=Y:
    # Y_{x=1}=? . Observed Y=1 when X=1. So Y_{x=1} is likely 1.
    # We should get something reasonable.
    # Main check is it runs and gives valid prob range.


def test_solver_expression_conditional():
    # P(Y=1 | X=0) + P(Y=0 | X=0) should be 1.0
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    data = {(0, 0): 0.5, (1, 1): 0.5}
    basis = VectorizedCanonicalBasis([X, Y])
    solver = LPSolver(basis, data, [X, Y])

    q1 = P(Y == 1, evidence=(X == 0))
    q2 = P(Y == 0, evidence=(X == 0))

    expr = q1 + q2

    lb, ub = solver.solve(expr)
    assert np.isclose(lb, 1.0)
    assert np.isclose(ub, 1.0)


def test_solver_expression_error_mixed_evidence():
    X = Variable("X", domain=(0, 1))

    q1 = P(X == 0)
    q2 = P(X == 0, evidence=(X == 1))  # Impossible evidence but valid query structure

    expr = q1 + q2

    # Mock everything else slightly as we just want to test validation overlap in solve()
    # But solve() instantiates actual solving process, so we need valid init.
    data = {(0,): 0.5, (1,): 0.5}
    basis = VectorizedCanonicalBasis([X])
    solver = LPSolver(basis, data, [X])

    with pytest.raises(ValueError, match="must share the same evidence"):
        solver.solve(expr)

    # Test with two different evidences
    q3 = P(X == 0, evidence=(X == 0))
    expr2 = q2 + q3
    with pytest.raises(ValueError, match="must share the same evidence"):
        solver.solve(expr2)
