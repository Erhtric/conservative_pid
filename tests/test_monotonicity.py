import numpy as np
import pytest

from canonical import VectorizedCanonicalBasis
from solver import LPSolver
from symbolic import CounterfactualTerm, P, Variable


def test_monotonicity_binary_defiers():
    """
    Test that monotonicity constraints force P(Defier) = 0 in a binary setting with uniform data.
    Defier: Y_{x=0}=1 AND Y_{x=1}=0.
    Monotonicity requires Y_{x=0} <= Y_{x=1}.
    """
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    # Uniform Data
    data = {
        (0, 0): 0.25,
        (0, 1): 0.25,
        (1, 0): 0.25,
        (1, 1): 0.25,
    }

    basis = VectorizedCanonicalBasis([X, Y])
    solver = LPSolver(basis, data, [X, Y])

    # Define Defier Event: Y(x=0) = 1 AND Y(x=1) = 0
    y_x0 = Y @ {X: 0}
    y_x1 = Y @ {X: 1}
    # Use & for joint event
    defier_query = P((y_x0 == 1) & (y_x1 == 0))

    # 1. Solve WITHOUT Monotonicity
    lb_std, ub_std = solver.solve(defier_query, monotonic=False)
    print(f"\nStandard Bounds for Defier: [{lb_std}, {ub_std}]")

    # Without constraints, defiers should be possible.
    assert ub_std > 0, "Defiers should be possible without monotonicity"

    # 2. Solve WITH Monotonicity (Global)
    lb_mono, ub_mono = solver.solve(defier_query, monotonic=True)
    print(f"Monotonic Bounds for Defier: [{lb_mono}, {ub_mono}]")

    # With monotonicity, defiers are impossible.
    assert np.isclose(ub_mono, 0.0), "Defiers should be probability 0 with monotonicity"
    assert np.isclose(lb_mono, 0.0)


def test_monotonicity_ternary():
    """
    Test monotonicity with a ternary variable.
    X (0,1) -> Y (0,1,2).
    Monotonicity: x <= x' => Y_x <= Y_x'.
    """
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1, 2))

    # Construct data compatible with monotonicity
    # e.g. Perfect correlation X=0->Y=0, X=1->Y=2
    data = {
        (0, 0): 0.5,
        (1, 2): 0.5,
        # Zero prob for others
        (0, 1): 0.0,
        (0, 2): 0.0,
        (1, 0): 0.0,
        (1, 1): 0.0,
    }

    basis = VectorizedCanonicalBasis([X, Y])
    solver = LPSolver(basis, data, [X, Y])

    # Query: P(Y_{x=0} > Y_{x=1}) -> Should be 0 under monotonicity
    y_x0 = Y @ {X: 0}
    y_x1 = Y @ {X: 1}

    q1 = P((y_x0 == 1) & (y_x1 == 0))
    q2 = P((y_x0 == 2) & (y_x1 == 0))
    q3 = P((y_x0 == 2) & (y_x1 == 1))

    violation_expr = q1 + q2 + q3

    lb_mono, ub_mono = solver.solve(violation_expr, monotonic=True)

    assert np.isclose(ub_mono, 0.0), "Monotonicity violation probability should be 0"


def test_monotonicity_subset():
    """
    Test enforcing monotonicity on a subset of variables.
    Chain X -> M -> Y.
    """
    X = Variable("X", domain=(0, 1))
    M = Variable("M", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    # Uniform data
    # (x, m, y)
    data = {
        (0, 0, 0): 0.125,
        (0, 0, 1): 0.125,
        (0, 1, 0): 0.125,
        (0, 1, 1): 0.125,
        (1, 0, 0): 0.125,
        (1, 0, 1): 0.125,
        (1, 1, 0): 0.125,
        (1, 1, 1): 0.125,
    }

    basis = VectorizedCanonicalBasis([X, M, Y])
    solver = LPSolver(basis, data, [X, M, Y])

    # Query: violations of M monotonicity + Y monotonicity
    # M defier: M_0=1, M_1=0
    m_x0 = M @ {X: 0}
    m_x1 = M @ {X: 1}
    m_defier = P((m_x0 == 1) & (m_x1 == 0))

    # Y defier (wrt M): Y_m0=1, Y_m1=0
    # Note: Y depends on M. Check Y_{M=0}=1, Y_{M=1}=0
    y_m0 = Y @ {M: 0}
    y_m1 = Y @ {M: 1}
    y_defier = P((y_m0 == 1) & (y_m1 == 0))

    # 1. Monotonic only on M
    # Should enforce M defier = 0, but allow Y defier > 0
    lb_M, ub_M = solver.solve(m_defier, monotonic=[M])
    lb_Y, ub_Y = solver.solve(y_defier, monotonic=[M])

    assert np.isclose(ub_M, 0.0), "M should be monotonic"
    assert ub_Y > 0, "Y should NOT be monotonic yet"

    # 2. Monotonic only on Y
    # Should enforce Y defier = 0, but allow M defier > 0
    lb_M2, ub_M2 = solver.solve(m_defier, monotonic=[Y])
    lb_Y2, ub_Y2 = solver.solve(y_defier, monotonic=[Y])

    assert ub_M2 > 0, "M should NOT be monotonic here"
    # Note: ub_Y2 might be > 0 if there are confounding paths,
    # but here Y only parent is M (in chain), so Y_{m=0} <= Y_{m=1} should hold if Y is monotonic w.r.t M.
    # Actually Y has parents X, M in the basis?
    # Wait, in the basis `VectorizedCanonicalBasis([X, M, Y])`,
    # M has parent X. Y has parents X and M.
    # Our data is uniform full joint.
    # Y's monotonicity is checked against ALL parents (X and M).
    # If we only intervene on M, X is random/observed.
    # Monotonicity definition: f(x,m) <= f(x',m') if (x,m) <= (x',m').
    # Here checking partial monotonicity w.r.t M means comparing (x,0) vs (x,1).
    # Since we enforce monotonicity on Y, it must hold that Y(x,0) <= Y(x,1).
    # Thus Y_{M=0} <= Y_{M=1} must hold for any specific x.
    # The atomic counterfactual Y_{m=0} integrates over X.
    # If for every u, Y_{m=0}(u) <= Y_{m=1}(u), then Y_{m=0} <= Y_{m=1} is almost certain?
    # Yes, Y_{m=0, u} <= Y_{m=1, u} implies the event (Y_{m=0}=1 & Y_{m=1}=0) is impossible.

    assert np.isclose(ub_Y2, 0.0), "Y should be monotonic"
