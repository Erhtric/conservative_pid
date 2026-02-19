import numpy as np
import pytest

from inference import ConservativePID
from symbolic import CounterfactualTerm, P, Variable


def test_infer_monotonicity_binary():
    """
    Test end-to-end monotonicity inference with ConservativePID.
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

    cp = ConservativePID([X, Y], data)

    # Defier Query: P((Y_0=1) & (Y_1=0))
    y_x0 = Y @ {X: 0}
    y_x1 = Y @ {X: 1}
    defier_query = P((y_x0 == 1) & (y_x1 == 0))

    # 1. Without Monotonicity
    lb_std, ub_std = cp.infer(defier_query, monotonic=False)

    # 2. With Monotonicity
    lb_mono, ub_mono = cp.infer(defier_query, monotonic=True)

    print(f"Standard: [{lb_std}, {ub_std}]")
    print(f"Monotonic: [{lb_mono}, {ub_mono}]")

    assert ub_std > 0, "Defiers should be possible without monotonicity"
    assert np.isclose(ub_mono, 0.0), "Defiers should be 0 with monotonicity"


def test_infer_monotonicity_subset():
    """
    Test passing specific variables for monotonicity to infer().
    Using X -> M -> Y chain example.
    """
    X = Variable("X", domain=(0, 1))
    M = Variable("M", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    # Uniform data
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

    cp = ConservativePID([X, M, Y], data)

    m_x0 = M @ {X: 0}
    m_x1 = M @ {X: 1}
    # M defier
    m_defier = P((m_x0 == 1) & (m_x1 == 0))

    # Check bounds when enforcing M monotonicity
    # We use fixed_order=['X', 'M', 'Y'] to ensure M only depends on X.
    # In this order, M is monotonic w.r.t X implies M_0 <= M_1.

    lb, ub = cp.infer(m_defier, fixed_order=["X", "M", "Y"], monotonic=[M])

    # If M is monotonic in all valid causal models, then bounds should be 0?
    assert np.isclose(ub, 0.0)

    # If we don't enforce M
    lb_std, ub_std = cp.infer(m_defier, fixed_order=["X", "M", "Y"], monotonic=[])
    assert ub_std > 0
