import pytest

from inference import ConservativePID
from symbolic import Variable


def test_marginalize_data():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))
    Z = Variable("Z", domain=(0, 1))

    # Simple distribution
    data = {
        (0, 0, 0): 0.1,
        (0, 0, 1): 0.2,
        (1, 1, 0): 0.3,
        (1, 1, 1): 0.4,
    }
    # Total sum 1.0

    all_vars = [X, Y, Z]

    # Marginalize out Y (keep X, Z)
    target_vars = [X, Z]
    marg_data = ConservativePID.marginalize_data(data, all_vars, target_vars)

    # Expected:
    # (0, 0) comes from (0,0,0) -> 0.1
    # (0, 1) comes from (0,0,1) -> 0.2
    # (1, 0) comes from (1,1,0) -> 0.3
    # (1, 1) comes from (1,1,1) -> 0.4

    assert marg_data[(0, 0)] == 0.1
    assert marg_data[(0, 1)] == 0.2
    assert marg_data[(1, 0)] == 0.3
    assert marg_data[(1, 1)] == 0.4

    # Marginalize out Z (keep X, Y)
    target_vars_xy = [X, Y]
    marg_data_xy = ConservativePID.marginalize_data(data, all_vars, target_vars_xy)

    # (0, 0) from (0,0,0) + (0,0,1) = 0.3
    # (1, 1) from (1,1,0) + (1,1,1) = 0.7

    assert pytest.approx(marg_data_xy[(0, 0)]) == 0.3
    assert pytest.approx(marg_data_xy[(1, 1)]) == 0.7
    assert (0, 1) not in marg_data_xy
    assert (1, 0) not in marg_data_xy


def test_marginalize_data_reorder():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    data = {(0, 1): 0.4, (1, 0): 0.6}
    all_vars = [X, Y]

    # Request [Y, X]
    target_vars = [Y, X]
    marg_data = ConservativePID.marginalize_data(data, all_vars, target_vars)

    # Should flip tuples
    # (1, 0) -> 0.4
    # (0, 1) -> 0.6

    assert marg_data[(1, 0)] == 0.4
    assert marg_data[(0, 1)] == 0.6
