import numpy as np
import pytest

from canonical import BasisGenerator, VectorizedCanonicalBasis
from symbolic import CounterfactualTerm, Variable


def test_vectorized_basis_structure():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    vec_basis = VectorizedCanonicalBasis([X, Y])

    assert vec_basis.n_worlds == 8
    assert vec_basis.func_tables[0].shape == (8, 1)
    assert vec_basis.func_tables[1].shape == (8, 2)


def test_vectorized_evaluation_consistency():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    gen = BasisGenerator([X, Y])
    std_basis = gen.generate_basis()

    vec_basis = VectorizedCanonicalBasis([X, Y])
    term_X = CounterfactualTerm(X, {})
    vals_X = vec_basis.evaluate(term_X)
    std_vals_X = [w.evaluate(term_X) for w in std_basis]
    assert np.array_equal(vals_X, std_vals_X)
    term_Y_do_X1 = Y @ {X: 1}
    vals_Y_do_X1 = vec_basis.evaluate(term_Y_do_X1)
    std_vals_Y_do_X1 = [w.evaluate(term_Y_do_X1) for w in std_basis]

    assert np.array_equal(vals_Y_do_X1, std_vals_Y_do_X1)


def test_vectorized_mask():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    vec_basis = VectorizedCanonicalBasis([X, Y])

    event = (X == 1) & (Y == 0)
    mask = vec_basis.get_event_mask(event)
    term_X = CounterfactualTerm(X, {})
    term_Y = CounterfactualTerm(Y, {})
    vals_X = vec_basis.evaluate(term_X)
    vals_Y = vec_basis.evaluate(term_Y)

    expected_mask = (vals_X == 1) & (vals_Y == 0)
    assert np.array_equal(mask, expected_mask)


def test_nested_intervention_vectorized():
    Z = Variable("Z", domain=(0, 1))
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    vec_basis = VectorizedCanonicalBasis([Z, X, Y])

    cf_Z_is_0 = Z @ {Z: 0}
    term_complex = Y @ {X: cf_Z_is_0}

    vals = vec_basis.evaluate(term_complex)

    term_simple = Y @ {X: 0}
    vals_simple = vec_basis.evaluate(term_simple)

    assert np.array_equal(vals, vals_simple)
