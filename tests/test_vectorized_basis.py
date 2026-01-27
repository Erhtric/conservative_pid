import numpy as np
import pytest

from canonical import BasisGenerator, VectorizedCanonicalBasis
from symbolic import CounterfactualTerm, Variable


def test_vectorized_basis_structure():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    vec_basis = VectorizedCanonicalBasis([X, Y])

    # Basis size should match standard generator
    # X: 2 funcs. Y: 4 funcs. Total 8.
    assert vec_basis.n_worlds == 8

    # Check func_tables shape
    # X has 0 parents -> 1 parent config. Table shape (8, 1)
    assert vec_basis.func_tables[0].shape == (8, 1)

    # Y has 1 parent (X) -> 2 parent configs. Table shape (8, 2)
    assert vec_basis.func_tables[1].shape == (8, 2)


def test_vectorized_evaluation_consistency():
    # Compare with standard basis
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    gen = BasisGenerator([X, Y])
    std_basis = gen.generate_basis()

    vec_basis = VectorizedCanonicalBasis([X, Y])

    # Check simple evaluation: X
    # vec_basis.evaluate(term) returns array of size 8.
    term_X = CounterfactualTerm(X, {})
    vals_X = vec_basis.evaluate(term_X)

    # In standard basis, we iterate
    std_vals_X = [w.evaluate(term_X) for w in std_basis]

    # Compare. Order implies we need to match rows.
    # Since both iterate itertools.product/cartesian logic in same order, they should match index-wise.
    assert np.array_equal(vals_X, std_vals_X)

    # Check counterfactual: Y_{X=1}
    term_Y_do_X1 = Y @ {X: 1}
    vals_Y_do_X1 = vec_basis.evaluate(term_Y_do_X1)
    std_vals_Y_do_X1 = [w.evaluate(term_Y_do_X1) for w in std_basis]

    assert np.array_equal(vals_Y_do_X1, std_vals_Y_do_X1)


def test_vectorized_mask():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    vec_basis = VectorizedCanonicalBasis([X, Y])

    # Event: X=1 & Y=0
    event = (X == 1) & (Y == 0)
    mask = vec_basis.get_event_mask(event)

    # Manual check
    term_X = CounterfactualTerm(X, {})
    term_Y = CounterfactualTerm(Y, {})
    vals_X = vec_basis.evaluate(term_X)
    vals_Y = vec_basis.evaluate(term_Y)

    expected_mask = (vals_X == 1) & (vals_Y == 0)
    assert np.array_equal(mask, expected_mask)


def test_nested_intervention_vectorized():
    # Z -> X -> Y
    Z = Variable("Z", domain=(0, 1))
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    vec_basis = VectorizedCanonicalBasis([Z, X, Y])

    # Term: Y_{X = Z_{Z=0}}
    # Evaluate Z_{Z=0} -> 0.
    # So Y_{X=0}

    cf_Z_is_0 = Z @ {Z: 0}
    term_complex = Y @ {X: cf_Z_is_0}

    vals = vec_basis.evaluate(term_complex)

    # Compare with Y_{X=0}
    term_simple = Y @ {X: 0}
    vals_simple = vec_basis.evaluate(term_simple)

    assert np.array_equal(vals, vals_simple)
