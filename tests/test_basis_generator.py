import pytest

from canonical import BasisGenerator, CanonicalConfiguration
from symbolic import Variable


def test_basis_generator_single_variable():
    X = Variable("X", domain=(0, 1))

    gen = BasisGenerator([X])
    basis = gen.generate_basis()

    # For a single variable X with domain {0, 1}:
    # Parents = empty.
    # Parent configs = [()].
    # Functions X -> {0, 1}.
    # Two possible functions: ()->0, ()->1.
    # So basis size should be 2.

    assert len(basis) == 2

    # Verify contents
    vals = set()
    for config in basis:
        assert isinstance(config, CanonicalConfiguration)
        # Check X evaluation (no intervention)
        # config.functions[X][()] should be 0 or 1
        vals.add(config.functions[X][()])

    assert vals == {0, 1}


def test_basis_generator_two_variables():
    # X -> Y
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    gen = BasisGenerator([X, Y])
    basis = gen.generate_basis()

    # Count:
    # X (root): 2 functions (as above).
    # Y (parent X):
    #   Domain X size 2.
    #   Domain Y size 2.
    #   Number of functions from X to Y = |Dom(Y)| ^ |Dom(X)| = 2^2 = 4.
    # Total basis size = 2 * 4 = 8.

    assert len(basis) == 8

    # Verify uniqueness (optional, but good)
    # Since basis is list of CanonicalConfiguration objects, and they might not implement equality based on content efficiently,
    # we can check distinctness of function maps.

    # ...


def test_basis_generator_three_variables():
    # X -> Y -> Z
    # Order matters.
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))
    Z = Variable("Z", domain=(0, 1))

    # Note: BasisGenerator uses the list order as topological order.
    # Parents of Z are X, Y (fully connected assumption in _build_parents_map logic of CanonicalConfiguration?)
    # Wait, BasisGenerator implementation:
    # parents = self.variables[:i]
    # So for Z (index 2), parents are X (0) and Y (1).

    gen = BasisGenerator([X, Y, Z])
    basis = gen.generate_basis()

    # Count:
    # X: 2 functions.
    # Y: 2^2 = 4 functions.
    # Z: Parents X, Y. |Dom(X)|=2, |Dom(Y)|=2. Parent configs = 4.
    #    Functions Z: |Dom(Z)| ^ 4 = 2^4 = 16.
    # Total: 2 * 4 * 16 = 128.

    assert len(basis) == 128
