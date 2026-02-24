import pytest

from canonical import BasisGenerator, CanonicalConfiguration
from symbolic import Variable


def test_basis_generator_single_variable():
    X = Variable("X", domain=(0, 1))

    gen = BasisGenerator([X])
    basis = gen.generate_basis()

    assert len(basis) == 2
    vals = set()
    for config in basis:
        assert isinstance(config, CanonicalConfiguration)
        vals.add(config.functions[X][()])

    assert vals == {0, 1}


def test_basis_generator_two_variables():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    gen = BasisGenerator([X, Y])
    basis = gen.generate_basis()

    assert len(basis) == 8


def test_basis_generator_three_variables():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))
    Z = Variable("Z", domain=(0, 1))

    gen = BasisGenerator([X, Y, Z])
    basis = gen.generate_basis()

    assert len(basis) == 128
