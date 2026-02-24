from canonical import CanonicalConfiguration
from symbolic import CounterfactualTerm, Variable


def test_canonical_configuration_simple():
    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    order = [X, Y]
    functions = {
        X: {(): 0},
        Y: {(0,): 1, (1,): 0},
    }

    config = CanonicalConfiguration(functions, order)

    term_X = CounterfactualTerm(X, {})
    term_Y = CounterfactualTerm(Y, {})

    assert config.evaluate(term_X) == 0
    assert config.evaluate(term_Y) == 1

    term_Y_do_X1 = Y @ {X: 1}
    assert config.evaluate(term_Y_do_X1) == 0
    term_Y_do_X0 = Y @ {X: 0}
    assert config.evaluate(term_Y_do_X0) == 1
    term_X_do_Y1 = X @ {Y: 1}
    assert config.evaluate(term_X_do_Y1) == 0
    term_Y_do_Y0 = Y @ {Y: 0}
    assert config.evaluate(term_Y_do_Y0) == 0


def test_satisfies_event():
    X = Variable("X")
    Y = Variable("Y")
    order = [X, Y]
    functions = {
        X: {(): 1},
        Y: {(0,): 0, (1,): 1},
    }
    config = CanonicalConfiguration(functions, order)
    e1 = (X == 1) & (Y == 1)
    assert config.satisfies(e1)
    e2 = Y @ {X: 0} == 0
    assert config.satisfies(e2)
    e3 = (X == 1) & (Y @ {X: 0} == 0)
    assert config.satisfies(e3)
    e4 = X == 0
    assert not config.satisfies(e4)


def test_nested_intervention_logic():
    Z = Variable("Z", (0, 1))
    X = Variable("X", (0, 1))
    Y = Variable("Y", (0, 1))

    order = [Z, X, Y]
    functions = {
        Z: {(): 1},
        X: {(0,): 0, (1,): 1},
        Y: {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1},
    }

    config = CanonicalConfiguration(functions, order)
    cf_Z_is_0 = Z @ {Z: 0}

    term_complex = Y @ {X: cf_Z_is_0}

    assert config.evaluate(term_complex) == 0


def test_is_compatible():
    X = Variable("X")
    functions = {X: {(): 1}}
    config = CanonicalConfiguration(functions, [X])
    obs1 = X == 1
    assert config.is_compatible(obs1)
    obs2 = X == 0
    assert not config.is_compatible(obs2)
