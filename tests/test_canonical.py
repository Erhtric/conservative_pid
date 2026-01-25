from canonical import CanonicalConfiguration
from symbolic import CounterfactualTerm, Variable


def test_canonical_configuration_simple():
    # Setup: X -> Y
    # X is root.
    # Y = 1 - X (NOT gate)

    X = Variable("X", domain=(0, 1))
    Y = Variable("Y", domain=(0, 1))

    order = [X, Y]

    # Functions
    # X takes no parents, so key is empty tuple
    # Y takes X as parent, so key is tuple (x_val,)
    functions = {
        X: {(): 0},  # X := 0
        Y: {(0,): 1, (1,): 0},  # Y := 1 - X
    }

    config = CanonicalConfiguration(functions, order)

    # Test 1: Evaluate atomic variables (observational level)
    # X should be 0
    # Y should be 1-0 = 1
    term_X = CounterfactualTerm(X, {})
    term_Y = CounterfactualTerm(Y, {})

    assert config.evaluate(term_X) == 0
    assert config.evaluate(term_Y) == 1

    # Test 2: Evaluate counterfactuals
    # Y_{X=1}
    # This implies setting X=1.
    # Y function sees parent X=1 -> returns 0.
    term_Y_do_X1 = Y @ {X: 1}
    assert config.evaluate(term_Y_do_X1) == 0

    # Y_{X=0} -> 1
    term_Y_do_X0 = Y @ {X: 0}
    assert config.evaluate(term_Y_do_X0) == 1

    # Test 3: Irrelevant intervention
    # X_{Y=1}
    # X is root, Y is downstream. Intervening on Y doesn't change X.
    term_X_do_Y1 = X @ {Y: 1}
    assert config.evaluate(term_X_do_Y1) == 0

    # Test 4: Self intervention
    # Y_{Y=0} -> 0
    term_Y_do_Y0 = Y @ {Y: 0}
    assert config.evaluate(term_Y_do_Y0) == 0


def test_satisfies_event():
    X = Variable("X")
    Y = Variable("Y")
    order = [X, Y]
    functions = {
        X: {(): 1},  # X := 1
        Y: {(0,): 0, (1,): 1},  # Y := X (Copy gate)
    }
    config = CanonicalConfiguration(functions, order)

    # Observational: X=1, Y=1
    e1 = (X == 1) & (Y == 1)
    assert config.satisfies(e1)

    # Counterfactual: Y_{X=0} should be 0
    e2 = Y @ {X: 0} == 0
    assert config.satisfies(e2)

    # Mixed event
    e3 = (X == 1) & (Y @ {X: 0} == 0)
    assert config.satisfies(e3)

    # False event
    e4 = X == 0
    assert not config.satisfies(e4)


def test_nested_intervention_logic():
    # If the code supports nested CF terms in intervention values (it seems to: recursive check)
    # term: Y @ {X: Z} (where Z is another var or term) - wait, Symbolic DSL maps {Var: Value}. Value can be CF term?
    # Symbolic.py: intervention: Dict[Variable, Union[Any, CounterfactualTerm]]

    # Let's try evaluating Y_{X = Z} where Z is another variable
    # Model: Z -> X -> Y
    Z = Variable("Z", (0, 1))
    X = Variable("X", (0, 1))
    Y = Variable("Y", (0, 1))

    order = [Z, X, Y]
    functions = {
        Z: {(): 1},  # Z := 1
        X: {(0,): 0, (1,): 1},  # X := Z
        # Y has parents Z, X (order is Z, X, Y)
        # We want Y := X. So we ignore Z.
        # Function maps (z, x) -> x.
        Y: {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1},
    }
    # So naturally Z=1, X=1, Y=1

    config = CanonicalConfiguration(functions, order)

    # Consider Y_{X = Z_{Z=0}}
    # Z_{Z=0} evaluates to 0. So effectively Y_{X=0} -> 0.

    cf_Z_is_0 = Z @ {
        Z: 0
    }  # Term representing Z set to 0? No, Z_{Z=0} is value of Z under do(Z=0)

    # Construct term: Y intervened such that X takes value of "Z_{Z=0}"
    # This requires Z_{Z=0} to be a CounterfactualTerm object? Yes.

    term_complex = Y @ {X: cf_Z_is_0}  # Y_{ X = Z_{Z=0} }

    # Expected evaluation:
    # 1. Evaluate intervention value: Z @ {Z:0}.
    #    config.evaluate(Z @ {Z:0}) -> 0.
    # 2. So we are evaluating Y_{X=0}.
    #    Y function with parent X=0 -> 0.

    assert config.evaluate(term_complex) == 0


def test_is_compatible():
    X = Variable("X")
    functions = {X: {(): 1}}
    config = CanonicalConfiguration(functions, [X])

    # Observation that matches
    obs1 = X == 1
    assert config.is_compatible(obs1)

    # Observation that incorrectly matches
    obs2 = X == 0
    assert not config.is_compatible(obs2)
