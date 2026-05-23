from cpid.mdd import ResponseSignatureMDD
from cpid.io import CausalQuery, AtomicCounterfactual, CausalExpression


def test_mdd_instantiation_and_attributes():
    """Verify ResponseSignatureMDD initializes correct properties."""
    domains = {"X": 2, "Y": 2}
    ordered_nodes = ["X", "Y"]
    structure = {"X": [], "Y": ["X"]}
    contexts = [{}, {"X": 0}, {"X": 1}]

    mdd = ResponseSignatureMDD(
        domains=domains,
        ordered_nodes=ordered_nodes,
        structure=structure,
        contexts=contexts,
    )

    assert mdd.domains == domains
    assert mdd.ordered_nodes == ordered_nodes
    assert mdd.structure == structure
    assert mdd.contexts == contexts
    assert mdd.paths == {(): 1}  # Initial root path with multiplicity 1


def test_mdd_build_path_generation_and_reduction():
    """Verify topological construction and state-merging reduction in MDD.

    For X -> Y with domains X: 2, Y: 2, and PNS contexts:
    - X (no parents): 2 values (0 and 1).
    - Y (parent X): active configurations are natural X, X=0, and X=1.
      This translates to 2 active configs (k=2) since natural X must match either 0 or 1.
      Therefore, Y has 2^2 = 4 outputs for each branch of X, leading to 8 paths.
      Multiplicity multiplier is 2^(2-2) = 1.
      All paths must have multiplicity 1, and total sum of multiplicities must be 8.
    """
    domains = {"X": 2, "Y": 2}
    ordered_nodes = ["X", "Y"]
    structure = {"X": [], "Y": ["X"]}
    contexts = [{}, {"X": 0}, {"X": 1}]

    mdd = ResponseSignatureMDD(
        domains=domains,
        ordered_nodes=ordered_nodes,
        structure=structure,
        contexts=contexts,
    )
    mdd.build()

    # Verify size and properties of generated paths
    assert len(mdd.paths) == 8
    assert sum(mdd.paths.values()) == 8
    assert all(mult == 1 for mult in mdd.paths.values())

    # Every path must have length 2 (X and Y context values)
    for path in mdd.paths.keys():
        assert len(path) == 2
        # Context 0: natural, Context 1: X=0, Context 2: X=1
        x_ctx = path[0]
        y_ctx = path[1]
        assert len(x_ctx) == 3
        assert len(y_ctx) == 3
        # Interventions in X: X context values must match interventions
        assert x_ctx[1] == 0  # Context 1 has intervention X=0
        assert x_ctx[2] == 1  # Context 2 has intervention X=1


def test_mdd_multiplicity_scaling():
    """Verify that combinatorial multipliers scale correctly for large domains.

    For X -> Y with X: 2, Y: 10, and PNS contexts:
    - X: 2 values.
    - Y: parent X. Total configurations for Y is |D(X)| = 2.
      Unique active parent configurations under contexts is k = 2.
      Inactive configurations = 2 - 2 = 0.
      Multipliers must be 10^0 = 1.
      Total sum of multiplicities must be 2 * 10^2 = 200.
    """
    domains = {"X": 2, "Y": 10}
    ordered_nodes = ["X", "Y"]
    structure = {"X": [], "Y": ["X"]}
    contexts = [{}, {"X": 0}, {"X": 1}]

    mdd = ResponseSignatureMDD(
        domains=domains,
        ordered_nodes=ordered_nodes,
        structure=structure,
        contexts=contexts,
    )
    mdd.build()

    assert sum(mdd.paths.values()) == 200


def test_mdd_equivalence_classes():
    """Verify MDD get_equivalence_classes output formatting and correctness."""
    domains = {"X": 2, "Y": 2}
    ordered_nodes = ["X", "Y"]
    structure = {"X": [], "Y": ["X"]}
    contexts = [{}, {"X": 0}, {"X": 1}]

    mdd = ResponseSignatureMDD(
        domains=domains,
        ordered_nodes=ordered_nodes,
        structure=structure,
        contexts=contexts,
    )
    mdd.build()

    pns_query = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=0, interventions={"X": 0}),
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1}),
        ]
    )
    flat_expr = CausalExpression({pns_query: 1.0})
    context_map = {frozenset(): 0, frozenset([("X", 0)]): 1, frozenset([("X", 1)]): 2}

    eq_classes = mdd.get_equivalence_classes(flat_expr, {}, context_map)

    # Check mapping format
    for (nat_tuple, num_coeff, den_coeff), count in eq_classes.items():
        assert len(nat_tuple) == 2
        assert isinstance(num_coeff, float)
        assert isinstance(den_coeff, float)
        assert isinstance(count, int)
        assert count > 0

    assert sum(eq_classes.values()) == 8
