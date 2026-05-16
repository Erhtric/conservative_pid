import pytest

from cpid.io import AtomicCounterfactual, CausalQuery, CausalExpression
from cpid.signature import TotalOrderSignature


def test_signature_compatible_with_intermediate_node():
    domains = {"X": 2, "W": 2, "Y": 2}
    sig = TotalOrderSignature(domains=domains, total_order=["X", "W", "Y"])

    # Query: X < Y (Y under intervention on X)
    q = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        ]
    )

    # Direct query and as a CausalExpression should both be compatible
    assert sig.is_compatible(q) is True
    expr = CausalExpression({q: 1.0})
    assert sig.is_compatible(expr) is True


def test_signature_missing_variable_raises():
    domains = {"X": 2, "W": 2}
    sig = TotalOrderSignature(domains=domains, total_order=["X", "W"])

    q = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        ]
    )
    with pytest.raises(ValueError):
        sig.is_compatible(q)


def test_conflicting_orders_raise():
    domains = {"X": 2, "Y": 2}
    sig = TotalOrderSignature(domains=domains, total_order=["X", "Y"])

    q1 = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        ]
    )
    q2 = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="X", target_val=1, interventions={"Y": 1})
        ]
    )

    expr = CausalExpression({q1: 1.0, q2: 1.0})
    with pytest.raises(ValueError):
        sig.is_compatible(expr)
