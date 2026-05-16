import pytest

from cpid.io import AtomicCounterfactual, CausalQuery
from cpid.signature import SignatureQueryEvaluator, TotalOrderSignature


def test_signature_query_evaluator_row_satisfies_query_true():
    domains = {"X": 2, "Y": 2}
    signature = TotalOrderSignature(domains=domains, total_order=["X", "Y"])

    query = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        ]
    )
    evaluator = SignatureQueryEvaluator(
        domains=domains, signature_obj=signature, query=query
    )

    # X is constant 1. Y copies X: y(0)=0, y(1)=1.
    sig_row = ((1,), (0, 1))

    assert evaluator.row_satisfies_query(sig_row) is True


def test_signature_query_evaluator_respects_evidence():
    domains = {"X": 2, "Y": 2}
    signature = TotalOrderSignature(domains=domains, total_order=["X", "Y"])

    query = CausalQuery(
        counterfactuals=[AtomicCounterfactual(target_var="Y", target_val=1)],
        evidence={"X": 0},
    )
    evaluator = SignatureQueryEvaluator(
        domains=domains, signature_obj=signature, query=query
    )

    # X is constant 1, so evidence X=0 should fail.
    sig_row = ((1,), (0, 1))

    assert evaluator.row_satisfies_query(sig_row) is False


def test_signature_query_evaluator_rejects_nested_interventions_without_unnesting():
    domains = {"X": 2, "Y": 2, "Z": 2}
    signature = TotalOrderSignature(domains=domains, total_order=["Z", "X", "Y"])

    # Nested intervention represented as dict value.
    query = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(
                target_var="Y", target_val=1, interventions={"X": {"Z": 0}}
            )
        ]
    )
    evaluator = SignatureQueryEvaluator(
        domains=domains, signature_obj=signature, query=query
    )

    sig_row = ((0,), (0, 1), (0, 1, 0, 1))

    with pytest.raises(ValueError, match="Nested interventions detected"):
        evaluator.row_satisfies_query(sig_row)
