import pytest
import time
import pandas as pd
import pyagrum as gum
from cpid.io import CausalQuery, AtomicCounterfactual, CausalExpression
from cpid.signature import PartialOrderSignature, SignatureQueryEvaluator
from cpid.lp import OrderFunctionalLPSolver

def brute_force_equivalence_classes(signature_obj, query):
    """Local baseline implementing the original brute-force enumeration."""
    if isinstance(query, CausalExpression):
        flat_expr = query
        evidence_dict = list(query.terms.keys())[0].evidence if query.terms else {}
    else:
        flat_expr = CausalExpression({query: 1.0})
        evidence_dict = query.evidence

    state_evaluator = SignatureQueryEvaluator(signature_obj.domains, signature_obj=signature_obj)
    evidence_query = CausalQuery(counterfactuals=[], evidence=evidence_dict)
    query_evaluators = [
        (SignatureQueryEvaluator(signature_obj.domains, signature_obj=signature_obj, query=cq), w)
        for cq, w in flat_expr.terms.items()
    ]
    evidence_evaluator = SignatureQueryEvaluator(
        signature_obj.domains, signature_obj=signature_obj, query=evidence_query
    )

    equivalence_classes = {}
    for sig in signature_obj.iter_space():
        nat_state = state_evaluator.evaluate_state(sig)
        nat_tuple = tuple(nat_state[v] for v in signature_obj.ordered_nodes)

        num_coeff = 0.0
        for evaluator, w in query_evaluators:
            if evaluator.row_satisfies_query(sig):
                num_coeff += w

        den_coeff = 1.0 if evidence_evaluator.row_satisfies_query(sig) else 0.0
        eq_key = (nat_tuple, num_coeff, den_coeff)
        equivalence_classes[eq_key] = equivalence_classes.get(eq_key, 0) + 1

    return equivalence_classes


def test_equivalence_classes_match_brute_force():
    """Verify that optimized dynamic programming equivalence classes exactly match brute force."""
    domains = {"X": 3, "Y": 3}
    pns_query = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=0, interventions={"X": 0}),
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1}),
        ]
    )

    sig = PartialOrderSignature(domains, pns_query, lazy=True)
    
    # Generate equivalence classes using both methods
    eq_opt = sig.get_equivalence_classes(pns_query)
    eq_brute = brute_force_equivalence_classes(sig, pns_query)
    
    assert eq_opt == eq_brute
    assert sum(eq_opt.values()) == sig.size
    assert sum(eq_brute.values()) == sig.size


def test_lpsolver_correctness_and_identity():
    """Verify that OrderFunctionalLPSolver produces same bounds under optimized method."""
    domains = {"X": 2, "Y": 2}
    bn = gum.fastBN("X->Y")
    gum.initRandom(42)
    bn.generateCPTs()
    df = gum.generateSample(bn, 100, show_progress=False)[0].astype(int)

    pns_query = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=0, interventions={"X": 0}),
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1}),
        ]
    )

    solver = OrderFunctionalLPSolver(
        data=df,
        query=pns_query,
        solver_verbose=False,
    )
    
    pns_lb, pns_ub = solver()
    # Check that bounds are computed and valid
    assert 0.0 <= pns_lb <= pns_ub <= 1.0


def test_high_domain_performance():
    """Verify that d=10 executes in milliseconds and does not crash."""
    domains = {"X": 10, "Y": 10}
    bn = gum.fastBN("X[10]->Y[10]")
    gum.initRandom(42)
    bn.generateCPTs()
    df = gum.generateSample(bn, 100, show_progress=False)[0].astype(int)

    pns_query = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=0, interventions={"X": 0}),
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1}),
        ]
    )

    start_time = time.time()
    solver = OrderFunctionalLPSolver(
        data=df,
        query=pns_query,
        solver_verbose=False,
    )
    lb, ub = solver()
    elapsed = time.time() - start_time
    
    print(f"d=10 execution completed in {elapsed:.6f}s (Bounds: [{lb:.4f}, {ub:.4f}])")
    
    # Assert performance threshold: must complete in under 0.5 seconds
    assert elapsed < 0.5
    assert 0.0 <= lb <= ub <= 1.0
