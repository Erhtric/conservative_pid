import pytest
import networkx as nx

from cpid.signature import TotalOrderSignature, PartialOrderSignature
from cpid.io import AtomicCounterfactual, CausalQuery, CausalExpression


def test_total_order_signature_basic():
    domains = {"X": 2, "Y": 2}
    total_order = ["X", "Y"]
    sig = TotalOrderSignature(domains, total_order)

    # ordered nodes and structure
    assert sig.ordered_nodes == ["X", "Y"]
    assert sig.structure["X"] == []
    assert sig.structure["Y"] == ["X"]

    # space size: X -> 2 funcs, Y -> 2^(|parents|)=2^2=4 funcs => total 8
    assert sig.size == 8


def test_partial_order_signature_from_query():
    # small example: intervention on X, outcome Y
    domains = {"X": 2, "Y": 2}
    ac = AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
    q = CausalQuery(counterfactuals=[ac], evidence={})

    sig = PartialOrderSignature(domains, q)

    # roots should include X
    assert "X" in sig.ordered_nodes
    # outcomes should include Y and come after roots
    assert sig.ordered_nodes[-1] == "Y"
    assert sig.structure["X"] == []
    assert sig.structure["Y"] == ["X"]

    # graph builder produces networkx graph with connectionstyle metadata
    G = sig.build_canonical_pscm()
    assert isinstance(G, nx.DiGraph)
    # edges exist and have 'connectionstyle' attribute
    found = False
    for u, v, data in G.edges(data=True):
        if "connectionstyle" in data:
            found = True
            break
    assert found


def test_partial_order_signature_from_expression():
    # Use four variables but keep signature space small by checking structure only
    domains = {"A": 2, "B": 2, "C": 2, "D": 2}
    q1 = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="C", target_val=1, interventions={"A": 1})
        ]
    )
    q2 = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="D", target_val=0, interventions={"B": 0})
        ]
    )

    expr = q1 - q2  # CausalExpression

    sig = PartialOrderSignature(domains, expr)

    # roots should be the intervention variables sorted
    assert sig.ordered_nodes[0:2] == ["A", "B"]
    # outcomes should include C and D
    assert set(sig.ordered_nodes[2:]) == {"C", "D"}
    # structure keys exist
    for v in ["A", "B", "C", "D"]:
        assert v in sig.structure
    # each root has empty parents
    assert sig.structure["A"] == []
    assert sig.structure["B"] == []
    # each outcome should have at least the roots as parents
    for out in ["C", "D"]:
        for r in ["A", "B"]:
            assert r in sig.structure[out]
