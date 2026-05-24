import pyagrum as gum
from cpid import AtomicCounterfactual, CausalQuery
from cpid import OrderFunctionalLPSolver
import pandas as pd


def generate_tian_df(
    domains: dict[str, int] = {"X": 2, "Y": 2}, seed: int = 42, sample_size: int = 10000
) -> pd.DataFrame:
    bn = gum.fastBN(f"X[{domains['X']}]->Y[{domains['Y']}]")
    gum.initRandom(seed)
    bn.generateCPTs()

    gum.initRandom(seed)
    df = gum.generateSample(bn, sample_size, show_progress=False)[0]
    return df


def tian_pns_bounds(df: pd.DataFrame):
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
    return pns_lb, pns_ub


def tian_pn_bounds(df: pd.DataFrame):
    pn_query = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
        ],
        evidence={"X": 0, "Y": 0},
    )

    solver = OrderFunctionalLPSolver(
        data=df,
        query=pn_query,
        solver_verbose=False,
    )

    pn_lb, pn_ub = solver()
    return pn_lb, pn_ub


def tian_ps_bounds(df: pd.DataFrame):
    ps_query = CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=0, interventions={"X": 0})
        ],
        evidence={"X": 1, "Y": 1},
    )

    solver = OrderFunctionalLPSolver(
        data=df,
        query=ps_query,
        solver_verbose=False,
    )
    ps_lb, ps_ub = solver()
    return ps_lb, ps_ub
