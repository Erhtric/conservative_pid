import pyagrum as gum
from cpid import AtomicCounterfactual, CausalQuery
from cpid import OrderFunctionalLPSolver
import pandas as pd
import numpy as np
import itertools


def generate_monotonic_tian_df(
    domains: dict[str, int] = {"X": 2, "Y": 2},
    sample_size: int = 10000,
    seed: int = 42,
    include_counterfactuals: bool = True,
) -> pd.DataFrame:
    np.random.seed(seed)

    dx = domains["X"]
    dy = domains["Y"]
    num_functions = dy**dx
    p_uy = np.random.dirichlet(np.ones(num_functions))

    # Force strict positive monotonicity
    p_uy[2] = 0.0
    p_uy = p_uy / p_uy.sum()  # Re-normalize

    p_x = np.random.dirichlet(np.ones(domains["X"]))

    X = np.random.choice(list(range(dx)), size=sample_size, p=p_x)
    Uy = np.random.choice(list(range(num_functions)), size=sample_size, p=p_uy)

    response_table = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = response_table[Uy, X]

    df = pd.DataFrame({"X": X, "Y": Y})
    if include_counterfactuals:
        for x_val in range(dx):
            df[f"Y_{{X={x_val}}}"] = response_table[Uy, x_val]

    return df


def generate_canonical_tian_df(
    domains: dict[str, int] = {"X": 2, "Y": 2},
    seed: int = 42,
    sample_size: int = 10000,
    include_counterfactuals: bool = True,
) -> pd.DataFrame:
    np.random.seed(seed)

    dx = domains["X"]
    dy = domains["Y"]
    num_functions = dy**dx

    # Sample the marginal exogenous distributions P(X) and P(U_Y)
    p_x = np.random.dirichlet(np.ones(dx))
    X = np.random.choice(range(dx), size=sample_size, p=p_x)

    p_uy = np.random.dirichlet(np.ones(num_functions))
    Uy = np.random.choice(range(num_functions), size=sample_size, p=p_uy)
    response_table = np.array(list(itertools.product(range(dy), repeat=dx)))

    Y = response_table[Uy, X]

    df = pd.DataFrame({"X": X, "Y": Y})

    if include_counterfactuals:
        for x_val in range(dx):
            df[f"Y_{{X={x_val}}}"] = response_table[Uy, x_val]

    return df


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
