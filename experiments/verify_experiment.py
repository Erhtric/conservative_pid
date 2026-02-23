import os
import sys

import numpy as np
import pandas as pd
import pymc as pm
from pymc.distributions.continuous import CircularContinuous

# Add parent directory to path
sys.path.append(os.getcwd())

from inference import ConservativePID
from symbolic import Event, P, Variable


def run_experiment():
    print("Running Experiment Verification...")

    # True Parameters
    px = 0.5
    p_m_cond_x = {0: 0.2, 1: 0.7}
    p_y_cond_xm = {(0, 0): 0.1, (0, 1): 0.3, (1, 0): 0.4, (1, 1): 0.8}

    n_samples = 500  # Small sample for speed
    RANDOM_SEED = 42

    print("Generating data...")
    with pm.Model() as model:
        # X
        X_obs = pm.Bernoulli("X", p=px, shape=n_samples)

        # M | X
        p_m = pm.math.switch(X_obs, p_m_cond_x[1], p_m_cond_x[0])
        M_obs = pm.Bernoulli("M", p=p_m, shape=n_samples)

        # Y | X, M
        # probs_y = np.array(
        #     [
        #         p_y_cond_xm[(0, 0)],
        #         p_y_cond_xm[(0, 1)],
        #         p_y_cond_xm[(1, 0)],
        #         p_y_cond_xm[(1, 1)],
        #     ]
        # )
        # p_y = probs_y[idx]

        # Use nested switch for safe symbolic indexing
        p_y_x1 = pm.math.switch(M_obs, p_y_cond_xm[(1, 1)], p_y_cond_xm[(1, 0)])
        p_y_x0 = pm.math.switch(M_obs, p_y_cond_xm[(0, 1)], p_y_cond_xm[(0, 0)])
        p_y = pm.math.switch(X_obs, p_y_x1, p_y_x0)

        Y_obs = pm.Bernoulli("Y", p=p_y, shape=n_samples)

        trace = pm.sample_prior_predictive(samples=1, random_seed=RANDOM_SEED)

    # Extract data
    df = pd.DataFrame(
        {
            "X": trace.prior["X"].values.flatten(),
            "M": trace.prior["M"].values.flatten(),
            "Y": trace.prior["Y"].values.flatten(),
        }
    )

    observational_data = {}
    total = len(df)

    for x in [0, 1]:
        for m in [0, 1]:
            for y in [0, 1]:
                count = len(df[(df["X"] == x) & (df["M"] == m) & (df["Y"] == y)])
                observational_data[(x, m, y)] = count / total

    print("Observational Data generated.")

    X_var = Variable("X", (0, 1))
    M_var = Variable("M", (0, 1))
    Y_var = Variable("Y", (0, 1))
    variables = [X_var, M_var, Y_var]

    cpid = ConservativePID(variables, observational_data)

    print("Computing TE...")
    q_te = P(Y_var @ {X_var: 1} == 1) - P(Y_var @ {X_var: 0} == 1)
    lb_te, ub_te = cpid.infer(q_te)
    print(f"TE: [{lb_te}, {ub_te}]")

    print("Computing NDE...")
    # NDE: P(Y_{X=1, M_{X=0}}=1) - P(Y_{X=0, M_{X=0}}=1)
    nested_M = M_var @ {X_var: 0}
    y_nde_term = Y_var @ {X_var: 1, M_var: nested_M}
    y_ref_term = Y_var @ {X_var: 0}

    q_nde = P(y_nde_term == 1) - P(y_ref_term == 1)
    lb_nde, ub_nde = cpid.infer(q_nde)
    print(f"NDE: [{lb_nde}, {ub_nde}]")

    print("Computing NIE...")
    # NIE: P(Y_{X=0, M_{X=1}}=1) - P(Y_{X=0, M_{X=0}}=1)
    nested_M_nie = M_var @ {X_var: 1}
    y_nie_term = Y_var @ {X_var: 0, M_var: nested_M_nie}

    q_nie = P(y_nie_term == 1) - P(y_ref_term == 1)
    lb_nie, ub_nie = cpid.infer(q_nie)
    print(f"NIE: [{lb_nie}, {ub_nie}]")

    print("Done!")


if __name__ == "__main__":
    run_experiment()
