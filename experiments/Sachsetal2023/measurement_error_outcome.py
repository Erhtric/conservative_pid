# %%
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import time
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling.Sampling import BayesianModelSampling


repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.symbolic import Variable, P
from src.inference import ConservativePID


def sachs_measurement_error_bounds(pxy: pd.DataFrame, x1, x2):
    """
    Computes analytical bounds for P(Y_x1=1) - P(Y_x2=1) under measurement error.
    """
    p_x1 = pxy[pxy["X"] == x1]["probability"].sum()
    p_x2 = pxy[pxy["X"] == x2]["probability"].sum()

    p_y2_0_given_x1 = (
        pxy[(pxy["X"] == x1) & (pxy["Y2"] == 0)]["probability"].sum() / p_x1
    )
    p_y2_0_given_x2 = (
        pxy[(pxy["X"] == x2) & (pxy["Y2"] == 0)]["probability"].sum() / p_x2
    )

    assert p_y2_0_given_x1 != p_y2_0_given_x2, "Vacous Bounds"

    lb = max(-1.0, 2 * p_y2_0_given_x2 - 2 * p_y2_0_given_x1 - 1.0)
    ub = min(1.0, 2 * p_y2_0_given_x2 - 2 * p_y2_0_given_x1 + 1.0)

    return lb, ub


# %%
model = DiscreteBayesianNetwork([("X", "Y"), ("Y", "Y2")])
cpd_x = TabularCPD(variable="X", variable_card=2, values=[[0.5], [0.5]])

ur_dist = np.random.dirichlet(np.ones(2), size=1).T
cpd_ur = TabularCPD(
    variable="U_r",
    variable_card=2,
    values=ur_dist,
)

cpd_y = TabularCPD(
    variable="Y",
    variable_card=2,
    values=[
        [0.5, 0.7],
        [0.5, 0.3],
    ],
    evidence=["X"],
    evidence_card=[2],
)

cpd_y2 = TabularCPD(
    variable="Y2",
    variable_card=2,
    values=[
        [0.9, 0.4],
        [0.1, 0.6],
    ],
    evidence=["Y"],
    evidence_card=[2],
)

model.add_cpds(cpd_x, cpd_y, cpd_y2)
data = BayesianModelSampling(model).forward_sample(
    size=10000, seed=42, show_progress=False
)

# %%
obs_data_df = data.groupby(["X", "Y", "Y2"]).size().reset_index(name="count")
obs_data_df["probability"] = obs_data_df["count"] / obs_data_df["count"].sum()
obs_data_df = obs_data_df.drop(columns=["count"])

print(obs_data_df)

# %%
contrast_pairs = [(1, 0)]

X = Variable("X", domain=(0, 1))
Y = Variable("Y", domain=(0, 1))
Y2 = Variable("Y2", domain=(0, 1))

pid = ConservativePID(obs_data_df)

monotonicity_constraint = (Y2 @ {Y: 1}) >= (Y2 @ {Y: 0})
pid.add_monotonicity(monotonicity_constraint)

print(f"\n{'Contrast':<15} | {'Sachs (6.3)':<20} | {'Ours (LP)':<20}")
print("-" * 65)

sachs_lb, sachs_ub = sachs_measurement_error_bounds(obs_data_df, 1, 0)

query = P(Y @ {X: 1} == 1) - P(
    Y @ {X: 0} == 1,
)

timestart = time.time()
lb, ub, probs = pid.infer(
    query,
    causal_order=[X, Y, Y2],
    return_problems=True,
)
timeend = time.time()

sachs_str = f"[{sachs_lb:.3f}, {sachs_ub:.3f}]"
cpid_str = f"[{lb:.3f}, {ub:.3f}]"
print(f"{1} vs {0: <10} | {sachs_str:<20} | {cpid_str:<20}")

records = [
    {
        "x1": 1,
        "x2": 0,
        "target": f"P(Y_{1}=1) - P(Y_{0}=1)",
        "sachs_lb": sachs_lb,
        "sachs_ub": sachs_ub,
        "cpid_lb": lb,
        "cpid_ub": ub,
        "cpid_time_sec": timeend - timestart,
    }
]

results_df = pd.DataFrame(records)
save_path = Path(__file__).parent / "sachs_6_3_measurement_error_results.csv"
results_df.to_csv(save_path, index=False)

# %%
