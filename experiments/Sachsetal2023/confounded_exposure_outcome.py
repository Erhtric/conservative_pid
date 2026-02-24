from pathlib import Path
import sys
import pandas as pd
import time
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.symbolic import Variable, P
from src.inference import ConservativePID


def sachs_ternary_bounds(pxy: pd.DataFrame, x1, x2):
    """
    Computes analytical bounds for P(Y_x1=1) - P(Y_x2=1)

    px_y: Dictionary {(x, y): probability}
    """
    # Notation mapping: P(X=x, Y=y)
    p_x1_y1 = pxy[(pxy["X"] == x1) & (pxy["Y"] == 1)]["probability"].sum()
    p_x2_y0 = pxy[(pxy["X"] == x2) & (pxy["Y"] == 0)]["probability"].sum()

    p_x1_y0 = pxy[(pxy["X"] == x1) & (pxy["Y"] == 0)]["probability"].sum()
    p_x2_y1 = pxy[(pxy["X"] == x2) & (pxy["Y"] == 1)]["probability"].sum()

    # Lower Bound: P(x1, y1) + P(x2, y0) - 1
    lb = p_x1_y1 + p_x2_y0 - 1.0

    # Upper Bound: 1 - P(x1, y0) - P(x2, y1)
    ub = 1.0 - p_x1_y0 - p_x2_y1

    return max(-1.0, lb), min(1.0, ub)


# %%
model = DiscreteBayesianNetwork([("U", "X"), ("U", "Y"), ("X", "Y")])

cpd_u = TabularCPD("U", 2, [[0.5], [0.5]])

# X (Ternary Exposure) depends on U
cpd_x = TabularCPD(
    "X",
    3,
    [
        [0.6, 0.2],  # P(X=0) - U affects assignment
        [0.3, 0.3],  # P(X=1)
        [0.1, 0.5],
    ],  # P(X=2)
    evidence=["U"],
    evidence_card=[2],
)

# Y (Binary Outcome) depends on X and U
cpd_y = TabularCPD(
    "Y",
    2,
    [
        [0.8, 0.7, 0.6, 0.5, 0.4, 0.2],  # P(Y=0)
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    ],  # P(Y=1)
    evidence=["X", "U"],
    evidence_card=[3, 2],
)

model.add_cpds(cpd_u, cpd_x, cpd_y)
ie = VariableElimination(model)
# %%
joint_xy = ie.query(variables=["X", "Y"], joint=True)

obs_data_df = pd.DataFrame(
    {
        "X": [0, 0, 1, 1, 2, 2],
        "Y": [0, 1, 0, 1, 0, 1],
        "probability": joint_xy.values.flatten(),
    }
)

print(obs_data_df)
# %%
contrast_pairs = [(1, 0), (2, 0), (2, 1)]

X = Variable("X", domain=(0, 1, 2))
Y = Variable("Y", domain=(0, 1))

pid = ConservativePID(obs_data_df)

print(f"\n{'Contrast':<15} | {'Sachs (6.1)':<20} | {'Ours (LP)':<20}")
print("-" * 65)

records = []
for x1, x2 in contrast_pairs:
    sachs_lb, sachs_ub = sachs_ternary_bounds(obs_data_df, x1, x2)

    query = P(Y @ {X: x1} == 1) - P(Y @ {X: x2} == 1)

    timestart = time.time()
    cpid_bounds = pid.infer(query)
    timeend = time.time()

    sachs_str = f"[{sachs_lb:.3f}, {sachs_ub:.3f}]"
    cpid_str = f"[{cpid_bounds[0]:.3f}, {cpid_bounds[1]:.3f}]"
    print(f"{x1} vs {x2: <10} | {sachs_str:<20} | {cpid_str:<20}")

    records.append(
        {
            "x1": x1,
            "x2": x2,
            "target": f"P(Y_{x1}=1) - P(Y_{x2}=1)",
            "sachs_lb": sachs_lb,
            "sachs_ub": sachs_ub,
            "cpid_lb": cpid_bounds[0],
            "cpid_ub": cpid_bounds[1],
            "cpid_time_sec": timeend - timestart,
        }
    )

results_df = pd.DataFrame(records)
save_path = Path(__file__).parent / "sachs_6_1_ternary_contrast_results.csv"
results_df.to_csv(save_path, index=False)
