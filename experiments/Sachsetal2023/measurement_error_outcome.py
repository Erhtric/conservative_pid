# %%
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

    lb = max(-1.0, 2 * p_y2_0_given_x2 - 2 * p_y2_0_given_x1 - 1.0)
    ub = min(1.0, 2 * p_y2_0_given_x2 - 2 * p_y2_0_given_x1 + 1.0)

    return lb, ub


# %%
model = DiscreteBayesianNetwork([("X", "Y"), ("U_r", "Y"), ("Y", "Y2"), ("U_r", "Y2")])

cpd_x = TabularCPD(variable="X", variable_card=2, values=[[0.5], [0.5]])
cpd_ur = TabularCPD(variable="U_r", variable_card=2, values=[[0.6], [0.4]])

cpd_y = TabularCPD(
    variable="Y",
    variable_card=2,
    values=[
        [0.7, 0.5, 0.3, 0.1],
        [0.3, 0.5, 0.7, 0.9],
    ],
    evidence=["X", "U_r"],
    evidence_card=[2, 2],
)

cpd_y2 = TabularCPD(
    variable="Y2",
    variable_card=2,
    values=[
        [0.9, 0.8, 0.1, 0.0],
        [0.1, 0.2, 0.9, 1.0],
    ],
    evidence=["Y", "U_r"],
    evidence_card=[2, 2],
)

model.add_cpds(cpd_x, cpd_ur, cpd_y, cpd_y2)
ie = VariableElimination(model)

# %%
joint_xy2 = ie.query(variables=["X", "Y2"], joint=True)

obs_data_df = pd.DataFrame(
    {
        "X": [0, 0, 1, 1],
        "Y2": [0, 1, 0, 1],
        "probability": joint_xy2.values.flatten(),
    }
)

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

query = P(Y2 @ {X: 1} == 1)
#  - P(Y @ {X: 0} == 1)

timestart = time.time()
lb, ub, probs = pid.infer(query, causal_order=[X, Y, Y2], return_problems=True)
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

# Show distribution of decision variables using matplotlib
import matplotlib.pyplot as plt

prob = probs[
    "lower"
]  # or "upper", they should have the same distribution of decision variables
decision_vars = prob.variables()  # Get the decision variables from the LP problem
values = [
    var.value() for var in decision_vars
]  # Get the values of the decision variables in the solution
# By definition of the decisions variables they are already a distribution, we simply plot them
plt.bar(range(len(values)), values)
plt.xlabel("World Index")
plt.ylabel("Probability")
plt.title("Distribution over Worlds in the LP Solution")
plt.show()

# %%
