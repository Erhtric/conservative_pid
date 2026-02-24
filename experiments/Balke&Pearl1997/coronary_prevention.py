# %%
import sys
from pathlib import Path
from src.symbolic import Variable
import pandas as pd
import time

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.symbolic import P
from src.inference import ConservativePID

# %%
# Data from Table 3 in Balke & Pearl (1997)
# Z: 0=Placebo Assignment, 1=Cholestyramine Assignment
# D: 0=Low Dose (d0), 1=High Dose (d1), 2=Middle/Partial (dm)
# Y: 0=No Reduction (<38), 1=Reduction (>=38)

# The table provides Conditional Probabilities P(y, d | z).
# We assume P(Z=0) = P(Z=1) = 0.5 to construct the joint distribution.

# Mappings based on text (Sec 4.2) and Table 3:
# z0 (Control):
#   d0 (Placebo taken): P(y0|z0)=0.971, P(y1|z0)=0.029
#   d1, dm: 0 probability
# z1 (Treatment):
#   d0 (Low):    P(y0,d0|z1)=0.024, P(y1,d0|z1)=0.000
#   dm (Middle): P(y0,dm|z1)=0.436, P(y1,dm|z1)=0.146
#   d1 (High):   P(y0,d1|z1)=0.103, P(y1,d1|z1)=0.291

# P(y,d,z) = P(y,d|z) * P(z)
data = [
    {"Z": 0, "D": 0, "Y": 0, "prob": 0.971 * 0.5},
    {"Z": 0, "D": 0, "Y": 1, "prob": 0.029 * 0.5},
    {"Z": 0, "D": 1, "Y": 0, "prob": 0.000 * 0.5},
    {"Z": 0, "D": 1, "Y": 1, "prob": 0.000 * 0.5},
    {"Z": 0, "D": 2, "Y": 0, "prob": 0.000 * 0.5},
    {"Z": 0, "D": 2, "Y": 1, "prob": 0.000 * 0.5},
    {"Z": 1, "D": 0, "Y": 0, "prob": 0.024 * 0.5},
    {"Z": 1, "D": 0, "Y": 1, "prob": 0.000 * 0.5},
    {"Z": 1, "D": 2, "Y": 0, "prob": 0.436 * 0.5},  # dm
    {"Z": 1, "D": 2, "Y": 1, "prob": 0.146 * 0.5},  # dm
    {"Z": 1, "D": 1, "Y": 0, "prob": 0.103 * 0.5},
    {"Z": 1, "D": 1, "Y": 1, "prob": 0.291 * 0.5},
]


obs_data_df = pd.DataFrame(data)
obs_data_df = obs_data_df.rename(columns={"prob": "probability"})

balke_lb = 0.262
balke_ub = 0.868

# Define Variables
# Z is binary, Y is binary
# D is Ternary: 0=d0, 1=d1, 2=dm
Z = Variable("Z", domain=(0, 1))
D = Variable("D", domain=(0, 1, 2))
Y = Variable("Y", domain=(0, 1))

pid = ConservativePID(obs_data_df)

# Query: ACE between High Dose (d1=1) and Low Dose (d0=0)
# ACE = P(Y_{d1}=1) - P(Y_{d0}=1)
ace = P(Y @ {D: 1} == 1) - P(Y @ {D: 0} == 1)

print(f"\n{'ACE':<15} | {'Balke (Lipid)':<20} | {'Ours (LP)':<20}")
print("-" * 65)

timestart = time.time()
lb, ub = pid.infer(ace)
timeend = time.time()

records = [
    {
        "x1": 1,
        "x2": 0,
        "target": f"P(Y_{1}=1) - P(Y_{0}=1)",
        "balke_lb": balke_lb,
        "balke_ub": balke_ub,
        "cpid_lb": lb,
        "cpid_ub": ub,
        "cpid_time_sec": timeend - timestart,
    }
]

balke_str = f"[{balke_lb:.3f}, {balke_ub:.3f}]"
cpid_str = f"[{lb:.3f}, {ub:.3f}]"
print(f"{1} vs {0: <10} | {balke_str:<20} | {cpid_str:<20}")

results_df = pd.DataFrame(records)
save_path = Path(__file__).parent / "balke_4_2_lipid_results.csv"
results_df.to_csv(save_path, index=False)
# %%
# Separately
p00_0 = obs_data_df[
    (obs_data_df["Z"] == 0) & (obs_data_df["D"] == 0) & (obs_data_df["Y"] == 0)
]["probability"].values[0]
p00_1 = obs_data_df[
    (obs_data_df["Z"] == 1) & (obs_data_df["D"] == 0) & (obs_data_df["Y"] == 0)
]["probability"].values[0]
p01_0 = obs_data_df[
    (obs_data_df["Z"] == 0) & (obs_data_df["D"] == 1) & (obs_data_df["Y"] == 0)
]["probability"].values[0]
p01_1 = obs_data_df[
    (obs_data_df["Z"] == 1) & (obs_data_df["D"] == 1) & (obs_data_df["Y"] == 0)
]["probability"].values[0]
p10_0 = obs_data_df[
    (obs_data_df["Z"] == 0) & (obs_data_df["D"] == 0) & (obs_data_df["Y"] == 1)
]["probability"].values[0]
p10_1 = obs_data_df[
    (obs_data_df["Z"] == 1) & (obs_data_df["D"] == 0) & (obs_data_df["Y"] == 1)
]["probability"].values[0]
p11_0 = obs_data_df[
    (obs_data_df["Z"] == 0) & (obs_data_df["D"] == 1) & (obs_data_df["Y"] == 1)
]["probability"].values[0]
p11_1 = obs_data_df[
    (obs_data_df["Z"] == 1) & (obs_data_df["D"] == 1) & (obs_data_df["Y"] == 1)
]["probability"].values[0]

balke_lb_control: float = max(
    p10_1, p10_0, p10_0 + p11_0 - p00_1 - p11_1, p01_0 + p10_0 - p00_1 - p01_1
)
balke_ub_control: float = min(
    1 - p00_1, 1 - p00_0, p01_0 + p10_0 + p10_1 + p11_1, p10_0 + p11_0 + p01_1 + p10_1
)
balke_lb_treated: float = max(
    p11_0, p11_1, -p00_0 - p01_0 + p00_1 + p11_1, -p01_0 - p10_0 + p10_1 + p11_1
)
balke_ub_treated: float = min(
    1 - p01_1, 1 - p01_0, p00_0 + p11_0 + p10_1 + p11_1, p10_0 + p11_0 + p00_1 + p11_1
)

ace_treated = P(Y @ {D: 1} == 1)
ace_control = P(Y @ {D: 0} == 1)
timestart = time.time()
lb_treated, ub_treated = pid.infer(ace_treated)
lb_control, ub_control = pid.infer(ace_control)
timeend = time.time()

records = [
    {
        "x1": 1,
        "x2": 0,
        "target": f"P(Y_{1}=1)",
        "balke_lb": balke_lb_treated,
        "balke_ub": balke_ub_treated,
        "cpid_lb": lb_treated,
        "cpid_ub": ub_treated,
        "cpid_time_sec": timeend - timestart,
    },
    {
        "x1": 0,
        "x2": 0,
        "target": f"P(Y_{0}=1)",
        "balke_lb": balke_lb_control,
        "balke_ub": balke_ub_control,
        "cpid_lb": lb_control,
        "cpid_ub": ub_control,
        "cpid_time_sec": timeend - timestart,
    },
]

results_df = pd.DataFrame(records)
save_path = Path(__file__).parent / "lipid_treated_control_results.csv"
results_df.to_csv(save_path, index=False)


print(f"\n{'Treated':<15} | {'Balke (Vitamin A)':<20} | {'Ours (LP)':<20}")
print("-" * 65)
balke_str = f"[{balke_lb_treated:.3f}, {balke_ub_treated:.3f}]"
cpid_str = f"[{lb_treated:.3f}, {ub_treated:.3f}]"
print(f"{'D=1':<15} | {balke_str:<20} | {cpid_str:<20}")

print(f"\n{'Control':<15} | {'Balke (Vitamin A)':<20} | {'Ours (LP)':<20}")
print("-" * 65)
balke_str = f"[{balke_lb_control:.3f}, {balke_ub_control:.3f}]"
cpid_str = f"[{lb_control:.3f}, {ub_control:.3f}]"
print(f"{'D=0':<15} | {balke_str:<20} | {cpid_str:<20}")
