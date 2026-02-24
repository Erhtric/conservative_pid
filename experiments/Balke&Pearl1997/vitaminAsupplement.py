# Study vitamin A supplementation in northern Sumatra by Sommer et al. and Sommer and Zeger
# 450 villagers, 221 were randomly assinged to the treatment group and 220 to the control
# %%
import sys
from pathlib import Path
from src.symbolic import Variable, P
import pandas as pd
import time

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.symbolic import Variable, P
from src.inference import ConservativePID

# %%
# Table Sec.4.1 counts
obs_data_df = pd.DataFrame(
    [
        [0, 0, 0, 74],
        [0, 0, 1, 11514],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 34],
        [1, 0, 1, 2385],
        [1, 1, 0, 12],
        [1, 1, 1, 9663],
    ],
    columns=["Z", "D", "Y", "counts"],
)
obs_data_df["probability"] = obs_data_df["counts"] / obs_data_df["counts"].sum()
obs_data_df = obs_data_df.drop(columns=["counts"])

balke_lb = -0.1946
balke_ub = 0.0054


Z = Variable("Z", domain=(0, 1))
D = Variable("D", domain=(0, 1))
Y = Variable("Y", domain=(0, 1))

pid = ConservativePID(obs_data_df)

ace = P(Y @ {D: 1} == 1) - P(Y @ {D: 0} == 1)
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

print(f"\n{'ACE':<15} | {'Balke (Vitamin A)':<20} | {'Ours (LP)':<20}")
print("-" * 65)

balke_str = f"[{balke_lb:.3f}, {balke_ub:.3f}]"
cpid_str = f"[{lb:.3f}, {ub:.3f}]"
print(f"{1} vs {0: <10} | {balke_str:<20} | {cpid_str:<20}")

results_df = pd.DataFrame(records)
save_path = Path(__file__).parent / "balke_4_1_vitaminA_results.csv"
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
save_path = Path(__file__).parent / "balke_4_1_vitaminA_treated_control_results.csv"
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

# %%
