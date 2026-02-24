import sys
from pathlib import Path
import pandas as pd
import time

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling.Sampling import BayesianModelSampling


repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.symbolic import Variable, P
from src.inference import ConservativePID


#  %%
model = DiscreteBayesianNetwork([("X", "Z"), ("Z", "Y")])

# CPD for X (Drug): P(X=1) = 0.5 (Assumption for joint gen)
cpd_x = TabularCPD(variable="X", variable_card=2, values=[[0.5], [0.5]])

# CPD for Z (Inflammation):
3
cpd_z = TabularCPD(
    variable="Z",
    variable_card=2,
    values=[[0.9, 0.9], [0.1, 0.1]],
    evidence=["X"],
    evidence_card=[2],
)

# CPD for Y (Recovery):
cpd_y = TabularCPD(
    variable="Y",
    variable_card=2,
    values=[[0.5, 0.5], [0.5, 0.5]],
    evidence=["Z"],
    evidence_card=[2],
)

model.add_cpds(cpd_x, cpd_z, cpd_y)

sampler = BayesianModelSampling(model)
obs_counts_df = sampler.forward_sample(size=10000, seed=42)

# %%

# Convert to probability distribution
obs_data_df = obs_counts_df.groupby(["X", "Z", "Y"]).size().reset_index(name="count")
obs_data_df["probability"] = obs_data_df["count"] / obs_data_df["count"].sum()
obs_data_df = obs_data_df.drop(columns=["count"])

obs_data_xy_df = obs_data_df.groupby(["X", "Y"])["probability"].sum().reset_index()

# %%
X = Variable("X", domain=[0, 1])
Y = Variable("Y", domain=[0, 1])
Z = Variable("Z", domain=[0, 1])

pid = ConservativePID(obs_data_df)
pns_event = (Y @ {X: 1} == 1) & (Y @ {X: 0} == 0)
pns_query = P(pns_event)

start_time = time.time()
lp_lb, lp_ub = pid.infer(pns_query, causal_order=[X, Z, Y])
duration = time.time() - start_time

print(f"PNS bounds: [{lp_lb:.4f}, {lp_ub:.4f}]")
print(f"Inference duration: {duration:.4f} seconds")

# %%
