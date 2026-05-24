from experiments.PoC.helpers import generate_canonical_tian_df
from cpid import CausalQuery, AtomicCounterfactual, OrderFunctionalLPSolver

domains = {"X": 2, "Y": 2}
seed = 15
df = generate_canonical_tian_df(domains=domains, seed=seed)

# We only need the observational component for this test
df = df.drop(columns=["Y_{X=0}", "Y_{X=1}"])

pns_query = CausalQuery(
    counterfactuals=[
        AtomicCounterfactual(target_var="Y", target_val=0, interventions={"X": 0}),
        AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1}),
    ]
)

print("----- PNS with Observational Data Only -----")
solver_obs = OrderFunctionalLPSolver(
    data=df,
    query=pns_query,
    solver_verbose=False,
)
lb_obs, ub_obs = solver_obs()
print(f"PNS bounds with observational data only: [{lb_obs:.4f}, {ub_obs:.4f}]")

print("\n----- PNS with Monotonicity (Y_{X=1} >= Y_{X=0}) -----")
solver_mono = OrderFunctionalLPSolver(
    data=df,
    query=pns_query,
    solver_verbose=False,
)
solver_mono.add_monotonicity_constraint(
    target_var="Y", 
    interventions_lower={"X": 0},
    interventions_upper={"X": 1}
)
lb_mono, ub_mono = solver_mono()
print(f"PNS bounds with monotonicity constraint: [{lb_mono:.4f}, {ub_mono:.4f}]")

print("\n----- PN with Observational Data Only -----")
pn_query = CausalQuery(
    counterfactuals=[
        AtomicCounterfactual(target_var="Y", target_val=1, interventions={"X": 1})
    ],
    evidence={"X": 0, "Y": 0},
)
solver_pn = OrderFunctionalLPSolver(
    data=df,
    query=pn_query,
    solver_verbose=False,
)
pn_lb, pn_ub = solver_pn()
print(f"PN bounds with observational data only: [{pn_lb:.4f}, {pn_ub:.4f}]")

print("\n----- PN with Monotonicity (Y_{X=1} >= Y_{X=0}) -----")
solver_pn_mono = OrderFunctionalLPSolver(
    data=df,
    query=pn_query,
    solver_verbose=False,
)
solver_pn_mono.add_monotonicity_constraint(
    target_var="Y", 
    interventions_lower={"X": 0},
    interventions_upper={"X": 1}
)
pn_lb_mono, pn_ub_mono = solver_pn_mono()
print(f"PN bounds with monotonicity constraint: [{pn_lb_mono:.4f}, {pn_ub_mono:.4f}]")

