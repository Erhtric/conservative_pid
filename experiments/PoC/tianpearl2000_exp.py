from cpid.io import MonotonicityConstraint
from experiments.PoC.helpers import (
    # generate_canonical_tian_df,
    generate_monotonic_tian_df,
)

from cpid import CausalQuery, AtomicCounterfactual, OrderFunctionalLPSolver


# PoC Bounds with additional experimental constraints on Tian's example
domains = {"X": 2, "Y": 2}
seed = 15
# df = generate_tian_df(domains=domains, seed=seed)
# df = generate_canonical_tian_df(domains=domains, seed=seed)
df = generate_monotonic_tian_df(domains=domains, seed=seed)

constraint_1 = {
    "target_expr": CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=0, interventions={"X": 0})
        ]
    ),
    "value": df["Y_{X=0}"].value_counts(normalize=True).values[0],
}

constraint_2 = {
    "target_expr": CausalQuery(
        counterfactuals=[
            AtomicCounterfactual(target_var="Y", target_val=0, interventions={"X": 1})
        ]
    ),
    "value": df["Y_{X=1}"].value_counts(normalize=True).values[0],
}

mono_constraint = MonotonicityConstraint(
    target_var="Y",
    interventions_lower={"X": 0},
    interventions_upper={"X": 1},
)

print(f"Experimental constraint 1: P(Y_{{X=0}}=0) = {constraint_1['value']}")
print(f"Experimental constraint 2: P(Y_{{X=1}}=0) = {constraint_2['value']}")

df = df.drop(columns=["Y_{X=0}", "Y_{X=1}"])

# PNS Query
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

lb_obs, ub_obs = solver()
print(f"PNS bounds with observational data only: [{lb_obs}, {ub_obs}]\n")

solver.add_experimental_constraint(**constraint_1)
solver.add_experimental_constraint(**constraint_2)

# Note: adding constraints will increase the number of equivalence classes that
# have non-empty support.
pns_lb, pns_ub = solver()
print(f"PNS bounds with experimental constraints: [{pns_lb}, {pns_ub}]\n")

solver.add_monotonicity_constraint(**mono_constraint.__dict__)
pns_lb_mono, pns_ub_mono = solver()
print(
    f"PNS bounds with monotonicity and experimental constraints: [{pns_lb_mono}, {pns_ub_mono}]\n"
)

# PN Query
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
print(f"PN bounds with observational data only: [{pn_lb}, {pn_ub}]\n")

solver.add_experimental_constraint(**constraint_1)
solver.add_experimental_constraint(**constraint_2)

pn_lb_exp, pn_ub_exp = solver()
print(f"PN bounds with experimental constraints: [{pn_lb_exp}, {pn_ub_exp}]\n")

solver.add_monotonicity_constraint(**mono_constraint.__dict__)
pn_lb_mono, pn_ub_mono = solver()
print(
    f"PN bounds with monotonicity and experimental constraints: [{pn_lb_mono}, {pn_ub_mono}]\n"
)

# PS Query
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
print(f"PS bounds with observational data only: [{ps_lb}, {ps_ub}]\n")

solver.add_experimental_constraint(**constraint_1)
solver.add_experimental_constraint(**constraint_2)

ps_lb_exp, ps_ub_exp = solver()
print(f"PS bounds with experimental constraints: [{ps_lb_exp}, {ps_ub_exp}]\n")

solver.add_monotonicity_constraint(**mono_constraint.__dict__)
ps_lb_mono, ps_ub_mono = solver()
print(
    f"PS bounds with monotonicity and experimental constraints: [{ps_lb_mono}, {ps_ub_mono}]\n"
)
