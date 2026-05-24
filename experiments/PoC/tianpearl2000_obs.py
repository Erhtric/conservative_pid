from experiments.PoC.helpers import (
    generate_tian_df,
    tian_pn_bounds,
    tian_pns_bounds,
    tian_ps_bounds,
)

# Classical Observational Bounds
domains = {"X": 2, "Y": 2}
seed = 15
df = generate_tian_df(domains=domains, seed=seed)
pns_lb, pns_ub = tian_pns_bounds(df)
print(f"Tian PNS bounds: [{pns_lb}, {pns_ub}]\n")
pn_lb, pn_ub = tian_pn_bounds(df)
print(f"Tian PN bounds: [{pn_lb}, {pn_ub}]\n")
ps_lb, ps_ub = tian_ps_bounds(df)
print(f"Tian PS bounds: [{ps_lb}, {ps_ub}]\n")
