from experiments.PoC.helpers import (
    generate_tian_df,
    tian_pns_bounds,
)
from time import time
import pandas as pd
from pathlib import Path
import plotly.express as px

# Stress test of Tian PNS bounds computation with increasing domain sizes
# Both variables will take the same domain size.
# Expected growth is exponential.

min_domain_size = 2
max_domain_size = 35
timings = []

for domain_size in range(min_domain_size, max_domain_size + 1):
    print(f"Domain size: {domain_size}")
    domains = {"X": domain_size, "Y": domain_size}
    seed = 15
    df = generate_tian_df(domains=domains, seed=seed)

    pns_start_time = time()
    pns_lb, pns_ub = tian_pns_bounds(df)
    pns_end_time = time()

    timings.append(
        {
            "domain_size": domain_size,
            "pns_time": pns_end_time - pns_start_time,
        }
    )

curr_dir = Path(__file__).parent
timings_df = pd.DataFrame(timings)
timings_df.to_csv(curr_dir / "timings.csv", index=False)

fig = px.line(
    timings_df,
    x="domain_size",
    y="pns_time",
    title="Tian PNS Bounds Computation Time vs Domain Size",
    labels={"domain_size": "Domain Size", "pns_time": "Computation Time (seconds)"},
)
fig.write_image(curr_dir / "timings.png")
