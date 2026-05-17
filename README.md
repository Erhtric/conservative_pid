# Conservative PID

Conservative PID is a Python library for partial identification under minimal assumptions.

## Install

```bash
uv sync
```

## Notebook

See [notebooks/lib_exploration.ipynb](notebooks/lib_exploration.ipynb) for a walkthrough focused on instantiating the solver and exploring query bounds.

## Project Layout

- [cpid/io.py](cpid/io.py): counterfactual and query data structures (pseudo-symbolic formulation)
- [cpid/signature.py](cpid/signature.py): signatures and query compatibility methods
- [cpid/lp.py](cpid/lp.py): linear-programming solver