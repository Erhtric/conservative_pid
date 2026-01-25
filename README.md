# Conservative PID

Conservative PID is a Python library for causal inference, providing tools to define symbolic causal models, generate canonical bases (deterministic worlds), and compute bounds on causal queries using Linear Programming.

## Features

- **Symbolic Causal Language**: Define variables, interventions (`@`), and counterfactual terms naturally.
- **Canonical Basis Generation**: transform variables and domains into a set of all possible deterministic worlds.
  - Includes a highly optimized, vectorized implementation using NumPy.
- **Solver**: Compute strict Lower and Upper bounds for causal queries (Standard and Conditional) given observational data.

## Getting Started

### Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

### Running Tests

```bash
uv run python -m pytest tests/
```

## Documentation

This project uses `mkdocs` for documentation. To view the documentation locally:

1.  **Install dependencies** (if not done):
    ```bash
    uv sync
    ```

2.  **Serve the documentation**:
    ```bash
    uv run mkdocs serve
    ```
    Open your browser at `http://127.0.0.1:8000` to view the site.

3.  **Build static site**:
    ```bash
    uv run mkdocs build
    ```
    The static HTML files will be generated in the `site/` directory.
