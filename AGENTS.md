# AGENTS.md — Black Hole Machine Learning

Inferring the parameters of a Kerr-Newman-NUT black hole given its shadow (form of the black sppot in the sky) with a fast vectorized Python forward model, a Bayesian baselinee (MCMC) and an amortized neural-network estimator.

## Stack and Structure
- Virtual environment created with uv
- See `pyproject.toml` for Python version and dependencies
- Structure:
```
blackhole-ml/
├── src/knn_shadow/
│   ├── metric.py           # KNN parameters, validity conditions (horizon, ranges)
│   ├── shadow.py           # photon region, shadow boundary, projection
│   ├── geodesics.py        # small numerical integrator (validation only)
│   ├── noise.py            # noise models
│   └── data_generation.py  # dataset generation, SQLite helpers
├── inference/
│   ├── mcmc.py             # emcee-based Bayesian baseline
│   ├── model.py            # PyTorch network
│   └── train.py
├── notebooks/              # analysis and figures
├── tests/                  # pytest
├── docs/                   # physical set up and math notes (equations to be used)
├── pyproject.toml
├── Dockerfile
├── .github/workflows/ci.yml
├── opencode/                   # .md-files used to work using SDD on Opoencode
│   ├── docs/constitution.md
└── └── specs/             
```

## Commands
- The .py-scripts to be run depend on which step we are in
- `uv sync`
- Testing: `uv run pytest`
- Linting: `uv run check .`

## Conventions
- Code like an expert in datascience, machine learning and astrophysics
- Comments should be short, but precise. Maximally one comment at the beginning of a function and/or code paragraph (and only if needed). Just a short description of what it does. Do not write to which spec, FR, NFR or task it is associated.
- Naming conventions: Camelcase for functions, Pascalcase for classes, Hungarian notation for variables
- When a function is defined, its argument's type shall be specified, and its output type as well (with ->) (Example: f(x: int) -> int: )
- If a function will be used several times (at least three), it shall be compiled using the decorators defined in `numba`, like @jit or @njit. If there are other cases where `numba` can be helpful consider it.
- Ruff will be used for linting, so write code in such a way, that it respects the given rules in `pyproject.toml`, so that ruff does not have to change a lot of things
- Everything should be in English

## Domain Rules / Known Pitfalls
- If there is a loss of understanding about the phyiscs, read again `docs/` to get the technical background.
- Never read the file `PROJECT_DESCRIPTION.md` unless it is stated explicitly. Man things inside them are not updated or even wrong.

## Workflow
- Before touching or creating any file, always plan and wait for the user's approval
- Read `docs/constitution.md` and the active spec (`specs/NNN-*/`) before touching code

## Boundaries
- **Always:**Ask in case of ambiguities and in case there is a potentially better plan, implementation, algorithm,...
- **Ask first:** New dependencies, new files, changes to data formats, etc.
- **Never:** Touch configuration files (.gitignore, pyproject.toml, .github)

## Verification
- First: `uv run pytest` to see if every test is bypassed
- Second: reread the expected goal and verify every point is fulfilled

## Memory
- At the beginning, read `MEMORY.md` to understand the project's current state and decisions made
- At the end of a task, update it with the current state, important decisions (and the reasons behind them), and mistakes to avoid
- Keep it brief (maximum 50-70 lines): summarize or remove anything that is no longer useful
- If something becomes a permanent rule, propose moving it to `AGENTS.md` instead of leaving it in memory
- Never store sensitive data (keys, tokens, personal information)

## Rules
- Read `opencode/docs/constitution.md` and the active spec (`opencode/specs/NNN-*/`) before touching code
