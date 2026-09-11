# Contributing

Contributions that improve correctness, point-in-time data handling, numerical
stability, or experiment reproducibility are welcome.

1. Create a branch from `main`.
2. Install the development environment with `python -m pip install -e '.[dev]'`.
3. Add or update tests for every behavioral change.
4. Run `ruff check src tests` and `pytest` before opening a pull request.
5. Keep generated data, model checkpoints, and licensed constituent histories
   out of Git. Small synthetic fixtures and paper-reported reference tables are
   acceptable.

Research-result changes must state the exact data snapshot, constituent file,
configuration hash, seed set, and whether values were produced by code or copied
from the manuscript. Never replace `results/paper_reported/` with newly generated
values; write generated results under `outputs/`.

