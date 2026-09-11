# CATT-RL

Official-style reference implementation for **“CATT-RL: Deep Reinforcement
Learning for Long-Horizon Portfolio Management with a Context-Aware Temporal
Transformer.”** The repository implements the model, the transaction-aware
portfolio environment, PPO/GAE training, rolling evaluation, all manuscript
metrics, and every reported ablation family.

> **Research-use notice.** This is a backtesting implementation, not investment
> advice and not a live-trading system. Yahoo Finance downloads are optional and
> governed by the data provider's terms. Exact historical index membership must
> be supplied as point-in-time snapshots for a defensible reproduction.

## What is implemented

- Causal OHLCV alignment, forward filling, 1/99 cross-sectional winsorization,
  and daily cross-sectional z-scoring.
- Price, return, momentum, Bollinger, CCI, RSI, true-range/ATR, DMI/ADX, MACD,
  MFI, volatility, and liquidity features, plus an optional Qlib Alpha158 bridge.
- Gymnasium portfolio MDP with cash, no shorting/leverage, daily rebalancing,
  proportional buy/sell costs, action blending, an allocation floor, and an
  L1 turnover cap.
- Context-Aware Temporal Transformer blocks with temporal attention,
  pre-attention InstanceNorm, LGU residual gating, a 128→64→32→128 PW-MLP,
  and a detached FIFO memory trajectory that augments attention keys/values.
- A scalable latent cross-asset attention option for large universes; full
  cross-asset attention remains the manuscript-faithful default for DJIA.
- Logistic-normal stochastic simplex policy, separate value head, PPO-Clip,
  GAE, AdamW, cosine decay, AMP, gradient clipping, validation early stopping,
  deterministic seed control, and configuration hashing.
- Net cumulative return, annualized Sharpe and Sortino, Omega, average L1
  turnover, maximum drawdown, and moving-block bootstrap confidence intervals.
- CATT/vanilla, memory-depth, normalization/gating, reward, and feature-group
  ablations, together with attention-weight export and heatmap generation.

## Repository map

```text
configs/                 experiment, protocol, and ablation YAML files
data/universes/          point-in-time DJIA examples; user-supplied S&P snapshots
docs/                    architecture, data, reproducibility, and paper alignment
results/paper_reported/  manuscript values (reference only, never generated)
scripts/                 end-to-end command examples
src/catt_rl/data/        download, features, panel I/O, splits, synthetic data, Qlib
src/catt_rl/envs/        portfolio accounting and action constraints
src/catt_rl/models/      CATT/LGU layers and actor–critic policy
src/catt_rl/rl/          rollout buffer and PPO trainer
src/catt_rl/             CLI, experiment orchestration, evaluation, metrics, plots
tests/                   causality, accounting, constraints, model, metrics, PPO
```

## Installation

The paper environment is Python 3.10, PyTorch 2.1, and CUDA 12 on one NVIDIA A100.

```bash
conda env create -f environment.yml
conda activate catt-rl
python -m pip install -e '.[dev]'
```

CPU-only development is supported through a normal pip installation. Qlib is
optional:

```bash
python -m pip install -e '.[qlib]'
```

## One-minute smoke run

```bash
catt-rl synthetic --output data/processed/smoke.npz --assets 4 --days 220 --seed 7
catt-rl train --config configs/smoke.yaml
pytest
```

The smoke configuration is intentionally tiny and is only a software check; it
cannot reproduce the paper's financial results.

## Real-data workflow

1. Put a frozen constituent file with a `ticker` column under `data/universes/`.
2. Download or ingest daily OHLCV.
3. Prepare a leakage-safe panel.
4. Train and evaluate one split or run all configured seeds/windows.

```bash
catt-rl download \
  --tickers data/universes/djia_2022.csv \
  --start 2018-01-01 --end 2023-01-01 \
  --output data/raw/djia_2022.csv.gz

catt-rl prepare \
  --raw data/raw/djia_2022.csv.gz \
  --output data/processed/djia_2022_full.npz \
  --availability 0.80 --feature-set full

catt-rl train --config configs/djia.yaml \
  --set data.panel_path=data/processed/djia_2022_full.npz \
  --set data.train_start=2019-01-01 \
  --set data.train_end=2021-12-31 \
  --set data.test_start=2022-01-01 \
  --set data.test_end=2022-12-31
```

Evaluate a checkpoint deterministically:

```bash
catt-rl evaluate --config outputs/djia/seed_7/resolved_config.yaml \
  --checkpoint outputs/djia/seed_7/best.pt --split test
```

Run the ablation matrix or ten manuscript seeds:

```bash
catt-rl ablate --config configs/djia.yaml --group all --seeds 7,17,27,37,47,57,67,77,87,97
catt-rl multiseed --config configs/djia.yaml --seeds 7,17,27,37,47,57,67,77,87,97
```

## Output contract

Every training directory contains:

- `resolved_config.yaml` and `run_manifest.json` (configuration/data hashes);
- `train_metrics.jsonl`, `best.pt`, and `last.pt`;
- `validation_metrics.json` and, when requested, `test_metrics.json`;
- deterministic `daily_returns.csv`, `portfolio_values.csv`, `weights.csv`,
  `trades.csv`, and optional `attention.npz`.

Generated outputs are deliberately separated from `results/paper_reported/`.
The latter contains values transcribed from the manuscript so that claimed and
newly reproduced results cannot be confused.

## Important manuscript alignment notes

The source manuscript contains mutually inconsistent DJIA dates (the dataset
table says 2022–2024 tests; the prose/implementation section also states
2023–2025), describes an “S&P500” universe of 3,152 retained securities, and
does not specify the stochastic distribution required to evaluate PPO action
log-probabilities. This repository does not hide those gaps:

- both DJIA date protocols are supplied in `configs/protocols/`;
- S&P500 requires a user-provided point-in-time constituent snapshot, while a
  broad-US-universe experiment should be named as such;
- the policy uses a documented logistic-normal distribution whose softmax mean
  exactly produces asset-plus-cash simplex weights;
- large-universe latent attention is explicit and configurable rather than
  being silently substituted for quadratic full attention.

See `docs/MANUSCRIPT_ALIGNMENT.md` for the complete specification ledger.

## Reproducibility boundary

Code can reproduce the stated procedure, but exact paper numbers additionally
require the authors' original price snapshot, corporate-action adjustments,
point-in-time constituent files, the exact 95-factor Alpha158 subset, and the
actual seed list. The repository records all of these inputs when supplied and
fails loudly on malformed panels. It never fabricates missing market data or
copies manuscript values into generated output folders.

## Citation

Update the repository URL in `CITATION.cff` after publishing, then cite the paper
and software release. The implementation is released under the MIT License.

