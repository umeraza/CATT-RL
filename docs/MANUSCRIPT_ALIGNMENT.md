# Manuscript specification ledger

This ledger distinguishes directly specified choices from necessary engineering
completions. It prevents undocumented assumptions from becoming “paper code.”

| Topic | Manuscript statement | Repository realization |
| --- | --- | --- |
| State | `[t,n,f]`, default `t=64`; prices, technical factors, weights, cash | Dict observation; state variables appended to per-asset tokens in the model |
| Core factors | BOLL, CCI, RSI, TR, DMI, MACD, MFI; lookbacks 14/20/63/126 | Causal vectorized implementations and documented feature groups |
| Preprocessing | calendar align, forward fill, adjusted prices, 1/99 winsorize, daily cross-sectional z-score | Implemented; no backfill; original availability retained as tradability mask |
| Action | asset+cash softmax, temperature 1, blend 0.35, floor 1e-4, L1 cap 0.25 | Implemented and unit-tested on the simplex |
| Frictions | daily next-close execution, no short/leverage, 10 bps buy/sell cost | One-action/one-next-return convention documented in environment info |
| Reward | cost-aware portfolio-value delta | Value delta divided by initial value by default for stable scale; objective ordering is unchanged |
| CATT | `d=128`, 3 blocks, 4 heads, pre-attention InstanceNorm, LGU | Implemented with axial time/context attention; vanilla FF width is solved to keep the total parameter gap below 0.1% |
| PW-MLP | hidden sizes 64 then 32, GELU, dropout 0.1 | `128→64→32→128` to permit the stated residual connection |
| Memory | FIFO of last `I=8` hidden states, truncated gradients | Detached pooled context cache, exact cache saved per PPO transition |
| Actor | logits MLP and fan-in/residual conditioning | Shared per-asset scorer plus cash scorer; works with variable N |
| PPO distribution | Not specified | Diagonal logistic-normal over raw logits; softmax gives simplex actions |
| PPO | gamma .995, GAE .95, clip .2, value .5, entropy .001 | Implemented with advantage normalization and analytic Gaussian entropy |
| Optimizer | AdamW 3e-4→3e-5 cosine, decay 1e-4, grad clip 1 | Implemented; norm and bias parameters excluded from weight decay |
| Rollout | 128×32, 10 epochs, minibatch 2048 | Exact values in `base.yaml`; `sp500.yaml` explicitly reduces state batching because the stated 3,152-asset tensor cannot fit in 40 GB |
| Hardware | Python 3.10, PyTorch 2.1, CUDA 12, A100 40 GB, AMP | Conda/Docker metadata supplied; CPU path retained for tests |
| Metrics | net CumRet, annualized SR, SoR, Omega, turnover, MDD | Implemented with explicit annual-to-daily threshold conversion |
| CIs | 95% moving-block bootstrap, block 21 | Implemented; seed-controlled resampling |
| Ablations | backbone, memory, norm/gate, reward, feature groups | YAML inventories plus executable ablation CLI |
| DJIA dates | Conflicting 2022–2024 and 2023–2025 statements | Two named protocol files; table protocol is default |
| S&P universe | “S&P500” but 3,152 retained stocks | No silent reconciliation; user must identify snapshot/universe |
| Alpha158 | “95 indicators” from a 158-factor library; subset absent | Qlib bridge plus required explicit selected-column file for exact subset |
| Large-N attention | No scaling mechanism specified | Full mode for small N; explicit latent mode for thousands of assets |
| Seeds | Ten runs but seed values absent | Default reproducible list in examples; replace with original values if known |
| Training length | Max update count absent; early stopping stated | Configurable `max_updates=500`, validation each update, patience 10 |

## Accounting convention

The paper's equations index the price gain under old holdings and the trade cost
at the new close, while the prose describes an action at each observed state.
The environment uses the equivalent causal convention common in RL backtests:
the action selected from information through close `t` is rebalanced at close
`t`, incurs cost, and earns the `t→t+1` price relative. The next observation
contains the drifted post-return weights. Dates, pre/post weights, gross return,
cost, and net return are emitted in `info` and evaluation CSVs.

## Interpretation of reported tables

All CSVs under `results/paper_reported/` are transcriptions, not outputs of this
code. Some main-table baselines may use different periods or universes, so the
repository does not treat those rows as controlled reruns. New runs are written
only under `outputs/` with hashes and provenance.
