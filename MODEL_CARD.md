# CATT-RL model card

## Intended use

CATT-RL is research code for studying long-horizon, transaction-aware portfolio
allocation with deep reinforcement learning. It is intended for offline
experiments on historical or synthetic data and for controlled architectural
ablations. It is not designed to place orders, handle brokerage credentials,
or provide individualized investment advice.

## Inputs and outputs

The policy consumes a causal daily tensor of adjusted-price/technical factors,
current asset-plus-cash weights, and a tradability mask. It emits a stochastic
logit vector mapped by softmax to nonnegative weights summing to one. The
environment further applies blending, an allocation floor, and an L1 turnover
cap before execution.

## Training

The actor–critic is optimized with PPO-Clip and generalized advantage estimation.
The default manuscript configuration uses gamma 0.995, GAE lambda 0.95, clip
0.20, AdamW, cosine learning-rate decay, and a cost-aware value-delta reward.
No pretrained weights or market data are bundled.

## Limitations and risks

- Backtests are sensitive to data revisions, delistings, corporate actions,
  universe construction, execution timing, and transaction-cost assumptions.
- A current constituent list used historically creates survivorship bias.
- The simulator omits impact, slippage, latency, partial fills, taxes, borrowing,
  sector/factor constraints, and time-varying cash returns.
- Attention diagnostics are descriptive, not causal explanations.
- Policy quality can deteriorate under regime shifts and distribution changes.
- The source manuscript leaves several protocol details ambiguous; they are
  listed in `docs/MANUSCRIPT_ALIGNMENT.md` rather than silently guessed.

Any practical financial use requires independent validation, licensed data,
stronger execution/risk controls, compliance review, and human oversight.

