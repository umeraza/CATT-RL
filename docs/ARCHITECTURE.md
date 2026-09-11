# Architecture and tensor contract

## State

At decision date `t`, the environment returns a causal market window
`market ∈ R[lookback, assets, features]`, current drifted portfolio weights
`weights ∈ Δ^(assets+cash)`, and a tradability mask. The model optionally appends
the risky weight and cash fraction to every asset token. All window rows are at
or before `t`; the environment applies the selected target to the `t→t+1`
transition.

## CATT encoder

1. A linear layer maps each feature vector to `d_model=128`.
2. Sinusoidal time codes provide within-window order. They can be disabled to
   test the manuscript's memory-only ordering interpretation.
3. Every CATT block applies pre-attention InstanceNorm and multi-head temporal
   attention independently per asset.
4. The latest representation of each asset enters cross-asset context attention.
   In `full` mode, asset queries attend to the current asset set plus cached
   memory. In `latent` mode, learned context tokens compress and redistribute
   cross-sectional context in O(NM), where M is `context_latents`.
5. The attention update and residual enter the Lightweight Gating Unit (LGU):
   `z=sigmoid(Wz*y+Uz*x-bz)`, `h=sigmoid(Wh*y+Uh*(z*x))`, and
   `g=(1-z)*x+z*h`.
6. A GELU PW-MLP follows the paper's `128→64→32` bottleneck and projects back to
   128 for a residual update.

The vanilla ablation replaces the CATT stack with parameter-aligned temporal and
cross-asset Transformer operations using LayerNorm and no LGU/memory.

## Memory trajectory

The encoder pools its final asset states to one context vector per environment.
That vector is appended to a FIFO cache `memory ∈ R[batch,I,d_model]`. During
rollout, the cache is detached after every transition; during PPO replay, the
exact pre-action cache and validity mask are restored from the rollout buffer.
This realizes the paper's “gradients truncated outside the memory window” rule
without accidentally evaluating actions under a different recurrent state.

## Policy and value heads

The actor applies a shared `128→128→1` asset scorer and a separate cash scorer.
Previous weights supply residual conditioning, and scores are divided by
`sqrt(d_model)` (fan-in scaling). A diagonal logistic-normal policy samples raw
logits; softmax with temperature 1 maps them to the asset-plus-cash simplex.
The critic pools asset states and applies an independent `128→128→1` head.

The stochastic distribution is a necessary implementation completion because
PPO requires action log-probabilities but the manuscript only specifies a
deterministic softmax mapping. Both raw logits and executed weights are stored,
so PPO ratios remain mathematically well-defined after environment constraints.

The rollout buffer stores date indices rather than copying overlapping market
windows. Every PPO minibatch reconstructs its causal windows from the immutable
panel while preserving the exact portfolio weights, action, and pre-action
memory. This is both lossless and substantially more memory-efficient.

## Action execution

The environment applies, in order: softmax output → allocation floor and
renormalization → blend with current weights (`η=0.35`) → L1 cap (`0.25`) →
tradability projection → renormalization. Costs are charged on risky-asset
turnover at 10 bps. Cash earns zero. No shorting or leverage is permitted.
