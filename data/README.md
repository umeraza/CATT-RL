# Data directory

Market data and most constituent histories are intentionally not distributed.
This directory keeps only small universe examples and documentation.

Expected raw long-form schema:

```text
date,ticker,open,high,low,close,adj_close,volume
```

`catt-rl prepare` converts that schema into a compressed panel (`.npz`) with:

- `dates`: trading dates;
- `tickers`: frozen asset order;
- `prices`: adjusted closes `[date, asset]` used for accounting;
- `features`: causal normalized factors `[date, asset, feature]`;
- `tradable`: observed-price mask `[date, asset]`;
- `feature_names`: ordered feature names.

The adjacent JSON manifest records preprocessing options and a SHA-256 digest.
Prepared panels, downloaded prices, and checkpoints are ignored by Git.

## Point-in-time membership

The included DJIA files are annual start-of-year snapshots intended to make the
workflow concrete. Confirm them against the licensed/source dataset used for a
submission. Do not fetch today's constituents and reuse them historically: that
introduces survivorship and look-ahead bias. The S&P 500 list is not redistributed;
provide a licensed or otherwise permitted point-in-time snapshot with a `ticker`
column.

