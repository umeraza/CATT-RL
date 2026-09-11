# Data, features, and temporal splits

## Data source boundary

The repository accepts any permitted long-form daily OHLCV source. The optional
Yahoo Finance helper exists for research convenience but the downloader does not
make the resulting data redistributable. Preserve the raw snapshot and its hash.
For publication-quality experiments, prefer a licensed source with corporate
actions and point-in-time constituent histories.

## Leakage controls

- Universe membership is frozen from the supplied snapshot for each window.
- Each asset is reindexed to the union of observed trading dates and only
  forward-filled; back-filling is prohibited.
- Rolling features use current/past observations only.
- Winsorization and normalization are cross-sectional at each date, so no future
  sample statistics enter a feature.
- The next adjusted close is accessed only inside `env.step` after the action.
- Validation and test boundaries are checked against the prepared panel.

The feature causality test perturbs future raw rows and asserts that earlier
prepared factors remain byte-equivalent within floating-point tolerance.

## Feature groups

| Group | Included information |
| --- | --- |
| `prices` | adjusted-close level/relative level, simple and log returns |
| `momentum` | prices plus RSI, MACD, and 20/63/126-day momentum |
| `volatility_range` | momentum plus Bollinger, TR/ATR, DMI/ADX, ranges, volatilities |
| `full` | all above plus CCI, MFI, volume z-score, and dollar-volume liquidity |

Portfolio weights and cash are separate switches, matching the feature ablation.
Qlib Alpha158 can be exported with `catt-rl qlib-alpha158` and merged during
preparation. The exact 95-factor subset mentioned by the paper is not listed in
the manuscript; pass a text file of selected columns to make that choice explicit.

## Conflicting DJIA protocols

`configs/protocols/djia_table_2022_2024.yaml` exactly follows the manuscript's
dataset table: train 2019–2021/test 2022, train 2020–2022/test 2023, and train
2021–2023/test 2024. `djia_impl_2023_2025.yaml` follows the separate Implementation
Details statement. Results from these protocols must not be averaged together.

## S&P500 naming

The S&P 500 contains about 500 leading companies, whereas the manuscript states
that 3,152 securities survive cleaning. The latter is a broad U.S. equity universe,
not an S&P500 constituent set. The code supports either, but the experiment name
and universe snapshot should report which one was actually used.

