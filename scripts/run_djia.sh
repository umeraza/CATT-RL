#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# End dates are exclusive for the downloader. The extra pre-window history is
# required by 126-day factors and the 64-day model lookback.
catt-rl download \
  --tickers data/universes/djia_2022.csv \
  --start 2018-01-01 \
  --end 2023-01-01 \
  --output data/raw/djia_2022.csv.gz

catt-rl prepare \
  --raw data/raw/djia_2022.csv.gz \
  --output data/processed/djia_2022_full.npz \
  --availability 0.80 \
  --availability-start 2019-01-01 \
  --availability-end 2021-12-31 \
  --feature-set full

catt-rl multiseed \
  --config configs/djia.yaml \
  --seeds 7,17,27,37,47,57,67,77,87,97

