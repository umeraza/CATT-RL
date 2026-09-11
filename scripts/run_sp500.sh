#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -f data/universes/sp500_2022.csv ]]; then
  echo "Provide a permitted point-in-time data/universes/sp500_2022.csv first." >&2
  exit 2
fi

catt-rl download \
  --tickers data/universes/sp500_2022.csv \
  --start 2018-01-01 \
  --end 2023-01-01 \
  --output data/raw/sp500_2022.csv.gz

prepare_args=(
  --raw data/raw/sp500_2022.csv.gz
  --output data/processed/sp500_2022_full_alpha158.npz
  --availability 0.80
  --availability-start 2019-01-01
  --availability-end 2021-12-31
  --feature-set full
)
if [[ -n "${ALPHA158_EXPORT:-}" ]]; then
  prepare_args+=(--alpha158 "$ALPHA158_EXPORT")
fi
if [[ -n "${ALPHA158_COLUMNS:-}" ]]; then
  prepare_args+=(--alpha158-columns "$ALPHA158_COLUMNS")
fi
catt-rl prepare "${prepare_args[@]}"

catt-rl multiseed \
  --config configs/sp500.yaml \
  --seeds 7,17,27,37,47,57,67,77,87,97

