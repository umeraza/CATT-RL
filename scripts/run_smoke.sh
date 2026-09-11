#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

catt-rl synthetic \
  --output data/processed/smoke.npz \
  --assets 4 \
  --days 220 \
  --seed 7
catt-rl train --config configs/smoke.yaml

