#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

catt-rl ablate \
  --config configs/djia.yaml \
  --group all \
  --seeds 7,17,27,37,47,57,67,77,87,97

