# Reproducibility checklist

Before claiming a reproduced result, record all of the following:

- raw-data source, retrieval date, terms, file hash, adjustment convention;
- point-in-time constituent snapshot and its effective date;
- exact Alpha158 column subset, if used;
- train, validation, and test boundaries;
- resolved YAML and configuration SHA-256;
- package versions, Python/PyTorch/CUDA versions, GPU model;
- every seed and deterministic-kernel warnings;
- checkpoint hash and evaluation output hash;
- transaction cost, allocation constraints, execution timing, and cash return;
- whether metrics aggregate windows by daily-return concatenation or by averaging.

## Paper-scale run

```bash
conda env create -f environment.yml
conda activate catt-rl
python -m pip install -e '.[dev,qlib]'

# Prepare one point-in-time panel per universe/window.
catt-rl download --tickers data/universes/djia_2022.csv \
  --start 2018-01-01 --end 2023-01-01 --output data/raw/djia_2022.csv.gz
catt-rl prepare --raw data/raw/djia_2022.csv.gz \
  --output data/processed/djia_2022_full.npz --feature-set full

# Ten independent runs. The seed list is explicit because the paper omits it.
catt-rl multiseed --config configs/djia.yaml \
  --seeds 7,17,27,37,47,57,67,77,87,97
```

For all three rolling windows, prepare each panel with its own universe snapshot,
then use `catt-rl protocol`. A window manifest can override `panel_path` as well
as dates. Never prepare one survivorship-biased present-day panel and relabel it
as point-in-time.

## Expected nondeterminism

CUDA reductions and some attention kernels may remain nondeterministic even when
deterministic mode is requested. The run manifest records PyTorch deterministic
flags. Report means over the ten independent seeds; do not choose the best seed.

