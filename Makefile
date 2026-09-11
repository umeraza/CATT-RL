.PHONY: install install-dev test lint smoke clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests

smoke:
	catt-rl synthetic --output data/processed/smoke.npz --assets 4 --days 220 --seed 7
	catt-rl train --config configs/smoke.yaml

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build', '.pytest_cache', '.ruff_cache', 'outputs/smoke']]"

