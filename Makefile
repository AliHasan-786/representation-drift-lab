.PHONY: smoke test validate reproduce-local benchmark-local methods-local early-warning web-test web-build

PYTHON ?= python3

smoke:
	PYTHONPATH=src $(PYTHON) -m driftlab smoke --config configs/smoke.yaml --resume

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

validate:
	PYTHONPATH=src $(PYTHON) -m driftlab validate-artifact public/data/smoke.json

reproduce-local:
	PYTHONPATH=src .venv/bin/python -m driftlab reproduce-clip --config configs/reproduction-local.yaml --resume

benchmark-local:
	PYTHONPATH=src .venv/bin/python -m driftlab benchmark-clip --suite configs/reproduction-local-multiseed.yaml

methods-local:
	PYTHONPATH=src .venv/bin/python -m driftlab classification-suite --suite configs/linear-probe-local-multiseed.yaml
	PYTHONPATH=src .venv/bin/python -m driftlab classification-suite --suite configs/full-finetune-local-multiseed.yaml
	PYTHONPATH=src .venv/bin/python -m driftlab classification-suite --suite configs/lp-ft-local-multiseed.yaml
	PYTHONPATH=src .venv/bin/python -m driftlab classification-suite --suite configs/wise-source-local-multiseed.yaml
	PYTHONPATH=src .venv/bin/python -m driftlab wise-ft
	PYTHONPATH=src .venv/bin/python -m driftlab benchmark-clip --suite configs/distillation-local-multiseed.yaml
	PYTHONPATH=src .venv/bin/python -m driftlab benchmark-clip --suite configs/nullspace-local-multiseed.yaml
	PYTHONPATH=src .venv/bin/python -m driftlab benchmark-clip --suite configs/selective-lora-local-multiseed.yaml
	PYTHONPATH=src .venv/bin/python -m driftlab aggregate-methods
	PYTHONPATH=src .venv/bin/python -m driftlab interpolate-lora

early-warning:
	PYTHONPATH=src .venv/bin/python -m driftlab early-warning-demo

web-test:
	cd apps/web && npm test

web-build:
	cd apps/web && npm run build
