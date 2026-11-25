# Reproduce everything: make venv data train eval figures readme
PY ?= .venv/bin/python

.PHONY: venv data train eval figures readme all test lint

venv:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

data:            ## download BloodMNIST, DermaMNIST, OrganAMNIST (about 93 MB in total) and write dataset.json
	$(PY) scripts/prepare_data.py

train:           ## 3 seeds x 5 CNNs (about 6 minutes on an M3 Pro)
	$(PY) scripts/train.py

eval:            ## calibration, OOD, deferral, ensemble-size ablation (about 15 minutes)
	$(PY) scripts/run_eval.py

figures:
	$(PY) scripts/make_figures.py

readme:          ## fill the README tables from results/metrics
	$(PY) scripts/update_readme.py

all: data train eval figures readme

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
