# Common tasks. Everything assumes `conda activate dl` (or any env with the deps installed) and MXDATA_ROOT pointing at the
# folder that holds SLIDE-*/ (default ~/Desktop/MedAI/Data/mxdata).
CONFIG ?= configs/dataset.yaml

.PHONY: install test lint report validate data dataset pretrain finetune compare notebooks clean

install:            ## editable install of the package + extras
	pip install -e ".[torch,baselines,train,kodiak,notebooks,dev]"

test:               ## unit + integration tests on synthetic slides (no real data, no GPU)
	pytest -q

lint:
	ruff check src tests

report:             ## every slide of the config: geometry, regions, QC, overlaps, validation summary
	mxdata report --config $(CONFIG)

validate:           ## full validation report (exit 1 on errors)
	mxdata validate --config $(CONFIG)

data:               ## the dataset: $MXDATA_ROOT/dataset/ — every marker, labeled + unlabeled patches of every slide in the config
	mxdata sample --config $(CONFIG)

dataset:            ## summarize the dataset
	mxdata dataset

pretrain:           ## KODIAK ViT-B/16 pretraining -> runs/kodiak_mxif_vitb/
	python -m kodiak.pretrain --config configs/kodiak_pretrain_vitb.yaml

finetune:           ## every supervised row, trained and evaluated on the test regions (scripts/finetune_all.sh --dry-run to preview)
	scripts/finetune_all.sh

compare:            ## the test-region table of every stored run
	mxseg compare --split test --unit region

notebooks:          ## execute the notebooks that need no trained model
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/00_dataset.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/00b_dataset_slide0236.ipynb

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
