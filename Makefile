.PHONY: setup check-base repos fetch fetch-status check encode encode-status train train-flatten eval mic all

setup:
	pip install -r requirements.txt

# run BEFORE anything else: is Lumma fit for English ASR? (IN/OUT report card, ~10min CPU)
check-base:
	python scripts/07_base_model_check.py --out base_check_results.json

repos:
	python scripts/00_create_repos.py

fetch:
	python scripts/01_fetch_data.py

fetch-status:
	python scripts/01_fetch_data.py --status

# gate: prints both ledgers, non-zero exit if below min_hours
check:
	python scripts/05_check_data.py

encode:
	python scripts/02_encode_data.py

encode-status:
	python scripts/02_encode_data.py --status

# primary Phase-1 run (per_frame_sum c0-c7)
train:
	python scripts/03_train.py

# A/B variant (flatten c0-c3) — lower batch size, it makes 4x longer sequences
train-flatten:
	python scripts/03_train.py --frontend flatten --codebooks 4 --bs 8

eval:
	python scripts/04_evaluate.py --model-dir $(MODEL) --push

mic:
	python scripts/06_mic_stream.py --model-dir $(MODEL)

# full Phase-1 pipeline on one box
all: repos fetch check encode train
