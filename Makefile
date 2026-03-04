PYTHONPATH=src

.PHONY: setup data nhanes rebuild train benchmark train_dataset analyze_errors compare_runs all

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	PYTHONPATH=$(PYTHONPATH) python scripts/generate_synthetic_dataset.py
	PYTHONPATH=$(PYTHONPATH) python scripts/export_separated_modalities_csv.py

nhanes:
	PYTHONPATH=$(PYTHONPATH) python scripts/download_nhanes_2017.py
	PYTHONPATH=$(PYTHONPATH) python scripts/export_separated_modalities_csv.py --input-csv data/raw/nhanes_2017_core_adults_dexa.csv --dataset-id nhanes_2017

train:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_ml_baseline.py

benchmark:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_multimodal_benchmark.py

train_dataset:
	PYTHONPATH=$(PYTHONPATH) python scripts/train_from_dataset_id.py --dataset-id cohort_v1 --stratify-age --save-predictions --save-feature-importance

analyze_errors:
	PYTHONPATH=$(PYTHONPATH) python scripts/error_analysis_from_predictions.py --dataset-id cohort_v1 --target-col muscle_deterioration_score

compare_runs:
	PYTHONPATH=$(PYTHONPATH) python scripts/compare_runs.py

rebuild:
	PYTHONPATH=$(PYTHONPATH) python scripts/build_training_table_from_modalities.py

all: data benchmark
