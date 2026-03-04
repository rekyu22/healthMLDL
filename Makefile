PYTHONPATH=src

.PHONY: setup \
	data_synth data_import data_nhanes data_rebuild \
	ml_baseline ml_benchmark ml_train ml_error ml_compare \
	dl_train all

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data_synth:
	PYTHONPATH=$(PYTHONPATH) python tasks/ingest/generate_synthetic_dataset.py
	PYTHONPATH=$(PYTHONPATH) python tasks/ingest/export_separated_modalities_csv.py

data_nhanes:
	PYTHONPATH=$(PYTHONPATH) python tasks/ingest/download_nhanes_2017.py
	PYTHONPATH=$(PYTHONPATH) python tasks/ingest/export_separated_modalities_csv.py --input-csv data/raw/nhanes_2017_core_adults_dexa.csv --dataset-id nhanes_2017

data_import:
	@echo "Usage: PYTHONPATH=src python tasks/ingest/import_csv_dataset.py /path/to/file.csv --name cohort_v1.csv"

data_rebuild:
	PYTHONPATH=$(PYTHONPATH) python tasks/ingest/build_training_table_from_modalities.py --dataset-id cohort_v1

ml_baseline:
	PYTHONPATH=$(PYTHONPATH) python tasks/classic_ml/run_ml_baseline.py

ml_benchmark:
	PYTHONPATH=$(PYTHONPATH) python tasks/classic_ml/run_multimodal_benchmark.py

ml_train:
	PYTHONPATH=$(PYTHONPATH) python tasks/classic_ml/train_from_dataset_id.py --dataset-id cohort_v1 --stratify-age --save-predictions --save-feature-importance

ml_error:
	PYTHONPATH=$(PYTHONPATH) python tasks/classic_ml/error_analysis_from_predictions.py --dataset-id cohort_v1 --target-col muscle_deterioration_score

ml_compare:
	PYTHONPATH=$(PYTHONPATH) python tasks/classic_ml/compare_runs.py

dl_train:
	PYTHONPATH=$(PYTHONPATH) python tasks/deep_learning/train_dl_from_dataset_id.py --dataset-id cohort_v1

all: data_synth ml_benchmark
