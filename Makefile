PYTHONPATH=src

.PHONY: setup data rebuild train benchmark all

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	PYTHONPATH=$(PYTHONPATH) python scripts/generate_synthetic_dataset.py
	PYTHONPATH=$(PYTHONPATH) python scripts/export_separated_modalities_csv.py

train:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_ml_baseline.py

benchmark:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_multimodal_benchmark.py

rebuild:
	PYTHONPATH=$(PYTHONPATH) python scripts/build_training_table_from_modalities.py

all: data benchmark
