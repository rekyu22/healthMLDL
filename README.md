# healthMLDL

Projet d'entrainement pour l'analyse multivariee de la deterioration musculaire.

## Arborescence claire

```text
healthMLDL/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── artifacts/
│   ├── ml/            # modeles ML entraines
│   └── dl/            # modeles DL entraines
├── reports/
│   ├── ml/
│   │   ├── figures/
│   │   └── tables/
│   └── dl/
│       └── tables/
├── tasks/
│   ├── ingest/        # import / download / split data
│   ├── classic_ml/    # entrainement + analyse ML
│   └── deep_learning/ # entrainement DL
└── src/health_mldl/
    ├── data/
    ├── features/
    ├── evaluation/
    ├── ml_core/
    ├── dl_core/
    └── utils/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ingestion (data)

```bash
PYTHONPATH=src python tasks/ingest/generate_synthetic_dataset.py
PYTHONPATH=src python tasks/ingest/export_separated_modalities_csv.py

PYTHONPATH=src python tasks/ingest/download_nhanes_2017.py
PYTHONPATH=src python tasks/ingest/export_separated_modalities_csv.py --input-csv data/raw/nhanes_2017_core_adults_dexa.csv --dataset-id nhanes_2017

PYTHONPATH=src python tasks/ingest/import_csv_dataset.py /chemin/vers/tes_donnees.csv --name cohort_v1.csv
PYTHONPATH=src python tasks/ingest/build_training_table_from_modalities.py --dataset-id cohort_v1
```

## ML (classic ML)

```bash
PYTHONPATH=src python tasks/classic_ml/run_ml_baseline.py
PYTHONPATH=src python tasks/classic_ml/run_multimodal_benchmark.py
PYTHONPATH=src python tasks/classic_ml/train_from_dataset_id.py --dataset-id cohort_v1 --stratify-age --save-predictions --save-feature-importance
PYTHONPATH=src python tasks/classic_ml/error_analysis_from_predictions.py --dataset-id cohort_v1 --target-col muscle_deterioration_score
PYTHONPATH=src python tasks/classic_ml/compare_runs.py
```

## DL (deep learning)

```bash
PYTHONPATH=src python tasks/deep_learning/train_dl_from_dataset_id.py --dataset-id cohort_v1
```

## Sorties

- ML artifacts: `artifacts/ml/`
- DL artifacts: `artifacts/dl/`
- ML reports: `reports/ml/`
- DL reports: `reports/dl/`
