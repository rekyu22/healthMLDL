# healthMLDL

Projet d'entrainement ML/DL pour l'etude multivariee de la deterioration musculaire.
Objectif: relier les mesures Ultrasound, MRI, DEXA et micro-ondes a un score de composition/deterioration tissulaire.

## Architecture claire du projet

```text
healthMLDL/
├── configs/
│   └── project_schema.md
├── data/
│   ├── raw/            # donnees brutes (jamais modifiees)
│   ├── interim/        # nettoyage et controle qualite
│   ├── processed/      # table finale d'entrainement
│   └── external/       # donnees externes telechargees
├── models/             # modeles entraines (.joblib)
├── notebooks/          # EDA et essais interactifs
├── reports/
│   ├── figures/        # heatmaps, graphs
│   └── tables/         # resultats benchmark, missingness
├── scripts/
│   ├── generate_synthetic_dataset.py
│   ├── download_nhanes_2017.py
│   ├── import_csv_dataset.py
│   ├── run_ml_baseline.py
│   ├── run_multimodal_benchmark.py
│   ├── train_from_dataset_id.py
│   ├── error_analysis_from_predictions.py
│   ├── compare_runs.py
│   └── train_dl_from_dataset_id.py
├── src/health_mldl/
│   ├── data/           # IO, nettoyage, split, validation
│   ├── features/       # schema de colonnes + feature engineering
│   ├── modeling/       # model zoo + stacking multimodal
│   ├── evaluation/     # metriques + cross-validation
│   ├── visualization/  # figures EDA
│   └── utils/          # serialisation
└── tests/
```

## Mini schema entree -> fusion -> prediction

```text
Entrees patient
├─ Bloc Clinical (age, bmi, sex, activity, inflammation) -> Modele 1
├─ Bloc Ultrasound (echo intensity)                      -> Modele 2
├─ Bloc MRI (fat fraction)                                -> Modele 3
├─ Bloc DEXA (lean mass index)                            -> Modele 4
└─ Bloc Microwave (phase, attenuation)                    -> Modele 5
                                                             |
                                                             v
                                              Meta-modele (stacking)
                                                             |
                                                             v
                                Sortie: muscle_deterioration_score
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commandes

1. Generer donnees synthetiques

```bash
PYTHONPATH=src python scripts/generate_synthetic_dataset.py
PYTHONPATH=src python scripts/export_separated_modalities_csv.py
```

2. Baseline simple (Random Forest)

```bash
PYTHONPATH=src python scripts/run_ml_baseline.py
```

3. Benchmark robuste (ElasticNet + RF + Gradient Boosting + Stacking multimodal + CV)

```bash
PYTHONPATH=src python scripts/run_multimodal_benchmark.py
```

4. Importer un vrai CSV

```bash
PYTHONPATH=src python scripts/import_csv_dataset.py /chemin/vers/tes_donnees.csv --name cohort_v1.csv
PYTHONPATH=src python scripts/export_separated_modalities_csv.py --input-csv data/raw/cohort_v1.csv --dataset-id cohort_v1
PYTHONPATH=src python scripts/build_training_table_from_modalities.py --dataset-id cohort_v1
```

5. Telecharger NHANES 2017-2018 (clinical + DEXA) et separer en modalites

```bash
PYTHONPATH=src python scripts/download_nhanes_2017.py
PYTHONPATH=src python scripts/export_separated_modalities_csv.py --input-csv data/raw/nhanes_2017_core_adults_dexa.csv --dataset-id nhanes_2017
```

6. Entrainer depuis un dataset-id de modalites

```bash
# Cas complet (avec target.csv): cohort_v1
PYTHONPATH=src python scripts/train_from_dataset_id.py --dataset-id cohort_v1 --stratify-age --save-predictions --save-feature-importance

# Cas NHANES (pas de target.csv): cible proxy via dexa_lean_mass_index
PYTHONPATH=src python scripts/train_from_dataset_id.py --dataset-id nhanes_2017 --target-col dexa_lean_mass_index --stratify-age --save-predictions --save-feature-importance
```

7. Analyser les erreurs par sous-groupes cliniques

```bash
PYTHONPATH=src python scripts/error_analysis_from_predictions.py --dataset-id cohort_v1 --target-col muscle_deterioration_score
```

8. Comparer automatiquement plusieurs runs

```bash
PYTHONPATH=src python scripts/compare_runs.py
```

9. Entrainer un modele DL tabulaire (PyTorch)

```bash
PYTHONPATH=src python scripts/train_dl_from_dataset_id.py --dataset-id cohort_v1
```

## Sorties principales

- `data/processed/training_table.csv`
- `data/raw/modalities/clinical.csv`
- `data/raw/modalities/ultrasound.csv`
- `data/raw/modalities/mri.csv`
- `data/raw/modalities/dexa.csv`
- `data/raw/modalities/microwave.csv`
- `data/raw/modalities/target.csv`
- `reports/tables/missingness_summary.csv`
- `reports/tables/benchmark_results.csv`
- `reports/tables/quality_report_<dataset>__<target>.json`
- `reports/tables/predictions_<dataset>__<target>.csv`
- `reports/tables/feature_importance_<dataset>__<target>.csv`
- `reports/tables/error_by_sex_<dataset>__<target>.csv`
- `reports/tables/error_by_age_bin_<dataset>__<target>.csv`
- `reports/tables/error_by_bmi_bin_<dataset>__<target>.csv`
- `reports/tables/run_comparison.csv`
- `reports/tables/model_comparison.csv`
- `reports/tables/dl_history_<dataset>__<target>.csv`
- `reports/tables/dl_predictions_<dataset>__<target>.csv`
- `reports/tables/dl_quality_report_<dataset>__<target>.json`
- `reports/dl_summary_<dataset>__<target>.json`
- `models/best_dl_model_<dataset>__<target>.pt`
- `models/dl_preprocessor_<dataset>__<target>.joblib`
- `reports/benchmark_summary.json`
- `models/best_model.joblib`

## Modeles utilises actuellement

- `ElasticNet`: baseline interpretable
- `RandomForest`: baseline non lineaire robuste
- `GradientBoosting`: boosting performant (version sklearn portable)
- `Multimodal Stacking`: fusion par modalite (clinical, US, MRI, DEXA, MW)
- `Tabular MLP (PyTorch)`: baseline DL tabulaire

Note: XGBoost/LightGBM/CatBoost peuvent etre ajoutes ensuite comme extensions.
