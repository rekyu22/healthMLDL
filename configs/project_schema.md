# Projet ML multimodal - schema

## Flux recommande
1. `data/raw`: depot brut des cohortes
2. `data/interim`: nettoyages, controles qualite, imputations initiales
3. `data/processed`: table finale modele
4. `scripts/run_multimodal_benchmark.py`: benchmark + CV + selection
5. `models/`: meilleur modele serialise
6. `reports/tables`: resultats comparatifs

## Blocs de modalites
- clinical: age, bmi, sex, physical_activity_score, inflammation_marker
- ultrasound: ultrasound_echo_intensity
- mri: mri_fat_fraction
- dexa: dexa_lean_mass_index
- microwave: microwave_phase_shift, microwave_attenuation

## Cible
- muscle_deterioration_score
