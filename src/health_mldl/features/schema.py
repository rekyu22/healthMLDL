"""Feature schema and modality mapping for multimodal training."""

PATIENT_ID_COL = "patient_id"
TARGET_COL = "muscle_deterioration_score"

CLINICAL_COLS = [
    "age",
    "bmi",
    "sex",
    "physical_activity_score",
    "inflammation_marker",
]

ULTRASOUND_COLS = [
    "ultrasound_echo_intensity",
]

MRI_COLS = [
    "mri_fat_fraction",
]

DEXA_COLS = [
    "dexa_lean_mass_index",
]

MICROWAVE_COLS = [
    "microwave_phase_shift",
    "microwave_attenuation",
]

ENGINEERED_COLS = [
    "fat_to_lean_ratio",
    "mw_composite_signal",
]

MODALITY_BLOCKS = {
    "clinical": CLINICAL_COLS,
    "ultrasound": ULTRASOUND_COLS,
    "mri": MRI_COLS,
    "dexa": DEXA_COLS,
    "microwave": MICROWAVE_COLS,
    "engineered": ENGINEERED_COLS,
}

BASE_REQUIRED_COLUMNS = sorted(
    set(
        [PATIENT_ID_COL]
        + CLINICAL_COLS
        + ULTRASOUND_COLS
        + MRI_COLS
        + DEXA_COLS
        + MICROWAVE_COLS
        + [TARGET_COL]
    )
)
