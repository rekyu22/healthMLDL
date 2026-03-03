import pandas as pd


def add_simple_interactions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if {"mri_fat_fraction", "dexa_lean_mass_index"}.issubset(data.columns):
        data["fat_to_lean_ratio"] = data["mri_fat_fraction"] / (
            data["dexa_lean_mass_index"].abs() + 1e-6
        )

    if {"microwave_phase_shift", "microwave_attenuation"}.issubset(data.columns):
        data["mw_composite_signal"] = (
            0.6 * data["microwave_phase_shift"]
            + 0.4 * data["microwave_attenuation"]
        )

    return data
