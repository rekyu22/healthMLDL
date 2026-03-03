import numpy as np
import pandas as pd

from health_mldl.config import RAW_DATA_DIR


def main(n_samples: int = 1200, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    patient_id = [f"P{idx:05d}" for idx in range(1, n_samples + 1)]

    age = rng.normal(62, 11, n_samples).clip(25, 90)
    bmi = rng.normal(27, 4.5, n_samples).clip(17, 45)
    sex = rng.choice(["F", "M"], size=n_samples)
    activity = rng.normal(55, 18, n_samples).clip(0, 100)
    inflammation = rng.normal(2.5, 1.0, n_samples).clip(0.2, 8.0)

    mri_fat_fraction = (
        8
        + 0.22 * (age - 40)
        + 0.35 * (bmi - 24)
        + 0.8 * inflammation
        - 0.08 * activity
        + rng.normal(0, 2.2, n_samples)
    ).clip(1, 70)

    dexa_lean_mass_index = (
        18
        - 0.06 * (age - 40)
        - 0.15 * (bmi - 24)
        + 0.07 * activity
        - 0.4 * inflammation
        + rng.normal(0, 1.1, n_samples)
    ).clip(6, 30)

    ultrasound_echo_intensity = (
        30
        + 0.9 * mri_fat_fraction
        - 0.5 * dexa_lean_mass_index
        + rng.normal(0, 6, n_samples)
    ).clip(5, 130)

    microwave_phase_shift = (
        0.4 * mri_fat_fraction
        - 0.15 * dexa_lean_mass_index
        + 0.2 * inflammation
        + rng.normal(0, 1.7, n_samples)
    )

    microwave_attenuation = (
        0.25 * mri_fat_fraction
        + 0.12 * bmi
        + 0.35 * inflammation
        + rng.normal(0, 1.5, n_samples)
    )

    sex_effect = np.where(sex == "M", -1.1, 0.9)
    muscle_deterioration_score = (
        0.45 * mri_fat_fraction
        - 0.55 * dexa_lean_mass_index
        + 0.18 * ultrasound_echo_intensity
        + 0.21 * microwave_phase_shift
        + 0.09 * microwave_attenuation
        + 0.09 * age
        - 0.07 * activity
        + 1.1 * inflammation
        + sex_effect
        + rng.normal(0, 2.0, n_samples)
    )

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age,
            "bmi": bmi,
            "sex": sex,
            "physical_activity_score": activity,
            "inflammation_marker": inflammation,
            "ultrasound_echo_intensity": ultrasound_echo_intensity,
            "mri_fat_fraction": mri_fat_fraction,
            "dexa_lean_mass_index": dexa_lean_mass_index,
            "microwave_phase_shift": microwave_phase_shift,
            "microwave_attenuation": microwave_attenuation,
            "muscle_deterioration_score": muscle_deterioration_score,
        }
    )

    output = RAW_DATA_DIR / "synthetic_muscle_multimodal.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Synthetic dataset saved to: {output}")
    print(df.head(3))


if __name__ == "__main__":
    main()
