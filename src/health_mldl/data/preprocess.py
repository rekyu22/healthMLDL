import pandas as pd

from health_mldl.features.schema import TARGET_COL
from health_mldl.features.schema import PATIENT_ID_COL


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop_duplicates()

    # Supprime lignes sans cible
    data = data.dropna(subset=[TARGET_COL])

    # Imputation simple des variables numériques
    num_cols = data.select_dtypes(include=["number"]).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())

    # Imputation mode pour variables catégorielles
    cat_cols = data.select_dtypes(exclude=["number"]).columns
    for col in cat_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode().iloc[0])

    return data


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [TARGET_COL]
    if PATIENT_ID_COL in df.columns:
        drop_cols.append(PATIENT_ID_COL)
    x = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    return x, y
