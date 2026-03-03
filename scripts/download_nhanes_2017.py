import argparse
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from health_mldl.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
XPT_FILES = {
    "DEMO_J": f"{NHANES_BASE_URL}/DEMO_J.xpt",
    "BMX_J": f"{NHANES_BASE_URL}/BMX_J.xpt",
    "PAQ_J": f"{NHANES_BASE_URL}/PAQ_J.xpt",
    "HSCRP_J": f"{NHANES_BASE_URL}/HSCRP_J.xpt",
    "DXX_J": f"{NHANES_BASE_URL}/DXX_J.xpt",
}


def download_xpt_files(out_dir: Path, force: bool = False) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for name, url in XPT_FILES.items():
        dst = out_dir / f"{name}.XPT"
        if dst.exists() and not force:
            print(f"Reuse: {dst}")
        else:
            print(f"Download: {url}")
            urlretrieve(url, dst)
        paths[name] = dst

    return paths


def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")


def decode_colnames(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [c.decode("utf-8") if isinstance(c, bytes) else c for c in data.columns]
    return data


def make_activity_score(df_paq: pd.DataFrame) -> pd.Series:
    # Binary indicators (1=yes, 2=no) aggregated to a simple 0-100 proxy score.
    candidate_cols = ["PAQ650", "PAQ665", "PAD680", "PAQ710", "PAQ715"]
    available = [c for c in candidate_cols if c in df_paq.columns]
    if not available:
        return pd.Series([pd.NA] * len(df_paq), index=df_paq.index, dtype="Float64")

    score = pd.Series(0.0, index=df_paq.index)
    for col in available:
        score += (df_paq[col] == 1).astype(float)

    return (score / len(available) * 100.0).astype("Float64")


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_core_table(demo: pd.DataFrame, bmx: pd.DataFrame, paq: pd.DataFrame, hscrp: pd.DataFrame, dxx: pd.DataFrame) -> pd.DataFrame:
    keys = ["SEQN"]

    demo_cols = ["SEQN", "RIDAGEYR", "RIAGENDR"]
    bmx_cols = ["SEQN", "BMXBMI"]
    paq_cols = ["SEQN"]
    hscrp_cols = ["SEQN", "LBXHSCRP"] if "LBXHSCRP" in hscrp.columns else ["SEQN"]

    # DEXA proxy selection: prefer lean-mass-like variables, otherwise derive from fat %.
    lean_candidates = [
        "DXXTLE",
        "DXXTLM",
        "DXXLTM",
        "DXXTOTLM",
        "DXXTRLI",
        "DXXHELI",
        "DXXLALI",
        "DXXLLLI",
        "DXXRALI",
        "DXXRLLI",
        "DXXPCTFAT",
    ]
    dxx_lean_col = first_existing(dxx, lean_candidates)
    dxx_cols = ["SEQN"] + ([dxx_lean_col] if dxx_lean_col else [])

    data = demo[demo_cols].merge(bmx[bmx_cols], on=keys, how="inner")
    data = data.merge(paq[paq_cols], on=keys, how="left")
    data = data.merge(hscrp[hscrp_cols], on=keys, how="left")
    data = data.merge(dxx[dxx_cols], on=keys, how="left")

    data["patient_id"] = data["SEQN"].astype("Int64").astype(str).map(lambda x: f"NHANES_{x}")
    data["age"] = pd.to_numeric(data["RIDAGEYR"], errors="coerce")
    data["bmi"] = pd.to_numeric(data["BMXBMI"], errors="coerce")
    data["sex"] = data["RIAGENDR"].map({1: "M", 2: "F"})
    data["physical_activity_score"] = make_activity_score(paq).reindex(data.index).values
    data["inflammation_marker"] = pd.to_numeric(data.get("LBXHSCRP"), errors="coerce")

    if dxx_lean_col and dxx_lean_col != "DXXPCTFAT":
        data["dexa_lean_mass_index"] = pd.to_numeric(data[dxx_lean_col], errors="coerce")
        print(f"DEXA lean proxy used: {dxx_lean_col}")
    elif dxx_lean_col == "DXXPCTFAT":
        data["dexa_lean_mass_index"] = 100.0 - pd.to_numeric(data[dxx_lean_col], errors="coerce")
        print("DEXA lean proxy used: 100 - DXXPCTFAT")
    else:
        data["dexa_lean_mass_index"] = pd.NA
        print("Warning: no suitable DEXA lean column found, dexa_lean_mass_index is NA")

    core_cols = [
        "patient_id",
        "age",
        "bmi",
        "sex",
        "physical_activity_score",
        "inflammation_marker",
        "dexa_lean_mass_index",
    ]
    return data[core_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build NHANES 2017 core CSV.")
    parser.add_argument("--force-download", action="store_true", help="Redownload files.")
    args = parser.parse_args()

    xpt_dir = EXTERNAL_DATA_DIR / "nhanes_2017_2018" / "xpt"
    paths = download_xpt_files(xpt_dir, force=args.force_download)

    demo = decode_colnames(read_xpt(paths["DEMO_J"]))
    bmx = decode_colnames(read_xpt(paths["BMX_J"]))
    paq = decode_colnames(read_xpt(paths["PAQ_J"]))
    hscrp = decode_colnames(read_xpt(paths["HSCRP_J"]))
    dxx = decode_colnames(read_xpt(paths["DXX_J"]))

    core = build_core_table(demo, bmx, paq, hscrp, dxx)

    out_csv = RAW_DATA_DIR / "nhanes_2017_core.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    core.to_csv(out_csv, index=False)

    curated = core[(core["age"] >= 20) & (core["dexa_lean_mass_index"].notna())].copy()
    curated_csv = RAW_DATA_DIR / "nhanes_2017_core_adults_dexa.csv"
    curated.to_csv(curated_csv, index=False)

    print(f"Core NHANES CSV written: {out_csv}")
    print(f"Shape: {core.shape}")
    print(core.head(3))
    print(f"Curated NHANES CSV written: {curated_csv}")
    print(f"Curated shape (age>=20 with DEXA): {curated.shape}")
    print(
        "Next: PYTHONPATH=src python scripts/export_separated_modalities_csv.py "
        "--input-csv data/raw/nhanes_2017_core_adults_dexa.csv --dataset-id nhanes_2017"
    )


if __name__ == "__main__":
    main()
