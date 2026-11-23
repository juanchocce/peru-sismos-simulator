"""Phase 1 - Step 4: Build territorial master table from FONDES PDF + GEOCATMIN FileGDB.

Extracts all 32 pages of `base_informativa/FONDES_relacion_distritos_nov26.pdf` using
`pdfplumber`, cleans line breaks, validates 6-digit UBIGEO uniqueness, ingests
`Peligros_Geologicos` and `Zonas_Criticas` from `data/raw/geocatmin/zipfolder/data.gdb`,
assigns H3 indices (res 6 and 7), saves interim Parquet files, and writes the district
master table `data/processed/distritos_maestro_ubigeo.parquet`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "base_informativa"
RAW_GDB = PROJECT_ROOT / "data" / "raw" / "geocatmin" / "zipfolder" / "data.gdb"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FONDES_PDF = BASE_DIR / "FONDES_relacion_distritos_nov26.pdf"
OUT_PELIGROS_PARQUET = INTERIM_DIR / "geocatmin_peligros.parquet"
OUT_ZONAS_PARQUET = INTERIM_DIR / "geocatmin_zonas_criticas.parquet"
OUT_DISTRITOS_PARQUET = PROCESSED_DIR / "distritos_maestro_ubigeo.parquet"
METRICS_JSON = INTERIM_DIR / "etl_fondes_geocatmin_metrics.json"


def _clean_text(val: Any) -> str:
    """Remove embedded newlines and collapse consecutive whitespace."""
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val).replace("\n", " ")).strip()


def extract_fondes_districts() -> tuple[pd.DataFrame, int]:
    """Extract district-level FONDES/SIGRID risk table across all 32 pages.

    Handles PDF horizontal cell overflow where long district names spill into the
    adjacent UBIGEO column (e.g., `GAR0A5Y0116` -> suffix `GARAY`, UBIGEO `050116`).
    """
    records: list[dict[str, Any]] = []
    overflow_repaired = 0
    with pdfplumber.open(FONDES_PDF) as pdf:
        assert len(pdf.pages) == 32, f"Expected 32 pages in FONDES PDF, got {len(pdf.pages)}"
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    if not row or len(row) < 6:
                        continue
                    nro = _clean_text(row[0])
                    if not nro.isdigit():
                        continue
                    distrito = _clean_text(row[3])
                    ub_raw = _clean_text(row[4])
                    ub_digits = re.sub(r"\D", "", ub_raw).zfill(6)
                    ub_letters = re.sub(r"[\d\s]+", "", ub_raw)
                    if ub_letters:
                        overflow_repaired += 1
                        sep = " " if distrito.endswith("ALBARRACIN") else ""
                        distrito = f"{distrito}{sep}{ub_letters}".strip()
                    records.append(
                        {
                            "fondes_nro": int(nro),
                            "departamento": _clean_text(row[1]),
                            "provincia": _clean_text(row[2]),
                            "distrito": distrito,
                            "ubigeo": ub_digits,
                            "nivel_riesgo_multipeligro_sigrid": _clean_text(row[5]),
                            "documento_sustento": _clean_text(row[6]) if len(row) > 6 else "",
                        }
                    )

    df = pd.DataFrame(records).sort_values("ubigeo").reset_index(drop=True)
    for col in ("departamento", "provincia", "distrito", "ubigeo", "nivel_riesgo_multipeligro_sigrid", "documento_sustento"):
        df[col] = df[col].astype("string")

    assert len(df) == 1893, f"Expected 1893 FONDES districts, extracted {len(df)}"
    assert df["ubigeo"].notna().all(), "Null UBIGEO in FONDES"
    assert (df["ubigeo"].str.len() == 6).all() and df["ubigeo"].str.match(r"^\d{6}$").all()
    assert df["ubigeo"].is_unique, f"Duplicate UBIGEOs in FONDES: {int(df['ubigeo'].duplicated().sum())}"
    assert (df["departamento"] != "").all() and (df["distrito"] != "").all()
    return df, overflow_repaired


def _clean_geocatmin_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace whitespace-only strings with pd.NA across string/object columns."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            s = out[col].astype("string").str.strip()
            out[col] = s.replace("", pd.NA)
    return out


def _add_h3_columns(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Assign H3 hex cell IDs at resolutions 6 and 7 in EPSG:4326."""
    lats = df[lat_col].to_numpy()
    lons = df[lon_col].to_numpy()
    h3_r6 = [
        h3.latlng_to_cell(float(la), float(lo), 6) if np.isfinite(la) and np.isfinite(lo) else pd.NA
        for la, lo in zip(lats, lons)
    ]
    h3_r7 = [
        h3.latlng_to_cell(float(la), float(lo), 7) if np.isfinite(la) and np.isfinite(lo) else pd.NA
        for la, lo in zip(lats, lons)
    ]
    df["h3_res6"] = pd.Series(h3_r6, index=df.index, dtype="string")
    df["h3_res7"] = pd.Series(h3_r7, index=df.index, dtype="string")
    return df


def process_geocatmin_layers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read, clean, H3-index, and save GEOCATMIN `Peligros_Geologicos` and `Zonas_Criticas`."""
    assert RAW_GDB.exists(), f"Missing GEOCATMIN GDB at {RAW_GDB}"

    # 1. Peligros_Geologicos
    g_pel = gpd.read_file(RAW_GDB, layer="Peligros_Geologicos")
    assert g_pel.crs is not None and g_pel.crs.to_epsg() == 4326
    pel = pd.DataFrame(g_pel.drop(columns=["geometry"]))
    pel["lon"] = g_pel.geometry.x.astype("float64")
    pel["lat"] = g_pel.geometry.y.astype("float64")
    pel = _clean_geocatmin_strings(pel)

    # Normalize timestamps to datetime64[us, UTC]
    for ts_col in ("FECHA_MUES", "FECHA_REGI"):
        if ts_col in pel.columns:
            pel[ts_col] = pd.to_datetime(pel[ts_col], utc=True, errors="coerce").astype("datetime64[us, UTC]")

    pel["ubigeo"] = pel["CD_DIST"].where(pel["CD_DIST"].isna(), pel["CD_DIST"].str.zfill(6)).astype("string")
    valid_ub = pel["ubigeo"].dropna()
    assert (valid_ub.str.len() == 6).all() and valid_ub.str.match(r"^\d{6}$").all()
    assert pel["lat"].between(-20.0, 0.5).all() and pel["lon"].between(-83.0, -68.0).all()

    pel = _add_h3_columns(pel, "lat", "lon")
    assert pel["h3_res6"].notna().all() and pel["h3_res7"].notna().all()
    assert len(pel) == 37023

    # 2. Zonas_Criticas
    g_zc = gpd.read_file(RAW_GDB, layer="Zonas_Criticas")
    assert g_zc.crs is not None and g_zc.crs.to_epsg() == 4326
    zc = pd.DataFrame(g_zc.drop(columns=["geometry"]))
    zc["lon"] = g_zc.geometry.x.astype("float64")
    zc["lat"] = g_zc.geometry.y.astype("float64")
    zc = _clean_geocatmin_strings(zc)
    zc["ubigeo"] = zc["CD_DIST"].where(zc["CD_DIST"].isna(), zc["CD_DIST"].str.zfill(6)).astype("string")
    valid_zc_ub = zc["ubigeo"].dropna()
    assert len(valid_zc_ub) == len(zc) == 2177
    assert (valid_zc_ub.str.len() == 6).all() and valid_zc_ub.str.match(r"^\d{6}$").all()

    zc = _add_h3_columns(zc, "lat", "lon")
    assert zc["h3_res6"].notna().all() and zc["h3_res7"].notna().all()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pel.to_parquet(OUT_PELIGROS_PARQUET, engine="pyarrow", compression="snappy", index=False)
    zc.to_parquet(OUT_ZONAS_PARQUET, engine="pyarrow", compression="snappy", index=False)
    return pel, zc


def build_district_master(
    df_fondes: pd.DataFrame, df_pel: pd.DataFrame, df_zc: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate GEOCATMIN hazards & critical zones by UBIGEO onto FONDES districts."""
    pel_valid = df_pel[df_pel["ubigeo"].notna()].copy()
    pel_valid["tipo_norm"] = pel_valid["TIPO_PELIG"].fillna("Sin_Clasificar").str.strip()

    pel_agg = pel_valid.groupby("ubigeo", as_index=False).agg(
        n_peligros_total=("ubigeo", "size"),
        n_peligros_flujo=("tipo_norm", lambda s: int((s == "Flujo").sum())),
        n_peligros_caida=("tipo_norm", lambda s: int((s == "Caida").sum())),
        n_peligros_deslizamiento=("tipo_norm", lambda s: int((s == "Deslizamiento").sum())),
        n_peligros_mov_complejo=("tipo_norm", lambda s: int((s == "Mov. Complejo").sum())),
        n_peligros_reptacion=("tipo_norm", lambda s: int((s == "Reptacion").sum())),
        n_peligros_otro=("tipo_norm", lambda s: int((~s.isin(["Flujo", "Caida", "Deslizamiento", "Mov. Complejo", "Reptacion"])).sum())),
        n_peligros_grado_alto=("GRADO_PELI", lambda s: int(s.fillna("").str.lower().isin(["alto", "muy alto"]).sum())),
        geocatmin_lat_mean=("lat", "mean"),
        geocatmin_lon_mean=("lon", "mean"),
    )

    zc_valid = df_zc[df_zc["ubigeo"].notna()].copy()
    zc_agg = zc_valid.groupby("ubigeo", as_index=False).agg(
        n_zonas_criticas_total=("ubigeo", "size"),
        n_zonas_criticas_activas=("INDICADORACTIVO", lambda s: int((s.fillna("").str.upper() == "A").sum())),
    )

    master = (
        df_fondes.merge(pel_agg, on="ubigeo", how="left")
        .merge(zc_agg, on="ubigeo", how="left")
        .sort_values("ubigeo")
        .reset_index(drop=True)
    )

    count_cols = [
        "n_peligros_total",
        "n_peligros_flujo",
        "n_peligros_caida",
        "n_peligros_deslizamiento",
        "n_peligros_mov_complejo",
        "n_peligros_reptacion",
        "n_peligros_otro",
        "n_peligros_grado_alto",
        "n_zonas_criticas_total",
        "n_zonas_criticas_activas",
    ]
    for col in count_cols:
        master[col] = master[col].fillna(0).astype("int32")

    assert len(master) == len(df_fondes) == 1893
    assert master["ubigeo"].is_unique and master["ubigeo"].notna().all()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUT_DISTRITOS_PARQUET, engine="pyarrow", compression="snappy", index=False)

    # Check unmatched GEOCATMIN ubigeos vs FONDES
    fondes_set = set(df_fondes["ubigeo"])
    pel_ubigeos = set(pel_valid["ubigeo"])
    zc_ubigeos = set(zc_valid["ubigeo"])

    metrics = {
        "fondes_districts": int(len(master)),
        "sigrid_risk_distribution": master["nivel_riesgo_multipeligro_sigrid"].value_counts().to_dict(),
        "peligros_total_points": int(len(df_pel)),
        "peligros_with_ubigeo": int(len(pel_valid)),
        "peligros_missing_ubigeo": int(df_pel["ubigeo"].isna().sum()),
        "peligros_ubigeos_not_in_fondes": int(len(pel_ubigeos - fondes_set)),
        "districts_with_peligros": int((master["n_peligros_total"] > 0).sum()),
        "districts_with_peligros_pct": round(float((master["n_peligros_total"] > 0).mean() * 100.0), 2),
        "zonas_criticas_total_points": int(len(df_zc)),
        "zonas_criticas_ubigeos_not_in_fondes": int(len(zc_ubigeos - fondes_set)),
        "districts_with_zonas_criticas": int((master["n_zonas_criticas_total"] > 0).sum()),
        "districts_with_zonas_criticas_pct": round(float((master["n_zonas_criticas_total"] > 0).mean() * 100.0), 2),
        "sum_peligros_assigned_to_fondes": int(master["n_peligros_total"].sum()),
        "sum_zonas_criticas_assigned_to_fondes": int(master["n_zonas_criticas_total"].sum()),
        "sum_zonas_criticas_activas_fondes": int(master["n_zonas_criticas_activas"].sum()),
    }
    return master, metrics


def main() -> None:
    """Execute FONDES + GEOCATMIN territorial ETL."""
    df_fondes, overflow_repaired = extract_fondes_districts()
    df_pel, df_zc = process_geocatmin_layers()
    _, metrics = build_district_master(df_fondes, df_pel, df_zc)
    metrics["fondes_overflow_cells_repaired"] = int(overflow_repaired)

    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== ETL TERRITORIAL FONDES + GEOCATMIN COMPLETADO ===")
    print(f"Distritos FONDES extraidos:        {metrics['fondes_districts']} (celdas overflow reparadas: {overflow_repaired})")
    print(f"Distribucion SIGRID multipeligro:  {metrics['sigrid_risk_distribution']}")
    print(f"Distritos con Peligros GEOCATMIN:  {metrics['districts_with_peligros']} ({metrics['districts_with_peligros_pct']}%) | pts asignados: {metrics['sum_peligros_assigned_to_fondes']}/{metrics['peligros_total_points']}")
    print(f"Distritos con Zonas Criticas:      {metrics['districts_with_zonas_criticas']} ({metrics['districts_with_zonas_criticas_pct']}%) | pts asignados: {metrics['sum_zonas_criticas_assigned_to_fondes']}/{metrics['zonas_criticas_total_points']}")


if __name__ == "__main__":
    main()
