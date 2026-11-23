"""Phase 1 - Step 1: Ingest and normalize IGP Historical and Instrumental catalogs.

Reads read-only Excel files from `base_informativa/`, converts timestamps to
`datetime64[us, UTC]`, handles pseudo-nulls (`-`), homogenizes magnitudes for the
historical catalog, computes seismic energy and scalar seismic moment, and writes
validated Parquet files to `data/interim/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "base_informativa"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

HIST_XLSX = BASE_DIR / "datos-sismicos_Historico_1471-01-01_1959-12-31.xlsx"
INST_XLSX = BASE_DIR / "datos-sismicos_Instrumental_1960-01-01_2026-11-22.xlsx"
SHEET_NAME = "IGP - Datos sísmicos"

OUT_HIST_PARQUET = INTERIM_DIR / "catalog_igp_historico.parquet"
OUT_INST_PARQUET = INTERIM_DIR / "catalog_igp_instrumental_base.parquet"
METRICS_JSON = INTERIM_DIR / "etl_catalogs_igp_metrics.json"


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """Convert strings with '-' or whitespace pseudo-nulls into float64."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "-": np.nan, "--": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _parse_igp_utc(fecha_s: pd.Series, hora_s: pd.Series) -> pd.Series:
    """Combine `fecha UTC` and `hora UTC` into `datetime64[us, UTC]` safely (pre-1677 safe)."""
    combined = fecha_s.astype(str).str.strip() + " " + hora_s.astype(str).str.strip()
    dt = pd.to_datetime(combined, format="mixed", utc=True, errors="raise")
    return dt.astype("datetime64[us, UTC]")


def _homogenize_historical_magnitude(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Build traceable Mw-equivalent `mag_homog` using Mw > Ms (Scordilis 2006) > mb (Scordilis 2006).

    Empirical relations (Scordilis, 2006, J. Seismol.):
      - Ms -> Mw: 0.67 * Ms + 2.07  (Ms <= 6.1)
                  0.99 * Ms + 0.08  (Ms > 6.1)
      - mb -> Mw: 0.85 * mb + 1.03
    """
    mag_homog = pd.Series(np.nan, index=df.index, dtype="float64")
    source = pd.Series(pd.NA, index=df.index, dtype="string")

    # 1. Direct Mw
    mw_mask = df["mag_mw"].notna()
    mag_homog.loc[mw_mask] = df.loc[mw_mask, "mag_mw"]
    source.loc[mw_mask] = "Mw_direct"

    # 2. Ms -> Mw (Scordilis 2006)
    ms_mask = mag_homog.isna() & df["mag_ms"].notna()
    ms_vals = df.loc[ms_mask, "mag_ms"]
    ms_conv = np.where(ms_vals <= 6.1, 0.67 * ms_vals + 2.07, 0.99 * ms_vals + 0.08)
    mag_homog.loc[ms_mask] = np.round(ms_conv, 3)
    source.loc[ms_mask] = "Ms_to_Mw_Scordilis2006"

    # 3. mb -> Mw (Scordilis 2006)
    mb_mask = mag_homog.isna() & df["mag_mb"].notna()
    mb_vals = df.loc[mb_mask, "mag_mb"]
    mag_homog.loc[mb_mask] = np.round(0.85 * mb_vals + 1.03, 3)
    source.loc[mb_mask] = "mb_to_Mw_Scordilis2006"

    return mag_homog, source


def _compute_energy_and_moment(mag: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compute seismic energy E (Joules) and scalar seismic moment M0 (N*m)."""
    energy_j = np.power(10.0, 1.5 * mag + 4.8).astype("float64")
    moment_nm = np.power(10.0, 1.5 * mag + 9.1).astype("float64")
    return energy_j, moment_nm


def process_historical_catalog() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest and validate the IGP historical seismic catalog (1552-1959)."""
    raw = pd.read_excel(HIST_XLSX, sheet_name=SHEET_NAME, dtype=str, engine="openpyxl")
    expected_cols = [
        "fecha UTC",
        "hora UTC",
        "latitud (º)",
        "longitud (º)",
        "profundidad (km)",
        "magnitud (mb)",
        "magnitud (Ms)",
        "magnitud (Mw)",
    ]
    assert list(raw.columns) == expected_cols, f"Unexpected historical columns: {list(raw.columns)}"

    df = pd.DataFrame(
        {
            "timestamp_utc": _parse_igp_utc(raw["fecha UTC"], raw["hora UTC"]),
            "lat": _to_numeric_clean(raw["latitud (º)"]),
            "lon": _to_numeric_clean(raw["longitud (º)"]),
            "depth_km": _to_numeric_clean(raw["profundidad (km)"]),
            "mag_mb": _to_numeric_clean(raw["magnitud (mb)"]),
            "mag_ms": _to_numeric_clean(raw["magnitud (Ms)"]),
            "mag_mw": _to_numeric_clean(raw["magnitud (Mw)"]),
        }
    )

    df["mag_homog"], df["mag_scale_source"] = _homogenize_historical_magnitude(df)
    df["energy_joules"], df["seismic_moment_nm"] = _compute_energy_and_moment(df["mag_homog"])

    df = df.sort_values(["timestamp_utc", "lat", "lon"], ascending=True).reset_index(drop=True)
    df.insert(0, "event_id", [f"IGP_HIST_{i:05d}" for i in range(1, len(df) + 1)])
    df["event_id"] = df["event_id"].astype("string")

    # Integrity assertions
    assert len(df) == len(raw) == 1098, f"Expected 1098 historical rows, got {len(df)}"
    assert str(df["timestamp_utc"].dtype) == "datetime64[us, UTC]", df["timestamp_utc"].dtype
    assert df["timestamp_utc"].notna().all(), "Null timestamps in historical catalog"
    assert df["timestamp_utc"].is_monotonic_increasing, "Historical timestamps not sorted"
    assert df["event_id"].is_unique and df["event_id"].notna().all()
    assert df["lat"].between(-25.0, 5.0).all(), "Historical lat out of Peru bounds"
    assert df["lon"].between(-90.0, -60.0).all(), "Historical lon out of Peru bounds"
    assert df["mag_homog"].notna().all(), "Null mag_homog in historical catalog"
    assert (df["energy_joules"] > 0).all() and (df["seismic_moment_nm"] > 0).all()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_HIST_PARQUET, engine="pyarrow", compression="snappy", index=False)

    metrics = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "min_utc": df["timestamp_utc"].min().isoformat(),
        "max_utc": df["timestamp_utc"].max().isoformat(),
        "depth_null_count": int(df["depth_km"].isna().sum()),
        "mag_scale_source_counts": df["mag_scale_source"].value_counts().to_dict(),
        "mag_homog_min": float(df["mag_homog"].min()),
        "mag_homog_max": float(df["mag_homog"].max()),
        "output_path": str(OUT_HIST_PARQUET),
    }
    return df, metrics


def process_instrumental_catalog() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest and validate the IGP instrumental seismic catalog (1960-2026)."""
    raw = pd.read_excel(INST_XLSX, sheet_name=SHEET_NAME, dtype=str, engine="openpyxl")
    expected_cols = [
        "fecha UTC",
        "hora UTC",
        "latitud (º)",
        "longitud (º)",
        "profundidad (km)",
        "magnitud (M)",
    ]
    assert list(raw.columns) == expected_cols, f"Unexpected instrumental columns: {list(raw.columns)}"

    df = pd.DataFrame(
        {
            "timestamp_utc": _parse_igp_utc(raw["fecha UTC"], raw["hora UTC"]),
            "lat": _to_numeric_clean(raw["latitud (º)"]),
            "lon": _to_numeric_clean(raw["longitud (º)"]),
            "depth_km": _to_numeric_clean(raw["profundidad (km)"]),
            "mag": _to_numeric_clean(raw["magnitud (M)"]),
        }
    )

    df["energy_joules"], df["seismic_moment_nm"] = _compute_energy_and_moment(df["mag"])
    df = df.sort_values("timestamp_utc", ascending=True).reset_index(drop=True)
    df.insert(0, "event_id", [f"IGP_INST_{i:05d}" for i in range(1, len(df) + 1)])
    df["event_id"] = df["event_id"].astype("string")

    # Integrity assertions
    assert len(df) == len(raw) == 25878, f"Expected 25878 instrumental rows, got {len(df)}"
    assert str(df["timestamp_utc"].dtype) == "datetime64[us, UTC]", df["timestamp_utc"].dtype
    assert df.notna().all().all(), f"Unexpected nulls in instrumental catalog: {df.isna().sum().to_dict()}"
    assert df["timestamp_utc"].is_monotonic_increasing, "Instrumental timestamps not sorted"
    assert df["timestamp_utc"].is_unique, "Duplicate timestamps in instrumental catalog"
    assert df["event_id"].is_unique
    assert df["lat"].between(-25.0, 5.0).all(), "Instrumental lat out of Peru bounds"
    assert df["lon"].between(-90.0, -60.0).all(), "Instrumental lon out of Peru bounds"
    assert df["depth_km"].between(0.0, 800.0).all(), "Instrumental depth out of range"
    assert df["mag"].between(1.0, 10.0).all(), "Instrumental magnitude out of range"

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_INST_PARQUET, engine="pyarrow", compression="snappy", index=False)

    metrics = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "min_utc": df["timestamp_utc"].min().isoformat(),
        "max_utc": df["timestamp_utc"].max().isoformat(),
        "depth_min_km": float(df["depth_km"].min()),
        "depth_max_km": float(df["depth_km"].max()),
        "mag_min": float(df["mag"].min()),
        "mag_max": float(df["mag"].max()),
        "events_m45_plus": int((df["mag"] >= 4.5).sum()),
        "output_path": str(OUT_INST_PARQUET),
    }
    return df, metrics


def main() -> None:
    """Run IGP catalog ingestion pipeline."""
    _, hist_metrics = process_historical_catalog()
    _, inst_metrics = process_instrumental_catalog()

    summary = {"historico": hist_metrics, "instrumental": inst_metrics}
    METRICS_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== ETL CATALOGOS IGP COMPLETADO ===")
    print(f"Historico:    {hist_metrics['rows']} filas | {hist_metrics['min_utc']} -> {hist_metrics['max_utc']}")
    print(f"              Fuentes mag_homog: {hist_metrics['mag_scale_source_counts']}")
    print(f"Instrumental: {inst_metrics['rows']} filas | {inst_metrics['min_utc']} -> {inst_metrics['max_utc']}")
    print(f"              Eventos M>=4.5: {inst_metrics['events_m45_plus']}")


if __name__ == "__main__":
    main()
