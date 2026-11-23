"""Phase 1 - Step 2: Clean ACELDAT dataset and perform temporal merge with IGP Instrumental.

Reads `base_informativa/IGP_ACELDAT_Peru_2019-2025_Dataset_0.csv` (cp1252, sep=';'),
fixes leading-zero truncation on `UBIGEO` and `HORA_EVENTO`, parses dual date formats
(`%Y%m%d` and `%m/%d/%Y`), empirically tests UTC (offset=0h) vs Peru local time
(offset=+5h) via `pd.merge_asof` (+-120 s) against IGP Instrumental, computes spherical
epicentral/hypocentral distances and GMPE PGA targets, and exports Parquet datasets.
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

ACELDAT_CSV = BASE_DIR / "IGP_ACELDAT_Peru_2019-2025_Dataset_0.csv"
IGP_INST_PARQUET = INTERIM_DIR / "catalog_igp_instrumental_base.parquet"

OUT_MERGED_PARQUET = INTERIM_DIR / "aceldat_merged_base.parquet"
OUT_UNMATCHED_PARQUET = INTERIM_DIR / "aceldat_unmatched.parquet"
METRICS_JSON = INTERIM_DIR / "etl_aceldat_metrics.json"

EARTH_RADIUS_KM = 6371.0
MATCH_TOLERANCE_SEC = 120


def _parse_dual_fecha_hora(fecha_s: pd.Series, hora_s: pd.Series) -> pd.Series:
    """Parse mixed `YYYYMMDD` and `M/D/YYYY` combined with zero-padded `HHMMSS`."""
    f = fecha_s.str.strip()
    h = hora_s.str.strip().str.zfill(6)
    assert (h.str.len() == 6).all() and h.str.match(r"^\d{6}$").all(), "Invalid HORA_EVENTO format"

    dt = pd.Series(pd.NaT, index=f.index, dtype="datetime64[us, UTC]")

    mask_ymd = f.str.match(r"^\d{8}$")
    if mask_ymd.any():
        dt.loc[mask_ymd] = pd.to_datetime(
            f.loc[mask_ymd] + h.loc[mask_ymd], format="%Y%m%d%H%M%S", utc=True
        ).astype("datetime64[us, UTC]")

    mask_slash = ~mask_ymd & f.str.contains("/", regex=False)
    if mask_slash.any():
        dt.loc[mask_slash] = pd.to_datetime(
            f.loc[mask_slash] + " " + h.loc[mask_slash],
            format="%m/%d/%Y %H%M%S",
            utc=True,
        ).astype("datetime64[us, UTC]")

    assert dt.notna().all(), f"Unparsed ACELDAT dates found: {int(dt.isna().sum())}"
    return dt


def _spherical_distance_and_azimuth(
    lat1_deg: np.ndarray, lon1_deg: np.ndarray, lat2_deg: np.ndarray, lon2_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D unit-sphere great-circle distance (km, R=6371.0) and forward azimuth [0, 360)."""
    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)

    # 3D unit-sphere Cartesian chord distance -> exact great-circle arc
    x1, y1, z1 = np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)
    chord = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    dist_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))

    # Forward azimuth from event (1) to station (2)
    dlon = lon2 - lon1
    y_az = np.sin(dlon) * np.cos(lat2)
    x_az = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    az_deg = (np.degrees(np.arctan2(y_az, x_az)) + 360.0) % 360.0
    return dist_km, az_deg


def _run_merge_asof_test(
    df_ac: pd.DataFrame, df_igp: pd.DataFrame, offset_hours: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Test `merge_asof` (+-120s) applying a candidate UTC hour offset to ACELDAT timestamps."""
    left = df_ac.copy()
    left["timestamp_utc"] = (
        left["aceldat_time_raw"] + pd.Timedelta(hours=offset_hours)
    ).astype("datetime64[us, UTC]")
    left = left.sort_values("timestamp_utc", ascending=True).reset_index(drop=True)

    right = df_igp[
        ["event_id", "timestamp_utc", "lat", "lon", "depth_km", "mag", "energy_joules", "seismic_moment_nm"]
    ].rename(
        columns={
            "timestamp_utc": "event_timestamp_utc",
            "lat": "event_lat",
            "lon": "event_lon",
            "depth_km": "event_depth_km",
            "mag": "event_mag",
        }
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="timestamp_utc",
        right_on="event_timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=MATCH_TOLERANCE_SEC),
    )
    matched_mask = merged["event_id"].notna()
    dt_sec = (merged.loc[matched_mask, "timestamp_utc"] - merged.loc[matched_mask, "event_timestamp_utc"]).dt.total_seconds()
    abs_dt = dt_sec.abs()

    stats = {
        "offset_hours": offset_hours,
        "label": "UTC (offset 0h)" if offset_hours == 0 else f"UTC-5->UTC (offset +{offset_hours}h)",
        "total_rows": int(len(merged)),
        "matched_rows": int(matched_mask.sum()),
        "unmatched_rows": int((~matched_mask).sum()),
        "match_rate_pct": round(float(matched_mask.mean() * 100.0), 4),
        "unique_events_matched": int(merged.loc[matched_mask, "event_id"].nunique()),
        "abs_dt_median_sec": round(float(abs_dt.median()), 4) if not abs_dt.empty else None,
        "abs_dt_p95_sec": round(float(abs_dt.quantile(0.95)), 4) if not abs_dt.empty else None,
        "abs_dt_max_sec": round(float(abs_dt.max()), 4) if not abs_dt.empty else None,
    }
    return merged, stats


def main() -> None:
    """Execute ACELDAT ETL, timezone verification, and IGP event matching."""
    assert IGP_INST_PARQUET.exists(), f"Missing {IGP_INST_PARQUET}. Run etl_catalogs_igp.py first."

    raw = pd.read_csv(ACELDAT_CSV, encoding="cp1252", sep=";", dtype=str)
    unnamed_cols = [c for c in raw.columns if c.startswith("Unnamed:")]
    raw = raw.drop(columns=unnamed_cols)
    assert len(raw) == 18313, f"Expected 18313 ACELDAT rows, got {len(raw)}"
    assert len(raw.columns) == 13, f"Expected 13 clean columns, got {list(raw.columns)}"

    df = pd.DataFrame(
        {
            "fecha_corte": raw["FECHA_CORTE"].str.strip().astype("string"),
            "ubigeo": raw["UBIGEO"].str.strip().str.zfill(6).astype("string"),
            "departamento": raw["DEPARTAMENTO"].str.strip().astype("string"),
            "provincia": raw["PROVINCIA"].str.strip().astype("string"),
            "distrito": raw["DISTRITO"].str.strip().astype("string"),
            "fecha_evento_raw": raw["FECHA_EVENTO"].str.strip().astype("string"),
            "hora_evento_str": raw["HORA_EVENTO"].str.strip().str.zfill(6).astype("string"),
            "aceldat_time_raw": _parse_dual_fecha_hora(raw["FECHA_EVENTO"], raw["HORA_EVENTO"]),
            "cod_estacion": raw["COD_ESTACION"].str.strip().astype("string"),
            "station_lat": pd.to_numeric(raw["LAT_ESTACION"].str.strip(), errors="raise").astype("float64"),
            "station_lon": pd.to_numeric(raw["LON_ESTACION"].str.strip(), errors="raise").astype("float64"),
            "pga_v": pd.to_numeric(raw["ACEL_MAX_VERTICAL"].str.strip(), errors="raise").astype("float64"),
            "pga_ns": pd.to_numeric(raw["ACEL_MAX_NORTE-SUR"].str.strip(), errors="raise").astype("float64"),
            "pga_ew": pd.to_numeric(raw["ACEL_MAX_ESTE-OESTE"].str.strip(), errors="raise").astype("float64"),
        }
    )

    # Structural assertions before merge
    assert (df["ubigeo"].str.len() == 6).all() and df["ubigeo"].str.match(r"^\d{6}$").all()
    assert df["station_lat"].between(-20.0, 0.0).all(), "Station lat out of Peru bounds"
    assert df["station_lon"].between(-83.0, -68.0).all(), "Station lon out of Peru bounds"
    assert (df["pga_v"] > 0).all() and (df["pga_ns"] > 0).all() and (df["pga_ew"] > 0).all()

    df_igp = pd.read_parquet(IGP_INST_PARQUET)
    df_igp = df_igp.sort_values("timestamp_utc", ascending=True).reset_index(drop=True)

    # Empirical timezone check: offset 0h (UTC) vs offset +5h (UTC-5 local time)
    merged_0h, stats_0h = _run_merge_asof_test(df, df_igp, offset_hours=0)
    merged_5h, stats_5h = _run_merge_asof_test(df, df_igp, offset_hours=5)

    if stats_0h["matched_rows"] >= stats_5h["matched_rows"]:
        chosen_merged, chosen_stats = merged_0h, stats_0h
    else:
        chosen_merged, chosen_stats = merged_5h, stats_5h

    assert chosen_stats["match_rate_pct"] >= 80.0, (
        f"Unexpectedly low match rate ({chosen_stats['match_rate_pct']}%) for both UTC and UTC-5"
    )

    chosen_merged["dt_seconds"] = (
        chosen_merged["timestamp_utc"] - chosen_merged["event_timestamp_utc"]
    ).dt.total_seconds()
    chosen_merged["abs_dt_seconds"] = chosen_merged["dt_seconds"].abs()

    matched_mask = chosen_merged["event_id"].notna() & (chosen_merged["abs_dt_seconds"] <= MATCH_TOLERANCE_SEC)
    df_valid = chosen_merged.loc[matched_mask].copy().reset_index(drop=True)
    df_unmatched = chosen_merged.loc[~matched_mask].copy().reset_index(drop=True)

    # Compute epicentral/hypocentral distances, azimuth, and GMPE PGA targets
    dist_epi, az_deg = _spherical_distance_and_azimuth(
        df_valid["event_lat"].to_numpy(),
        df_valid["event_lon"].to_numpy(),
        df_valid["station_lat"].to_numpy(),
        df_valid["station_lon"].to_numpy(),
    )
    df_valid["dist_epicentral_km"] = dist_epi
    df_valid["dist_hypocentral_km"] = np.sqrt(dist_epi ** 2 + df_valid["event_depth_km"].to_numpy() ** 2)
    df_valid["azimuth_deg"] = az_deg

    df_valid["pga_h_max"] = np.maximum(df_valid["pga_ns"], df_valid["pga_ew"])
    df_valid["pga_h_geom"] = np.sqrt(df_valid["pga_ns"] * df_valid["pga_ew"])
    df_valid["pga_3d"] = np.sqrt(df_valid["pga_v"] ** 2 + df_valid["pga_ns"] ** 2 + df_valid["pga_ew"] ** 2)
    df_valid["log10_pga_h_geom"] = np.log10(df_valid["pga_h_geom"])

    # Drop temporary raw timestamp column now that `timestamp_utc` is validated
    df_valid = df_valid.drop(columns=["aceldat_time_raw"])
    df_unmatched = df_unmatched.drop(
        columns=[
            "event_id",
            "event_timestamp_utc",
            "event_lat",
            "event_lon",
            "event_depth_km",
            "event_mag",
            "energy_joules",
            "seismic_moment_nm",
            "dt_seconds",
            "abs_dt_seconds",
        ]
    )

    # Final integrity assertions
    assert len(df_valid) + len(df_unmatched) == len(raw)
    assert df_valid["event_id"].notna().all()
    assert str(df_valid["timestamp_utc"].dtype) == "datetime64[us, UTC]"
    assert str(df_valid["event_timestamp_utc"].dtype) == "datetime64[us, UTC]"
    assert (df_valid["dist_hypocentral_km"] >= df_valid["dist_epicentral_km"]).all()
    assert np.isfinite(df_valid["log10_pga_h_geom"]).all()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df_valid.to_parquet(OUT_MERGED_PARQUET, engine="pyarrow", compression="snappy", index=False)
    df_unmatched.to_parquet(OUT_UNMATCHED_PARQUET, engine="pyarrow", compression="snappy", index=False)

    metrics = {
        "input_rows": int(len(raw)),
        "dropped_unnamed_cols": len(unnamed_cols),
        "timezone_comparison": {"offset_0h_utc": stats_0h, "offset_5h_local": stats_5h},
        "selected_alignment": chosen_stats,
        "valid_rows": int(len(df_valid)),
        "unmatched_rows": int(len(df_unmatched)),
        "unique_event_ids": int(df_valid["event_id"].nunique()),
        "unique_stations": int(df_valid["cod_estacion"].nunique()),
        "unique_ubigeos": int(df_valid["ubigeo"].nunique()),
        "unmatched_unique_timestamps": int(df_unmatched["timestamp_utc"].nunique()),
        "dist_epicentral_median_km": round(float(df_valid["dist_epicentral_km"].median()), 2),
        "dist_epicentral_p95_km": round(float(df_valid["dist_epicentral_km"].quantile(0.95)), 2),
        "pga_h_geom_median_gal": round(float(df_valid["pga_h_geom"].median()), 4),
        "pga_h_geom_max_gal": round(float(df_valid["pga_h_geom"].max()), 4),
        "output_merged_path": str(OUT_MERGED_PARQUET),
        "output_unmatched_path": str(OUT_UNMATCHED_PARQUET),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== ETL ACELDAT + CRUCE IGP COMPLETADO ===")
    print(f"Prueba offset  0h (UTC):         {stats_0h['matched_rows']}/{stats_0h['total_rows']} ({stats_0h['match_rate_pct']}%) | med |dt|={stats_0h['abs_dt_median_sec']}s")
    print(f"Prueba offset +5h (UTC-5->UTC):  {stats_5h['matched_rows']}/{stats_5h['total_rows']} ({stats_5h['match_rate_pct']}%) | med |dt|={stats_5h['abs_dt_median_sec']}s")
    print(f"Alineacion seleccionada:         {chosen_stats['label']}")
    print(f"Registros validos emparejados:   {len(df_valid)} ({chosen_stats['match_rate_pct']}%) | Eventos unicos (GroupKFold): {metrics['unique_event_ids']}")
    print(f"Registros no emparejados:        {len(df_unmatched)} ({metrics['unmatched_unique_timestamps']} timestamps unicos)")


if __name__ == "__main__":
    main()
