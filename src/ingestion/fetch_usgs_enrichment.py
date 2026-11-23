"""Phase 1 - Step 3: Fetch USGS FDSN Event catalog (M>=4.5, Peru bbox) and enrich IGP Instrumental.

Queries `https://earthquake.usgs.gov/fdsnws/event/1/query` in 5-year blocks from
1960-01-01 to 2026-11-23 with retries, caches raw GeoJSON responses under
`data/raw/usgs/`, normalizes timestamps to `datetime64[us, UTC]`, saves
`data/interim/catalog_usgs_m45.parquet`, and performs space-time matching
(|dt| <= 30 s, epicentral distance <= 50 km) against `catalog_igp_instrumental_base.parquet`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_USGS_DIR = PROJECT_ROOT / "data" / "raw" / "usgs"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

IGP_INST_PARQUET = INTERIM_DIR / "catalog_igp_instrumental_base.parquet"
OUT_USGS_PARQUET = INTERIM_DIR / "catalog_usgs_m45.parquet"
OUT_ENRICH_PARQUET = INTERIM_DIR / "igp_usgs_enrichment.parquet"
METRICS_JSON = INTERIM_DIR / "fetch_usgs_metrics.json"

USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
BBOX_PARAMS = {
    "format": "geojson",
    "minlatitude": -21.0,
    "maxlatitude": 0.5,
    "minlongitude": -84.5,
    "maxlongitude": -67.0,
    "minmagnitude": 4.5,
}
TIME_BLOCKS = [
    ("1960-01-01", "1964-12-31T23:59:59"),
    ("1965-01-01", "1969-12-31T23:59:59"),
    ("1970-01-01", "1974-12-31T23:59:59"),
    ("1975-01-01", "1979-12-31T23:59:59"),
    ("1980-01-01", "1984-12-31T23:59:59"),
    ("1985-01-01", "1989-12-31T23:59:59"),
    ("1990-01-01", "1994-12-31T23:59:59"),
    ("1995-01-01", "1999-12-31T23:59:59"),
    ("2000-01-01", "2004-12-31T23:59:59"),
    ("2005-01-01", "2009-12-31T23:59:59"),
    ("2010-01-01", "2014-12-31T23:59:59"),
    ("2015-01-01", "2019-12-31T23:59:59"),
    ("2020-01-01", "2024-12-31T23:59:59"),
    ("2025-01-01", "2026-11-23T23:59:59"),
]

EARTH_RADIUS_KM = 6371.0
MAX_DT_SEC = 30.0
MAX_DIST_KM = 50.0


def _build_session() -> requests.Session:
    """Create requests session with automatic retry/backoff."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "PeruSeismoML-Phase1-ETL/1.0"})
    return session


def _fetch_block(session: requests.Session, start_str: str, end_str: str) -> dict[str, Any]:
    """Fetch a single 5-year USGS GeoJSON block, reusing local cache in `data/raw/usgs/`."""
    RAW_USGS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start_str[:4]}_{end_str[:4]}"
    cache_path = RAW_USGS_DIR / f"usgs_m45_{tag}.json"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = {**BBOX_PARAMS, "starttime": start_str, "endtime": end_str, "orderby": "time-asc"}
    last_err: Exception | None = None
    for attempt in range(1, 5):
        try:
            resp = session.get(USGS_ENDPOINT, params=params, timeout=60)
            if resp.status_code == 204:
                payload = {"type": "FeatureCollection", "features": []}
            else:
                resp.raise_for_status()
                payload = resp.json()
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.25)
            return payload
        except Exception as exc:
            last_err = exc
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"Failed fetching USGS block {tag}: {last_err}")


def _parse_usgs_features(blocks: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten USGS GeoJSON FeatureCollections into a typed DataFrame."""
    rows: list[dict[str, Any]] = []
    for payload in blocks:
        for feat in payload.get("features", []):
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None, None]
            types_str = props.get("types") or ""
            rows.append(
                {
                    "usgs_id": feat.get("id"),
                    "time_ms": props.get("time"),
                    "usgs_lon": coords[0] if len(coords) > 0 else None,
                    "usgs_lat": coords[1] if len(coords) > 1 else None,
                    "usgs_depth_km": coords[2] if len(coords) > 2 else None,
                    "usgs_mag": props.get("mag"),
                    "usgs_mag_type": props.get("magType"),
                    "azimuthal_gap": props.get("gap"),
                    "rms": props.get("rms"),
                    "mmi_shakemap": props.get("mmi"),
                    "pager_alert": props.get("alert"),
                    "has_moment_tensor": "moment-tensor" in types_str,
                    "usgs_event_type": props.get("type"),
                }
            )

    df = pd.DataFrame(rows)
    assert not df.empty, "USGS query returned 0 events"

    df["usgs_id"] = df["usgs_id"].astype("string")
    df["usgs_timestamp_utc"] = pd.to_datetime(
        df["time_ms"], unit="ms", utc=True
    ).astype("datetime64[us, UTC]")
    df = df.drop(columns=["time_ms"])

    for col in ("usgs_lon", "usgs_lat", "usgs_depth_km", "usgs_mag", "azimuthal_gap", "rms", "mmi_shakemap"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df["usgs_mag_type"] = df["usgs_mag_type"].astype("string")
    df["pager_alert"] = df["pager_alert"].astype("string")
    df["usgs_event_type"] = df["usgs_event_type"].astype("string")
    df["has_moment_tensor"] = df["has_moment_tensor"].astype("bool")

    df = df.drop_duplicates(subset=["usgs_id"]).sort_values("usgs_timestamp_utc").reset_index(drop=True)

    # Assertions
    assert df["usgs_id"].is_unique and df["usgs_id"].notna().all()
    assert str(df["usgs_timestamp_utc"].dtype) == "datetime64[us, UTC]"
    assert df["usgs_lat"].between(-21.5, 1.0).all()
    assert df["usgs_lon"].between(-85.0, -66.5).all()
    return df


def _spherical_dist_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance on 3D unit sphere (R=6371.0 km) from one point to an array."""
    r_lat1, r_lon1 = np.radians(lat1), np.radians(lon1)
    r_lat2, r_lon2 = np.radians(lat2), np.radians(lon2)
    x1, y1, z1 = np.cos(r_lat1) * np.cos(r_lon1), np.cos(r_lat1) * np.sin(r_lon1), np.sin(r_lat1)
    x2, y2, z2 = np.cos(r_lat2) * np.cos(r_lon2), np.cos(r_lat2) * np.sin(r_lon2), np.sin(r_lat2)
    chord = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def _match_igp_to_usgs(df_igp: pd.DataFrame, df_usgs: pd.DataFrame) -> pd.DataFrame:
    """Match IGP instrumental events to USGS within |dt| <= 30 s and dist <= 50 km."""
    igp_ts_us = df_igp["timestamp_utc"].astype("int64").to_numpy()
    usgs_ts_us = df_usgs["usgs_timestamp_utc"].astype("int64").to_numpy()
    usgs_lat = df_usgs["usgs_lat"].to_numpy()
    usgs_lon = df_usgs["usgs_lon"].to_numpy()

    tol_us = int(MAX_DT_SEC * 1_000_000)
    left_idx = np.searchsorted(usgs_ts_us, igp_ts_us - tol_us, side="left")
    right_idx = np.searchsorted(usgs_ts_us, igp_ts_us + tol_us, side="right")

    matched_rows: list[dict[str, Any]] = []
    igp_ids = df_igp["event_id"].to_numpy()
    igp_lats = df_igp["lat"].to_numpy()
    igp_lons = df_igp["lon"].to_numpy()

    for i in range(len(df_igp)):
        lo, hi = left_idx[i], right_idx[i]
        if lo >= hi:
            continue
        dists = _spherical_dist_km(igp_lats[i], igp_lons[i], usgs_lat[lo:hi], usgs_lon[lo:hi])
        valid_rel = np.where(dists <= MAX_DIST_KM)[0]
        if len(valid_rel) == 0:
            continue
        dts = np.abs(usgs_ts_us[lo:hi][valid_rel] - igp_ts_us[i]) / 1_000_000.0
        # Score by normalized time + space error to pick unambiguous best match
        best_pos = valid_rel[int(np.argmin(dts / MAX_DT_SEC + dists[valid_rel] / MAX_DIST_KM))]
        u_idx = lo + best_pos
        dt_signed = (usgs_ts_us[u_idx] - igp_ts_us[i]) / 1_000_000.0
        matched_rows.append(
            {
                "event_id": igp_ids[i],
                "usgs_row_idx": int(u_idx),
                "usgs_dt_sec": float(dt_signed),
                "usgs_dist_km": float(dists[best_pos]),
            }
        )

    if not matched_rows:
        raise RuntimeError("No matches found between IGP Instrumental and USGS M>=4.5")

    m_df = pd.DataFrame(matched_rows)
    # Enforce 1-to-1 on usgs_row_idx by keeping the closest IGP event if two collided
    m_df["score"] = m_df["usgs_dt_sec"].abs() / MAX_DT_SEC + m_df["usgs_dist_km"] / MAX_DIST_KM
    m_df = m_df.sort_values("score").drop_duplicates(subset=["usgs_row_idx"], keep="first")
    m_df = m_df.drop(columns=["score"]).reset_index(drop=True)

    usgs_cols = [
        "usgs_id",
        "usgs_mag",
        "usgs_mag_type",
        "azimuthal_gap",
        "rms",
        "mmi_shakemap",
        "pager_alert",
        "has_moment_tensor",
    ]
    usgs_selected = df_usgs.iloc[m_df["usgs_row_idx"].to_numpy()][usgs_cols].reset_index(drop=True)
    enrich = pd.concat(
        [m_df[["event_id", "usgs_dt_sec", "usgs_dist_km"]].reset_index(drop=True), usgs_selected],
        axis=1,
    )
    enrich["event_id"] = enrich["event_id"].astype("string")

    assert enrich["event_id"].is_unique and enrich["usgs_id"].is_unique
    assert (enrich["usgs_dt_sec"].abs() <= MAX_DT_SEC).all()
    assert (enrich["usgs_dist_km"] <= MAX_DIST_KM).all()
    return enrich


def main() -> None:
    """Fetch USGS catalog, save interim parquet, and cross-match against IGP Instrumental."""
    assert IGP_INST_PARQUET.exists(), f"Missing {IGP_INST_PARQUET}. Run etl_catalogs_igp.py first."

    session = _build_session()
    blocks: list[dict[str, Any]] = []
    block_counts: dict[str, int] = {}
    for start_str, end_str in TIME_BLOCKS:
        payload = _fetch_block(session, start_str, end_str)
        blocks.append(payload)
        block_counts[f"{start_str[:4]}-{end_str[:4]}"] = len(payload.get("features", []))

    df_usgs = _parse_usgs_features(blocks)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df_usgs.to_parquet(OUT_USGS_PARQUET, engine="pyarrow", compression="snappy", index=False)

    df_igp = pd.read_parquet(IGP_INST_PARQUET).sort_values("timestamp_utc").reset_index(drop=True)
    df_enrich = _match_igp_to_usgs(df_igp, df_usgs)
    df_enrich.to_parquet(OUT_ENRICH_PARQUET, engine="pyarrow", compression="snappy", index=False)

    # Enrichment rates for all IGP events and specifically M >= 4.5
    igp_merged = df_igp.merge(df_enrich, on="event_id", how="left")
    m45 = igp_merged[igp_merged["mag"] >= 4.5]
    m45_matched = m45[m45["usgs_id"].notna()]

    metrics = {
        "usgs_total_events": int(len(df_usgs)),
        "usgs_min_utc": df_usgs["usgs_timestamp_utc"].min().isoformat(),
        "usgs_max_utc": df_usgs["usgs_timestamp_utc"].max().isoformat(),
        "block_counts": block_counts,
        "igp_total_events": int(len(df_igp)),
        "igp_matched_all": int(len(df_enrich)),
        "igp_m45_total": int(len(m45)),
        "igp_m45_matched": int(len(m45_matched)),
        "igp_m45_match_rate_pct": round(float(len(m45_matched) / len(m45) * 100.0), 4),
        "m45_with_azimuthal_gap": int(m45_matched["azimuthal_gap"].notna().sum()),
        "m45_with_mmi_shakemap": int(m45_matched["mmi_shakemap"].notna().sum()),
        "m45_with_pager_alert": int(m45_matched["pager_alert"].notna().sum()),
        "m45_with_moment_tensor": int(m45_matched["has_moment_tensor"].fillna(False).sum()),
        "all_with_azimuthal_gap": int(df_enrich["azimuthal_gap"].notna().sum()),
        "all_with_mmi_shakemap": int(df_enrich["mmi_shakemap"].notna().sum()),
        "all_with_moment_tensor": int(df_enrich["has_moment_tensor"].sum()),
        "match_dt_median_sec": round(float(df_enrich["usgs_dt_sec"].abs().median()), 3),
        "match_dist_median_km": round(float(df_enrich["usgs_dist_km"].median()), 3),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== ENRIQUECIMIENTO USGS COMPLETADO ===")
    print(f"Eventos USGS descargados (M>=4.5): {metrics['usgs_total_events']} ({metrics['usgs_min_utc']} -> {metrics['usgs_max_utc']})")
    print(f"Cruces IGP-USGS totales:           {metrics['igp_matched_all']} / {metrics['igp_total_events']}")
    print(f"Cruce en IGP M>=4.5:               {metrics['igp_m45_matched']} / {metrics['igp_m45_total']} ({metrics['igp_m45_match_rate_pct']}%)")
    print(f"  con azimuthal_gap:               {metrics['m45_with_azimuthal_gap']}")
    print(f"  con mmi_shakemap:                {metrics['m45_with_mmi_shakemap']}")
    print(f"  con has_moment_tensor:           {metrics['m45_with_moment_tensor']}")


if __name__ == "__main__":
    main()
