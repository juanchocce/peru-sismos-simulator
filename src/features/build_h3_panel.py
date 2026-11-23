"""Phase 2 - Step 5: Build Strictly Causal Spatio-Temporal H3 (Res 5) Panel for Phase 3 ML.

Constructs a hexagonal Uber H3 resolution 5 grid (~252 km^2 per cell) covering Peru
(`Limite_Paises` + 150 km offshore coastal buffer to the trench), generates monthly
forecasting cuts from 2000-01-01 to 2026-11-01 with strictly causal historical
features (t_event < cut_date_utc; zero look-ahead leakage), Omori/ETAS decay scores
in cell and 1-ring neighborhood (`h3.grid_disk(cell, 1)`), static 3D tectonic and
GEOCATMIN multi-hazard attributes, and right-censoring-safe future targets
(30d, 90d, 365d for M>=4.5 and 90d, 365d for M>=5.0).

Exports `data/processed/h3_spatiotemporal_panel.parquet`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RAW_GDB = PROJECT_ROOT / "data" / "raw" / "geocatmin" / "zipfolder" / "data.gdb"
RAW_TECTONIC_ROOT = PROJECT_ROOT / "data" / "raw" / "tectonic_plates"

INST_PARQUET = PROCESSED_DIR / "catalog_igp_instrumental.parquet"
PELIGROS_PARQUET = INTERIM_DIR / "geocatmin_peligros.parquet"
ZONAS_PARQUET = INTERIM_DIR / "geocatmin_zonas_criticas.parquet"
SLAB_MODEL_JSON = INTERIM_DIR / "wadati_benioff_slab_model.json"

OUT_PANEL_PARQUET = PROCESSED_DIR / "h3_spatiotemporal_panel.parquet"
METRICS_JSON = INTERIM_DIR / "h3_panel_metrics.json"

H3_RES = 5
BUFFER_M = 150_000.0
EARTH_RADIUS_KM = 6371.0
DEG_LAT_TO_KM = 111.1949266
NAZCA_RIDGE_LAT = -15.0
NAZCA_TRENCH_BOUNDS = {"NZ\\SA", "NZ\\AP", "NZ\\ND"}

MC_MODERN = 4.7
ETAS_ALPHA = 0.8
ETAS_C_DAYS = 0.05
ETAS_P = 1.1
US_PER_DAY = 86_400_000_000  # microseconds per day


def _resolve_geojson_dir() -> Path:
    for p in (RAW_TECTONIC_ROOT / "tectonicplates-master" / "GeoJSON", RAW_TECTONIC_ROOT / "GeoJSON"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"PB2002 GeoJSON not found under {RAW_TECTONIC_ROOT}")


def _latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    c = np.cos(lat)
    return np.column_stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)))


def _predict_slab_depth(lat: np.ndarray, dist_trench_km: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Bilinearly interpolate Wadati-Benioff slab depth from Phase 2 Step 3 model."""
    lat_orig = np.asarray(model["lat_knots_deg"], dtype="float64")
    order = np.argsort(lat_orig)
    lat_k = lat_orig[order]
    dist_k = np.asarray(model["dist_knots_km"], dtype="float64")
    z_grid = np.asarray(model["slab_depth_grid_km"], dtype="float64")[order, :]

    lat_c = np.clip(lat, lat_k[0], lat_k[-1])
    dist_c = np.clip(dist_trench_km, dist_k[0], dist_k[-1])

    li_hi = np.clip(np.searchsorted(lat_k, lat_c, side="right"), 1, len(lat_k) - 1)
    li_lo = li_hi - 1
    wl = (lat_c - lat_k[li_lo]) / (lat_k[li_hi] - lat_k[li_lo])

    di_hi = np.clip(np.searchsorted(dist_k, dist_c, side="right"), 1, len(dist_k) - 1)
    di_lo = di_hi - 1
    wd = (dist_c - dist_k[di_lo]) / (dist_k[di_hi] - dist_k[di_lo])

    z0 = (1.0 - wd) * z_grid[li_lo, di_lo] + wd * z_grid[li_lo, di_hi]
    z1 = (1.0 - wd) * z_grid[li_hi, di_lo] + wd * z_grid[li_hi, di_hi]
    return ((1.0 - wl) * z0 + wl * z1).astype("float64")


def _build_peru_h3_grid() -> pd.DataFrame:
    """Generate H3 res-5 cells covering Peru + 150 km coastal/offshore buffer."""
    gdf_peru = gpd.read_file(RAW_GDB, layer="Limite_Paises")
    assert len(gdf_peru) == 1
    # Project to UTM 18S (EPSG:32718) for accurate 150 km metric buffer, then back to WGS84
    buffered = gdf_peru.to_crs(epsg=32718).buffer(BUFFER_M).simplify(1500.0).to_crs(epsg=4326)
    geom = buffered.iloc[0]

    cells = sorted(h3.geo_to_cells(geom.__geo_interface__, H3_RES))
    assert len(cells) > 3000, f"Unexpectedly few H3 res-5 cells: {len(cells)}"

    lats, lons = zip(*(h3.cell_to_latlng(c) for c in cells))
    grid = pd.DataFrame(
        {
            "h3_res5": pd.Series(cells, dtype="string"),
            "cell_lat": np.asarray(lats, dtype="float64"),
            "cell_lon": np.asarray(lons, dtype="float64"),
        }
    )
    return grid


def _attach_static_cell_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Attach 3D trench distance, PB2002 convergence/seafloor age, slab depth, and GEOCATMIN counts."""
    geojson_dir = _resolve_geojson_dir()
    geod = Geod(ellps="WGS84")

    # Trench steps densified <= 2 km
    gdf_steps = gpd.read_file(geojson_dir / "PB2002_steps.json")
    trench = gdf_steps[gdf_steps["PLATEBOUND"].isin(NAZCA_TRENCH_BOUNDS)]
    pts: list[tuple[float, float]] = []
    vel_list: list[float] = []
    age_list: list[float] = []
    for _, r in trench.iterrows():
        coords = list(r.geometry.coords)
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i][0], coords[i][1]
            lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
            _, _, dm = geod.inv(lon1, lat1, lon2, lat2)
            n_int = max(int(np.ceil(dm / 2000.0)) - 1, 0)
            seg_pts = [(lon1, lat1)] + (geod.npts(lon1, lat1, lon2, lat2, n_int) if n_int > 0 else []) + [(lon2, lat2)]
            for lo, la in seg_pts:
                pts.append((lo, la))
                vel_list.append(float(r["VELOCITYLE"]))
                age_list.append(float(r["SEAFLOORAG"]))

    arr = np.asarray(pts, dtype="float64")
    tree = cKDTree(_latlon_to_unit_xyz(arr[:, 1], arr[:, 0]))

    cell_xyz = _latlon_to_unit_xyz(grid["cell_lat"].to_numpy(), grid["cell_lon"].to_numpy())
    d_chord, idx = tree.query(cell_xyz, k=1)
    grid["dist_trench_km"] = (2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(d_chord / 2.0, 0.0, 1.0))).astype("float32")
    grid["convergence_vel_mm_yr"] = np.asarray(vel_list, dtype="float32")[idx]
    grid["seafloor_age_myr"] = np.asarray(age_list, dtype="float32")[idx]
    grid["dist_lat_15s_km"] = ((grid["cell_lat"] - NAZCA_RIDGE_LAT) * DEG_LAT_TO_KM).astype("float32")
    grid["regimen_subduccion_lat"] = pd.Series(
        np.where(grid["cell_lat"] > NAZCA_RIDGE_LAT, "flat_slab_norte_centro", "normal_sur"),
        dtype="string",
    )

    slab_payload = json.loads(SLAB_MODEL_JSON.read_text(encoding="utf-8"))["model"]
    grid["slab_depth_est_km"] = _predict_slab_depth(
        grid["cell_lat"].to_numpy(), grid["dist_trench_km"].to_numpy(dtype="float64"), slab_payload
    ).astype("float32")

    # GEOCATMIN hazards & critical zones aggregated to H3 res 5
    df_pel = pd.read_parquet(PELIGROS_PARQUET, columns=["lat", "lon"])
    pel_h3 = pd.Series([h3.latlng_to_cell(float(la), float(lo), H3_RES) for la, lo in zip(df_pel["lat"], df_pel["lon"])], dtype="string")
    pel_cnt = pel_h3.value_counts().rename("n_geocatmin_peligros")

    df_zc = pd.read_parquet(ZONAS_PARQUET, columns=["lat", "lon"])
    zc_h3 = pd.Series([h3.latlng_to_cell(float(la), float(lo), H3_RES) for la, lo in zip(df_zc["lat"], df_zc["lon"])], dtype="string")
    zc_cnt = zc_h3.value_counts().rename("n_geocatmin_zonas_criticas")

    grid = grid.merge(pel_cnt, left_on="h3_res5", right_index=True, how="left")
    grid = grid.merge(zc_cnt, left_on="h3_res5", right_index=True, how="left")
    grid["n_geocatmin_peligros"] = grid["n_geocatmin_peligros"].fillna(0).astype("int32")
    grid["n_geocatmin_zonas_criticas"] = grid["n_geocatmin_zonas_criticas"].fillna(0).astype("int32")
    return grid


def _index_events_by_cell(df_inst: pd.DataFrame, valid_cells: set[str]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Group event arrays (ts_us, mag, energy, moment) by H3 res-5 cell and by 1-ring disk."""
    df_s = df_inst.sort_values("timestamp_utc").reset_index(drop=True)
    ev_cells = [h3.latlng_to_cell(float(la), float(lo), H3_RES) for la, lo in zip(df_s["lat"].to_numpy(), df_s["lon"].to_numpy())]
    ts_us = df_s["timestamp_utc"].astype("int64").to_numpy()
    mags = df_s["mag"].to_numpy(dtype="float64")
    energy = df_s["energy_joules"].to_numpy(dtype="float64")
    moment = df_s["seismic_moment_nm"].to_numpy(dtype="float64")

    # Direct cell buckets
    cell_indices: dict[str, list[int]] = {}
    nbhd_indices: dict[str, list[int]] = {}
    for i, c in enumerate(ev_cells):
        if c in valid_cells:
            cell_indices.setdefault(c, []).append(i)
        for nb in h3.grid_disk(c, 1):
            if nb in valid_cells:
                nbhd_indices.setdefault(nb, []).append(i)

    cell_events = {
        c: (ts_us[idxs], mags[idxs], energy[idxs], moment[idxs])
        for c, idxs in cell_indices.items()
    }
    nbhd_events = {
        c: (ts_us[idxs], mags[idxs])
        for c, idxs in nbhd_indices.items()
    }
    return cell_events, nbhd_events


def _compute_etas_decay_vec(
    cut_ts_us: np.ndarray, ev_ts_us: np.ndarray, ev_mag: np.ndarray, window_days: int = 365
) -> np.ndarray:
    """Strictly causal ETAS/Omori decay sum for events in [cut - window_days, cut) with M >= 4.0."""
    n_cuts = len(cut_ts_us)
    out = np.zeros(n_cuts, dtype="float32")
    mask = ev_mag >= 4.0
    if not np.any(mask):
        return out
    t_ev = ev_ts_us[mask]
    w_ev = np.power(10.0, ETAS_ALPHA * (ev_mag[mask] - MC_MODERN))
    win_us = window_days * US_PER_DAY
    lo_arr = np.searchsorted(t_ev, cut_ts_us - win_us, side="left")
    hi_arr = np.searchsorted(t_ev, cut_ts_us, side="left")  # strictly < cut_ts_us
    for k in range(n_cuts):
        lo, hi = lo_arr[k], hi_arr[k]
        if lo < hi:
            dt_days = (cut_ts_us[k] - t_ev[lo:hi]) / float(US_PER_DAY)
            out[k] = float(np.sum(w_ev[lo:hi] / np.power(dt_days + ETAS_C_DAYS, ETAS_P)))
    return out


def _window_sum_prefix(
    cut_ts_us: np.ndarray, ev_ts_us: np.ndarray, ev_vals: np.ndarray, window_days: int
) -> np.ndarray:
    """Fast O(log N) causal window sum over [cut - window_days, cut) using prefix sums."""
    if len(ev_ts_us) == 0:
        return np.zeros(len(cut_ts_us), dtype="float64")
    prefix = np.concatenate(([0.0], np.cumsum(ev_vals, dtype="float64")))
    lo = np.searchsorted(ev_ts_us, cut_ts_us - window_days * US_PER_DAY, side="left")
    hi = np.searchsorted(ev_ts_us, cut_ts_us, side="left")  # strict < cut_ts_us
    return prefix[hi] - prefix[lo]


def _future_count_prefix(
    cut_ts_us: np.ndarray, ev_ts_us: np.ndarray, horizon_days: int
) -> np.ndarray:
    """Fast O(log N) future event count over [cut, cut + horizon_days)."""
    if len(ev_ts_us) == 0:
        return np.zeros(len(cut_ts_us), dtype="int32")
    lo = np.searchsorted(ev_ts_us, cut_ts_us, side="left")
    hi = np.searchsorted(ev_ts_us, cut_ts_us + horizon_days * US_PER_DAY, side="left")
    return (hi - lo).astype("int32")


def _days_since_last(cut_ts_us: np.ndarray, ev_ts_us: np.ndarray) -> np.ndarray:
    """Days elapsed since most recent event strictly before cut_ts_us (NaN if none since 1960)."""
    out = np.full(len(cut_ts_us), np.nan, dtype="float32")
    if len(ev_ts_us) == 0:
        return out
    idx = np.searchsorted(ev_ts_us, cut_ts_us, side="left") - 1
    valid = idx >= 0
    if np.any(valid):
        out[valid] = ((cut_ts_us[valid] - ev_ts_us[idx[valid]]) / float(US_PER_DAY)).astype("float32")
    return out


def main() -> None:
    """Build and persist the H3 res-5 spatio-temporal panel."""
    for p in (INST_PARQUET, PELIGROS_PARQUET, ZONAS_PARQUET, SLAB_MODEL_JSON, RAW_GDB):
        assert p.exists(), f"Missing prerequisite: {p}"

    df_inst = pd.read_parquet(INST_PARQUET)
    max_catalog_ts = df_inst["timestamp_utc"].max()

    grid = _build_peru_h3_grid()
    grid = _attach_static_cell_features(grid)
    cells = grid["h3_res5"].tolist()
    n_cells = len(cells)
    valid_cells_set = set(cells)

    # Monthly forecasting cuts: 2000-01-01 to 2026-11-01 (strict UTC us)
    cuts = pd.date_range("2000-01-01", "2026-11-01", freq="MS", tz="UTC").astype("datetime64[us, UTC]")
    n_cuts = len(cuts)
    cut_ts_us = cuts.astype("int64").to_numpy()
    max_cat_us = int(max_catalog_ts.value // 1000) if hasattr(max_catalog_ts, "value") else int(pd.Series([max_catalog_ts]).astype("datetime64[us, UTC]").astype("int64").iloc[0])

    # Right-censoring masks per cut date
    complete_30d = (cut_ts_us + 30 * US_PER_DAY) <= max_cat_us
    complete_90d = (cut_ts_us + 90 * US_PER_DAY) <= max_cat_us
    complete_365d = (cut_ts_us + 365 * US_PER_DAY) <= max_cat_us

    cell_events, nbhd_events = _index_events_by_cell(df_inst, valid_cells_set)

    # Pre-allocate dense 2D arrays [n_cells, n_cuts] for fast memory-contiguous construction
    shape = (n_cells, n_cuts)
    past_windows = [("30d", 30), ("90d", 90), ("365d", 365), ("5y", 1826)]
    arrays: dict[str, np.ndarray] = {}
    for w_name, _ in past_windows:
        arrays[f"cnt_mc_{w_name}"] = np.zeros(shape, dtype="int16")
        arrays[f"cnt_m40_{w_name}"] = np.zeros(shape, dtype="int16")
        arrays[f"cnt_m45_{w_name}"] = np.zeros(shape, dtype="int16")
        arrays[f"nbhd_cnt_m45_{w_name}"] = np.zeros(shape, dtype="int16")
        arrays[f"log10_energy_{w_name}"] = np.zeros(shape, dtype="float32")
        arrays[f"log10_moment_{w_name}"] = np.zeros(shape, dtype="float32")

    arrays["etas_decay_cell_365d"] = np.zeros(shape, dtype="float32")
    arrays["etas_decay_nbhd_365d"] = np.zeros(shape, dtype="float32")
    arrays["days_since_last_m50_nbhd"] = np.full(shape, np.nan, dtype="float32")
    arrays["days_since_last_m60_nbhd"] = np.full(shape, np.nan, dtype="float32")

    # Future target counts (use float32 so censored cuts cleanly hold NaN)
    tgt_specs = [
        ("target_m45_30d", 4.5, 30, complete_30d),
        ("target_m45_90d", 4.5, 90, complete_90d),
        ("target_m45_365d", 4.5, 365, complete_365d),
        ("target_m50_90d", 5.0, 90, complete_90d),
        ("target_m50_365d", 5.0, 365, complete_365d),
    ]
    for t_name, _, _, comp_mask in tgt_specs:
        c_arr = np.zeros(shape, dtype="float32")
        b_arr = np.zeros(shape, dtype="float32")
        c_arr[:, ~comp_mask] = np.nan
        b_arr[:, ~comp_mask] = np.nan
        arrays[f"{t_name}_count"] = c_arr
        arrays[t_name] = b_arr

    # Populate cell-level features & targets (only loop over cells with events!)
    for ci, cell in enumerate(cells):
        if cell in cell_events:
            t_ev, m_ev, e_ev, m0_ev = cell_events[cell]
            m_mc = (m_ev >= MC_MODERN).astype("float64")
            m_40 = (m_ev >= 4.0).astype("float64")
            m_45 = (m_ev >= 4.5).astype("float64")

            for w_name, w_days in past_windows:
                arrays[f"cnt_mc_{w_name}"][ci] = _window_sum_prefix(cut_ts_us, t_ev, m_mc, w_days).astype("int16")
                arrays[f"cnt_m40_{w_name}"][ci] = _window_sum_prefix(cut_ts_us, t_ev, m_40, w_days).astype("int16")
                arrays[f"cnt_m45_{w_name}"][ci] = _window_sum_prefix(cut_ts_us, t_ev, m_45, w_days).astype("int16")
                e_sum = _window_sum_prefix(cut_ts_us, t_ev, e_ev, w_days)
                m0_sum = _window_sum_prefix(cut_ts_us, t_ev, m0_ev, w_days)
                arrays[f"log10_energy_{w_name}"][ci] = np.log10(1.0 + np.maximum(e_sum, 0.0)).astype("float32")
                arrays[f"log10_moment_{w_name}"][ci] = np.log10(1.0 + np.maximum(m0_sum, 0.0)).astype("float32")

            arrays["etas_decay_cell_365d"][ci] = _compute_etas_decay_vec(cut_ts_us, t_ev, m_ev, 365)

            t_m45 = t_ev[m_ev >= 4.5]
            t_m50 = t_ev[m_ev >= 5.0]
            for t_name, mag_thr, h_days, comp_mask in tgt_specs:
                t_sub = t_m45 if mag_thr == 4.5 else t_m50
                cnts = _future_count_prefix(cut_ts_us, t_sub, h_days).astype("float32")
                cnts[~comp_mask] = np.nan
                bins = np.where(comp_mask, (cnts > 0).astype("float32"), np.nan).astype("float32")
                arrays[f"{t_name}_count"][ci] = cnts
                arrays[t_name][ci] = bins

        if cell in nbhd_events:
            nt_ev, nm_ev = nbhd_events[cell]
            nm_45 = (nm_ev >= 4.5).astype("float64")
            for w_name, w_days in past_windows:
                arrays[f"nbhd_cnt_m45_{w_name}"][ci] = _window_sum_prefix(cut_ts_us, nt_ev, nm_45, w_days).astype("int16")
            arrays["etas_decay_nbhd_365d"][ci] = _compute_etas_decay_vec(cut_ts_us, nt_ev, nm_ev, 365)
            arrays["days_since_last_m50_nbhd"][ci] = _days_since_last(cut_ts_us, nt_ev[nm_ev >= 5.0])
            arrays["days_since_last_m60_nbhd"][ci] = _days_since_last(cut_ts_us, nt_ev[nm_ev >= 6.0])

    # Assemble flat DataFrame (cell-major order -> sort by cut_date_utc, h3_res5)
    n_total = n_cells * n_cuts
    panel_dict: dict[str, Any] = {
        "h3_res5": np.repeat(grid["h3_res5"].to_numpy(), n_cuts),
        "cut_date_utc": np.tile(cuts.to_numpy(), n_cells),
        "horizon_complete_30d": np.tile(complete_30d, n_cells),
        "horizon_complete_90d": np.tile(complete_90d, n_cells),
        "horizon_complete_365d": np.tile(complete_365d, n_cells),
    }

    # Static cell columns repeated across cuts
    for col in (
        "cell_lat",
        "cell_lon",
        "dist_trench_km",
        "slab_depth_est_km",
        "convergence_vel_mm_yr",
        "seafloor_age_myr",
        "dist_lat_15s_km",
        "regimen_subduccion_lat",
        "n_geocatmin_peligros",
        "n_geocatmin_zonas_criticas",
    ):
        arr_col = grid[col].to_numpy()
        if col in ("cell_lat", "cell_lon"):
            arr_col = arr_col.astype("float32")
        panel_dict[col] = np.repeat(arr_col, n_cuts)

    for k, v in arrays.items():
        panel_dict[k] = v.reshape(n_total)

    panel = pd.DataFrame(panel_dict)
    panel["h3_res5"] = panel["h3_res5"].astype("string")
    panel["regimen_subduccion_lat"] = panel["regimen_subduccion_lat"].astype("string")
    panel["cut_date_utc"] = pd.to_datetime(panel["cut_date_utc"], utc=True).astype("datetime64[us, UTC]")

    # Strict integrity & anti-leakage assertions
    assert len(panel) == n_cells * n_cuts
    assert str(panel["cut_date_utc"].dtype) == "datetime64[us, UTC]"
    assert panel["h3_res5"].notna().all() and panel["dist_trench_km"].notna().all()
    for t_name, _, _, _ in tgt_specs:
        valid = panel[t_name].notna()
        assert np.array_equal(panel.loc[valid, t_name] > 0, panel.loc[valid, f"{t_name}_count"] > 0)
    # Verify non-negativity of elapsed days
    ds50 = panel["days_since_last_m50_nbhd"].dropna()
    assert (ds50 >= 0).all(), "Negative days_since_last_m50_nbhd indicates look-ahead leakage!"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PANEL_PARQUET, engine="pyarrow", compression="snappy", index=False)

    # Target class balance metrics on complete (uncensored) rows
    target_stats: dict[str, dict[str, Any]] = {}
    for t_name, _, _, _ in tgt_specs:
        valid_s = panel[t_name].dropna()
        pos = int((valid_s == 1.0).sum())
        target_stats[t_name] = {
            "uncensored_rows": int(len(valid_s)),
            "positive_rows": pos,
            "positive_rate_pct": round(float(pos / len(valid_s) * 100.0), 4),
            "total_event_count_sum": int(panel[f"{t_name}_count"].dropna().sum()),
        }

    metrics = {
        "h3_resolution": H3_RES,
        "n_cells": int(n_cells),
        "n_active_cells_with_events": int(len(cell_events)),
        "n_cuts_monthly": int(n_cuts),
        "cut_min_utc": cuts.min().isoformat(),
        "cut_max_utc": cuts.max().isoformat(),
        "total_rows": int(len(panel)),
        "total_cols": int(len(panel.columns)),
        "parquet_size_mb": round(OUT_PANEL_PARQUET.stat().st_size / (1024 * 1024), 4),
        "complete_cuts_30d": int(complete_30d.sum()),
        "complete_cuts_90d": int(complete_90d.sum()),
        "complete_cuts_365d": int(complete_365d.sum()),
        "target_class_balance": target_stats,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== PANEL ESPACIO-TEMPORAL H3 (RES 5) COMPLETADO ===")
    print(f"Grilla H3 res 5: {n_cells} celdas ({len(cell_events)} con sismicidad) x {n_cuts} cortes mensuales = {len(panel):,} filas x {len(panel.columns)} cols ({metrics['parquet_size_mb']} MB)")
    for t_name, st in target_stats.items():
        print(f"  {t_name:16s}: {st['positive_rows']:7d} / {st['uncensored_rows']:,} positivos ({st['positive_rate_pct']:.3f}%)")


if __name__ == "__main__":
    main()
