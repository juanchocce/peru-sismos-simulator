"""Phase 4 - Step 1: Territorial Multi-Hazard Integration (UBIGEO + H3) & Megathrust ShakeMap Engine.

Integrates all 3 Phase 3 models (GMPE ML + Site Factors, Probabilistic Seismic Gaps
MC/BPT, H3 Operational Forecaster) with FONDES/SIGRID and GEOCATMIN onto:
  1. `data/processed/territorial_multihazard_ubigeo.parquet` (1,893 districts)
  2. `data/processed/territorial_multihazard_h3.parquet`     (8,205 H3 res-5 cells)

Applies mandatory hybrid physical-ML extrapolation for megathrust scenarios (Mw >= 8.0):
    log10(PGA_hybrid) = log10(PGA_LGBM(min(Mw, 7.5), ...)) + c1_Huber * max(0, Mw - 7.5) + delta_S2S

Includes extensible multi-hazard contract columns (`ext_hydromet_flood_idx`,
`ext_wildfire_firms_idx`, `ext_social_vuln_idx`, `ext_composite_multirisk_idx`)
and performs a quantitative discrepancy audit against FONDES SIGRID multi-hazard risk.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import joblib
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree
from sklearn.linear_model import HuberRegressor

import sys
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from nonlinear_site import apply_nonlinear_site_damping, simulate_hybrid_nonlinear_pga

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_TECTONIC_ROOT = PROJECT_ROOT / "data" / "raw" / "tectonic_plates"

DISTRITOS_PARQUET = PROCESSED_DIR / "distritos_maestro_ubigeo.parquet"
ACELDAT_PARQUET = PROCESSED_DIR / "aceldat_merged.parquet"
SITE_FACTORS_PARQUET = PROCESSED_DIR / "aceldat_station_site_factors.parquet"
GAPS_PROB_PARQUET = PROCESSED_DIR / "seismic_gaps_probabilistic.parquet"
FORECAST_H3_PARQUET = PROCESSED_DIR / "h3_forecast_latest.parquet"
PELIGROS_PARQUET = INTERIM_DIR / "geocatmin_peligros.parquet"
ZONAS_PARQUET = INTERIM_DIR / "geocatmin_zonas_criticas.parquet"
SLAB_MODEL_JSON = INTERIM_DIR / "wadati_benioff_slab_model.json"
GMPE_JOBLIB = MODELS_DIR / "gmpe_lgbm.joblib"

OUT_UBIGEO_PARQUET = PROCESSED_DIR / "territorial_multihazard_ubigeo.parquet"
OUT_H3_PARQUET = PROCESSED_DIR / "territorial_multihazard_h3.parquet"
METRICS_JSON = INTERIM_DIR / "phase4_integration_metrics.json"

EARTH_RADIUS_KM = 6371.0
DEG_LAT_TO_KM = 111.1949266
NAZCA_RIDGE_LAT = -15.0
NAZCA_TRENCH_BOUNDS = {"NZ\\SA", "NZ\\AP", "NZ\\ND"}
MW_SATURATION_CAP = 7.5

# 3 Design / Stress-Test Seismic Scenarios
SCENARIOS = [
    {
        "key": "lima_mw86",
        "label": "Laguna Central Lima-Callao-Chincha (Mw 8.6)",
        "mw": 8.6,
        "lat": -12.05,
        "lon": -77.80,
        "depth_km": 25.0,
        "regime": "subduccion_interplaca",
    },
    {
        "key": "sur_mw84",
        "label": "Sub-bloque Sur Ilo-Moquegua-Tacna (Mw 8.4)",
        "mw": 8.4,
        "lat": -17.65,
        "lon": -71.80,
        "depth_km": 28.0,
        "regime": "subduccion_interplaca",
    },
    {
        "key": "intraplaca_mw68",
        "label": "Intraplaca Intermedio Sierra Central / Junin (Mw 6.8)",
        "mw": 6.8,
        "lat": -11.20,
        "lon": -75.80,
        "depth_km": 120.0,
        "regime": "intramanto_intermedio",
    },
]


def _resolve_geojson_dir() -> Path:
    for p in (RAW_TECTONIC_ROOT / "tectonicplates-master" / "GeoJSON", RAW_TECTONIC_ROOT / "GeoJSON"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"PB2002 GeoJSON not found under {RAW_TECTONIC_ROOT}")


def _latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    c = np.cos(lat)
    return np.column_stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)))


def _spherical_dist_and_az(
    lat1_deg: float, lon1_deg: float, lat2_deg: np.ndarray, lon2_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """3D unit-sphere great-circle distance (km, R=6371.0) and forward azimuth [0, 360)."""
    lat1, lon1 = np.radians(lat1_deg), np.radians(lon1_deg)
    lat2, lon2 = np.radians(lat2_deg), np.radians(lon2_deg)
    x1, y1, z1 = np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)
    chord = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    dist_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))

    dlon = lon2 - lon1
    y_az = np.sin(dlon) * np.cos(lat2)
    x_az = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    az_deg = (np.degrees(np.arctan2(y_az, x_az)) + 360.0) % 360.0
    return dist_km, az_deg


def _predict_slab_depth(lat: np.ndarray, dist_trench_km: np.ndarray, model: dict[str, Any]) -> np.ndarray:
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


def _pga_gal_to_mmi(pga_gal: np.ndarray) -> np.ndarray:
    """Wald et al. (1999) USGS ShakeMap empirical PGA (cm/s^2 = Gal) to Instrumental MMI."""
    log_pga = np.log10(np.maximum(pga_gal, 1e-4))
    mmi_low = 2.20 * log_pga + 1.00
    mmi_high = 3.66 * log_pga - 1.66
    # Smooth crossover at log10(PGA) = 1.82 (~66 Gal, MMI ~ 5.0)
    mmi = np.where(mmi_low < 5.0, mmi_low, mmi_high)
    return np.clip(mmi, 1.0, 10.0).astype("float64")


def _build_trench_tree() -> tuple[cKDTree, np.ndarray, np.ndarray]:
    """Densify PB2002 Nazca trench steps (<= 2 km) and return 3D cKDTree + vel/age arrays."""
    geod = Geod(ellps="WGS84")
    gdf_steps = gpd.read_file(_resolve_geojson_dir() / "PB2002_steps.json")
    trench = gdf_steps[gdf_steps["PLATEBOUND"].isin(NAZCA_TRENCH_BOUNDS)]
    pts: list[tuple[float, float]] = []
    vels: list[float] = []
    ages: list[float] = []
    for _, r in trench.iterrows():
        coords = list(r.geometry.coords)
        for i in range(len(coords) - 1):
            lo1, la1 = coords[i][0], coords[i][1]
            lo2, la2 = coords[i + 1][0], coords[i + 1][1]
            _, _, dm = geod.inv(lo1, la1, lo2, la2)
            n_int = max(int(np.ceil(dm / 2000.0)) - 1, 0)
            seg_pts = [(lo1, la1)] + (geod.npts(lo1, la1, lo2, la2, n_int) if n_int > 0 else []) + [(lo2, la2)]
            for lo, la in seg_pts:
                pts.append((lo, la))
                vels.append(float(r["VELOCITYLE"]))
                ages.append(float(r["SEAFLOORAG"]))
    arr = np.asarray(pts, dtype="float64")
    tree = cKDTree(_latlon_to_unit_xyz(arr[:, 1], arr[:, 0]))
    return tree, np.asarray(vels, dtype="float64"), np.asarray(ages, dtype="float64")


def _fit_huber_magnitude_slope(aceldat_path: Path, h0_km: float) -> float:
    """Compute physical unstandardized Huber magnitude slope c1 (log10 PGA per Mw)."""
    ac = pd.read_parquet(aceldat_path)
    log10_r_hyp = np.log10(np.sqrt(ac["dist_hypocentral_km"] ** 2 + h0_km ** 2))
    X = np.column_stack(
        (
            ac["event_mag"].to_numpy(dtype="float64"),
            log10_r_hyp.to_numpy(dtype="float64"),
            ac["dist_hypocentral_km"].to_numpy(dtype="float64"),
            ac["event_depth_km"].to_numpy(dtype="float64"),
            (ac["regimen_sismotectonico"] == "subduccion_interplaca").to_numpy(dtype="float64"),
            (ac["regimen_sismotectonico"] == "cortical_continental").to_numpy(dtype="float64"),
            (ac["regimen_sismotectonico"] == "intramanto_intermedio").to_numpy(dtype="float64"),
        )
    )
    y = ac["log10_pga_h_geom"].to_numpy(dtype="float64")
    huber = HuberRegressor(max_iter=1000, alpha=1.0)
    huber.fit(X, y)
    c1 = float(huber.coef_[0])
    assert 0.35 <= c1 <= 0.85, f"Unphysical Huber magnitude slope c1={c1}"
    return c1


def _georeference_all_districts(
    df_dist: pd.DataFrame, df_ac: pd.DataFrame, df_pel: pd.DataFrame, df_zc: pd.DataFrame
) -> pd.DataFrame:
    """Assign representative (dist_lat, dist_lon) and H3 indices to 100% of 1,893 districts."""
    out = df_dist.copy()
    out["ubigeo"] = out["ubigeo"].astype(str).str.strip().str.zfill(6).astype("string")
    out["prov_code"] = out["ubigeo"].str.slice(0, 4)
    out["depa_code"] = out["ubigeo"].str.slice(0, 2)

    # 1. ACELDAT station median per UBIGEO
    ac_coords = df_ac.groupby("ubigeo", as_index=False).agg(
        ac_lat=("station_lat", "median"),
        ac_lon=("station_lon", "median"),
    )

    # 2. Combined GEOCATMIN points median per UBIGEO
    geo_pts = pd.concat(
        [
            df_pel[df_pel["ubigeo"].notna()][["ubigeo", "lat", "lon"]],
            df_zc[df_zc["ubigeo"].notna()][["ubigeo", "lat", "lon"]],
        ],
        ignore_index=True,
    )
    geo_coords = geo_pts.groupby("ubigeo", as_index=False).agg(
        geo_lat=("lat", "median"),
        geo_lon=("lon", "median"),
    )

    out = out.merge(ac_coords, on="ubigeo", how="left").merge(geo_coords, on="ubigeo", how="left")

    out["dist_lat"] = np.nan
    out["dist_lon"] = np.nan
    out["geo_source"] = pd.Series(pd.NA, index=out.index, dtype="string")

    m1 = out["ac_lat"].notna()
    out.loc[m1, "dist_lat"] = out.loc[m1, "ac_lat"]
    out.loc[m1, "dist_lon"] = out.loc[m1, "ac_lon"]
    out.loc[m1, "geo_source"] = "ACELDAT_station"

    m2 = out["dist_lat"].isna() & out["geo_lat"].notna()
    out.loc[m2, "dist_lat"] = out.loc[m2, "geo_lat"]
    out.loc[m2, "dist_lon"] = out.loc[m2, "geo_lon"]
    out.loc[m2, "geo_source"] = "GEOCATMIN_points"

    # 3. Province centroid fallback from already georeferenced districts
    prov_ref = out[out["dist_lat"].notna()].groupby("prov_code", as_index=False).agg(
        prov_lat=("dist_lat", "median"),
        prov_lon=("dist_lon", "median"),
    )
    out = out.merge(prov_ref, on="prov_code", how="left")
    m3 = out["dist_lat"].isna() & out["prov_lat"].notna()
    out.loc[m3, "dist_lat"] = out.loc[m3, "prov_lat"]
    out.loc[m3, "dist_lon"] = out.loc[m3, "prov_lon"]
    out.loc[m3, "geo_source"] = "PROVINCIA_centroid"

    # 4. Department centroid fallback
    depa_ref = out[out["dist_lat"].notna()].groupby("depa_code", as_index=False).agg(
        depa_lat=("dist_lat", "median"),
        depa_lon=("dist_lon", "median"),
    )
    out = out.merge(depa_ref, on="depa_code", how="left")
    m4 = out["dist_lat"].isna() & out["depa_lat"].notna()
    out.loc[m4, "dist_lat"] = out.loc[m4, "depa_lat"]
    out.loc[m4, "dist_lon"] = out.loc[m4, "depa_lon"]
    out.loc[m4, "geo_source"] = "DEPARTAMENTO_centroid"

    out = out.drop(columns=["ac_lat", "ac_lon", "geo_lat", "geo_lon", "prov_lat", "prov_lon", "depa_lat", "depa_lon", "prov_code", "depa_code"])

    # Assign H3 res 5, 6, 7
    lats = out["dist_lat"].to_numpy(dtype="float64")
    lons = out["dist_lon"].to_numpy(dtype="float64")
    out["h3_res5"] = pd.Series([h3.latlng_to_cell(float(la), float(lo), 5) for la, lo in zip(lats, lons)], dtype="string")
    out["h3_res6"] = pd.Series([h3.latlng_to_cell(float(la), float(lo), 6) for la, lo in zip(lats, lons)], dtype="string")
    out["h3_res7"] = pd.Series([h3.latlng_to_cell(float(la), float(lo), 7) for la, lo in zip(lats, lons)], dtype="string")

    assert len(out) == 1893
    assert out["dist_lat"].notna().all() and out["dist_lon"].notna().all() and out["h3_res5"].notna().all()
    assert out["dist_lat"].between(-18.6, 0.2).all() and out["dist_lon"].between(-81.6, -68.4).all()
    return out


def _simulate_hybrid_scenario_pga(
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    target_dist_trench_km: np.ndarray,
    target_dist_lat_15s_km: np.ndarray,
    target_geocatmin_df: pd.DataFrame,
    target_delta_s2s: np.ndarray,
    scenario: dict[str, Any],
    gmpe_bundle: dict[str, Any],
    c1_huber: float,
    trench_tree: cKDTree,
    trench_vels: np.ndarray,
    trench_ages: np.ndarray,
    slab_model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run hybrid GMPE (LightGBM capped at Mw 7.5 + Huber extrapolation > 7.5 + site delta_S2S)."""
    ml_cols: list[str] = gmpe_bundle["ml_features_no_site"]
    model_no_site = gmpe_bundle["model_no_site"]
    h0_km = float(gmpe_bundle["h0_saturation_km"])

    ev_lat = float(scenario["lat"])
    ev_lon = float(scenario["lon"])
    ev_depth = float(scenario["depth_km"])
    ev_mw = float(scenario["mw"])
    ev_regime = str(scenario["regime"])

    # Event tectonic properties
    ev_xyz = _latlon_to_unit_xyz(np.array([ev_lat]), np.array([ev_lon]))
    d_ch, idx_t = trench_tree.query(ev_xyz, k=1)
    ev_dist_trench = float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(d_ch[0] / 2.0, 0.0, 1.0)))
    ev_vel = float(trench_vels[idx_t[0]])
    ev_age = float(trench_ages[idx_t[0]])
    ev_slab_z = float(_predict_slab_depth(np.array([ev_lat]), np.array([ev_dist_trench]), slab_model)[0])
    ev_dist_slab = abs(ev_depth - ev_slab_z)
    ev_dist_15s = (ev_lat - NAZCA_RIDGE_LAT) * DEG_LAT_TO_KM
    ev_is_sur = 1.0 if ev_lat <= NAZCA_RIDGE_LAT else 0.0

    # Path distances & azimuths from focus to every target point
    dist_epi, az_deg = _spherical_dist_and_az(ev_lat, ev_lon, target_lats, target_lons)
    dist_hyp = np.sqrt(dist_epi ** 2 + ev_depth ** 2)

    n = len(target_lats)
    syn = pd.DataFrame(0.0, index=range(n), columns=ml_cols)
    syn["event_mag"] = min(ev_mw, MW_SATURATION_CAP)
    syn["event_depth_km"] = ev_depth
    syn["dist_epicentral_km"] = dist_epi
    syn["dist_hypocentral_km"] = dist_hyp
    syn["log10_r_hyp"] = np.log10(np.sqrt(dist_hyp ** 2 + h0_km ** 2))
    syn["log10_r_epi"] = np.log10(np.sqrt(dist_epi ** 2 + h0_km ** 2))
    syn["azimuth_deg"] = az_deg
    syn["sin_azimuth"] = np.sin(np.radians(az_deg))
    syn["cos_azimuth"] = np.cos(np.radians(az_deg))
    syn["event_dist_trench_km"] = ev_dist_trench
    syn["station_dist_trench_km"] = target_dist_trench_km
    syn["event_slab_depth_est_km"] = ev_slab_z
    syn["event_dist_to_slab_km"] = ev_dist_slab
    syn["dist_lat_15s_km"] = ev_dist_15s
    syn["station_dist_lat_15s_km"] = target_dist_lat_15s_km
    syn["convergence_vel_mm_yr"] = ev_vel
    syn["seafloor_age_myr"] = ev_age
    syn["is_normal_sur"] = ev_is_sur
    syn[f"reg_{ev_regime}"] = 1.0

    for gcol in ("n_peligros_total", "n_peligros_flujo", "n_peligros_caida", "n_peligros_deslizamiento", "n_zonas_criticas_total"):
        if gcol in target_geocatmin_df.columns:
            syn[gcol] = target_geocatmin_df[gcol].to_numpy(dtype="float64")

    # Hybrid extrapolation above Mw 7.5 using Huber physical magnitude slope + nonlinear site damping
    geo_dict = {
        gcol: target_geocatmin_df[gcol].to_numpy(dtype="float64")
        for gcol in ("n_peligros_total", "n_peligros_flujo", "n_peligros_caida", "n_peligros_deslizamiento", "n_zonas_criticas_total")
        if gcol in target_geocatmin_df.columns
    }
    res = simulate_hybrid_nonlinear_pga(
        target_lats=target_lats,
        target_lons=target_lons,
        target_dist_trench_km=target_dist_trench_km,
        target_dist_lat_15s_km=target_dist_lat_15s_km,
        target_geocatmin_matrix=geo_dict,
        target_delta_s2s=target_delta_s2s,
        ev_lat=float(scenario["lat"]),
        ev_lon=float(scenario["lon"]),
        ev_depth_km=float(scenario["depth_km"]),
        ev_mw=float(scenario["mw"]),
        ev_regime=str(scenario["regime"]),
        gmpe_bundle=gmpe_bundle,
        c1_huber=c1_huber,
        trench_tree=trench_tree,
        trench_vels=trench_vels,
        trench_ages=trench_ages,
        slab_model=slab_model,
        use_nonlinear_damping=True,
    )
    return res["pga_site_gal"], res["pga_rock_gal"], res["mmi_site"]


def _normalize_0_100(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(arr, dtype="float64")
    return np.clip((arr - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def main() -> None:
    """Execute Phase 4 multi-hazard territorial integration onto UBIGEO and H3."""
    for p in (
        DISTRITOS_PARQUET,
        ACELDAT_PARQUET,
        SITE_FACTORS_PARQUET,
        GAPS_PROB_PARQUET,
        FORECAST_H3_PARQUET,
        PELIGROS_PARQUET,
        ZONAS_PARQUET,
        SLAB_MODEL_JSON,
        GMPE_JOBLIB,
    ):
        assert p.exists(), f"Missing prerequisite: {p}"

    df_dist_raw = pd.read_parquet(DISTRITOS_PARQUET)
    df_ac = pd.read_parquet(ACELDAT_PARQUET)
    df_site = pd.read_parquet(SITE_FACTORS_PARQUET)
    df_gaps_prob = pd.read_parquet(GAPS_PROB_PARQUET)
    df_fc_h3 = pd.read_parquet(FORECAST_H3_PARQUET)
    df_pel = pd.read_parquet(PELIGROS_PARQUET, columns=["ubigeo", "lat", "lon"])
    df_zc = pd.read_parquet(ZONAS_PARQUET, columns=["ubigeo", "lat", "lon"])
    slab_model = json.loads(SLAB_MODEL_JSON.read_text(encoding="utf-8"))["model"]
    gmpe_bundle = joblib.load(GMPE_JOBLIB)

    trench_tree, trench_vels, trench_ages = _build_trench_tree()
    c1_huber = _fit_huber_magnitude_slope(ACELDAT_PARQUET, float(gmpe_bundle["h0_saturation_km"]))

    # 1. Georeference all 1,893 districts
    df_u = _georeference_all_districts(df_dist_raw, df_ac, df_pel, df_zc)

    # 2. Exact 3D trench distance & tectonic context for all 1,893 districts
    u_xyz = _latlon_to_unit_xyz(df_u["dist_lat"].to_numpy(), df_u["dist_lon"].to_numpy())
    d_ch, idx_t = trench_tree.query(u_xyz, k=1)
    df_u["dist_trench_km"] = (2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(d_ch / 2.0, 0.0, 1.0))).astype("float64")
    df_u["convergence_vel_mm_yr"] = trench_vels[idx_t]
    df_u["seafloor_age_myr"] = trench_ages[idx_t]
    df_u["dist_lat_15s_km"] = (df_u["dist_lat"] - NAZCA_RIDGE_LAT) * DEG_LAT_TO_KM
    df_u["regimen_subduccion_lat"] = pd.Series(
        np.where(df_u["dist_lat"] > NAZCA_RIDGE_LAT, "flat_slab_norte_centro", "normal_sur"),
        dtype="string",
    )
    df_u["slab_depth_est_km"] = _predict_slab_depth(
        df_u["dist_lat"].to_numpy(), df_u["dist_trench_km"].to_numpy(), slab_model
    )

    # 3. Couple Model C (H3 forecast + 1-ring neighborhood max)
    fc_lookup = df_fc_h3.set_index("h3_res5")[["prob_m45_90d", "prob_m45_365d", "prob_m50_90d"]].to_dict("index")
    fc_tree = cKDTree(_latlon_to_unit_xyz(df_fc_h3["cell_lat"].to_numpy(), df_fc_h3["cell_lon"].to_numpy()))
    fc_cells_arr = df_fc_h3["h3_res5"].to_numpy()

    p90_cell, p365_cell, pm50_cell = [], [], []
    p90_nbhd, p365_nbhd, pm50_nbhd = [], [], []
    for i, cell in enumerate(df_u["h3_res5"].tolist()):
        info = fc_lookup.get(cell)
        if info is None:
            # Nearest H3 cell fallback (e.g., extreme border)
            _, nn_i = fc_tree.query(u_xyz[i], k=1)
            cell_ref = str(fc_cells_arr[int(nn_i)])
            info = fc_lookup[cell_ref]
        else:
            cell_ref = cell
        p90_cell.append(float(info["prob_m45_90d"]))
        p365_cell.append(float(info["prob_m45_365d"]))
        pm50_cell.append(float(info["prob_m50_90d"]))

        nb_infos = [fc_lookup[nb] for nb in h3.grid_disk(cell_ref, 1) if nb in fc_lookup]
        p90_nbhd.append(max(float(x["prob_m45_90d"]) for x in nb_infos))
        p365_nbhd.append(max(float(x["prob_m45_365d"]) for x in nb_infos))
        pm50_nbhd.append(max(float(x["prob_m50_90d"]) for x in nb_infos))

    df_u["prob_m45_90d"] = np.asarray(p90_cell, dtype="float64")
    df_u["prob_m45_365d"] = np.asarray(p365_cell, dtype="float64")
    df_u["prob_m50_90d"] = np.asarray(pm50_cell, dtype="float64")
    df_u["prob_m45_90d_nbhd_max"] = np.asarray(p90_nbhd, dtype="float64")
    df_u["prob_m45_365d_nbhd_max"] = np.asarray(p365_nbhd, dtype="float64")
    df_u["prob_m50_90d_nbhd_max"] = np.asarray(pm50_nbhd, dtype="float64")

    # 4. Couple Model B (Probabilistic Seismic Gaps + smooth inland attenuation > 200 km)
    segs = df_gaps_prob[df_gaps_prob["entity_type"] == "segment"].reset_index(drop=True)
    # Match each district to nearest coastal gap segment by latitude projection onto coastal strip
    seg_lats = segs["lat_center"].to_numpy(dtype="float64")
    nearest_seg_idx = np.argmin(np.abs(df_u["dist_lat"].to_numpy()[:, None] - seg_lats[None, :]), axis=1)

    df_u["nearest_gap_segment_id"] = segs["entity_id"].to_numpy()[nearest_seg_idx].astype(str)
    df_u["nearest_gap_zone"] = segs["coastal_zone"].to_numpy()[nearest_seg_idx].astype(str)
    df_u["gap_mw_pot_p50"] = segs["mw_pot_p50"].to_numpy(dtype="float64")[nearest_seg_idx]
    df_u["gap_mw_pot_p90"] = segs["mw_pot_p90"].to_numpy(dtype="float64")[nearest_seg_idx]
    df_u["gap_bpt_p30_a05_raw"] = segs["bpt_p30_a05"].to_numpy(dtype="float64")[nearest_seg_idx]
    df_u["gap_bpt_p50_a05_raw"] = segs["bpt_p50_a05"].to_numpy(dtype="float64")[nearest_seg_idx]

    coastal_atten = np.where(df_u["dist_trench_km"] <= 200.0, 1.0, np.exp(-(df_u["dist_trench_km"] - 200.0) / 180.0))
    df_u["gap_coastal_atten_factor"] = coastal_atten.astype("float64")
    df_u["gap_bpt_p30_a05_eff"] = (df_u["gap_bpt_p30_a05_raw"] * coastal_atten).astype("float64")

    # 5. Couple Model A (Station site factors per UBIGEO + 3 Hybrid ShakeMap Scenarios)
    df_site_sorted = df_site.sort_values("n_records", ascending=False)
    site_u = (
        df_site_sorted.groupby("ubigeo", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "primary_station_code": str(g["cod_estacion"].iloc[0]),
                    "n_aceldat_stations_dist": int(len(g)),
                    "n_aceldat_records_dist": int(g["n_records"].sum()),
                    "site_delta_s2s": float(np.average(g["delta_s2s_log10"], weights=g["n_records"])),
                    "site_delta_s2s_rel_msol": float(np.average(g["delta_s2s_rel_msol"], weights=g["n_records"])),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    site_u["site_amp_factor_vs_msol"] = np.power(10.0, site_u["site_delta_s2s_rel_msol"])

    df_u = df_u.merge(site_u, on="ubigeo", how="left")
    df_u["has_aceldat_station"] = df_u["primary_station_code"].notna()
    df_u["n_aceldat_stations_dist"] = df_u["n_aceldat_stations_dist"].fillna(0).astype("int32")
    df_u["n_aceldat_records_dist"] = df_u["n_aceldat_records_dist"].fillna(0).astype("int32")
    df_u["site_delta_s2s"] = df_u["site_delta_s2s"].fillna(0.0).astype("float64")
    df_u["site_delta_s2s_rel_msol"] = df_u["site_delta_s2s_rel_msol"].fillna(0.0).astype("float64")
    df_u["site_amp_factor_vs_msol"] = df_u["site_amp_factor_vs_msol"].fillna(1.0).astype("float64")
    df_u["primary_station_code"] = df_u["primary_station_code"].astype("string")

    for sc in SCENARIOS:
        pga_site, pga_rock, mmi_site = _simulate_hybrid_scenario_pga(
            target_lats=df_u["dist_lat"].to_numpy(),
            target_lons=df_u["dist_lon"].to_numpy(),
            target_dist_trench_km=df_u["dist_trench_km"].to_numpy(),
            target_dist_lat_15s_km=df_u["dist_lat_15s_km"].to_numpy(),
            target_geocatmin_df=df_u,
            target_delta_s2s=df_u["site_delta_s2s"].to_numpy(),
            scenario=sc,
            gmpe_bundle=gmpe_bundle,
            c1_huber=c1_huber,
            trench_tree=trench_tree,
            trench_vels=trench_vels,
            trench_ages=trench_ages,
            slab_model=slab_model,
        )
        key = sc["key"]
        df_u[f"pga_escenario_{key}_gal"] = np.round(pga_site, 2)
        df_u[f"pga_escenario_{key}_rock_gal"] = np.round(pga_rock, 2)
        df_u[f"mmi_escenario_{key}"] = np.round(mmi_site, 2)

    df_u["pga_escenario_max_gal"] = np.maximum.reduce(
        [
            df_u["pga_escenario_lima_mw86_gal"].to_numpy(),
            df_u["pga_escenario_sur_mw84_gal"].to_numpy(),
            df_u["pga_escenario_intraplaca_mw68_gal"].to_numpy(),
        ]
    )
    df_u["mmi_escenario_max"] = np.round(_pga_gal_to_mmi(df_u["pga_escenario_max_gal"].to_numpy()), 2)

    # 6. Composite Pure Seismic Hazard Index (0-100): 45% Max PGA + 30% BPT Gap + 25% H3 Annual Rate
    comp_pga = _normalize_0_100(np.log10(np.maximum(df_u["pga_escenario_max_gal"].to_numpy(), 0.1)))
    comp_gap = _normalize_0_100(df_u["gap_bpt_p30_a05_eff"].to_numpy())
    comp_rate = _normalize_0_100(np.log10(1e-4 + df_u["prob_m45_365d_nbhd_max"].to_numpy()))

    df_u["idx_comp_pga_0_100"] = np.round(comp_pga, 2)
    df_u["idx_comp_gap_0_100"] = np.round(comp_gap, 2)
    df_u["idx_comp_rate_0_100"] = np.round(comp_rate, 2)
    df_u["seismic_hazard_index_0_100"] = np.round(0.45 * comp_pga + 0.30 * comp_gap + 0.25 * comp_rate, 2)

    bins = [-0.01, 20.0, 40.0, 60.0, 80.0, 100.01]
    labels = ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]
    df_u["nivel_peligro_sismico_puro"] = pd.cut(df_u["seismic_hazard_index_0_100"], bins=bins, labels=labels).astype("string")

    # 7. Extensible Multi-Hazard Contract Columns (ready for ANA/SENAMHI, FIRMS, INEI)
    for ext_col in ("ext_hydromet_flood_idx", "ext_wildfire_firms_idx", "ext_social_vuln_idx", "ext_composite_multirisk_idx"):
        df_u[ext_col] = np.nan

    # Final assertions for UBIGEO table
    assert len(df_u) == 1893 and df_u["ubigeo"].is_unique
    assert (df_u["ubigeo"].str.len() == 6).all()
    assert df_u[["dist_lat", "dist_lon", "h3_res5", "seismic_hazard_index_0_100", "pga_escenario_max_gal"]].notna().all().all()
    assert df_u["seismic_hazard_index_0_100"].between(0.0, 100.0).all()

    df_u.to_parquet(OUT_UBIGEO_PARQUET, engine="pyarrow", compression="snappy", index=False)

    # 8. Build Territorial Multi-Hazard H3 Table (`8,205` cells)
    df_h3 = df_fc_h3.copy()
    h3_seg_idx = np.argmin(np.abs(df_h3["cell_lat"].to_numpy()[:, None] - seg_lats[None, :]), axis=1)
    df_h3["nearest_gap_segment_id"] = segs["entity_id"].to_numpy()[h3_seg_idx].astype(str)
    df_h3["gap_mw_pot_p50"] = segs["mw_pot_p50"].to_numpy(dtype="float64")[h3_seg_idx]
    df_h3["gap_bpt_p30_a05_raw"] = segs["bpt_p30_a05"].to_numpy(dtype="float64")[h3_seg_idx]
    h3_atten = np.where(df_h3["dist_trench_km"] <= 200.0, 1.0, np.exp(-(df_h3["dist_trench_km"] - 200.0) / 180.0))
    df_h3["gap_bpt_p30_a05_eff"] = (df_h3["gap_bpt_p30_a05_raw"] * h3_atten).astype("float64")

    # Simulate neutral-site scenario PGA across all 8,205 H3 cells
    h3_geo_proxy = pd.DataFrame(
        {
            "n_peligros_total": df_h3["n_geocatmin_peligros"],
            "n_peligros_flujo": df_h3["n_geocatmin_peligros"] * 0.26,
            "n_peligros_caida": df_h3["n_geocatmin_peligros"] * 0.25,
            "n_peligros_deslizamiento": df_h3["n_geocatmin_peligros"] * 0.20,
            "n_zonas_criticas_total": df_h3["n_geocatmin_zonas_criticas"],
        }
    )
    h3_dist_15s = (df_h3["cell_lat"].to_numpy(dtype="float64") - NAZCA_RIDGE_LAT) * DEG_LAT_TO_KM
    zero_site = np.zeros(len(df_h3), dtype="float64")
    for sc in SCENARIOS:
        pga_h3, _, _ = _simulate_hybrid_scenario_pga(
            target_lats=df_h3["cell_lat"].to_numpy(dtype="float64"),
            target_lons=df_h3["cell_lon"].to_numpy(dtype="float64"),
            target_dist_trench_km=df_h3["dist_trench_km"].to_numpy(dtype="float64"),
            target_dist_lat_15s_km=h3_dist_15s,
            target_geocatmin_df=h3_geo_proxy,
            target_delta_s2s=zero_site,
            scenario=sc,
            gmpe_bundle=gmpe_bundle,
            c1_huber=c1_huber,
            trench_tree=trench_tree,
            trench_vels=trench_vels,
            trench_ages=trench_ages,
            slab_model=slab_model,
        )
        df_h3[f"pga_escenario_{sc['key']}_gal"] = np.round(pga_h3, 2)

    df_h3["pga_escenario_max_gal"] = np.maximum.reduce(
        [
            df_h3["pga_escenario_lima_mw86_gal"].to_numpy(),
            df_h3["pga_escenario_sur_mw84_gal"].to_numpy(),
            df_h3["pga_escenario_intraplaca_mw68_gal"].to_numpy(),
        ]
    )
    h3_comp_pga = _normalize_0_100(np.log10(np.maximum(df_h3["pga_escenario_max_gal"].to_numpy(), 0.1)))
    h3_comp_gap = _normalize_0_100(df_h3["gap_bpt_p30_a05_eff"].to_numpy())
    h3_comp_rate = _normalize_0_100(np.log10(1e-4 + df_h3["prob_m45_365d"].to_numpy()))
    df_h3["seismic_hazard_index_0_100"] = np.round(0.45 * h3_comp_pga + 0.30 * h3_comp_gap + 0.25 * h3_comp_rate, 2)

    for ext_col in ("ext_hydromet_flood_idx", "ext_wildfire_firms_idx", "ext_social_vuln_idx", "ext_composite_multirisk_idx"):
        df_h3[ext_col] = np.nan

    assert len(df_h3) == 8205 and df_h3["h3_res5"].is_unique
    assert df_h3["seismic_hazard_index_0_100"].notna().all()
    df_h3.to_parquet(OUT_H3_PARQUET, engine="pyarrow", compression="snappy", index=False)

    # 9. Audit Discrepancy vs FONDES SIGRID & Lima Summary
    crosstab = pd.crosstab(df_u["nivel_peligro_sismico_puro"], df_u["nivel_riesgo_multipeligro_sigrid"])
    high_seis = df_u["nivel_peligro_sismico_puro"].isin(["Alto", "Muy Alto"])
    low_fondes = df_u["nivel_riesgo_multipeligro_sigrid"].isin(["Muy Bajo", "Bajo"])
    underestimated = df_u[high_seis & low_fondes]

    lima_callao = df_u[df_u["provincia"].isin(["LIMA", "CALLAO"])].sort_values("pga_escenario_lima_mw86_gal", ascending=False)
    lima_under = lima_callao[
        lima_callao["nivel_peligro_sismico_puro"].isin(["Alto", "Muy Alto"])
        & lima_callao["nivel_riesgo_multipeligro_sigrid"].isin(["Muy Bajo", "Bajo"])
    ]

    metrics = {
        "c1_huber_mag_slope": round(c1_huber, 4),
        "ubigeo_rows": int(len(df_u)),
        "ubigeo_cols": int(len(df_u.columns)),
        "ubigeo_size_mb": round(OUT_UBIGEO_PARQUET.stat().st_size / (1024 * 1024), 4),
        "h3_rows": int(len(df_h3)),
        "h3_cols": int(len(df_h3.columns)),
        "h3_size_mb": round(OUT_H3_PARQUET.stat().st_size / (1024 * 1024), 4),
        "georef_source_counts": df_u["geo_source"].value_counts().to_dict(),
        "districts_with_aceldat_site_factor": int(df_u["has_aceldat_station"].sum()),
        "pure_seismic_distribution": df_u["nivel_peligro_sismico_puro"].value_counts().to_dict(),
        "fondes_sigrid_distribution": df_u["nivel_riesgo_multipeligro_sigrid"].value_counts().to_dict(),
        "crosstab_pure_vs_fondes": crosstab.to_dict(),
        "underestimated_national_count": int(len(underestimated)),
        "underestimated_national_of_high_seis_total": int(high_seis.sum()),
        "underestimated_national_pct": round(float(len(underestimated) / max(int(high_seis.sum()), 1) * 100.0), 2),
        "lima_callao_total_districts": int(len(lima_callao)),
        "lima_callao_high_seismic_count": int(lima_callao["nivel_peligro_sismico_puro"].isin(["Alto", "Muy Alto"]).sum()),
        "lima_callao_underestimated_in_fondes_count": int(len(lima_under)),
        "top15_national": df_u.sort_values("seismic_hazard_index_0_100", ascending=False).head(15)[
            [
                "ubigeo",
                "departamento",
                "provincia",
                "distrito",
                "seismic_hazard_index_0_100",
                "nivel_peligro_sismico_puro",
                "nivel_riesgo_multipeligro_sigrid",
                "pga_escenario_max_gal",
                "mmi_escenario_max",
                "site_amp_factor_vs_msol",
                "gap_bpt_p30_a05_eff",
                "prob_m45_365d_nbhd_max",
            ]
        ].to_dict("records"),
        "lima_callao_top20": lima_callao.head(20)[
            [
                "ubigeo",
                "provincia",
                "distrito",
                "primary_station_code",
                "site_delta_s2s",
                "site_amp_factor_vs_msol",
                "pga_escenario_lima_mw86_rock_gal",
                "pga_escenario_lima_mw86_gal",
                "mmi_escenario_lima_mw86",
                "seismic_hazard_index_0_100",
                "nivel_peligro_sismico_puro",
                "nivel_riesgo_multipeligro_sigrid",
            ]
        ].to_dict("records"),
        "lima_key_benchmarks": lima_callao[
            lima_callao["distrito"].isin(["PUENTE PIEDRA", "CALLAO", "LA MOLINA", "LOS OLIVOS", "LURIN", "VENTANILLA", "CHORRILLOS", "MIRAFLORES", "LIMA", "SAN JUAN DE LURIGANCHO"])
        ][
            [
                "ubigeo",
                "provincia",
                "distrito",
                "primary_station_code",
                "site_delta_s2s",
                "site_amp_factor_vs_msol",
                "pga_escenario_lima_mw86_rock_gal",
                "pga_escenario_lima_mw86_gal",
                "mmi_escenario_lima_mw86",
                "nivel_peligro_sismico_puro",
                "nivel_riesgo_multipeligro_sigrid",
            ]
        ].to_dict("records"),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== FASE 4.1: INTEGRACION TERRITORIAL MULTI-AMENAZA COMPLETADA ===")
    print(f"Pendiente fisica Huber c1_mag: {c1_huber:.4f} log10(PGA)/Mw (extrapolacion hibrida > Mw 7.5)")
    print(f"UBIGEO maestro: {len(df_u)} filas x {len(df_u.columns)} cols ({metrics['ubigeo_size_mb']} MB) | georref={metrics['georef_source_counts']}")
    print(f"H3 maestro:     {len(df_h3)} filas x {len(df_h3.columns)} cols ({metrics['h3_size_mb']} MB)")
    print(f"Discrepancia FONDES: {len(underestimated)}/{int(high_seis.sum())} distritos de Peligro Sismico Alto/Muy Alto ({metrics['underestimated_national_pct']}%) figuran como Muy Bajo/Bajo en FONDES SIGRID")
    print(f"Lima+Callao: {metrics['lima_callao_underestimated_in_fondes_count']}/{metrics['lima_callao_high_seismic_count']} distritos de Alto/Muy Alto peligro sismico subestimados en FONDES")
    print("Benchmarks Lima/Callao bajo Escenario Laguna Central Mw 8.6:")
    for r in metrics["lima_key_benchmarks"]:
        st = r["primary_station_code"] if pd.notna(r["primary_station_code"]) else "-"
        print(
            f"  {r['distrito']:22s} ({st:5s}) | Amp={r['site_amp_factor_vs_msol']:.2f}x | PGA={r['pga_escenario_lima_mw86_gal']:6.1f} Gal (MMI {r['mmi_escenario_lima_mw86']:.1f}) | Sism={r['nivel_peligro_sismico_puro']:8s} vs FONDES={r['nivel_riesgo_multipeligro_sigrid']}"
        )


if __name__ == "__main__":
    main()
