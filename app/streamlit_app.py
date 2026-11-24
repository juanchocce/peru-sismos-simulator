"""Phase 5D: CivicTech / GovTech Interactive Seismic & Multi-Hazard Platform for Peru (Streamlit).

Self-contained Python + Streamlit web application ("12-year-old clarity test" + scientific rigor):
  - Pedagogically ordered 9-section vertical sidebar navigation (from Seismic School to Technical Docs).
  - Real Dual Light (`Claro Editorial`) / Dark (`Oscuro Nocturno`) visual theme switcher.
  - Fullscreen button + visual legend panel on EVERY Folium and Plotly map (including 3D Peru contour).
  - Dual Magnitude (Mw) + Mercalli scale everywhere, plain-Spanish `Gal`/`%g` & linear vs. nonlinear soil notes.
  - 100%-centered responsive tables (`render_centered_table`) + mandatory explanatory legend below every table.
  - Dedicated Lowest Seismic Hazard Zones territorial analysis (`🛡️ Zonas de Menor Peligro Sísmico`).

Reads exclusively from `data/processed/`, `data/interim/`, `models/`, and `docs/`
(`base_informativa/` remains strictly read-only and untouched).

Launch:
    python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import h3
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Fullscreen, MarkerCluster
from pyproj import Geod
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_theme import (
    MACRO_BLOCK_LABELS_ES,
    MERCALLI_DESC_ES,
    MERCALLI_TO_MAG_RANGE_ES,
    REGIME_LABELS_ES,
    REGIME_SHORT_ES,
    RISK_LEVEL_COLORS,
    THEME_DARK,
    THEME_LIGHT,
    describe_mercalli,
    describe_mercalli_dual,
    folium_tiles_for_theme,
    format_mercalli_with_mag,
    inject_custom_css,
    plotly_theme_kwargs,
    render_badge,
    render_centered_table,
    render_kpi_card,
    render_map_legend,
    render_table_legend,
    translate_feature,
)
from src.models.nonlinear_site import (
    apply_nonlinear_site_damping,
    latlon_to_unit_xyz,
    simulate_hybrid_nonlinear_pga,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
RAW_TECTONIC_ROOT = PROJECT_ROOT / "data" / "raw" / "tectonic_plates"

UBIGEO_PARQUET = PROCESSED_DIR / "territorial_multihazard_ubigeo.parquet"
H3_PARQUET = PROCESSED_DIR / "territorial_multihazard_h3.parquet"
GAPS_PROB_PARQUET = PROCESSED_DIR / "seismic_gaps_probabilistic.parquet"
GAPS_RAW_PARQUET = PROCESSED_DIR / "coastal_seismic_gaps.parquet"
SITE_FACTORS_PARQUET = PROCESSED_DIR / "aceldat_station_site_factors.parquet"
CAT_INST_PARQUET = PROCESSED_DIR / "catalog_igp_instrumental.parquet"
CAT_HIST_PARQUET = PROCESSED_DIR / "catalog_igp_historico.parquet"
ZONAS_PARQUET = INTERIM_DIR / "geocatmin_zonas_criticas.parquet"
GMPE_JOBLIB = MODELS_DIR / "gmpe_lgbm.joblib"
SLAB_MODEL_JSON = INTERIM_DIR / "wadati_benioff_slab_model.json"
PHASE4_METRICS_JSON = INTERIM_DIR / "phase4_integration_metrics.json"
H3_METRICS_JSON = INTERIM_DIR / "h3_forecaster_metrics.json"

INEI_DEPT_GEOJSON = INTERIM_DIR / "inei_departamentos.geojson"
INEI_DIST_GEOJSON = INTERIM_DIR / "inei_distritos.geojson"
DOCS_COMPLETE_MD = DOCS_DIR / "DOCUMENTACION_TECNICA_COMPLETA.md"

NAZCA_TRENCH_BOUNDS = {"NZ\\SA", "NZ\\AP", "NZ\\ND"}
PLOTLY_CFG = {"displayModeBar": True, "displaylogo": False}

# Pedagogically ordered from easiest citizen guide to deepest technical rigor (9 sections)
NAV_SECTIONS = [
    "🎓 Escuela Sísmica (Empieza Aquí — Guía Fácil)",
    "🗺️ Mapa Nacional Multi-Amenaza",
    "🏘️ Consultor de Mi Distrito (Probabilidad y Riesgo)",
    "⚡ Simulador de Terremotos en Vivo",
    "📜 Historia Sísmica (1552–2026) y Corte 3D",
    "🌊 Radar de Silencios Sísmicos en la Costa",
    "🛡️ Zonas de Menor Peligro Sísmico (Análisis Territorial)",
    "🏛️ Auditoría vs. FONDES / SIGRID",
    "📚 Documentación Técnica y Rigor Estadístico",
]

PRESET_SCENARIOS = {
    "Megaterremoto frente a Lima–Callao (M 8.6 — Brecha de 1746)": {
        "lat": -12.05,
        "lon": -77.80,
        "depth": 25.0,
        "mw": 8.6,
    },
    "Terremoto en el Sur Ilo–Tacna (M 8.4 — Brecha de 1868)": {
        "lat": -17.65,
        "lon": -71.80,
        "depth": 28.0,
        "mw": 8.4,
    },
    "Sismo profundo en Sierra Central (M 6.8 — Estilo Junín)": {
        "lat": -11.20,
        "lon": -75.80,
        "depth": 120.0,
        "mw": 6.8,
    },
    "Terremoto Histórico de Pisco 2007 (M 8.0)": {
        "lat": -13.55,
        "lon": -76.86,
        "depth": 40.0,
        "mw": 8.0,
    },
}

HISTORICAL_MILESTONES = [
    (1746, 8.8, "1746 Lima–Callao (M 8.8)"),
    (1868, 8.8, "1868 Arica–Tacna (M 8.8)"),
    (1940, 7.9, "1940 Lima–Huacho (M 7.9)"),
    (1966, 8.1, "1966 Huacho–Lima (M 8.1)"),
    (1970, 7.8, "1970 Áncash / Yungay (M 7.8)"),
    (1974, 8.1, "1974 Lima–Cañete (M 8.1)"),
    (2001, 8.4, "2001 Arequipa–Atico (M 8.4)"),
    (2007, 8.0, "2007 Pisco–Ica (M 8.0)"),
    (2019, 8.0, "2019 Lagunas, Loreto (M 8.0)"),
]

SCHOOL_MAJOR_QUAKES = [
    {"Año": 1746, "Zona": "Lima – Callao", "Magnitud (Mw)": "M 8.8", "Tiempo al Siguiente": "—", "Lección Ciudadana": "Mayor megaterremoto de la costa central; generó tsunami destructivo en el Callao."},
    {"Año": 1868, "Zona": "Arica – Tacna – Moquegua", "Magnitud (Mw)": "M 8.8", "Tiempo al Siguiente": "122 años", "Lección Ciudadana": "Gran ruptura del extremo sur; desde entonces ese tramo sigue acumulando energía."},
    {"Año": 1940, "Zona": "Lima – Huacho", "Magnitud (Mw)": "M 7.9", "Tiempo al Siguiente": "72 años", "Lección Ciudadana": "Mostró que los suelos blandos sacuden mucho más fuerte que la roca firme."},
    {"Año": 1966, "Zona": "Huacho – Barranca", "Magnitud (Mw)": "M 8.1", "Tiempo al Siguiente": "26 años", "Lección Ciudadana": "Rompió al norte de Lima pero dejó trabada la brecha frente al Callao y Chincha."},
    {"Año": 1970, "Zona": "Áncash (Chimbote / Yungay)", "Magnitud (Mw)": "M 7.8", "Tiempo al Siguiente": "4 años", "Lección Ciudadana": "Demostró cómo un sismo puede detonar aludes y deslizamientos gigantes (peligro compuesto)."},
    {"Año": 1974, "Zona": "Lima Sur – Cañete", "Magnitud (Mw)": "M 8.1", "Tiempo al Siguiente": "4 años", "Lección Ciudadana": "Ruptura parcial al sur de Lima; gran parte del silencio de 1746 sigue sin liberarse."},
    {"Año": 2001, "Zona": "Arequipa – Atico – Ocoña", "Magnitud (Mw)": "M 8.4", "Tiempo al Siguiente": "27 años", "Lección Ciudadana": "Liberó energía en Arequipa, pero se detuvo antes de Ilo y Tacna."},
    {"Año": 2007, "Zona": "Pisco – Chincha – Ica", "Magnitud (Mw)": "M 8.0", "Tiempo al Siguiente": "6 años", "Lección Ciudadana": "Evidenció la alta vulnerabilidad de las viviendas de adobe y autoconstrucción."},
]

MERCALLI_PGA_HINTS: dict[int, tuple[str, str]] = {
    1: ("< 0.1% g (< 1 Gal)", "Sin daños en viviendas"),
    2: ("0.1 – 0.3% g (1–3 Gal)", "Sin daños en viviendas"),
    3: ("0.3 – 1.4% g (3–14 Gal)", "Sin daños; vibración ligera"),
    4: ("1.4 – 3.9% g (14–39 Gal)", "Sin daños; suenan ventanas y platos"),
    5: ("3.9 – 9.2% g (39–90 Gal)", "Caen objetos pequeños; fisuras mínimas"),
    6: ("9.2 – 18% g (90–177 Gal)", "Grietas en tarrajeo y muros antiguos débiles"),
    7: ("18 – 34% g (177–334 Gal)", "Daños fuertes en adobe y muros sin columnas"),
    8: ("34 – 65% g (334–638 Gal)", "Colapso parcial en adobe; daño moderado en concreto"),
    9: ("65 – 124% g (638–1216 Gal)", "Destrucción severa en construcciones vulnerables"),
    10: ("> 124% g (> 1216 Gal)", "Destrucción masiva, licuefacción y grandes derrumbes"),
}


def _resolve_geojson_dir() -> Path:
    for p in (RAW_TECTONIC_ROOT / "tectonicplates-master" / "GeoJSON", RAW_TECTONIC_ROOT / "GeoJSON"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"PB2002 GeoJSON not found under {RAW_TECTONIC_ROOT}")


def _add_fullscreen(m: folium.Map) -> None:
    """Attach mandatory Fullscreen button to every Folium map."""
    Fullscreen(
        position="topleft",
        title="Ver en pantalla completa",
        title_cancel="Salir de pantalla completa",
        force_separate_button=True,
    ).add_to(m)


def _extract_peru_outline_3d(dept_geojson: dict[str, Any] | None) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Extract lightweight (lon, lat, z=0) line segments from INEI departmental boundaries for 3D overlay."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    if not dept_geojson:
        return xs, ys, zs

    def _add_ring(ring: list[list[float]]) -> None:
        if len(ring) < 3:
            return
        step = max(1, len(ring) // 80)
        sampled = ring[::step]
        if sampled[-1] != ring[-1]:
            sampled.append(ring[-1])
        for lon, lat in sampled:
            xs.append(float(lon))
            ys.append(float(lat))
            zs.append(0.0)
        xs.append(None)
        ys.append(None)
        zs.append(None)

    for feat in dept_geojson.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "Polygon" and coords:
            _add_ring(coords[0])
        elif gtype == "MultiPolygon":
            for poly in coords:
                if poly:
                    _add_ring(poly[0])
    return xs, ys, zs


def load_simulation_assets() -> dict[str, Any]:
    """Load precomputed GMPE model, PB2002 trench tree, slab geometry, and district arrays."""
    df_u = pd.read_parquet(UBIGEO_PARQUET)
    df_u["ubigeo"] = df_u["ubigeo"].astype(str).str.zfill(6)
    gmpe_bundle = joblib.load(GMPE_JOBLIB)
    slab_model = json.loads(SLAB_MODEL_JSON.read_text(encoding="utf-8"))["model"]
    p4_metrics = json.loads(PHASE4_METRICS_JSON.read_text(encoding="utf-8"))
    c1_huber = float(p4_metrics["c1_huber_mag_slope"])

    geod = Geod(ellps="WGS84")
    gdf_steps = gpd.read_file(_resolve_geojson_dir() / "PB2002_steps.json")
    trench = gdf_steps[gdf_steps["PLATEBOUND"].isin(NAZCA_TRENCH_BOUNDS)]
    pts, vels, ages = [], [], []
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
    trench_tree = cKDTree(latlon_to_unit_xyz(arr[:, 1], arr[:, 0]))

    geo_matrix = {
        c: df_u[c].to_numpy(dtype="float64")
        for c in ("n_peligros_total", "n_peligros_flujo", "n_peligros_caida", "n_peligros_deslizamiento", "n_zonas_criticas_total")
    }

    return {
        "df_u": df_u,
        "gmpe_bundle": gmpe_bundle,
        "slab_model": slab_model,
        "c1_huber": c1_huber,
        "trench_tree": trench_tree,
        "trench_vels": np.asarray(vels, dtype="float64"),
        "trench_ages": np.asarray(ages, dtype="float64"),
        "target_lats": df_u["dist_lat"].to_numpy(dtype="float64"),
        "target_lons": df_u["dist_lon"].to_numpy(dtype="float64"),
        "target_dist_trench_km": df_u["dist_trench_km"].to_numpy(dtype="float64"),
        "target_dist_lat_15s_km": df_u["dist_lat_15s_km"].to_numpy(dtype="float64"),
        "target_delta_s2s": df_u["site_delta_s2s"].to_numpy(dtype="float64"),
        "geo_matrix": geo_matrix,
    }


def run_live_simulation(
    assets: dict[str, Any],
    ev_lat: float,
    ev_lon: float,
    ev_depth_km: float,
    ev_mw: float,
    use_nonlinear_damping: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run real-time hybrid ShakeMap simulation across all 1,893 districts (< 50 ms)."""
    t0 = time.perf_counter()
    sim = simulate_hybrid_nonlinear_pga(
        target_lats=assets["target_lats"],
        target_lons=assets["target_lons"],
        target_dist_trench_km=assets["target_dist_trench_km"],
        target_dist_lat_15s_km=assets["target_dist_lat_15s_km"],
        target_geocatmin_matrix=assets["geo_matrix"],
        target_delta_s2s=assets["target_delta_s2s"],
        ev_lat=ev_lat,
        ev_lon=ev_lon,
        ev_depth_km=ev_depth_km,
        ev_mw=ev_mw,
        ev_regime=None,
        gmpe_bundle=assets["gmpe_bundle"],
        c1_huber=assets["c1_huber"],
        trench_tree=assets["trench_tree"],
        trench_vels=assets["trench_vels"],
        trench_ages=assets["trench_ages"],
        slab_model=assets["slab_model"],
        use_nonlinear_damping=use_nonlinear_damping,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    df_u = assets["df_u"]
    mmi_arr = np.round(sim["mmi_site"], 2)
    mercalli_grades = [describe_mercalli_dual(m)[0] for m in mmi_arr]
    mercalli_names = [describe_mercalli_dual(m)[1] for m in mmi_arr]
    mercalli_mags = [describe_mercalli_dual(m)[4] for m in mmi_arr]

    res_df = pd.DataFrame(
        {
            "ubigeo": df_u["ubigeo"],
            "departamento": df_u["departamento"],
            "provincia": df_u["provincia"],
            "distrito": df_u["distrito"],
            "dist_lat": df_u["dist_lat"],
            "dist_lon": df_u["dist_lon"],
            "primary_station_code": df_u["primary_station_code"],
            "site_amp_normal": np.round(df_u["site_amp_factor_vs_msol"], 2),
            "site_delta_s2s_eff": np.round(sim["delta_s2s_eff"], 3),
            "site_amp_eff": np.round(np.power(10.0, sim["delta_s2s_eff"]), 2),
            "dist_epicentral_km": np.round(sim["dist_epicentral_km"], 1),
            "pga_rock_gal": np.round(sim["pga_rock_gal"], 1),
            "pga_site_gal": np.round(sim["pga_site_gal"], 1),
            "pga_site_pct_g": np.round(sim["pga_site_gal"] / 9.81, 1),
            "mmi_est": mmi_arr,
            "mercalli_romano": mercalli_grades,
            "mercalli_mag_eq": mercalli_mags,
            "nivel_impacto": mercalli_names,
            "n_zonas_criticas_activas": df_u["n_zonas_criticas_activas"],
            "n_peligros_total": df_u["n_peligros_total"],
            "nivel_riesgo_multipeligro_sigrid": df_u["nivel_riesgo_multipeligro_sigrid"],
        }
    )
    raw_regime = str(sim["resolved_regime"])
    meta = {
        "latency_ms": round(elapsed_ms, 2),
        "resolved_regime": raw_regime,
        "resolved_regime_es": REGIME_SHORT_ES.get(raw_regime, raw_regime),
        "resolved_regime_full_es": REGIME_LABELS_ES.get(raw_regime, raw_regime),
        "max_pga_gal": round(float(res_df["pga_site_gal"].max()), 1),
        "max_pga_pct_g": round(float(res_df["pga_site_gal"].max() / 9.81), 1),
        "max_mmi": round(float(res_df["mmi_est"].max()), 2),
        "n_districts_mmi7_plus": int((res_df["mmi_est"] >= 7.0).sum()),
        "n_districts_mmi8_plus": int((res_df["mmi_est"] >= 8.0).sum()),
        "exposed_zonas_criticas_mmi7": int(res_df.loc[res_df["mmi_est"] >= 7.0, "n_zonas_criticas_activas"].sum()),
    }
    return res_df, meta


@st.cache_resource(show_spinner="Cargando motores geofísicos y de inteligencia artificial...")
def _cached_sim_assets() -> dict[str, Any]:
    return load_simulation_assets()


@st.cache_data(show_spinner="Cargando catálogos históricos (1552–2026) y capas territoriales...")
def _cached_datasets() -> dict[str, Any]:
    df_u = pd.read_parquet(UBIGEO_PARQUET)
    df_u["ubigeo"] = df_u["ubigeo"].astype(str).str.zfill(6)
    df_h3 = pd.read_parquet(H3_PARQUET)
    df_gaps = pd.read_parquet(GAPS_PROB_PARQUET)
    df_gaps_raw = pd.read_parquet(GAPS_RAW_PARQUET)[["segment_id", "trench_lon_mean"]]
    df_site = pd.read_parquet(SITE_FACTORS_PARQUET)
    df_site["ubigeo"] = df_site["ubigeo"].astype(str).str.zfill(6)
    df_zc = pd.read_parquet(ZONAS_PARQUET)
    if "ubigeo" in df_zc.columns:
        df_zc["ubigeo"] = df_zc["ubigeo"].astype(str).str.zfill(6)
    h3_metrics = json.loads(H3_METRICS_JSON.read_text(encoding="utf-8"))

    u_tree = cKDTree(latlon_to_unit_xyz(df_u["dist_lat"].to_numpy(), df_u["dist_lon"].to_numpy()))
    h3_xyz = latlon_to_unit_xyz(df_h3["cell_lat"].to_numpy(), df_h3["cell_lon"].to_numpy())
    _, nn_idx = u_tree.query(h3_xyz, k=1)
    df_h3["departamento_cercano"] = df_u["departamento"].to_numpy()[nn_idx]
    df_h3["provincia_cercana"] = df_u["provincia"].to_numpy()[nn_idx]
    df_h3["distrito_cercano"] = df_u["distrito"].to_numpy()[nn_idx]
    df_h3["ubicacion_legible"] = (
        "Cerca de " + df_h3["distrito_cercano"].astype(str)
        + ", " + df_h3["provincia_cercana"].astype(str)
        + " (" + df_h3["departamento_cercano"].astype(str) + ")"
    )

    df_hist = pd.read_parquet(CAT_HIST_PARQUET)[
        ["event_id", "timestamp_utc", "lat", "lon", "depth_km", "mag_homog", "regimen_sismotectonico", "dist_trench_km"]
    ].rename(columns={"mag_homog": "mag"})
    df_hist["periodo"] = "Histórico (1552–1959)"

    df_inst = pd.read_parquet(CAT_INST_PARQUET)[
        ["event_id", "timestamp_utc", "lat", "lon", "depth_km", "mag", "regimen_sismotectonico", "dist_trench_km"]
    ].copy()
    df_inst["periodo"] = "Instrumental (1960–2026)"

    df_cat = pd.concat([df_hist, df_inst], ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    df_cat["anio"] = df_cat["timestamp_utc"].dt.year.astype("int32")
    df_cat["regimen_es"] = df_cat["regimen_sismotectonico"].map(REGIME_SHORT_ES).fillna("Otro")
    df_cat["macro_zona"] = np.where(
        df_cat["lat"] > -9.0,
        "Norte (Tumbes a Áncash)",
        np.where(df_cat["lat"] > -15.0, "Centro (Lima, Ica, Sierra Central)", "Sur (Arequipa, Moquegua, Tacna, Cusco)"),
    )

    return {
        "df_h3": df_h3,
        "df_gaps": df_gaps,
        "df_gaps_raw": df_gaps_raw,
        "df_site": df_site,
        "df_zc": df_zc,
        "df_cat": df_cat,
        "h3_metrics": h3_metrics,
    }


@st.cache_resource(show_spinner="Cargando polígonos oficiales del INEI...")
def _cached_inei_boundaries() -> dict[str, Any]:
    """Load cached simplified INEI GeoJSON layers, UBIGEO lookup, and 3D Peru surface contour."""
    dept_geojson: dict[str, Any] | None = None
    dist_by_ubigeo: dict[str, dict[str, Any]] = {}

    if INEI_DEPT_GEOJSON.is_file():
        dept_geojson = json.loads(INEI_DEPT_GEOJSON.read_text(encoding="utf-8"))

    if INEI_DIST_GEOJSON.is_file():
        raw_dist = json.loads(INEI_DIST_GEOJSON.read_text(encoding="utf-8"))
        for feat in raw_dist.get("features", []):
            props = feat.get("properties") or {}
            ub = str(props.get("ubigeo", "")).strip().zfill(6)
            if len(ub) == 6 and ub != "000000":
                dist_by_ubigeo[ub] = feat

    peru_3d_x, peru_3d_y, peru_3d_z = _extract_peru_outline_3d(dept_geojson)

    return {
        "dept_geojson": dept_geojson,
        "dist_by_ubigeo": dist_by_ubigeo,
        "peru_3d_x": peru_3d_x,
        "peru_3d_y": peru_3d_y,
        "peru_3d_z": peru_3d_z,
    }


@st.cache_data(show_spinner="Generando mapa interactivo con capas seleccionadas...")
def _build_dynamic_folium_html(
    show_inei_dept: bool,
    show_h3: bool,
    show_gaps: bool,
    show_stations: bool,
    show_geocatmin: bool,
    theme_mode: str,
) -> str:
    """Build interactive multi-layer Folium map with Fullscreen control."""
    assets = _cached_sim_assets()
    data = _cached_datasets()
    inei = _cached_inei_boundaries()
    df_u = assets["df_u"]
    df_h3 = data["df_h3"]
    df_gaps = data["df_gaps"]
    df_gaps_raw = data["df_gaps_raw"]
    df_site = data["df_site"]
    df_zc = data["df_zc"]

    m = folium.Map(location=[-10.2, -75.2], zoom_start=6, tiles=folium_tiles_for_theme(theme_mode), control_scale=True)
    _add_fullscreen(m)

    dept_border = "#CBD5E1" if theme_mode == THEME_DARK else "#44403C"
    if show_inei_dept and inei["dept_geojson"] is not None:
        folium.GeoJson(
            inei["dept_geojson"],
            name="Límites Departamentales INEI",
            style_function=lambda _f: {
                "fillColor": "#FFFFFF",
                "color": dept_border,
                "weight": 1.3,
                "fillOpacity": 0.03,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["departamento"],
                aliases=["Departamento (INEI):"],
                sticky=False,
            ),
        ).add_to(m)

    if show_h3:
        active = df_h3[df_h3["prob_m45_365d"] >= 0.02].copy()
        features = []
        for _, r in active.iterrows():
            p = float(r["prob_m45_365d"])
            color = "#7F1D1D" if p >= 0.20 else "#B91C1C" if p >= 0.12 else "#EA580C" if p >= 0.06 else "#F59E0B"
            boundary = h3.cell_to_boundary(str(r["h3_res5"]))
            coords = [[float(lon), float(lat)] for lat, lon in boundary]
            coords.append(coords[0])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "ubicacion": str(r["ubicacion_legible"]),
                        "prob_12m": f"{p * 100.0:.1f}%",
                        "prob_3m": f"{float(r['prob_m45_90d']) * 100.0:.1f}%",
                        "dist_fosa": f"{float(r['dist_trench_km']):.0f} km",
                        "fill_color": color,
                    },
                }
            )
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            style_function=lambda f: {
                "fillColor": f["properties"]["fill_color"],
                "color": "#57534E",
                "weight": 0.35,
                "fillOpacity": 0.50,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["ubicacion", "prob_12m", "prob_3m", "dist_fosa"],
                aliases=["Ubicación:", "Prob. sismo M≥4.5 (12 meses):", "Prob. sismo M≥4.5 (3 meses):", "Distancia a la fosa marina:"],
                sticky=True,
            ),
        ).add_to(m)

    if show_gaps:
        segs = df_gaps[df_gaps["entity_type"] == "segment"].merge(
            df_gaps_raw, left_on="entity_id", right_on="segment_id", how="left"
        )
        for _, r in segs.iterrows():
            mw_p50 = float(r["mw_pot_p50"])
            p30 = float(r["bpt_p30_a05"])
            yr_str = f"{r['last_major_rupture_year']:.0f}" if pd.notna(r["last_major_rupture_year"]) else "Sin registro desde 1552"
            is_crit = (-13.5 <= float(r["lat_center"]) <= -11.5) or (-18.5 <= float(r["lat_center"]) <= -17.0)
            color = "#7F1D1D" if mw_p50 >= 8.30 else "#B91C1C" if mw_p50 >= 8.10 else "#EA580C" if mw_p50 >= 7.80 else "#F59E0B"
            popup_html = (
                f"<b>Tramo Costero: {r['coastal_zone']}</b><br>"
                f"Último gran terremoto: <b>{yr_str}</b> ({r['years_since_major_rupture']:.0f} años)<br>"
                f"Magnitud potencial acumulada: <b>Mw {mw_p50:.2f}</b><br>"
                f"Probabilidad de gran ruptura: <b>30 años = {p30:.1f}%</b> | 50 años = {r['bpt_p50_a05']:.1f}%"
            )
            folium.CircleMarker(
                location=[float(r["lat_center"]), float(r["trench_lon_mean"])],
                radius=11 if is_crit else 7,
                color="#F8FAFC" if (is_crit and theme_mode == THEME_DARK) else ("#1C1917" if is_crit else "#57534E"),
                weight=2.2 if is_crit else 1.0,
                fill=True,
                fill_color=color,
                fill_opacity=0.88,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{r['coastal_zone']} · Magnitud potencial: Mw {mw_p50:.2f} · Prob. 30 años: {p30:.1f}%",
            ).add_to(m)

    if show_stations:
        u_lookup = df_u.set_index("ubigeo")[["pga_escenario_lima_mw86_gal", "mmi_escenario_lima_mw86"]].to_dict("index")
        for _, r in df_site.iterrows():
            amp = float(r["amp_factor_rel_msol"])
            color = "#991B1B" if amp >= 1.50 else "#D97706" if amp >= 1.20 else "#0F766E"
            u_info = u_lookup.get(str(r["ubigeo"]), {})
            pga_l = u_info.get("pga_escenario_lima_mw86_gal", np.nan)
            mmi_l = u_info.get("mmi_escenario_lima_mw86", np.nan)
            mmi_txt = format_mercalli_with_mag(mmi_l) if pd.notna(mmi_l) else "—"
            pga_txt = f"{pga_l:.0f} Gal ({pga_l/9.81:.0f}% g)" if pd.notna(pga_l) else "—"
            popup_html = (
                f"<b>Estación IGP {r['cod_estacion']}</b> — {r['distrito']} ({r['departamento']})<br>"
                f"Sismos registrados: {int(r['n_records'])}<br>"
                f"Amplificación del suelo en sismos normales: <b>{amp:.2f}×</b> vs roca firme<br>"
                f"Sacudimiento en escenario Lima M 8.6: <b>{pga_txt}</b> ({mmi_txt})"
            )
            folium.CircleMarker(
                location=[float(r["station_lat"]), float(r["station_lon"])],
                radius=5.5,
                color="#1C1917",
                weight=0.8,
                fill=True,
                fill_color=color,
                fill_opacity=0.90,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"Estación IGP {r['cod_estacion']} ({r['distrito']}): suelo {amp:.2f}×",
            ).add_to(m)

    if show_geocatmin:
        mc = MarkerCluster().add_to(m)
        for _, r in df_zc.iterrows():
            pel = str(r["PELIGROS_G"]) if pd.notna(r["PELIGROS_G"]) else "Peligro geológico"
            dist_n = str(r["DISTRITO"]) if pd.notna(r["DISTRITO"]) else ""
            reg_n = str(r["REGION"]) if pd.notna(r["REGION"]) else ""
            folium.CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=4,
                color="#6D28D9",
                fill=True,
                fill_color="#7C3AED",
                fill_opacity=0.8,
                popup=f"<b>Zona Crítica GEOCATMIN</b><br>{pel}<br>{dist_n}, {reg_n}",
            ).add_to(mc)

    return m._repr_html_()


def _build_district_consultant_map(
    dist_row: pd.Series,
    dist_feature: dict[str, Any] | None,
    df_site: pd.DataFrame,
    df_zc: pd.DataFrame,
    theme_mode: str,
) -> str:
    """Build focused Folium map with Fullscreen control, INEI polygon, ACELDAT sensors & GEOCATMIN zones."""
    lat0 = float(dist_row["dist_lat"])
    lon0 = float(dist_row["dist_lon"])
    nivel_seis = str(dist_row["nivel_peligro_sismico_puro"])
    poly_col = RISK_LEVEL_COLORS.get(nivel_seis, "#B91C1C")
    poly_border = "#F8FAFC" if theme_mode == THEME_DARK else "#1C1917"

    m = folium.Map(location=[lat0, lon0], zoom_start=11, tiles=folium_tiles_for_theme(theme_mode), control_scale=True)
    _add_fullscreen(m)

    if dist_feature is not None:
        gj = folium.GeoJson(
            {"type": "FeatureCollection", "features": [dist_feature]},
            style_function=lambda _f: {
                "fillColor": poly_col,
                "color": poly_border,
                "weight": 2.4,
                "fillOpacity": 0.28,
            },
            tooltip=(
                f"Polígono Oficial INEI: {dist_row['distrito']} ({dist_row['provincia']}, {dist_row['departamento']}) "
                f"· UBIGEO {dist_row['ubigeo']}"
            ),
        )
        gj.add_to(m)
        try:
            bounds = gj.get_bounds()
            if bounds and bounds[0][0] is not None:
                m.fit_bounds(bounds, padding=(25, 25))
        except Exception:
            pass
    else:
        folium.Circle(
            location=[lat0, lon0],
            radius=4500,
            color=poly_border,
            weight=2.0,
            fill=True,
            fill_color=poly_col,
            fill_opacity=0.28,
            tooltip=f"Centroide distrital {dist_row['distrito']} (UBIGEO {dist_row['ubigeo']})",
        ).add_to(m)

    # Nearby ACELDAT sensor stations (with sensor icon marker)
    st_d = np.sqrt(((df_site["station_lat"] - lat0) * 111.0) ** 2 + ((df_site["station_lon"] - lon0) * 111.0 * np.cos(np.radians(lat0))) ** 2)
    near_st = df_site[(df_site["ubigeo"] == str(dist_row["ubigeo"])) | (st_d <= 35.0)].copy()
    for _, s in near_st.iterrows():
        amp = float(s["amp_factor_rel_msol"])
        icon_col = "darkred" if amp >= 1.50 else "orange" if amp >= 1.20 else "green"
        folium.Marker(
            location=[float(s["station_lat"]), float(s["station_lon"])],
            icon=folium.Icon(color=icon_col, icon="wave-square", prefix="fa"),
            tooltip=f"Sensor IGP {s['cod_estacion']} ({s['distrito']}) · Suelo {amp:.2f}× en sismos normales",
            popup=(
                f"<b>Estación Acelerográfica IGP {s['cod_estacion']}</b><br>"
                f"{s['distrito']} ({s['departamento']})<br>"
                f"Amplificación medida en sismos normales: <b>{amp:.2f}×</b> ({int(s['n_records'])} sismos)"
            ),
        ).add_to(m)

    # Nearby GEOCATMIN critical zones (purple points)
    zc_d = np.sqrt(((df_zc["lat"] - lat0) * 111.0) ** 2 + ((df_zc["lon"] - lon0) * 111.0 * np.cos(np.radians(lat0))) ** 2)
    cond_ub = (df_zc["ubigeo"] == str(dist_row["ubigeo"])) if "ubigeo" in df_zc.columns else pd.Series(False, index=df_zc.index)
    near_zc = df_zc[cond_ub | (zc_d <= 25.0)].head(120)
    for _, z in near_zc.iterrows():
        pel = str(z["PELIGROS_G"]) if pd.notna(z["PELIGROS_G"]) else "Peligro geológico"
        dn = str(z["DISTRITO"]) if pd.notna(z["DISTRITO"]) else ""
        folium.CircleMarker(
            location=[float(z["lat"]), float(z["lon"])],
            radius=5.2,
            color="#4C1D95",
            weight=1.0,
            fill=True,
            fill_color="#7C3AED",
            fill_opacity=0.88,
            tooltip=f"Punto Crítico GEOCATMIN ({dn}): {pel}",
        ).add_to(m)

    return m._repr_html_()


@st.cache_data(show_spinner="Generando mapa de zonas de menor peligro sísmico...")
def _build_low_hazard_folium_html(theme_mode: str) -> str:
    """Build Folium map with Fullscreen control highlighting Peru's Muy Bajo & Bajo seismic hazard districts."""
    assets = _cached_sim_assets()
    inei = _cached_inei_boundaries()
    df_u = assets["df_u"]

    m = folium.Map(location=[-9.5, -74.0], zoom_start=6, tiles=folium_tiles_for_theme(theme_mode), control_scale=True)
    _add_fullscreen(m)

    dept_border = "#94A3B8" if theme_mode == THEME_DARK else "#57534E"
    if inei["dept_geojson"] is not None:
        folium.GeoJson(
            inei["dept_geojson"],
            style_function=lambda _f: {
                "fillColor": "#FFFFFF",
                "color": dept_border,
                "weight": 1.1,
                "fillOpacity": 0.03,
            },
            tooltip=folium.GeoJsonTooltip(fields=["departamento"], aliases=["Departamento:"]),
        ).add_to(m)

    low_df = df_u[df_u["nivel_peligro_sismico_puro"].isin(["Muy Bajo", "Bajo"])].copy()
    for _, r in low_df.iterrows():
        is_mb = r["nivel_peligro_sismico_puro"] == "Muy Bajo"
        col = "#0F766E" if is_mb else "#0284C7"
        folium.CircleMarker(
            location=[float(r["dist_lat"]), float(r["dist_lon"])],
            radius=5.0 if is_mb else 3.8,
            color=col,
            weight=0.8,
            fill=True,
            fill_color=col,
            fill_opacity=0.80,
            tooltip=(
                f"{r['distrito']} ({r['departamento']}) · Nivel: {r['nivel_peligro_sismico_puro']} "
                f"(Índice {r['seismic_hazard_index_0_100']:.1f}/100) · Sacudimiento máx.: {r['pga_escenario_max_gal']:.1f} Gal"
            ),
            popup=(
                f"<b>{r['distrito']} ({r['provincia']}, {r['departamento']})</b><br>"
                f"Peligro Sísmico: <b>{r['nivel_peligro_sismico_puro']}</b> ({r['seismic_hazard_index_0_100']:.1f}/100)<br>"
                f"Sacudimiento máximo estimado: <b>{r['pga_escenario_max_gal']/9.81:.1f}% g ({r['pga_escenario_max_gal']:.1f} Gal)</b><br>"
                f"Peligros geológicos (INGEMMET): <b>{int(r['n_peligros_total'])}</b> · FONDES: <b>{r['nivel_riesgo_multipeligro_sigrid']}</b>"
            ),
        ).add_to(m)

    return m._repr_html_()


def _render_citizen_header() -> None:
    st.markdown(
        """
        <div class="civic-hero">
          <h1>Observatorio Sísmico y Multi-Amenaza del Perú</h1>
          <p>
            Plataforma ciudadana y científica diseñada para que cualquier estudiante, familia o autoridad entienda el
            peligro sísmico en los <strong>1,893 distritos del Perú</strong>. Basada en 474 años de registros del IGP
            (1552–2026), acelerómetros ACELDAT, USGS, límites oficiales INEI, peligros INGEMMET/GEOCATMIN y FONDES/SIGRID.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sec1_seismic_school(data: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🎓 Escuela Sísmica: Empieza Aquí (Guía Fácil para Todos)")
    st.write(
        "Antes de explorar mapas y simuladores, aquí explicamos **en español de todos los días** qué significa cada "
        "número que verás en la plataforma, con experimentos interactivos y datos reales del Perú."
    )

    # 1. Visual Glossary of Observatory Units (6 Cards)
    st.markdown("#### 1. Glosario Visual: ¿Qué significan las palabras y números de esta plataforma?")
    g1, g2, g3 = st.columns(3)
    g1.markdown(
        """
        <div class="guide-card" style="--accent:#B91C1C;">
          <h4>📏 Magnitud (Mw — la escala moderna)</h4>
          <p>Es el <strong>tamaño único del terremoto</strong> en el lugar donde nace (reemplazó a la antigua escala de Richter).
          No cambia según dónde vivas: un sismo de <strong>M 8.0</strong> libera la misma energía total aquí y en todo el planeta.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    g2.markdown(
        """
        <div class="guide-card" style="--accent:#EA580C;">
          <h4>🏠 Intensidad Mercalli (Grados I al X)</h4>
          <p>Mide <strong>qué tan fuerte se siente y qué daños causa en tu barrio</strong>. Un mismo terremoto puede ser
          <strong>Mercalli IX</strong> en la costa y apenas <strong>Mercalli III</strong> en la selva. En toda la web verás su equivalencia aproximada en Magnitud (Mw).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    g3.markdown(
        """
        <div class="guide-card" style="--accent:#D97706;">
          <h4>⚡ Aceleración del Suelo (%g y Gal)</h4>
          <p><strong>1 g (= 981 Gal o cm/s²)</strong> es la fuerza de gravedad de la Tierra.
          A <strong>10% g (~98 Gal)</strong> cuesta pararse y caen objetos; a <strong>50% g (~490 Gal)</strong> se agrietan paredes;
          cerca de <strong>100% g (~981 Gal)</strong> la sacudida iguala a la gravedad.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g4, g5, g6 = st.columns(3)
    g4.markdown(
        """
        <div class="guide-card" style="--accent:#0F766E;">
          <h4>🌊 Brecha o Laguna Sísmica</h4>
          <p>Es un tramo de la costa donde la placa oceánica de Nazca lleva <strong>siglos trabada</strong> contra Sudamérica sin romperse
          en un gran terremoto (ej. Lima–Callao desde 1746 o Tacna desde 1868). Mientras más años pasan, más energía se acumula.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    g5.markdown(
        """
        <div class="guide-card" style="--accent:#7C3AED;">
          <h4>⛰️ Peligros Geológicos (INGEMMET)</h4>
          <p>Además del temblor, el terreno puede ceder: <strong>Flujos</strong> son huaicos o avalanchas de lodo;
          <strong>Caídas</strong> son derrumbes o caída de rocas en cerros; <strong>Zonas Críticas</strong> son sectores habitados
          que pueden activarse durante un terremoto o lluvias intensas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    g6.markdown(
        """
        <div class="guide-card" style="--accent:#0284C7;">
          <h4>🏛️ ¿Qué es FONDES / SIGRID?</h4>
          <p>Es el listado oficial del Estado peruano (CENEPRED) que clasifica a los 1,893 distritos para priorizar presupuestos
          de prevención. Como promedia lluvias, heladas y huaicos, a veces <strong>oculta el peligro sísmico real</strong> de distritos costeros secos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Interactive Magnitude & Energy Comparator
    st.markdown("---")
    st.markdown("#### 2. Comparador Interactivo: ¿Por qué un sismo de 7 no es «un poquito» más fuerte que uno de 6?")
    st.markdown(
        """
        <div class="school-callout">
          La magnitud crece de forma <strong>exponencial</strong>, no como una regla común. Subir apenas <strong>+0.2 grados duplica la energía</strong>,
          subir <strong>+1.0 grado multiplica la energía por 31.6 veces</strong>, y pasar de un temblor común <strong>M 5.0</strong> al terremoto de
          <strong>Pisco 2007 (M 8.0)</strong> equivale a <strong>31,623 temblores juntos</strong>. ¡Pruébalo abajo!
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Temblor M 5.0 vs Pisco M 8.0", width="stretch"):
        st.session_state["sch_m1"], st.session_state["sch_m2"] = 5.0, 8.0
    if b2.button("M 6.0 vs M 7.0 (+1.0 grado)", width="stretch"):
        st.session_state["sch_m1"], st.session_state["sch_m2"] = 6.0, 7.0
    if b3.button("M 6.0 vs M 8.6 (Lima 1746)", width="stretch"):
        st.session_state["sch_m1"], st.session_state["sch_m2"] = 6.0, 8.6
    if b4.button("M 6.7 vs M 8.8 (1,413× energía)", width="stretch"):
        st.session_state["sch_m1"], st.session_state["sch_m2"] = 6.7, 8.8

    st.session_state.setdefault("sch_m1", 5.0)
    st.session_state.setdefault("sch_m2", 8.0)

    s1, s2 = st.columns(2)
    m_a = s1.slider("Magnitud del Sismo A (más pequeño)", 3.0, 9.0, float(st.session_state["sch_m1"]), 0.1)
    m_b = s2.slider("Magnitud del Sismo B (más grande)", 3.0, 9.0, float(st.session_state["sch_m2"]), 0.1)

    lo_m, hi_m = min(m_a, m_b), max(m_a, m_b)
    dm = round(hi_m - lo_m, 2)
    amp_ratio = 10.0 ** dm
    energy_ratio = 10.0 ** (1.5 * dm)

    mk1, mk2, mk3 = st.columns(3)
    mk1.markdown(render_kpi_card("Diferencia de Magnitud", f"+{dm:.1f} grados", f"De Mw {lo_m:.1f} a Mw {hi_m:.1f}", "#0F766E"), unsafe_allow_html=True)
    mk2.markdown(
        render_kpi_card(
            "Tamaño de la Onda Sísmica",
            f"{amp_ratio:,.1f}× veces" if amp_ratio < 100 else f"{amp_ratio:,.0f}× veces",
            "Fórmula física: 10^(ΔM)",
            "#D97706",
            "Cuántas veces más grande se dibuja la onda en un sismógrafo a igual distancia.",
        ),
        unsafe_allow_html=True,
    )
    mk3.markdown(
        render_kpi_card(
            "Energía Total Liberada",
            f"{energy_ratio:,.1f}× veces" if energy_ratio < 100 else f"{energy_ratio:,.0f}× veces",
            "Fórmula física: 10^(1.5 · ΔM)",
            "#B91C1C",
            "Cuántos sismos del tamaño pequeño se necesitarían al mismo tiempo para igualar al grande.",
        ),
        unsafe_allow_html=True,
    )

    col_g, col_t = st.columns([1.0, 1.15])
    with col_g:
        fig_e = go.Figure()
        fig_e.add_trace(
            go.Bar(
                x=["Tamaño de Onda (10^ΔM)", "Energía Liberada (10^1.5·ΔM)"],
                y=[amp_ratio, energy_ratio],
                marker_color=["#D97706", "#B91C1C"],
                text=[f"{amp_ratio:,.1f}×" if amp_ratio < 100 else f"{amp_ratio:,.0f}×", f"{energy_ratio:,.1f}×" if energy_ratio < 100 else f"{energy_ratio:,.0f}×"],
                textposition="outside",
            )
        )
        fig_e.update_yaxes(type="log", title_text="Multiplicador (escala logarítmica)")
        fig_e.update_layout(
            **plotly_theme_kwargs(theme_mode),
            height=330,
            title=f"Comparación Mw {hi_m:.1f} frente a Mw {lo_m:.1f}",
            margin=dict(l=45, r=20, t=48, b=35),
        )
        st.plotly_chart(fig_e, width="stretch", config=PLOTLY_CFG)

    with col_t:
        jumps_df = pd.DataFrame(
            [
                {"Salto de Magnitud": "+0.2 grados (ej. 6.0 → 6.2)", "Tamaño de Onda": "1.6×", "Energía Liberada": "2.0× (el doble)"},
                {"Salto de Magnitud": "+1.0 grado (ej. 6.0 → 7.0)", "Tamaño de Onda": "10×", "Energía Liberada": "31.6×"},
                {"Salto de Magnitud": "+2.0 grados (ej. 6.0 → 8.0)", "Tamaño de Onda": "100×", "Energía Liberada": "1,000×"},
                {"Salto de Magnitud": "6.7 → 8.8 (+2.1 grados)", "Tamaño de Onda": "126×", "Energía Liberada": "1,413×"},
                {"Salto de Magnitud": "5.0 → 8.0 (Temblor vs Pisco)", "Tamaño de Onda": "1,000×", "Energía Liberada": "31,623×"},
            ]
        )
        st.markdown("**Tabla Rápida de Equivalencias de Energía**")
        st.markdown(render_centered_table(jumps_df, max_height_px=275), unsafe_allow_html=True)
        st.markdown(
            render_table_legend(
                [
                    ("Salto de Magnitud", "Diferencia en grados Mw entre un sismo menor y otro mayor."),
                    ("Tamaño de Onda", "Cuántas veces crece la amplitud de vibración registrada en un sismógrafo (10^ΔM)."),
                    ("Energía Liberada", "Cuántos sismos pequeños juntos equivalen al grande (10^(1.5·ΔM))."),
                ]
            ),
            unsafe_allow_html=True,
        )

    # 3. Mercalli I-X + Equivalent Mw Range visual centered table
    st.markdown("---")
    st.markdown("#### 3. Escala Dual: Intensidad Mercalli (Grados I al X) y su Equivalencia en Magnitud (Mw)")
    st.write(
        "Esta tabla conecta lo que **sientes en tu casa (Mercalli)** con la **fuerza de sacudida (%g y Gal)** y el "
        "**rango aproximado de Magnitud (Mw)** que suele producir ese nivel cerca del epicentro."
    )
    mercalli_rows = []
    for grade in range(1, 11):
        rom, short, desc, _, mag_rng = describe_mercalli_dual(float(grade))
        pga_hint, house_hint = MERCALLI_PGA_HINTS[grade]
        mercalli_rows.append(
            {
                "Grado Mercalli": f"Grado {rom}",
                "Nivel": short,
                "Rango Magnitud Eq. (Mw)": mag_rng,
                "Sacudida (%g y Gal)": pga_hint,
                "¿Qué siente una persona?": desc,
                "¿Qué pasa con las viviendas?": house_hint,
            }
        )
    st.markdown(
        render_centered_table(pd.DataFrame(mercalli_rows), badge_cols={"Nivel"}, max_height_px=470),
        unsafe_allow_html=True,
    )
    st.markdown(
        render_table_legend(
            [
                ("Grado Mercalli (I–X)", "Indica la fuerza real del movimiento en un lugar específico; disminuye conforme te alejas del epicentro."),
                ("Rango Magnitud Eq. (Mw)", "Magnitud aproximada de un sismo superficial que produce esa intensidad cerca del epicentro."),
                ("Sacudida (%g y Gal)", "Aceleración horizontal del suelo. 100% g = 981 Gal = fuerza de gravedad de la Tierra."),
            ]
        ),
        unsafe_allow_html=True,
    )

    # 4. Soil Amplification Effect in Lima (Linear vs Nonlinear explanation)
    st.markdown("---")
    st.markdown("#### 4. El Efecto del Suelo de tu Barrio (Mediciones Reales del IGP en Lima y Callao)")
    df_site = data["df_site"]
    target_codes = [
        ("MSOL", "Morro Solar — Chorrillos (Roca firme)"),
        ("RINC", "La Molina — La Rinconada (Suelo blando)"),
        ("OLIV", "Los Olivos (Depósito aluvial)"),
        ("MARQ", "Callao — Márquez (Suelo marino/blando)"),
        ("PIDR", "Puente Piedra (Suelo muy amplificador)"),
    ]
    soil_rows = []
    for code, label_es in target_codes:
        m_st = df_site[df_site["cod_estacion"] == code]
        if not m_st.empty:
            row_s = m_st.iloc[0]
            soil_rows.append(
                {
                    "Estación": label_es,
                    "Código": code,
                    "Factor": round(float(row_s["amp_factor_rel_msol"]), 2),
                    "Sismos": int(row_s["n_records"]),
                }
            )
    soil_df = pd.DataFrame(soil_rows)

    sc1, sc2 = st.columns([1.22, 0.98])
    with sc1:
        colors_soil = ["#0F766E" if v <= 1.05 else "#D97706" if v < 1.50 else "#B91C1C" for v in soil_df["Factor"]]
        fig_soil = go.Figure(
            go.Bar(
                x=soil_df["Factor"],
                y=soil_df["Estación"],
                orientation="h",
                marker_color=colors_soil,
                text=[f"{v:.2f}× ({n} sismos)" for v, n in zip(soil_df["Factor"], soil_df["Sismos"])],
                textposition="outside",
            )
        )
        fig_soil.add_vline(x=1.0, line_dash="dash", line_color="#64748B", annotation_text="Roca firme (1.00×)")
        fig_soil.update_xaxes(range=[0, 2.75], title_text="Amplificación en sismos normales frente a roca firme")
        fig_soil.update_layout(**plotly_theme_kwargs(theme_mode), height=340, margin=dict(l=10, r=45, t=20, b=40))
        st.plotly_chart(fig_soil, width="stretch", config=PLOTLY_CFG)

    with sc2:
        st.markdown(
            """
            <div class="school-callout" style="border-left-color:#B91C1C;">
              <h4 style="margin:0 0 0.42rem 0;color:var(--accent-title);">Sismos normales vs. Megaterremotos (¿Por qué cambia de 2.17× a 1.24×?)</h4>
              <p style="margin:0 0 0.5rem 0;">
                En temblores cotidianos, la onda entra a suelos arenosos o rellenos y se amplifica como una ola que llega a la playa:
                en <strong>Puente Piedra</strong> se sacude <strong>2.17 veces más</strong> que sobre la roca del Morro Solar (<code>1.00×</code>).
              </p>
              <p style="margin:0;">
                Pero en un <strong>megaterremoto extremo (M ≥ 8.0)</strong>, el suelo blando se deforma y absorbe parte del golpe
                (<em>amortiguamiento no lineal</em>): por eso en el escenario Lima M 8.6 la amplificación efectiva de Puente Piedra
                se modera de <strong>2.17×</strong> a <strong>1.24×</strong>, evitando números irreales.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Historical Cadence — Debunking the 100-year clock myth
    st.markdown("---")
    st.markdown("#### 5. ¿Cada cuánto ocurren los grandes terremotos en el Perú?")
    st.info(
        "🧭 **Desmintiendo el mito del «reloj exacto cada 100 años»:** Los terremotos NO tienen calendario fijo. "
        "En el Perú ocurrieron grandes terremotos separados por apenas **4 años** (1970 y 1974) o **6 años** (2001 y 2007), "
        "mientras Lima–Callao lleva **280 años** sin una ruptura total desde 1746. Lo único seguro es que las placas siguen chocando a **6–8 cm/año**."
    )

    hist_df = pd.DataFrame(SCHOOL_MAJOR_QUAKES)
    st.markdown(render_centered_table(hist_df, max_height_px=370), unsafe_allow_html=True)
    st.markdown(
        render_table_legend(
            [
                ("Magnitud (Mw)", "Energía total del terremoto histórico homogeneizada a la escala moderna de momento."),
                ("Tiempo al Siguiente", "Años transcurridos desde el gran terremoto anterior de esta lista (demuestra que no hay ciclos fijos)."),
                ("Lección Ciudadana", "Qué aprendió la ingeniería y la gestión del riesgo tras cada evento."),
            ]
        ),
        unsafe_allow_html=True,
    )

    # 6. Citizen FAQ (Expanders)
    st.markdown("---")
    st.markdown("#### 6. Preguntas Frecuentes Ciudadanas")
    with st.expander("❓ ¿Se puede predecir la fecha exacta de un terremoto?"):
        st.write(
            "**No.** Ninguna institución científica ni inteligencia artificial en el mundo puede predecir el día, la hora ni el minuto "
            "de un terremoto. Esta plataforma calcula **probabilidades estadísticas** (por ejemplo, probabilidad en 12 meses o 30 años) "
            "y **simulaciones de impacto** (*si ocurre un sismo aquí, cuánto se sacudiría mi distrito*) para preparar viviendas y planes de emergencia."
        )
    with st.expander("❓ ¿Por qué en mi distrito se siente más fuerte que en el distrito vecino?"):
        st.write(
            "Por dos razones: (1) **el tipo de suelo** —los suelos arenosos, humedales o rellenos amplifican la vibración hasta más del doble "
            "comparados con la roca dura—, y (2) **el tipo de edificio** —pisos altos o viviendas sin columnas bien amarradas oscilan mucho más."
        )
    with st.expander("❓ ¿Es verdad que cada 100 años hay un terremoto fijo en Lima?"):
        st.write(
            "**No es un reloj fijo.** Es cierto que el tramo Lima–Callao–Chincha no tiene un megaterremoto similar al de **1746 (M 8.8)** desde hace "
            "**280 años**, y que la placa de Nazca acumula tensión constantemente; por eso la probabilidad a 30 años es alta (~36% a 63% según el modelo), "
            "pero la ruptura puede ocurrir mañana o dentro de décadas."
        )
    with st.expander("❓ ¿Qué significa que mi distrito tenga Índice Sísmico 96/100?"):
        st.write(
            "**NO significa 96% de probabilidad de terremoto.** Es un puntaje comparativo de **0 (mínimo del Perú)** a **100 (máximo del Perú)** "
            "que combina tres factores físicos: (1) qué tan fuerte sería la sacudida en megaterremotos considerando tu suelo (45%), "
            "(2) cercanía a una laguna sísmica costera con energía acumulada (30%) y (3) frecuencia anual de sismos sensibles en la zona (25%)."
        )


def _render_sec2_national_map(assets: dict[str, Any], data: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🗺️ Mapa Nacional Multi-Amenaza y Pronóstico por Zonas")
    st.write(
        "Activa o desactiva capas para ver dónde coinciden los límites departamentales oficiales del INEI, la probabilidad "
        "anual de sismos, las lagunas costeras, la amplificación del suelo y las zonas críticas de deslizamientos o huaicos."
    )

    ctrl_col, map_col = st.columns([0.31, 0.69])
    with ctrl_col:
        st.markdown("#### Capas del Mapa")
        show_inei_dept = st.checkbox("Límites Departamentales Oficiales (INEI WFS)", value=True)
        show_h3 = st.checkbox("Probabilidad de sismos M≥4.5 en 12 meses (Grilla H3)", value=True)
        show_gaps = st.checkbox("Lagunas Sísmicas Costeras (Energía acumulada)", value=True)
        show_stations = st.checkbox("Estaciones Acelerográficas IGP (Efecto de suelo)", value=True)
        show_geocatmin = st.checkbox("Zonas Críticas de Huaicos/Derrumbes (GEOCATMIN)", value=False)
        st.info(
            "💡 Haz clic en cualquier círculo del mar o estación en tierra para ver su detalle, o usa el botón de "
            "**pantalla completa** en la esquina superior izquierda del mapa."
        )

    with map_col:
        html_map = _build_dynamic_folium_html(show_inei_dept, show_h3, show_gaps, show_stations, show_geocatmin, theme_mode)
        st.iframe(html_map, height=565, width="stretch")
        st.markdown(
            render_map_legend(
                [
                    ("#7F1D1D", "Hexágono rojo oscuro", "Probabilidad anual ≥ 12% de sismo M≥4.5"),
                    ("#EA580C", "Hexágono naranja/amarillo", "Probabilidad anual entre 2% y 12%"),
                    ("#1C1917", "Círculo costero con borde oscuro", "Laguna sísmica crítica (Lima–Chincha / Ilo–Tacna)"),
                    ("#0F766E", "Sensor verde/naranja/rojo", "Estación IGP según amplificación del suelo (≤1.2× / 1.2–1.5× / ≥1.5×)"),
                    ("#7C3AED", "Punto morado", "Zona crítica de huaicos o deslizamientos (INGEMMET)"),
                ]
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📊 Explorador de Pronóstico Probabilístico por Zonas (Próximos 3 y 12 Meses)")
    st.write(
        "Estima la **probabilidad estadística** de que ocurra al menos un sismo sensible (M ≥ 4.5) o fuerte (M ≥ 5.0) "
        "por zona. *No predice fechas exactas.*"
    )

    df_h3 = data["df_h3"].copy()
    h3_metrics = data["h3_metrics"]

    c1, c2, c3 = st.columns([1.3, 1.2, 1.1])
    horizontes = {
        "Próximos 12 meses — Sismos sensibles (M ≥ 4.5)": "prob_m45_365d",
        "Próximos 3 meses — Sismos sensibles (M ≥ 4.5)": "prob_m45_90d",
        "Próximos 3 meses — Sismos fuertes (M ≥ 5.0)": "prob_m50_90d",
    }
    h_lbl = c1.selectbox("Horizonte de pronóstico", list(horizontes.keys()))
    prob_col = horizontes[h_lbl]

    depas = ["Todo el Perú"] + sorted(df_h3["departamento_cercano"].dropna().unique().tolist())
    sel_depa = c2.selectbox("Filtrar por departamento", depas)
    min_pct = c3.slider("Umbral mínimo de probabilidad (%)", 0.0, 20.0, 2.0 if "12 meses" in h_lbl else 0.5, 0.5)

    sub = df_h3 if sel_depa == "Todo el Perú" else df_h3[df_h3["departamento_cercano"] == sel_depa].copy()
    vis = sub[sub[prob_col] * 100.0 >= min_pct].copy()
    vis["prob_pct"] = np.round(vis[prob_col] * 100.0, 2)

    col_m, col_s = st.columns([1.15, 1.0])
    with col_m:
        if not vis.empty:
            fig_f = go.Figure(
                go.Scattermap(
                    lat=vis["cell_lat"],
                    lon=vis["cell_lon"],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=vis["prob_pct"],
                        colorscale="YlOrRd",
                        colorbar=dict(title=dict(text="Probabilidad (%)", side="right"), thickness=14),
                        opacity=0.82,
                    ),
                    text=[
                        f"<b>{r['ubicacion_legible']}</b><br>"
                        f"Probabilidad estimada: <b>{r['prob_pct']:.1f}%</b><br>"
                        f"Distancia a la fosa marina: {r['dist_trench_km']:.0f} km<br>"
                        f"Peligros geológicos cercanos: {int(r['n_geocatmin_peligros'])}"
                        for _, r in vis.iterrows()
                    ],
                    hoverinfo="text",
                )
            )
            fig_f.update_layout(
                **plotly_theme_kwargs(theme_mode),
                map=dict(
                    style="carto-darkmatter" if theme_mode == THEME_DARK else "open-street-map",
                    center=dict(lat=float(vis["cell_lat"].mean()), lon=float(vis["cell_lon"].mean())),
                    zoom=5.2 if sel_depa == "Todo el Perú" else 7.0,
                ),
                height=405,
                margin=dict(l=0, r=0, t=8, b=0),
            )
            st.plotly_chart(fig_f, width="stretch", config=PLOTLY_CFG)
            st.caption("💡 Usa el ícono de expansión en la esquina superior derecha del mapa para verlo en pantalla completa.")
        else:
            st.info("No hay zonas por encima del umbral seleccionado en este departamento.")

    with col_s:
        st.markdown("**¿Qué factores explican este pronóstico?**")
        shap_raw = pd.DataFrame(h3_metrics["top15_shap_features_m45_90d"]).head(10).iloc[::-1]
        shap_raw["Factor"] = [translate_feature(f) for f in shap_raw["feature"]]
        fig_sh = px.bar(
            shap_raw,
            x="mean_abs_shap",
            y="Factor",
            orientation="h",
            color_discrete_sequence=["#B91C1C"],
            labels={"mean_abs_shap": "Peso explicativo promedio", "Factor": ""},
        )
        fig_sh.update_layout(**plotly_theme_kwargs(theme_mode), height=405, margin=dict(l=10, r=15, t=10, b=35))
        st.plotly_chart(fig_sh, width="stretch", config=PLOTLY_CFG)

    st.markdown(f"**Top 15 Zonas con Mayor Probabilidad Estimada ({h_lbl})**")
    top_z = vis.sort_values(prob_col, ascending=False).head(15)
    tbl_z = pd.DataFrame(
        {
            "Ubicación de Referencia": top_z["ubicacion_legible"],
            "Departamento": top_z["departamento_cercano"],
            "Probabilidad Estimada": [f"{v:.1f}%" for v in top_z["prob_pct"]],
            "Distancia a la Fosa": [f"{v:.0f} km" for v in top_z["dist_trench_km"]],
            "Profundidad de Placa": [f"{v:.0f} km" for v in top_z["slab_depth_est_km"]],
            "Peligros Geológicos": [f"{int(v):,}" for v in top_z["n_geocatmin_peligros"]],
        }
    )
    st.markdown(render_centered_table(tbl_z, max_height_px=390), unsafe_allow_html=True)
    st.markdown(
        render_table_legend(
            [
                ("Probabilidad Estimada (%)", "Probabilidad porcentual de que ocurra al menos un sismo del umbral elegido dentro de esa celda (~252 km²) en el periodo seleccionado."),
                ("Distancia a la Fosa", "Kilómetros en línea recta hasta la fosa marina donde chocan las placas de Nazca y Sudamérica."),
                ("Profundidad de Placa", "Profundidad estimada bajo tierra a la que pasa la placa oceánica de Nazca debajo de esa zona."),
                ("Peligros Geológicos", "Cantidad de puntos de huaicos, deslizamientos o caída de rocas inventariados por INGEMMET/GEOCATMIN en la zona."),
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_sec3_district_consultant(assets: dict[str, Any], data: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🏘️ Consultor de Mi Distrito: Probabilidad Sísmica y Riesgo Local")
    st.write(
        "Elige tu distrito para ver su **polígono oficial del INEI**, cuánta probabilidad estadística hay de sentir un sismo "
        "en los próximos meses o años, cómo responde tu suelo en sismos normales vs. megaterremotos y cómo se compara con tus vecinos."
    )

    df_u = assets["df_u"].copy()
    inei = _cached_inei_boundaries()
    df_u["etiqueta"] = (
        df_u["distrito"] + " — " + df_u["provincia"] + ", " + df_u["departamento"] + " (" + df_u["ubigeo"] + ")"
    )

    default_matches = df_u.index[df_u["ubigeo"] == "150125"].tolist()
    default_idx = int(default_matches[0]) if default_matches else 0
    sel_lbl = st.selectbox("🔎 Selecciona tu distrito (1,893 distritos oficiales del Perú)", df_u["etiqueta"].tolist(), index=default_idx)
    r = df_u[df_u["etiqueta"] == sel_lbl].iloc[0]

    ubigeo = str(r["ubigeo"])
    dist_feat = inei["dist_by_ubigeo"].get(ubigeo)

    p3m_m45 = float(np.clip(r["prob_m45_90d_nbhd_max"], 0.0, 1.0))
    p12m_m45 = float(np.clip(r["prob_m45_365d_nbhd_max"], 0.0, 1.0))
    p6m_m45 = 1.0 - (1.0 - p12m_m45) ** 0.5
    p5y_m45 = 1.0 - (1.0 - p12m_m45) ** 5.0
    p3m_m50 = float(np.clip(r["prob_m50_90d_nbhd_max"], 0.0, 1.0))
    p12m_m50 = 1.0 - (1.0 - p3m_m50) ** (365.0 / 90.0)

    bpt_30_raw = float(r["gap_bpt_p30_a05_raw"])
    bpt_50_raw = float(r["gap_bpt_p50_a05_raw"])
    bpt_30_eff = float(r["gap_bpt_p30_a05_eff"])
    gap_zone_lbl = str(r["nearest_gap_zone"]) if pd.notna(r["nearest_gap_zone"]) else "Tramo costero cercano"

    nivel_seis = str(r["nivel_peligro_sismico_puro"])
    nivel_fondes = str(r["nivel_riesgo_multipeligro_sigrid"])
    is_under = nivel_seis in ("Alto", "Muy Alto") and nivel_fondes in ("Muy Bajo", "Bajo")

    # Compute BOTH normal (linear) and megathrust (nonlinear) soil amplification dynamically
    amp_normal = float(r["site_amp_factor_vs_msol"])
    # Determine rock PGA of the critical scenario
    scen_pairs = [
        (float(r["pga_escenario_lima_mw86_gal"]), float(r["pga_escenario_lima_mw86_rock_gal"])),
        (float(r["pga_escenario_sur_mw84_gal"]), float(r["pga_escenario_sur_mw84_rock_gal"])),
        (float(r["pga_escenario_intraplaca_mw68_gal"]), float(r["pga_escenario_intraplaca_mw68_rock_gal"])),
    ]
    crit_site_gal, crit_rock_gal = max(scen_pairs, key=lambda pair: pair[0])
    amp_mega_eff = crit_site_gal / crit_rock_gal if crit_rock_gal > 0 else 1.0

    st.markdown("#### ¿Qué tan probable es un sismo en mi distrito?")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.markdown(render_kpi_card("Próximos 3 Meses", f"{p3m_m45 * 100:.1f}%", "Prob. sismo sensible M ≥ 4.5", "#0F766E", "Probabilidad porcentual calibrada (Modelo C) de al menos un sismo M≥4.5 en los próximos 90 días cerca del distrito."), unsafe_allow_html=True)
    h2.markdown(render_kpi_card("Próximos 6 Meses", f"{p6m_m45 * 100:.1f}%", "Proyección semestral (tasa anual)", "#0284C7", "Calculado desde la tasa anual calibrada: P(6m) = 1 - (1 - P12m)^0.5."), unsafe_allow_html=True)
    h3.markdown(render_kpi_card("Próximos 12 Meses", f"{p12m_m45 * 100:.1f}%", f"M≥4.5 · Fuerte M≥5.0: {p12m_m50 * 100:.1f}%", "#D97706", "Probabilidad anual calibrada de sismo sensible M≥4.5 y proyección anual de sismo fuerte M≥5.0."), unsafe_allow_html=True)
    h4.markdown(render_kpi_card("Próximos 5 Años", f"{p5y_m45 * 100:.1f}%", "M ≥ 4.5 (tasa moderna)", "#C2410C", "Proyección multianual P(5 años) = 1 - (1 - P12m)^5 asumiendo continuidad de la tasa 2000–2026."), unsafe_allow_html=True)
    h5.markdown(render_kpi_card("Megaterremoto Mw ≥ 8.0", f"{bpt_30_raw:.0f}% (30a) · {bpt_50_raw:.0f}% (50a)", f"Tramo {gap_zone_lbl} (efecto {bpt_30_eff:.0f}%)", "#991B1B", "Probabilidad condicional BPT de gran ruptura en el tramo costero más cercano."), unsafe_allow_html=True)

    if is_under:
        st.warning(
            f"⚠️ **Brecha de priorización detectada:** **{r['distrito']}** tiene Peligro Sísmico Puro **{nivel_seis}** "
            f"(Índice `{r['seismic_hazard_index_0_100']:.1f}/100`), pero aparece como **{nivel_fondes}** en el padrón oficial FONDES/SIGRID."
        )

    st.markdown("")
    col_map, col_info = st.columns([1.12, 1.0])
    with col_map:
        poly_status = "Polígono oficial INEI WFS" if dist_feat is not None else "Área referencial (centroide oficial)"
        st.markdown(f"**Mapa Distrital ({poly_status})**")
        dist_map_html = _build_district_consultant_map(r, dist_feat, data["df_site"], data["df_zc"], theme_mode)
        st.iframe(dist_map_html, height=430, width="stretch")
        st.markdown(
            render_map_legend(
                [
                    (RISK_LEVEL_COLORS.get(nivel_seis, "#B91C1C"), "Polígono sombreado INEI", f"Límite oficial de {r['distrito']} (color según nivel {nivel_seis})"),
                    ("#D97706", "Marcador con ícono de onda", "Estaciones Acelerográficas del IGP cercanas (miden amplificación real del suelo)"),
                    ("#7C3AED", "Puntos morados", "Puntos Críticos de deslizamientos, caída de rocas o huaicos (INGEMMET/GEOCATMIN)"),
                ]
            ),
            unsafe_allow_html=True,
        )

    with col_info:
        m_rom, m_short, m_desc, m_col, m_mag_rng = describe_mercalli_dual(float(r["mmi_escenario_max"]))
        st_code = str(r["primary_station_code"]) if pd.notna(r["primary_station_code"]) else "Sin estación directa (neutro)"
        pga_max = float(r["pga_escenario_max_gal"])
        pga_pct_g = pga_max / 9.81

        st.markdown(
            f"""
            <div class="district-spotlight" style="--accent:{m_col};margin-top:0;">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
                <strong style="font-size:1.05rem;">Impacto si Ocurre el Escenario Crítico</strong>
                {render_badge(f"Mercalli {m_rom} ({m_mag_rng}) — {m_short}", m_col)}
              </div>
              <p style="margin:0.42rem 0 0.65rem 0;font-size:0.88rem;color:var(--text-secondary);">
                {m_desc}
              </p>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.84rem;">
                <div><strong>Sacudimiento máximo:</strong><br>{pga_pct_g:.1f}% g ({pga_max:.0f} Gal)<br><span style="color:var(--text-muted);font-size:0.76rem;">100% g = 981 Gal (gravedad)</span></div>
                <div><strong>Efecto del suelo local ({st_code}):</strong><br>Sismos normales: <strong>{amp_normal:.2f}×</strong><br>En megaterremoto: <strong>{amp_mega_eff:.2f}×</strong></div>
                <div><strong>Peligro Sísmico Puro:</strong><br>{render_badge(nivel_seis, RISK_LEVEL_COLORS.get(nivel_seis, '#B91C1C'))} ({r['seismic_hazard_index_0_100']:.1f}/100)</div>
                <div><strong>Oficial FONDES / SIGRID:</strong><br>{render_badge(nivel_fondes, RISK_LEVEL_COLORS.get(nivel_fondes, '#475569'))}</div>
              </div>
              <div style="margin-top:0.5rem;font-size:0.78rem;color:var(--text-muted);">
                ℹ️ <em>¿Por qué difieren los factores de suelo?</em> En temblores cotidianos el suelo blando amplifica de forma lineal ({amp_normal:.2f}×),
                pero ante megaterremotos extremos se deforma y amortigua parte del golpe ({amp_mega_eff:.2f}× efectivo).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        detail_df = pd.DataFrame(
            [
                {"Escenario / Amenaza": "Escenario Lima–Callao (Mw 8.6)", "Valor del Distrito": f"{r['pga_escenario_lima_mw86_gal']/9.81:.1f}% g ({r['pga_escenario_lima_mw86_gal']:.0f} Gal)", "Equivalencia": format_mercalli_with_mag(r["mmi_escenario_lima_mw86"])},
                {"Escenario / Amenaza": "Escenario Sur Ilo–Tacna (Mw 8.4)", "Valor del Distrito": f"{r['pga_escenario_sur_mw84_gal']/9.81:.1f}% g ({r['pga_escenario_sur_mw84_gal']:.0f} Gal)", "Equivalencia": format_mercalli_with_mag(r["mmi_escenario_sur_mw84"])},
                {"Escenario / Amenaza": "Escenario Sierra Central (Mw 6.8)", "Valor del Distrito": f"{r['pga_escenario_intraplaca_mw68_gal']/9.81:.1f}% g ({r['pga_escenario_intraplaca_mw68_gal']:.0f} Gal)", "Equivalencia": format_mercalli_with_mag(r["mmi_escenario_intraplaca_mw68"])},
                {"Escenario / Amenaza": "Peligros Geológicos (INGEMMET)", "Valor del Distrito": f"{int(r['n_peligros_total'])} puntos", "Equivalencia": f"Flujos (huaicos): {int(r['n_peligros_flujo'])} · Caídas (rocas): {int(r['n_peligros_caida'])} · Desliz.: {int(r['n_peligros_deslizamiento'])}"},
                {"Escenario / Amenaza": "Zonas Críticas Activas", "Valor del Distrito": f"{int(r['n_zonas_criticas_activas'])} sectores", "Equivalencia": f"Distancia a la fosa marina: {r['dist_trench_km']:.0f} km"},
            ]
        )
        st.markdown(render_centered_table(detail_df, max_height_px=265), unsafe_allow_html=True)
        st.markdown(
            render_table_legend(
                [
                    ("Valor del Distrito (%g y Gal)", "Aceleración esperada del suelo. 10% g (~98 Gal) dificulta mantenerse en pie; 50% g (~490 Gal) agrieta paredes."),
                    ("Equivalencia Mercalli + Mw", "Grado de sacudida local (Mercalli) y rango aproximado de magnitud Mw que genera ese impacto en epicentro."),
                    ("Flujos / Caídas / Zonas Críticas", "Inventario INGEMMET de huaicos (Flujos), derrumbes de rocas (Caídas) y áreas habitadas vulnerables (Zonas Críticas)."),
                ]
            ),
            unsafe_allow_html=True,
        )

    prov_peers = df_u[
        (df_u["departamento"] == r["departamento"]) & (df_u["provincia"] == r["provincia"])
    ].sort_values("seismic_hazard_index_0_100", ascending=False).copy()

    st.markdown(f"#### Comparación con los Distritos Vecinos de la Provincia de {r['provincia']} ({len(prov_peers)} distritos)")
    peer_tbl = pd.DataFrame(
        {
            "Distrito": prov_peers["distrito"],
            "UBIGEO": prov_peers["ubigeo"],
            "Índice Sísmico (0–100)": [f"{v:.1f}" for v in prov_peers["seismic_hazard_index_0_100"]],
            "Peligro Sísmico": prov_peers["nivel_peligro_sismico_puro"],
            "Oficial FONDES": prov_peers["nivel_riesgo_multipeligro_sigrid"],
            "Prob. 12m (M≥4.5)": [f"{v * 100:.1f}%" for v in prov_peers["prob_m45_365d_nbhd_max"]],
            "Sacudimiento Máx.": [f"{v/9.81:.0f}% g ({v:.0f} Gal)" for v in prov_peers["pga_escenario_max_gal"]],
            "Mercalli + Rango Mw": [format_mercalli_with_mag(m) for m in prov_peers["mmi_escenario_max"]],
            "Suelo Normal": [f"{v:.2f}×" for v in prov_peers["site_amp_factor_vs_msol"]],
            "Zonas Críticas": [f"{int(v)}" for v in prov_peers["n_zonas_criticas_activas"]],
        }
    )
    st.markdown(
        render_centered_table(
            peer_tbl,
            badge_cols={"Peligro Sísmico", "Oficial FONDES"},
            highlight_col="UBIGEO",
            highlight_value=ubigeo,
            max_height_px=415,
            caption=f"Fila resaltada: {r['distrito']} (UBIGEO {ubigeo}).",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        render_table_legend(
            [
                ("Índice Sísmico (0–100)", "Puntaje relativo de peligro físico de 0 (mínimo nacional) a 100 (máximo); NO es un porcentaje de probabilidad."),
                ("Prob. 12m (M≥4.5)", "Probabilidad porcentual de al menos un sismo sensible (M≥4.5) en los próximos 12 meses en el vecindario."),
                ("Suelo Normal", "Cuántas veces amplifica el suelo local frente a roca firme en sismos cotidianos (en megaterremotos el amortiguamiento no lineal lo reduce)."),
                ("Zonas Críticas", "Sectores críticos habitados registrados por INGEMMET susceptibles a derrumbes, deslizamientos o huaicos."),
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_sec4_simulator(assets: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### ⚡ Simulador de Terremotos en Vivo: ¿Cómo Afectaría a Mi Distrito?")
    st.write(
        "Elige un escenario real o mueve el epicentro y la magnitud para calcular en milisegundos cómo se sacudirían "
        "los 1,893 distritos del Perú. Recuerda: **100% g (= 981 Gal)** equivale a la fuerza de gravedad de la Tierra."
    )

    pcols = st.columns(len(PRESET_SCENARIOS))
    for col, (label, vals) in zip(pcols, PRESET_SCENARIOS.items()):
        if col.button(label, width="stretch"):
            st.session_state["sim_lat"] = vals["lat"]
            st.session_state["sim_lon"] = vals["lon"]
            st.session_state["sim_depth"] = vals["depth"]
            st.session_state["sim_mw"] = vals["mw"]

    st.session_state.setdefault("sim_lat", -12.05)
    st.session_state.setdefault("sim_lon", -77.80)
    st.session_state.setdefault("sim_depth", 25.0)
    st.session_state.setdefault("sim_mw", 8.6)

    c1, c2, c3, c4, c5 = st.columns([1.15, 1.15, 1.15, 1.15, 1.4])
    ev_lat = c1.slider("Latitud del epicentro (°S)", -18.5, -3.0, float(st.session_state["sim_lat"]), 0.05)
    ev_lon = c2.slider("Longitud del epicentro (°W)", -82.0, -68.5, float(st.session_state["sim_lon"]), 0.05)
    ev_depth = c3.slider("Profundidad focal (km)", 5.0, 350.0, float(st.session_state["sim_depth"]), 1.0)
    ev_mw = c4.slider("Magnitud del terremoto (Mw)", 4.0, 9.0, float(st.session_state["sim_mw"]), 0.1)
    use_nl = c5.toggle(
        "Amortiguamiento no lineal de suelos blandos",
        value=True,
        help="En megaterremotos (>150 Gal en roca), los suelos blandos absorben parte de la sacudida (Seyhan & Stewart 2014).",
    )

    res_df, meta = run_live_simulation(assets, ev_lat, ev_lon, ev_depth, ev_mw, use_nonlinear_damping=use_nl)

    res_df["etiqueta_distrito"] = res_df["distrito"] + " — " + res_df["provincia"] + " (" + res_df["departamento"] + ") · UBIGEO " + res_df["ubigeo"]
    default_idx = int(res_df.index[res_df["ubigeo"] == "150125"][0]) if (res_df["ubigeo"] == "150125").any() else 0
    sel_dist_lbl = st.selectbox("🔍 Consulta rápida — ¿Cómo se sentiría este sismo en mi distrito?", res_df["etiqueta_distrito"].tolist(), index=default_idx)
    drow = res_df[res_df["etiqueta_distrito"] == sel_dist_lbl].iloc[0]
    m_rom, m_short, m_desc, m_col, m_mag_rng = describe_mercalli_dual(float(drow["mmi_est"]))

    st.markdown(
        f"""
        <div class="district-spotlight" style="--accent:{m_col};">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <div>
              <strong style="font-size:1.10rem;">{drow['distrito']} ({drow['provincia']}, {drow['departamento']})</strong>
              <span style="color:var(--text-muted);font-size:0.86rem;"> · Distancia al epicentro: <strong>{drow['dist_epicentral_km']:.0f} km</strong></span>
            </div>
            <div>{render_badge(f"Mercalli {m_rom} ({m_mag_rng}) — {m_short}", m_col)}</div>
          </div>
          <p style="margin:0.42rem 0 0.62rem 0;color:var(--text-secondary);font-size:0.91rem;">
            <strong>Impacto cotidiano esperado:</strong> {m_desc}
          </p>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:10px;font-size:0.86rem;">
            <div><strong>Sacudimiento real:</strong> {drow['pga_site_pct_g']:.1f}% g ({drow['pga_site_gal']:.0f} Gal)</div>
            <div><strong>Suelo normal vs. este sismo:</strong> {drow['site_amp_normal']:.2f}× normal → <strong>{drow['site_amp_eff']:.2f}× efectivo</strong></div>
            <div><strong>Zonas críticas INGEMMET:</strong> {int(drow['n_zonas_criticas_activas'])} activas ({int(drow['n_peligros_total'])} peligros)</div>
            <div><strong>Clasificación oficial FONDES:</strong> {drow['nivel_riesgo_multipeligro_sigrid']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(render_kpi_card("Tipo de Sismo", meta["resolved_regime_es"], f"Calculado en {meta['latency_ms']:.0f} ms", "#0F766E", meta["resolved_regime_full_es"]), unsafe_allow_html=True)
    max_rom, max_short, _, max_col, max_mag_rng = describe_mercalli_dual(meta["max_mmi"])
    k2.markdown(render_kpi_card("Sacudimiento Máximo", f"{meta['max_pga_pct_g']:.0f}% g", f"{meta['max_pga_gal']:.0f} Gal (100% g = 981 Gal)", max_col), unsafe_allow_html=True)
    k3.markdown(render_kpi_card("Intensidad Máxima", f"Mercalli {max_rom}", f"{max_short} · eq. {max_mag_rng}", max_col), unsafe_allow_html=True)
    k4.markdown(render_kpi_card("Distritos en Daño Fuerte", f"{meta['n_districts_mmi7_plus']} distritos", f"Mercalli ≥ VII ({meta['n_districts_mmi8_plus']} en ≥ VIII)", "#C2410C"), unsafe_allow_html=True)
    k5.markdown(render_kpi_card("Zonas Críticas Expuestas", f"{meta['exposed_zonas_criticas_mmi7']:,}", "Huaicos/derrumbes en MMI ≥ VII", "#7C3AED"), unsafe_allow_html=True)

    st.markdown("")
    map_col, tbl_col = st.columns([1.0, 1.2])
    with map_col:
        vis = res_df[res_df["pga_site_gal"] >= 1.5].copy()
        fig_map = go.Figure()
        fig_map.add_trace(
            go.Scattermap(
                lat=vis["dist_lat"],
                lon=vis["dist_lon"],
                mode="markers",
                marker=dict(
                    size=np.clip(vis["mmi_est"] * 1.9, 5, 18),
                    color=vis["mmi_est"],
                    colorscale="YlOrRd",
                    cmin=3.0,
                    cmax=9.5,
                    colorbar=dict(title=dict(text="Mercalli (MMI)", side="right"), thickness=14),
                    opacity=0.82,
                ),
                text=[
                    f"<b>{r['distrito']} ({r['departamento']})</b><br>"
                    f"Intensidad: <b>Mercalli {r['mercalli_romano']} (eq. {r['mercalli_mag_eq']}) — {r['nivel_impacto']}</b><br>"
                    f"Sacudimiento: {r['pga_site_pct_g']:.1f}% g ({r['pga_site_gal']:.0f} Gal)<br>"
                    f"Suelo efectivo: {r['site_amp_eff']:.2f}× (normal {r['site_amp_normal']:.2f}×) | Dist.: {r['dist_epicentral_km']:.0f} km"
                    for _, r in vis.iterrows()
                ],
                hoverinfo="text",
                name="Distritos afectados",
            )
        )
        fig_map.add_trace(
            go.Scattermap(
                lat=[ev_lat],
                lon=[ev_lon],
                mode="markers",
                marker=dict(size=18, color="#000000" if theme_mode == THEME_LIGHT else "#F8FAFC"),
                text=[f"<b>Epicentro simulado</b><br>Magnitud: Mw {ev_mw:.1f}<br>Profundidad: {ev_depth:.0f} km"],
                hoverinfo="text",
                name="Epicentro",
            )
        )
        fig_map.update_layout(
            **plotly_theme_kwargs(theme_mode),
            map=dict(
                style="carto-darkmatter" if theme_mode == THEME_DARK else "open-street-map",
                center=dict(lat=ev_lat, lon=ev_lon),
                zoom=5.5,
            ),
            height=475,
            margin=dict(l=0, r=0, t=8, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_map, width="stretch", config=PLOTLY_CFG)
        st.markdown(
            render_map_legend(
                [
                    ("#FACC15", "Amarillo (Mercalli III–V)", "Vibración leve a moderada sin daño estructural"),
                    ("#EA580C", "Naranja (Mercalli VI–VII)", "Daños en muros de adobe y caída de objetos"),
                    ("#991B1B", "Rojo oscuro (Mercalli VIII–X)", "Sacudimiento severo a extremo"),
                    ("#1C1917", "Punto negro/blanco grande", "Epicentro del terremoto simulado"),
                ],
                note="💡 Usa el ícono de expansión en la esquina superior derecha del mapa para verlo en pantalla completa.",
            ),
            unsafe_allow_html=True,
        )

    with tbl_col:
        st.markdown("**Top 20 Distritos con Mayor Sacudimiento Esperado**")
        top20 = res_df.sort_values("pga_site_gal", ascending=False).head(20).copy()
        tbl_top20 = pd.DataFrame(
            {
                "Distrito": top20["distrito"],
                "Provincia / Dpto.": top20["provincia"] + " / " + top20["departamento"],
                "En Roca": [f"{v/9.81:.0f}% g ({v:.0f} Gal)" for v in top20["pga_rock_gal"]],
                "Real c/ Suelo": [f"{v/9.81:.0f}% g ({v:.0f} Gal)" for v in top20["pga_site_gal"]],
                "Suelo (Norm. → Efect.)": [f"{n:.2f}× → {e:.2f}×" for n, e in zip(top20["site_amp_normal"], top20["site_amp_eff"])],
                "Mercalli + Rango Mw": [f"{rom} ({eq})" for rom, eq in zip(top20["mercalli_romano"], top20["mercalli_mag_eq"])],
                "Impacto": top20["nivel_impacto"],
            }
        )
        st.markdown(render_centered_table(tbl_top20, badge_cols={"Impacto"}, max_height_px=400), unsafe_allow_html=True)
        st.markdown(
            render_table_legend(
                [
                    ("En Roca vs. Real c/ Suelo", "Sacudimiento si el distrito fuera roca sólida vs. sacudimiento real incorporando su tipo de suelo (100% g = 981 Gal)."),
                    ("Suelo (Norm. → Efect.)", "Amplificación medida en sismos cotidianos vs. factor efectivo tras amortiguamiento no lineal en este escenario."),
                    ("Mercalli + Rango Mw", "Intensidad local estimada y rango aproximado de Magnitud Mw equivalente en epicentro."),
                ]
            ),
            unsafe_allow_html=True,
        )
        export_df = res_df.sort_values("pga_site_gal", ascending=False)[
            ["ubigeo", "departamento", "provincia", "distrito", "dist_epicentral_km", "pga_rock_gal", "pga_site_gal", "pga_site_pct_g", "site_amp_normal", "site_amp_eff", "mercalli_romano", "mercalli_mag_eq", "nivel_impacto", "n_zonas_criticas_activas"]
        ].rename(
            columns={
                "ubigeo": "UBIGEO",
                "departamento": "Departamento",
                "provincia": "Provincia",
                "distrito": "Distrito",
                "dist_epicentral_km": "Distancia_Epicentral_km",
                "pga_rock_gal": "Aceleracion_Roca_Gal",
                "pga_site_gal": "Aceleracion_Real_Suelo_Gal",
                "pga_site_pct_g": "Aceleracion_Porcentaje_Gravedad",
                "site_amp_normal": "Factor_Suelo_Sismos_Normales",
                "site_amp_eff": "Factor_Suelo_Efectivo_NoLineal",
                "mercalli_romano": "Intensidad_Mercalli",
                "mercalli_mag_eq": "Rango_Magnitud_Equivalente_Mw",
                "nivel_impacto": "Nivel_Impacto",
                "n_zonas_criticas_activas": "Zonas_Criticas_GEOCATMIN",
            }
        )
        st.download_button(
            "⬇️ Descargar simulación completa de los 1,893 distritos (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"simulacion_sismo_peru_Mw{ev_mw:.1f}.csv",
            mime="text/csv",
            width="stretch",
        )


def _render_sec5_history_and_3d(data: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 📜 Historia Sísmica del Perú (1552–2026) y Corte 3D de la Placa de Nazca")
    st.write(
        "Explora 474 años de sismos documentados por el IGP y gira el modelo 3D —con el mapa oficial del Perú sobre "
        "la superficie (`0 km`)— para ver cómo la placa de Nazca se hunde bajo la cordillera y la selva."
    )

    df_cat = data["df_cat"]
    inei = _cached_inei_boundaries()

    f1, f2, f3, f4 = st.columns([1.3, 1.2, 1.2, 1.1])
    yr_range = f1.slider("Rango de años", 1552, 2026, (1552, 2026))
    zona_sel = f2.selectbox("Macro-zona geográfica", ["Todo el Perú", "Norte (Tumbes a Áncash)", "Centro (Lima, Ica, Sierra Central)", "Sur (Arequipa, Moquegua, Tacna, Cusco)"])
    reg_opts = ["Todos los tipos"] + sorted(df_cat["regimen_es"].unique().tolist())
    reg_sel = f3.selectbox("Tipo de origen sísmico", reg_opts)
    min_mag = f4.slider("Magnitud mínima (Mw)", 3.0, 8.0, 4.5, 0.1)

    sub = df_cat[(df_cat["anio"].between(yr_range[0], yr_range[1])) & (df_cat["mag"] >= min_mag)].copy()
    if zona_sel != "Todo el Perú":
        sub = sub[sub["macro_zona"] == zona_sel]
    if reg_sel != "Todos los tipos":
        sub = sub[sub["regimen_es"] == reg_sel]

    n_total = len(sub)
    n_destr = int((sub["mag"] >= 7.5).sum())
    max_mag = float(sub["mag"].max()) if n_total > 0 else 0.0
    max_yr = int(sub.loc[sub["mag"].idxmax(), "anio"]) if n_total > 0 else 0
    inst_sens = sub[(sub["anio"] >= 2000) & (sub["mag"] >= 4.5)]
    yrs_mod = max(min(yr_range[1], 2026) - max(yr_range[0], 2000) + 1, 1)
    avg_yr = len(inst_sens) / yrs_mod

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(render_kpi_card("Sismos Registrados", f"{n_total:,}", f"Entre {yr_range[0]} y {yr_range[1]} (M ≥ {min_mag:.1f})", "#0F766E"), unsafe_allow_html=True)
    k2.markdown(render_kpi_card("Terremotos Destructivos", f"{n_destr}", "Eventos con magnitud M ≥ 7.5", "#B91C1C"), unsafe_allow_html=True)
    k3.markdown(render_kpi_card("Sismo Más Fuerte", f"Mw {max_mag:.1f}" if n_total else "—", f"Registrado en el año {max_yr}" if n_total else "Sin datos", "#991B1B"), unsafe_allow_html=True)
    k4.markdown(render_kpi_card("Promedio Anual Moderno", f"{avg_yr:.0f} / año", "Sismos sensibles M ≥ 4.5 (desde 2000)", "#B45309"), unsafe_allow_html=True)

    st.markdown("")
    maj = sub[sub["mag"] >= 6.5].copy()
    fig_tl = go.Figure()
    if not maj.empty:
        fig_tl.add_trace(
            go.Scatter(
                x=maj["anio"],
                y=maj["mag"],
                mode="markers",
                marker=dict(
                    size=np.clip((maj["mag"] - 5.8) * 6.0, 6, 22),
                    color=maj["mag"],
                    colorscale="YlOrRd",
                    cmin=6.5,
                    cmax=9.0,
                    line=dict(color="#1C1917", width=0.7),
                ),
                text=[
                    f"<b>Año {int(r['anio'])}</b> · Magnitud Mw {r['mag']:.1f}<br>Origen: {r['regimen_es']}<br>Profundidad: {r['depth_km']:.0f} km"
                    if pd.notna(r["depth_km"])
                    else f"<b>Año {int(r['anio'])}</b> · Magnitud Mw {r['mag']:.1f}<br>Origen: {r['regimen_es']}"
                    for _, r in maj.iterrows()
                ],
                hoverinfo="text",
                name="Grandes terremotos (M ≥ 6.5)",
            )
        )
        ann_bg = "rgba(30,41,59,0.88)" if theme_mode == THEME_DARK else "rgba(255,255,255,0.88)"
        ann_fg = "#F8FAFC" if theme_mode == THEME_DARK else "#1C1917"
        for yr_m, mag_m, txt_m in HISTORICAL_MILESTONES:
            if yr_range[0] <= yr_m <= yr_range[1] and mag_m >= min_mag:
                fig_tl.add_annotation(
                    x=yr_m,
                    y=mag_m,
                    text=txt_m,
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30 if yr_m not in (1940, 1974, 2007) else 28,
                    font=dict(size=10, color=ann_fg),
                    bgcolor=ann_bg,
                    bordercolor="#64748B",
                )
    fig_tl.update_layout(
        **plotly_theme_kwargs(theme_mode),
        height=385,
        title="Línea de Tiempo de Grandes Terremotos del Perú (Magnitud ≥ 6.5)",
        xaxis_title="Año",
        yaxis_title="Magnitud (Mw)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_tl, width="stretch", config=PLOTLY_CFG)

    # 3D Wadati-Benioff Cutaway WITH Peru INEI contour at z=0 km and SW->NE camera
    st.markdown("#### Corte 3D de la Placa de Nazca bajo el Perú (Geometría Wadati–Benioff)")
    st.info(
        "🧭 **¿Por qué hay volcanes en Arequipa y Tacna, pero no en Lima o Piura?** "
        "Mira el mapa del Perú dibujado en la superficie (`0 km`): al **Norte y Centro** (arriba de -15°S), la placa de Nazca "
        "entra casi horizontal (**11°–12°**), bloqueando el vulcanismo. Al **Sur de Ica/Nasca**, se hunde bruscamente (**30.1°**) "
        "hasta 269 km de profundidad, fundiendo roca y alimentando los volcanes del sur."
    )
    sub_3d = sub[(sub["depth_km"].notna()) & (sub["depth_km"] <= 350.0)].copy()
    if len(sub_3d) > 3500:
        sub_3d = sub_3d.sample(3500, random_state=42)
    color_map_reg = {
        "Subducción Costera": "#DC2626",
        "Falla Cortical Andina": "#F59E0B",
        "Losa Intermedia": "#14B8A6",
        "Foco Profundo": "#3B82F6",
    }
    fig_3d = go.Figure()

    # Overlay Peru INEI departmental outline on z = 0 km surface plane
    if inei["peru_3d_x"]:
        fig_3d.add_trace(
            go.Scatter3d(
                x=inei["peru_3d_x"],
                y=inei["peru_3d_y"],
                z=inei["peru_3d_z"],
                mode="lines",
                name="Contorno del Perú (0 km)",
                line=dict(color="#F8FAFC" if theme_mode == THEME_DARK else "#1C1917", width=2.2),
                hoverinfo="skip",
            )
        )

    for reg_lbl, col_hex in color_map_reg.items():
        r_df = sub_3d[sub_3d["regimen_es"] == reg_lbl]
        if r_df.empty:
            continue
        fig_3d.add_trace(
            go.Scatter3d(
                x=r_df["lon"],
                y=r_df["lat"],
                z=-r_df["depth_km"],
                mode="markers",
                name=reg_lbl,
                marker=dict(size=2.8, color=col_hex, opacity=0.72),
                text=[f"Año {int(a)} | Mw {m:.1f} | Prof. {d:.0f} km" for a, m, d in zip(r_df["anio"], r_df["mag"], r_df["depth_km"])],
                hoverinfo="text",
            )
        )
    fig_3d.update_layout(
        **plotly_theme_kwargs(theme_mode),
        height=560,
        scene=dict(
            xaxis_title="Longitud (°W — Mar → Selva)",
            yaxis_title="Latitud (°S — Sur → Norte)",
            zaxis_title="Profundidad subterránea (-km)",
            aspectratio=dict(x=1.0, y=1.35, z=0.60),
            camera=dict(eye=dict(x=-1.65, y=-1.35, z=0.82)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=35, b=0),
    )
    st.plotly_chart(fig_3d, width="stretch", config=PLOTLY_CFG)
    st.markdown(
        render_map_legend(
            [
                ("#DC2626", "Rojo — Subducción Costera", "Choque superficial (<60 km) frente al mar"),
                ("#F59E0B", "Ámbar — Falla Cortical Andina", "Sismos superficiales (<60 km) dentro de la cordillera"),
                ("#14B8A6", "Verde azulado — Losa Intermedia", "Interior de la placa de Nazca (60 a 350 km bajo tierra)"),
                ("#1C1917", "Línea superficial (0 km)", "Contorno oficial del Perú (INEI) proyectado sobre la superficie"),
            ],
            note="💡 Usa el ícono de expansión en la esquina superior derecha del gráfico 3D para verlo en pantalla completa y arrastra con el mouse para rotarlo.",
        ),
        unsafe_allow_html=True,
    )

    if not sub.empty:
        st.markdown("**Top 10 Sismos Más Fuertes del Filtro Actual**")
        top_hist = sub.sort_values("mag", ascending=False).head(10)
        tbl_h = pd.DataFrame(
            {
                "Fecha (UTC)": top_hist["timestamp_utc"].dt.strftime("%Y-%m-%d"),
                "Magnitud": [f"Mw {v:.1f}" for v in top_hist["mag"]],
                "Profundidad": [f"{v:.0f} km" if pd.notna(v) else "No estimada" for v in top_hist["depth_km"]],
                "Tipo de Origen": top_hist["regimen_es"],
                "Macro-Zona": top_hist["macro_zona"],
                "Catálogo": top_hist["periodo"],
            }
        )
        st.markdown(render_centered_table(tbl_h, max_height_px=340), unsafe_allow_html=True)
        st.markdown(
            render_table_legend(
                [
                    ("Fecha (UTC)", "Fecha universal coordinada del terremoto según el catálogo oficial del IGP."),
                    ("Magnitud (Mw)", "Energía total del terremoto homogeneizada a la escala de momento."),
                    ("Profundidad", "Kilómetros bajo la superficie donde inició la ruptura; los menores a 60 km son los más sentidos."),
                    ("Catálogo", "Histórico (1552–1959, crónicas e isosistas) vs. Instrumental (1960–2026, red de sismógrafos)."),
                ]
            ),
            unsafe_allow_html=True,
        )


def _render_sec6_gaps(data: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🌊 Radar de Silencios Sísmicos en la Costa del Perú")
    st.write(
        "Mediante **10,000 simulaciones Monte Carlo**, estimamos cuánta energía elástica se ha acumulado en cada tramo del litoral "
        "desde su último megaterremoto y la probabilidad condicional de una gran ruptura (Mw ≥ 8.0) en los próximos 30 y 50 años."
    )

    df_gaps = data["df_gaps"].copy()
    c1, c2 = st.columns([1.2, 1.2])
    vista = c1.radio("Nivel de detalle", ["5 Grandes Bloques Costeros", "30 Tramos Detallados del Litoral (cada ~60 km)"], horizontal=True)
    modelo_bpt = c2.radio(
        "Supuesto de recurrencia (BPT)",
        ["Conservador — irregularidad normal (α = 0.5)", "Cuasi-periódico — ciclos más regulares (α = 0.3)"],
        horizontal=True,
    )
    p30_col = "bpt_p30_a05" if "0.5" in modelo_bpt else "bpt_p30_a03"
    p50_col = "bpt_p50_a05" if "0.5" in modelo_bpt else "bpt_p50_a03"

    if "5 Grandes" in vista:
        sub = df_gaps[df_gaps["entity_type"] == "macro_block"].copy()
        sub["nombre_es"] = sub["entity_id"].map(MACRO_BLOCK_LABELS_ES).fillna(sub["coastal_zone"])
        sub = sub.sort_values("mw_pot_p50", ascending=False)

        col_l, col_r = st.columns(2)
        with col_l:
            fig_mw = go.Figure()
            fig_mw.add_trace(
                go.Bar(
                    x=sub["nombre_es"],
                    y=sub["mw_pot_p50"],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=sub["mw_pot_p90"] - sub["mw_pot_p50"],
                        arrayminus=sub["mw_pot_p50"] - np.maximum(sub["mw_pot_p10"], 6.5),
                        color="#F8FAFC" if theme_mode == THEME_DARK else "#1C1917",
                    ),
                    marker_color="#B91C1C",
                    text=[f"Mw {v:.2f}" for v in sub["mw_pot_p50"]],
                    textposition="outside",
                    name="Magnitud potencial mediana",
                )
            )
            fig_mw.update_yaxes(range=[7.5, 9.05], title_text="Magnitud potencial acumulada (Mw)")
            fig_mw.update_layout(**plotly_theme_kwargs(theme_mode), height=415, title="Energía Acumulada (Mediana y rango P10–P90)", margin=dict(l=40, r=20, t=50, b=90))
            st.plotly_chart(fig_mw, width="stretch", config=PLOTLY_CFG)

        with col_r:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Bar(x=sub["nombre_es"], y=sub[p30_col], name="Próximos 30 años (%)", marker_color="#0F766E", text=[f"{v:.0f}%" for v in sub[p30_col]], textposition="outside"))
            fig_pr.add_trace(go.Bar(x=sub["nombre_es"], y=sub[p50_col], name="Próximos 50 años (%)", marker_color="#D97706", text=[f"{v:.0f}%" for v in sub[p50_col]], textposition="outside"))
            fig_pr.update_yaxes(range=[0, 100], title_text="Probabilidad condicional (%)")
            fig_pr.update_layout(**plotly_theme_kwargs(theme_mode), height=415, barmode="group", title="Probabilidad de Gran Terremoto (Mw ≥ 8.0)", margin=dict(l=40, r=20, t=50, b=90))
            st.plotly_chart(fig_pr, width="stretch", config=PLOTLY_CFG)
    else:
        sub = df_gaps[df_gaps["entity_type"] == "segment"].sort_values("lat_center", ascending=False).copy()
        sub["nombre_es"] = sub["coastal_zone"]
        p10_floor = np.maximum(sub["mw_pot_p10"].to_numpy(dtype="float64"), 6.5)

        fig_seg = go.Figure()
        fig_seg.add_trace(go.Scatter(x=sub["mw_pot_p90"], y=sub["lat_center"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_seg.add_trace(
            go.Scatter(
                x=p10_floor,
                y=sub["lat_center"],
                mode="lines",
                fill="tonextx",
                fillcolor="rgba(185,28,28,0.22)",
                line=dict(width=0),
                name="Rango de incertidumbre (P10–P90)",
            )
        )
        seg_hover = [
            (
                f"<b>{r['coastal_zone']}</b><br>Mediana: Mw {r['mw_pot_p50']:.2f}<br>"
                + ("Rango: Sin déficit mayor (ruptura reciente)" if float(r["mw_pot_p10"]) < 6.5 else f"Rango: Mw {float(r['mw_pot_p10']):.2f}")
                + f" a Mw {r['mw_pot_p90']:.2f}<br>"
                + f"Prob. 30 años: {r[p30_col]:.1f}% | Prob. 50 años: {r[p50_col]:.1f}%"
            )
            for _, r in sub.iterrows()
        ]
        fig_seg.add_trace(
            go.Scatter(
                x=sub["mw_pot_p50"],
                y=sub["lat_center"],
                mode="lines+markers",
                name="Magnitud potencial mediana",
                line=dict(color="#B91C1C", width=2.5),
                text=seg_hover,
                hoverinfo="text",
            )
        )
        fig_seg.update_xaxes(title_text="Magnitud potencial acumulada (Mw, piso visual 6.5)", range=[6.4, 8.85])
        fig_seg.update_yaxes(title_text="Latitud del litoral peruano (°S)")
        fig_seg.update_layout(**plotly_theme_kwargs(theme_mode), height=460, title="Perfil de Energía Acumulada de Norte a Sur (-3.5°S a -18.5°S)")
        st.plotly_chart(fig_seg, width="stretch", config=PLOTLY_CFG)

    tbl = pd.DataFrame(
        {
            "Tramo Costero": sub["nombre_es"],
            "Último Gran Terremoto": [f"Año {int(round(y))}" if pd.notna(y) else "Sin registro desde 1552" for y in sub["last_major_rupture_year"]],
            "Años Acumulando": [f"{int(round(y))} años" for y in sub["years_since_major_rupture"]],
            "Magnitud Potencial": [f"Mw {v:.2f}" for v in sub["mw_pot_p50"]],
            "Rango P10 – P90": [
                f"Sin déficit mayor – Mw {hi:.2f}" if lo < 6.5 else f"Mw {lo:.2f} – {hi:.2f}"
                for lo, hi in zip(sub["mw_pot_p10"], sub["mw_pot_p90"])
            ],
            "Prob. 30 Años": [f"{v:.1f}%" for v in sub[p30_col]],
            "Prob. 50 Años": [f"{v:.1f}%" for v in sub[p50_col]],
            "Actividad Reciente (/100 km)": [f"{v:.1f}" for v in sub["modern_density_per_100km"]],
        }
    )
    st.markdown(render_centered_table(tbl, max_height_px=450), unsafe_allow_html=True)
    st.markdown(
        render_table_legend(
            [
                ("Magnitud Potencial (Mw)", "Tamaño aproximado del terremoto que se liberaría si toda la energía acumulada en ese tramo saliera en una sola ruptura."),
                ("Rango P10 – P90", "Intervalo de confianza del 80% obtenido tras 10,000 simulaciones Monte Carlo (refleja la incertidumbre física del acoplamiento de placas)."),
                ("Prob. 30 y 50 Años (BPT)", "Probabilidad condicional de que ocurra una gran ruptura (Mw ≥ 8.0) dado el tiempo ya transcurrido sin terremoto."),
                ("Actividad Reciente (/100 km)", "Sismos moderados M≥4.7 entre 2000 y 2026 por cada 100 km de costa; valores bajos en zonas de alto déficit confirman silencio sísmico."),
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_sec7_low_hazard(assets: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🛡️ Zonas de Menor Peligro Sísmico del Perú (Análisis Territorial)")
    st.write(
        "No todo el Perú tiene el mismo nivel de amenaza sísmica. Aquí analizamos los departamentos y distritos más "
        "tranquilos sísmicamente —principalmente en la llanura amazónica alejada de la fosa marina— **contrastándolos "
        "con sus peligros geológicos e hidrometeorológicos** para mantener rigor multi-amenaza."
    )

    df_u = assets["df_u"].copy()
    low_df = df_u[df_u["nivel_peligro_sismico_puro"].isin(["Muy Bajo", "Bajo"])].copy()
    lowest_dist = df_u.sort_values("seismic_hazard_index_0_100", ascending=True).iloc[0]

    dep_agg = (
        df_u.groupby("departamento", as_index=False)
        .agg(
            n_dist=("ubigeo", "count"),
            mean_idx=("seismic_hazard_index_0_100", "mean"),
            mean_pga=("pga_escenario_max_gal", "mean"),
            peligros_geo=("n_peligros_total", "sum"),
            zonas_crit=("n_zonas_criticas_activas", "sum"),
        )
        .sort_values("mean_idx", ascending=True)
    )
    lowest_dep = dep_agg.iloc[0]
    pct_low = len(low_df) / len(df_u) * 100.0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        render_kpi_card(
            "Distrito Más Tranquilo",
            f"{lowest_dist['distrito']}",
            f"{lowest_dist['departamento']} · Índice {lowest_dist['seismic_hazard_index_0_100']:.1f}/100 ({lowest_dist['pga_escenario_max_gal']:.1f} Gal)",
            "#0F766E",
        ),
        unsafe_allow_html=True,
    )
    k2.markdown(
        render_kpi_card(
            "Dpto. Menor Peligro Sísmico",
            f"{lowest_dep['departamento']}",
            f"Promedio {lowest_dep['mean_idx']:.1f}/100 ({int(lowest_dep['n_dist'])} distritos)",
            "#0284C7",
        ),
        unsafe_allow_html=True,
    )
    k3.markdown(
        render_kpi_card(
            "Distritos Muy Bajo / Bajo",
            f"{len(low_df):,} distritos",
            f"{pct_low:.1f}% del territorio nacional (142 Muy Bajo + 565 Bajo)",
            "#0D9488",
        ),
        unsafe_allow_html=True,
    )
    k4.markdown(
        render_kpi_card(
            "Otros Peligros en Estas Zonas",
            f"{int(low_df['n_peligros_total'].sum()):,} puntos",
            f"{int(low_df['n_zonas_criticas_activas'].sum()):,} zonas críticas INGEMMET (huaicos/inundaciones)",
            "#7C3AED",
            "Menor peligro sísmico NO significa ausencia de otros desastres naturales como inundaciones, erosión fluvial o deslizamientos.",
        ),
        unsafe_allow_html=True,
    )

    st.info(
        "⚖️ **Mirada multi-amenaza con rigor:** Aunque **Madre de Dios** y **Loreto** tienen los índices sísmicos más bajos "
        "del Perú (lejos de la costa), muchos de sus distritos figuran en **FONDES Alto o Muy Alto** debido a **inundaciones "
        "amazónicas, erosión fluvial y deslizamientos** en la selva alta (como San Martín con 3,029 peligros geológicos)."
    )

    c_chart, c_map = st.columns([1.05, 1.05])
    with c_chart:
        top15_dep = dep_agg.head(15).iloc[::-1]
        fig_dep = go.Figure()
        fig_dep.add_trace(
            go.Bar(
                y=top15_dep["departamento"],
                x=top15_dep["mean_idx"],
                orientation="h",
                marker_color="#0F766E",
                text=[f"{idx:.1f}/100 · {int(pg):,} peligros geo." for idx, pg in zip(top15_dep["mean_idx"], top15_dep["peligros_geo"])],
                textposition="outside",
                hovertext=[
                    f"<b>{r['departamento']}</b><br>"
                    f"Índice Sísmico promedio: <b>{r['mean_idx']:.1f} / 100</b><br>"
                    f"Sacudimiento máx. promedio: {r['mean_pga']:.1f} Gal ({r['mean_pga']/9.81:.1f}% g)<br>"
                    f"Peligros geológicos INGEMMET: <b>{int(r['peligros_geo']):,}</b> ({int(r['zonas_crit'])} zonas críticas)"
                    for _, r in top15_dep.iterrows()
                ],
                hoverinfo="text",
            )
        )
        fig_dep.update_xaxes(range=[0, 72], title_text="Índice Sísmico Promedio (0 = mínimo, 100 = máximo)")
        fig_dep.update_layout(
            **plotly_theme_kwargs(theme_mode),
            height=485,
            title="Top 15 Departamentos con Menor Peligro Sísmico Promedio (vs. Peligros Geológicos)",
            margin=dict(l=10, r=40, t=48, b=40),
        )
        st.plotly_chart(fig_dep, width="stretch", config=PLOTLY_CFG)
        st.caption("💡 Usa el ícono de expansión en la esquina superior derecha del gráfico para verlo en pantalla completa.")

    with c_map:
        st.markdown("**Mapa de los 707 Distritos en Peligro Sísmico Muy Bajo y Bajo**")
        low_map_html = _build_low_hazard_folium_html(theme_mode)
        st.iframe(low_map_html, height=420, width="stretch")
        st.markdown(
            render_map_legend(
                [
                    ("#0F766E", "Verde petróleo — Muy Bajo (142 distritos)", "Principalmente Amazonía oriental y fronteras alejadas de la costa"),
                    ("#0284C7", "Azul — Bajo (565 distritos)", "Sierra oriental, altiplano y selva alta con menor sacudimiento en megaterremotos"),
                ]
            ),
            unsafe_allow_html=True,
        )

    st.markdown("#### Top 25 Distritos con Menor Peligro Sísmico del Perú")
    dep_opts = ["Todo el Perú"] + sorted(df_u["departamento"].unique().tolist())
    sel_d_low = st.selectbox("Filtrar ranking por departamento (opcional)", dep_opts, index=0)
    ranked_all = df_u.sort_values("seismic_hazard_index_0_100", ascending=True).copy()
    sub_low = ranked_all if sel_d_low == "Todo el Perú" else ranked_all[ranked_all["departamento"] == sel_d_low]
    top25 = sub_low.head(25)

    tbl_low = pd.DataFrame(
        {
            "Distrito": top25["distrito"],
            "Provincia / Dpto.": top25["provincia"] + " / " + top25["departamento"],
            "Índice Sísmico (0–100)": [f"{v:.2f}" for v in top25["seismic_hazard_index_0_100"]],
            "Nivel Sísmico": top25["nivel_peligro_sismico_puro"],
            "Sacudimiento Máx.": [f"{v/9.81:.1f}% g ({v:.1f} Gal)" for v in top25["pga_escenario_max_gal"]],
            "Mercalli + Rango Mw": [format_mercalli_with_mag(m) for m in top25["mmi_escenario_max"]],
            "Peligros Geológicos (Flujos / Caídas)": [
                f"{int(t)} total ({int(fl)} flujos / {int(ca)} caídas)"
                for t, fl, ca in zip(top25["n_peligros_total"], top25["n_peligros_flujo"], top25["n_peligros_caida"])
            ],
            "Clasificación FONDES": top25["nivel_riesgo_multipeligro_sigrid"],
        }
    )
    st.markdown(
        render_centered_table(tbl_low, badge_cols={"Nivel Sísmico", "Clasificación FONDES"}, max_height_px=450),
        unsafe_allow_html=True,
    )
    st.markdown(
        render_table_legend(
            [
                ("Índice Sísmico (0–100)", "Puntaje relativo de amenaza sísmica física (0 = mínimo del país, 100 = máximo); no es un porcentaje."),
                ("Sacudimiento Máx. (%g y Gal)", "Aceleración máxima esperada entre los 3 escenarios nacionales; debajo de 1% g (<10 Gal) suele ser imperceptible o muy débil."),
                ("Mercalli + Rango Mw", "Intensidad estimada en el distrito y rango de Magnitud Mw equivalente en epicentro."),
                ("Peligros Geológicos (Flujos / Caídas)", "Puntos registrados por INGEMMET de huaicos/flujos de lodo y derrumbes/caídas de rocas."),
                ("Clasificación FONDES", "Nivel oficial multi-peligro de CENEPRED/SIGRID (incluye lluvias, inundaciones y heladas, por eso puede ser Alto aun con bajo peligro sísmico)."),
            ]
        ),
        unsafe_allow_html=True,
    )

    export_low = ranked_all[
        [
            "ubigeo", "departamento", "provincia", "distrito",
            "seismic_hazard_index_0_100", "nivel_peligro_sismico_puro",
            "pga_escenario_max_gal", "mmi_escenario_max",
            "n_peligros_total", "n_peligros_flujo", "n_peligros_caida",
            "n_zonas_criticas_activas", "nivel_riesgo_multipeligro_sigrid",
        ]
    ].rename(
        columns={
            "ubigeo": "UBIGEO",
            "departamento": "Departamento",
            "provincia": "Provincia",
            "distrito": "Distrito",
            "seismic_hazard_index_0_100": "Indice_Peligro_Sismico_0_100",
            "nivel_peligro_sismico_puro": "Nivel_Peligro_Sismico",
            "pga_escenario_max_gal": "Sacudimiento_Maximo_Gal",
            "mmi_escenario_max": "Intensidad_Mercalli_Numerica",
            "n_peligros_total": "Peligros_Geologicos_Total",
            "n_peligros_flujo": "Peligros_Flujos_Huaicos",
            "n_peligros_caida": "Peligros_Caidas_Rocas",
            "n_zonas_criticas_activas": "Zonas_Criticas_Activas",
            "nivel_riesgo_multipeligro_sigrid": "Clasificacion_Oficial_FONDES",
        }
    )
    st.download_button(
        "📥 Descargar ranking completo de los 1,893 distritos ordenados de menor a mayor peligro (CSV)",
        data=export_low.to_csv(index=False).encode("utf-8"),
        file_name="ranking_menor_a_mayor_peligro_sismico_peru_1893_distritos.csv",
        mime="text/csv",
        width="stretch",
    )


def _render_sec8_audit(assets: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🏛️ Auditoría Territorial: Peligro Sísmico Real vs. Clasificación Oficial (FONDES / SIGRID)")
    st.info(
        "🏛️ **¿Qué es FONDES / SIGRID y por qué encontramos una brecha?** "
        "El padrón FONDES usa el índice multi-peligro de SIGRID para priorizar fondos de prevención. Al promediar lluvias, "
        "inundaciones y huaicos, **150 distritos con Peligro Sísmico Alto o Muy Alto (35.2% del total)** aparecen clasificados "
        "como *'Muy Bajo'* o *'Bajo'* — incluyendo **21 de los 50 distritos de Lima y Callao**."
    )

    df_u = assets["df_u"].copy()
    df_u["busqueda"] = df_u["distrito"] + " — " + df_u["provincia"] + ", " + df_u["departamento"] + " (UBIGEO " + df_u["ubigeo"] + ")"
    default_idx = int(df_u.index[df_u["ubigeo"] == "150142"][0]) if (df_u["ubigeo"] == "150142").any() else 0
    sel_lbl = st.selectbox("🔎 Busca cualquier distrito del Perú (1,893 distritos)", df_u["busqueda"].tolist(), index=default_idx)
    r = df_u[df_u["busqueda"] == sel_lbl].iloc[0]

    nivel_seis = str(r["nivel_peligro_sismico_puro"])
    nivel_fondes = str(r["nivel_riesgo_multipeligro_sigrid"])
    is_gap = nivel_seis in ("Alto", "Muy Alto") and nivel_fondes in ("Muy Bajo", "Bajo")

    if is_gap:
        st.warning(
            f"⚠️ **Alerta de Subestimación Oficial:** **{r['distrito']}** tiene Peligro Sísmico Puro **{nivel_seis}** "
            f"(Índice `{r['seismic_hazard_index_0_100']:.1f}/100`, sacudimiento estimado de **{r['pga_escenario_max_gal']/9.81:.0f}% g / {r['pga_escenario_max_gal']:.0f} Gal**), "
            f"pero figura como **{nivel_fondes}** en el listado oficial FONDES / SIGRID."
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(render_kpi_card("Peligro Sísmico Puro", nivel_seis, f"Índice {r['seismic_hazard_index_0_100']:.1f} / 100", RISK_LEVEL_COLORS.get(nivel_seis, "#B91C1C")), unsafe_allow_html=True)
    c2.markdown(render_kpi_card("Oficial FONDES / SIGRID", nivel_fondes, "Índice multi-peligro estatal", RISK_LEVEL_COLORS.get(nivel_fondes, "#475569")), unsafe_allow_html=True)
    m_rom, _, _, m_col, m_mag_rng = describe_mercalli_dual(float(r["mmi_escenario_max"]))
    c3.markdown(render_kpi_card("Escenario Crítico", f"{r['pga_escenario_max_gal'] / 9.81:.0f}% g", f"{r['pga_escenario_max_gal']:.0f} Gal · Mercalli {m_rom} ({m_mag_rng})", m_col), unsafe_allow_html=True)
    st_lbl = f"Estación IGP {r['primary_station_code']}" if pd.notna(r["primary_station_code"]) else "Sin estación (neutro)"
    c4.markdown(render_kpi_card("Efecto de Suelo Normal", f"{r['site_amp_factor_vs_msol']:.2f}×", st_lbl, "#B45309"), unsafe_allow_html=True)
    c5.markdown(render_kpi_card("Peligros Geológicos", f"{int(r['n_peligros_total'])} registros", f"{int(r['n_zonas_criticas_activas'])} zonas críticas activas", "#6D28D9"), unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([0.90, 1.20])
    with left:
        order = ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]
        ct = pd.crosstab(df_u["nivel_peligro_sismico_puro"], df_u["nivel_riesgo_multipeligro_sigrid"]).reindex(index=order, columns=order, fill_value=0)
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=ct.to_numpy(),
                x=order,
                y=order,
                text=ct.to_numpy().astype(str),
                texttemplate="%{text}",
                colorscale="YlOrRd",
                showscale=False,
            )
        )
        fig_hm.update_layout(
            **plotly_theme_kwargs(theme_mode),
            height=405,
            title="Cruce de los 1,893 Distritos: Sísmico Real vs. FONDES",
            xaxis_title="Clasificación Oficial FONDES / SIGRID",
            yaxis_title="Peligro Sísmico Puro (Esta Plataforma)",
            margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(fig_hm, width="stretch", config=PLOTLY_CFG)

    with right:
        under = df_u[
            df_u["nivel_peligro_sismico_puro"].isin(["Alto", "Muy Alto"])
            & df_u["nivel_riesgo_multipeligro_sigrid"].isin(["Muy Bajo", "Bajo"])
        ].sort_values("seismic_hazard_index_0_100", ascending=False)
        st.markdown(f"**{len(under)} Distritos con Peligro Sísmico Alto/Muy Alto Subestimados en FONDES**")
        tbl_u = pd.DataFrame(
            {
                "UBIGEO": under["ubigeo"],
                "Departamento": under["departamento"],
                "Distrito": under["distrito"],
                "Índice Sísmico": [f"{v:.1f} / 100" for v in under["seismic_hazard_index_0_100"]],
                "Peligro Sísmico": under["nivel_peligro_sismico_puro"],
                "Oficial FONDES": under["nivel_riesgo_multipeligro_sigrid"],
                "Sacudimiento Máx.": [f"{v/9.81:.0f}% g ({v:.0f} Gal)" for v in under["pga_escenario_max_gal"]],
                "Mercalli + Rango Mw": [format_mercalli_with_mag(m) for m in under["mmi_escenario_max"]],
            }
        )
        st.markdown(render_centered_table(tbl_u, badge_cols={"Peligro Sísmico", "Oficial FONDES"}, max_height_px=370), unsafe_allow_html=True)

    st.markdown(
        render_table_legend(
            [
                ("Índice Sísmico (0–100)", "Puntaje físico de amenaza sísmica (sacudimiento 45%, laguna costera 30%, tasa anual 25%); no es probabilidad."),
                ("Peligro Sísmico vs. Oficial FONDES", "Compara nuestra evaluación sísmica pura contra la categoría multi-peligro vigente de CENEPRED/SIGRID."),
                ("Sacudimiento Máx. y Mercalli + Mw", "Aceleración esperada en el escenario más crítico (%g y Gal) junto con su grado Mercalli y rango equivalente en Magnitud Mw."),
            ]
        ),
        unsafe_allow_html=True,
    )

    export_all = pd.DataFrame(
        {
            "UBIGEO": df_u["ubigeo"],
            "Departamento": df_u["departamento"],
            "Provincia": df_u["provincia"],
            "Distrito": df_u["distrito"],
            "Indice_Peligro_Sismico_0_100": df_u["seismic_hazard_index_0_100"],
            "Nivel_Peligro_Sismico_Puro": df_u["nivel_peligro_sismico_puro"],
            "Nivel_Oficial_FONDES_SIGRID": df_u["nivel_riesgo_multipeligro_sigrid"],
            "Sacudimiento_Lima_Mw86_Gal": df_u["pga_escenario_lima_mw86_gal"],
            "Sacudimiento_Sur_Mw84_Gal": df_u["pga_escenario_sur_mw84_gal"],
            "Sacudimiento_Intraplaca_Mw68_Gal": df_u["pga_escenario_intraplaca_mw68_gal"],
            "Factor_Amplificacion_Suelo_vs_Roca": np.round(df_u["site_amp_factor_vs_msol"], 2),
            "Probabilidad_Sismo_M45_12Meses_Pct": np.round(df_u["prob_m45_365d_nbhd_max"] * 100.0, 2),
            "Peligros_Geologicos_GEOCATMIN": df_u["n_peligros_total"],
            "Zonas_Criticas_Activas_GEOCATMIN": df_u["n_zonas_criticas_activas"],
        }
    )
    st.download_button(
        "⬇️ Descargar base territorial completa de los 1,893 distritos (CSV en español)",
        data=export_all.to_csv(index=False).encode("utf-8"),
        file_name="auditoria_peligro_sismico_peru_1893_distritos.csv",
        mime="text/csv",
    )


def _render_sec9_documentation() -> None:
    st.markdown("### 📚 Documentación Técnica Completa y Rigor Estadístico")
    st.write(
        "Transparencia metodológica total para auditores, jurados técnicos y ciudadanos: fechas exactas de corte, "
        "auditoría ETL, formulaciones geofísicas, validación anti-leakage y limitaciones científicas explícitas."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(render_kpi_card("Bondad de Ajuste G–R", "GFT 94.75%", "Mc = 4.7 · b = 1.170 ± 0.010", "#0F766E"), unsafe_allow_html=True)
    k2.markdown(render_kpi_card("GMPE Modelo A (CV)", "R² = 0.807", "GroupKFold-5 · RMSE 0.283", "#B91C1C"), unsafe_allow_html=True)
    k3.markdown(render_kpi_card("Pronóstico H3 (90d)", "ROC-AUC 0.853", "Walk-Forward 4 splits (2018–2025)", "#D97706"), unsafe_allow_html=True)
    k4.markdown(render_kpi_card("Skill H3 (12 meses)", "BSS +0.203", "Mejora vs. Poisson y ETAS", "#6D28D9"), unsafe_allow_html=True)

    st.markdown("")
    if not DOCS_COMPLETE_MD.is_file():
        st.error(f"No se encontró el documento técnico en {DOCS_COMPLETE_MD}")
        return

    md_text = DOCS_COMPLETE_MD.read_text(encoding="utf-8")
    parts = md_text.split("\n## ")
    intro = parts[0]
    chapters: dict[str, str] = {"📖 Documento Completo (Todos los Capítulos)": md_text}
    for part in parts[1:]:
        first_line = part.splitlines()[0].strip()
        chapters[f"Capítulo {first_line}"] = "## " + part

    nav_col, dl_col = st.columns([1.45, 0.55])
    sel_chap = nav_col.selectbox("📑 Índice navegable de capítulos técnicos", list(chapters.keys()))
    dl_col.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    dl_col.download_button(
        "⬇️ Descargar Documentación (.md)",
        data=md_text.encode("utf-8"),
        file_name="DOCUMENTACION_TECNICA_COMPLETA_PERU_SISMOS.md",
        mime="text/markdown",
        width="stretch",
    )

    st.markdown("---")
    if sel_chap == "📖 Documento Completo (Todos los Capítulos)":
        st.markdown(intro)
        for part in parts[1:]:
            st.markdown("## " + part)
    else:
        st.markdown(chapters[sel_chap])


def main() -> None:
    st.set_page_config(
        page_title="Observatorio Sísmico y Multi-Amenaza del Perú",
        page_icon="🌎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    assets = _cached_sim_assets()
    data = _cached_datasets()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <h2>🌎 Observatorio Sísmico del Perú</h2>
              <p>Plataforma Ciudadana y Científica · 1,893 Distritos</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        theme_mode = st.radio(
            "🎨 Tema Visual",
            [THEME_LIGHT, THEME_DARK],
            index=0,
            horizontal=True,
        )
        section = st.radio(
            "Navegación Principal",
            NAV_SECTIONS,
            index=0,
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div class="sidebar-data-card">
              <h4>📅 Acerca de los Datos (Fechas de Corte)</h4>
              <ul>
                <li><strong>IGP Histórico:</strong> 02/07/1552 – 02/12/1959 (1,098 sismos)</li>
                <li><strong>IGP Instrumental:</strong> 13/01/1960 – 22/11/2026 (25,878 sismos)</li>
                <li><strong>ACELDAT (IGP):</strong> 03/01/2019 – 28/12/2025 (17,794 reg., 277 est.)</li>
                <li><strong>FONDES / SIGRID:</strong> Corte 05/11/2026 (1,893 distritos)</li>
                <li><strong>GEOCATMIN:</strong> 37,023 peligros · 2,177 zonas críticas</li>
                <li><strong>INEI WFS Oficial:</strong> 25 dptos · 196 prov · 1,890 distritos</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(inject_custom_css(theme_mode), unsafe_allow_html=True)
    _render_citizen_header()

    if section == NAV_SECTIONS[0]:
        _render_sec1_seismic_school(data, theme_mode)
    elif section == NAV_SECTIONS[1]:
        _render_sec2_national_map(assets, data, theme_mode)
    elif section == NAV_SECTIONS[2]:
        _render_sec3_district_consultant(assets, data, theme_mode)
    elif section == NAV_SECTIONS[3]:
        _render_sec4_simulator(assets, theme_mode)
    elif section == NAV_SECTIONS[4]:
        _render_sec5_history_and_3d(data, theme_mode)
    elif section == NAV_SECTIONS[5]:
        _render_sec6_gaps(data, theme_mode)
    elif section == NAV_SECTIONS[6]:
        _render_sec7_low_hazard(assets, theme_mode)
    elif section == NAV_SECTIONS[7]:
        _render_sec8_audit(assets, theme_mode)
    else:
        _render_sec9_documentation()


if __name__ == "__main__":
    main()
