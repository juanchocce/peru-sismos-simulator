"""Phase 5E: CivicTech / GovTech Interactive Seismic & Multi-Hazard Platform for Peru (Streamlit).

Self-contained Python + Streamlit web application featuring:
  - Compact single-button Light/Dark theme switcher (`🌙 Cambiar a Modo Oscuro` / `☀️ Cambiar a Modo Claro`).
  - Pedagogically ordered 9-section vertical sidebar navigation.
  - Animated 3D Seismic Wave Propagation Simulator (`Vs ≈ 3.8 km/s`, 12 frames `5s–120s`, underground
    hypocenter, expanding 3D wavefront, Peru INEI surface contour at `z = 0 km`, and soil-amplified
    3D district shaking columns) + 2D Cartographic view toggle.
  - Magnitude-scaled 3D spheres + translucent outer aura (`M ≥ 7.0`) in the 3D Wadati–Benioff cutaway.
  - District Consultant with 5 interactive scenario modes + Housing Vulnerability Guide (Norma E.030/CISMID).
  - Dual Lowest & Highest Seismic Hazard Territorial Analysis (`🟢 Menor` / `🔴 Mayor`).
  - Citizen Confidence Translator in Technical Documentation.
  - 100%-centered zero-clipping tables (`render_centered_table`) + mandatory legend below every table.

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
    REGIME_LABELS_ES,
    REGIME_SHORT_ES,
    RISK_LEVEL_COLORS,
    THEME_DARK,
    THEME_LIGHT,
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
WAVE_VP_KM_S = 6.5
WAVE_VS_KM_S = 3.8
WAVE_FRAME_TIMES_S = [5, 10, 18, 26, 36, 48, 60, 74, 88, 100]
ASSET_CACHE_VERSION = "v5f"

# Pedagogically ordered from easiest citizen guide to deepest technical rigor (9 sections)
NAV_SECTIONS = [
    "🎓 Escuela Sísmica (Empieza Aquí — Guía Fácil)",
    "🗺️ Mapa Nacional Multi-Amenaza",
    "🏘️ Consultor de Mi Distrito (Probabilidad y Riesgo)",
    "⚡ Simulador de Terremotos en Vivo",
    "📜 Historia Sísmica (1552–2026) y Corte 3D",
    "🌊 Radar de Silencios Sísmicos en la Costa",
    "🛡️ Zonas de Menor y Mayor Peligro Sísmico",
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

DISTRICT_SCENARIO_OPTIONS = [
    "Peor escenario crítico para este distrito (automático)",
    "Megaterremoto frente a Lima–Callao (M 8.6 — Brecha de 1746)",
    "Terremoto en el Sur Ilo–Tacna (M 8.4 — Brecha de 1868)",
    "Sismo profundo en Sierra Central (M 6.8 — Estilo Junín)",
    "Simulación personalizada (elige magnitud M 5.0 a M 9.0)",
]

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
        "trench_pts": arr,
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
            "dist_hypocentral_km": np.round(np.sqrt(sim["dist_epicentral_km"] ** 2 + ev_depth_km ** 2), 1),
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


def _nearest_trench_lonlat(assets: dict[str, Any], dist_lat: float, dist_lon: float) -> tuple[float, float, float]:
    """Return nearest PB2002 trench vertex (lon, lat, epicentral_dist_km) with stale-cache fallback."""
    d_xyz = latlon_to_unit_xyz(np.array([dist_lat], dtype="float64"), np.array([dist_lon], dtype="float64"))
    chord_dist, t_idx = assets["trench_tree"].query(d_xyz, k=1)
    idx_int = int(t_idx[0])
    # Exact great-circle distance from unit-sphere chord length
    arc_rad = 2.0 * np.arcsin(np.clip(float(chord_dist[0]) / 2.0, 0.0, 1.0))
    epi_km = float(arc_rad * 6371.0)

    trench_pts = assets.get("trench_pts")
    if trench_pts is not None and len(trench_pts) > idx_int:
        ev_lon, ev_lat = float(trench_pts[idx_int][0]), float(trench_pts[idx_int][1])
        return ev_lon, ev_lat, epi_km

    # Defensive fallback: recover (lon, lat) directly from unit-sphere XYZ stored inside cKDTree
    xyz_pt = np.asarray(assets["trench_tree"].data[idx_int], dtype="float64")
    ev_lat = float(np.degrees(np.arcsin(np.clip(xyz_pt[2], -1.0, 1.0))))
    ev_lon = float(np.degrees(np.arctan2(xyz_pt[1], xyz_pt[0])))
    return ev_lon, ev_lat, epi_km


def _build_3d_animated_wave_figure(
    res_df: pd.DataFrame,
    ev_lat: float,
    ev_lon: float,
    ev_depth_km: float,
    ev_mw: float,
    inei: dict[str, Any],
    theme_mode: str,
) -> go.Figure:
    """Build clean 10-frame Plotly 3D subsoil + P/S surface wave model (zero vertical bars)."""
    cos_lat = max(float(np.cos(np.radians(ev_lat))), 0.25)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * cos_lat

    vis = res_df[res_df["pga_site_gal"] >= 1.8].sort_values("pga_site_gal", ascending=False).head(320).copy()
    if vis.empty:
        vis = res_df.sort_values("pga_site_gal", ascending=False).head(100).copy()

    vis["t_arrival_p_s"] = vis["dist_hypocentral_km"] / WAVE_VP_KM_S
    vis["t_arrival_s_s"] = vis["dist_hypocentral_km"] / WAVE_VS_KM_S

    def _surface_ring(r_km: float, z_plane: float) -> tuple[list[float | None], list[float | None], list[float | None]]:
        if r_km <= ev_depth_km:
            return [ev_lon], [ev_lat], [z_plane]
        r_surf = float(np.sqrt(max(r_km ** 2 - ev_depth_km ** 2, 0.0)))
        ang = np.linspace(0.0, 2.0 * np.pi, 64)
        rx = [float(ev_lon + (r_surf * np.cos(a)) / km_per_deg_lon) for a in ang]
        ry = [float(ev_lat + (r_surf * np.sin(a)) / km_per_deg_lat) for a in ang]
        rz = [z_plane] * len(ang)
        return rx, ry, rz

    def _subsoil_dome_mesh(r_km: float) -> go.Mesh3d:
        """Build a lightweight semi-transparent spherical wave dome in subsoil (z <= 0)."""
        r_eff = max(r_km, 8.0)
        # Upper hemisphere from hypocenter up to surface z = 0
        sin_max = min(ev_depth_km / r_eff, 1.0)
        phi_max = float(np.arcsin(np.clip(sin_max, -1.0, 1.0)))
        n_lat, n_lon = 7, 20
        phis = np.linspace(-np.pi / 2.0, phi_max, n_lat)
        thetas = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

        vx, vy, vz = [], [], []
        for p in phis:
            z_val = min(-ev_depth_km + r_eff * float(np.sin(p)), 0.0)
            rad_h = r_eff * float(np.cos(p))
            for th in thetas:
                vx.append(float(ev_lon + (rad_h * np.cos(th)) / km_per_deg_lon))
                vy.append(float(ev_lat + (rad_h * np.sin(th)) / km_per_deg_lat))
                vz.append(float(z_val))

        ii, jj, kk = [], [], []
        for r_i in range(n_lat - 1):
            for c_i in range(n_lon):
                c_next = (c_i + 1) % n_lon
                a = r_i * n_lon + c_i
                b = r_i * n_lon + c_next
                c = (r_i + 1) * n_lon + c_i
                d = (r_i + 1) * n_lon + c_next
                ii.extend([a, b])
                jj.extend([b, d])
                kk.extend([c, c])

        return go.Mesh3d(
            x=vx,
            y=vy,
            z=vz,
            i=ii,
            j=jj,
            k=kk,
            color="#EF4444",
            opacity=0.18,
            name="Domo de onda subterránea",
            hoverinfo="skip",
            showlegend=True,
        )

    def _surface_districts_trace(t_sec: float) -> go.Scatter3d:
        """Render district markers resting directly on z = 0 (zero vertical bars)."""
        reached = vis[vis["t_arrival_s_s"] <= t_sec]
        if not reached.empty:
            sx = reached["dist_lon"].tolist()
            sy = reached["dist_lat"].tolist()
            sz = [0.0] * len(reached)
            sc = reached["mmi_est"].tolist()
            ss = np.clip(4.8 + np.sqrt(np.maximum(reached["pga_site_pct_g"].to_numpy(dtype="float64"), 0.0)) * 1.25, 4.5, 16.0).tolist()
            stxt = [
                f"<b>{r['distrito']} ({r['departamento']})</b><br>"
                f"Llegada Onda P: <b>{r['t_arrival_p_s']:.1f} s</b> · Onda S: <b>{r['t_arrival_s_s']:.1f} s</b><br>"
                f"Sacudimiento real: <b>{r['pga_site_pct_g']:.1f}% g ({r['pga_site_gal']:.0f} Gal)</b><br>"
                f"Intensidad: <b>Mercalli {r['mercalli_romano']} (eq. {r['mercalli_mag_eq']}) — {r['nivel_impacto']}</b><br>"
                f"Suelo: {r['site_amp_normal']:.2f}× normal → {r['site_amp_eff']:.2f}× efectivo"
                for _, r in reached.iterrows()
            ]
        else:
            sx, sy, sz, sc, ss, stxt = [ev_lon], [ev_lat], [0.0], [3.0], [1.0], ["Onda S aún viajando hacia la superficie..."]

        return go.Scatter3d(
            x=sx,
            y=sy,
            z=sz,
            mode="markers",
            name="Distritos alcanzados (z = 0 km)",
            marker=dict(
                size=ss,
                color=sc,
                colorscale="YlOrRd",
                cmin=3.0,
                cmax=9.5,
                colorbar=dict(title=dict(text="Mercalli", side="right"), thickness=13, len=0.62, y=0.52),
                opacity=0.90,
                line=dict(color="#1C1917", width=0.5),
            ),
            text=stxt,
            hoverinfo="text",
        )

    outline_col = "#E2E8F0" if theme_mode == THEME_DARK else "#292524"
    trace_peru = go.Scatter3d(
        x=inei["peru_3d_x"],
        y=inei["peru_3d_y"],
        z=inei["peru_3d_z"],
        mode="lines",
        name="Contorno Perú (0 km)",
        line=dict(color=outline_col, width=2.0),
        hoverinfo="skip",
    )
    trace_hypo = go.Scatter3d(
        x=[ev_lon, ev_lon],
        y=[ev_lat, ev_lat],
        z=[-ev_depth_km, 0.0],
        mode="lines+markers",
        name=f"Hipocentro (-{ev_depth_km:.0f} km)",
        line=dict(color="#DC2626", width=4.0, dash="dot"),
        marker=dict(size=[13, 8], color=["#DC2626", "#F97316"], symbol=["circle", "diamond"]),
        text=[
            f"<b>Hipocentro real</b><br>Magnitud: Mw {ev_mw:.1f}<br>Profundidad: {ev_depth_km:.0f} km bajo tierra",
            f"<b>Epicentro en superficie (0 km)</b><br>Lat {ev_lat:.2f}°, Lon {ev_lon:.2f}°",
        ],
        hoverinfo="text",
    )

    frames: list[go.Frame] = []
    init_traces: list[Any] | None = None
    init_idx = 4  # t = 36 s so P-ring, S-ring, dome & lit surface districts are immediately visible

    p_col = "#38BDF8" if theme_mode == THEME_DARK else "#0284C7"
    s_col = "#F43F5E" if theme_mode == THEME_DARK else "#BE123C"

    for f_idx, t_sec in enumerate(WAVE_FRAME_TIMES_S):
        px_r, py_r, pz_r = _surface_ring(WAVE_VP_KM_S * float(t_sec), 0.35)
        sx_r, sy_r, sz_r = _surface_ring(WAVE_VS_KM_S * float(t_sec), 0.55)

        tr_dome = _subsoil_dome_mesh(WAVE_VS_KM_S * float(t_sec))
        tr_wave_p = go.Scatter3d(
            x=px_r,
            y=py_r,
            z=pz_r,
            mode="lines",
            name="Onda P (6.5 km/s — aviso)",
            line=dict(color=p_col, width=3.2, dash="dot"),
            hoverinfo="skip",
        )
        tr_wave_s = go.Scatter3d(
            x=sx_r,
            y=sy_r,
            z=sz_r,
            mode="lines",
            name="Onda S (3.8 km/s — sacudida)",
            line=dict(color=s_col, width=4.2),
            hoverinfo="skip",
        )
        tr_dists = _surface_districts_trace(float(t_sec))

        n_hit = int((vis["t_arrival_s_s"] <= t_sec).sum())
        frame = go.Frame(
            name=f"{t_sec}s",
            data=[tr_dome, tr_wave_p, tr_wave_s, tr_dists],
            traces=[2, 3, 4, 5],
            layout=go.Layout(
                title=dict(
                    text=f"Ondas P y S en Subsuelo y Superficie (Mw {ev_mw:.1f}, Prof. {ev_depth_km:.0f} km) — t = {t_sec} s · {n_hit} distritos alcanzados",
                    x=0.5,
                    xanchor="center",
                    y=0.97,
                    font=dict(size=14),
                )
            ),
        )
        frames.append(frame)
        if f_idx == init_idx:
            init_traces = [trace_peru, trace_hypo, tr_dome, tr_wave_p, tr_wave_s, tr_dists]

    assert init_traces is not None
    t_init = WAVE_FRAME_TIMES_S[init_idx]
    n_hit_init = int((vis["t_arrival_s_s"] <= t_init).sum())

    z_min = min(-max(ev_depth_km + 60.0, 120.0), -120.0)
    fig = go.Figure(data=init_traces, frames=frames)
    fig.update_layout(
        **plotly_theme_kwargs(theme_mode),
        height=610,
        title=dict(
            text=f"Ondas P y S en Subsuelo y Superficie (Mw {ev_mw:.1f}, Prof. {ev_depth_km:.0f} km) — t = {t_init} s · {n_hit_init} distritos alcanzados",
            x=0.5,
            xanchor="center",
            y=0.97,
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="Longitud (°W)",
            yaxis_title="Latitud (°S)",
            zaxis_title="Profundidad subterránea (km)",
            xaxis=dict(range=[-82.2, -68.2]),
            yaxis=dict(range=[-18.8, -0.2]),
            zaxis=dict(range=[z_min, 8.0]),
            aspectratio=dict(x=1.0, y=1.28, z=0.48),
            camera=dict(eye=dict(x=-1.52, y=-1.32, z=0.92)),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=70, b=10),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.01,
                y=0.06,
                xanchor="left",
                yanchor="bottom",
                bgcolor="#1E293B" if theme_mode == THEME_DARK else "#F3EDE4",
                font=dict(color="#F8FAFC" if theme_mode == THEME_DARK else "#1C1917", size=12),
                buttons=[
                    dict(
                        label="▶ Reproducir Onda",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 550, "redraw": True},
                                "transition": {"duration": 180},
                                "fromcurrent": False,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pausar",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=init_idx,
                x=0.24,
                y=0.05,
                len=0.73,
                currentvalue={"prefix": "Tiempo transcurrido: ", "visible": True, "font": {"size": 12}},
                steps=[
                    dict(
                        label=f"{t_s}s",
                        method="animate",
                        args=[[f"{t_s}s"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    )
                    for t_s in WAVE_FRAME_TIMES_S
                ],
            )
        ],
    )
    return fig


load_assets = load_simulation_assets


@st.cache_resource(show_spinner="Cargando motores geofísicos y de inteligencia artificial...")
def _cached_sim_assets(cache_version: str = ASSET_CACHE_VERSION) -> dict[str, Any]:
    _ = cache_version
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

    if show_inei_dept and inei["dept_geojson"] is not None:
        folium.GeoJson(
            inei["dept_geojson"],
            name="Límites Departamentales INEI",
            style_function=lambda _f: {
                "fillColor": "#FFFFFF",
                "color": "#44403C",
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
                color="#1C1917" if is_crit else "#57534E",
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

    m = folium.Map(location=[lat0, lon0], zoom_start=11, tiles=folium_tiles_for_theme(theme_mode), control_scale=True)
    _add_fullscreen(m)

    if dist_feature is not None:
        gj = folium.GeoJson(
            {"type": "FeatureCollection", "features": [dist_feature]},
            style_function=lambda _f: {
                "fillColor": poly_col,
                "color": "#1C1917",
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
            color="#1C1917",
            weight=2.0,
            fill=True,
            fill_color=poly_col,
            fill_opacity=0.28,
            tooltip=f"Centroide distrital {dist_row['distrito']} (UBIGEO {dist_row['ubigeo']})",
        ).add_to(m)

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


@st.cache_data(show_spinner="Generando mapa territorial de peligro sísmico...")
def _build_hazard_extremes_folium_html(mode: str, theme_mode: str) -> str:
    """Build Folium map with Fullscreen control highlighting either lowest (Muy Bajo/Bajo) or highest (Alto/Muy Alto) districts."""
    assets = _cached_sim_assets()
    inei = _cached_inei_boundaries()
    df_u = assets["df_u"]

    m = folium.Map(location=[-9.8, -74.8], zoom_start=6, tiles=folium_tiles_for_theme(theme_mode), control_scale=True)
    _add_fullscreen(m)

    if inei["dept_geojson"] is not None:
        folium.GeoJson(
            inei["dept_geojson"],
            style_function=lambda _f: {
                "fillColor": "#FFFFFF",
                "color": "#57534E",
                "weight": 1.1,
                "fillOpacity": 0.03,
            },
            tooltip=folium.GeoJsonTooltip(fields=["departamento"], aliases=["Departamento:"]),
        ).add_to(m)

    if mode == "low":
        sub = df_u[df_u["nivel_peligro_sismico_puro"].isin(["Muy Bajo", "Bajo"])].copy()
        for _, r in sub.iterrows():
            is_ext = r["nivel_peligro_sismico_puro"] == "Muy Bajo"
            col = "#0F766E" if is_ext else "#0284C7"
            folium.CircleMarker(
                location=[float(r["dist_lat"]), float(r["dist_lon"])],
                radius=5.0 if is_ext else 3.8,
                color=col,
                weight=0.8,
                fill=True,
                fill_color=col,
                fill_opacity=0.80,
                tooltip=f"{r['distrito']} ({r['departamento']}) · {r['nivel_peligro_sismico_puro']} ({r['seismic_hazard_index_0_100']:.1f}/100)",
            ).add_to(m)
    else:
        sub = df_u[df_u["nivel_peligro_sismico_puro"].isin(["Alto", "Muy Alto"])].copy()
        for _, r in sub.iterrows():
            is_ext = r["nivel_peligro_sismico_puro"] == "Muy Alto"
            col = "#991B1B" if is_ext else "#EA580C"
            folium.CircleMarker(
                location=[float(r["dist_lat"]), float(r["dist_lon"])],
                radius=5.4 if is_ext else 4.0,
                color=col,
                weight=0.8,
                fill=True,
                fill_color=col,
                fill_opacity=0.82,
                tooltip=f"{r['distrito']} ({r['departamento']}) · {r['nivel_peligro_sismico_puro']} ({r['seismic_hazard_index_0_100']:.1f}/100 · {r['pga_escenario_max_gal']:.0f} Gal)",
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

    st.markdown("---")
    st.markdown("#### 🧠 ¿Cómo nace un terremoto y por qué aún no se puede predecir la fecha exacta?")
    st.markdown(
        """
        <div class="step-grid">
          <div class="step-card" style="--step-accent:#B91C1C;">
            <span class="step-num">Paso 1 · Acumulación</span>
            <h4>La regla de madera que se dobla</h4>
            <p>
              Frente al Perú, la placa oceánica de Nazca empuja a Sudamérica a
              <span class="tech-pill" style="--pill-accent:#B91C1C;">6 – 8 cm/año</span>.
              Como una regla doblada entre las manos, la roca se deforma durante décadas o siglos hasta quebrarse a unos
              <span class="tech-pill" style="--pill-accent:#991B1B;">~30 km bajo el mar</span>.
            </p>
            <p>
              <strong>¿Por qué no sabemos el día exacto?</strong> Hoy vigilamos milímetro a milímetro cuánto se dobla la costa usando
              <strong>estaciones GPS/GNSS continuas</strong>, <strong>radares satelitales InSAR</strong> y <strong>sensores sísmicos profundos</strong>;
              eso nos permite medir <em>cuánta energía hay acumulada</em>, pero no ver el microsegundo exacto en que cederá la roca.
            </p>
          </div>
          <div class="step-card" style="--step-accent:#0284C7;">
            <span class="step-num">Paso 2 · Propagación</span>
            <h4>Onda P vs. Onda S: segundos que salvan vidas</h4>
            <p>
              Al romperse la falla salen dos ondas principales desde el foco subterráneo:
            </p>
            <p>
              <span class="tech-pill" style="--pill-accent:#0284C7;">Onda P · ~6.5 km/s</span>
              Es la <strong>onda primaria rápida</strong>: viaja primero como un golpe vertical leve. Los sistemas de alerta temprana la detectan antes de que llegue el movimiento fuerte.
            </p>
            <p>
              <span class="tech-pill" style="--pill-accent:#BE123C;">Onda S · ~3.8 km/s</span>
              Es la <strong>onda secundaria destructiva</strong>: llega segundos después moviendo el suelo de lado a lado con mucha mayor energía.
            </p>
          </div>
          <div class="step-card" style="--step-accent:#D97706;">
            <span class="step-num">Paso 3 · Amplificación Local</span>
            <h4>El «Efecto Gelatina» de los suelos blandos</h4>
            <p>
              Cuando la Onda S llega a tu barrio, la roca dura vibra poco, pero los sedimentos sueltos, arenas o rellenos
              amplifican el movimiento como gelatina sobre un plato.
            </p>
            <p>
              <strong>Mediciones reales del IGP en Lima (vs. roca firme):</strong><br>
              <span class="tech-pill" style="--pill-accent:#0F766E;">Morro Solar 1.00×</span>
              <span class="tech-pill" style="--pill-accent:#D97706;">La Molina 1.42×</span>
              <span class="tech-pill" style="--pill-accent:#D97706;">Los Olivos 1.45×</span>
              <span class="tech-pill" style="--pill-accent:#EA580C;">Callao 1.55×</span>
              <span class="tech-pill" style="--pill-accent:#B91C1C;">Puente Piedra 2.17×</span>
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    mk2.markdown(render_kpi_card("Tamaño de la Onda Sísmica", f"{amp_ratio:,.1f}× veces" if amp_ratio < 100 else f"{amp_ratio:,.0f}× veces", "Fórmula física: 10^(ΔM)", "#D97706"), unsafe_allow_html=True)
    mk3.markdown(render_kpi_card("Energía Total Liberada", f"{energy_ratio:,.1f}× veces" if energy_ratio < 100 else f"{energy_ratio:,.0f}× veces", "Fórmula física: 10^(1.5 · ΔM)", "#B91C1C"), unsafe_allow_html=True)

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
        fig_e.update_layout(**plotly_theme_kwargs(theme_mode), height=330, title=f"Comparación Mw {hi_m:.1f} frente a Mw {lo_m:.1f}", margin=dict(l=45, r=20, t=48, b=35))
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

    st.markdown("---")
    st.markdown("#### 3. Escala Dual: Intensidad Mercalli (Grados I al X) y su Equivalencia en Magnitud (Mw)")
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
    st.markdown(render_centered_table(pd.DataFrame(mercalli_rows), badge_cols={"Nivel"}, max_height_px=470), unsafe_allow_html=True)
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
            soil_rows.append({"Estación": label_es, "Código": code, "Factor": round(float(row_s["amp_factor_rel_msol"]), 2), "Sismos": int(row_s["n_records"])})
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

    st.markdown("---")
    st.markdown("#### 6. Preguntas Frecuentes Ciudadanas")
    with st.expander("❓ ¿Se puede predecir la fecha exacta de un terremoto?"):
        st.write(
            "**No.** Ninguna institución científica ni inteligencia artificial en el mundo puede predecir el día, la hora ni el minuto "
            "de un terremoto. Esta plataforma calcula **probabilidades estadísticas** y **simulaciones de impacto** para prevención."
        )
    with st.expander("❓ ¿Por qué en mi distrito se siente más fuerte que en el distrito vecino?"):
        st.write(
            "Por dos razones: (1) **el tipo de suelo** —los suelos arenosos, humedales o rellenos amplifican la vibración hasta más del doble "
            "comparados con la roca dura—, y (2) **el tipo de edificio** —viviendas sin columnas bien amarradas oscilan mucho más."
        )
    with st.expander("❓ ¿Es verdad que cada 100 años hay un terremoto fijo en Lima?"):
        st.write(
            "**No es un reloj fijo.** El tramo Lima–Callao–Chincha acumula tensión desde **1746 (280 años)**, por lo que la probabilidad a 30 años "
            "es alta (~36% a 63% según el modelo), pero la ruptura puede ocurrir mañana o dentro de décadas."
        )
    with st.expander("❓ ¿Qué significa que mi distrito tenga Índice Sísmico 96/100?"):
        st.write(
            "**NO significa 96% de probabilidad.** Es un puntaje comparativo de **0 (mínimo nacional)** a **100 (máximo nacional)** "
            "que combina sacudimiento con suelo local (**45%**), cercanía a laguna costera BPT a 30 años (**30%**) y tasa anual H3 (**25%**)."
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
        st.info("💡 Haz clic en cualquier círculo del mar o estación en tierra para ver su detalle, o usa el botón de **pantalla completa** arriba a la izquierda.")

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
                    marker=dict(size=9, color=vis["prob_pct"], colorscale="YlOrRd", colorbar=dict(title=dict(text="Probabilidad (%)", side="right"), thickness=14), opacity=0.82),
                    text=[
                        f"<b>{r['ubicacion_legible']}</b><br>Probabilidad estimada: <b>{r['prob_pct']:.1f}%</b><br>Distancia a la fosa: {r['dist_trench_km']:.0f} km<br>Peligros geológicos: {int(r['n_geocatmin_peligros'])}"
                        for _, r in vis.iterrows()
                    ],
                    hoverinfo="text",
                )
            )
            fig_f.update_layout(
                **plotly_theme_kwargs(theme_mode),
                map=dict(style="carto-darkmatter" if theme_mode == THEME_DARK else "open-street-map", center=dict(lat=float(vis["cell_lat"].mean()), lon=float(vis["cell_lon"].mean())), zoom=5.2 if sel_depa == "Todo el Perú" else 7.0),
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
        fig_sh = px.bar(shap_raw, x="mean_abs_shap", y="Factor", orientation="h", color_discrete_sequence=["#B91C1C"], labels={"mean_abs_shap": "Peso explicativo promedio", "Factor": ""})
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
                ("Probabilidad Estimada (%)", "Probabilidad porcentual de al menos un sismo del umbral elegido dentro de esa celda (~252 km²) en el periodo seleccionado."),
                ("Distancia a la Fosa", "Kilómetros en línea recta hasta la fosa marina donde chocan las placas de Nazca y Sudamérica."),
                ("Profundidad de Placa", "Profundidad estimada bajo tierra a la que pasa la placa oceánica de Nazca debajo de esa zona."),
                ("Peligros Geológicos", "Cantidad de puntos de huaicos, deslizamientos o caída de rocas inventariados por INGEMMET/GEOCATMIN."),
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_sec3_district_consultant(assets: dict[str, Any], data: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🏘️ Consultor de Mi Distrito: Probabilidad Sísmica y Riesgo Local")
    st.write(
        "Elige tu distrito y el **escenario sísmico que deseas evaluar** para ver su polígono oficial del INEI, "
        "su probabilidad estadística, la respuesta de su suelo y una guía clara de resistencia de viviendas."
    )

    df_u = assets["df_u"].copy()
    inei = _cached_inei_boundaries()
    df_u["etiqueta"] = df_u["distrito"] + " — " + df_u["provincia"] + ", " + df_u["departamento"] + " (" + df_u["ubigeo"] + ")"

    default_matches = df_u.index[df_u["ubigeo"] == "150125"].tolist()
    default_idx = int(default_matches[0]) if default_matches else 0

    sel_c1, sel_c2 = st.columns([1.15, 1.05])
    sel_lbl = sel_c1.selectbox("🔎 Selecciona tu distrito (1,893 distritos oficiales)", df_u["etiqueta"].tolist(), index=default_idx)
    sel_scen = sel_c2.selectbox("⚡ Escenario a evaluar en este distrito", DISTRICT_SCENARIO_OPTIONS, index=0)

    r = df_u[df_u["etiqueta"] == sel_lbl].iloc[0]
    ubigeo = str(r["ubigeo"])
    dist_feat = inei["dist_by_ubigeo"].get(ubigeo)

    amp_normal = float(r["site_amp_factor_vs_msol"])
    if sel_scen == DISTRICT_SCENARIO_OPTIONS[0]:
        scen_pairs = [
            ("Lima–Callao Mw 8.6", float(r["pga_escenario_lima_mw86_gal"]), float(r["pga_escenario_lima_mw86_rock_gal"]), float(r["mmi_escenario_lima_mw86"])),
            ("Sur Ilo–Tacna Mw 8.4", float(r["pga_escenario_sur_mw84_gal"]), float(r["pga_escenario_sur_mw84_rock_gal"]), float(r["mmi_escenario_sur_mw84"])),
            ("Sierra Central Mw 6.8", float(r["pga_escenario_intraplaca_mw68_gal"]), float(r["pga_escenario_intraplaca_mw68_rock_gal"]), float(r["mmi_escenario_intraplaca_mw68"])),
        ]
        active_scen_lbl, active_site_gal, active_rock_gal, active_mmi = max(scen_pairs, key=lambda t: t[1])
        active_scen_lbl = f"Peor escenario ({active_scen_lbl})"
    elif sel_scen == DISTRICT_SCENARIO_OPTIONS[1]:
        active_scen_lbl = "Lima–Callao Mw 8.6 (Brecha de 1746)"
        active_site_gal = float(r["pga_escenario_lima_mw86_gal"])
        active_rock_gal = float(r["pga_escenario_lima_mw86_rock_gal"])
        active_mmi = float(r["mmi_escenario_lima_mw86"])
    elif sel_scen == DISTRICT_SCENARIO_OPTIONS[2]:
        active_scen_lbl = "Sur Ilo–Tacna Mw 8.4 (Brecha de 1868)"
        active_site_gal = float(r["pga_escenario_sur_mw84_gal"])
        active_rock_gal = float(r["pga_escenario_sur_mw84_rock_gal"])
        active_mmi = float(r["mmi_escenario_sur_mw84"])
    elif sel_scen == DISTRICT_SCENARIO_OPTIONS[3]:
        active_scen_lbl = "Sierra Central Mw 6.8 (Estilo Junín)"
        active_site_gal = float(r["pga_escenario_intraplaca_mw68_gal"])
        active_rock_gal = float(r["pga_escenario_intraplaca_mw68_rock_gal"])
        active_mmi = float(r["mmi_escenario_intraplaca_mw68"])
    else:
        custom_mw = st.slider(
            "🎛️ Elige la magnitud personalizada a simular cerca de este distrito (Mw)",
            5.0,
            9.0,
            7.5,
            0.1,
            help="Simula un sismo de subducción (25 km de profundidad) ubicado en el punto más cercano de la fosa marina PB2002.",
        )
        ev_lon_c, ev_lat_c, epi_trench_km = _nearest_trench_lonlat(assets, float(r["dist_lat"]), float(r["dist_lon"]))
        hypo_trench_km = float(np.sqrt(epi_trench_km ** 2 + 25.0 ** 2))
        sim_one = simulate_hybrid_nonlinear_pga(
            target_lats=np.array([float(r["dist_lat"])]),
            target_lons=np.array([float(r["dist_lon"])]),
            target_dist_trench_km=np.array([float(r["dist_trench_km"])]),
            target_dist_lat_15s_km=np.array([float(r["dist_lat_15s_km"])]),
            target_geocatmin_matrix={k: np.array([float(r[k])]) for k in ("n_peligros_total", "n_peligros_flujo", "n_peligros_caida", "n_peligros_deslizamiento", "n_zonas_criticas_total")},
            target_delta_s2s=np.array([float(r["site_delta_s2s"])]),
            ev_lat=float(ev_lat_c),
            ev_lon=float(ev_lon_c),
            ev_depth_km=25.0,
            ev_mw=float(custom_mw),
            ev_regime=None,
            gmpe_bundle=assets["gmpe_bundle"],
            c1_huber=assets["c1_huber"],
            trench_tree=assets["trench_tree"],
            trench_vels=assets["trench_vels"],
            trench_ages=assets["trench_ages"],
            slab_model=assets["slab_model"],
            use_nonlinear_damping=True,
        )
        reg_custom = REGIME_SHORT_ES.get(str(sim_one["resolved_regime"]), str(sim_one["resolved_regime"]))
        active_scen_lbl = f"Simulación personalizada Mw {custom_mw:.1f} ({reg_custom} · {hypo_trench_km:.0f} km hipoc.)"
        active_site_gal = float(sim_one["pga_site_gal"][0])
        active_rock_gal = float(sim_one["pga_rock_gal"][0])
        active_mmi = float(sim_one["mmi_site"][0])

    amp_eff_active = active_site_gal / active_rock_gal if active_rock_gal > 0 else 1.0

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

    st.markdown("#### ¿Qué tan probable es un sismo en mi distrito?")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.markdown(render_kpi_card("Próximos 3 Meses", f"{p3m_m45 * 100:.1f}%", "Prob. sismo sensible M ≥ 4.5", "#0F766E"), unsafe_allow_html=True)
    h2.markdown(render_kpi_card("Próximos 6 Meses", f"{p6m_m45 * 100:.1f}%", "Proyección semestral (tasa anual)", "#0284C7"), unsafe_allow_html=True)
    h3.markdown(render_kpi_card("Próximos 12 Meses", f"{p12m_m45 * 100:.1f}%", f"M≥4.5 · Fuerte M≥5.0: {p12m_m50 * 100:.1f}%", "#D97706"), unsafe_allow_html=True)
    h4.markdown(render_kpi_card("Próximos 5 Años", f"{p5y_m45 * 100:.1f}%", "M ≥ 4.5 (tasa moderna)", "#C2410C"), unsafe_allow_html=True)
    h5.markdown(render_kpi_card("Megaterremoto Mw ≥ 8.0", f"{bpt_30_raw:.0f}% (30a) · {bpt_50_raw:.0f}% (50a)", f"Tramo {gap_zone_lbl} (efecto {bpt_30_eff:.0f}%)", "#991B1B"), unsafe_allow_html=True)

    if is_under:
        st.warning(
            f"⚠️ **Brecha de priorización detectada:** **{r['distrito']}** tiene Peligro Sísmico Puro **{nivel_seis}** "
            f"(Índice `{r['seismic_hazard_index_0_100']:.1f}/100`), pero aparece como **{nivel_fondes}** en el padrón oficial FONDES/SIGRID."
        )

    st.markdown("")
    col_map, col_info = st.columns([1.10, 1.02])
    with col_map:
        poly_status = "Polígono oficial INEI WFS" if dist_feat is not None else "Área referencial (centroide oficial)"
        st.markdown(f"**Mapa Distrital ({poly_status})**")
        dist_map_html = _build_district_consultant_map(r, dist_feat, data["df_site"], data["df_zc"], theme_mode)
        st.iframe(dist_map_html, height=430, width="stretch")
        st.markdown(
            render_map_legend(
                [
                    (RISK_LEVEL_COLORS.get(nivel_seis, "#B91C1C"), "Polígono sombreado INEI", f"Límite oficial de {r['distrito']} (color según nivel {nivel_seis})"),
                    ("#D97706", "Marcador con ícono de onda", "Estaciones Acelerográficas del IGP cercanas"),
                    ("#7C3AED", "Puntos morados", "Puntos Críticos de huaicos/derrumbes (INGEMMET)"),
                ]
            ),
            unsafe_allow_html=True,
        )

    with col_info:
        m_rom, m_short, m_desc, m_col, m_mag_rng = describe_mercalli_dual(active_mmi)
        st_code = str(r["primary_station_code"]) if pd.notna(r["primary_station_code"]) else "Sin estación directa (neutro)"
        active_pct_g = active_site_gal / 9.81

        st.markdown(
            f"""
            <div class="district-spotlight" style="--accent:{m_col};margin-top:0;">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
                <strong style="font-size:1.02rem;">Escenario Evaluado: {active_scen_lbl}</strong>
                {render_badge(f"Mercalli {m_rom} ({m_mag_rng}) — {m_short}", m_col)}
              </div>
              <p style="margin:0.40rem 0 0.60rem 0;font-size:0.87rem;color:var(--text-secondary);">
                {m_desc}
              </p>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.83rem;">
                <div><strong>Sacudimiento esperado:</strong><br>{active_pct_g:.1f}% g ({active_site_gal:.0f} Gal)<br><span style="color:var(--text-muted);font-size:0.75rem;">100% g = 981 Gal (gravedad)</span></div>
                <div><strong>Efecto del suelo ({st_code}):</strong><br>Sismos normales: <strong>{amp_normal:.2f}×</strong><br>En este escenario: <strong>{amp_eff_active:.2f}×</strong></div>
                <div><strong>Peligro Sísmico Puro:</strong><br>{render_badge(nivel_seis, RISK_LEVEL_COLORS.get(nivel_seis, '#B91C1C'))} ({r['seismic_hazard_index_0_100']:.1f}/100)</div>
                <div><strong>Oficial FONDES / SIGRID:</strong><br>{render_badge(nivel_fondes, RISK_LEVEL_COLORS.get(nivel_fondes, '#475569'))}</div>
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
                {"Escenario / Amenaza": "Peligros Geológicos (INGEMMET)", "Valor del Distrito": f"{int(r['n_peligros_total'])} puntos", "Equivalencia": f"Flujos: {int(r['n_peligros_flujo'])} · Caídas: {int(r['n_peligros_caida'])} · Desliz.: {int(r['n_peligros_deslizamiento'])}"},
                {"Escenario / Amenaza": "Zonas Críticas Activas", "Valor del Distrito": f"{int(r['n_zonas_criticas_activas'])} sectores", "Equivalencia": f"Distancia a la fosa: {r['dist_trench_km']:.0f} km"},
            ]
        )
        st.markdown(render_centered_table(detail_df, max_height_px=255), unsafe_allow_html=True)
        st.markdown(
            render_table_legend(
                [
                    ("Valor del Distrito (%g y Gal)", "Aceleración esperada del suelo. 10% g (~98 Gal) dificulta mantenerse en pie; 50% g (~490 Gal) agrieta paredes."),
                    ("Equivalencia Mercalli + Mw", "Grado de sacudida local (Mercalli) y rango aproximado de Magnitud Mw equivalente en epicentro."),
                    ("Flujos / Caídas / Zonas Críticas", "Huaicos/lodo (Flujos), derrumbes de rocas (Caídas) y áreas habitadas vulnerables (INGEMMET)."),
                ]
            ),
            unsafe_allow_html=True,
        )

    if active_mmi >= 8.5:
        dist_verdict = (
            f"En {r['distrito']} bajo {active_scen_lbl} ({active_pct_g:.1f}% g, Mercalli {m_rom}), "
            f"las viviendas de adobe o autoconstrucción enfrentan riesgo severo de colapso; las casas formales de albañilería confinada "
            f"requerirán inspección estructural inmediata y solo las edificaciones modernas Norma E.030 protegen contra el colapso."
        )
        verdict_col = "#991B1B"
    elif active_mmi >= 6.5:
        dist_verdict = (
            f"En {r['distrito']} bajo {active_scen_lbl} ({active_pct_g:.1f}% g, Mercalli {m_rom}), "
            f"las viviendas de adobe o sin columnas presentan riesgo alto de grietas y colapso parcial, mientras que las casas de ladrillo formal "
            f"con columnas y vigas suelen resistir con daños menores o reparables."
        )
        verdict_col = "#EA580C"
    elif active_mmi >= 5.0:
        dist_verdict = (
            f"En {r['distrito']} bajo {active_scen_lbl} ({active_pct_g:.1f}% g, Mercalli {m_rom}), "
            f"las viviendas formales no sufren daño estructural, pero muros antiguos debilitados o de adobe pueden mostrar fisuras finas y caída de objetos."
        )
        verdict_col = "#D97706"
    else:
        dist_verdict = (
            f"En {r['distrito']} bajo {active_scen_lbl} ({active_pct_g:.1f}% g, Mercalli {m_rom}), "
            f"la sacudida esperada es leve o imperceptible y no compromete la estabilidad estructural de las viviendas."
        )
        verdict_col = "#0F766E"

    st.markdown(
        f"""
        <div class="district-spotlight" style="--accent:{verdict_col};">
          <strong style="font-size:1.02rem;">🏠 ¿Hasta qué intensidad resisten las viviendas de este distrito? (Guía de Vulnerabilidad · Ref. Norma E.030 / CISMID)</strong>
          <div class="vuln-grid">
            <div class="vuln-item" style="--v-accent:#DC2626;">
              <strong>1. Adobe, quincha o autoconstrucción sin columnas</strong><br>
              Empiezan a sufrir grietas y colapso parcial desde <strong>Mercalli VI–VII</strong> (<code>&gt; 10% g</code> / <code>~100 Gal</code>).
            </div>
            <div class="vuln-item" style="--v-accent:#D97706;">
              <strong>2. Ladrillo formal con columnas y vigas (albañilería confinada)</strong><br>
              Resisten bien hasta <strong>Mercalli VII–VIII</strong> (<code>20%–45% g</code>); requieren revisión estructural tras <strong>Mercalli IX</strong>.
            </div>
            <div class="vuln-item" style="--v-accent:#0F766E;">
              <strong>3. Edificaciones modernas sismorresistentes (Norma E.030)</strong><br>
              Diseñadas para evitar el colapso incluso en <strong>Mercalli IX</strong> (<code>&gt; 50% g</code>), aunque pueden dañar muros divisorios.
            </div>
          </div>
          <p style="margin:0.65rem 0 0 0;font-size:0.86rem;color:var(--text-primary);">
            <strong>Diagnóstico para {r['distrito']}:</strong> {dist_verdict}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prov_peers = df_u[(df_u["departamento"] == r["departamento"]) & (df_u["provincia"] == r["provincia"])].sort_values("seismic_hazard_index_0_100", ascending=False).copy()
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
        render_centered_table(peer_tbl, badge_cols={"Peligro Sísmico", "Oficial FONDES"}, highlight_col="UBIGEO", highlight_value=ubigeo, max_height_px=415, caption=f"Fila resaltada: {r['distrito']} (UBIGEO {ubigeo})."),
        unsafe_allow_html=True,
    )
    st.markdown(
        render_table_legend(
            [
                ("Índice Sísmico (0–100)", "Puntaje relativo de peligro físico de 0 (mínimo nacional) a 100 (máximo); NO es porcentaje de probabilidad."),
                ("Prob. 12m (M≥4.5)", "Probabilidad porcentual de al menos un sismo sensible (M≥4.5) en los próximos 12 meses."),
                ("Suelo Normal", "Amplificación medida frente a roca firme en sismos cotidianos (en megaterremotos el amortiguamiento no lineal la reduce)."),
                ("Zonas Críticas", "Sectores críticos habitados registrados por INGEMMET susceptibles a huaicos o derrumbes."),
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_sec4_simulator(assets: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### ⚡ Simulador de Terremotos en Vivo: Propagación 3D de Ondas y ShakeMap")
    st.write(
        "Elige un escenario real o mueve el epicentro y la magnitud para simular en milisegundos cómo nacen las ondas bajo "
        "tierra y sacuden los **1,893 distritos del Perú** según su distancia y tipo de suelo."
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
    use_nl = c5.toggle("Amortiguamiento no lineal de suelos blandos", value=True, help="En megaterremotos (>150 Gal en roca), los suelos blandos absorben parte de la sacudida (Seyhan & Stewart 2014).")

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
              <strong style="font-size:1.08rem;">{drow['distrito']} ({drow['provincia']}, {drow['departamento']})</strong>
              <span style="color:var(--text-muted);font-size:0.86rem;"> · Distancia: <strong>{drow['dist_epicentral_km']:.0f} km</strong> · Onda P: <strong>{drow['dist_hypocentral_km']/WAVE_VP_KM_S:.1f} s</strong> · Onda S: <strong>{drow['dist_hypocentral_km']/WAVE_VS_KM_S:.1f} s</strong></span>
            </div>
            <div>{render_badge(f"Mercalli {m_rom} ({m_mag_rng}) — {m_short}", m_col)}</div>
          </div>
          <p style="margin:0.40rem 0 0.58rem 0;color:var(--text-secondary);font-size:0.90rem;">
            <strong>Impacto cotidiano esperado:</strong> {m_desc}
          </p>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:10px;font-size:0.85rem;">
            <div><strong>Sacudimiento real:</strong> {drow['pga_site_pct_g']:.1f}% g ({drow['pga_site_gal']:.0f} Gal)</div>
            <div><strong>Suelo normal vs. este sismo:</strong> {drow['site_amp_normal']:.2f}× → <strong>{drow['site_amp_eff']:.2f}× efectivo</strong></div>
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
    view_mode = st.radio(
        "Modo de visualización del simulador",
        [
            "🌐 Vista 3D Subsuelo + Ondas",
            "🗺️ Vista 2D Cartográfica (Mapa Detallado)",
        ],
        index=0,
        horizontal=True,
    )

    if "Vista 3D" in view_mode:
        inei = _cached_inei_boundaries()
        fig_wave = _build_3d_animated_wave_figure(res_df, ev_lat, ev_lon, ev_depth, ev_mw, inei, theme_mode)
        st.plotly_chart(fig_wave, width="stretch", config=PLOTLY_CFG)
        st.markdown(
            render_map_legend(
                [
                    ("#DC2626", "Esfera roja bajo tierra + domo translúcido", f"Hipocentro real (-{ev_depth:.0f} km) y frente subterráneo subiendo hacia la superficie"),
                    ("#0284C7", "Anillo cian punteado — Onda P (6.5 km/s)", "Onda primaria rápida que avisa primero (base de los sistemas de alerta temprana)"),
                    ("#BE123C", "Anillo carmesí sólido — Onda S (3.8 km/s)", "Onda secundaria destructiva que va encendiendo los distritos sobre la superficie (z = 0)"),
                    ("#EA580C", "Puntos en superficie (z = 0 km)", "Color según Intensidad Mercalli y tamaño proporcional al sacudimiento real (%g)"),
                ],
                note="Observa el nacimiento del terremoto bajo tierra y los dos anillos concéntricos (Onda P rápida y Onda S destructiva) expandiéndose sobre el Perú. Usa ▶ Reproducir Onda o el ícono de expansión arriba a la derecha para pantalla completa.",
            ),
            unsafe_allow_html=True,
        )
    else:
        vis = res_df[res_df["pga_site_gal"] >= 1.5].copy()
        fig_map = go.Figure()
        fig_map.add_trace(
            go.Scattermap(
                lat=vis["dist_lat"],
                lon=vis["dist_lon"],
                mode="markers",
                marker=dict(size=np.clip(vis["mmi_est"] * 1.9, 5, 18), color=vis["mmi_est"], colorscale="YlOrRd", cmin=3.0, cmax=9.5, colorbar=dict(title=dict(text="Mercalli (MMI)", side="right"), thickness=14), opacity=0.82),
                text=[
                    f"<b>{r['distrito']} ({r['departamento']})</b><br>"
                    f"Intensidad: <b>Mercalli {r['mercalli_romano']} (eq. {r['mercalli_mag_eq']}) — {r['nivel_impacto']}</b><br>"
                    f"Sacudimiento: {r['pga_site_pct_g']:.1f}% g ({r['pga_site_gal']:.0f} Gal)<br>"
                    f"Suelo: {r['site_amp_normal']:.2f}× normal → {r['site_amp_eff']:.2f}× efectivo | Dist.: {r['dist_epicentral_km']:.0f} km"
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
            map=dict(style="carto-darkmatter" if theme_mode == THEME_DARK else "open-street-map", center=dict(lat=ev_lat, lon=ev_lon), zoom=5.5),
            height=490,
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

    st.markdown("#### Top 20 Distritos con Mayor Sacudimiento Esperado")
    top20 = res_df.sort_values("pga_site_gal", ascending=False).head(20).copy()
    tbl_top20 = pd.DataFrame(
        {
            "Distrito": top20["distrito"],
            "Provincia / Dpto.": top20["provincia"] + " / " + top20["departamento"],
            "Llegada Onda P / S": [f"{d / WAVE_VP_KM_S:.0f} s / {d / WAVE_VS_KM_S:.0f} s" for d in top20["dist_hypocentral_km"]],
            "En Roca": [f"{v/9.81:.0f}% g ({v:.0f} Gal)" for v in top20["pga_rock_gal"]],
            "Real c/ Suelo": [f"{v/9.81:.0f}% g ({v:.0f} Gal)" for v in top20["pga_site_gal"]],
            "Suelo (Norm. → Efect.)": [f"{n:.2f}× → {e:.2f}×" for n, e in zip(top20["site_amp_normal"], top20["site_amp_eff"])],
            "Mercalli + Rango Mw": [f"{rom} ({eq})" for rom, eq in zip(top20["mercalli_romano"], top20["mercalli_mag_eq"])],
            "Impacto": top20["nivel_impacto"],
        }
    )
    st.markdown(render_centered_table(tbl_top20, badge_cols={"Impacto"}, max_height_px=420), unsafe_allow_html=True)
    st.markdown(
        render_table_legend(
            [
                ("Llegada Onda P / S", "Segundos aproximados desde la ruptura hasta la Onda P de aviso (~6.5 km/s) y la Onda S destructiva (~3.8 km/s)."),
                ("En Roca vs. Real c/ Suelo", "Sacudimiento sobre roca firme vs. sacudimiento real incorporando tu tipo de suelo (100% g = 981 Gal)."),
                ("Suelo (Norm. → Efect.)", "Amplificación en sismos cotidianos vs. factor efectivo tras amortiguamiento no lineal en este escenario."),
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
        "Explora 474 años de sismos del IGP y gira el modelo 3D —con el mapa del Perú en superficie (`0 km`) y "
        "**esferas proporcionales a la magnitud con aura luminosa en terremotos M ≥ 7.0**."
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
                marker=dict(size=np.clip((maj["mag"] - 5.8) * 6.0, 6, 22), color=maj["mag"], colorscale="YlOrRd", cmin=6.5, cmax=9.0, line=dict(color="#1C1917", width=0.7)),
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
                fig_tl.add_annotation(x=yr_m, y=mag_m, text=txt_m, showarrow=True, arrowhead=2, ax=0, ay=-30 if yr_m not in (1940, 1974, 2007) else 28, font=dict(size=10, color=ann_fg), bgcolor=ann_bg, bordercolor="#64748B")
    fig_tl.update_layout(**plotly_theme_kwargs(theme_mode), height=385, title="Línea de Tiempo de Grandes Terremotos del Perú (Magnitud ≥ 6.5)", xaxis_title="Año", yaxis_title="Magnitud (Mw)", margin=dict(l=40, r=20, t=50, b=40))
    st.plotly_chart(fig_tl, width="stretch", config=PLOTLY_CFG)

    st.markdown("#### Corte 3D de la Placa de Nazca bajo el Perú (Esferas Proporcionales a Magnitud + Aura en M ≥ 7.0)")
    st.info(
        "🧭 **¿Por qué hay volcanes en Arequipa y Tacna, pero no en Lima o Piura?** "
        "Al **Norte y Centro** (arriba de -15°S), la placa de Nazca entra casi horizontal (**11°–12°**), bloqueando el vulcanismo. "
        "Al **Sur de Ica/Nasca**, se hunde bruscamente (**30.1°**) hasta 269 km, alimentando los volcanes del sur. "
        "Las esferas más grandes con halo luminoso destacan los terremotos mayores (**M ≥ 7.0**)."
    )
    sub_3d_all = sub[(sub["depth_km"].notna()) & (sub["depth_km"] <= 350.0)].copy()
    major_3d = sub_3d_all[sub_3d_all["mag"] >= 6.5]
    small_3d = sub_3d_all[sub_3d_all["mag"] < 6.5]
    if len(small_3d) > 3200:
        small_3d = small_3d.sample(3200, random_state=42)
    sub_3d = pd.concat([small_3d, major_3d], ignore_index=True)

    color_map_reg = {
        "Subducción Costera": "#DC2626",
        "Falla Cortical Andina": "#F59E0B",
        "Losa Intermedia": "#14B8A6",
        "Foco Profundo": "#3B82F6",
    }
    fig_3d = go.Figure()

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
        sizes = np.clip((r_df["mag"].to_numpy(dtype="float64") - 3.9) ** 1.85 * 1.45, 2.2, 16.5)
        fig_3d.add_trace(
            go.Scatter3d(
                x=r_df["lon"],
                y=r_df["lat"],
                z=-r_df["depth_km"],
                mode="markers",
                name=reg_lbl,
                marker=dict(size=sizes, color=col_hex, opacity=0.74, line=dict(width=0)),
                text=[f"<b>Año {int(a)}</b> · Mw {m:.1f}<br>Prof.: {d:.0f} km · {reg_lbl}" for a, m, d in zip(r_df["anio"], r_df["mag"], r_df["depth_km"])],
                hoverinfo="text",
            )
        )

    aura_df = sub_3d[sub_3d["mag"] >= 7.0]
    if not aura_df.empty:
        aura_sizes = np.clip((aura_df["mag"].to_numpy(dtype="float64") - 3.9) ** 1.85 * 1.45, 8.0, 16.5) * 2.15
        fig_3d.add_trace(
            go.Scatter3d(
                x=aura_df["lon"],
                y=aura_df["lat"],
                z=-aura_df["depth_km"],
                mode="markers",
                name="Aura Terremotos Mayores (M ≥ 7.0)",
                marker=dict(size=aura_sizes, color="#F97316", opacity=0.24, line=dict(color="#DC2626", width=1.2)),
                text=[f"<b>GRAN TERREMOTO · Año {int(a)}</b><br>Magnitud: <b>Mw {m:.1f}</b><br>Profundidad: {d:.0f} km ({reg})" for a, m, d, reg in zip(aura_df["anio"], aura_df["mag"], aura_df["depth_km"], aura_df["regimen_es"])],
                hoverinfo="text",
            )
        )

    fig_3d.update_layout(
        **plotly_theme_kwargs(theme_mode),
        height=575,
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
                ("#DC2626", "Rojo — Subducción Costera", "Choque superficial (<60 km) frente al mar; tamaño proporcional a Mw"),
                ("#F59E0B", "Ámbar — Falla Cortical Andina", "Sismos superficiales (<60 km) dentro de la cordillera"),
                ("#14B8A6", "Verde azulado — Losa Intermedia", "Interior de la placa de Nazca (60–350 km bajo tierra)"),
                ("#F97316", "Halo/Aura translúcida", "Destaca todos los terremotos mayores (M ≥ 7.0) por energía y profundidad"),
                ("#1C1917", "Contorno en 0 km", "Mapa oficial del Perú (INEI) proyectado sobre la superficie"),
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
                    ("Profundidad", "Kilómetros bajo la superficie donde inició la ruptura; los menores a 60 km se sienten más fuerte."),
                    ("Catálogo", "Histórico (1552–1959, crónicas e isosistas) vs. Instrumental (1960–2026, red de sismógrafos)."),
                ]
            ),
            unsafe_allow_html=True,
        )


def _render_sec6_gaps(data: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🌊 Radar de Silencios Sísmicos en la Costa del Perú")
    st.write(
        "Mediante **10,000 simulaciones Monte Carlo**, estimamos cuánta energía elástica se ha acumulado en cada tramo del litoral "
        "desde su último megaterremoto y la probabilidad condicional de una gran ruptura (Mw ≥ 8.0) en 30 y 50 años."
    )

    df_gaps = data["df_gaps"].copy()
    c1, c2 = st.columns([1.2, 1.2])
    vista = c1.radio("Nivel de detalle", ["5 Grandes Bloques Costeros", "30 Tramos Detallados del Litoral (cada ~60 km)"], horizontal=True)
    modelo_bpt = c2.radio("Supuesto de recurrencia (BPT)", ["Conservador — irregularidad normal (α = 0.5)", "Cuasi-periódico — ciclos más regulares (α = 0.3)"], horizontal=True)
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
                    error_y=dict(type="data", symmetric=False, array=sub["mw_pot_p90"] - sub["mw_pot_p50"], arrayminus=sub["mw_pot_p50"] - np.maximum(sub["mw_pot_p10"], 6.5), color="#F8FAFC" if theme_mode == THEME_DARK else "#1C1917"),
                    marker_color="#B91C1C",
                    text=[f"Mw {v:.2f}" for v in sub["mw_pot_p50"]],
                    textposition="outside",
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
        fig_seg.add_trace(go.Scatter(x=p10_floor, y=sub["lat_center"], mode="lines", fill="tonextx", fillcolor="rgba(185,28,28,0.22)", line=dict(width=0), name="Rango P10–P90"))
        seg_hover = [
            f"<b>{r['coastal_zone']}</b><br>Mediana: Mw {r['mw_pot_p50']:.2f}<br>"
            + ("Rango: Sin déficit mayor (ruptura reciente)" if float(r["mw_pot_p10"]) < 6.5 else f"Rango: Mw {float(r['mw_pot_p10']):.2f}")
            + f" a Mw {r['mw_pot_p90']:.2f}<br>Prob. 30 años: {r[p30_col]:.1f}% | 50 años: {r[p50_col]:.1f}%"
            for _, r in sub.iterrows()
        ]
        fig_seg.add_trace(go.Scatter(x=sub["mw_pot_p50"], y=sub["lat_center"], mode="lines+markers", name="Magnitud potencial mediana", line=dict(color="#B91C1C", width=2.5), text=seg_hover, hoverinfo="text"))
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
            "Rango P10 – P90": [f"Sin déficit mayor – Mw {hi:.2f}" if lo < 6.5 else f"Mw {lo:.2f} – {hi:.2f}" for lo, hi in zip(sub["mw_pot_p10"], sub["mw_pot_p90"])],
            "Prob. 30 Años": [f"{v:.1f}%" for v in sub[p30_col]],
            "Prob. 50 Años": [f"{v:.1f}%" for v in sub[p50_col]],
            "Actividad Reciente (/100 km)": [f"{v:.1f}" for v in sub["modern_density_per_100km"]],
        }
    )
    st.markdown(render_centered_table(tbl, max_height_px=450), unsafe_allow_html=True)
    st.markdown(
        render_table_legend(
            [
                ("Magnitud Potencial (Mw)", "Tamaño aproximado del terremoto si toda la energía acumulada en ese tramo saliera en una sola ruptura."),
                ("Rango P10 – P90", "Intervalo de confianza del 80% obtenido tras 10,000 simulaciones Monte Carlo."),
                ("Prob. 30 y 50 Años (BPT)", "Probabilidad condicional de gran ruptura (Mw ≥ 8.0) dado el tiempo ya transcurrido sin megaterremoto."),
                ("Actividad Reciente (/100 km)", "Sismos moderados M≥4.7 entre 2000 y 2026 por cada 100 km de costa; valores bajos en alto déficit confirman silencio sísmico."),
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_sec7_hazard_extremes(assets: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🛡️ Zonas de Menor y Mayor Peligro Sísmico del Perú (Análisis Territorial)")
    st.markdown(
        """
        <div class="school-callout">
          <strong>¿Cómo se calcula el Índice Sísmico (0–100)?</strong> Es un puntaje comparativo entre todos los distritos del Perú
          (<strong>0 = mínimo nacional</strong>, <strong>100 = máximo nacional</strong>; <em>no es un porcentaje de probabilidad</em>) que pondera:
          <strong>45%</strong> sacudimiento máximo con efecto de suelo local + <strong>30%</strong> cercanía a una laguna costera (probabilidad BPT a 30 años)
          + <strong>25%</strong> tasa anual calibrada de sismos sensibles H3.
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode_lbl = st.radio(
        "Elige qué extremo del país deseas analizar",
        [
            "🟢 Ver Zonas de Menor Peligro Sísmico (Más Tranquilas)",
            "🔴 Ver Zonas de Mayor Peligro Sísmico (Más Expuestas)",
        ],
        index=0,
        horizontal=True,
    )
    is_low = "Menor" in mode_lbl

    df_u = assets["df_u"].copy()
    dep_agg = (
        df_u.groupby("departamento", as_index=False)
        .agg(
            n_dist=("ubigeo", "count"),
            mean_idx=("seismic_hazard_index_0_100", "mean"),
            mean_pga=("pga_escenario_max_gal", "mean"),
            peligros_geo=("n_peligros_total", "sum"),
            zonas_crit=("n_zonas_criticas_activas", "sum"),
        )
        .sort_values("mean_idx", ascending=is_low)
    )

    if is_low:
        subset = df_u[df_u["nivel_peligro_sismico_puro"].isin(["Muy Bajo", "Bajo"])].copy()
        ranked_all = df_u.sort_values("seismic_hazard_index_0_100", ascending=True).copy()
        ext_dist = ranked_all.iloc[0]
        ext_dep = dep_agg.iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(render_kpi_card("Distrito Más Tranquilo", f"{ext_dist['distrito']}", f"{ext_dist['departamento']} · Índice {ext_dist['seismic_hazard_index_0_100']:.1f}/100 ({ext_dist['pga_escenario_max_gal']:.1f} Gal)", "#0F766E"), unsafe_allow_html=True)
        k2.markdown(render_kpi_card("Dpto. Menor Peligro Sísmico", f"{ext_dep['departamento']}", f"Promedio {ext_dep['mean_idx']:.1f}/100 ({int(ext_dep['n_dist'])} distritos)", "#0284C7"), unsafe_allow_html=True)
        k3.markdown(render_kpi_card("Distritos Muy Bajo / Bajo", f"{len(subset):,} distritos", f"{len(subset)/len(df_u)*100:.1f}% del país (142 Muy Bajo + 565 Bajo)", "#0D9488"), unsafe_allow_html=True)
        k4.markdown(render_kpi_card("Otros Peligros en Estas Zonas", f"{int(subset['n_peligros_total'].sum()):,} puntos", f"{int(subset['n_zonas_criticas_activas'].sum()):,} zonas críticas (huaicos/inundaciones)", "#7C3AED"), unsafe_allow_html=True)

        st.info(
            "⚖️ **Mirada multi-amenaza:** Aunque **Madre de Dios** (`8.1/100`) y **Loreto** (`14.1/100`) tienen los índices sísmicos "
            "más bajos del Perú, varios de sus distritos figuran en **FONDES Alto o Muy Alto** por inundaciones amazónicas, erosión fluvial y deslizamientos."
        )
        bar_col = "#0F766E"
        chart_title = "Top 15 Departamentos con Menor Peligro Sísmico Promedio (vs. Peligros Geológicos)"
        map_title = "**Mapa de los 707 Distritos en Peligro Sísmico Muy Bajo y Bajo**"
        map_legend_items = [
            ("#0F766E", "Verde petróleo — Muy Bajo (142 distritos)", "Principalmente Amazonía oriental y fronteras alejadas de la costa"),
            ("#0284C7", "Azul — Bajo (565 distritos)", "Sierra oriental, altiplano y selva alta con menor sacudimiento en megaterremotos"),
        ]
        tbl_title = "#### Top 25 Distritos con Menor Peligro Sísmico del Perú"
        dl_label = "📥 Descargar ranking completo de los 1,893 distritos ordenados de menor a mayor peligro (CSV)"
        dl_name = "ranking_menor_a_mayor_peligro_sismico_peru_1893_distritos.csv"
    else:
        subset = df_u[df_u["nivel_peligro_sismico_puro"].isin(["Alto", "Muy Alto"])].copy()
        ranked_all = df_u.sort_values("seismic_hazard_index_0_100", ascending=False).copy()
        ext_dist = ranked_all.iloc[0]
        ext_dep = dep_agg.iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(render_kpi_card("Distrito Más Expuesto", f"{ext_dist['distrito']}", f"{ext_dist['departamento']} · Índice {ext_dist['seismic_hazard_index_0_100']:.1f}/100 ({ext_dist['pga_escenario_max_gal']:.0f} Gal)", "#991B1B"), unsafe_allow_html=True)
        k2.markdown(render_kpi_card("Dpto. Mayor Peligro Sísmico", f"{ext_dep['departamento']}", f"Promedio {ext_dep['mean_idx']:.1f}/100 ({int(ext_dep['n_dist'])} distritos)", "#B91C1C"), unsafe_allow_html=True)
        k3.markdown(render_kpi_card("Distritos Alto / Muy Alto", f"{len(subset):,} distritos", f"{len(subset)/len(df_u)*100:.1f}% del país (104 Muy Alto + 322 Alto)", "#C2410C"), unsafe_allow_html=True)
        k4.markdown(render_kpi_card("Zonas Críticas Expuestas", f"{int(subset['n_zonas_criticas_activas'].sum()):,} sectores", f"{int(subset['n_peligros_total'].sum()):,} peligros geológicos en zonas de alto peligro", "#7C3AED"), unsafe_allow_html=True)

        st.warning(
            "🚨 **Concentración costera:** **Callao (`93.9/100`)**, **Lima (`77.9/100`)**, **Tacna (`72.6/100`)** y **Moquegua (`68.7/100`)** "
            "encabezan el peligro sísmico nacional por su cercanía a las brechas de 1746 y 1868 sumada a la amplificación de suelos blandos."
        )
        bar_col = "#B91C1C"
        chart_title = "Top 15 Departamentos con Mayor Peligro Sísmico Promedio (vs. Peligros Geológicos)"
        map_title = "**Mapa de los 426 Distritos en Peligro Sísmico Alto y Muy Alto**"
        map_legend_items = [
            ("#991B1B", "Rojo oscuro — Muy Alto (104 distritos)", "Costa central (Lima–Callao–Ica) y extremo sur (Moquegua–Tacna)"),
            ("#EA580C", "Naranja — Alto (322 distritos)", "Franja costera y estribaciones andinas occidentales con fuerte exposición"),
        ]
        tbl_title = "#### Top 25 Distritos con Mayor Peligro Sísmico del Perú"
        dl_label = "📥 Descargar ranking completo de los 1,893 distritos ordenados de mayor a menor peligro (CSV)"
        dl_name = "ranking_mayor_a_menor_peligro_sismico_peru_1893_distritos.csv"

    c_chart, c_map = st.columns([1.05, 1.05])
    with c_chart:
        top15_dep = dep_agg.head(15).iloc[::-1]
        fig_dep = go.Figure()
        fig_dep.add_trace(
            go.Bar(
                y=top15_dep["departamento"],
                x=top15_dep["mean_idx"],
                orientation="h",
                marker_color=bar_col,
                text=[f"{idx:.1f}/100 · {int(pg):,} peligros geo." for idx, pg in zip(top15_dep["mean_idx"], top15_dep["peligros_geo"])],
                textposition="outside",
                hovertext=[
                    f"<b>{r['departamento']}</b><br>Índice Sísmico prom.: <b>{r['mean_idx']:.1f}/100</b><br>"
                    f"Sacudimiento prom.: {r['mean_pga']:.1f} Gal ({r['mean_pga']/9.81:.1f}% g)<br>"
                    f"Peligros geológicos: <b>{int(r['peligros_geo']):,}</b> ({int(r['zonas_crit'])} zonas críticas)"
                    for _, r in top15_dep.iterrows()
                ],
                hoverinfo="text",
            )
        )
        fig_dep.update_xaxes(range=[0, 118 if not is_low else 72], title_text="Índice Sísmico Promedio (0 = mínimo, 100 = máximo)")
        fig_dep.update_layout(**plotly_theme_kwargs(theme_mode), height=485, title=chart_title, margin=dict(l=10, r=45, t=48, b=40))
        st.plotly_chart(fig_dep, width="stretch", config=PLOTLY_CFG)
        st.caption("💡 Usa el ícono de expansión en la esquina superior derecha del gráfico para verlo en pantalla completa.")

    with c_map:
        st.markdown(map_title)
        map_html = _build_hazard_extremes_folium_html("low" if is_low else "high", theme_mode)
        st.iframe(map_html, height=420, width="stretch")
        st.markdown(render_map_legend(map_legend_items), unsafe_allow_html=True)

    st.markdown(tbl_title)
    dep_opts = ["Todo el Perú"] + sorted(df_u["departamento"].unique().tolist())
    sel_d = st.selectbox("Filtrar ranking por departamento (opcional)", dep_opts, index=0)
    sub_tbl = ranked_all if sel_d == "Todo el Perú" else ranked_all[ranked_all["departamento"] == sel_d]
    top25 = sub_tbl.head(25)

    tbl_ext = pd.DataFrame(
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
    st.markdown(render_centered_table(tbl_ext, badge_cols={"Nivel Sísmico", "Clasificación FONDES"}, max_height_px=450), unsafe_allow_html=True)
    st.markdown(
        render_table_legend(
            [
                ("Índice Sísmico (0–100)", "45% sacudimiento con suelo + 30% laguna costera BPT (30 años) + 25% tasa sísmica anual H3; no es porcentaje."),
                ("Sacudimiento Máx. (%g y Gal)", "Aceleración máxima esperada entre los 3 escenarios nacionales (100% g = 981 Gal)."),
                ("Mercalli + Rango Mw", "Intensidad estimada en el distrito y rango aproximado de Magnitud Mw equivalente en epicentro."),
                ("Peligros Geológicos (Flujos / Caídas)", "Puntos registrados por INGEMMET de huaicos/flujos de lodo y derrumbes/caídas de rocas."),
                ("Clasificación FONDES", "Nivel oficial multi-peligro de CENEPRED/SIGRID (incluye lluvias, inundaciones y heladas)."),
            ]
        ),
        unsafe_allow_html=True,
    )

    export_ext = ranked_all[
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
    st.download_button(dl_label, data=export_ext.to_csv(index=False).encode("utf-8"), file_name=dl_name, mime="text/csv", width="stretch")


def _render_sec8_audit(assets: dict[str, Any], theme_mode: str) -> None:
    st.markdown("### 🏛️ Auditoría Territorial: Peligro Sísmico Real vs. Clasificación Oficial (FONDES / SIGRID)")
    st.info(
        "🏛️ **¿Qué es FONDES / SIGRID y por qué encontramos una brecha?** "
        "Al promediar lluvias, inundaciones y huaicos, **150 distritos con Peligro Sísmico Alto o Muy Alto (35.2% del total)** "
        "aparecen clasificados como *'Muy Bajo'* o *'Bajo'* en el padrón oficial — incluyendo **21 de los 50 distritos de Lima y Callao**."
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
            f"(Índice `{r['seismic_hazard_index_0_100']:.1f}/100`, **{r['pga_escenario_max_gal']/9.81:.0f}% g / {r['pga_escenario_max_gal']:.0f} Gal**), "
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
        fig_hm = go.Figure(data=go.Heatmap(z=ct.to_numpy(), x=order, y=order, text=ct.to_numpy().astype(str), texttemplate="%{text}", colorscale="YlOrRd", showscale=False))
        fig_hm.update_layout(**plotly_theme_kwargs(theme_mode), height=405, title="Cruce de los 1,893 Distritos: Sísmico Real vs. FONDES", xaxis_title="Clasificación Oficial FONDES / SIGRID", yaxis_title="Peligro Sísmico Puro (Esta Plataforma)", margin=dict(l=60, r=20, t=50, b=50))
        st.plotly_chart(fig_hm, width="stretch", config=PLOTLY_CFG)

    with right:
        under = df_u[df_u["nivel_peligro_sismico_puro"].isin(["Alto", "Muy Alto"]) & df_u["nivel_riesgo_multipeligro_sigrid"].isin(["Muy Bajo", "Bajo"])].sort_values("seismic_hazard_index_0_100", ascending=False)
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
    st.download_button("⬇️ Descargar base territorial completa de los 1,893 distritos (CSV en español)", data=export_all.to_csv(index=False).encode("utf-8"), file_name="auditoria_peligro_sismico_peru_1893_distritos.csv", mime="text/csv")


def _render_sec9_documentation() -> None:
    st.markdown("### 📚 Documentación Técnica Completa y Rigor Estadístico")
    st.write(
        "Transparencia metodológica total para auditores, jurados técnicos y ciudadanos: fechas exactas de corte, "
        "auditoría ETL, formulaciones geofísicas, validación anti-leakage y limitaciones científicas explícitas."
    )

    st.markdown(
        """
        <div class="school-callout" style="border-left-color:#0284C7;">
          <h4 style="margin:0 0 0.45rem 0;color:var(--accent-title);">🧭 Traductor Ciudadano: ¿Qué tan confiables son estos cálculos?</h4>
          <ul style="margin:0;padding-left:1.2rem;line-height:1.55;">
            <li><strong>94.75% de bondad de ajuste del catálogo (GFT):</strong> Significa que los 12,390 sismos completos del IGP desde 1960 siguen casi a la perfección la ley física universal de Gutenberg–Richter (el estándar científico exige al menos 90%).</li>
            <li><strong>80.7% de precisión explicada en sacudimiento del suelo (R² = 0.807):</strong> Probamos el simulador ocultándole terremotos enteros que nunca había visto (<em>GroupKFold</em> sobre 956 sismos reales de ACELDAT) para asegurarnos de que no «haga trampa» memorizando eventos.</li>
            <li><strong>0.853 de capacidad de discriminación (ROC-AUC) en pronóstico por zonas:</strong> Evaluado año por año hacia el futuro entre <strong>2018 y 2025</strong> (sin mirar datos futuros), distingue con alta precisión qué zonas tienen mayor probabilidad relativa de registrar sismos sensibles.</li>
            <li><strong>Regla de honestidad científica:</strong> <em>Probabilidad estadística</em> (por ejemplo, «25% en 12 meses») indica qué tan frecuente es un evento en esa zona; <strong>nunca es una predicción exacta de fecha ni hora</strong>.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
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
    dl_col.download_button("⬇️ Descargar Documentación (.md)", data=md_text.encode("utf-8"), file_name="DOCUMENTACION_TECNICA_COMPLETA_PERU_SISMOS.md", mime="text/markdown", width="stretch")

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

    st.session_state.setdefault("theme_mode", THEME_LIGHT)
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
        toggle_label = "🌙 Cambiar a Modo Oscuro" if st.session_state["theme_mode"] == THEME_LIGHT else "☀️ Cambiar a Modo Claro"
        if st.button(toggle_label, key="btn_theme_toggle", width="stretch"):
            st.session_state["theme_mode"] = THEME_DARK if st.session_state["theme_mode"] == THEME_LIGHT else THEME_LIGHT
            st.rerun()

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

    theme_mode = str(st.session_state["theme_mode"])
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
        _render_sec7_hazard_extremes(assets, theme_mode)
    elif section == NAV_SECTIONS[7]:
        _render_sec8_audit(assets, theme_mode)
    else:
        _render_sec9_documentation()


if __name__ == "__main__":
    main()
