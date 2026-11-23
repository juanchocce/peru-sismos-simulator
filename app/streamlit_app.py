"""Phase 5B: CivicTech / GovTech Interactive Seismic & Multi-Hazard Platform for Peru (Streamlit).

Unified, citizen-friendly web application presenting all Phase 0-5A datasets and models
in clear Spanish with zero raw snake_case jargon, editorial telluric styling, interactive
maps, live ShakeMap simulation (< 50 ms), 474-year seismic history + 3D Wadati-Benioff
cutaway, coastal seismic gaps radar, H3 probabilistic forecast, and FONDES/SIGRID audit.

Reads exclusively from `data/processed/`, `data/interim/`, and `models/`
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
from folium.plugins import MarkerCluster
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
    describe_mercalli,
    inject_custom_css,
    render_badge,
    render_kpi_card,
    translate_feature,
)
from src.models.nonlinear_site import (
    latlon_to_unit_xyz,
    simulate_hybrid_nonlinear_pga,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
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

NAZCA_TRENCH_BOUNDS = {"NZ\\SA", "NZ\\AP", "NZ\\ND"}

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
    (1970, 7.8, "1970 Áncash / Yungay (M 7.8)"),
    (2001, 8.4, "2001 Arequipa–Atico (M 8.4)"),
    (2007, 8.0, "2007 Pisco–Ica (M 8.0)"),
    (2019, 8.0, "2019 Lagunas, Loreto (M 8.0)"),
]


def _resolve_geojson_dir() -> Path:
    for p in (RAW_TECTONIC_ROOT / "tectonicplates-master" / "GeoJSON", RAW_TECTONIC_ROOT / "GeoJSON"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"PB2002 GeoJSON not found under {RAW_TECTONIC_ROOT}")


def load_simulation_assets() -> dict[str, Any]:
    """Load precomputed GMPE model, PB2002 trench tree, slab geometry, and district arrays."""
    df_u = pd.read_parquet(UBIGEO_PARQUET)
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
    mercalli_grades = [describe_mercalli(m)[0] for m in mmi_arr]
    mercalli_names = [describe_mercalli(m)[1] for m in mmi_arr]

    res_df = pd.DataFrame(
        {
            "ubigeo": df_u["ubigeo"],
            "departamento": df_u["departamento"],
            "provincia": df_u["provincia"],
            "distrito": df_u["distrito"],
            "dist_lat": df_u["dist_lat"],
            "dist_lon": df_u["dist_lon"],
            "primary_station_code": df_u["primary_station_code"],
            "site_delta_s2s_eff": np.round(sim["delta_s2s_eff"], 3),
            "site_amp_eff": np.round(np.power(10.0, sim["delta_s2s_eff"]), 2),
            "dist_epicentral_km": np.round(sim["dist_epicentral_km"], 1),
            "pga_rock_gal": np.round(sim["pga_rock_gal"], 1),
            "pga_site_gal": np.round(sim["pga_site_gal"], 1),
            "pga_site_pct_g": np.round(sim["pga_site_gal"] / 9.81, 1),
            "mmi_est": mmi_arr,
            "mercalli_romano": mercalli_grades,
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
    df_h3 = pd.read_parquet(H3_PARQUET)
    df_gaps = pd.read_parquet(GAPS_PROB_PARQUET)
    df_gaps_raw = pd.read_parquet(GAPS_RAW_PARQUET)[["segment_id", "trench_lon_mean"]]
    df_site = pd.read_parquet(SITE_FACTORS_PARQUET)
    df_zc = pd.read_parquet(ZONAS_PARQUET)
    h3_metrics = json.loads(H3_METRICS_JSON.read_text(encoding="utf-8"))

    # Map every H3 cell to its nearest Peruvian district for citizen-friendly location names
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

    # Unified 474-year historical + instrumental seismic catalog (1552-2026)
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


@st.cache_data(show_spinner="Generando mapa interactivo con capas seleccionadas...")
def _build_dynamic_folium_html(
    show_h3: bool, show_gaps: bool, show_stations: bool, show_geocatmin: bool
) -> str:
    """Build interactive OpenStreetMap multi-layer Folium HTML cached by toggle state."""
    assets = _cached_sim_assets()
    data = _cached_datasets()
    df_u = assets["df_u"]
    df_h3 = data["df_h3"]
    df_gaps = data["df_gaps"]
    df_gaps_raw = data["df_gaps_raw"]
    df_site = data["df_site"]
    df_zc = data["df_zc"]

    m = folium.Map(location=[-10.5, -75.5], zoom_start=6, tiles="OpenStreetMap", control_scale=True)

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
                "fillOpacity": 0.52,
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
            popup_html = (
                f"<b>Estación IGP {r['cod_estacion']}</b> — {r['distrito']} ({r['departamento']})<br>"
                f"Sismos registrados: {int(r['n_records'])}<br>"
                f"Amplificación del suelo frente a roca firme: <b>{amp:.2f} veces</b><br>"
                f"Sacudimiento en escenario Lima M 8.6: <b>{pga_l:.0f} Gal</b> (Mercalli {describe_mercalli(mmi_l)[0]})"
            )
            folium.CircleMarker(
                location=[float(r["station_lat"]), float(r["station_lon"])],
                radius=5.5,
                color="#1C1917",
                weight=0.8,
                fill=True,
                fill_color=color,
                fill_opacity=0.90,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Estación {r['cod_estacion']} ({r['distrito']}): amplificación {amp:.2f}x",
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


def _render_citizen_header() -> None:
    st.markdown(
        """
        <div class="civic-hero">
          <h1>Observatorio Sísmico y Multi-Amenaza del Perú</h1>
          <p>
            Plataforma ciudadana de análisis sismotectónico, simulación de terremotos en tiempo real y auditoría territorial
            para los <strong>1,893 distritos del Perú</strong>. Basada en 474 años de registros del IGP (1552–2026),
            acelerómetros ACELDAT, catálogo USGS, zonas críticas de INGEMMET/GEOCATMIN y el listado oficial FONDES/SIGRID.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📖 ¿Cómo leer este monitor en 30 segundos? (Guía Ciudadana)", expanded=False):
        g1, g2, g3, g4 = st.columns(4)
        g1.markdown(
            """
            <div class="guide-card">
              <h4>1. Magnitud (Mw) vs. Intensidad (Mercalli)</h4>
              <p>La <strong>magnitud</strong> es un único número que mide la energía total liberada por el sismo.
              La <strong>intensidad (escala de Mercalli I–X)</strong> cambia según dónde estés: mide qué tan fuerte se sacude
              el suelo y qué daños puede causar en tu distrito.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        g2.markdown(
            """
            <div class="guide-card">
              <h4>2. Aceleración del Suelo y Efecto de Suelo</h4>
              <p>Un mismo terremoto puede sentirse <strong>hasta 2 veces más fuerte</strong> sobre suelos arenosos, rellenos o valles
              blandos (como Puente Piedra, Villa El Salvador o La Molina) que sobre roca firme (como el Morro Solar en Chorrillos).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        g3.markdown(
            """
            <div class="guide-card">
              <h4>3. ¿Qué es una Laguna o Silencio Sísmico?</h4>
              <p>Son tramos de la costa donde la placa de Nazca lleva <strong>siglos trabada</strong> contra Sudamérica sin liberar
              energía en un gran terremoto. Mientras más años pasan sin ruptura (ej. Lima desde 1746 o Tacna desde 1868), mayor energía se acumula.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        g4.markdown(
            """
            <div class="guide-card">
              <h4>4. Transparencia Científica</h4>
              <p><strong>Ningún sistema en el mundo puede predecir el día ni la hora exacta de un terremoto.</strong>
              Esta plataforma calcula probabilidades estadísticas por zona, energía acumulada en la costa y el impacto esperado
              en cada distrito para prevención y gestión de emergencias.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_tab1_national_map(assets: dict[str, Any], data: dict[str, Any]) -> None:
    st.markdown("### Mapa Nacional Multi-Amenaza (Capas Interactivas)")
    st.write("Activa o desactiva las capas para explorar cómo coinciden la probabilidad sísmica, las lagunas costeras, la amplificación de suelos y las zonas críticas de huaicos o deslizamientos.")

    ctrl_col, map_col = st.columns([0.34, 0.66])
    with ctrl_col:
        st.markdown("#### Capas del Mapa")
        show_h3 = st.checkbox("Probabilidad de sismos M≥4.5 en los próximos 12 meses (Grilla hexagonal)", value=True)
        show_gaps = st.checkbox("Lagunas Sísmicas Costeras (Zonas con energía acumulada)", value=True)
        show_stations = st.checkbox("Estaciones Acelerográficas del IGP (Amplificación del suelo medida)", value=True)
        show_geocatmin = st.checkbox("Zonas Críticas de Deslizamientos y Huaicos (GEOCATMIN — 2,177 puntos)", value=False)

        st.markdown("---")
        st.markdown("#### Leyenda Ciudadana")
        st.markdown(
            """
            - 🔴 **Hexágonos rojos oscuros:** Mayor probabilidad de sismos sensibles (≥ 12% anual).
            - 🟠 **Hexágonos naranjas/amarillos:** Probabilidad moderada (2% a 12% anual).
            - ⚫ **Círculos costeros con borde negro:** Tramos con gran energía acumulada (**Lima–Chincha** e **Ilo–Tacna**).
            - 🟢/🟠/🔴 **Puntos pequeños en ciudades:** Estaciones del IGP según cuánto amplifica el suelo local frente a roca firme (`🟢 ≤1.2x`, `🟠 1.2–1.5x`, `🔴 ≥1.5x`).
            - 🟣 **Puntos morados agrupados:** Zonas críticas activas de deslizamientos, caída de rocas o huaicos (INGEMMET).
            """
        )
        st.info(
            "💡 **Consejo:** Haz clic sobre cualquier círculo en el mar o estación en tierra para ver el detalle de energía acumulada y sacudimiento estimado."
        )

    with map_col:
        html_map = _build_dynamic_folium_html(show_h3, show_gaps, show_stations, show_geocatmin)
        st.iframe(html_map, height=610, width="stretch")


def _render_tab2_simulator(assets: dict[str, Any]) -> None:
    st.markdown("### Simulador de Terremotos en Vivo: ¿Cómo afectaría a mi distrito?")
    st.write("Elige un escenario real o mueve el epicentro y la magnitud para calcular en milisegundos la sacudida esperada en los 1,893 distritos del país, incluyendo el tipo de suelo local.")

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
    ev_lat = c1.slider("Latitud del epicentro (°S)", -18.5, -3.0, float(st.session_state["sim_lat"]), 0.05, help="Posición Norte–Sur del terremoto. Lima está cerca de -12.0° y Tacna cerca de -18.0°.")
    ev_lon = c2.slider("Longitud del epicentro (°W)", -82.0, -68.5, float(st.session_state["sim_lon"]), 0.05, help="Posición Este–Oeste. Valores más negativos están mar adentro cerca de la fosa.")
    ev_depth = c3.slider("Profundidad focal (km)", 5.0, 350.0, float(st.session_state["sim_depth"]), 1.0, help="Los sismos superficiales (<60 km) se sienten mucho más intensos en superficie que los profundos.")
    ev_mw = c4.slider("Magnitud del terremoto (Mw)", 4.0, 9.0, float(st.session_state["sim_mw"]), 0.1, help="Escala de magnitud momento. Cada +1.0 libera ~32 veces más energía.")
    use_nl = c5.toggle(
        "Amortiguamiento no lineal de suelos blandos",
        value=True,
        help="En terremotos extremos (>150 Gal en roca), los suelos blandos pierden rigidez y amortiguan los picos lineales irreales (modelo Seyhan & Stewart 2014).",
    )

    res_df, meta = run_live_simulation(assets, ev_lat, ev_lon, ev_depth, ev_mw, use_nonlinear_damping=use_nl)

    # Personalized "My District" spotlight BEFORE the general table
    res_df["etiqueta_distrito"] = res_df["distrito"] + " — " + res_df["provincia"] + " (" + res_df["departamento"] + ") · UBIGEO " + res_df["ubigeo"]
    default_idx = int(res_df.index[res_df["ubigeo"] == "150142"][0]) if (res_df["ubigeo"] == "150142").any() else 0
    sel_dist_lbl = st.selectbox("🔍 Consulta rápida — ¿Cómo se sentiría este sismo en mi distrito?", res_df["etiqueta_distrito"].tolist(), index=default_idx)
    drow = res_df[res_df["etiqueta_distrito"] == sel_dist_lbl].iloc[0]
    m_rom, m_short, m_desc, m_col = describe_mercalli(float(drow["mmi_est"]))

    st.markdown(
        f"""
        <div class="district-spotlight" style="--accent:{m_col};">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <div>
              <strong style="font-size:1.12rem;">{drow['distrito']} ({drow['provincia']}, {drow['departamento']})</strong>
              <span style="color:#78716C;font-size:0.88rem;"> · Distancia al epicentro: <strong>{drow['dist_epicentral_km']:.0f} km</strong></span>
            </div>
            <div>{render_badge(f"Mercalli {m_rom} — {m_short}", m_col)}</div>
          </div>
          <p style="margin:0.45rem 0 0.65rem 0;color:#44403C;font-size:0.93rem;">
            <strong>Impacto cotidiano esperado:</strong> {m_desc}
          </p>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;font-size:0.88rem;">
            <div><strong>Sacudimiento real:</strong> {drow['pga_site_pct_g']:.1f}% de gravedad ({drow['pga_site_gal']:.0f} Gal)</div>
            <div><strong>Efecto del suelo local:</strong> {drow['site_amp_eff']:.2f}× frente a roca firme</div>
            <div><strong>Zonas críticas GEOCATMIN:</strong> {int(drow['n_zonas_criticas_activas'])} activas ({int(drow['n_peligros_total'])} peligros)</div>
            <div><strong>Clasificación oficial FONDES:</strong> {drow['nivel_riesgo_multipeligro_sigrid']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Non-truncating HTML KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(render_kpi_card("Tipo de Sismo", meta["resolved_regime_es"], f"Calculado en {meta['latency_ms']:.0f} ms", "#0F766E", meta["resolved_regime_full_es"]), unsafe_allow_html=True)
    max_rom, max_short, _, max_col = describe_mercalli(meta["max_mmi"])
    k2.markdown(render_kpi_card("Sacudimiento Máximo", f"{meta['max_pga_pct_g']:.0f}% g", f"{meta['max_pga_gal']:.0f} Gal (cm/s²)", max_col, "Porcentaje de la aceleración de la gravedad (100% g = 981 Gal). A partir de 30% g ocurren daños graves en viviendas vulnerables."), unsafe_allow_html=True)
    k3.markdown(render_kpi_card("Intensidad Máxima", f"Mercalli {max_rom}", max_short, max_col, "Intensidad instrumental estimada en la escala modificada de Mercalli (I a X)."), unsafe_allow_html=True)
    k4.markdown(render_kpi_card("Distritos en Daño Fuerte", f"{meta['n_districts_mmi7_plus']} distritos", f"Mercalli ≥ VII ({meta['n_districts_mmi8_plus']} en ≥ VIII)", "#C2410C", "Número de distritos donde el sacudimiento supera Mercalli VII (daños importantes en adobe y muros débiles)."), unsafe_allow_html=True)
    k5.markdown(render_kpi_card("Zonas Críticas Expuestas", f"{meta['exposed_zonas_criticas_mmi7']:,}", "Deslizamientos/huaicos en MMI ≥ VII", "#7C3AED", "Cantidad de zonas críticas activas de INGEMMET/GEOCATMIN ubicadas dentro de distritos con sacudimiento muy fuerte."), unsafe_allow_html=True)

    st.markdown("")
    map_col, tbl_col = st.columns([1.15, 1.05])
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
                    f"Intensidad: <b>Mercalli {r['mercalli_romano']} ({r['nivel_impacto']})</b><br>"
                    f"Sacudimiento: {r['pga_site_pct_g']:.1f}% g ({r['pga_site_gal']:.0f} Gal)<br>"
                    f"Amplificación suelo: {r['site_amp_eff']:.2f}× | Dist.: {r['dist_epicentral_km']:.0f} km"
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
                marker=dict(size=18, color="#1C1917"),
                text=[f"<b>Epicentro simulado</b><br>Magnitud: Mw {ev_mw:.1f}<br>Profundidad: {ev_depth:.0f} km"],
                hoverinfo="text",
                name="Epicentro",
            )
        )
        fig_map.update_layout(
            map=dict(style="open-street-map", center=dict(lat=ev_lat, lon=ev_lon), zoom=5.5),
            height=480,
            margin=dict(l=0, r=0, t=8, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_map, width="stretch")

    with tbl_col:
        st.markdown("**Top 20 Distritos con Mayor Sacudimiento Esperado**")
        top20 = res_df.sort_values("pga_site_gal", ascending=False).head(20).copy()
        top20["ubicacion"] = top20["provincia"] + " / " + top20["departamento"]
        top20["mercalli_etq"] = top20["mercalli_romano"] + " (" + top20["mmi_est"].map(lambda x: f"{x:.1f}") + ")"
        st.dataframe(
            top20[["distrito", "ubicacion", "pga_rock_gal", "pga_site_gal", "site_amp_eff", "mercalli_etq", "nivel_impacto"]],
            hide_index=True,
            width="stretch",
            height=425,
            column_config={
                "distrito": st.column_config.TextColumn("Distrito"),
                "ubicacion": st.column_config.TextColumn("Provincia / Dpto."),
                "pga_rock_gal": st.column_config.NumberColumn("En Roca (Gal)", format="%.0f", help="Aceleración estimada si el distrito estuviera sobre roca firme (100 Gal ≈ 10% de gravedad)."),
                "pga_site_gal": st.column_config.NumberColumn("Real c/ Suelo (Gal)", format="%.0f", help="Aceleración esperada incorporando el efecto no lineal del suelo local."),
                "site_amp_eff": st.column_config.NumberColumn("Suelo", format="%.2fx", help="Factor multiplicador efectivo del suelo local frente a roca firme."),
                "mercalli_etq": st.column_config.TextColumn("Mercalli", help="Grado en la Escala Modificada de Mercalli (I a X)."),
                "nivel_impacto": st.column_config.TextColumn("Nivel de Impacto"),
            },
        )
        export_df = res_df.sort_values("pga_site_gal", ascending=False)[
            ["ubigeo", "departamento", "provincia", "distrito", "dist_epicentral_km", "pga_rock_gal", "pga_site_gal", "pga_site_pct_g", "site_amp_eff", "mercalli_romano", "nivel_impacto", "n_zonas_criticas_activas"]
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
                "site_amp_eff": "Factor_Amplificacion_Suelo",
                "mercalli_romano": "Intensidad_Mercalli",
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


def _render_tab3_history_and_3d(data: dict[str, Any]) -> None:
    st.markdown("### Historia Sísmica del Perú (1552–2026) y Corte 3D de la Placa de Nazca")
    st.write("Explora casi cinco siglos de actividad sísmica documentada por el IGP y observa en 3D cómo cambia el ángulo de la placa de Nazca al cruzar la latitud 15°S (Ica/Nasca).")

    df_cat = data["df_cat"]
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
    # Timeline of major earthquakes M >= 6.5
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
        for yr_m, mag_m, txt_m in HISTORICAL_MILESTONES:
            if yr_range[0] <= yr_m <= yr_range[1] and mag_m >= min_mag:
                fig_tl.add_annotation(
                    x=yr_m,
                    y=mag_m,
                    text=txt_m,
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-32 if yr_m not in (1940, 2007) else 28,
                    font=dict(size=11, color="#1C1917"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#D6CEC2",
                )
    fig_tl.update_layout(
        height=390,
        title="Línea de Tiempo de Grandes Terremotos del Perú (Magnitud ≥ 6.5)",
        xaxis_title="Año",
        yaxis_title="Magnitud (Mw)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_tl, width="stretch")

    # 3D Wadati-Benioff Subduction Cutaway
    st.markdown("#### Corte 3D de la Placa de Nazca bajo el Perú (Geometría Wadati–Benioff)")
    st.info(
        "🧭 **¿Por qué hay volcanes en Arequipa y Tacna, pero no en Lima o Piura?** "
        "Gira este modelo 3D: al **Norte y Centro** (arriba de los -15°S), la placa oceánica de Nazca se desliza casi horizontal "
        "(**subducción plana a ~12°**), impidiendo el vulcanismo pero generando fuerte acoplamiento costero. Al **Sur de Ica/Nasca**, "
        "la placa se hunde abruptamente (**~30°**), alcanzando 270 km de profundidad y alimentando el arco volcánico del sur."
    )
    sub_3d = sub[(sub["depth_km"].notna()) & (sub["depth_km"] <= 350.0)].copy()
    if len(sub_3d) > 3500:
        sub_3d = sub_3d.sample(3500, random_state=42)
    color_map_reg = {
        "Subducción Costera": "#B91C1C",
        "Falla Cortical Andina": "#D97706",
        "Losa Intermedia": "#0F766E",
        "Foco Profundo": "#1E3A8A",
    }
    fig_3d = go.Figure()
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
                marker=dict(size=2.8, color=col_hex, opacity=0.68),
                text=[f"Año {int(a)} | Mw {m:.1f} | Prof. {d:.0f} km" for a, m, d in zip(r_df["anio"], r_df["mag"], r_df["depth_km"])],
                hoverinfo="text",
            )
        )
    fig_3d.update_layout(
        height=560,
        template="plotly_white",
        scene=dict(
            xaxis_title="Longitud (°W)",
            yaxis_title="Latitud (°S)",
            zaxis_title="Profundidad subterránea (-km)",
            aspectratio=dict(x=1.0, y=1.35, z=0.58),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=35, b=0),
    )
    st.plotly_chart(fig_3d, width="stretch")


def _render_tab4_gaps(data: dict[str, Any]) -> None:
    st.markdown("### Radar de Silencios Sísmicos: Energía Acumulada en la Costa")
    st.write(
        "Mediante 10,000 simulaciones Monte Carlo, estimamos cuánta energía elástica se ha acumulado en cada tramo del litoral desde su último megaterremoto "
        "y la probabilidad condicional de una gran ruptura (Mw ≥ 8.0) en los próximos 30 y 50 años."
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
                        color="#1C1917",
                    ),
                    marker_color="#B91C1C",
                    text=[f"Mw {v:.2f}" for v in sub["mw_pot_p50"]],
                    textposition="outside",
                    name="Magnitud potencial mediana",
                )
            )
            fig_mw.update_yaxes(range=[7.5, 9.05], title_text="Magnitud potencial acumulada (Mw)")
            fig_mw.update_layout(height=420, template="plotly_white", title="Energía Acumulada (Mediana y rango P10–P90)", margin=dict(l=40, r=20, t=50, b=90))
            st.plotly_chart(fig_mw, width="stretch")

        with col_r:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Bar(x=sub["nombre_es"], y=sub[p30_col], name="Próximos 30 años (%)", marker_color="#0F766E", text=[f"{v:.0f}%" for v in sub[p30_col]], textposition="outside"))
            fig_pr.add_trace(go.Bar(x=sub["nombre_es"], y=sub[p50_col], name="Próximos 50 años (%)", marker_color="#D97706", text=[f"{v:.0f}%" for v in sub[p50_col]], textposition="outside"))
            fig_pr.update_yaxes(range=[0, 100], title_text="Probabilidad condicional (%)")
            fig_pr.update_layout(height=420, barmode="group", template="plotly_white", title="Probabilidad de Gran Terremoto (Mw ≥ 8.0)", margin=dict(l=40, r=20, t=50, b=90))
            st.plotly_chart(fig_pr, width="stretch")
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
                fillcolor="rgba(185,28,28,0.20)",
                line=dict(width=0),
                name="Rango de incertidumbre (P10–P90)",
            )
        )
        seg_hover = [
            (
                f"<b>{r['coastal_zone']}</b><br>Mediana: Mw {r['mw_pot_p50']:.2f}<br>"
                + (
                    "Rango: Sin déficit mayor (ruptura reciente)"
                    if float(r["mw_pot_p10"]) < 6.5
                    else f"Rango: Mw {float(r['mw_pot_p10']):.2f}"
                )
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
        fig_seg.update_layout(height=470, template="plotly_white", title="Perfil de Energía Acumulada de Norte a Sur (-3.5°S a -18.5°S)")
        st.plotly_chart(fig_seg, width="stretch")

    # Clean Spanish table
    tbl = pd.DataFrame(
        {
            "Tramo Costero": sub["nombre_es"],
            "Último Gran Terremoto": [f"Año {int(round(y))}" if pd.notna(y) else "Sin registro desde 1552" for y in sub["last_major_rupture_year"]],
            "Años Acumulando Energía": np.round(sub["years_since_major_rupture"], 0).astype(int),
            "Magnitud Potencial Estimada": [f"Mw {v:.2f}" for v in sub["mw_pot_p50"]],
            "Rango de Incertidumbre (P10–P90)": [
                f"Sin déficit mayor – Mw {hi:.2f}" if lo < 6.5 else f"Mw {lo:.2f} – {hi:.2f}"
                for lo, hi in zip(sub["mw_pot_p10"], sub["mw_pot_p90"])
            ],
            "Probabilidad a 30 años (%)": sub[p30_col],
            "Probabilidad a 50 años (%)": sub[p50_col],
            "Sismos Recientes (cada 100 km)": sub["modern_density_per_100km"],
        }
    )
    st.dataframe(
        tbl,
        hide_index=True,
        width="stretch",
        column_config={
            "Probabilidad a 30 años (%)": st.column_config.ProgressColumn("Probabilidad a 30 años", format="%.1f%%", min_value=0, max_value=100, help="Probabilidad condicional de un megaterremoto Mw ≥ 8.0 en los próximos 30 años."),
            "Probabilidad a 50 años (%)": st.column_config.ProgressColumn("Probabilidad a 50 años", format="%.1f%%", min_value=0, max_value=100, help="Probabilidad condicional de un megaterremoto Mw ≥ 8.0 en los próximos 50 años."),
            "Sismos Recientes (cada 100 km)": st.column_config.NumberColumn("Actividad Reciente (/100 km)", format="%.1f", help="Sismos moderados M≥4.7 entre 2000 y 2026 por cada 100 km de costa. Valores bajos en zonas de alto déficit confirman silencio sísmico (acoplamiento trabado)."),
        },
    )


def _render_tab5_forecast(assets: dict[str, Any], data: dict[str, Any]) -> None:
    st.markdown("### Pronóstico Probabilístico por Zonas (Próximos 3 y 12 Meses)")
    st.write(
        "Este modelo de inteligencia artificial (calibrado mediante validación temporal estricta 2018–2025) estima la **probabilidad de que ocurra al menos un sismo moderado o fuerte** "
        "en cada zona del país según la actividad reciente, réplicas y cercanía a la fosa marina. *No predice fechas exactas.*"
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
    min_pct = c3.slider("Mostrar zonas con probabilidad mínima (%)", 0.0, 20.0, 2.0 if "12 meses" in h_lbl else 0.5, 0.5)

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
                        opacity=0.80,
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
                map=dict(
                    style="open-street-map",
                    center=dict(lat=float(vis["cell_lat"].mean()), lon=float(vis["cell_lon"].mean())),
                    zoom=5.2 if sel_depa == "Todo el Perú" else 7.0,
                ),
                height=440,
                margin=dict(l=0, r=0, t=8, b=0),
            )
            st.plotly_chart(fig_f, width="stretch")
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
        fig_sh.update_layout(height=410, template="plotly_white", margin=dict(l=10, r=15, t=10, b=35))
        st.plotly_chart(fig_sh, width="stretch")

    st.markdown(f"**Top 20 Zonas con Mayor Probabilidad Estimada ({h_lbl})**")
    top_z = vis.sort_values(prob_col, ascending=False).head(20)
    tbl_z = pd.DataFrame(
        {
            "Ubicación de Referencia": top_z["ubicacion_legible"],
            "Departamento": top_z["departamento_cercano"],
            "Probabilidad Estimada (%)": top_z["prob_pct"],
            "Cercanía a la Fosa Marina (km)": np.round(top_z["dist_trench_km"], 0),
            "Profundidad de Placa (km)": np.round(top_z["slab_depth_est_km"], 0),
            "Peligros Geológicos Cercanos": top_z["n_geocatmin_peligros"],
        }
    )
    st.dataframe(
        tbl_z,
        hide_index=True,
        width="stretch",
        column_config={
            "Probabilidad Estimada (%)": st.column_config.ProgressColumn("Probabilidad Estimada", format="%.1f%%", min_value=0, max_value=float(max(50.0, top_z["prob_pct"].max() if not top_z.empty else 50.0))),
        },
    )


def _render_tab6_audit(assets: dict[str, Any]) -> None:
    st.markdown("### Auditoría Territorial: Peligro Sísmico Real vs. Clasificación Oficial (FONDES / SIGRID)")
    st.info(
        "🏛️ **¿Qué es FONDES / SIGRID y por qué encontramos una brecha?** "
        "El Fondo para Intervenciones ante la Ocurrencia de Desastres Naturales (FONDES) utiliza la calificación multi-peligro de SIGRID "
        "para priorizar presupuestos de prevención. Sin embargo, al promediar lluvias, inundaciones y huaicos, **150 distritos con Peligro Sísmico "
        "Alto o Muy Alto (35% del total)** aparecen clasificados como *'Riesgo Muy Bajo'* o *'Bajo'* — incluyendo **21 de los 50 distritos de Lima y Callao** "
        "(como Villa El Salvador, San Juan de Miraflores, Villa María del Triunfo, Independencia y La Molina)."
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
            f"({r['seismic_hazard_index_0_100']:.1f}/100, sacudimiento estimado de **{r['pga_escenario_max_gal']:.0f} Gal**), "
            f"pero figura como **{nivel_fondes}** en el listado oficial FONDES / SIGRID."
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(render_kpi_card("Peligro Sísmico Puro", nivel_seis, f"Índice {r['seismic_hazard_index_0_100']:.1f} / 100", RISK_LEVEL_COLORS.get(nivel_seis, "#B91C1C"), "Combina sacudimiento en megaterremotos con suelo local (45%), laguna costera BPT (30%) y frecuencia anual (25%)."), unsafe_allow_html=True)
    c2.markdown(render_kpi_card("Oficial FONDES / SIGRID", nivel_fondes, "Índice multi-peligro estatal", RISK_LEVEL_COLORS.get(nivel_fondes, "#475569"), "Nivel oficial reportado por CENEPRED/SIGRID en el padrón FONDES vigente."), unsafe_allow_html=True)
    m_rom, m_short, _, m_col = describe_mercalli(float(r["mmi_escenario_max"]))
    c3.markdown(render_kpi_card("Escenario Crítico", f"{r['pga_escenario_max_gal'] / 9.81:.0f}% g", f"{r['pga_escenario_max_gal']:.0f} Gal · Mercalli {m_rom}", m_col, "Máximo sacudimiento esperado entre los 3 escenarios simulados (Lima M 8.6, Sur M 8.4, Intraplaca M 6.8)."), unsafe_allow_html=True)
    st_lbl = f"Estación IGP {r['primary_station_code']}" if pd.notna(r["primary_station_code"]) else "Sin estación (valor neutro)"
    c4.markdown(render_kpi_card("Efecto de Suelo Local", f"{r['site_amp_factor_vs_msol']:.2f}×", st_lbl, "#B45309", "Factor de amplificación del suelo frente a roca firme (Morro Solar = 1.00x)."), unsafe_allow_html=True)
    c5.markdown(render_kpi_card("Peligros Geológicos", f"{int(r['n_peligros_total'])} registros", f"{int(r['n_zonas_criticas_activas'])} zonas críticas activas", "#6D28D9", "Inventario INGEMMET/GEOCATMIN de deslizamientos, huaicos y caídas de rocas en el distrito."), unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([0.95, 1.15])
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
            height=390,
            title="Cruce de los 1,893 Distritos: Sísmico Real vs. FONDES",
            xaxis_title="Clasificación Oficial FONDES / SIGRID",
            yaxis_title="Peligro Sísmico Puro (Esta Plataforma)",
            template="plotly_white",
            margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(fig_hm, width="stretch")

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
                "Índice Sísmico (0–100)": under["seismic_hazard_index_0_100"],
                "Peligro Sísmico Real": under["nivel_peligro_sismico_puro"],
                "Clasificación FONDES": under["nivel_riesgo_multipeligro_sigrid"],
                "Sacudimiento Máx. (Gal)": under["pga_escenario_max_gal"],
            }
        )
        st.dataframe(tbl_u, hide_index=True, width="stretch", height=345)

    # Full downloadable territorial table with clean Spanish headers
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


def main() -> None:
    st.set_page_config(
        page_title="Observatorio Sísmico y Multi-Amenaza del Perú",
        page_icon="🌎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(inject_custom_css(), unsafe_allow_html=True)

    assets = _cached_sim_assets()
    data = _cached_datasets()

    with st.sidebar:
        st.markdown("### Acerca de los Datos")
        st.markdown(
            "- **Catálogo Sísmico IGP + USGS:** 26,976 sismos (1552–2026)\n"
            "- **Red Acelerográfica ACELDAT:** 17,794 registros en 277 estaciones\n"
            "- **Cobertura Territorial:** 1,893 distritos del Perú (100% UBIGEO)\n"
            "- **Peligros Geológicos:** 37,023 peligros y 2,177 zonas críticas (INGEMMET)\n"
            "- **Rigor Científico:** Validación sin fuga temporal ni por evento; amortiguamiento no lineal de suelos blandos."
        )

    _render_citizen_header()

    tabs = st.tabs(
        [
            "🗺️ Mapa Nacional Multi-Amenaza",
            "⚡ Simulador de Terremotos en Vivo",
            "📜 Historia Sísmica (1552–2026) y Corte 3D",
            "🌊 Radar de Silencios Sísmicos",
            "📊 Pronóstico por Zonas (3 y 12 Meses)",
            "🏛️ Auditoría vs. FONDES / SIGRID",
        ]
    )
    with tabs[0]:
        _render_tab1_national_map(assets, data)
    with tabs[1]:
        _render_tab2_simulator(assets)
    with tabs[2]:
        _render_tab3_history_and_3d(data)
    with tabs[3]:
        _render_tab4_gaps(data)
    with tabs[4]:
        _render_tab5_forecast(assets, data)
    with tabs[5]:
        _render_tab6_audit(assets)


if __name__ == "__main__":
    main()
