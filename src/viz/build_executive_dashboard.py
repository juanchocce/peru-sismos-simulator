"""Phase 4 - Step 2: Executive Interactive Map (Folium) & Executive Multi-Panel Dashboard (Plotly).

Generates two standalone HTML deliverables in `reports/`:
  1. `reports/mapa_peligro_sismico_peru.html` (Folium multi-layer GIS map):
       - Layer 1: H3 Res-5 hexagonal choropleth (`prob_m45_365d` forecast 2026-11-01)
       - Layer 2: Coastal Seismic Gaps (`mw_pot_p50` & `bpt_p30_a05`, highlighting Lima-Chincha & Ilo-Tacna)
       - Layer 3: ACELDAT accelerometric stations colored by site amplification vs MSOL + Lima Mw 8.6 PGA
       - Layer 4: GEOCATMIN Critical Zones (2,177 active geological hazard points via MarkerCluster)
  2. `reports/dashboard_ejecutivo_sismos_peru.html` (Plotly 2x2 executive dashboard):
       - Panel 1: Top 20 Lima Metropolitana & Callao districts by simulated PGA (Mw 8.6) & site amplification
       - Panel 2: Monte Carlo P10/P50/P90 Mw_pot + BPT 30/50yr probability across coastal macro-blocks
       - Panel 3: Cross-matrix Pure Seismic Hazard vs FONDES SIGRID Multi-Hazard (1,893 districts)
       - Panel 4: Top 15 Departments by seismic hazard index & GEOCATMIN geological hazards
"""

from __future__ import annotations

from pathlib import Path

import folium
import h3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from folium.plugins import MarkerCluster
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORTS_DIR = PROJECT_ROOT / "reports"

UBIGEO_PARQUET = PROCESSED_DIR / "territorial_multihazard_ubigeo.parquet"
H3_PARQUET = PROCESSED_DIR / "territorial_multihazard_h3.parquet"
GAPS_PROB_PARQUET = PROCESSED_DIR / "seismic_gaps_probabilistic.parquet"
SITE_FACTORS_PARQUET = PROCESSED_DIR / "aceldat_station_site_factors.parquet"
ZONAS_PARQUET = INTERIM_DIR / "geocatmin_zonas_criticas.parquet"

OUT_FOLIUM_MAP = REPORTS_DIR / "mapa_peligro_sismico_peru.html"
OUT_PLOTLY_DASH = REPORTS_DIR / "dashboard_ejecutivo_sismos_peru.html"


def _prob_color(p: float) -> str:
    """Color ramp for H3 annual M>=4.5 probability."""
    if p >= 0.20:
        return "#800026"
    if p >= 0.12:
        return "#bd0026"
    if p >= 0.07:
        return "#e31a1c"
    if p >= 0.04:
        return "#fc4e2a"
    if p >= 0.02:
        return "#fd8d3c"
    return "#feb24c"


def _amp_color(amp: float) -> str:
    """Color ramp for station amplification factor relative to rock MSOL."""
    if amp >= 1.60:
        return "#b10026"
    if amp >= 1.35:
        return "#e31a1c"
    if amp >= 1.15:
        return "#fd8d3c"
    if amp >= 0.90:
        return "#2b8cbe"
    return "#084081"


def _gap_color(mw_p50: float) -> str:
    if mw_p50 >= 8.35:
        return "#99000d"
    if mw_p50 >= 8.20:
        return "#cb181d"
    if mw_p50 >= 8.00:
        return "#ef3b2c"
    if mw_p50 >= 7.75:
        return "#fb6a4a"
    return "#fc9272"


def build_folium_map(
    df_u: pd.DataFrame,
    df_h3: pd.DataFrame,
    df_gaps: pd.DataFrame,
    df_site: pd.DataFrame,
    df_zc: pd.DataFrame,
) -> None:
    """Create multi-layer interactive Folium GIS map of Peruvian seismic hazard."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    m = folium.Map(location=[-10.5, -75.5], zoom_start=6, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satelite (Esri World Imagery)",
    ).add_to(m)

    # Layer 1: H3 Res-5 Hexagonal Choropleth (active cells prob_m45_365d >= 1.5%)
    fg_h3 = folium.FeatureGroup(name="1. Pronostico H3 Res-5: P(M>=4.5 en 365d, 2026-11-01)", show=True)
    active_h3 = df_h3[df_h3["prob_m45_365d"] >= 0.015].copy()
    features = []
    for _, r in active_h3.iterrows():
        boundary = h3.cell_to_boundary(str(r["h3_res5"]))
        coords = [[float(lon), float(lat)] for lat, lon in boundary]
        coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "h3_res5": str(r["h3_res5"]),
                    "prob_365d_pct": round(float(r["prob_m45_365d"]) * 100.0, 2),
                    "prob_90d_pct": round(float(r["prob_m45_90d"]) * 100.0, 2),
                    "pga_max_gal": round(float(r["pga_escenario_max_gal"]), 1),
                    "dist_trench_km": round(float(r["dist_trench_km"]), 0),
                    "fill_color": _prob_color(float(r["prob_m45_365d"])),
                },
            }
        )
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=lambda feat: {
            "fillColor": feat["properties"]["fill_color"],
            "color": "#555555",
            "weight": 0.35,
            "fillOpacity": 0.55,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["h3_res5", "prob_365d_pct", "prob_90d_pct", "pga_max_gal", "dist_trench_km"],
            aliases=["Celda H3:", "P(M>=4.5, 365d) [%]:", "P(M>=4.5, 90d) [%]:", "PGA Max Escenario (Gal):", "Dist. Fosa (km):"],
            sticky=True,
        ),
    ).add_to(fg_h3)
    fg_h3.add_to(m)

    # Layer 2: Coastal Seismic Gaps (30 segments + prominent macro-block highlights)
    fg_gaps = folium.FeatureGroup(name="2. Lagunas Sismicas Costeras (Mw_pot P50 y BPT 30a)", show=True)
    segs = df_gaps[df_gaps["entity_type"] == "segment"].copy()
    # Recover approximate trench lon per segment from coastal_seismic_gaps.parquet
    raw_gaps = pd.read_parquet(PROCESSED_DIR / "coastal_seismic_gaps.parquet")[["segment_id", "trench_lon_mean"]]
    segs = segs.merge(raw_gaps, left_on="entity_id", right_on="segment_id", how="left")
    for _, r in segs.iterrows():
        mw_p50 = float(r["mw_pot_p50"])
        mw_p10 = float(r["mw_pot_p10"])
        p10_str = "Sin déficit mayor (ruptura reciente)" if mw_p10 < 6.5 else f"Mw {mw_p10:.2f}"
        p30 = float(r["bpt_p30_a05"])
        yr_str = f"{r['last_major_rupture_year']:.0f}" if pd.notna(r["last_major_rupture_year"]) else "Sin registro desde 1552"
        is_critical = (-13.5 <= float(r["lat_center"]) <= -11.5) or (-18.5 <= float(r["lat_center"]) <= -17.0)
        popup_html = (
            f"<b>{r['coastal_zone']}</b><br>"
            f"Último gran terremoto (Mw ≥ 7.8): <b>{yr_str}</b> ({r['years_since_major_rupture']:.0f} años)<br>"
            f"Magnitud potencial acumulada: <b>Mw {mw_p50:.2f}</b> ({p10_str} – Mw {r['mw_pot_p90']:.2f})<br>"
            f"Probabilidad de gran ruptura: <b>30 años = {p30:.1f}%</b> | 50 años = {r['bpt_p50_a05']:.1f}%<br>"
            f"Actividad sísmica reciente: {r['modern_density_per_100km']:.1f} sismos/100 km"
        )
        folium.CircleMarker(
            location=[float(r["lat_center"]), float(r["trench_lon_mean"])],
            radius=11 if is_critical else 7,
            color="#111111" if is_critical else "#444444",
            weight=2.0 if is_critical else 1.0,
            fill=True,
            fill_color=_gap_color(mw_p50),
            fill_opacity=0.88,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{r['coastal_zone']} | Magnitud potencial: Mw {mw_p50:.2f} | Prob. 30 años: {p30:.1f}%",
        ).add_to(fg_gaps)
    fg_gaps.add_to(m)

    # Layer 3: ACELDAT Stations (amplification vs MSOL + Lima Mw 8.6 scenario PGA)
    fg_st = folium.FeatureGroup(name="3. Estaciones ACELDAT (Amplificacion de Sitio vs MSOL + PGA Lima Mw 8.6)", show=True)
    u_pga_lookup = df_u.set_index("ubigeo")[["pga_escenario_lima_mw86_gal", "mmi_escenario_lima_mw86", "nivel_peligro_sismico_puro"]].to_dict("index")
    for _, r in df_site.iterrows():
        u_info = u_pga_lookup.get(str(r["ubigeo"]), {})
        pga_lima = u_info.get("pga_escenario_lima_mw86_gal", np.nan)
        mmi_lima = u_info.get("mmi_escenario_lima_mw86", np.nan)
        amp = float(r["amp_factor_rel_msol"])
        popup_html = (
            f"<b>Estacion {r['cod_estacion']}</b> ({r['distrito']}, {r['departamento']})<br>"
            f"UBIGEO: {r['ubigeo']} | Registros ACELDAT: {int(r['n_records'])}<br>"
            f"Termino de sitio delta_S2S: <b>{r['delta_s2s_log10']:+.3f}</b><br>"
            f"Amplificacion vs Roca MSOL: <b>{amp:.2f}x</b><br>"
            f"Escenario Lima Mw 8.6: <b>{pga_lima:.1f} Gal</b> (MMI {mmi_lima:.1f})"
        )
        folium.CircleMarker(
            location=[float(r["station_lat"]), float(r["station_lon"])],
            radius=5.5,
            color="#222222",
            weight=0.8,
            fill=True,
            fill_color=_amp_color(amp),
            fill_opacity=0.90,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{r['cod_estacion']} ({r['distrito']}): {amp:.2f}x vs MSOL | Lima Mw8.6={pga_lima:.0f} Gal",
        ).add_to(fg_st)
    fg_st.add_to(m)

    # Layer 4: GEOCATMIN Critical Zones (2,177 active points clustered for smooth browser UX)
    fg_zc = folium.FeatureGroup(name="4. GEOCATMIN Zonas Criticas (2,177 peligros geologicos)", show=False)
    mc = MarkerCluster(name="Cluster Zonas Criticas").add_to(fg_zc)
    for _, r in df_zc.iterrows():
        pel = str(r["PELIGROS_G"]) if pd.notna(r["PELIGROS_G"]) else "Peligro Geologico"
        dist_n = str(r["DISTRITO"]) if pd.notna(r["DISTRITO"]) else ""
        reg_n = str(r["REGION"]) if pd.notna(r["REGION"]) else ""
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=4,
            color="#6a3d9a",
            fill=True,
            fill_color="#984ea3",
            fill_opacity=0.8,
            popup=f"<b>Zona Critica GEOCATMIN</b><br>{pel}<br>{dist_n}, {reg_n} (UBIGEO: {r['ubigeo']})",
        ).add_to(mc)
    fg_zc.add_to(m)

    # Compact floating legend
    legend_html = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: rgba(255,255,255,0.93);
                padding: 10px 14px; border: 1px solid #999; border-radius: 6px; font-family: Arial, sans-serif;
                font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
      <b>Mapa Multi-Amenaza Sismica — Peru (Fase 4)</b><br>
      <span style="color:#800026;">&#9632;</span> H3 P(M&ge;4.5, 365d) &ge; 20% &nbsp;
      <span style="color:#fc4e2a;">&#9632;</span> 4–20% &nbsp;
      <span style="color:#feb24c;">&#9632;</span> 1.5–4%<br>
      <span style="color:#99000d;">&#9679;</span> Laguna Costera Mw_pot &ge; 8.35 (borde negro = Lima-Chincha / Ilo-Tacna)<br>
      <span style="color:#b10026;">&#9679;</span> Estacion ACELDAT Amp &ge; 1.6x &nbsp;
      <span style="color:#fd8d3c;">&#9679;</span> 1.15–1.6x &nbsp;
      <span style="color:#2b8cbe;">&#9679;</span> &le; 1.15x (Roca MSOL=1.0x)<br>
      <span style="color:#984ea3;">&#9679;</span> Zonas Criticas GEOCATMIN (activar capa en control superior derecho)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(OUT_FOLIUM_MAP))


def build_plotly_executive_dashboard(df_u: pd.DataFrame, df_gaps: pd.DataFrame) -> None:
    """Create 2x2 executive interactive Plotly dashboard."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. Lima y Callao (Top 20): PGA Simulado Escenario Mw 8.6 (Gal) y Amplificacion de Sitio",
            "2. Macro-Bloques Costeros: Mw_pot (P10-P50-P90) y Probabilidad BPT 30/50 años",
            "3. Brecha Metodologica: Peligro Sismico Puro vs FONDES SIGRID (1,893 Distritos)",
            "4. Top 15 Departamentos: Indice de Peligro Sismico Medio y Peligros GEOCATMIN",
        ),
        specs=[
            [{"type": "xy"}, {"secondary_y": True}],
            [{"type": "xy"}, {"secondary_y": True}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.11,
    )

    # Panel 1: Top 20 Lima & Callao districts by Lima Mw 8.6 PGA
    lima20 = (
        df_u[df_u["provincia"].isin(["LIMA", "CALLAO"])]
        .sort_values("pga_escenario_lima_mw86_gal", ascending=False)
        .head(20)
        .iloc[::-1]
    )
    labels_lima = [
        f"{r['distrito']} ({r['primary_station_code']})" if pd.notna(r["primary_station_code"]) else r["distrito"]
        for _, r in lima20.iterrows()
    ]
    fig.add_trace(
        go.Bar(
            x=lima20["pga_escenario_lima_mw86_gal"],
            y=labels_lima,
            orientation="h",
            marker=dict(
                color=lima20["site_amp_factor_vs_msol"],
                colorscale="YlOrRd",
                cmin=0.9,
                cmax=2.1,
                showscale=False,
            ),
            text=[f"{p:.0f} Gal ({a:.2f}x | MMI {m:.1f})" for p, a, m in zip(lima20["pga_escenario_lima_mw86_gal"], lima20["site_amp_factor_vs_msol"], lima20["mmi_escenario_lima_mw86"])],
            textposition="outside",
            name="PGA Escenario Mw 8.6 (Gal)",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Macro-blocks MC P10/P50/P90 + BPT 30yr & 50yr
    mb = df_gaps[df_gaps["entity_type"] == "macro_block"].copy()
    short_names = {
        "BLOQUE_LAGUNA_CENTRAL_LIMA_CHINCHA": "Laguna Central<br>Lima-Chincha",
        "SUBBLOQUE_SUR_ILO_TACNA": "Sub-bloque Sur<br>Ilo-Tacna",
        "BLOQUE_NORTE": "Bloque Norte<br>Tumbes-Chimbote",
        "BLOQUE_CENTRO_NORTE": "Centro-Norte<br>Ancash-Huacho",
        "BLOQUE_SUR_NASCA_TACNA": "Sur Global<br>Nasca-Tacna",
    }
    mb["short"] = mb["entity_id"].map(short_names).fillna(mb["entity_id"])
    mb = mb.sort_values("mw_pot_p50", ascending=False)

    fig.add_trace(
        go.Bar(
            x=mb["short"],
            y=mb["mw_pot_p50"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(mb["mw_pot_p90"] - mb["mw_pot_p50"]).to_numpy(),
                arrayminus=(mb["mw_pot_p50"] - np.maximum(mb["mw_pot_p10"], 6.5)).to_numpy(),
                color="#1C1917",
            ),
            marker_color="#B91C1C",
            opacity=0.85,
            name="Magnitud potencial mediana (rango P10–P90)",
        ),
        row=1,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=mb["short"],
            y=mb["bpt_p30_a05"],
            mode="lines+markers+text",
            text=[f"{v:.0f}%" for v in mb["bpt_p30_a05"]],
            textposition="top center",
            line=dict(color="#1d3557", width=2.5),
            marker=dict(size=9),
            name="BPT P(30a, alpha=0.5) [%]",
        ),
        row=1,
        col=2,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=mb["short"],
            y=mb["bpt_p50_a05"],
            mode="lines+markers",
            line=dict(color="#457b9d", width=2, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            name="BPT P(50a, alpha=0.5) [%]",
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    # Panel 3: Cross-matrix Pure Seismic Hazard vs FONDES SIGRID (1,893 districts)
    order = ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]
    ct = pd.crosstab(df_u["nivel_peligro_sismico_puro"], df_u["nivel_riesgo_multipeligro_sigrid"]).reindex(index=order, columns=order, fill_value=0)
    z_mat = ct.to_numpy()
    fig.add_trace(
        go.Heatmap(
            z=z_mat,
            x=order,
            y=order,
            text=z_mat.astype(str),
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            name="Distritos",
        ),
        row=2,
        col=1,
    )

    # Panel 4: Top 15 Departments by mean seismic hazard index & total GEOCATMIN hazards
    dep_agg = (
        df_u.groupby("departamento", as_index=False)
        .agg(
            mean_seis_idx=("seismic_hazard_index_0_100", "mean"),
            max_pga_gal=("pga_escenario_max_gal", "max"),
            total_peligros=("n_peligros_total", "sum"),
            total_zonas_crit=("n_zonas_criticas_total", "sum"),
            n_dist=("ubigeo", "size"),
        )
        .sort_values("mean_seis_idx", ascending=False)
        .head(15)
    )
    fig.add_trace(
        go.Bar(
            x=dep_agg["departamento"],
            y=dep_agg["mean_seis_idx"],
            marker_color="#2a9d8f",
            name="Indice Peligro Sismico Medio (0-100)",
        ),
        row=2,
        col=2,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dep_agg["departamento"],
            y=dep_agg["total_peligros"],
            mode="lines+markers",
            line=dict(color="#e76f51", width=2.5),
            marker=dict(size=8),
            name="Peligros Geologicos GEOCATMIN (#)",
        ),
        row=2,
        col=2,
        secondary_y=True,
    )

    # Axes styling
    fig.update_xaxes(title_text="PGA Simulado (Gal = cm/s²)", range=[0, float(lima20["pga_escenario_lima_mw86_gal"].max()) * 1.32], row=1, col=1)
    fig.update_yaxes(title_text="Mw Potencial (MC)", range=[7.5, 9.0], row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Probabilidad Condicional BPT (%)", range=[0, 70], row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="FONDES SIGRID (Multi-Peligro)", row=2, col=1)
    fig.update_yaxes(title_text="Peligro Sismico Puro (Fase 4)", row=2, col=1)
    fig.update_yaxes(title_text="Indice Sismico Medio (0-100)", range=[0, 100], row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Total Peligros GEOCATMIN", row=2, col=2, secondary_y=True)
    fig.update_xaxes(tickangle=-35, row=2, col=2)

    fig.update_layout(
        height=1020,
        title_text="Dashboard Ejecutivo — Sistema Sismotectonico y Multi-Amenaza Territorial del Peru (1,893 Distritos)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="center", x=0.5),
        margin=dict(l=160, r=60, t=80, b=110),
    )
    fig.write_html(OUT_PLOTLY_DASH, include_plotlyjs="cdn")


def main() -> None:
    """Build Folium GIS map and Plotly executive dashboard."""
    for p in (UBIGEO_PARQUET, H3_PARQUET, GAPS_PROB_PARQUET, SITE_FACTORS_PARQUET, ZONAS_PARQUET):
        assert p.exists(), f"Missing prerequisite: {p}"

    df_u = pd.read_parquet(UBIGEO_PARQUET)
    df_h3 = pd.read_parquet(H3_PARQUET)
    df_gaps = pd.read_parquet(GAPS_PROB_PARQUET)
    df_site = pd.read_parquet(SITE_FACTORS_PARQUET)
    df_zc = pd.read_parquet(ZONAS_PARQUET)

    build_folium_map(df_u, df_h3, df_gaps, df_site, df_zc)
    build_plotly_executive_dashboard(df_u, df_gaps)

    assert OUT_FOLIUM_MAP.exists() and OUT_FOLIUM_MAP.stat().st_size > 50_000
    assert OUT_PLOTLY_DASH.exists() and OUT_PLOTLY_DASH.stat().st_size > 5_000

    print("=== FASE 4.2: VISUALIZACIONES EJECUTIVAS COMPLETADAS ===")
    print(f"Mapa Folium Multi-Capa: {OUT_FOLIUM_MAP} ({OUT_FOLIUM_MAP.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Dashboard Plotly 2x2:   {OUT_PLOTLY_DASH} ({OUT_PLOTLY_DASH.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
