"""CivicTech / GovTech Editorial Visual Theme & Citizen-Language Dictionaries (Phase 5C).

Provides:
  - Custom CSS for warm telluric institutional styling (cards, badges, sidebar nav, centered tables).
  - `render_centered_table(df, ...)`: responsive 100%-centered HTML table renderer with zero
    truncation, sticky header, clean hover highlight, and optional semantic pill badges.
  - `FEATURE_LABELS_ES`, `REGIME_LABELS_ES`, `REGIME_SHORT_ES`, `MACRO_BLOCK_LABELS_ES`.
  - `MERCALLI_DESC_ES` & `describe_mercalli(mmi)`: everyday physical impact descriptions.
  - Reusable HTML KPI card & badge components that never truncate text.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

FEATURE_LABELS_ES: dict[str, str] = {
    # Model A (GMPE) features
    "event_mag": "Magnitud del sismo (Mw)",
    "dist_epicentral_km": "Distancia desde el epicentro (km)",
    "station_dist_trench_km": "Distancia del distrito a la fosa marina (km)",
    "dist_hypocentral_km": "Distancia real desde el foco subterráneo (km)",
    "station_site_term_oof": "Tipo de suelo local (amplificación medida)",
    "log10_r_epi": "Atenuación geométrica por distancia superficial",
    "event_slab_depth_est_km": "Profundidad estimada de la placa de Nazca (km)",
    "event_depth_km": "Profundidad del foco sísmico (km)",
    "station_dist_lat_15s_km": "Posición del distrito respecto a la Dorsal de Nazca (15°S)",
    "log10_r_hyp": "Atenuación geométrica por distancia focal 3D",
    "event_dist_trench_km": "Distancia del sismo a la fosa marina (km)",
    "dist_lat_15s_km": "Posición norte–sur respecto a Ica/Nasca (15°S)",
    "event_dist_to_slab_km": "Distancia del sismo al contacto entre placas (km)",
    "n_peligros_caida": "Peligros de caída de rocas en el distrito (GEOCATMIN)",
    "n_peligros_flujo": "Peligros de huaicos/flujos en el distrito (GEOCATMIN)",
    "n_peligros_deslizamiento": "Peligros de deslizamientos en el distrito (GEOCATMIN)",
    "n_peligros_total": "Total de peligros geológicos en el distrito",
    "n_zonas_criticas_total": "Zonas críticas geológicas en el distrito",
    "azimuth_deg": "Dirección (azimut) desde el epicentro",
    "sin_azimuth": "Componente direccional Este–Oeste de propagación",
    "cos_azimuth": "Componente direccional Norte–Sur de propagación",
    # Model C (H3 Forecaster) features
    "dist_trench_km": "Cercanía a la fosa marina (km)",
    "nbhd_cnt_m45_5y": "Sismos M≥4.5 en la zona durante los últimos 5 años",
    "convergence_vel_mm_yr": "Velocidad de choque entre placas Nazca–Sudamérica (mm/año)",
    "days_since_last_m50_nbhd": "Días desde el último sismo fuerte (M≥5.0) cercano",
    "causal_expanding_cell_rate": "Frecuencia histórica acumulada de sismos en la zona",
    "cell_lon": "Ubicación Este–Oeste (Costa, Sierra o Selva)",
    "cell_lat": "Ubicación Norte–Sur en el territorio peruano",
    "nbhd_cnt_m45_365d": "Sismos M≥4.5 en la zona durante el último año",
    "days_since_last_m60_nbhd": "Días desde el último sismo muy fuerte (M≥6.0) cercano",
    "slab_depth_est_km": "Profundidad de la placa de Nazca bajo la zona (km)",
    "etas_decay_nbhd_365d": "Efecto de réplicas y enjambres recientes en la zona",
    "etas_decay_cell_365d": "Efecto de réplicas locales en la celda",
    "seafloor_age_myr": "Antigüedad de la placa oceánica de Nazca (millones de años)",
    "n_geocatmin_peligros": "Peligros geológicos registrados en la zona (GEOCATMIN)",
    "n_geocatmin_zonas_criticas": "Zonas críticas geológicas activas en la zona",
    "log10_energy_5y": "Energía sísmica liberada en los últimos 5 años",
    "log10_moment_5y": "Momento sísmico liberado en los últimos 5 años",
    "cnt_m45_365d": "Sismos locales M≥4.5 en el último año",
    "cnt_m45_5y": "Sismos locales M≥4.5 en los últimos 5 años",
    "cnt_mc_5y": "Sismos completos (M≥4.7) en los últimos 5 años",
}

REGIME_LABELS_ES: dict[str, str] = {
    "subduccion_interplaca": "Subducción Costera (choque entre placas)",
    "cortical_continental": "Falla Cortical Andina (superficial en tierra)",
    "intramanto_intermedio": "Losa Intermedia (interior de la placa, 60–350 km)",
    "intramanto_profundo": "Foco Profundo Amazónico (>350 km)",
    "indeterminado_sin_profundidad": "Histórico sin profundidad estimada",
}

REGIME_SHORT_ES: dict[str, str] = {
    "subduccion_interplaca": "Subducción Costera",
    "cortical_continental": "Falla Cortical Andina",
    "intramanto_intermedio": "Losa Intermedia",
    "intramanto_profundo": "Foco Profundo",
    "indeterminado_sin_profundidad": "Sin profundidad",
}

MACRO_BLOCK_LABELS_ES: dict[str, str] = {
    "BLOQUE_LAGUNA_CENTRAL_LIMA_CHINCHA": "Laguna Central: Lima – Callao – Chincha",
    "SUBBLOQUE_SUR_ILO_TACNA": "Extremo Sur: Mollendo – Ilo – Tacna",
    "BLOQUE_NORTE": "Costa Norte: Tumbes – Piura – Chimbote",
    "BLOQUE_CENTRO_NORTE": "Costa Centro-Norte: Áncash – Huacho",
    "BLOQUE_SUR_NASCA_TACNA": "Bloque Sur Amplio: Nasca – Arequipa – Tacna",
}

RISK_LEVEL_COLORS: dict[str, str] = {
    "Muy Alto": "#991B1B",
    "Alto": "#C2410C",
    "Medio": "#B45309",
    "Bajo": "#0F766E",
    "Muy Bajo": "#475569",
}

MERCALLI_DESC_ES: dict[int, tuple[str, str, str, str]] = {
    1: ("I", "Imperceptible", "No sentido por la población; solo registrado por sismógrafos.", "#64748B"),
    2: ("II", "Muy débil", "Sentido apenas por personas en reposo en pisos altos.", "#475569"),
    3: ("III", "Débil", "Vibración ligera similar al paso de un camión; sin daños.", "#0284C7"),
    4: ("IV", "Moderado leve", "Vibración leve; ventanas y vajilla vibran, sin daños.", "#0D9488"),
    5: ("V", "Fuerte leve", "Sentido por casi todos; objetos pequeños se desplazan o caen.", "#CA8A04"),
    6: ("VI", "Fuerte", "Susto general; caída de objetos, grietas finas en tarrajeo y muros débiles.", "#D97706"),
    7: ("VII", "Muy fuerte", "Difícil mantenerse de pie; daños considerables en adobe y muros no reforzados.", "#EA580C"),
    8: ("VIII", "Severo", "Daño severo en muros débiles/adobe, caída de cornisas y daños moderados en concreto.", "#DC2626"),
    9: ("IX", "Violento", "Destrucción grave en construcciones no sismorresistentes y daños fuertes en infraestructura.", "#991B1B"),
    10: ("X", "Extremo", "Destrucción masiva de estructuras vulnerables, licuefacción y deslizamientos severos.", "#7F1D1D"),
}

BADGE_AUTO_COLORS: dict[str, str] = {
    **RISK_LEVEL_COLORS,
    "Imperceptible": "#64748B",
    "Muy débil": "#475569",
    "Débil": "#0284C7",
    "Moderado leve": "#0D9488",
    "Fuerte leve": "#CA8A04",
    "Fuerte": "#D97706",
    "Muy fuerte": "#EA580C",
    "Severo": "#DC2626",
    "Violento": "#991B1B",
    "Extremo": "#7F1D1D",
    "Subestimado": "#991B1B",
    "Alineado": "#0F766E",
}


def translate_feature(name: str) -> str:
    """Translate any internal technical feature name into clear citizen Spanish."""
    return FEATURE_LABELS_ES.get(name, name.replace("_", " ").capitalize())


def describe_mercalli(mmi_value: float) -> tuple[str, str, str, str]:
    """Return (roman_numeral, short_name, everyday_impact_desc, hex_color) for a numeric MMI."""
    grade = int(min(max(round(float(mmi_value)), 1), 10))
    return MERCALLI_DESC_ES[grade]


def inject_custom_css() -> str:
    """Return institutional warm telluric CSS including sidebar nav and 100%-centered tables."""
    return """
    <style>
      .stApp {
        background-color: #FAF8F5;
        color: #1C1917;
      }
      section[data-testid="stSidebar"] {
        background-color: #F1ECE4;
        border-right: 1px solid #DFD5C7;
      }
      section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.28rem;
      }
      section[data-testid="stSidebar"] .stRadio label {
        background-color: #FAF8F5;
        border: 1px solid #E5DDD0;
        border-radius: 8px;
        padding: 0.48rem 0.7rem;
        margin-bottom: 0.12rem;
        font-weight: 600;
        font-size: 0.88rem;
        color: #292524;
        transition: all 0.15s ease-in-out;
        width: 100%;
      }
      section[data-testid="stSidebar"] .stRadio label:hover {
        border-color: #B91C1C;
        background-color: #FFF7F5;
      }
      .sidebar-brand {
        background: linear-gradient(135deg, #991B1B 0%, #B91C1C 100%);
        color: #FFFFFF;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 6px rgba(153, 27, 27, 0.20);
      }
      .sidebar-brand h2 {
        margin: 0;
        font-size: 1.05rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.25;
      }
      .sidebar-brand p {
        margin: 0.25rem 0 0 0;
        font-size: 0.76rem;
        color: #FEE2E2;
        line-height: 1.3;
      }
      .sidebar-data-card {
        background: #FFFFFF;
        border: 1px solid #E2D9CC;
        border-left: 4px solid #0F766E;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        font-size: 0.78rem;
        color: #44403C;
        line-height: 1.45;
        margin-top: 0.65rem;
      }
      .sidebar-data-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.84rem;
        font-weight: 800;
        color: #1C1917;
      }
      .sidebar-data-card ul {
        margin: 0;
        padding-left: 1.05rem;
      }
      .sidebar-data-card li {
        margin-bottom: 0.25rem;
      }
      .civic-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #F5EFE6 100%);
        border: 1px solid #E5DDD0;
        border-left: 6px solid #B91C1C;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 8px rgba(28, 25, 23, 0.05);
      }
      .civic-hero h1 {
        font-size: 1.65rem;
        margin: 0 0 0.3rem 0;
        color: #1C1917;
        font-weight: 800;
        letter-spacing: -0.02em;
      }
      .civic-hero p {
        margin: 0;
        color: #44403C;
        font-size: 0.95rem;
        line-height: 1.48;
      }
      .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E7E0D6;
        border-top: 4px solid var(--accent, #B91C1C);
        border-radius: 9px;
        padding: 0.82rem 0.95rem;
        min-height: 102px;
        box-shadow: 0 1px 4px rgba(28, 25, 23, 0.04);
      }
      .kpi-title {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #57534E;
        margin-bottom: 0.24rem;
      }
      .kpi-value {
        font-size: 1.28rem;
        font-weight: 800;
        color: #1C1917;
        line-height: 1.2;
        word-wrap: break-word;
      }
      .kpi-sub {
        font-size: 0.78rem;
        color: #78716C;
        margin-top: 0.24rem;
        line-height: 1.25;
      }
      .guide-card {
        background: #FFFFFF;
        border: 1px solid #E7E0D6;
        border-radius: 9px;
        padding: 0.92rem 1.02rem;
        height: 100%;
      }
      .guide-card h4 {
        margin: 0 0 0.38rem 0;
        font-size: 0.94rem;
        color: #991B1B;
      }
      .guide-card p {
        margin: 0;
        font-size: 0.85rem;
        color: #44403C;
        line-height: 1.45;
      }
      .district-spotlight {
        background: #FFFFFF;
        border: 1px solid #E5DDD0;
        border-left: 6px solid var(--accent, #B91C1C);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.6rem 0 1rem 0;
        box-shadow: 0 2px 6px rgba(28, 25, 23, 0.05);
      }
      .civic-badge {
        display: inline-block;
        padding: 0.18rem 0.62rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #FFFFFF;
        vertical-align: middle;
        white-space: nowrap;
      }
      /* 100% Centered Responsive Civic Tables */
      .civic-table-wrap {
        width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        border: 1px solid #E2D9CC;
        border-radius: 10px;
        background: #FFFFFF;
        box-shadow: 0 1px 5px rgba(28, 25, 23, 0.05);
        margin: 0.45rem 0 0.85rem 0;
      }
      table.civic-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        color: #1C1917;
      }
      table.civic-table thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #F3EDE4;
        color: #292524;
        font-weight: 800;
        font-size: 0.80rem;
        text-transform: none;
        letter-spacing: 0.01em;
        padding: 0.68rem 0.72rem;
        border-bottom: 2px solid #D6CBB9;
        text-align: center !important;
        vertical-align: middle !important;
        white-space: normal;
        line-height: 1.25;
      }
      table.civic-table tbody td {
        padding: 0.56rem 0.70rem;
        border-bottom: 1px solid #EFEAE2;
        text-align: center !important;
        vertical-align: middle !important;
        white-space: normal;
        line-height: 1.32;
      }
      table.civic-table tbody tr:nth-child(even) {
        background-color: #FCFBF9;
      }
      table.civic-table tbody tr:hover {
        background-color: #FEF3EC;
        transition: background-color 0.12s ease;
      }
      .civic-table-caption {
        font-size: 0.78rem;
        color: #78716C;
        margin-top: -0.4rem;
        margin-bottom: 0.75rem;
        text-align: center;
      }
      .school-callout {
        background: #FFFFFF;
        border: 1px solid #E5DDD0;
        border-left: 5px solid #0F766E;
        border-radius: 9px;
        padding: 0.95rem 1.15rem;
        margin: 0.6rem 0;
      }
    </style>
    """


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    accent: str = "#B91C1C",
    tooltip: str = "",
) -> str:
    """Render a non-truncating HTML KPI card with optional hover tooltip."""
    tip_icon = f' <span title="{escape(tooltip)}" style="cursor:help;color:#991B1B;">ⓘ</span>' if tooltip else ""
    sub_html = f'<div class="kpi-sub">{escape(subtitle)}</div>' if subtitle else ""
    return (
        f'<div class="kpi-card" style="--accent:{accent};">'
        f'<div class="kpi-title">{escape(title)}{tip_icon}</div>'
        f'<div class="kpi-value">{escape(value)}</div>'
        f'{sub_html}'
        f'</div>'
    )


def render_badge(text: str, color: str) -> str:
    """Render a clean semantic pill badge."""
    return f'<span class="civic-badge" style="background:{color};">{escape(text)}</span>'


def _badge_color_for_value(val: str, custom_map: dict[str, str] | None = None) -> str | None:
    if custom_map and val in custom_map:
        return custom_map[val]
    if val in BADGE_AUTO_COLORS:
        return BADGE_AUTO_COLORS[val]
    # Support Mercalli Roman + description badges like "VIII — Severo" or "VII (7.4)"
    for key, col in BADGE_AUTO_COLORS.items():
        if key in val and len(key) >= 4:
            return col
    return None


def render_centered_table(
    df: pd.DataFrame,
    badge_cols: set[str] | list[str] | tuple[str, ...] | None = None,
    badge_color_map: dict[str, str] | None = None,
    highlight_col: str | None = None,
    highlight_value: Any = None,
    max_height_px: int = 440,
    caption: str = "",
    empty_message: str = "No hay registros disponibles para mostrar con los filtros actuales.",
) -> str:
    """Render a 100%-centered responsive HTML/CSS table with zero column truncation.

    Every `<th>` and `<td>` enforces `text-align: center !important`. Columns listed in
    `badge_cols` automatically render semantic pill badges.
    """
    if df is None or df.empty:
        return f'<div class="school-callout" style="text-align:center;">{escape(empty_message)}</div>'

    badge_set = set(badge_cols or ())
    headers_html = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)

    rows_html: list[str] = []
    for _, row in df.iterrows():
        is_hl = highlight_col is not None and highlight_col in df.columns and row[highlight_col] == highlight_value
        tr_style = ' style="background-color:#FEF2F2;font-weight:700;"' if is_hl else ""
        cells: list[str] = []
        for col in df.columns:
            raw_val = row[col]
            if pd.isna(raw_val):
                cell_html = "—"
            else:
                txt = str(raw_val)
                if col in badge_set:
                    bcol = _badge_color_for_value(txt, badge_color_map) or "#475569"
                    cell_html = render_badge(txt, bcol)
                else:
                    cell_html = escape(txt)
            cells.append(f"<td>{cell_html}</td>")
        rows_html.append(f"<tr{tr_style}>{''.join(cells)}</tr>")

    max_h_style = f"max-height:{int(max_height_px)}px;" if max_height_px and max_height_px > 0 else ""
    cap_html = f'<div class="civic-table-caption">{escape(caption)}</div>' if caption else ""
    return (
        f'<div class="civic-table-wrap" style="{max_h_style}">'
        f'<table class="civic-table">'
        f'<thead><tr>{headers_html}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
        f'</div>'
        f'{cap_html}'
    )
