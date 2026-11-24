"""CivicTech / GovTech Editorial Visual Theme, Dual Light/Dark Mode & Citizen Glossaries (Phase 5D).

Provides:
  - Dual Light (`Claro Editorial`) / Dark (`Oscuro Nocturno`) high-contrast CSS via CSS variables.
  - `plotly_theme_kwargs(theme_mode)`: matching Plotly template/colors for light and dark modes.
  - `MERCALLI_DESC_ES` & `MERCALLI_TO_MAG_RANGE_ES`: dual Mercalli + equivalent epicentral Mw scale.
  - `render_centered_table(df, ...)`: 100%-centered responsive HTML table (`text-align: center !important`).
  - `render_table_legend(items, ...)`: mandatory plain-Spanish column legend card placed under every table.
  - `render_map_legend(items, ...)`: visual swatch legend panel placed under every map.
  - `render_kpi_card(...)` & `render_badge(...)`: non-truncating cards and semantic color pills.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd

THEME_LIGHT = "Claro Editorial"
THEME_DARK = "Oscuro Nocturno"

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
    "n_peligros_caida": "Peligros de derrumbes/caída de rocas en el distrito (GEOCATMIN)",
    "n_peligros_flujo": "Peligros de huaicos/flujos de lodo en el distrito (GEOCATMIN)",
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
    "Muy Bajo": "#334155",
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

MERCALLI_TO_MAG_RANGE_ES: dict[int, str] = {
    1: "M < 2.5",
    2: "M 2.5 – 3.0",
    3: "M 3.0 – 3.4",
    4: "M 3.5 – 4.2",
    5: "M 4.3 – 4.8",
    6: "M 4.9 – 5.4",
    7: "M 5.5 – 6.1",
    8: "M 6.2 – 6.9",
    9: "M 7.0 – 7.9",
    10: "M ≥ 8.0",
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


def describe_mercalli_dual(mmi_value: float) -> tuple[str, str, str, str, str]:
    """Return (roman, short_name, everyday_desc, hex_color, equiv_epicentral_mw_range)."""
    grade = int(min(max(round(float(mmi_value)), 1), 10))
    rom, short, desc, col = MERCALLI_DESC_ES[grade]
    return rom, short, desc, col, MERCALLI_TO_MAG_RANGE_ES[grade]


def format_mercalli_with_mag(mmi_value: float) -> str:
    """Format Mercalli Roman grade together with its approximate epicentral Mw range."""
    rom, _, _, _, mag_rng = describe_mercalli_dual(mmi_value)
    return f"Mercalli {rom} (eq. {mag_rng})"


def plotly_theme_kwargs(theme_mode: str = THEME_LIGHT) -> dict[str, Any]:
    """Return Plotly layout kwargs matching the active Light or Dark visual theme."""
    if theme_mode == THEME_DARK:
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "#111827",
            "plot_bgcolor": "#111827",
            "font": {"color": "#F1F5F9"},
        }
    return {
        "template": "plotly_white",
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": {"color": "#1C1917"},
    }


def folium_tiles_for_theme(theme_mode: str = THEME_LIGHT) -> str:
    """Return free, API-keyless Folium basemap tiles reliable in both Light and Dark modes."""
    _ = theme_mode
    return "OpenStreetMap"


def inject_custom_css(theme_mode: str = THEME_LIGHT) -> str:
    """Return institutional CSS supporting both Claro Editorial and Oscuro Nocturno modes."""
    is_dark = theme_mode == THEME_DARK

    if is_dark:
        v = {
            "bg_app": "#0B1120",
            "bg_header": "#0B1120",
            "bg_sidebar": "#111827",
            "bg_surface": "#1E293B",
            "bg_surface_alt": "#162032",
            "hero_grad": "linear-gradient(135deg, #1E293B 0%, #172033 100%)",
            "text_primary": "#F8FAFC",
            "text_secondary": "#CBD5E1",
            "text_muted": "#94A3B8",
            "border": "#334155",
            "border_strong": "#475569",
            "table_head_bg": "#0F172A",
            "table_head_txt": "#F8FAFC",
            "table_stripe": "#172033",
            "table_hover": "#283548",
            "table_hl": "#3B1D24",
            "nav_bg": "#1E293B",
            "nav_hover": "#283548",
            "nav_txt": "#F1F5F9",
            "legend_bg": "#162032",
            "accent_title": "#FCA5A5",
            "input_bg": "#1E293B",
        }
    else:
        v = {
            "bg_app": "#FAF8F5",
            "bg_header": "#FAF8F5",
            "bg_sidebar": "#F1ECE4",
            "bg_surface": "#FFFFFF",
            "bg_surface_alt": "#F7F3EC",
            "hero_grad": "linear-gradient(135deg, #FFFFFF 0%, #F5EFE6 100%)",
            "text_primary": "#1C1917",
            "text_secondary": "#44403C",
            "text_muted": "#78716C",
            "border": "#E2D9CC",
            "border_strong": "#D6CBB9",
            "table_head_bg": "#F3EDE4",
            "table_head_txt": "#292524",
            "table_stripe": "#FCFBF9",
            "table_hover": "#FEF3EC",
            "table_hl": "#FEF2F2",
            "nav_bg": "#FAF8F5",
            "nav_hover": "#FFF7F5",
            "nav_txt": "#292524",
            "legend_bg": "#F8F5EF",
            "accent_title": "#991B1B",
            "input_bg": "#FFFFFF",
        }

    dark_extra = ""
    if is_dark:
        dark_extra = """
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
      .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] p {
        color: #F1F5F9 !important;
      }
      div[data-baseweb="select"] > div,
      div[data-baseweb="base-input"] > input {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-color: #475569 !important;
      }
      div[data-testid="stExpander"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 9px;
      }
      button[kind="secondary"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-color: #475569 !important;
      }
      button[kind="secondary"]:hover {
        border-color: #F87171 !important;
        background-color: #283548 !important;
      }
        """

    return f"""
    <style>
      :root {{
        --bg-app: {v['bg_app']};
        --bg-sidebar: {v['bg_sidebar']};
        --bg-surface: {v['bg_surface']};
        --bg-surface-alt: {v['bg_surface_alt']};
        --text-primary: {v['text_primary']};
        --text-secondary: {v['text_secondary']};
        --text-muted: {v['text_muted']};
        --border-color: {v['border']};
        --border-strong: {v['border_strong']};
        --table-head-bg: {v['table_head_bg']};
        --table-head-txt: {v['table_head_txt']};
        --table-stripe: {v['table_stripe']};
        --table-hover: {v['table_hover']};
        --table-hl: {v['table_hl']};
        --legend-bg: {v['legend_bg']};
        --accent-title: {v['accent_title']};
      }}
      .stApp {{
        background-color: var(--bg-app);
        color: var(--text-primary);
      }}
      header[data-testid="stHeader"] {{
        background-color: {v['bg_header']};
      }}
      section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border-color);
      }}
      section[data-testid="stSidebar"] .stRadio > div {{
        gap: 0.24rem;
      }}
      section[data-testid="stSidebar"] .stRadio label {{
        background-color: {v['nav_bg']};
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.44rem 0.68rem;
        margin-bottom: 0.08rem;
        font-weight: 600;
        font-size: 0.86rem;
        color: {v['nav_txt']} !important;
        transition: all 0.15s ease-in-out;
        width: 100%;
      }}
      section[data-testid="stSidebar"] .stRadio label:hover {{
        border-color: #B91C1C;
        background-color: {v['nav_hover']};
      }}
      .sidebar-brand {{
        background: linear-gradient(135deg, #991B1B 0%, #B91C1C 100%);
        color: #FFFFFF;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 2px 6px rgba(153, 27, 27, 0.24);
      }}
      .sidebar-brand h2 {{
        margin: 0;
        font-size: 1.03rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.25;
      }}
      .sidebar-brand p {{
        margin: 0.22rem 0 0 0;
        font-size: 0.75rem;
        color: #FEE2E2 !important;
        line-height: 1.3;
      }}
      .sidebar-data-card {{
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-left: 4px solid #0F766E;
        border-radius: 8px;
        padding: 0.72rem 0.82rem;
        font-size: 0.77rem;
        color: var(--text-secondary);
        line-height: 1.44;
        margin-top: 0.6rem;
      }}
      .sidebar-data-card h4 {{
        margin: 0 0 0.38rem 0;
        font-size: 0.83rem;
        font-weight: 800;
        color: var(--text-primary) !important;
      }}
      .sidebar-data-card ul {{
        margin: 0;
        padding-left: 1.05rem;
      }}
      .sidebar-data-card li {{
        margin-bottom: 0.22rem;
        color: var(--text-secondary) !important;
      }}
      .civic-hero {{
        background: {v['hero_grad']};
        border: 1px solid var(--border-color);
        border-left: 6px solid #B91C1C;
        border-radius: 10px;
        padding: 1.08rem 1.35rem;
        margin-bottom: 0.88rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
      }}
      .civic-hero h1 {{
        font-size: 1.62rem;
        margin: 0 0 0.28rem 0;
        color: var(--text-primary) !important;
        font-weight: 800;
        letter-spacing: -0.02em;
      }}
      .civic-hero p {{
        margin: 0;
        color: var(--text-secondary) !important;
        font-size: 0.94rem;
        line-height: 1.48;
      }}
      .kpi-card {{
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-top: 4px solid var(--accent, #B91C1C);
        border-radius: 9px;
        padding: 0.80rem 0.92rem;
        min-height: 106px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
      }}
      .kpi-title {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: var(--text-muted);
        margin-bottom: 0.24rem;
      }}
      .kpi-value {{
        font-size: 1.24rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.22;
        word-wrap: break-word;
      }}
      .kpi-sub {{
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.24rem;
        line-height: 1.28;
      }}
      .guide-card {{
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-top: 4px solid var(--accent, #B91C1C);
        border-radius: 9px;
        padding: 0.92rem 1.02rem;
        height: 100%;
        margin-bottom: 0.6rem;
      }}
      .guide-card h4 {{
        margin: 0 0 0.36rem 0;
        font-size: 0.95rem;
        color: var(--accent-title) !important;
      }}
      .guide-card p {{
        margin: 0;
        font-size: 0.85rem;
        color: var(--text-secondary) !important;
        line-height: 1.46;
      }}
      .district-spotlight {{
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-left: 6px solid var(--accent, #B91C1C);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.55rem 0 0.95rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        color: var(--text-primary);
      }}
      .civic-badge {{
        display: inline-block;
        padding: 0.18rem 0.62rem;
        border-radius: 999px;
        font-size: 0.77rem;
        font-weight: 700;
        color: #FFFFFF !important;
        vertical-align: middle;
        white-space: nowrap;
      }}
      /* 100% Centered Responsive Civic Tables */
      .civic-table-wrap {{
        width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        background: var(--bg-surface);
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.06);
        margin: 0.42rem 0 0.35rem 0;
      }}
      table.civic-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
        color: var(--text-primary);
      }}
      table.civic-table thead th {{
        position: sticky;
        top: 0;
        z-index: 2;
        background: var(--table-head-bg);
        color: var(--table-head-txt);
        font-weight: 800;
        font-size: 0.79rem;
        letter-spacing: 0.01em;
        padding: 0.66rem 0.68rem;
        border-bottom: 2px solid var(--border-strong);
        text-align: center !important;
        vertical-align: middle !important;
        white-space: normal;
        line-height: 1.25;
      }}
      table.civic-table tbody td {{
        padding: 0.54rem 0.66rem;
        border-bottom: 1px solid var(--border-color);
        text-align: center !important;
        vertical-align: middle !important;
        white-space: normal;
        line-height: 1.32;
        color: var(--text-primary);
      }}
      table.civic-table tbody tr:nth-child(even) {{
        background-color: var(--table-stripe);
      }}
      table.civic-table tbody tr:hover {{
        background-color: var(--table-hover);
        transition: background-color 0.12s ease;
      }}
      table.civic-table tbody tr.civic-row-highlight {{
        background-color: var(--table-hl) !important;
        font-weight: 700;
      }}
      .civic-table-caption {{
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.15rem;
        margin-bottom: 0.45rem;
        text-align: center;
      }}
      .table-legend-box {{
        background: var(--legend-bg);
        border: 1px solid var(--border-color);
        border-left: 4px solid #0F766E;
        border-radius: 8px;
        padding: 0.72rem 0.95rem;
        margin: 0.25rem 0 1.05rem 0;
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.48;
      }}
      .table-legend-title {{
        font-weight: 800;
        font-size: 0.83rem;
        color: var(--text-primary);
        margin-bottom: 0.35rem;
      }}
      .table-legend-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 0.35rem 1.0rem;
      }}
      .table-legend-item strong {{
        color: var(--text-primary);
      }}
      .map-legend-box {{
        background: var(--legend-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.62rem 0.9rem;
        margin: 0.35rem 0 0.85rem 0;
        font-size: 0.81rem;
        color: var(--text-secondary);
      }}
      .map-legend-items {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem 1.15rem;
        align-items: center;
      }}
      .map-legend-swatch {{
        display: inline-block;
        width: 11px;
        height: 11px;
        border-radius: 50%;
        margin-right: 0.38rem;
        vertical-align: middle;
        border: 1px solid rgba(0, 0, 0, 0.35);
      }}
      .map-legend-note {{
        margin-top: 0.38rem;
        font-size: 0.78rem;
        color: var(--text-muted);
      }}
      .school-callout {{
        background: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-left: 5px solid #0F766E;
        border-radius: 9px;
        padding: 0.92rem 1.12rem;
        margin: 0.55rem 0;
        color: var(--text-secondary);
        font-size: 0.90rem;
        line-height: 1.5;
      }}
      {dark_extra}
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
    tip_icon = f' <span title="{escape(tooltip)}" style="cursor:help;color:#DC2626;">ⓘ</span>' if tooltip else ""
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
    """Render a 100%-centered responsive HTML/CSS table with zero column truncation."""
    if df is None or df.empty:
        return f'<div class="school-callout" style="text-align:center;">{escape(empty_message)}</div>'

    badge_set = set(badge_cols or ())
    headers_html = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)

    rows_html: list[str] = []
    for _, row in df.iterrows():
        is_hl = highlight_col is not None and highlight_col in df.columns and row[highlight_col] == highlight_value
        tr_cls = ' class="civic-row-highlight"' if is_hl else ""
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
        rows_html.append(f"<tr{tr_cls}>{''.join(cells)}</tr>")

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


def render_table_legend(
    items: list[tuple[str, str]],
    title: str = "📖 ¿Cómo leer esta tabla? (Explicación en español sencillo)",
) -> str:
    """Render a clean explanatory legend card directly beneath every data table."""
    items_html = "".join(
        f'<div class="table-legend-item"><strong>{escape(k)}:</strong> {escape(v)}</div>'
        for k, v in items
    )
    return (
        f'<div class="table-legend-box">'
        f'<div class="table-legend-title">{escape(title)}</div>'
        f'<div class="table-legend-grid">{items_html}</div>'
        f'</div>'
    )


def render_map_legend(
    items: list[tuple[str, str, str]],
    note: str = "💡 Usa el botón de pantalla completa (esquina superior izquierda del mapa) para explorar con mayor detalle.",
) -> str:
    """Render a visual swatch legend bar directly beneath a Folium or Plotly map.

    Each item is `(hex_color, bold_label, description)`.
    """
    chips = "".join(
        f'<span><span class="map-legend-swatch" style="background:{escape(col)};"></span>'
        f'<strong>{escape(lbl)}:</strong> {escape(desc)}</span>'
        for col, lbl, desc in items
    )
    note_html = f'<div class="map-legend-note">{escape(note)}</div>' if note else ""
    return (
        f'<div class="map-legend-box">'
        f'<div class="map-legend-items">{chips}</div>'
        f'{note_html}'
        f'</div>'
    )
