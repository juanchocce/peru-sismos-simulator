"""Phase 2 - Step 4: Coastal Seismic Gaps & Seismic Moment Deficit Quantification.

Divides the Peruvian subduction margin (-3.5S to -18.5S) into 30 contiguous 0.5-deg
latitudinal segments projected onto the continuous PB2002 Nazca trench (NZ\\SA + NZ\\AP
+ NZ\\ND), computes theoretical seismic moment accumulation rate
    M0_dot = chi * mu * A_seg * v_conv   (mu = 3.0e10 Pa, W = 100 km, chi in {0.50, 0.65, 0.80})
combines historical (1552-1959) and instrumental (1960-2026) interplate seismicity
(h <= 60 km, dist_trench_km <= 180 km) to determine years since the last megathrust
rupture (Mw >= 7.8), accumulated released moment, net moment deficit Delta_M0, and
equivalent potential magnitude Mw_pot.

Exports `data/processed/coastal_seismic_gaps.parquet` and `reports/figures/seismic_gaps_map.html`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyproj import Geod

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RAW_TECTONIC_ROOT = PROJECT_ROOT / "data" / "raw" / "tectonic_plates"

INST_PARQUET = PROCESSED_DIR / "catalog_igp_instrumental.parquet"
HIST_PARQUET = PROCESSED_DIR / "catalog_igp_historico.parquet"
OUT_GAPS_PARQUET = PROCESSED_DIR / "coastal_seismic_gaps.parquet"
OUT_MAP_HTML = FIGURES_DIR / "seismic_gaps_map.html"
METRICS_JSON = INTERIM_DIR / "seismic_gaps_metrics.json"

MU_PA = 3.0e10  # Rigidity (30 GPa)
SEISMOGENIC_WIDTH_KM = 100.0
CHI_VALUES = (0.50, 0.65, 0.80)
REFERENCE_DATE_UTC = pd.Timestamp("2026-11-23T00:00:00+00:00")
CATALOG_HIST_START_YEAR = 1552.5
MAJOR_RUPTURE_MW = 7.8
NAZCA_TRENCH_BOUNDS = {"NZ\\SA", "NZ\\AP", "NZ\\ND"}

# Coastal reference place names for readability
COASTAL_LABELS = [
    (-3.75, "Tumbes / Talara Norte"),
    (-4.25, "Talara / Mancora"),
    (-4.75, "Paita / Piura Norte"),
    (-5.25, "Sechura / Bayovar"),
    (-5.75, "Bayovar / Lambayeque N"),
    (-6.25, "Chiclayo / Morrope"),
    (-6.75, "Chiclayo / Eten"),
    (-7.25, "Pacasmayo / Chepen"),
    (-7.75, "Trujillo / Chicama"),
    (-8.25, "Salaverry / Viru"),
    (-8.75, "Chimbote / Santa"),
    (-9.25, "Casma / Ancash"),
    (-9.75, "Huarmey"),
    (-10.25, "Paramonga / Barranca"),
    (-10.75, "Supe / Huacho Norte"),
    (-11.25, "Huacho / Chancay"),
    (-11.75, "Lima / Callao Norte"),
    (-12.25, "Lima / Callao Sur - Lurin"),
    (-12.75, "Canete / Mala"),
    (-13.25, "Chincha / Pisco Norte"),
    (-13.75, "Pisco / Paracas"),
    (-14.25, "Ica / Bahía Independencia"),
    (-14.75, "Nasca / Marcona Norte"),
    (-15.25, "San Juan de Marcona"),
    (-15.75, "Lomas / Chala"),
    (-16.25, "Atico / Caraveli"),
    (-16.75, "Ocona / Camana"),
    (-17.25, "Mollendo / Islay"),
    (-17.75, "Ilo / Moquegua"),
    (-18.25, "Tacna / Arica Norte"),
]


def _resolve_geojson_dir() -> Path:
    for p in (RAW_TECTONIC_ROOT / "tectonicplates-master" / "GeoJSON", RAW_TECTONIC_ROOT / "GeoJSON"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"PB2002 GeoJSON not found under {RAW_TECTONIC_ROOT}")


def _extract_trench_samples(geojson_dir: Path) -> pd.DataFrame:
    """Densify PB2002_steps along Peru trench to ~1 km spacing to measure exact segment length & velocity."""
    geod = Geod(ellps="WGS84")
    gdf_steps = gpd.read_file(geojson_dir / "PB2002_steps.json")
    trench = gdf_steps[gdf_steps["PLATEBOUND"].isin(NAZCA_TRENCH_BOUNDS)].copy()

    rows: list[dict[str, float | str]] = []
    for _, row in trench.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i][0], coords[i][1]
            lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
            _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
            n_sub = max(int(np.ceil(dist_m / 1000.0)), 1)
            pts = [(lon1, lat1)] + (geod.npts(lon1, lat1, lon2, lat2, n_sub - 1) if n_sub > 1 else []) + [(lon2, lat2)]
            sub_len_km = (dist_m / 1000.0) / n_sub
            for j in range(len(pts) - 1):
                mid_lon = 0.5 * (pts[j][0] + pts[j + 1][0])
                mid_lat = 0.5 * (pts[j][1] + pts[j + 1][1])
                rows.append(
                    {
                        "lat": mid_lat,
                        "lon": mid_lon,
                        "sub_len_km": sub_len_km,
                        "convergence_vel_mm_yr": float(row["VELOCITYLE"]),
                        "seafloor_age_myr": float(row["SEAFLOORAG"]),
                        "platebound": str(row["PLATEBOUND"]),
                    }
                )
    return pd.DataFrame(rows)


def _mw_from_m0(m0_nm: float) -> float:
    """Hanks & Kanamori (1979) moment magnitude from scalar seismic moment (N*m)."""
    return float((np.log10(max(m0_nm, 1.0)) - 9.1) / 1.5)


def _generate_gaps_map(df_gaps: pd.DataFrame, df_major: pd.DataFrame) -> None:
    """Generate interactive Plotly geographic map of coastal moment deficit & Mw_pot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()

    # Trench segments colored by Mw_potential (chi=0.65)
    hover_text = [
        (
            f"<b>{r['segment_id']} ({r['coastal_zone']})</b><br>"
            f"Lat: [{r['lat_min']:.1f}, {r['lat_max']:.1f}] | L_seg: {r['segment_length_km']:.1f} km<br>"
            f"Ultima ruptura M>=7.8: {int(r['last_major_rupture_year']) if pd.notna(r['last_major_rupture_year']) else 'Sin registro >=1552'} "
            f"(Mw {r['last_major_rupture_mw']:.1f})<br>"
            f"Años transcurridos: {r['years_since_major_rupture']:.1f} yr<br>"
            f"Deficit M0 (chi=0.65): {r['m0_deficit_chi065_nm']:.2e} N·m<br>"
            f"<b>Mw potencial (chi=0.65): {r['mw_pot_chi065']:.2f}</b> "
            f"[chi=0.50: {r['mw_pot_chi050']:.2f} | chi=0.80: {r['mw_pot_chi080']:.2f}]"
        )
        for _, r in df_gaps.iterrows()
    ]

    fig.add_trace(
        go.Scattergeo(
            lon=df_gaps["trench_lon_mean"],
            lat=df_gaps["lat_center"],
            mode="markers+lines",
            name="Segmentos costeros (0.5 deg)",
            text=hover_text,
            hoverinfo="text",
            line=dict(color="rgba(80,80,80,0.45)", width=2),
            marker=dict(
                size=13,
                color=df_gaps["mw_pot_chi065"],
                colorscale="YlOrRd",
                cmin=7.0,
                cmax=8.9,
                colorbar=dict(title="Mw Potencial<br>(chi=0.65)"),
                line=dict(color="#222222", width=1),
            ),
        )
    )

    # Overlay historical + instrumental major megathrust epicenters (Mw >= 7.8)
    fig.add_trace(
        go.Scattergeo(
            lon=df_major["lon"],
            lat=df_major["lat"],
            mode="markers",
            name="Megaterremotos Mw >= 7.8 (1552-2026)",
            text=[
                f"<b>{r['timestamp_utc'].strftime('%Y-%m-%d')}</b> | Mw={r['mag_unified']:.2f} | Prof={r['depth_km']:.0f} km"
                for _, r in df_major.iterrows()
            ],
            hoverinfo="text",
            marker=dict(symbol="star", size=11, color="#1d3557", line=dict(color="white", width=0.8)),
        )
    )

    fig.update_geos(
        scope="south america",
        lataxis_range=[-20.0, -2.0],
        lonaxis_range=[-84.0, -68.0],
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="rgb(243, 243, 243)",
        oceancolor="rgb(225, 240, 250)",
        showocean=True,
    )
    fig.update_layout(
        title="Lagunas Sismicas Costeras del Peru: Deficit de Momento Sismico y Mw Potencial (chi=0.65)",
        height=760,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    fig.write_html(OUT_MAP_HTML, include_plotlyjs="cdn")


def main() -> None:
    """Compute coastal moment deficit across 30 segments of 0.5 deg latitude."""
    assert INST_PARQUET.exists() and HIST_PARQUET.exists()
    geojson_dir = _resolve_geojson_dir()
    trench_df = _extract_trench_samples(geojson_dir)

    df_hist = pd.read_parquet(HIST_PARQUET)
    df_inst = pd.read_parquet(INST_PARQUET)

    # Harmonize historical + instrumental interplate interface catalogs (h <= 60 km, dist_trench_km <= 180 km)
    h_sub = df_hist[
        (df_hist["depth_km"].fillna(30.0) <= 60.0) & (df_hist["dist_trench_km"] <= 180.0)
    ][["event_id", "timestamp_utc", "lat", "lon", "depth_km", "mag_homog", "seismic_moment_nm", "dist_trench_km"]].rename(
        columns={"mag_homog": "mag_unified"}
    )
    h_sub["catalog"] = "IGP_HIST"

    i_sub = df_inst[
        (df_inst["depth_km"] <= 60.0) & (df_inst["dist_trench_km"] <= 180.0)
    ][["event_id", "timestamp_utc", "lat", "lon", "depth_km", "mag", "seismic_moment_nm", "dist_trench_km"]].rename(
        columns={"mag": "mag_unified"}
    )
    i_sub["catalog"] = "IGP_INST"

    interplate = pd.concat([h_sub, i_sub], ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    interplate["year_decimal"] = (
        interplate["timestamp_utc"].dt.year
        + (interplate["timestamp_utc"].dt.dayofyear - 1) / 365.25
    )

    major_events = interplate[interplate["mag_unified"] >= MAJOR_RUPTURE_MW].copy()
    assert len(major_events) >= 10, f"Expected >=10 major interplate ruptures Mw>=7.8, found {len(major_events)}"

    ref_year = REFERENCE_DATE_UTC.year + (REFERENCE_DATE_UTC.dayofyear - 1) / 365.25
    label_map = dict(COASTAL_LABELS)

    lat_edges = np.round(np.arange(-3.5, -18.5 - 1e-9, -0.5), 2)  # -3.5, -4.0, ..., -18.5
    rows: list[dict[str, Any]] = []

    for idx in range(len(lat_edges) - 1):
        lat_max = float(lat_edges[idx])
        lat_min = float(lat_edges[idx + 1])
        lat_center = round(0.5 * (lat_min + lat_max), 2)
        seg_id = f"SEG_{idx + 1:02d}_{abs(lat_max):.1f}S_{abs(lat_min):.1f}S"

        # Trench geometry & convergence velocity inside [lat_min, lat_max)
        t_seg = trench_df[(trench_df["lat"] >= lat_min) & (trench_df["lat"] < lat_max)]
        assert not t_seg.empty, f"No trench geometry in segment {seg_id}"
        l_seg_km = float(t_seg["sub_len_km"].sum())
        v_conv_mm_yr = float(np.average(t_seg["convergence_vel_mm_yr"], weights=t_seg["sub_len_km"]))
        seafloor_age = float(np.average(t_seg["seafloor_age_myr"], weights=t_seg["sub_len_km"]))
        trench_lon_mean = float(t_seg["lon"].mean())
        trench_bound = str(t_seg["platebound"].mode().iloc[0])

        # Area (m^2) and base full-coupling moment rate (chi=1.0)
        area_m2 = (l_seg_km * 1000.0) * (SEISMOGENIC_WIDTH_KM * 1000.0)
        v_conv_m_yr = v_conv_mm_yr / 1000.0
        m0_rate_chi1 = MU_PA * area_m2 * v_conv_m_yr

        # Interplate events in segment
        ev_seg = interplate[(interplate["lat"] >= lat_min) & (interplate["lat"] < lat_max)]
        maj_seg = ev_seg[ev_seg["mag_unified"] >= MAJOR_RUPTURE_MW]

        if not maj_seg.empty:
            last_maj = maj_seg.iloc[-1]
            last_ts = last_maj["timestamp_utc"]
            last_year = float(last_maj["year_decimal"])
            last_mw = float(last_maj["mag_unified"])
            has_rec_m78 = True
            # Moment released strictly AFTER the last major megathrust rupture
            ev_after = ev_seg[ev_seg["timestamp_utc"] > last_ts]
        else:
            # No Mw>=7.8 epicenter inside this 0.5 deg bin since 1552; check nearby rupture within +-0.75 deg
            # and also record strictly catalog-based since 1552
            last_ts = pd.NaT
            last_year = np.nan
            last_mw = np.nan
            has_rec_m78 = False
            ev_after = ev_seg

        # Years elapsed: from last major rupture in segment, or from 1552.5 catalog start if none recorded
        years_since = ref_year - (last_year if has_rec_m78 else CATALOG_HIST_START_YEAR)
        m0_released_nm = float(ev_after["seismic_moment_nm"].sum())
        n_events_after = int(len(ev_after))
        n_inst_m45 = int(len(ev_seg[(ev_seg["catalog"] == "IGP_INST") & (ev_seg["mag_unified"] >= 4.5)]))

        row_dict: dict[str, Any] = {
            "segment_id": seg_id,
            "coastal_zone": label_map.get(lat_center, f"Lat {lat_center:.2f}"),
            "lat_max": lat_max,
            "lat_min": lat_min,
            "lat_center": lat_center,
            "trench_lon_mean": round(trench_lon_mean, 4),
            "trench_bound": trench_bound,
            "regimen_subduccion_lat": "flat_slab_norte_centro" if lat_center > -15.0 else "normal_sur",
            "segment_length_km": round(l_seg_km, 2),
            "seismogenic_width_km": SEISMOGENIC_WIDTH_KM,
            "convergence_vel_mm_yr": round(v_conv_mm_yr, 2),
            "seafloor_age_myr": round(seafloor_age, 2),
            "has_recorded_m78_in_segment": has_rec_m78,
            "last_major_rupture_year": round(last_year, 1) if has_rec_m78 else np.nan,
            "last_major_rupture_mw": round(last_mw, 2) if has_rec_m78 else np.nan,
            "years_since_major_rupture": round(float(years_since), 2),
            "n_interplate_events_after_rupture": n_events_after,
            "n_instrumental_m45_events": n_inst_m45,
            "m0_released_nm": m0_released_nm,
        }

        for chi in CHI_VALUES:
            tag = f"chi{int(round(chi * 100)):03d}"
            m0_rate = chi * m0_rate_chi1
            m0_accum = m0_rate * years_since
            m0_def = max(m0_accum - m0_released_nm, 0.0)
            row_dict[f"m0_accum_rate_{tag}_nm_yr"] = m0_rate
            row_dict[f"m0_accum_{tag}_nm"] = m0_accum
            row_dict[f"m0_deficit_{tag}_nm"] = m0_def
            row_dict[f"mw_pot_{tag}"] = round(_mw_from_m0(m0_def), 3) if m0_def > 0 else 0.0

        rows.append(row_dict)

    df_gaps = pd.DataFrame(rows)
    for col in ("segment_id", "coastal_zone", "trench_bound", "regimen_subduccion_lat"):
        df_gaps[col] = df_gaps[col].astype("string")

    # Integrity assertions
    assert len(df_gaps) == 30, f"Expected 30 coastal segments, got {len(df_gaps)}"
    assert (df_gaps["segment_length_km"] >= 50.0).all()
    assert (df_gaps["convergence_vel_mm_yr"] > 40.0).all()
    assert (df_gaps["mw_pot_chi080"] >= df_gaps["mw_pot_chi065"]).all()
    assert (df_gaps["mw_pot_chi065"] >= df_gaps["mw_pot_chi050"]).all()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_gaps.to_parquet(OUT_GAPS_PARQUET, engine="pyarrow", compression="snappy", index=False)
    _generate_gaps_map(df_gaps, major_events)
    assert OUT_MAP_HTML.exists() and OUT_MAP_HTML.stat().st_size > 10000

    # Top 10 overall & Top 10 with documented historical Mw>=7.8 rupture
    top10_all = df_gaps.sort_values("m0_deficit_chi065_nm", ascending=False).head(10)
    top10_doc = df_gaps[df_gaps["has_recorded_m78_in_segment"]].sort_values("m0_deficit_chi065_nm", ascending=False).head(10)

    metrics = {
        "total_segments": int(len(df_gaps)),
        "segments_with_recorded_m78": int(df_gaps["has_recorded_m78_in_segment"].sum()),
        "segments_without_recorded_m78": int((~df_gaps["has_recorded_m78_in_segment"]).sum()),
        "top10_documented_rupture": top10_doc[
            ["segment_id", "coastal_zone", "lat_center", "last_major_rupture_year", "last_major_rupture_mw", "years_since_major_rupture", "m0_deficit_chi065_nm", "mw_pot_chi050", "mw_pot_chi065", "mw_pot_chi080"]
        ].to_dict("records"),
        "top10_all_segments": top10_all[
            ["segment_id", "coastal_zone", "lat_center", "has_recorded_m78_in_segment", "last_major_rupture_year", "years_since_major_rupture", "m0_deficit_chi065_nm", "mw_pot_chi065"]
        ].to_dict("records"),
        "lima_segment": df_gaps[df_gaps["lat_center"].isin([-11.75, -12.25])][
            ["segment_id", "coastal_zone", "last_major_rupture_year", "last_major_rupture_mw", "years_since_major_rupture", "m0_deficit_chi065_nm", "mw_pot_chi050", "mw_pot_chi065", "mw_pot_chi080"]
        ].to_dict("records"),
        "map_path": str(OUT_MAP_HTML),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== CUANTIFICACION DE LAGUNAS SISMICAS COMPLETADA ===")
    print(f"Segmentos evaluados: {len(df_gaps)} (con ruptura Mw>=7.8 registrada: {metrics['segments_with_recorded_m78']})")
    print("Top 10 segmentos con ruptura historica Mw>=7.8 documentada (ordenados por deficit chi=0.65):")
    print(
        top10_doc[["segment_id", "coastal_zone", "last_major_rupture_year", "years_since_major_rupture", "m0_deficit_chi065_nm", "mw_pot_chi065"]].to_string(index=False)
    )
    print(f"Mapa interactivo guardado en: {OUT_MAP_HTML}")


if __name__ == "__main__":
    main()
