"""Phase 3 - Model B: Probabilistic Seismic Gaps (Monte Carlo, BPT Renewal & HDBSCAN).

Quantifies epistemic geophysical uncertainty over Peru's 30 coastal subduction segments
and 4 tectonic macro-blocks using:
  1. N = 10,000 Monte Carlo simulations sampling coupling chi ~ U(0.50, 0.85),
     seismogenic width W ~ U(85, 115) km, rigidity mu ~ N(3.0e10, 0.2e10) Pa, and
     historical magnitude error sigma_M = 0.25 -> P10 / P50 / P90 of Delta_M0 and Mw_pot.
  2. Brownian Passage Time (BPT) conditional rupture probability for Mw >= 8.0 over
     30-year and 50-year horizons given elapsed time since the last major rupture,
     with sensitivity for aperiodicity alpha in {0.3, 0.5} and Tr in [150, 300] yr.
  3. HDBSCAN 3D spatial clustering (`sklearn.cluster.HDBSCAN` in UTM 18S + depth) on
     modern interplate seismicity (2000-2026, M >= Mc) to contrast creeping/active zones
     against locked seismic silence cores.

Exports:
  - `data/processed/seismic_gaps_probabilistic.parquet`
  - `reports/figures/seismic_gaps_probabilistic.html`
  - `data/interim/seismic_gaps_prob_metrics.json`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer
from scipy.stats import norm
from sklearn.cluster import HDBSCAN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

GAPS_PARQUET = PROCESSED_DIR / "coastal_seismic_gaps.parquet"
INST_PARQUET = PROCESSED_DIR / "catalog_igp_instrumental.parquet"
HIST_PARQUET = PROCESSED_DIR / "catalog_igp_historico.parquet"

OUT_PROB_PARQUET = PROCESSED_DIR / "seismic_gaps_probabilistic.parquet"
OUT_HTML = FIGURES_DIR / "seismic_gaps_probabilistic.html"
METRICS_JSON = INTERIM_DIR / "seismic_gaps_prob_metrics.json"

N_MC = 10_000
RANDOM_STATE = 42
MC_MODERN = 4.7
M0_MW80_NM = 10.0 ** (1.5 * 8.0 + 9.1)  # 1.2589e21 N*m

MACRO_BLOCKS = [
    ("BLOQUE_NORTE", "Norte: Tumbes - Chimbote (-3.5 a -9.0)", -3.5, -9.0),
    ("BLOQUE_CENTRO_NORTE", "Centro-Norte: Ancash - Huacho (-9.0 a -11.5)", -9.0, -11.5),
    ("BLOQUE_LAGUNA_CENTRAL_LIMA_CHINCHA", "Laguna Central: Lima - Chincha (-11.5 a -13.5)", -11.5, -13.5),
    ("BLOQUE_SUR_NASCA_TACNA", "Sur: Nasca - Arequipa - Ilo - Tacna (-14.5 a -18.5)", -14.5, -18.5),
    ("SUBBLOQUE_SUR_ILO_TACNA", "Sub-bloque Sur Critico: Mollendo - Ilo - Tacna (-17.0 a -18.5)", -17.0, -18.5),
]


def bpt_cdf(t: float, t_r: float, alpha: float) -> float:
    """Brownian Passage Time (Inverse Gaussian) CDF (Matthews et al. 2002)."""
    if t <= 0.0:
        return 0.0
    u = t / t_r
    sqrt_u = np.sqrt(u)
    term1 = norm.cdf((u - 1.0) / (alpha * sqrt_u))
    # Numerically stable second term using log-sf/cdf for small alpha
    z2 = -(u + 1.0) / (alpha * sqrt_u)
    log_term2 = (2.0 / (alpha ** 2)) + norm.logcdf(z2)
    term2 = float(np.exp(min(log_term2, 0.0))) if np.isfinite(log_term2) else 0.0
    return float(np.clip(term1 + term2, 0.0, 1.0))


def bpt_conditional_prob(t_elapsed: float, delta_t: float, t_r: float, alpha: float) -> float:
    """Conditional probability P(t_elapsed < T <= t_elapsed + delta_t | T > t_elapsed)."""
    f0 = bpt_cdf(t_elapsed, t_r, alpha)
    f1 = bpt_cdf(t_elapsed + delta_t, t_r, alpha)
    surv = max(1.0 - f0, 1e-12)
    return float(np.clip((f1 - f0) / surv, 0.0, 1.0))


def _mw_from_m0_arr(m0: np.ndarray) -> np.ndarray:
    return (np.log10(np.maximum(m0, 1.0)) - 9.1) / 1.5


def _run_mc_for_entity(
    rng: np.random.Generator,
    length_km: float,
    v_conv_mm_yr: float,
    years_elapsed: float,
    m0_released_hist: float,
    m0_released_inst: float,
) -> dict[str, float]:
    """Run N=10,000 Monte Carlo draws for moment deficit and Mw_pot."""
    chi = rng.uniform(0.50, 0.85, size=N_MC)
    width_km = rng.uniform(85.0, 115.0, size=N_MC)
    mu = np.clip(rng.normal(3.0e10, 0.2e10, size=N_MC), 1.5e10, 4.5e10)

    area_m2 = (length_km * 1000.0) * (width_km * 1000.0)
    m0_rate = chi * mu * area_m2 * (v_conv_mm_yr / 1000.0)
    m0_accum = m0_rate * years_elapsed

    # Magnitude uncertainty: sigma_M = 0.25 on historical release, 0.12 on instrumental release
    dm_hist = rng.normal(0.0, 0.25, size=N_MC)
    dm_inst = rng.normal(0.0, 0.12, size=N_MC)
    m0_rel_mc = m0_released_hist * np.power(10.0, 1.5 * dm_hist) + m0_released_inst * np.power(10.0, 1.5 * dm_inst)

    m0_def = np.maximum(m0_accum - m0_rel_mc, 1.0)
    mw_pot = _mw_from_m0_arr(m0_def)

    p10_def, p50_def, p90_def = np.percentile(m0_def, [10, 50, 90])
    p10_mw, p50_mw, p90_mw = np.percentile(mw_pot, [10, 50, 90])
    m0_rate_p50 = float(np.median(m0_rate))

    # Physical mean recurrence time Tr for Mw >= 8.0 clipped to [150, 300] yr
    tr_raw = float(M0_MW80_NM / max(m0_rate_p50, 1.0))
    tr_clamped = float(np.clip(tr_raw, 150.0, 300.0))

    return {
        "m0_rate_p50_nm_yr": m0_rate_p50,
        "m0_deficit_p10_nm": float(p10_def),
        "m0_deficit_p50_nm": float(p50_def),
        "m0_deficit_p90_nm": float(p90_def),
        "mw_pot_p10": round(float(p10_mw), 3),
        "mw_pot_p50": round(float(p50_mw), 3),
        "mw_pot_p90": round(float(p90_mw), 3),
        "bpt_tr_years": round(tr_clamped, 1),
        "bpt_p30_a03": round(bpt_conditional_prob(years_elapsed, 30.0, tr_clamped, 0.3) * 100.0, 2),
        "bpt_p30_a05": round(bpt_conditional_prob(years_elapsed, 30.0, tr_clamped, 0.5) * 100.0, 2),
        "bpt_p50_a03": round(bpt_conditional_prob(years_elapsed, 50.0, tr_clamped, 0.3) * 100.0, 2),
        "bpt_p50_a05": round(bpt_conditional_prob(years_elapsed, 50.0, tr_clamped, 0.5) * 100.0, 2),
    }


def _run_hdbscan_modern_interplate(df_inst: pd.DataFrame) -> pd.DataFrame:
    """Cluster modern (2000-2026) interplate events M >= Mc using HDBSCAN in UTM18S + depth."""
    mod = df_inst[
        (df_inst["timestamp_utc"].dt.year >= 2000)
        & (df_inst["mag"] >= MC_MODERN)
        & (df_inst["regimen_sismotectonico"] == "subduccion_interplaca")
    ].copy()
    assert len(mod) >= 500, f"Too few modern interplate events for HDBSCAN: {len(mod)}"

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    easting_m, northing_m = transformer.transform(mod["lon"].to_numpy(), mod["lat"].to_numpy())
    coords_km = np.column_stack((easting_m / 1000.0, northing_m / 1000.0, mod["depth_km"].to_numpy()))

    clusterer = HDBSCAN(min_cluster_size=15, min_samples=8, metric="euclidean", copy=True)
    labels = clusterer.fit_predict(coords_km)
    mod["hdbscan_cluster"] = labels.astype("int32")
    return mod


def _generate_probabilistic_html(df_out: pd.DataFrame, mod_clustered: pd.DataFrame) -> None:
    """Create interactive 2-panel Plotly chart: Mw_pot (P10-P90) + BPT P(30/50yr) & HDBSCAN clusters."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    segs = df_out[df_out["entity_type"] == "segment"].sort_values("lat_center", ascending=False)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Deficit Probabilistico Mw_pot (P10-P50-P90, N=10,000 MC) por Latitud",
            "Prob. Condicional BPT (Mw>=8.0 en 30/50 años) vs Actividad HDBSCAN",
        ),
        horizontal_spacing=0.12,
    )

    # Panel 1: P10-P90 band and P50 curve (floor P10 visually at Mw 6.5 for recently ruptured segments)
    mw_p10_vis = np.maximum(segs["mw_pot_p10"].to_numpy(dtype="float64"), 6.5)
    mw_p50_vis = np.maximum(segs["mw_pot_p50"].to_numpy(dtype="float64"), 6.5)
    mw_p90_vis = np.maximum(segs["mw_pot_p90"].to_numpy(dtype="float64"), 6.5)

    fig.add_trace(
        go.Scatter(
            x=mw_p90_vis,
            y=segs["lat_center"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mw_p10_vis,
            y=segs["lat_center"],
            mode="lines",
            fill="tonextx",
            fillcolor="rgba(185, 28, 28, 0.20)",
            line=dict(width=0),
            name="Rango de incertidumbre (P10 – P90)",
        ),
        row=1,
        col=1,
    )
    hover_labels = []
    for _, r in segs.iterrows():
        yr_lbl = f"{r['last_major_rupture_year']:.0f}" if pd.notna(r["last_major_rupture_year"]) else "Sin registro desde 1552"
        p10_lbl = "Sin déficit mayor (ruptura reciente)" if float(r["mw_pot_p10"]) < 6.5 else f"Mw {r['mw_pot_p10']:.2f}"
        hover_labels.append(
            f"<b>{r['coastal_zone']}</b><br>Último gran sismo: {yr_lbl}<br>"
            f"Magnitud potencial (mediana): <b>Mw {r['mw_pot_p50']:.2f}</b><br>"
            f"Rango P10–P90: {p10_lbl} a Mw {r['mw_pot_p90']:.2f}"
        )
    fig.add_trace(
        go.Scatter(
            x=mw_p50_vis,
            y=segs["lat_center"],
            mode="lines+markers",
            name="Magnitud potencial mediana (P50)",
            text=hover_labels,
            hoverinfo="text",
            line=dict(color="#B91C1C", width=2.5),
            marker=dict(size=7),
        ),
        row=1,
        col=1,
    )

    # Panel 2: BPT 30-year & 50-year conditional probability (alpha=0.5) on documented segments
    doc = segs[segs["has_recorded_m78_in_segment"]]
    fig.add_trace(
        go.Bar(
            y=doc["lat_center"],
            x=doc["bpt_p30_a05"],
            orientation="h",
            name="Probabilidad en próximos 30 años (%)",
            marker_color="#0F766E",
            width=0.35,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            y=doc["lat_center"],
            x=doc["bpt_p50_a05"],
            mode="markers+lines",
            name="Probabilidad en próximos 50 años (%)",
            marker=dict(color="#1C1917", size=8, symbol="diamond"),
            line=dict(color="#1C1917", dash="dot"),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="Latitud costera (°S)", range=[-18.8, -3.2], row=1, col=1)
    fig.update_yaxes(range=[-18.8, -3.2], row=1, col=2)
    fig.update_xaxes(title_text="Magnitud potencial acumulada (Mw, piso visual 6.5)", range=[6.4, 8.8], row=1, col=1)
    fig.update_xaxes(title_text="Probabilidad de gran terremoto (Mw ≥ 8.0) [%]", range=[0, 100], row=1, col=2)
    fig.update_layout(
        height=720,
        title_text="Radar de Silencios Sísmicos en la Costa Peruana (Simulación Monte Carlo N=10,000 + BPT)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
    )
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")


def main() -> None:
    """Execute Model B: Monte Carlo gaps, BPT probabilities, and HDBSCAN clustering."""
    for p in (GAPS_PARQUET, INST_PARQUET, HIST_PARQUET):
        assert p.exists(), f"Missing prerequisite: {p}"

    df_gaps = pd.read_parquet(GAPS_PARQUET)
    df_inst = pd.read_parquet(INST_PARQUET)
    df_hist = pd.read_parquet(HIST_PARQUET)

    # Split released moment into historical vs instrumental with exact UTC timestamps
    h_ip = df_hist[(df_hist["depth_km"].fillna(30.0) <= 60.0) & (df_hist["dist_trench_km"] <= 180.0)].copy()
    i_ip = df_inst[(df_inst["depth_km"] <= 60.0) & (df_inst["dist_trench_km"] <= 180.0)].copy()

    all_ip_major = pd.concat(
        [
            h_ip[h_ip["mag_homog"] >= 7.8][["timestamp_utc", "lat", "mag_homog"]].rename(columns={"mag_homog": "mag_u"}),
            i_ip[i_ip["mag"] >= 7.8][["timestamp_utc", "lat", "mag"]].rename(columns={"mag": "mag_u"}),
        ],
        ignore_index=True,
    ).sort_values("timestamp_utc")
    ref_date_utc = pd.Timestamp("2026-11-23T00:00:00+00:00")
    cat_start_utc = pd.Timestamp("1552-07-02T00:00:00+00:00")

    # HDBSCAN on modern (2000-2026) interplate M >= Mc
    mod_clustered = _run_hdbscan_modern_interplate(df_inst)
    n_clusters_found = int((mod_clustered["hdbscan_cluster"] >= 0).any() and (mod_clustered["hdbscan_cluster"].max() + 1))

    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, Any]] = []

    # 1. All 30 individual coastal segments
    for _, seg in df_gaps.iterrows():
        lat_min, lat_max = float(seg["lat_min"]), float(seg["lat_max"])
        has_rec = bool(seg["has_recorded_m78_in_segment"])
        maj_in_seg = all_ip_major[(all_ip_major["lat"] >= lat_min) & (all_ip_major["lat"] < lat_max)]
        if has_rec and not maj_in_seg.empty:
            ref_ts_utc = maj_in_seg["timestamp_utc"].iloc[-1]
        else:
            ref_ts_utc = cat_start_utc

        m0_h = float(h_ip[(h_ip["lat"] >= lat_min) & (h_ip["lat"] < lat_max) & (h_ip["timestamp_utc"] > ref_ts_utc)]["seismic_moment_nm"].sum())
        m0_i = float(i_ip[(i_ip["lat"] >= lat_min) & (i_ip["lat"] < lat_max) & (i_ip["timestamp_utc"] > ref_ts_utc)]["seismic_moment_nm"].sum())

        mc_stats = _run_mc_for_entity(
            rng=rng,
            length_km=float(seg["segment_length_km"]),
            v_conv_mm_yr=float(seg["convergence_vel_mm_yr"]),
            years_elapsed=float(seg["years_since_major_rupture"]),
            m0_released_hist=m0_h,
            m0_released_inst=m0_i,
        )

        # HDBSCAN activity metrics in segment
        seg_mod = mod_clustered[(mod_clustered["lat"] >= lat_min) & (mod_clustered["lat"] < lat_max)]
        n_mod_total = int(len(seg_mod))
        n_mod_clustered = int((seg_mod["hdbscan_cluster"] >= 0).sum())
        density_per_100km = round(float(n_mod_total / (float(seg["segment_length_km"]) / 100.0)), 1)

        rows.append(
            {
                "entity_type": "segment",
                "entity_id": str(seg["segment_id"]),
                "coastal_zone": str(seg["coastal_zone"]),
                "lat_max": lat_max,
                "lat_min": lat_min,
                "lat_center": float(seg["lat_center"]),
                "length_km": float(seg["segment_length_km"]),
                "convergence_vel_mm_yr": float(seg["convergence_vel_mm_yr"]),
                "has_recorded_m78_in_segment": has_rec,
                "last_major_rupture_year": float(seg["last_major_rupture_year"]) if has_rec else np.nan,
                "years_since_major_rupture": float(seg["years_since_major_rupture"]),
                "hdbscan_modern_events": n_mod_total,
                "hdbscan_clustered_events": n_mod_clustered,
                "hdbscan_clustered_pct": round(float(n_mod_clustered / n_mod_total * 100.0), 1) if n_mod_total > 0 else 0.0,
                "modern_density_per_100km": density_per_100km,
                **mc_stats,
            }
        )

    # 2. Macro-blocks (4 main + 1 critical south sub-block)
    for b_id, b_name, lat_max, lat_min in MACRO_BLOCKS:
        sub = df_gaps[(df_gaps["lat_center"] < lat_max) & (df_gaps["lat_center"] >= lat_min)]
        total_len = float(sub["segment_length_km"].sum())
        v_mean = float(np.average(sub["convergence_vel_mm_yr"], weights=sub["segment_length_km"]))
        maj_in_blk = all_ip_major[(all_ip_major["lat"] >= lat_min) & (all_ip_major["lat"] < lat_max)]

        # Exact timestamp of the latest Mw >= 7.8 megathrust in the block so the rupture
        # event itself is NEVER double-subtracted as post-rupture release
        if not maj_in_blk.empty:
            ref_ts_utc = maj_in_blk["timestamp_utc"].iloc[-1]
            last_yr_rep = round(float(ref_ts_utc.year + (ref_ts_utc.dayofyear - 1) / 365.25), 1)
            years_eff = float((ref_date_utc - ref_ts_utc).total_seconds() / (365.25 * 86400.0))
        else:
            ref_ts_utc = cat_start_utc
            last_yr_rep = np.nan
            years_eff = float((ref_date_utc - cat_start_utc).total_seconds() / (365.25 * 86400.0))

        m0_h = float(h_ip[(h_ip["lat"] >= lat_min) & (h_ip["lat"] < lat_max) & (h_ip["timestamp_utc"] > ref_ts_utc)]["seismic_moment_nm"].sum())
        m0_i = float(i_ip[(i_ip["lat"] >= lat_min) & (i_ip["lat"] < lat_max) & (i_ip["timestamp_utc"] > ref_ts_utc)]["seismic_moment_nm"].sum())

        mc_stats = _run_mc_for_entity(
            rng=rng,
            length_km=total_len,
            v_conv_mm_yr=v_mean,
            years_elapsed=years_eff,
            m0_released_hist=m0_h,
            m0_released_inst=m0_i,
        )

        blk_mod = mod_clustered[(mod_clustered["lat"] >= lat_min) & (mod_clustered["lat"] < lat_max)]
        n_mod_total = int(len(blk_mod))
        n_mod_clustered = int((blk_mod["hdbscan_cluster"] >= 0).sum())
        rows.append(
            {
                "entity_type": "macro_block",
                "entity_id": b_id,
                "coastal_zone": b_name,
                "lat_max": lat_max,
                "lat_min": lat_min,
                "lat_center": round(0.5 * (lat_min + lat_max), 2),
                "length_km": round(total_len, 2),
                "convergence_vel_mm_yr": round(v_mean, 2),
                "has_recorded_m78_in_segment": not np.isnan(last_yr_rep),
                "last_major_rupture_year": last_yr_rep,
                "years_since_major_rupture": round(years_eff, 2),
                "hdbscan_modern_events": n_mod_total,
                "hdbscan_clustered_events": n_mod_clustered,
                "hdbscan_clustered_pct": round(float(n_mod_clustered / n_mod_total * 100.0), 1) if n_mod_total > 0 else 0.0,
                "modern_density_per_100km": round(float(n_mod_total / (total_len / 100.0)), 1),
                **mc_stats,
            }
        )

    df_out = pd.DataFrame(rows)
    for c in ("entity_type", "entity_id", "coastal_zone"):
        df_out[c] = df_out[c].astype("string")

    # Integrity assertions
    assert len(df_out) == 30 + len(MACRO_BLOCKS)
    assert (df_out["mw_pot_p50"] > 6.5).all(), f"Unphysical Mw_pot P50: {df_out[['entity_id', 'mw_pot_p50']]}"
    assert (df_out["mw_pot_p90"] >= df_out["mw_pot_p50"]).all()
    assert (df_out["mw_pot_p50"] >= df_out["mw_pot_p10"]).all()
    assert (df_out["bpt_p50_a05"] >= df_out["bpt_p30_a05"]).all()
    assert (df_out["bpt_p50_a03"] >= df_out["bpt_p30_a03"]).all()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT_PROB_PARQUET, engine="pyarrow", compression="snappy", index=False)
    _generate_probabilistic_html(df_out, mod_clustered)
    assert OUT_HTML.exists() and OUT_PROB_PARQUET.exists()

    summary = {
        "n_mc_iterations": N_MC,
        "hdbscan_modern_interplate_events": int(len(mod_clustered)),
        "hdbscan_clusters_found": n_clusters_found,
        "hdbscan_noise_pct": round(float((mod_clustered["hdbscan_cluster"] == -1).mean() * 100.0), 2),
        "macro_blocks": df_out[df_out["entity_type"] == "macro_block"].to_dict("records"),
        "top_documented_segments": df_out[
            (df_out["entity_type"] == "segment") & (df_out["has_recorded_m78_in_segment"])
        ]
        .sort_values("m0_deficit_p50_nm", ascending=False)
        .head(10)
        .to_dict("records"),
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== MODELO B (LAGUNAS PROBABILISTICAS MC + BPT + HDBSCAN) COMPLETADO ===")
    print(f"HDBSCAN (2000-2026 interplaca M>=4.7): {len(mod_clustered)} eventos -> {n_clusters_found} clusters ({summary['hdbscan_noise_pct']}% ruido)")
    print("Macro-bloques costeros (P10 / P50 / P90 Mw_pot y BPT 30/50yr):")
    mb = df_out[df_out["entity_type"] == "macro_block"]
    print(
        mb[["entity_id", "last_major_rupture_year", "years_since_major_rupture", "mw_pot_p10", "mw_pot_p50", "mw_pot_p90", "bpt_p30_a03", "bpt_p30_a05", "bpt_p50_a05"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
