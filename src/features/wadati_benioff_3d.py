"""Phase 2 - Step 3: 3D Wadati-Benioff Subduction Slab Geometry & Regime Refinement.

Builds cross-trench depth profiles (Norte -3..-9, Centro/Lima -9..-15, Sur -15..-18.5),
fits a robust L1 monotonically non-decreasing 2D Wadati-Benioff slab surface
`slab_depth_est_km(lat, dist_trench_km)` on complete subduction seismicity (M >= Mc),
computes `slab_depth_est_km` and `dist_to_slab_km` for instrumental & historical
catalogs, refines `regimen_sismotectonico` (separating interplate subduction interface
from shallow continental crustal seismicity), synchronizes `aceldat_merged.parquet`,
and exports an interactive Plotly HTML visualization to
`reports/figures/wadati_benioff_3d.html`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

INST_PARQUET = PROCESSED_DIR / "catalog_igp_instrumental.parquet"
HIST_PARQUET = PROCESSED_DIR / "catalog_igp_historico.parquet"
ACELDAT_PARQUET = PROCESSED_DIR / "aceldat_merged.parquet"
OUT_SLAB_MODEL_JSON = INTERIM_DIR / "wadati_benioff_slab_model.json"
OUT_HTML = FIGURES_DIR / "wadati_benioff_3d.html"

MC_SUBDUCTION = 4.7
DIST_KNOTS_KM = np.array([0.0, 60.0, 120.0, 180.0, 250.0, 330.0, 420.0, 520.0, 650.0], dtype="float64")
LAT_KNOTS_DEG = np.array([-4.0, -7.0, -10.0, -12.5, -14.2, -15.5, -17.0, -18.5], dtype="float64")
LAT_HALF_WINDOW = 1.75


def _fit_monotone_l1_profile(dist_km: np.ndarray, depth_km: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Fit piecewise-linear monotonically non-decreasing median (L1 + mild smoothing) slab depth profile."""
    # Parameterize knot depths via non-negative increments: z[0] >= 5, dz[k] >= 0
    k = len(knots)
    # Initial guess from binned medians
    init_z = np.zeros(k, dtype="float64")
    init_z[0] = 6.0
    for j in range(1, k):
        lo = 0.5 * (knots[j - 1] + knots[j])
        hi = 0.5 * (knots[j] + knots[j + 1]) if j + 1 < k else knots[j] + 80.0
        mask = (dist_km >= lo) & (dist_km <= hi)
        if np.sum(mask) >= 5:
            init_z[j] = max(init_z[j - 1], float(np.median(depth_km[mask])))
        else:
            init_z[j] = init_z[j - 1] + 15.0

    init_params = np.zeros(k, dtype="float64")
    init_params[0] = init_z[0]
    init_params[1:] = np.maximum(np.diff(init_z), 0.5)

    # Precompute linear interpolation weights for each observation onto `knots`
    d_clip = np.clip(dist_km, knots[0], knots[-1])
    idx_hi = np.clip(np.searchsorted(knots, d_clip, side="right"), 1, k - 1)
    idx_lo = idx_hi - 1
    w_hi = (d_clip - knots[idx_lo]) / (knots[idx_hi] - knots[idx_lo])
    w_lo = 1.0 - w_hi

    def objective(params: np.ndarray) -> float:
        z = np.cumsum(params)
        pred = w_lo * z[idx_lo] + w_hi * z[idx_hi]
        # Huber / L1-like robust loss (delta = 5 km) + small second-derivative curvature smoothing
        res = depth_km - pred
        delta = 5.0
        huber = np.where(np.abs(res) <= delta, 0.5 * (res ** 2) / delta, np.abs(res) - 0.5 * delta)
        # Curvature regularization on slopes
        slopes = np.diff(z) / np.diff(knots)
        smooth_pen = 35.0 * float(np.sum(np.diff(slopes) ** 2))
        return float(np.mean(huber)) + smooth_pen

    bounds = [(5.0, 12.0)] + [(0.0, 120.0)] * (k - 1)
    res = minimize(objective, init_params, method="L-BFGS-B", bounds=bounds)
    z_opt = np.cumsum(res.x)
    return z_opt


def fit_wadati_benioff_surface(df_inst: pd.DataFrame) -> dict[str, Any]:
    """Fit 2D slab depth table `z_grid[lat_idx, dist_idx]` across latitude knots."""
    # Use M >= Mc complete events with depth <= 350 km, excluding clearly shallow inland crustal (>220 km & depth<=55 km)
    sub = df_inst[
        (df_inst["mag"] >= MC_SUBDUCTION)
        & (df_inst["depth_km"] <= 350.0)
        & (df_inst["dist_trench_km"] <= 700.0)
        & (~((df_inst["dist_trench_km"] > 220.0) & (df_inst["depth_km"] <= 55.0)))
    ].copy()
    assert len(sub) >= 5000, f"Too few subduction events for slab fit: {len(sub)}"

    z_grid = np.zeros((len(LAT_KNOTS_DEG), len(DIST_KNOTS_KM)), dtype="float64")
    knot_counts: list[int] = []
    for i, lat_k in enumerate(LAT_KNOTS_DEG):
        # Asymmetric narrow window near the 15S transition to preserve sharp flat-vs-normal dip change
        if lat_k == -14.2:
            mask = (sub["lat"] >= -15.0) & (sub["lat"] <= -12.5)
        elif lat_k == -15.5:
            mask = (sub["lat"] <= -15.0) & (sub["lat"] >= -17.0)
        else:
            mask = (sub["lat"] >= lat_k - LAT_HALF_WINDOW) & (sub["lat"] <= lat_k + LAT_HALF_WINDOW)
        band = sub.loc[mask]
        knot_counts.append(int(len(band)))
        z_grid[i, :] = _fit_monotone_l1_profile(
            band["dist_trench_km"].to_numpy(), band["depth_km"].to_numpy(), DIST_KNOTS_KM
        )

    return {
        "mc_used": MC_SUBDUCTION,
        "n_events_fit": int(len(sub)),
        "lat_knots_deg": LAT_KNOTS_DEG.tolist(),
        "dist_knots_km": DIST_KNOTS_KM.tolist(),
        "slab_depth_grid_km": np.round(z_grid, 2).tolist(),
        "band_event_counts": knot_counts,
    }


def predict_slab_depth(
    lat: np.ndarray, dist_trench_km: np.ndarray, model: dict[str, Any]
) -> np.ndarray:
    """Bilinearly interpolate `slab_depth_est_km` from the fitted 2D Wadati-Benioff grid."""
    # Sort lat_knots ascending (-18.5 .. -4.0) for np.searchsorted
    lat_knots_orig = np.asarray(model["lat_knots_deg"], dtype="float64")
    order = np.argsort(lat_knots_orig)
    lat_knots = lat_knots_orig[order]
    dist_knots = np.asarray(model["dist_knots_km"], dtype="float64")
    z_grid = np.asarray(model["slab_depth_grid_km"], dtype="float64")[order, :]

    lat_c = np.clip(lat, lat_knots[0], lat_knots[-1])
    dist_c = np.clip(dist_trench_km, dist_knots[0], dist_knots[-1])

    li_hi = np.clip(np.searchsorted(lat_knots, lat_c, side="right"), 1, len(lat_knots) - 1)
    li_lo = li_hi - 1
    wl = (lat_c - lat_knots[li_lo]) / (lat_knots[li_hi] - lat_knots[li_lo])

    di_hi = np.clip(np.searchsorted(dist_knots, dist_c, side="right"), 1, len(dist_knots) - 1)
    di_lo = di_hi - 1
    wd = (dist_c - dist_knots[di_lo]) / (dist_knots[di_hi] - dist_knots[di_lo])

    z00 = z_grid[li_lo, di_lo]
    z01 = z_grid[li_lo, di_hi]
    z10 = z_grid[li_hi, di_lo]
    z11 = z_grid[li_hi, di_hi]

    z0 = (1.0 - wd) * z00 + wd * z01
    z1 = (1.0 - wd) * z10 + wd * z11
    return ((1.0 - wl) * z0 + wl * z1).astype("float64")


def refine_regime_with_slab(
    depth_km: pd.Series, dist_trench_km: pd.Series, dist_to_slab_km: pd.Series
) -> pd.Series:
    """Refine 4 seismotectonic regimes using Wadati-Benioff slab proximity.

    - intramanto_profundo:   depth_km > 350
    - intramanto_intermedio: 60 < depth_km <= 350
    - subduccion_interplaca: depth_km <= 60 AND (dist_trench_km <= 140 OR (dist_trench_km <= 210 AND dist_to_slab_km <= 35))
    - cortical_continental:  depth_km <= 60 AND outside interplate interface envelope
    """
    d = depth_km.to_numpy(dtype="float64")
    t = dist_trench_km.to_numpy(dtype="float64")
    s = dist_to_slab_km.to_numpy(dtype="float64")

    is_interplate = (d <= 60.0) & ((t <= 140.0) | ((t <= 210.0) & (s <= 35.0)))
    conds = [
        np.isnan(d),
        d > 350.0,
        (d > 60.0) & (d <= 350.0),
        is_interplate,
        d <= 60.0,
    ]
    choices = [
        "indeterminado_sin_profundidad",
        "intramanto_profundo",
        "intramanto_intermedio",
        "subduccion_interplaca",
        "cortical_continental",
    ]
    return pd.Series(np.select(conds, choices, default="indeterminado_sin_profundidad"), index=depth_km.index, dtype="string")


def _compute_profile_diagnostics(df_inst: pd.DataFrame, model: dict[str, Any]) -> dict[str, Any]:
    """Compute apparent slab dip angles and depth benchmarks across Norte, Centro, and Sur."""
    bands = [
        ("Norte (-3 a -9)", -9.0, -3.0, -6.5),
        ("Centro / Lima (-9 a -15)", -15.0, -9.0, -12.0),
        ("Sur (-15 a -18.5)", -18.5, -15.0, -16.8),
    ]
    dist_eval = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    out: dict[str, Any] = {}
    for name, lat_lo, lat_hi, lat_rep in bands:
        sub = df_inst[(df_inst["lat"] >= lat_lo) & (df_inst["lat"] < lat_hi) & (df_inst["mag"] >= MC_SUBDUCTION) & (df_inst["depth_km"] <= 350.0)]
        z_eval = predict_slab_depth(np.full_like(dist_eval, lat_rep), dist_eval, model)
        # Apparent dip angles: shallow-to-mid (100->300 km) and deep/flat bench (250->500 km)
        z_250 = float(predict_slab_depth(np.array([lat_rep]), np.array([250.0]), model)[0])
        dip_100_300 = float(np.degrees(np.arctan2(z_eval[2] - z_eval[0], 200.0)))
        dip_250_500 = float(np.degrees(np.arctan2(z_eval[4] - z_250, 250.0)))
        dip_100_500 = float(np.degrees(np.arctan2(z_eval[4] - z_eval[0], 400.0)))
        out[name] = {
            "lat_rep": lat_rep,
            "n_events_m_ge_mc": int(len(sub)),
            "slab_depth_100km": round(float(z_eval[0]), 1),
            "slab_depth_200km": round(float(z_eval[1]), 1),
            "slab_depth_300km": round(float(z_eval[2]), 1),
            "slab_depth_400km": round(float(z_eval[3]), 1),
            "slab_depth_500km": round(float(z_eval[4]), 1),
            "dip_deg_100_300km": round(dip_100_300, 1),
            "dip_deg_250_500km": round(dip_250_500, 1),
            "dip_deg_100_500km": round(dip_100_500, 1),
            "median_dist_to_slab_km": round(float(sub["dist_to_slab_km"].median()), 1),
        }
    return out


def _generate_interactive_html(df_inst: pd.DataFrame, model: dict[str, Any]) -> None:
    """Create interactive 3D Wadati-Benioff + 2D latitudinal profile HTML figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sub_plot = df_inst[(df_inst["mag"] >= 4.7) & (df_inst["depth_km"] <= 350.0)].copy()

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "scene", "colspan": 3}, None, None],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.58, 0.42],
        subplot_titles=(
            "Geometria 3D Wadati-Benioff (M >= 4.7, Prof <= 350 km) y Superficie del Slab",
            "Perfil Norte (-3 a -9 S): Flat-Slab",
            "Perfil Centro / Lima (-9 a -15 S): Flat-Slab",
            "Perfil Sur (-15 a -18.5 S): Subduccion Normal",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    regime_colors = {
        "subduccion_interplaca": "#e63946",
        "cortical_continental": "#f4a261",
        "intramanto_intermedio": "#2a9d8f",
        "intramanto_profundo": "#264653",
    }

    # 3D Scatter by regime (downsample slightly if >8000 for crisp browser rendering)
    for reg, color in regime_colors.items():
        r_df = sub_plot[sub_plot["regimen_sismotectonico"] == reg]
        if r_df.empty:
            continue
        if len(r_df) > 2500:
            r_df = r_df.sample(2500, random_state=42)
        fig.add_trace(
            go.Scatter3d(
                x=r_df["lon"],
                y=r_df["lat"],
                z=-r_df["depth_km"],
                mode="markers",
                name=reg,
                marker=dict(size=2.5, color=color, opacity=0.65),
            ),
            row=1,
            col=1,
        )

    # 2D Profiles
    bands = [
        (1, -9.0, -3.0, -6.5),
        (2, -15.0, -9.0, -12.0),
        (3, -18.5, -15.0, -16.8),
    ]
    d_line = np.linspace(0.0, 620.0, 125)
    for col_idx, lat_lo, lat_hi, lat_rep in bands:
        b_df = sub_plot[(sub_plot["lat"] >= lat_lo) & (sub_plot["lat"] < lat_hi) & (sub_plot["dist_trench_km"] <= 650.0)]
        fig.add_trace(
            go.Scattergl(
                x=b_df["dist_trench_km"],
                y=b_df["depth_km"],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3.5,
                    color=[regime_colors.get(r, "#888") for r in b_df["regimen_sismotectonico"]],
                    opacity=0.45,
                ),
            ),
            row=2,
            col=col_idx,
        )
        z_curve = predict_slab_depth(np.full_like(d_line, lat_rep), d_line, model)
        fig.add_trace(
            go.Scatter(
                x=d_line,
                y=z_curve,
                mode="lines",
                name="Slab L1 ajustado" if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                line=dict(color="#111111", width=3),
            ),
            row=2,
            col=col_idx,
        )
        fig.update_yaxes(autorange="reversed", range=[320, 0], title_text="Profundidad (km)" if col_idx == 1 else "", row=2, col=col_idx)
        fig.update_xaxes(range=[0, 650], title_text="Dist. a Fosa (km)", row=2, col=col_idx)

    fig.update_layout(
        height=920,
        title_text="Subduccion Nazca-Sudamerica en Peru: Plano de Wadati-Benioff 3D y Transicion 15 S",
        scene=dict(
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            zaxis_title="Elevacion / -Profundidad (km)",
            aspectratio=dict(x=1.0, y=1.3, z=0.6),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white",
    )
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")


def main() -> None:
    """Fit Wadati-Benioff 3D slab, refine regimes across processed catalogs, and plot HTML."""
    for p in (INST_PARQUET, HIST_PARQUET, ACELDAT_PARQUET):
        assert p.exists(), f"Missing prerequisite: {p}"

    df_inst = pd.read_parquet(INST_PARQUET)
    df_hist = pd.read_parquet(HIST_PARQUET)
    df_ac = pd.read_parquet(ACELDAT_PARQUET)

    regimes_before = df_inst["regimen_sismotectonico"].value_counts().to_dict()

    model = fit_wadati_benioff_surface(df_inst)

    # Apply to Instrumental
    df_inst["slab_depth_est_km"] = predict_slab_depth(
        df_inst["lat"].to_numpy(), df_inst["dist_trench_km"].to_numpy(), model
    )
    df_inst["dist_to_slab_km"] = np.abs(df_inst["depth_km"] - df_inst["slab_depth_est_km"]).astype("float64")
    df_inst["regimen_sismotectonico_v1"] = df_inst["regimen_sismotectonico"]
    df_inst["regimen_sismotectonico"] = refine_regime_with_slab(
        df_inst["depth_km"], df_inst["dist_trench_km"], df_inst["dist_to_slab_km"]
    )

    assert df_inst["slab_depth_est_km"].notna().all()
    assert df_inst["dist_to_slab_km"].notna().all()
    assert set(df_inst["regimen_sismotectonico"].unique()) == {
        "subduccion_interplaca",
        "cortical_continental",
        "intramanto_intermedio",
        "intramanto_profundo",
    }

    # Apply to Historical
    df_hist["slab_depth_est_km"] = predict_slab_depth(
        df_hist["lat"].to_numpy(), df_hist["dist_trench_km"].to_numpy(), model
    )
    df_hist["dist_to_slab_km"] = np.abs(df_hist["depth_km"] - df_hist["slab_depth_est_km"]).astype("float64")
    df_hist["regimen_sismotectonico"] = refine_regime_with_slab(
        df_hist["depth_km"], df_hist["dist_trench_km"], df_hist["dist_to_slab_km"]
    )

    # Synchronize ACELDAT merged with refined event slab & regime
    drop_existing = [c for c in ("event_slab_depth_est_km", "event_dist_to_slab_km", "regimen_sismotectonico") if c in df_ac.columns]
    df_ac = df_ac.drop(columns=drop_existing).merge(
        df_inst[["event_id", "slab_depth_est_km", "dist_to_slab_km", "regimen_sismotectonico"]].rename(
            columns={
                "slab_depth_est_km": "event_slab_depth_est_km",
                "dist_to_slab_km": "event_dist_to_slab_km",
            }
        ),
        on="event_id",
        how="left",
    )
    assert df_ac["regimen_sismotectonico"].notna().all()

    df_inst.to_parquet(INST_PARQUET, engine="pyarrow", compression="snappy", index=False)
    df_hist.to_parquet(HIST_PARQUET, engine="pyarrow", compression="snappy", index=False)
    df_ac.to_parquet(ACELDAT_PARQUET, engine="pyarrow", compression="snappy", index=False)

    diagnostics = _compute_profile_diagnostics(df_inst, model)
    regimes_after = df_inst["regimen_sismotectonico"].value_counts().to_dict()
    reclassified = int((df_inst["regimen_sismotectonico"] != df_inst["regimen_sismotectonico_v1"]).sum())

    model_report = {
        "model": model,
        "profile_diagnostics": diagnostics,
        "regimes_before_slab": regimes_before,
        "regimes_after_slab": regimes_after,
        "reclassified_events": reclassified,
        "aceldat_regimes_after_slab": df_ac["regimen_sismotectonico"].value_counts().to_dict(),
        "figure_path": str(OUT_HTML),
    }
    OUT_SLAB_MODEL_JSON.write_text(json.dumps(model_report, indent=2, ensure_ascii=False), encoding="utf-8")

    _generate_interactive_html(df_inst, model)
    assert OUT_HTML.exists() and OUT_HTML.stat().st_size > 10000

    print("=== GEOMETRIA 3D WADATI-BENIOFF Y REFINAMIENTO COMPLETADO ===")
    for k, v in diagnostics.items():
        print(
            f"  {k:26s} | z(100/200/300/400/500km)={v['slab_depth_100km']}/{v['slab_depth_200km']}/{v['slab_depth_300km']}/{v['slab_depth_400km']}/{v['slab_depth_500km']} km"
            f" | dip(100-300)={v['dip_deg_100_300km']} deg | dip(250-500)={v['dip_deg_250_500km']} deg"
        )
    print(f"Regimenes antes: {regimes_before}")
    print(f"Regimenes tras slab: {regimes_after} (reclasificados: {reclassified})")
    print(f"Figura 3D guardada en: {OUT_HTML}")


if __name__ == "__main__":
    main()
