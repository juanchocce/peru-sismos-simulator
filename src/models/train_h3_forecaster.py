"""Phase 3 - Model C: Spatio-Temporal Seismic Forecaster on H3 (Res 5) Grid.

Evaluates 4 modeling approaches on `data/processed/h3_spatiotemporal_panel.parquet`
(2,650,215 rows x 53 cols) under strict Expanding-Window Walk-Forward Validation
with mandatory purge gaps (4 months for 90-day targets, 12 months for 365-day target):
  1. Baseline 1: Stationary Spatial Poisson (Empirical Bayes cell rate).
  2. Baseline 2: Calibrated Parametric ETAS / Omori Logistic Regression.
  3. Model ML 1: Probabilistic LightGBM + Internal-Split Isotonic/Platt Calibration.
  4. Model ML 2: Probabilistic XGBoost + Internal-Split Isotonic/Platt Calibration.

Evaluates `target_m45_90d` (primary), `target_m45_365d` (annual), and `target_m50_90d`
using PR-AUC, ROC-AUC, Log-Loss, Brier Score, and Brier Skill Score (BSS vs Poisson).
Generates operational calibrated forecast for `2026-11-01` across all 8,205 H3 cells
(`data/processed/h3_forecast_latest.parquet`), persists `models/h3_forecaster_lgbm.joblib`,
and exports SHAP + forecast map to `reports/figures/h3_forecaster_shap.html`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import xgboost as xgb
from plotly.subplots import make_subplots
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PANEL_PARQUET = PROCESSED_DIR / "h3_spatiotemporal_panel.parquet"
OUT_FORECAST_LATEST = PROCESSED_DIR / "h3_forecast_latest.parquet"
OUT_MODEL_JOBLIB = MODELS_DIR / "h3_forecaster_lgbm.joblib"
OUT_SHAP_HTML = FIGURES_DIR / "h3_forecaster_shap.html"
METRICS_JSON = INTERIM_DIR / "h3_forecaster_metrics.json"

RANDOM_STATE = 42
POISSON_PRIOR_STRENGTH = 12.0

# 4 Expanding-Window Walk-Forward Splits (with exact purge gap per target horizon)
SPLITS_90D = [
    ("Split_1_2018_2019", "2017-09-01", "2018-01-01", "2019-12-01"),
    ("Split_2_2020_2021", "2019-09-01", "2020-01-01", "2021-12-01"),
    ("Split_3_2022_2023", "2021-09-01", "2022-01-01", "2023-12-01"),
    ("Split_4_2024_2025", "2023-09-01", "2024-01-01", "2025-10-01"),
]

SPLITS_365D = [
    ("Split_1_2018_2019", "2017-01-01", "2018-01-01", "2019-12-01"),
    ("Split_2_2020_2021", "2019-01-01", "2020-01-01", "2021-12-01"),
    ("Split_3_2022_2023", "2021-01-01", "2022-01-01", "2023-12-01"),
    ("Split_4_2024_2025", "2023-01-01", "2024-01-01", "2025-10-01"),
]

ML_BASE_FEATURES = [
    "cell_lat",
    "cell_lon",
    "dist_trench_km",
    "slab_depth_est_km",
    "convergence_vel_mm_yr",
    "seafloor_age_myr",
    "dist_lat_15s_km",
    "is_normal_sur",
    "n_geocatmin_peligros",
    "n_geocatmin_zonas_criticas",
    "cnt_mc_30d",
    "cnt_mc_90d",
    "cnt_mc_365d",
    "cnt_mc_5y",
    "cnt_m40_30d",
    "cnt_m40_90d",
    "cnt_m40_365d",
    "cnt_m40_5y",
    "cnt_m45_30d",
    "cnt_m45_90d",
    "cnt_m45_365d",
    "cnt_m45_5y",
    "nbhd_cnt_m45_30d",
    "nbhd_cnt_m45_90d",
    "nbhd_cnt_m45_365d",
    "nbhd_cnt_m45_5y",
    "log10_energy_30d",
    "log10_energy_90d",
    "log10_energy_365d",
    "log10_energy_5y",
    "log10_moment_30d",
    "log10_moment_90d",
    "log10_moment_365d",
    "log10_moment_5y",
    "etas_decay_cell_365d",
    "etas_decay_nbhd_365d",
    "days_since_last_m50_nbhd",
    "days_since_last_m60_nbhd",
]


def _fit_poisson_spatial_rate(train_df: pd.DataFrame, target_col: str) -> tuple[dict[str, float], dict[int, float]]:
    """Empirical Bayes stationary Poisson cell probability estimated strictly on `train_df`."""
    reg_priors = train_df.groupby("is_normal_sur")[target_col].mean().to_dict()
    cell_agg = train_df.groupby(["h3_res5", "is_normal_sur"], as_index=False)[target_col].agg(["count", "sum"])
    cell_agg["prior"] = cell_agg["is_normal_sur"].map(reg_priors).fillna(float(train_df[target_col].mean()))
    cell_agg["rate"] = (cell_agg["sum"] + POISSON_PRIOR_STRENGTH * cell_agg["prior"]) / (
        cell_agg["count"] + POISSON_PRIOR_STRENGTH
    )
    cell_rate_map = dict(zip(cell_agg["h3_res5"].tolist(), cell_agg["rate"].astype("float64").tolist()))
    return cell_rate_map, {int(k): float(v) for k, v in reg_priors.items()}


def _predict_poisson(df: pd.DataFrame, cell_rate_map: dict[str, float], reg_priors: dict[int, float]) -> np.ndarray:
    fallback = float(np.mean(list(reg_priors.values())))
    cells = df["h3_res5"].tolist()
    regs = df["is_normal_sur"].to_numpy()
    return np.array(
        [cell_rate_map.get(c, reg_priors.get(int(r), fallback)) for c, r in zip(cells, regs)],
        dtype="float64",
    )


def _logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def _calibrate_probs(
    raw_cal: np.ndarray, y_cal: np.ndarray, raw_test: np.ndarray
) -> tuple[np.ndarray, str, Any]:
    """Fit monotonic Platt calibrator strictly on internal calibration split (preserves PR-AUC ranking)."""
    platt = LogisticRegression(C=10.0, max_iter=200)
    platt.fit(_logit(raw_cal).reshape(-1, 1), y_cal)
    p_test = np.clip(platt.predict_proba(_logit(raw_test).reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6)
    return p_test, "platt", platt


def _add_causal_expanding_cell_rate(panel: pd.DataFrame, target_col: str, lag_months: int) -> np.ndarray:
    """Compute strictly causal expanding cell event rate using ONLY targets matured >= lag_months earlier."""
    # Sort by cell then time so shifted cumsum is 100% leak-free
    tmp = panel[["h3_res5", "cut_date_utc", target_col]].copy()
    tmp["_orig_idx"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["h3_res5", "cut_date_utc"])
    g = tmp.groupby("h3_res5", sort=False)[target_col]
    y_fill = tmp[target_col].fillna(0.0)
    valid_flag = tmp[target_col].notna().astype("float64")
    tmp["_y_shift"] = y_fill.groupby(tmp["h3_res5"], sort=False).shift(lag_months).fillna(0.0)
    tmp["_n_shift"] = valid_flag.groupby(tmp["h3_res5"], sort=False).shift(lag_months).fillna(0.0)
    cum_y = tmp.groupby("h3_res5", sort=False)["_y_shift"].cumsum()
    cum_n = tmp.groupby("h3_res5", sort=False)["_n_shift"].cumsum()
    prior_rate = float(
        panel.loc[panel["cut_date_utc"] <= pd.Timestamp("2010-01-01", tz="UTC"), target_col].dropna().mean()
    )
    rate = (cum_y + POISSON_PRIOR_STRENGTH * prior_rate) / (cum_n + POISSON_PRIOR_STRENGTH)
    tmp["_rate"] = rate.astype("float32")
    return tmp.sort_values("_orig_idx")["_rate"].to_numpy(dtype="float32")


def _eval_prob_metrics(y_true: np.ndarray, y_prob: np.ndarray, brier_poisson: float) -> dict[str, float]:
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    pr_auc = float(average_precision_score(y_true, p))
    roc_auc = float(roc_auc_score(y_true, p))
    ll = float(log_loss(y_true, p))
    bs = float(brier_score_loss(y_true, p))
    bss = float(1.0 - bs / max(brier_poisson, 1e-12))
    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "log_loss": round(ll, 5),
        "brier_score": round(bs, 6),
        "bss_vs_poisson": round(bss, 4),
    }


def _evaluate_target_walk_forward(
    panel: pd.DataFrame,
    target_col: str,
    horizon_days: int,
    splits: list[tuple[str, str, str, str]],
) -> dict[str, Any]:
    """Run 4-split expanding walk-forward evaluation with mandatory purge gap."""
    purge_months = 4 if horizon_days <= 90 else 12
    causal_rate_all = _add_causal_expanding_cell_rate(panel, target_col, lag_months=purge_months)
    panel = panel.copy()
    panel["_causal_bg_rate"] = causal_rate_all
    valid_panel = panel[panel[target_col].notna()].copy()

    per_split_results: list[dict[str, Any]] = []
    model_names = ["poisson_stationary", "etas_parametric", "lgbm_calibrated", "xgb_calibrated"]

    for split_name, train_end_s, test_start_s, test_end_s in splits:
        train_end = pd.Timestamp(train_end_s, tz="UTC")
        test_start = pd.Timestamp(test_start_s, tz="UTC")
        test_end = pd.Timestamp(test_end_s, tz="UTC")

        # Strict purge gap verification
        gap_days = (test_start - train_end).days
        assert gap_days >= horizon_days, f"Purge gap violation in {split_name}: {gap_days}d < {horizon_days}d"

        tr_df = valid_panel[valid_panel["cut_date_utc"] <= train_end]
        te_df = valid_panel[(valid_panel["cut_date_utc"] >= test_start) & (valid_panel["cut_date_utc"] <= test_end)]
        assert not tr_df.empty and not te_df.empty

        y_te = te_df[target_col].to_numpy(dtype="int8")

        # 1. Baseline 1: Stationary Spatial Poisson (strictly on train_df)
        cell_map_full, reg_priors_full = _fit_poisson_spatial_rate(tr_df, target_col)
        p_poisson_te = _predict_poisson(te_df, cell_map_full, reg_priors_full)
        brier_pois = float(brier_score_loss(y_te, p_poisson_te))

        # Internal temporal split inside train_df for leak-free ETAS fitting & ML calibration
        cal_start = train_end - pd.DateOffset(months=23)
        fit_end = cal_start - pd.DateOffset(months=purge_months)
        fit_df = tr_df[tr_df["cut_date_utc"] <= fit_end]
        cal_df = tr_df[tr_df["cut_date_utc"] >= cal_start]
        assert not fit_df.empty and not cal_df.empty
        assert (cal_df["cut_date_utc"].min() - fit_df["cut_date_utc"].max()).days >= horizon_days

        # Spatial background probability estimated ONLY on fit_df for cal_df/ETAS
        cell_map_fit, reg_priors_fit = _fit_poisson_spatial_rate(fit_df, target_col)
        bg_cal = _predict_poisson(cal_df, cell_map_fit, reg_priors_fit)

        # 2. Baseline 2: Parametric ETAS / Omori Logistic Regression (fit strictly on cal_df using out-of-time bg)
        X_etas_cal = np.column_stack(
            (
                _logit(bg_cal),
                np.log1p(cal_df["etas_decay_cell_365d"].to_numpy(dtype="float64")),
                np.log1p(cal_df["etas_decay_nbhd_365d"].to_numpy(dtype="float64")),
            )
        )
        X_etas_te = np.column_stack(
            (
                _logit(p_poisson_te),
                np.log1p(te_df["etas_decay_cell_365d"].to_numpy(dtype="float64")),
                np.log1p(te_df["etas_decay_nbhd_365d"].to_numpy(dtype="float64")),
            )
        )
        y_cal = cal_df[target_col].to_numpy(dtype="int8")
        etas_lr = LogisticRegression(C=1.0, max_iter=300)
        etas_lr.fit(X_etas_cal, y_cal)
        p_etas_te = etas_lr.predict_proba(X_etas_te)[:, 1]

        # Build ML feature matrices using strictly causal expanding cell rate (zero in-sample target leak)
        ml_cols_with_bg = ML_BASE_FEATURES + ["_causal_bg_rate"]
        X_fit = fit_df[ml_cols_with_bg].to_numpy(dtype="float32")
        y_fit = fit_df[target_col].to_numpy(dtype="int8")
        X_cal = cal_df[ml_cols_with_bg].to_numpy(dtype="float32")
        X_te = te_df[ml_cols_with_bg].to_numpy(dtype="float32")

        # 3. Model ML 1: LightGBM + internal-split Platt calibration
        lgbm = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=120,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(X_fit, y_fit)
        raw_lgb_cal = lgbm.predict_proba(X_cal)[:, 1]
        raw_lgb_te = lgbm.predict_proba(X_te)[:, 1]
        p_lgb_te, cal_type_lgb, _ = _calibrate_probs(raw_lgb_cal, y_cal, raw_lgb_te)

        # 4. Model ML 2: XGBoost + internal-split Platt calibration
        xgbm = xgb.XGBClassifier(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=80,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgbm.fit(X_fit, y_fit)
        raw_xgb_cal = xgbm.predict_proba(X_cal)[:, 1]
        raw_xgb_te = xgbm.predict_proba(X_te)[:, 1]
        p_xgb_te, cal_type_xgb, _ = _calibrate_probs(raw_xgb_cal, y_cal, raw_xgb_te)

        split_res = {
            "split": split_name,
            "train_rows": int(len(tr_df)),
            "test_rows": int(len(te_df)),
            "test_pos_rate_pct": round(float(y_te.mean() * 100.0), 4),
            "lgb_calibrator": cal_type_lgb,
            "xgb_calibrator": cal_type_xgb,
            "metrics": {
                "poisson_stationary": _eval_prob_metrics(y_te, p_poisson_te, brier_pois),
                "etas_parametric": _eval_prob_metrics(y_te, p_etas_te, brier_pois),
                "lgbm_calibrated": _eval_prob_metrics(y_te, p_lgb_te, brier_pois),
                "xgb_calibrated": _eval_prob_metrics(y_te, p_xgb_te, brier_pois),
            },
        }
        per_split_results.append(split_res)

    # Aggregate 4-split averages
    mean_metrics: dict[str, dict[str, float]] = {}
    for m_name in model_names:
        mean_metrics[m_name] = {
            k: round(float(np.mean([s["metrics"][m_name][k] for s in per_split_results])), 6 if k == "brier_score" else 4)
            for k in ("pr_auc", "roc_auc", "log_loss", "brier_score", "bss_vs_poisson")
        }

    return {"target": target_col, "horizon_days": horizon_days, "mean_4splits": mean_metrics, "splits": per_split_results}


def _train_final_and_forecast_latest(
    panel: pd.DataFrame,
    target_col: str,
    horizon_days: int,
    latest_cut: pd.Timestamp,
) -> tuple[np.ndarray, lgb.LGBMClassifier, pd.DataFrame, dict[str, Any]]:
    """Train final calibrated LightGBM on all uncensored cuts using strictly causal expanding rate."""
    purge_months = 4 if horizon_days <= 90 else 12
    panel_w = panel.copy()
    panel_w["_causal_bg_rate"] = _add_causal_expanding_cell_rate(panel_w, target_col, lag_months=purge_months)

    valid = panel_w[panel_w[target_col].notna()].copy()
    max_valid_cut = valid["cut_date_utc"].max()
    cal_start = max_valid_cut - pd.DateOffset(months=23)
    fit_end = cal_start - pd.DateOffset(months=purge_months)

    fit_df = valid[valid["cut_date_utc"] <= fit_end]
    cal_df = valid[valid["cut_date_utc"] >= cal_start]
    latest_df = panel_w[panel_w["cut_date_utc"] == latest_cut].reset_index(drop=True)

    ml_cols_with_bg = ML_BASE_FEATURES + ["_causal_bg_rate"]
    X_fit = fit_df[ml_cols_with_bg].to_numpy(dtype="float32")
    y_fit = fit_df[target_col].to_numpy(dtype="int8")
    X_cal = cal_df[ml_cols_with_bg].to_numpy(dtype="float32")
    y_cal = cal_df[target_col].to_numpy(dtype="int8")
    X_latest = latest_df[ml_cols_with_bg].to_numpy(dtype="float32")

    model = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_fit, y_fit)
    raw_cal = model.predict_proba(X_cal)[:, 1]
    raw_latest = model.predict_proba(X_latest)[:, 1]
    p_latest, cal_name, calibrator = _calibrate_probs(raw_cal, y_cal, raw_latest)

    feature_names = ML_BASE_FEATURES + ["causal_expanding_cell_rate"]
    X_latest_df = pd.DataFrame(X_latest, columns=feature_names)
    meta = {"calibrator_type": cal_name, "calibrator": calibrator, "max_valid_cut": max_valid_cut.isoformat()}
    return p_latest, model, X_latest_df, meta


def _generate_shap_and_forecast_html(
    model_m45_90d: lgb.LGBMClassifier,
    X_latest_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Compute TreeSHAP for H3 forecaster and render SHAP + 2026-11-01 Peru probability map."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    # Focus SHAP sample on active + stratified background cells for clear seismic signal
    sample_idx = rng.choice(len(X_latest_df), size=min(3000, len(X_latest_df)), replace=False)
    X_sample = X_latest_df.iloc[sample_idx]

    explainer = shap.TreeExplainer(model_m45_90d)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    shap_df = (
        pd.DataFrame({"feature": X_latest_df.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    # Translate SHAP features to plain Spanish and separate map title from colorbar cleanly
    import sys as _sys
    if str(PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_ROOT))
    from app.ui_theme import translate_feature

    top15 = shap_df.head(15).iloc[::-1].copy()
    top15["feature_es"] = [translate_feature(f) for f in top15["feature"]]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "geo"}]],
        column_widths=[0.48, 0.52],
        subplot_titles=(
            "Factores Principales del Pronóstico (Próximos 3 Meses, M ≥ 4.5)",
            "Mapa de Probabilidad Calibrada: Sismos M ≥ 4.5 en 90 Días",
        ),
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Bar(
            x=top15["mean_abs_shap"],
            y=top15["feature_es"],
            orientation="h",
            marker_color="#B91C1C",
            name="Impacto relativo",
        ),
        row=1,
        col=1,
    )

    # Show top active cells (prob >= 0.5%) on Peru map with non-overlapping colorbar
    active = forecast_df[forecast_df["prob_m45_90d"] >= 0.005].copy()
    fig.add_trace(
        go.Scattergeo(
            lon=active["cell_lon"],
            lat=active["cell_lat"],
            mode="markers",
            text=[
                f"<b>Zona Sísmica</b><br>Probabilidad 3 meses (M≥4.5): {r['prob_m45_90d']*100:.2f}%<br>"
                f"Probabilidad 12 meses (M≥4.5): {r['prob_m45_365d']*100:.2f}%<br>"
                f"Probabilidad 3 meses (M≥5.0): {r['prob_m50_90d']*100:.2f}%<br>"
                f"Distancia a la fosa marina: {r['dist_trench_km']:.0f} km"
                for _, r in active.iterrows()
            ],
            hoverinfo="text",
            marker=dict(
                size=5.5,
                color=active["prob_m45_90d"] * 100.0,
                colorscale="YlOrRd",
                colorbar=dict(
                    title=dict(text="Probabilidad (%)", side="right"),
                    x=1.02,
                    y=0.45,
                    len=0.72,
                    thickness=14,
                ),
                opacity=0.84,
            ),
            name="Zonas hexagonales",
        ),
        row=1,
        col=2,
    )

    fig.update_geos(
        scope="south america",
        lataxis_range=[-19.5, 0.0],
        lonaxis_range=[-83.5, -67.5],
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="rgb(245, 243, 238)",
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Importancia promedio del factor (SHAP)", row=1, col=1)
    fig.update_layout(
        height=700,
        title=dict(
            text="Pronóstico Probabilístico por Zonas en el Perú (Corte 2026-11-01)",
            y=0.97,
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=290, r=90, t=95, b=55),
        template="plotly_white",
        showlegend=False,
    )
    fig.write_html(OUT_SHAP_HTML, include_plotlyjs="cdn")
    return shap_df.head(15).to_dict("records")


def main() -> None:
    """Execute Model C Walk-Forward evaluation, final operational forecast, and SHAP export."""
    assert PANEL_PARQUET.exists(), f"Missing {PANEL_PARQUET}"
    panel = pd.read_parquet(PANEL_PARQUET)
    panel["is_normal_sur"] = (panel["regimen_subduccion_lat"] == "normal_sur").astype("int8")

    # 1. Walk-Forward Validation across 3 targets
    res_m45_90d = _evaluate_target_walk_forward(panel, "target_m45_90d", 90, SPLITS_90D)
    res_m45_365d = _evaluate_target_walk_forward(panel, "target_m45_365d", 365, SPLITS_365D)
    res_m50_90d = _evaluate_target_walk_forward(panel, "target_m50_90d", 90, SPLITS_90D)

    # 2. Final Operational Forecast on latest cut (`2026-11-01`)
    latest_cut = panel["cut_date_utc"].max()
    latest_df = panel[panel["cut_date_utc"] == latest_cut].copy().reset_index(drop=True)
    assert len(latest_df) == 8205, f"Expected 8205 cells at {latest_cut}, got {len(latest_df)}"

    p_m45_90d, model_m45_90d, X_latest_df, meta_90d = _train_final_and_forecast_latest(panel, "target_m45_90d", 90, latest_cut)
    p_m45_365d, model_m45_365d, _, meta_365d = _train_final_and_forecast_latest(panel, "target_m45_365d", 365, latest_cut)
    p_m50_90d, model_m50_90d, _, meta_m50 = _train_final_and_forecast_latest(panel, "target_m50_90d", 90, latest_cut)

    forecast_out = latest_df[
        [
            "h3_res5",
            "cut_date_utc",
            "cell_lat",
            "cell_lon",
            "dist_trench_km",
            "slab_depth_est_km",
            "convergence_vel_mm_yr",
            "regimen_subduccion_lat",
            "n_geocatmin_peligros",
            "n_geocatmin_zonas_criticas",
            "cnt_m45_90d",
            "cnt_m45_365d",
            "etas_decay_nbhd_365d",
            "days_since_last_m50_nbhd",
        ]
    ].copy()
    forecast_out["prob_m45_90d"] = p_m45_90d.astype("float64")
    forecast_out["prob_m45_365d"] = p_m45_365d.astype("float64")
    forecast_out["prob_m50_90d"] = p_m50_90d.astype("float64")

    assert forecast_out[["prob_m45_90d", "prob_m45_365d", "prob_m50_90d"]].notna().all().all()
    assert (forecast_out["prob_m45_90d"].between(0.0, 1.0)).all()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    forecast_out.to_parquet(OUT_FORECAST_LATEST, engine="pyarrow", compression="snappy", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_m45_90d": model_m45_90d,
            "model_m45_365d": model_m45_365d,
            "model_m50_90d": model_m50_90d,
            "meta_m45_90d": meta_90d,
            "meta_m45_365d": meta_365d,
            "meta_m50_90d": meta_m50,
            "feature_names": ML_BASE_FEATURES + ["poisson_bg_rate_causal"],
        },
        OUT_MODEL_JOBLIB,
    )

    top15_shap = _generate_shap_and_forecast_html(model_m45_90d, X_latest_df, forecast_out)
    assert OUT_SHAP_HTML.exists() and OUT_FORECAST_LATEST.exists() and OUT_MODEL_JOBLIB.exists()

    summary = {
        "latest_cut_utc": latest_cut.isoformat(),
        "n_forecast_cells": int(len(forecast_out)),
        "forecast_summary_2026_11_01": {
            "prob_m45_90d_mean_pct": round(float(forecast_out["prob_m45_90d"].mean() * 100.0), 4),
            "prob_m45_90d_max_pct": round(float(forecast_out["prob_m45_90d"].max() * 100.0), 2),
            "prob_m45_365d_mean_pct": round(float(forecast_out["prob_m45_365d"].mean() * 100.0), 4),
            "prob_m45_365d_max_pct": round(float(forecast_out["prob_m45_365d"].max() * 100.0), 2),
            "prob_m50_90d_mean_pct": round(float(forecast_out["prob_m50_90d"].mean() * 100.0), 4),
            "prob_m50_90d_max_pct": round(float(forecast_out["prob_m50_90d"].max() * 100.0), 2),
        },
        "results_target_m45_90d": res_m45_90d,
        "results_target_m45_365d": res_m45_365d,
        "results_target_m50_90d": res_m50_90d,
        "top15_shap_features_m45_90d": top15_shap,
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== MODELO C (PRONOSTICO ESPACIO-TEMPORAL H3 WALK-FORWARD) COMPLETADO ===")
    for label, res in [
        ("target_m45_90d (4-split mean)", res_m45_90d),
        ("target_m45_365d (4-split mean)", res_m45_365d),
        ("target_m50_90d (4-split mean)", res_m50_90d),
    ]:
        print(f"--- {label} ---")
        for m_name, mv in res["mean_4splits"].items():
            print(
                f"  {m_name:20s} | PR-AUC={mv['pr_auc']:.4f} | ROC-AUC={mv['roc_auc']:.4f} | LogLoss={mv['log_loss']:.5f} | Brier={mv['brier_score']:.6f} | BSS={mv['bss_vs_poisson']:+.4f}"
            )
    print(f"Pronostico vigente ({latest_cut.date()}): {len(forecast_out)} celdas guardadas en {OUT_FORECAST_LATEST}")


if __name__ == "__main__":
    main()
