"""Phase 3 - Model A: Machine Learning GMPE & Empirical Station Site Amplification (delta_S2S).

Trains and evaluates Ground Motion Prediction Equations (GMPEs) on
`data/processed/aceldat_merged.parquet` (17,794 station-event records, 956 unique
events) under strict anti-leakage validation:
  - 5-fold GroupKFold strictly grouped by `event_id` (zero event leakage).
  - Out-of-time temporal holdout: Train 2019-2024 -> Test 2025.
  - Any station site term (delta_S2S) is estimated STRICTLY inside each training fold
    via nested inner GroupKFold + Empirical Bayes shrinkage.

Compares 3 modeling tiers:
  1. Classical Empirical Attenuation Baseline (Huber linear model).
  2. Nationwide Generalizable ML GMPE without station ID (LightGBM & XGBoost).
  3. Site-Aware ML GMPE with OOF Empirical Bayes station term (LightGBM & XGBoost).

Exports:
  - `data/processed/aceldat_station_site_factors.parquet` (276 stations rel. to MSOL)
  - `data/interim/gmpe_oof_predictions.parquet`
  - `models/gmpe_lgbm.joblib`
  - `reports/figures/gmpe_shap_attenuation.html`
  - `data/interim/gmpe_metrics.json`
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
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

ACELDAT_PARQUET = PROCESSED_DIR / "aceldat_merged.parquet"
DISTRITOS_PARQUET = PROCESSED_DIR / "distritos_maestro_ubigeo.parquet"

OUT_SITE_FACTORS = PROCESSED_DIR / "aceldat_station_site_factors.parquet"
OUT_OOF_PREDS = INTERIM_DIR / "gmpe_oof_predictions.parquet"
OUT_MODEL_JOBLIB = MODELS_DIR / "gmpe_lgbm.joblib"
OUT_SHAP_HTML = FIGURES_DIR / "gmpe_shap_attenuation.html"
METRICS_JSON = INTERIM_DIR / "gmpe_metrics.json"

H0_SATURATION_KM = 10.0
SHRINKAGE_LAMBDA = 5.0
N_SPLITS = 5
RANDOM_STATE = 42


def _prepare_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load ACELDAT, join GEOCATMIN district proxies, and build physical feature matrix."""
    df = pd.read_parquet(ACELDAT_PARQUET)
    dist = pd.read_parquet(
        DISTRITOS_PARQUET,
        columns=[
            "ubigeo",
            "n_peligros_total",
            "n_peligros_flujo",
            "n_peligros_caida",
            "n_peligros_deslizamiento",
            "n_zonas_criticas_total",
        ],
    )
    df = df.merge(dist, on="ubigeo", how="left")
    geo_cols = [
        "n_peligros_total",
        "n_peligros_flujo",
        "n_peligros_caida",
        "n_peligros_deslizamiento",
        "n_zonas_criticas_total",
    ]
    for c in geo_cols:
        df[c] = df[c].fillna(0).astype("float64")

    # Fill exact geometric station features for any newly recovered 2025 station
    if df["station_dist_lat_15s_km"].isna().any():
        miss = df["station_dist_lat_15s_km"].isna()
        df.loc[miss, "station_dist_lat_15s_km"] = (df.loc[miss, "station_lat"] + 15.0) * 111.1949266
    if df["station_dist_trench_km"].isna().any():
        import geopandas as gpd
        from scipy.spatial import cKDTree

        g_dir = PROJECT_ROOT / "data" / "raw" / "tectonic_plates" / "tectonicplates-master" / "GeoJSON"
        tb = gpd.read_file(g_dir / "PB2002_boundaries.json")
        tb = tb[tb["Name"].isin({"NZ\\SA", "NZ\\AP", "NZ\\ND"})]
        t_pts = np.asarray([c[:2] for geom in tb.geometry for c in geom.coords], dtype="float64")
        t_lat, t_lon = np.radians(t_pts[:, 1]), np.radians(t_pts[:, 0])
        t_xyz = np.column_stack((np.cos(t_lat) * np.cos(t_lon), np.cos(t_lat) * np.sin(t_lon), np.sin(t_lat)))
        tree = cKDTree(t_xyz)
        miss = df["station_dist_trench_km"].isna()
        s_lat = np.radians(df.loc[miss, "station_lat"].to_numpy())
        s_lon = np.radians(df.loc[miss, "station_lon"].to_numpy())
        s_xyz = np.column_stack((np.cos(s_lat) * np.cos(s_lon), np.cos(s_lat) * np.sin(s_lon), np.sin(s_lat)))
        d_ch, _ = tree.query(s_xyz, k=1)
        df.loc[miss, "station_dist_trench_km"] = 2.0 * 6371.0 * np.arcsin(np.clip(d_ch / 2.0, 0.0, 1.0))

    df["log10_pga_v"] = np.log10(df["pga_v"].to_numpy(dtype="float64"))
    df["log10_r_hyp"] = np.log10(np.sqrt(df["dist_hypocentral_km"] ** 2 + H0_SATURATION_KM ** 2))
    df["log10_r_epi"] = np.log10(np.sqrt(df["dist_epicentral_km"] ** 2 + H0_SATURATION_KM ** 2))
    df["sin_azimuth"] = np.sin(np.radians(df["azimuth_deg"]))
    df["cos_azimuth"] = np.cos(np.radians(df["azimuth_deg"]))
    df["is_normal_sur"] = (df["regimen_subduccion_lat"] == "normal_sur").astype("float64")
    df["year"] = df["timestamp_utc"].dt.year.astype("int32")

    # One-hot encode the 4 seismotectonic regimes deterministically
    regimes = [
        "subduccion_interplaca",
        "cortical_continental",
        "intramanto_intermedio",
        "intramanto_profundo",
    ]
    for r in regimes:
        df[f"reg_{r}"] = (df["regimen_sismotectonico"] == r).astype("float64")
    regime_cols = [f"reg_{r}" for r in regimes]

    # Tier 1: Parametric empirical attenuation features
    baseline_cols = ["event_mag", "log10_r_hyp", "dist_hypocentral_km", "event_depth_km"] + regime_cols

    # Tier 2: Generalizable nationwide physical + tectonic + GEOCATMIN features (NO station_lat/lon/ID)
    ml_no_site_cols = [
        "event_mag",
        "event_depth_km",
        "dist_epicentral_km",
        "dist_hypocentral_km",
        "log10_r_hyp",
        "log10_r_epi",
        "azimuth_deg",
        "sin_azimuth",
        "cos_azimuth",
        "event_dist_trench_km",
        "station_dist_trench_km",
        "event_slab_depth_est_km",
        "event_dist_to_slab_km",
        "dist_lat_15s_km",
        "station_dist_lat_15s_km",
        "convergence_vel_mm_yr",
        "seafloor_age_myr",
        "is_normal_sur",
    ] + regime_cols + geo_cols

    assert df["event_id"].notna().all()
    assert df[ml_no_site_cols].notna().all().all(), f"NaNs in ML features: {df[ml_no_site_cols].isna().sum().to_dict()}"
    assert np.isfinite(df["log10_pga_h_geom"]).all() and np.isfinite(df["log10_pga_v"]).all()
    return df, baseline_cols, ml_no_site_cols


def _eval_gmpe_metrics(y_true_log10: np.ndarray, y_pred_log10: np.ndarray, event_ids: np.ndarray) -> dict[str, float]:
    """Compute log10 RMSE, MAE, R2, between-event tau, within-event phi, and linear-scale error metrics."""
    rmse = float(root_mean_squared_error(y_true_log10, y_pred_log10))
    mae = float(mean_absolute_error(y_true_log10, y_pred_log10))
    r2 = float(r2_score(y_true_log10, y_pred_log10))

    # Residual decomposition: total = between-event (tau) + within-event (phi)
    res_df = pd.DataFrame({"event_id": event_ids, "res": y_true_log10 - y_pred_log10})
    ev_means = res_df.groupby("event_id")["res"].transform("mean")
    tau = float(res_df.groupby("event_id")["res"].mean().std(ddof=1))
    phi = float((res_df["res"] - ev_means).std(ddof=1))

    # Linear scale (Gal): median absolute percentage error & geometric factor
    y_true_gal = np.power(10.0, y_true_log10)
    y_pred_gal = np.power(10.0, y_pred_log10)
    rel_err_pct = np.abs(y_pred_gal - y_true_gal) / np.maximum(y_true_gal, 1e-6) * 100.0
    mdape_pct = float(np.median(rel_err_pct))
    mae_gal = float(mean_absolute_error(y_true_gal, y_pred_gal))
    geom_factor_mae = float(np.power(10.0, mae))

    return {
        "rmse_log10": round(rmse, 4),
        "mae_log10": round(mae, 4),
        "r2_log10": round(r2, 4),
        "tau_between_event": round(tau, 4),
        "phi_within_event": round(phi, 4),
        "mdape_linear_pct": round(mdape_pct, 2),
        "geom_factor_mae": round(geom_factor_mae, 3),
        "mae_gal": round(mae_gal, 4),
    }


def _make_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def _make_xgb() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def _estimate_station_site_terms_train_only(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute leak-free inner-OOF station Empirical Bayes site terms strictly inside a training split."""
    gkf_inner = GroupKFold(n_splits=4)
    X_tr = df_train[feature_cols].to_numpy(dtype="float64")
    y_tr = df_train[target_col].to_numpy(dtype="float64")
    groups_tr = df_train["event_id"].to_numpy()
    stations_tr = df_train["cod_estacion"].to_numpy()

    # Nested OOF residuals inside train fold to prevent memorizing train residuals
    inner_oof_pred = np.zeros(len(df_train), dtype="float64")
    for itr, iva in gkf_inner.split(X_tr, y_tr, groups_tr):
        m = _make_lgbm()
        m.fit(X_tr[itr], y_tr[itr])
        inner_oof_pred[iva] = m.predict(X_tr[iva])

    inner_res = y_tr - inner_oof_pred

    # Out-of-fold station encoding for train rows (computed from other inner folds)
    train_site_feature = np.zeros(len(df_train), dtype="float64")
    for itr, iva in gkf_inner.split(X_tr, y_tr, groups_tr):
        s_df = pd.DataFrame({"st": stations_tr[itr], "r": inner_res[itr]})
        agg = s_df.groupby("st")["r"].agg(["count", "mean"])
        shrunk = (agg["count"] / (agg["count"] + SHRINKAGE_LAMBDA)) * agg["mean"]
        st_map = shrunk.to_dict()
        train_site_feature[iva] = [st_map.get(s, 0.0) for s in stations_tr[iva]]

    # Full train-fold station lookup table for mapping onto the held-out validation/test fold
    full_s_df = pd.DataFrame({"st": stations_tr, "r": inner_res})
    full_agg = full_s_df.groupby("st")["r"].agg(["count", "mean"])
    full_shrunk = ((full_agg["count"] / (full_agg["count"] + SHRINKAGE_LAMBDA)) * full_agg["mean"]).to_dict()
    return train_site_feature, full_shrunk


def _run_group_cv_suite(
    df: pd.DataFrame,
    baseline_cols: list[str],
    ml_cols: list[str],
    target_col: str,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Run 5-fold GroupKFold (by event_id) for all 3 GMPE tiers."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = df["event_id"].to_numpy()
    y = df[target_col].to_numpy(dtype="float64")
    X_base = df[baseline_cols].to_numpy(dtype="float64")
    X_ml = df[ml_cols].to_numpy(dtype="float64")

    oof_preds: dict[str, np.ndarray] = {
        "baseline_huber": np.zeros(len(df), dtype="float64"),
        "lgbm_no_site": np.zeros(len(df), dtype="float64"),
        "xgb_no_site": np.zeros(len(df), dtype="float64"),
        "lgbm_with_site_oof": np.zeros(len(df), dtype="float64"),
        "xgb_with_site_oof": np.zeros(len(df), dtype="float64"),
    }

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_ml, y, groups), start=1):
        # Anti-leakage assertion: ZERO shared event_id between train and val
        assert len(set(groups[tr_idx]).intersection(set(groups[va_idx]))) == 0, f"Event leakage in fold {fold}!"

        # Tier 1: Parametric empirical attenuation baseline (HuberRegressor)
        scaler = StandardScaler()
        Xb_tr = scaler.fit_transform(X_base[tr_idx])
        Xb_va = scaler.transform(X_base[va_idx])
        huber = HuberRegressor(max_iter=500, alpha=1.0)
        huber.fit(Xb_tr, y[tr_idx])
        oof_preds["baseline_huber"][va_idx] = huber.predict(Xb_va)

        # Tier 2: Nationwide ML GMPE (no station ID)
        m_lgb = _make_lgbm()
        m_lgb.fit(X_ml[tr_idx], y[tr_idx])
        oof_preds["lgbm_no_site"][va_idx] = m_lgb.predict(X_ml[va_idx])

        m_xgb = _make_xgb()
        m_xgb.fit(X_ml[tr_idx], y[tr_idx])
        oof_preds["xgb_no_site"][va_idx] = m_xgb.predict(X_ml[va_idx])

        # Tier 3: ML GMPE + strictly train-only Empirical Bayes station site term
        tr_site_feat, st_lookup = _estimate_station_site_terms_train_only(
            df.iloc[tr_idx], ml_cols, target_col
        )
        va_site_feat = np.array(
            [st_lookup.get(s, 0.0) for s in df.iloc[va_idx]["cod_estacion"].to_numpy()],
            dtype="float64",
        )
        X_ml_tr_site = np.column_stack((X_ml[tr_idx], tr_site_feat))
        X_ml_va_site = np.column_stack((X_ml[va_idx], va_site_feat))

        m_lgb_s = _make_lgbm()
        m_lgb_s.fit(X_ml_tr_site, y[tr_idx])
        oof_preds["lgbm_with_site_oof"][va_idx] = m_lgb_s.predict(X_ml_va_site)

        m_xgb_s = _make_xgb()
        m_xgb_s.fit(X_ml_tr_site, y[tr_idx])
        oof_preds["xgb_with_site_oof"][va_idx] = m_xgb_s.predict(X_ml_va_site)

    metrics = {
        name: _eval_gmpe_metrics(y, pred, groups)
        for name, pred in oof_preds.items()
    }
    return metrics, oof_preds


def _run_temporal_holdout_2025(
    df: pd.DataFrame,
    baseline_cols: list[str],
    ml_cols: list[str],
    target_col: str,
) -> dict[str, dict[str, float]]:
    """Strict out-of-time evaluation: Train 2019-2024 -> Evaluate 2025."""
    tr_mask = df["year"] <= 2024
    te_mask = df["year"] == 2025
    df_tr, df_te = df.loc[tr_mask].copy(), df.loc[te_mask].copy()
    assert len(df_tr) > 10000 and len(df_te) > 2000
    assert len(set(df_tr["event_id"]).intersection(set(df_te["event_id"]))) == 0, "Event overlap in 2025 holdout!"

    y_tr = df_tr[target_col].to_numpy(dtype="float64")
    y_te = df_te[target_col].to_numpy(dtype="float64")
    ev_te = df_te["event_id"].to_numpy()

    results: dict[str, dict[str, float]] = {}

    # Tier 1
    scaler = StandardScaler()
    huber = HuberRegressor(max_iter=500, alpha=1.0)
    huber.fit(scaler.fit_transform(df_tr[baseline_cols].to_numpy()), y_tr)
    results["baseline_huber"] = _eval_gmpe_metrics(y_te, huber.predict(scaler.transform(df_te[baseline_cols].to_numpy())), ev_te)

    # Tier 2
    X_tr = df_tr[ml_cols].to_numpy(dtype="float64")
    X_te = df_te[ml_cols].to_numpy(dtype="float64")

    m_lgb = _make_lgbm()
    m_lgb.fit(X_tr, y_tr)
    results["lgbm_no_site"] = _eval_gmpe_metrics(y_te, m_lgb.predict(X_te), ev_te)

    m_xgb = _make_xgb()
    m_xgb.fit(X_tr, y_tr)
    results["xgb_no_site"] = _eval_gmpe_metrics(y_te, m_xgb.predict(X_te), ev_te)

    # Tier 3 (station site terms estimated strictly from 2019-2024)
    tr_site_feat, st_lookup = _estimate_station_site_terms_train_only(df_tr, ml_cols, target_col)
    te_site_feat = np.array([st_lookup.get(s, 0.0) for s in df_te["cod_estacion"].to_numpy()], dtype="float64")

    m_lgb_s = _make_lgbm()
    m_lgb_s.fit(np.column_stack((X_tr, tr_site_feat)), y_tr)
    results["lgbm_with_site_oof"] = _eval_gmpe_metrics(y_te, m_lgb_s.predict(np.column_stack((X_te, te_site_feat))), ev_te)

    m_xgb_s = _make_xgb()
    m_xgb_s.fit(np.column_stack((X_tr, tr_site_feat)), y_tr)
    results["xgb_with_site_oof"] = _eval_gmpe_metrics(y_te, m_xgb_s.predict(np.column_stack((X_te, te_site_feat))), ev_te)
    return results


def _extract_station_site_factors(
    df: pd.DataFrame, oof_h_no_site: np.ndarray, oof_v_no_site: np.ndarray
) -> pd.DataFrame:
    """Compute regularized delta_S2S and amplification factor relative to rock reference MSOL."""
    tmp = pd.DataFrame(
        {
            "cod_estacion": df["cod_estacion"],
            "ubigeo": df["ubigeo"],
            "departamento": df["departamento"],
            "provincia": df["provincia"],
            "distrito": df["distrito"],
            "station_lat": df["station_lat"],
            "station_lon": df["station_lon"],
            "station_h3_res6": df["station_h3_res6"],
            "station_h3_res7": df["station_h3_res7"],
            "res_h": df["log10_pga_h_geom"].to_numpy() - oof_h_no_site,
            "res_v": df["log10_pga_v"].to_numpy() - oof_v_no_site,
        }
    )
    agg = (
        tmp.groupby("cod_estacion", as_index=False)
        .agg(
            n_records=("res_h", "size"),
            ubigeo=("ubigeo", "first"),
            departamento=("departamento", "first"),
            provincia=("provincia", "first"),
            distrito=("distrito", "first"),
            station_lat=("station_lat", "mean"),
            station_lon=("station_lon", "mean"),
            station_h3_res6=("station_h3_res6", "first"),
            station_h3_res7=("station_h3_res7", "first"),
            res_h_mean=("res_h", "mean"),
            res_h_std=("res_h", "std"),
            res_v_mean=("res_v", "mean"),
        )
    )
    shrink = agg["n_records"] / (agg["n_records"] + SHRINKAGE_LAMBDA)
    agg["delta_s2s_log10"] = (shrink * agg["res_h_mean"]).astype("float64")
    agg["delta_s2s_v_log10"] = (shrink * agg["res_v_mean"]).astype("float64")
    agg["amp_factor_abs"] = np.power(10.0, agg["delta_s2s_log10"])

    msol_row = agg[agg["cod_estacion"] == "MSOL"]
    assert len(msol_row) == 1, "Reference rock station MSOL (Morro Solar) not found!"
    msol_delta = float(msol_row["delta_s2s_log10"].iloc[0])

    agg["delta_s2s_rel_msol"] = (agg["delta_s2s_log10"] - msol_delta).astype("float64")
    agg["amp_factor_rel_msol"] = np.power(10.0, agg["delta_s2s_rel_msol"]).astype("float64")
    agg = agg.sort_values("amp_factor_rel_msol", ascending=False).reset_index(drop=True)

    assert len(agg) == df["cod_estacion"].nunique()
    assert abs(float(agg.loc[agg["cod_estacion"] == "MSOL", "amp_factor_rel_msol"].iloc[0]) - 1.0) < 1e-9
    return agg


def _generate_shap_and_attenuation_html(
    final_lgbm_no_site: lgb.LGBMRegressor,
    final_lgbm_with_site: lgb.LGBMRegressor,
    df: pd.DataFrame,
    ml_cols: list[str],
    site_cols: list[str],
    site_terms_all: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute TreeSHAP on site-aware GMPE and plot SHAP + physical attenuation curves."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_full_site = np.column_stack((df[ml_cols].to_numpy(dtype="float64"), site_terms_all))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(df), size=min(2500, len(df)), replace=False)
    X_sample = pd.DataFrame(X_full_site[sample_idx], columns=site_cols)

    explainer = shap.TreeExplainer(final_lgbm_with_site)
    shap_vals = explainer.shap_values(X_sample)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = (
        pd.DataFrame({"feature": site_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top15 = shap_df.head(15).iloc[::-1]

    # Build synthetic physical attenuation curves (Lima interplaca depth=35 km, lat=-12.0, dist_trench=120 km)
    r_epi_grid = np.geomspace(15.0, 600.0, 80)
    depth_syn = 35.0
    r_hyp_grid = np.sqrt(r_epi_grid ** 2 + depth_syn ** 2)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Importancia SHAP (Top 15 Features — GMPE LightGBM + Sitio)",
            "Curvas de Atenuacion GMPE (Interplaca, Prof=35 km, Roca Ref.)",
        ),
        horizontal_spacing=0.14,
    )

    fig.add_trace(
        go.Bar(
            x=top15["mean_abs_shap"],
            y=top15["feature"],
            orientation="h",
            marker_color="#2a9d8f",
            name="Mean |SHAP|",
        ),
        row=1,
        col=1,
    )

    mag_colors = {5.0: "#457b9d", 6.0: "#2a9d8f", 7.0: "#e9c46a", 8.0: "#e63946"}
    for mag_val, col in mag_colors.items():
        syn = pd.DataFrame(0.0, index=range(len(r_epi_grid)), columns=ml_cols)
        syn["event_mag"] = mag_val
        syn["event_depth_km"] = depth_syn
        syn["dist_epicentral_km"] = r_epi_grid
        syn["dist_hypocentral_km"] = r_hyp_grid
        syn["log10_r_hyp"] = np.log10(np.sqrt(r_hyp_grid ** 2 + H0_SATURATION_KM ** 2))
        syn["log10_r_epi"] = np.log10(np.sqrt(r_epi_grid ** 2 + H0_SATURATION_KM ** 2))
        syn["azimuth_deg"] = 90.0
        syn["sin_azimuth"] = 1.0
        syn["cos_azimuth"] = 0.0
        syn["event_dist_trench_km"] = 120.0
        syn["station_dist_trench_km"] = 180.0
        syn["event_slab_depth_est_km"] = 45.0
        syn["event_dist_to_slab_km"] = 10.0
        syn["dist_lat_15s_km"] = 333.6
        syn["station_dist_lat_15s_km"] = 333.6
        syn["convergence_vel_mm_yr"] = 67.0
        syn["seafloor_age_myr"] = 42.0
        syn["is_normal_sur"] = 0.0
        syn["reg_subduccion_interplaca"] = 1.0
        pga_pred = np.power(10.0, final_lgbm_no_site.predict(syn.to_numpy(dtype="float64")))
        fig.add_trace(
            go.Scatter(
                x=r_hyp_grid,
                y=pga_pred,
                mode="lines",
                name=f"Mw = {mag_val:.1f} (Interplaca)",
                line=dict(color=col, width=3),
            ),
            row=1,
            col=2,
        )

    # Overlay observed ACELDAT interplaca M=5.0..6.5 scatter sample
    obs_sub = df[(df["regimen_sismotectonico"] == "subduccion_interplaca") & (df["event_mag"].between(5.0, 6.5))]
    if len(obs_sub) > 1200:
        obs_sub = obs_sub.sample(1200, random_state=RANDOM_STATE)
    fig.add_trace(
        go.Scattergl(
            x=obs_sub["dist_hypocentral_km"],
            y=obs_sub["pga_h_geom"],
            mode="markers",
            name="ACELDAT obs. (M 5.0-6.5)",
            marker=dict(size=4, color="rgba(100,100,100,0.28)"),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Mean |SHAP| en log10(PGA)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Distancia Hipocentral R_hyp (km)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="PGA Horizontal Geom. (Gal = cm/s²)", row=1, col=2)
    fig.update_layout(
        height=620,
        title_text="Modelo A (GMPE Peru): Explicabilidad SHAP y Curvas de Atenuacion Sismica",
        template="plotly_white",
    )
    fig.write_html(OUT_SHAP_HTML, include_plotlyjs="cdn")
    return shap_df.head(15).to_dict("records")


def main() -> None:
    """Execute Model A (GMPE) training, validation, site factor extraction, and SHAP export."""
    assert ACELDAT_PARQUET.exists() and DISTRITOS_PARQUET.exists()
    df, baseline_cols, ml_cols = _prepare_dataset()

    # Primary target: log10_pga_h_geom
    cv_metrics_h, oof_preds_h = _run_group_cv_suite(df, baseline_cols, ml_cols, "log10_pga_h_geom")
    oot_metrics_h = _run_temporal_holdout_2025(df, baseline_cols, ml_cols, "log10_pga_h_geom")

    # Secondary target: log10_pga_v
    cv_metrics_v, oof_preds_v = _run_group_cv_suite(df, baseline_cols, ml_cols, "log10_pga_v")
    oot_metrics_v = _run_temporal_holdout_2025(df, baseline_cols, ml_cols, "log10_pga_v")

    # Extract regularized station site factors (relative to MSOL)
    site_df = _extract_station_site_factors(df, oof_preds_h["lgbm_no_site"], oof_preds_v["lgbm_no_site"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    site_df.to_parquet(OUT_SITE_FACTORS, engine="pyarrow", compression="snappy", index=False)

    # Save OOF predictions
    oof_out = df[
        [
            "event_id",
            "timestamp_utc",
            "cod_estacion",
            "ubigeo",
            "event_mag",
            "dist_hypocentral_km",
            "regimen_sismotectonico",
            "log10_pga_h_geom",
            "log10_pga_v",
        ]
    ].copy()
    for name, pred in oof_preds_h.items():
        oof_out[f"pred_h_{name}"] = pred
    oof_out["pred_v_lgbm_no_site"] = oof_preds_v["lgbm_no_site"]
    oof_out["pred_v_lgbm_with_site_oof"] = oof_preds_v["lgbm_with_site_oof"]
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    oof_out.to_parquet(OUT_OOF_PREDS, engine="pyarrow", compression="snappy", index=False)

    # Train final production LightGBM models (nationwide + site-aware) and persist joblib
    X_all = df[ml_cols].to_numpy(dtype="float64")
    y_all_h = df["log10_pga_h_geom"].to_numpy(dtype="float64")
    final_lgbm_no_site = _make_lgbm()
    final_lgbm_no_site.fit(X_all, y_all_h)

    all_site_feat, full_site_lookup = _estimate_station_site_terms_train_only(df, ml_cols, "log10_pga_h_geom")
    site_cols = ml_cols + ["station_site_term_oof"]
    final_lgbm_with_site = _make_lgbm()
    final_lgbm_with_site.fit(np.column_stack((X_all, all_site_feat)), y_all_h)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_no_site": final_lgbm_no_site,
            "model_with_site": final_lgbm_with_site,
            "ml_features_no_site": ml_cols,
            "ml_features_with_site": site_cols,
            "station_site_lookup": full_site_lookup,
            "h0_saturation_km": H0_SATURATION_KM,
            "shrinkage_lambda": SHRINKAGE_LAMBDA,
        },
        OUT_MODEL_JOBLIB,
    )

    top_shap = _generate_shap_and_attenuation_html(
        final_lgbm_no_site, final_lgbm_with_site, df, ml_cols, site_cols, all_site_feat
    )
    assert OUT_SHAP_HTML.exists() and OUT_MODEL_JOBLIB.exists() and OUT_SITE_FACTORS.exists()

    # Check key Lima reference stations
    lima_bench = site_df[site_df["cod_estacion"].isin(["MSOL", "RINC", "OLIV", "PAVE"])][
        ["cod_estacion", "distrito", "n_records", "delta_s2s_log10", "delta_s2s_rel_msol", "amp_factor_rel_msol"]
    ].to_dict("records")
    # Top 10 high-amplification stations with >= 10 records (statistically stable)
    top10_amp = site_df[site_df["n_records"] >= 10].head(10)[
        ["cod_estacion", "departamento", "distrito", "n_records", "delta_s2s_log10", "amp_factor_rel_msol"]
    ].to_dict("records")

    metrics = {
        "n_rows": int(len(df)),
        "n_events": int(df["event_id"].nunique()),
        "n_stations": int(df["cod_estacion"].nunique()),
        "n_train_2019_2024": int((df["year"] <= 2024).sum()),
        "n_test_2025": int((df["year"] == 2025).sum()),
        "cv_5fold_group_h": cv_metrics_h,
        "holdout_2025_h": oot_metrics_h,
        "cv_5fold_group_v": cv_metrics_v,
        "holdout_2025_v": oot_metrics_v,
        "lima_benchmark_stations": lima_bench,
        "top10_amplification_stations_n10": top10_amp,
        "top15_shap_features": top_shap,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sanity assertions: site-aware ML must beat empirical baseline on GroupKFold RMSE
    assert cv_metrics_h["lgbm_with_site_oof"]["rmse_log10"] < cv_metrics_h["lgbm_no_site"]["rmse_log10"]
    assert cv_metrics_h["lgbm_no_site"]["rmse_log10"] < cv_metrics_h["baseline_huber"]["rmse_log10"]

    print("=== MODELO A (GMPE ML + SITIO OOF) COMPLETADO ===")
    print("GroupKFold (5 folds por event_id) — Target: log10_pga_h_geom")
    for k, v in cv_metrics_h.items():
        print(f"  {k:20s} | RMSE={v['rmse_log10']:.4f} | MAE={v['mae_log10']:.4f} | R2={v['r2_log10']:.4f} | tau={v['tau_between_event']:.3f} | phi={v['phi_within_event']:.3f} | MdAPE={v['mdape_linear_pct']:.1f}%")
    print("Holdout Out-of-Time 2025 (Train 2019-2024 -> Test 2025):")
    for k, v in oot_metrics_h.items():
        print(f"  {k:20s} | RMSE={v['rmse_log10']:.4f} | MAE={v['mae_log10']:.4f} | R2={v['r2_log10']:.4f} | MdAPE={v['mdape_linear_pct']:.1f}%")
    print("Estaciones referencia Lima vs MSOL (Roca):")
    for r in lima_bench:
        print(f"  {r['cod_estacion']:5s} ({r['distrito']:18s}, n={r['n_records']:3d}): delta_S2S={r['delta_s2s_log10']:+.3f} | Amp rel. MSOL = {r['amp_factor_rel_msol']:.2f}x")


if __name__ == "__main__":
    main()
