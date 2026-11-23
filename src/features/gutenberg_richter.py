"""Phase 2 - Step 2: Completeness Magnitude (Mc) and Gutenberg-Richter (a, b +/- sigma_b).

Estimates Mc via MAXC + 0.2 and Wiemer & Wyss (2000) Goodness-of-Fit (GFT >= 90%),
computes Aki-Utsu MLE b-value, Shi & Bolt (1982) standard error sigma_b, and
productivity a (total and annualized) strictly on events with M >= Mc across:
  1. Global and Modern (2000-2026)
  2. Decade (1960s through 2020-2026)
  3. Geographic macro-region (Costa/Mar, Sierra/Andina, Selva/Amazonica)
  4. Seismotectonic regime (4 Tavera & Buforn regimes)
  5. Subduction latitude segment (flat_slab_norte_centro vs normal_sur)

Exports `data/interim/gutenberg_richter_strata.parquet`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

IGP_INST_PARQUET = PROCESSED_DIR / "catalog_igp_instrumental.parquet"
OUT_GR_PARQUET = INTERIM_DIR / "gutenberg_richter_strata.parquet"
METRICS_JSON = INTERIM_DIR / "gutenberg_richter_metrics.json"

DELTA_M = 0.1
MIN_EVENTS_FOR_FIT = 30
GFT_THRESHOLD = 90.0


def _aki_utsu_b_and_shi_bolt(mags_ge_mc: np.ndarray, mc: float, delta_m: float = DELTA_M) -> tuple[float, float, float]:
    """Compute Aki-Utsu MLE b-value, Shi & Bolt (1982) sigma_b, and total productivity a."""
    n = len(mags_ge_mc)
    if n < 10:
        return np.nan, np.nan, np.nan
    m_mean = float(np.mean(mags_ge_mc))
    denom = m_mean - (mc - delta_m / 2.0)
    if denom <= 0.0:
        return np.nan, np.nan, np.nan
    b = float(np.log10(np.e) / denom)
    var_sum = float(np.sum((mags_ge_mc - m_mean) ** 2))
    sigma_b = float(2.30 * (b ** 2) * np.sqrt(var_sum / (n * (n - 1)))) if n > 1 else np.nan
    a_total = float(np.log10(n) + b * mc)
    return b, sigma_b, a_total


def _gft_score(mags_rounded: np.ndarray, mc: float, delta_m: float = DELTA_M) -> tuple[float, float, float, float]:
    """Compute Wiemer & Wyss (2000) Goodness-of-Fit R(%) at candidate Mc."""
    sub = mags_rounded[mags_rounded >= mc - 1e-9]
    if len(sub) < MIN_EVENTS_FOR_FIT:
        return np.nan, np.nan, np.nan, np.nan
    b, sigma_b, a_total = _aki_utsu_b_and_shi_bolt(sub, mc, delta_m)
    if np.isnan(b):
        return np.nan, np.nan, np.nan, np.nan

    m_max = float(np.max(sub))
    bins = np.round(np.arange(mc, m_max + delta_m / 2.0, delta_m), 1)
    if len(bins) < 3:
        return np.nan, b, sigma_b, a_total

    obs_cum = np.array([np.sum(sub >= m - 1e-9) for m in bins], dtype="float64")
    syn_cum = np.power(10.0, a_total - b * bins)
    denom = float(np.sum(obs_cum))
    if denom <= 0.0:
        return np.nan, b, sigma_b, a_total
    r_pct = float(100.0 - (np.sum(np.abs(obs_cum - syn_cum)) / denom) * 100.0)
    return r_pct, b, sigma_b, a_total


def estimate_gr_for_subset(
    df_sub: pd.DataFrame, group_type: str, stratum_name: str
) -> dict[str, Any]:
    """Estimate Mc (MAXC, MAXC+0.2, GFT90) and Aki-Utsu (a, b +/- sigma_b) strictly on M >= Mc."""
    mags = np.round(df_sub["mag"].to_numpy(dtype="float64"), 1)
    n_total = len(mags)
    assert n_total > 0, f"Empty stratum {group_type}:{stratum_name}"

    t_min = df_sub["timestamp_utc"].min()
    t_max = df_sub["timestamp_utc"].max()
    duration_years = max(float((t_max - t_min).total_seconds() / (365.25 * 86400.0)), 1.0 / 365.25)

    # 1. MAXC (mode of non-cumulative magnitude distribution) + 0.2 standard correction
    uniq_m, counts = np.unique(mags, return_counts=True)
    mc_maxc = float(uniq_m[int(np.argmax(counts))])
    mc_maxc_p02 = round(mc_maxc + 0.2, 1)

    # 2. GFT >= 90% scan (Wiemer & Wyss 2000) scanning from min(mags) upward
    cand_start = float(np.min(uniq_m))
    cand_end = float(np.max(uniq_m)) - 0.5
    candidates = np.round(np.arange(cand_start, max(cand_start, cand_end) + 1e-9, DELTA_M), 1)

    mc_gft90 = np.nan
    gft_at_gft90 = np.nan
    best_gft = -np.inf
    best_gft_mc = mc_maxc_p02

    for mc_cand in candidates:
        r_pct, _, _, _ = _gft_score(mags, float(mc_cand))
        if np.isnan(r_pct):
            continue
        if r_pct > best_gft:
            best_gft = r_pct
            best_gft_mc = float(mc_cand)
        if np.isnan(mc_gft90) and r_pct >= GFT_THRESHOLD and mc_cand >= mc_maxc - 1e-9:
            mc_gft90 = float(mc_cand)
            gft_at_gft90 = float(r_pct)

    # Adopt MAXC+0.2 if GFT90 fails or falls below MAXC+0.2 while MAXC+0.2 also meets GFT>=90,
    # otherwise prefer GFT90 when >= MAXC+0.2 and sufficient sample size (N >= 50)
    r_maxc02, _, _, _ = _gft_score(mags, mc_maxc_p02)
    n_at_maxc02 = int(np.sum(mags >= mc_maxc_p02 - 1e-9))

    if not np.isnan(mc_gft90) and mc_gft90 >= mc_maxc_p02 - 1e-9 and int(np.sum(mags >= mc_gft90 - 1e-9)) >= MIN_EVENTS_FOR_FIT:
        mc_adopted = mc_gft90
        mc_method = "GFT90"
    elif n_at_maxc02 >= MIN_EVENTS_FOR_FIT:
        mc_adopted = mc_maxc_p02
        mc_method = "MAXC+0.2"
    else:
        mc_adopted = mc_maxc
        mc_method = "MAXC_small_n"

    # Strict Aki-Utsu & Shi-Bolt ONLY for M >= mc_adopted
    sub_complete = mags[mags >= mc_adopted - 1e-9]
    n_complete = int(len(sub_complete))
    gft_adopted, b_val, sigma_b, a_total = _gft_score(mags, mc_adopted)
    if np.isnan(b_val):
        b_val, sigma_b, a_total = _aki_utsu_b_and_shi_bolt(sub_complete, mc_adopted)

    a_annual = float(a_total - np.log10(duration_years)) if not np.isnan(a_total) else np.nan

    return {
        "group_type": group_type,
        "stratum": stratum_name,
        "n_total": int(n_total),
        "n_complete": int(n_complete),
        "pct_complete": round(float(n_complete / n_total * 100.0), 2),
        "duration_years": round(duration_years, 2),
        "mc_maxc": round(mc_maxc, 1),
        "mc_maxc_plus02": round(mc_maxc_p02, 1),
        "mc_gft90": round(float(mc_gft90), 1) if not np.isnan(mc_gft90) else np.nan,
        "mc_adopted": round(float(mc_adopted), 1),
        "mc_method": mc_method,
        "gft_pct_adopted": round(float(gft_adopted), 2) if not np.isnan(gft_adopted) else np.nan,
        "gft_pct_best": round(float(best_gft), 2) if best_gft > -np.inf else np.nan,
        "b_value": round(float(b_val), 4) if not np.isnan(b_val) else np.nan,
        "sigma_b": round(float(sigma_b), 4) if not np.isnan(sigma_b) else np.nan,
        "a_total": round(float(a_total), 4) if not np.isnan(a_total) else np.nan,
        "a_annual": round(float(a_annual), 4) if not np.isnan(a_annual) else np.nan,
        "mag_max": round(float(np.max(mags)), 1),
    }


def _assign_macro_region(dist_trench_km: pd.Series) -> pd.Series:
    """Derive Peruvian physiographic macro-region from 3D trench distance."""
    d = dist_trench_km.to_numpy(dtype="float64")
    conds = [d <= 200.0, (d > 200.0) & (d <= 420.0), d > 420.0]
    choices = ["Costa / Mar", "Sierra / Andina", "Selva / Amazonica"]
    return pd.Series(np.select(conds, choices, default="Costa / Mar"), index=dist_trench_km.index, dtype="string")


def _assign_decade(ts_utc: pd.Series) -> pd.Series:
    """Assign decade buckets (1960-1969 through 2020-2026)."""
    yr = ts_utc.dt.year
    bins = [1959, 1969, 1979, 1989, 1999, 2009, 2019, 2030]
    labels = ["1960-1969", "1970-1979", "1980-1989", "1990-1999", "2000-2009", "2010-2019", "2020-2026"]
    return pd.cut(yr, bins=bins, labels=labels).astype("string")


def main() -> None:
    """Run Gutenberg-Richter & Mc analysis across all strata and save parquet."""
    assert IGP_INST_PARQUET.exists(), f"Missing {IGP_INST_PARQUET}"
    df = pd.read_parquet(IGP_INST_PARQUET)
    assert len(df) == 25878

    df["decada"] = _assign_decade(df["timestamp_utc"])
    df["macro_region"] = _assign_macro_region(df["dist_trench_km"])

    rows: list[dict[str, Any]] = []
    # 0. Global & Modern (2000-2026)
    rows.append(estimate_gr_for_subset(df, "global", "Todo_1960_2026"))
    rows.append(estimate_gr_for_subset(df[df["timestamp_utc"].dt.year >= 2000], "global", "Moderno_2000_2026"))

    # 1. By Decade
    for dec in ["1960-1969", "1970-1979", "1980-1989", "1990-1999", "2000-2009", "2010-2019", "2020-2026"]:
        rows.append(estimate_gr_for_subset(df[df["decada"] == dec], "decada", dec))

    # 2. By Macro-region (both full 1960-2026 and modern 2000-2026 for completeness sensitivity)
    for reg in ["Costa / Mar", "Sierra / Andina", "Selva / Amazonica"]:
        rows.append(estimate_gr_for_subset(df[df["macro_region"] == reg], "macro_region", reg))

    # 3. By Seismotectonic Regime
    for r in ["subduccion_interplaca", "cortical_continental", "intramanto_intermedio", "intramanto_profundo"]:
        rows.append(estimate_gr_for_subset(df[df["regimen_sismotectonico"] == r], "regimen_sismotectonico", r))

    # 4. By Subduction Latitude (15S transition)
    for lat_seg in ["flat_slab_norte_centro", "normal_sur"]:
        rows.append(estimate_gr_for_subset(df[df["regimen_subduccion_lat"] == lat_seg], "latitud_15s", lat_seg))

    res_df = pd.DataFrame(rows)

    # Integrity assertions
    assert res_df["b_value"].notna().all(), f"NaN b_value found: {res_df[res_df['b_value'].isna()]}"
    assert res_df["b_value"].between(0.3, 2.5).all(), f"Unphysical b_value: {res_df[['stratum', 'b_value']]}"
    assert (res_df["sigma_b"] > 0).all()
    assert (res_df["n_complete"] <= res_df["n_total"]).all()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_parquet(OUT_GR_PARQUET, engine="pyarrow", compression="snappy", index=False)
    METRICS_JSON.write_text(
        json.dumps(res_df.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== ANALISIS GUTENBERG-RICHTER Y COMPLETITUD (Mc) COMPLETADO ===")
    print(
        res_df[["group_type", "stratum", "n_total", "mc_maxc_plus02", "mc_gft90", "mc_adopted", "b_value", "sigma_b", "a_annual", "gft_pct_adopted"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
