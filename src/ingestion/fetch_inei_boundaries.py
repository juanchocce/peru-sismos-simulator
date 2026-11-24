"""Phase 5C: Download and Cache Official INEI Administrative Boundaries (WFS).

Queries official INEI GeoServer WFS layers (Departamentos, Provincias, Distritos),
normalizes official territorial keys (`ubigeo` zfill(6), `ccdd` zfill(2)), ensures
WGS84 (EPSG:4326), simplifies geometries topologically (`0.002` deg) for sub-second
interactive map rendering, and caches lightweight GeoJSON files under `data/interim/`:
  - `data/interim/inei_departamentos.geojson`
  - `data/interim/inei_provincias.geojson`
  - `data/interim/inei_distritos.geojson`
  - `data/interim/inei_boundaries_metrics.json`

If any WFS endpoint fails due to network/timeout errors, automatically builds a
clean fallback GeoJSON from territorial centroids / buffered envelopes (and Peru
country boundary if available in `data/raw/geocatmin/`) without failing the pipeline.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UBIGEO_PARQUET = PROCESSED_DIR / "territorial_multihazard_ubigeo.parquet"

SIMPLIFY_TOLERANCE_DEG = 0.002
REQUEST_TIMEOUT_SEC = 45
MAX_RETRIES = 3

WFS_LAYERS: dict[str, dict[str, str]] = {
    "departamentos": {
        "url": (
            "https://geoespacial.inei.gob.pe/geoserver/Interoperabilidad/ows"
            "?service=WFS&version=1.0.0&request=GetFeature"
            "&typeName=Interoperabilidad%3Aig_departamento"
            "&maxFeatures=100&outputFormat=application%2Fjson"
        ),
        "output": "inei_departamentos.geojson",
    },
    "provincias": {
        "url": (
            "https://geoespacial.inei.gob.pe/geoserver/Interoperabilidad/ows"
            "?service=WFS&version=1.0.0&request=GetFeature"
            "&typeName=Interoperabilidad%3Aig_provincia"
            "&maxFeatures=500&outputFormat=application%2Fjson"
        ),
        "output": "inei_provincias.geojson",
    },
    "distritos": {
        "url": (
            "https://geoespacial.inei.gob.pe/geoserver/Interoperabilidad/ows"
            "?service=WFS&version=1.0.0&request=GetFeature"
            "&typeName=Interoperabilidad%3Aig_distrito"
            "&maxFeatures=2500&outputFormat=application%2Fjson"
        ),
        "output": "inei_distritos.geojson",
    },
}


def _find_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _clean_digits(series: pd.Series, width: int) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(width)
    )


def _fetch_geojson_with_retries(url: str, layer_name: str) -> tuple[dict[str, Any] | None, str]:
    headers = {
        "User-Agent": "PeruSismosSimulator/5C (CivicTech Geospatial Platform)",
        "Accept": "application/json",
    }
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[INEI WFS] Descargando {layer_name} (intento {attempt}/{MAX_RETRIES})...")
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
                n_feat = len(payload.get("features", []))
                if n_feat > 0:
                    print(f"[INEI WFS] OK {layer_name}: {n_feat} features recibidos.")
                    return payload, "ok"
                last_err = "FeatureCollection vacio"
            else:
                last_err = f"Respuesta inesperada (keys={list(payload.keys())[:5] if isinstance(payload, dict) else type(payload)})"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            print(f"[INEI WFS] Fallo intento {attempt} en {layer_name}: {last_err}")
        if attempt < MAX_RETRIES:
            time.sleep(2.0 * attempt)
    return None, last_err


def _normalize_wfs_gdf(gdf: gpd.GeoDataFrame, level: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    cols = [c for c in gdf.columns if c != "geometry"]

    ccdd_col = _find_col(cols, ("ccdd", "cod_dep", "id_dpto", "coddpto"))
    ccpp_col = _find_col(cols, ("ccpp", "cod_prov", "id_prov"))
    ccdi_col = _find_col(cols, ("ccdi", "cod_dist", "id_dist"))
    ubigeo_col = _find_col(cols, ("ubigeo", "iddist", "cd_dist", "cod_ubigeo", "ubigeo_reniec"))
    dep_name_col = _find_col(cols, ("nombdep", "departamento", "nom_dep", "departam", "dpto"))
    prov_name_col = _find_col(cols, ("nombprov", "provincia", "nom_prov"))
    dist_name_col = _find_col(cols, ("nombdist", "distrito", "nom_dist"))

    out = pd.DataFrame(index=gdf.index)

    if ubigeo_col is not None:
        out["ubigeo"] = _clean_digits(gdf[ubigeo_col], 6)
    elif ccdd_col is not None and ccpp_col is not None and ccdi_col is not None:
        out["ubigeo"] = (
            _clean_digits(gdf[ccdd_col], 2)
            + _clean_digits(gdf[ccpp_col], 2)
            + _clean_digits(gdf[ccdi_col], 2)
        )

    if ccdd_col is not None:
        out["ccdd"] = _clean_digits(gdf[ccdd_col], 2)
    elif "ubigeo" in out.columns:
        out["ccdd"] = out["ubigeo"].str.slice(0, 2)
    else:
        out["ccdd"] = ""

    if ccpp_col is not None:
        raw_ccpp = _clean_digits(gdf[ccpp_col], 2)
        out["ccpp"] = raw_ccpp.str.slice(-2)
        out["cod_prov"] = out["ccdd"] + out["ccpp"]
    elif "ubigeo" in out.columns:
        out["ccpp"] = out["ubigeo"].str.slice(2, 4)
        out["cod_prov"] = out["ubigeo"].str.slice(0, 4)

    if dep_name_col is not None:
        out["departamento"] = gdf[dep_name_col].astype(str).str.strip().str.upper()
    if prov_name_col is not None:
        out["provincia"] = gdf[prov_name_col].astype(str).str.strip().str.upper()
    if dist_name_col is not None:
        out["distrito"] = gdf[dist_name_col].astype(str).str.strip().str.upper()

    if level == "distritos" and "ubigeo" in out.columns:
        assert (out["ubigeo"].str.len() == 6).all(), "UBIGEO debe tener exactamente 6 digitos"
    if "ccdd" in out.columns and (out["ccdd"] != "").all():
        assert (out["ccdd"].str.len() == 2).all(), "CCDD debe tener exactamente 2 digitos"

    geom = gdf.geometry.buffer(0)
    geom = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    res = gpd.GeoDataFrame(out, geometry=geom, crs="EPSG:4326")
    res = res[res.geometry.notna() & (~res.geometry.is_empty)].reset_index(drop=True)
    res["source_layer"] = f"INEI_WFS_{level}"
    return res


def _build_fallback_gdf(level: str) -> gpd.GeoDataFrame:
    """Build deterministic geometric fallback from processed UBIGEO centroids if WFS is unreachable."""
    df_u = pd.read_parquet(UBIGEO_PARQUET).copy()
    df_u["ubigeo"] = df_u["ubigeo"].astype(str).str.zfill(6)
    df_u["ccdd"] = df_u["ubigeo"].str.slice(0, 2)
    df_u["ccpp"] = df_u["ubigeo"].str.slice(2, 4)
    df_u["cod_prov"] = df_u["ubigeo"].str.slice(0, 4)

    if level == "distritos":
        geoms = [Point(lon, lat).buffer(0.06, resolution=8) for lon, lat in zip(df_u["dist_lon"], df_u["dist_lat"])]
        res = gpd.GeoDataFrame(
            df_u[["ubigeo", "ccdd", "ccpp", "cod_prov", "departamento", "provincia", "distrito"]],
            geometry=geoms,
            crs="EPSG:4326",
        )
    elif level == "provincias":
        rows, geoms = [], []
        for (ccdd, cod_prov, dep, prov), grp in df_u.groupby(["ccdd", "cod_prov", "departamento", "provincia"]):
            pts = [Point(lo, la).buffer(0.12, resolution=8) for lo, la in zip(grp["dist_lon"], grp["dist_lat"])]
            poly = unary_union(pts).convex_hull.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
            rows.append({"ccdd": ccdd, "ccpp": cod_prov[2:], "cod_prov": cod_prov, "departamento": dep, "provincia": prov})
            geoms.append(poly)
        res = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    else:
        rows, geoms = [], []
        for (ccdd, dep), grp in df_u.groupby(["ccdd", "departamento"]):
            pts = [Point(lo, la).buffer(0.18, resolution=8) for lo, la in zip(grp["dist_lon"], grp["dist_lat"])]
            poly = unary_union(pts).convex_hull.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
            rows.append({"ccdd": ccdd, "departamento": dep})
            geoms.append(poly)
        res = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")

    res["source_layer"] = f"FALLBACK_CENTROIDS_{level}"
    return res


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "simplify_tolerance_deg": SIMPLIFY_TOLERANCE_DEG,
        "layers": {},
    }

    for level, cfg in WFS_LAYERS.items():
        out_path = INTERIM_DIR / cfg["output"]
        payload, status_msg = _fetch_geojson_with_retries(cfg["url"], level)

        if payload is not None:
            raw_gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
            raw_cols = [c for c in raw_gdf.columns if c != "geometry"]
            norm_gdf = _normalize_wfs_gdf(raw_gdf, level)
            source_mode = "INEI_WFS"
        else:
            print(f"[INEI WFS] Activando respaldo local para {level} (motivo: {status_msg})")
            raw_cols = []
            norm_gdf = _build_fallback_gdf(level)
            source_mode = "FALLBACK_LOCAL"

        norm_gdf.to_file(out_path, driver="GeoJSON")
        size_mb = round(out_path.stat().st_size / (1024 * 1024), 3)
        summary["layers"][level] = {
            "source_mode": source_mode,
            "status": status_msg,
            "feature_count": int(len(norm_gdf)),
            "raw_columns": raw_cols,
            "normalized_columns": [c for c in norm_gdf.columns if c != "geometry"],
            "output_path": str(out_path.relative_to(PROJECT_ROOT)),
            "file_size_mb": size_mb,
        }
        print(
            f"[INEI] {level.upper():13s} | modo={source_mode:14s} | features={len(norm_gdf):4d} "
            f"| size={size_mb:.3f} MB -> {out_path.name}"
        )

    metrics_path = INTERIM_DIR / "inei_boundaries_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INEI] Resumen guardado en {metrics_path}")


if __name__ == "__main__":
    main()
