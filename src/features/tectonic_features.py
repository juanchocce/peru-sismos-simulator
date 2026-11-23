"""Phase 1 - Step 5: 3D Seismotectonic Feature Engineering and Phase 1 Consolidation.

Builds a densified (<= 2 km geodesic spacing) 3D unit-sphere cKDTree (R = 6371.0 km)
of the continuous Nazca subduction trench (`NZ\\SA`, `NZ\\AP`, `NZ\\ND` in PB2002),
computes great-circle distance to the trench, transfers PB2002_steps geophysical
parameters (convergence velocity/azimuth, seafloor age), classifies Peru's 4
seismotectonic regimes (Tavera & Buforn) and the ~15S Nazca Ridge subduction
transition, assigns Uber H3 indices (res 6 and 7), merges USGS enrichment, and
exports consolidated Phase 1 datasets to `data/processed/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TECTONIC_ROOT = PROJECT_ROOT / "data" / "raw" / "tectonic_plates"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INTERIM_INST = INTERIM_DIR / "catalog_igp_instrumental_base.parquet"
INTERIM_HIST = INTERIM_DIR / "catalog_igp_historico.parquet"
INTERIM_ACELDAT = INTERIM_DIR / "aceldat_merged_base.parquet"
INTERIM_USGS_ENRICH = INTERIM_DIR / "igp_usgs_enrichment.parquet"

OUT_INST = PROCESSED_DIR / "catalog_igp_instrumental.parquet"
OUT_HIST = PROCESSED_DIR / "catalog_igp_historico.parquet"
OUT_ACELDAT = PROCESSED_DIR / "aceldat_merged.parquet"
OUT_DISTRITOS = PROCESSED_DIR / "distritos_maestro_ubigeo.parquet"
METRICS_JSON = INTERIM_DIR / "tectonic_features_metrics.json"

EARTH_RADIUS_KM = 6371.0
MAX_STEP_KM = 2.0
DEG_LAT_TO_KM = 111.1949266  # 2 * pi * 6371 / 360
NAZCA_TRENCH_BOUNDS = {"NZ\\SA", "NZ\\AP", "NZ\\ND"}
INTERPLATE_MAX_DIST_KM = 180.0
SHALLOW_MAX_DEPTH_KM = 60.0
INTERMEDIATE_MAX_DEPTH_KM = 350.0
NAZCA_RIDGE_LAT = -15.0


def _resolve_geojson_dir() -> Path:
    """Locate PB2002 GeoJSON directory regardless of zip nesting."""
    candidates = [
        RAW_TECTONIC_ROOT / "tectonicplates-master" / "GeoJSON",
        RAW_TECTONIC_ROOT / "GeoJSON",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Could not find PB2002 GeoJSON folder under {RAW_TECTONIC_ROOT}")


def _latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert WGS84 lat/lon (degrees) to 3D Cartesian coordinates on unit sphere."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def _densify_line_coords(
    coords: list[tuple[float, float]], geod: Geod, max_step_km: float = MAX_STEP_KM
) -> list[tuple[float, float]]:
    """Densify a sequence of (lon, lat) vertices so consecutive points are <= `max_step_km` apart."""
    out: list[tuple[float, float]] = []
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i][0], coords[i][1]
        lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
        _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
        dist_km = dist_m / 1000.0
        out.append((lon1, lat1))
        if dist_km > max_step_km:
            n_intermediate = int(np.ceil(dist_km / max_step_km)) - 1
            if n_intermediate > 0:
                pts = geod.npts(lon1, lat1, lon2, lat2, n_intermediate)
                out.extend(pts)
    if coords:
        out.append((coords[-1][0], coords[-1][1]))
    return out


def _build_trench_indices(geojson_dir: Path) -> tuple[cKDTree, cKDTree, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Densify Nazca subduction boundaries and PB2002_steps (<= 2 km) and build 3D cKDTrees."""
    geod = Geod(ellps="WGS84")

    # 1. Boundaries tree (`NZ\SA` + continuous Peru segments `NZ\AP` & `NZ\ND`)
    gdf_bound = gpd.read_file(geojson_dir / "PB2002_boundaries.json")
    trench_bound = gdf_bound[gdf_bound["Name"].isin(NAZCA_TRENCH_BOUNDS)].copy()
    assert len(trench_bound) >= 2, f"Expected >=2 trench boundaries, found {len(trench_bound)}"

    bound_pts: list[tuple[float, float]] = []
    for geom in trench_bound.geometry:
        if geom.geom_type == "LineString":
            bound_pts.extend(_densify_line_coords(list(geom.coords), geod))
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                bound_pts.extend(_densify_line_coords(list(part.coords), geod))

    bound_arr = np.asarray(bound_pts, dtype="float64")  # (lon, lat)
    bound_tree = cKDTree(_latlon_to_unit_xyz(bound_arr[:, 1], bound_arr[:, 0]))

    # 2. Steps tree with geophysical attributes per segment
    gdf_steps = gpd.read_file(geojson_dir / "PB2002_steps.json")
    trench_steps = gdf_steps[gdf_steps["PLATEBOUND"].isin(NAZCA_TRENCH_BOUNDS)].copy().reset_index(drop=True)
    assert len(trench_steps) >= 60, f"Expected >=60 trench steps, found {len(trench_steps)}"

    step_pts: list[tuple[float, float]] = []
    step_meta: list[dict[str, Any]] = []
    for _, row in trench_steps.iterrows():
        geom = row.geometry
        coords = list(geom.coords) if geom.geom_type == "LineString" else list(geom.geoms[0].coords)
        dense = _densify_line_coords(coords, geod)
        meta = {
            "trench_segment_bound": str(row["PLATEBOUND"]),
            "convergence_vel_mm_yr": float(row["VELOCITYLE"]),
            "convergence_az_deg": float(row["VELOCITYAZ"]),
            "seafloor_age_myr": float(row["SEAFLOORAG"]),
            "trench_elevation_m": float(row["ELEVATION"]),
        }
        for lon, lat in dense:
            step_pts.append((lon, lat))
            step_meta.append(meta)

    step_arr = np.asarray(step_pts, dtype="float64")
    steps_tree = cKDTree(_latlon_to_unit_xyz(step_arr[:, 1], step_arr[:, 0]))
    steps_meta_df = pd.DataFrame(step_meta)

    # 3. Plates & Orogens
    gdf_plates = gpd.read_file(geojson_dir / "PB2002_plates.json")[["Code", "PlateName", "geometry"]].copy()
    gdf_orogens = gpd.read_file(geojson_dir / "PB2002_orogens.json")
    peru_orogen = gdf_orogens[gdf_orogens["Name"] == "Peru"][["Name", "geometry"]].copy()
    assert len(peru_orogen) == 1, "Peru orogen polygon not found in PB2002_orogens.json"

    return bound_tree, steps_tree, steps_meta_df, gdf_plates, peru_orogen


def _compute_h3_series(lat_s: pd.Series, lon_s: pd.Series, res: int) -> pd.Series:
    """Vector-style assignment of H3 cell IDs in WGS84."""
    return pd.Series(
        [h3.latlng_to_cell(float(la), float(lo), res) for la, lo in zip(lat_s.to_numpy(), lon_s.to_numpy())],
        index=lat_s.index,
        dtype="string",
    )


def _spatial_plate_and_orogen(
    lat_s: pd.Series, lon_s: pd.Series, gdf_plates: gpd.GeoDataFrame, peru_orogen: gpd.GeoDataFrame
) -> tuple[pd.Series, pd.Series]:
    """Assign tectonic `plate_code` and boolean `in_peru_orogen` via point-in-polygon."""
    pts = gpd.GeoDataFrame(
        {"_row_idx": np.arange(len(lat_s))},
        geometry=gpd.points_from_xy(lon_s.to_numpy(), lat_s.to_numpy()),
        crs="EPSG:4326",
    )
    joined_plate = gpd.sjoin(pts, gdf_plates[["Code", "geometry"]], how="left", predicate="intersects")
    joined_plate = joined_plate.drop_duplicates(subset=["_row_idx"], keep="first").set_index("_row_idx")
    plate_code = joined_plate.reindex(range(len(lat_s)))["Code"].fillna("UNK").astype("string").reset_index(drop=True)
    plate_code.index = lat_s.index

    joined_oro = gpd.sjoin(pts, peru_orogen[["Name", "geometry"]], how="left", predicate="intersects")
    joined_oro = joined_oro.drop_duplicates(subset=["_row_idx"], keep="first").set_index("_row_idx")
    in_orogen = joined_oro.reindex(range(len(lat_s)))["Name"].notna().astype("bool").reset_index(drop=True)
    in_orogen.index = lat_s.index
    return plate_code, in_orogen


def _classify_seismotectonic_regime(depth_km: pd.Series, dist_trench_km: pd.Series) -> pd.Series:
    """Classify events into Peru's 4 seismotectonic regimes (Tavera & Buforn).

    1. subduccion_interplaca: depth <= 60 km and dist_trench_km <= 180 km
    2. cortical_continental:  depth <= 60 km and dist_trench_km > 180 km
    3. intramanto_intermedio: 60 < depth <= 350 km
    4. intramanto_profundo:   depth > 350 km
    """
    d = depth_km.to_numpy(dtype="float64")
    t = dist_trench_km.to_numpy(dtype="float64")

    conds = [
        np.isnan(d),
        (d <= SHALLOW_MAX_DEPTH_KM) & (t <= INTERPLATE_MAX_DIST_KM),
        (d <= SHALLOW_MAX_DEPTH_KM) & (t > INTERPLATE_MAX_DIST_KM),
        (d > SHALLOW_MAX_DEPTH_KM) & (d <= INTERMEDIATE_MAX_DEPTH_KM),
        d > INTERMEDIATE_MAX_DEPTH_KM,
    ]
    choices = [
        "indeterminado_sin_profundidad",
        "subduccion_interplaca",
        "cortical_continental",
        "intramanto_intermedio",
        "intramanto_profundo",
    ]
    return pd.Series(np.select(conds, choices, default="indeterminado_sin_profundidad"), index=depth_km.index, dtype="string")


def _attach_tectonic_features(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    depth_col: str | None,
    bound_tree: cKDTree,
    steps_tree: cKDTree,
    steps_meta_df: pd.DataFrame,
    gdf_plates: gpd.GeoDataFrame,
    peru_orogen: gpd.GeoDataFrame,
    prefix: str = "",
) -> pd.DataFrame:
    """Compute 3D trench distance, PB2002_steps parameters, regimes, 15S transition, and H3."""
    out = df.copy()
    lats = out[lat_col].to_numpy(dtype="float64")
    lons = out[lon_col].to_numpy(dtype="float64")
    xyz = _latlon_to_unit_xyz(lats, lons)

    d_chord, _ = bound_tree.query(xyz, k=1)
    dist_trench_km = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(d_chord / 2.0, 0.0, 1.0))

    _, step_idx = steps_tree.query(xyz, k=1)
    nearest_meta = steps_meta_df.iloc[step_idx].reset_index(drop=True)

    out[f"{prefix}dist_trench_km"] = dist_trench_km.astype("float64")
    out[f"{prefix}trench_segment_bound"] = nearest_meta["trench_segment_bound"].astype("string").to_numpy()
    out[f"{prefix}convergence_vel_mm_yr"] = nearest_meta["convergence_vel_mm_yr"].astype("float64").to_numpy()
    out[f"{prefix}convergence_az_deg"] = nearest_meta["convergence_az_deg"].astype("float64").to_numpy()
    out[f"{prefix}seafloor_age_myr"] = nearest_meta["seafloor_age_myr"].astype("float64").to_numpy()
    out[f"{prefix}trench_elevation_m"] = nearest_meta["trench_elevation_m"].astype("float64").to_numpy()

    plate_code, in_orogen = _spatial_plate_and_orogen(out[lat_col], out[lon_col], gdf_plates, peru_orogen)
    out[f"{prefix}plate_code"] = plate_code
    out[f"{prefix}in_peru_orogen"] = in_orogen

    out[f"{prefix}regimen_subduccion_lat"] = pd.Series(
        np.where(lats > NAZCA_RIDGE_LAT, "flat_slab_norte_centro", "normal_sur"),
        index=out.index,
        dtype="string",
    )
    out[f"{prefix}dist_lat_15s_km"] = ((lats - NAZCA_RIDGE_LAT) * DEG_LAT_TO_KM).astype("float64")

    if depth_col is not None:
        out[f"{prefix}regimen_sismotectonico"] = _classify_seismotectonic_regime(
            out[depth_col], out[f"{prefix}dist_trench_km"]
        )

    out[f"{prefix}h3_res6"] = _compute_h3_series(out[lat_col], out[lon_col], 6)
    out[f"{prefix}h3_res7"] = _compute_h3_series(out[lat_col], out[lon_col], 7)
    return out


def main() -> None:
    """Execute 3D tectonic feature engineering and export consolidated Phase 1 tables."""
    for p in (INTERIM_INST, INTERIM_HIST, INTERIM_ACELDAT, INTERIM_USGS_ENRICH, OUT_DISTRITOS):
        assert p.exists(), f"Missing required prerequisite file: {p}"

    geojson_dir = _resolve_geojson_dir()
    bound_tree, steps_tree, steps_meta_df, gdf_plates, peru_orogen = _build_trench_indices(geojson_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Instrumental catalog (+ USGS enrichment + 3D tectonic features + H3)
    df_inst = pd.read_parquet(INTERIM_INST)
    df_usgs_enrich = pd.read_parquet(INTERIM_USGS_ENRICH)
    df_inst = df_inst.merge(df_usgs_enrich, on="event_id", how="left")
    df_inst["has_moment_tensor"] = df_inst["has_moment_tensor"].fillna(False).astype("bool")
    df_inst = _attach_tectonic_features(
        df_inst,
        lat_col="lat",
        lon_col="lon",
        depth_col="depth_km",
        bound_tree=bound_tree,
        steps_tree=steps_tree,
        steps_meta_df=steps_meta_df,
        gdf_plates=gdf_plates,
        peru_orogen=peru_orogen,
    )
    assert len(df_inst) == 25878
    assert df_inst["event_id"].is_unique and df_inst["dist_trench_km"].notna().all()
    assert (df_inst["dist_trench_km"] >= 0).all()
    assert set(df_inst["regimen_sismotectonico"].unique()) == {
        "subduccion_interplaca",
        "cortical_continental",
        "intramanto_intermedio",
        "intramanto_profundo",
    }
    assert str(df_inst["timestamp_utc"].dtype) == "datetime64[us, UTC]"
    df_inst.to_parquet(OUT_INST, engine="pyarrow", compression="snappy", index=False)

    # 2. Historical catalog (+ 3D tectonic features + H3)
    df_hist = pd.read_parquet(INTERIM_HIST)
    df_hist = _attach_tectonic_features(
        df_hist,
        lat_col="lat",
        lon_col="lon",
        depth_col="depth_km",
        bound_tree=bound_tree,
        steps_tree=steps_tree,
        steps_meta_df=steps_meta_df,
        gdf_plates=gdf_plates,
        peru_orogen=peru_orogen,
    )
    assert len(df_hist) == 1098
    assert df_hist["event_id"].is_unique and df_hist["dist_trench_km"].notna().all()
    assert str(df_hist["timestamp_utc"].dtype) == "datetime64[us, UTC]"
    df_hist.to_parquet(OUT_HIST, engine="pyarrow", compression="snappy", index=False)

    # 3. ACELDAT merged (+ station tectonic/H3 + event tectonic/USGS from processed instrumental)
    df_ac = pd.read_parquet(INTERIM_ACELDAT)
    df_ac = _attach_tectonic_features(
        df_ac,
        lat_col="station_lat",
        lon_col="station_lon",
        depth_col=None,
        bound_tree=bound_tree,
        steps_tree=steps_tree,
        steps_meta_df=steps_meta_df,
        gdf_plates=gdf_plates,
        peru_orogen=peru_orogen,
        prefix="station_",
    )
    # Also attach event-level tectonic regime, trench distance, and USGS fields via event_id
    event_cols = [
        "event_id",
        "dist_trench_km",
        "trench_segment_bound",
        "convergence_vel_mm_yr",
        "convergence_az_deg",
        "seafloor_age_myr",
        "plate_code",
        "in_peru_orogen",
        "regimen_sismotectonico",
        "regimen_subduccion_lat",
        "dist_lat_15s_km",
        "h3_res6",
        "h3_res7",
        "usgs_id",
        "usgs_mag",
        "usgs_mag_type",
        "azimuthal_gap",
        "rms",
        "mmi_shakemap",
        "pager_alert",
        "has_moment_tensor",
    ]
    df_ac = df_ac.merge(
        df_inst[event_cols].rename(
            columns={
                "dist_trench_km": "event_dist_trench_km",
                "h3_res6": "event_h3_res6",
                "h3_res7": "event_h3_res7",
            }
        ),
        on="event_id",
        how="left",
    )
    # Canonical `h3_res6` / `h3_res7` on ACELDAT point to station location for spatial aggregation
    df_ac["h3_res6"] = df_ac["station_h3_res6"]
    df_ac["h3_res7"] = df_ac["station_h3_res7"]
    assert df_ac["event_id"].notna().all() and df_ac["regimen_sismotectonico"].notna().all()
    assert (df_ac["ubigeo"].str.len() == 6).all()
    assert str(df_ac["timestamp_utc"].dtype) == "datetime64[us, UTC]"
    df_ac.to_parquet(OUT_ACELDAT, engine="pyarrow", compression="snappy", index=False)

    # 4. Enrich distritos_maestro_ubigeo.parquet with approximate trench distance where mean lat/lon exists
    df_dist = pd.read_parquet(OUT_DISTRITOS)
    has_coords = df_dist["geocatmin_lat_mean"].notna() & df_dist["geocatmin_lon_mean"].notna()
    df_dist["dist_trench_km_approx"] = np.nan
    df_dist["regimen_subduccion_lat_approx"] = pd.Series(pd.NA, index=df_dist.index, dtype="string")
    if has_coords.any():
        xyz_d = _latlon_to_unit_xyz(
            df_dist.loc[has_coords, "geocatmin_lat_mean"].to_numpy(),
            df_dist.loc[has_coords, "geocatmin_lon_mean"].to_numpy(),
        )
        d_ch, _ = bound_tree.query(xyz_d, k=1)
        df_dist.loc[has_coords, "dist_trench_km_approx"] = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(d_ch / 2.0, 0.0, 1.0))
        df_dist.loc[has_coords, "regimen_subduccion_lat_approx"] = np.where(
            df_dist.loc[has_coords, "geocatmin_lat_mean"] > NAZCA_RIDGE_LAT,
            "flat_slab_norte_centro",
            "normal_sur",
        )
    assert len(df_dist) == 1893 and df_dist["ubigeo"].is_unique
    df_dist.to_parquet(OUT_DISTRITOS, engine="pyarrow", compression="snappy", index=False)

    metrics = {
        "trench_boundaries_included": sorted(NAZCA_TRENCH_BOUNDS),
        "densified_boundary_points": int(bound_tree.n),
        "densified_steps_points": int(steps_tree.n),
        "instrumental_rows": int(len(df_inst)),
        "instrumental_regimes": df_inst["regimen_sismotectonico"].value_counts().to_dict(),
        "instrumental_subduction_lat": df_inst["regimen_subduccion_lat"].value_counts().to_dict(),
        "instrumental_regime_x_lat": (
            df_inst.groupby(["regimen_sismotectonico", "regimen_subduccion_lat"]).size().unstack(fill_value=0).to_dict()
        ),
        "instrumental_trench_bound_counts": df_inst["trench_segment_bound"].value_counts().to_dict(),
        "instrumental_dist_trench_median_km": round(float(df_inst["dist_trench_km"].median()), 2),
        "instrumental_dist_trench_max_km": round(float(df_inst["dist_trench_km"].max()), 2),
        "historico_rows": int(len(df_hist)),
        "historico_regimes": df_hist["regimen_sismotectonico"].value_counts().to_dict(),
        "aceldat_rows": int(len(df_ac)),
        "aceldat_regimes": df_ac["regimen_sismotectonico"].value_counts().to_dict(),
        "aceldat_unique_events": int(df_ac["event_id"].nunique()),
        "aceldat_usgs_enriched_rows": int(df_ac["usgs_id"].notna().sum()),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== FEATURES SISMOTECTONICAS 3D Y CONSOLIDACION COMPLETADAS ===")
    print(f"Fosa densificada (<=2 km): {metrics['densified_boundary_points']} nodos ({metrics['trench_boundaries_included']})")
    print(f"Instrumental ({len(df_inst)}): regimenes = {metrics['instrumental_regimes']}")
    print(f"Instrumental latitud 15S: {metrics['instrumental_subduction_lat']}")
    print(f"ACELDAT consolidado ({len(df_ac)} filas, {metrics['aceldat_unique_events']} eventos): regimenes = {metrics['aceldat_regimes']}")


if __name__ == "__main__":
    main()
