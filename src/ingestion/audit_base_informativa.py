"""Deep read-only audit of `base_informativa/` for Phase 0 data governance.

Inspects files in `base_informativa/` without modifying any source files, extracts
clean copies of ZIP archives to `data/raw/`, and prints/saves a comprehensive
inventory of tabular, geospatial, PDF, image, and text assets.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import openpyxl
import pandas as pd
import pdfplumber
import pyogrio
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "base_informativa"
RAW_GEOCATMIN_DIR = PROJECT_ROOT / "data" / "raw" / "geocatmin"
RAW_TECTONIC_DIR = PROJECT_ROOT / "data" / "raw" / "tectonic_plates"
AUDIT_JSON_PATH = PROJECT_ROOT / "data" / "interim" / "audit_summary.json"

PSEUDO_NULL_STRINGS = {"", "-", "--", "na", "n/a", "null", "none", "nan"}


def _clean_json(obj: Any) -> Any:
    """Convert numpy/pandas scalars and paths into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd.isna(obj) if not isinstance(obj, (str, bytes, bool)) else False:
        return None
    return obj


def inventory_files(base_dir: Path) -> list[dict[str, Any]]:
    """Recursively list all files in `base_dir` with size and extension."""
    entries: list[dict[str, Any]] = []
    for path in sorted(base_dir.rglob("*")):
        if path.is_file():
            size_bytes = path.stat().st_size
            entries.append(
                {
                    "relative_path": path.relative_to(base_dir).as_posix(),
                    "name": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 4),
                }
            )
    return entries


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> list[dict[str, Any]]:
    """List ZIP entries and extract clean copy to `target_dir` without touching source."""
    target_dir.mkdir(parents=True, exist_ok=True)
    internal_files: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            internal_files.append(
                {
                    "filename": info.filename,
                    "is_dir": info.is_dir(),
                    "file_size_bytes": info.file_size,
                    "compress_size_bytes": info.compress_size,
                }
            )
        zf.extractall(target_dir)
    return internal_files


def _inspect_vector_layer(source_path: Path, layer: str | None = None) -> dict[str, Any]:
    """Inspect vector layer metadata, CRS, geometry types, columns, and feature count."""
    info = pyogrio.read_info(source_path, layer=layer)
    gdf = gpd.read_file(source_path, layer=layer)
    geom_types = (
        gdf.geometry.geom_type.value_counts(dropna=False).to_dict()
        if "geometry" in gdf.columns
        else {}
    )
    non_geom_cols = [c for c in gdf.columns if c != "geometry"]
    col_summary = []
    for col in non_geom_cols:
        s = gdf[col]
        sample_vals = [str(v) for v in s.dropna().head(3).tolist()]
        col_summary.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "null_count": int(s.isna().sum()),
                "sample_values": sample_vals,
            }
        )
    bounds = [round(float(x), 6) for x in gdf.total_bounds] if len(gdf) > 0 and gdf.geometry is not None else []
    return {
        "layer": layer or source_path.name,
        "driver": info.get("driver"),
        "feature_count": int(len(gdf)),
        "crs": str(gdf.crs) if gdf.crs else None,
        "geometry_types": {str(k): int(v) for k, v in geom_types.items()},
        "bounds": bounds,
        "columns": col_summary,
    }


def audit_zips(base_dir: Path) -> list[dict[str, Any]]:
    """Audit ZIP archives (GEOCATMIN & Tectonic Plates) and inspect spatial layers."""
    results: list[dict[str, Any]] = []
    zip_targets = {
        "geocatmin_datos.zip": RAW_GEOCATMIN_DIR,
        "tectonicplates-master.zip": RAW_TECTONIC_DIR,
    }

    for zip_path in sorted(base_dir.rglob("*.zip")):
        target_dir = zip_targets.get(zip_path.name, RAW_GEOCATMIN_DIR.parent / zip_path.stem)
        internal_files = _safe_extract_zip(zip_path, target_dir)

        layers_report: list[dict[str, Any]] = []
        text_docs: list[dict[str, Any]] = []

        # Inspect .gdb folders if present
        for gdb_dir in sorted(target_dir.rglob("*.gdb")):
            if gdb_dir.is_dir():
                for layer_name, geom_type in pyogrio.list_layers(gdb_dir):
                    layer_info = _inspect_vector_layer(gdb_dir, layer=layer_name)
                    layer_info["source_subpath"] = gdb_dir.relative_to(target_dir).as_posix()
                    layer_info["declared_geom_type"] = str(geom_type)
                    layers_report.append(layer_info)

        # Inspect GeoJSON / Shapefiles
        for spatial_file in sorted(
            list(target_dir.rglob("*.json"))
            + list(target_dir.rglob("*.geojson"))
            + list(target_dir.rglob("*.shp"))
        ):
            layer_info = _inspect_vector_layer(spatial_file)
            layer_info["source_subpath"] = spatial_file.relative_to(target_dir).as_posix()
            layers_report.append(layer_info)

        # Inspect README / LICENSE / TXT inside extracted zip
        for doc_file in sorted(
            list(target_dir.rglob("*.md")) + list(target_dir.rglob("*.txt"))
        ):
            content = doc_file.read_text(encoding="utf-8", errors="replace")
            text_docs.append(
                {
                    "subpath": doc_file.relative_to(target_dir).as_posix(),
                    "lines": len(content.splitlines()),
                    "preview": "\n".join(content.splitlines()[:15]),
                }
            )

        results.append(
            {
                "zip_relative_path": zip_path.relative_to(base_dir).as_posix(),
                "extracted_to": target_dir.relative_to(PROJECT_ROOT).as_posix(),
                "internal_file_count": len(internal_files),
                "internal_files": internal_files,
                "spatial_layers": layers_report,
                "text_docs": text_docs,
            }
        )
    return results


def _compute_null_stats(df_raw_str: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute true null and pseudo-null ('-', whitespace, NA) percentages per column."""
    stats: dict[str, dict[str, Any]] = {}
    total = len(df_raw_str)
    for col in df_raw_str.columns:
        s = df_raw_str[col]
        is_na = s.isna()
        s_clean = s.fillna("").astype(str).str.strip()
        is_pseudo = (~is_na) & (s_clean.str.lower().isin(PSEUDO_NULL_STRINGS))
        total_missing = int((is_na | is_pseudo).sum())
        stats[col] = {
            "na_count": int(is_na.sum()),
            "pseudo_null_count": int(is_pseudo.sum()),
            "total_missing_count": total_missing,
            "missing_pct": round((total_missing / total) * 100, 4) if total > 0 else 0.0,
        }
    return stats


def _numeric_distribution(series: pd.Series) -> dict[str, Any]:
    """Return summary percentiles for a numeric series."""
    num = pd.to_numeric(series, errors="coerce").dropna()
    if num.empty:
        return {"valid_count": 0}
    return {
        "valid_count": int(len(num)),
        "min": float(num.min()),
        "p25": float(num.quantile(0.25)),
        "median": float(num.median()),
        "p75": float(num.quantile(0.75)),
        "max": float(num.max()),
        "mean": round(float(num.mean()), 4),
    }


def _check_leading_zero_loss(
    series_raw_str: pd.Series, expected_len: int, label: str
) -> dict[str, Any]:
    """Check whether string values are shorter than `expected_len` (leading-zero loss)."""
    s = series_raw_str.dropna().astype(str).str.strip()
    s = s[s != ""]
    lengths = s.str.len().value_counts().sort_index().to_dict()
    short_mask = s.str.len() < expected_len
    return {
        "column": label,
        "expected_length": expected_len,
        "non_empty_count": int(len(s)),
        "length_distribution": {int(k): int(v) for k, v in lengths.items()},
        "shorter_than_expected_count": int(short_mask.sum()),
        "shorter_examples": s[short_mask].head(5).tolist(),
        "valid_examples": s[~short_mask].head(5).tolist(),
    }


def _parse_igp_datetime(fecha_s: pd.Series, hora_s: pd.Series) -> dict[str, Any]:
    """Parse mixed YYYYMMDD / M/D/YYYY + HHMMSS or ISO date/time strings into UTC."""
    f = fecha_s.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    h = hora_s.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    # If time has no colons, it's numeric HHMMSS that lost leading zeros
    no_colon_mask = ~h.str.contains(":", regex=False)
    h_norm = h.where(~no_colon_mask, h.str.zfill(6))

    # Try explicit YYYYMMDD + HHMMSS first, then slash dates M/D/YYYY + HHMMSS, then ISO/mixed
    dt = pd.to_datetime(f + h_norm, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    slash_mask = dt.isna() & f.str.contains("/", regex=False)
    if slash_mask.any():
        dt_slash = pd.to_datetime(
            f[slash_mask] + " " + h_norm[slash_mask],
            format="%m/%d/%Y %H%M%S",
            utc=True,
            errors="coerce",
        )
        dt = dt.fillna(dt_slash)
    rem_mask = dt.isna()
    if rem_mask.any():
        dt_mixed = pd.to_datetime(
            f[rem_mask] + " " + h_norm[rem_mask],
            format="mixed",
            utc=True,
            errors="coerce",
        )
        dt = dt.fillna(dt_mixed)

    valid = dt.dropna()
    return {
        "valid_timestamps": int(len(valid)),
        "invalid_timestamps": int(dt.isna().sum()),
        "slash_format_rows": int(f.str.contains("/", regex=False).sum()),
        "min_utc": valid.min().isoformat() if not valid.empty else None,
        "max_utc": valid.max().isoformat() if not valid.empty else None,
    }


def audit_excel_files(base_dir: Path) -> list[dict[str, Any]]:
    """Audit all `.xlsx` files in `base_dir` (IGP Historical & Instrumental catalogs)."""
    results: list[dict[str, Any]] = []
    for xlsx_path in sorted(base_dir.rglob("*.xlsx")):
        xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
        sheets_info: list[dict[str, Any]] = []
        for sheet in xl.sheet_names:
            df_inferred = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
            df_str = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str, engine="openpyxl")

            # Also inspect raw openpyxl cell types for row 2 (first data row)
            wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
            ws = wb[sheet]
            raw_sample_types = {}
            rows_iter = ws.iter_rows(min_row=1, max_row=2, values_only=False)
            header_cells = next(rows_iter, [])
            first_data_cells = next(rows_iter, [])
            for hc, dc in zip(header_cells, first_data_cells):
                raw_sample_types[str(hc.value)] = {
                    "openpyxl_data_type": dc.data_type,
                    "sample_value": str(dc.value),
                }
            wb.close()

            null_stats = _compute_null_stats(df_str)
            cols_meta = [
                {
                    "column": col,
                    "inferred_dtype": str(df_inferred[col].dtype),
                    "openpyxl_sample": raw_sample_types.get(col, {}),
                    **null_stats[col],
                }
                for col in df_inferred.columns
            ]

            date_range = {}
            if "FECHA_UTC" in df_str.columns and "HORA_UTC" in df_str.columns:
                date_range = _parse_igp_datetime(df_str["FECHA_UTC"], df_str["HORA_UTC"])
            elif "fecha UTC" in df_str.columns and "hora UTC" in df_str.columns:
                date_range = _parse_igp_datetime(df_str["fecha UTC"], df_str["hora UTC"])

            distributions = {}
            for col in df_inferred.columns:
                if any(k in col.upper() for k in ("MAGNITUD", "PROFUNDIDAD", "LAT", "LON")):
                    distributions[col] = _numeric_distribution(df_inferred[col])

            leading_zero_checks = []
            if "HORA_UTC" in df_str.columns:
                leading_zero_checks.append(
                    _check_leading_zero_loss(df_str["HORA_UTC"], 6, "HORA_UTC")
                )

            sheets_info.append(
                {
                    "sheet_name": sheet,
                    "rows": int(len(df_inferred)),
                    "cols": int(len(df_inferred.columns)),
                    "columns": cols_meta,
                    "date_range_utc": date_range,
                    "distributions": distributions,
                    "leading_zero_checks": leading_zero_checks,
                }
            )

        results.append(
            {
                "relative_path": xlsx_path.relative_to(base_dir).as_posix(),
                "sheets": sheets_info,
            }
        )
    return results


def _detect_csv_encoding_and_sep(csv_path: Path) -> tuple[str, str]:
    """Detect full-file encoding and delimiter for a CSV file."""
    raw_bytes = csv_path.read_bytes()
    encoding = "latin-1"
    sample = ""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            decoded = raw_bytes.decode(enc)
            encoding = enc
            sample = decoded[:8192]
            break
        except UnicodeDecodeError:
            continue
    else:
        sample = raw_bytes[:8192].decode("latin-1", errors="replace")

    first_lines = "\n".join(sample.splitlines()[:10])
    try:
        dialect = csv.Sniffer().sniff(first_lines, delimiters=",;\t|")
        sep = dialect.delimiter
    except csv.Error:
        sep = "," if first_lines.count(",") >= first_lines.count(";") else ";"
    return encoding, sep


def audit_csv_files(base_dir: Path) -> list[dict[str, Any]]:
    """Audit all `.csv` files in `base_dir` (e.g. ACELDAT)."""
    results: list[dict[str, Any]] = []
    for csv_path in sorted(base_dir.rglob("*.csv")):
        encoding, sep = _detect_csv_encoding_and_sep(csv_path)
        df_inferred = pd.read_csv(csv_path, encoding=encoding, sep=sep)
        df_str = pd.read_csv(csv_path, encoding=encoding, sep=sep, dtype=str, keep_default_na=False)
        # Re-read with default NA detection on str for consistent NA vs pseudo-null reporting
        df_str_na = pd.read_csv(csv_path, encoding=encoding, sep=sep, dtype=str)

        null_stats = _compute_null_stats(df_str_na)
        cols_meta = [
            {
                "column": col,
                "inferred_dtype": str(df_inferred[col].dtype),
                **null_stats[col],
            }
            for col in df_inferred.columns
        ]

        date_range = {}
        if "FECHA_EVENTO" in df_str.columns and "HORA_EVENTO" in df_str.columns:
            date_range = _parse_igp_datetime(df_str["FECHA_EVENTO"], df_str["HORA_EVENTO"])

        distributions = {}
        for col in df_inferred.columns:
            if any(k in col.upper() for k in ("ACEL", "LAT", "LON", "MAG", "PROF")):
                distributions[col] = _numeric_distribution(df_inferred[col])

        leading_zero_checks = []
        for col, expected_len in (("UBIGEO", 6), ("HORA_EVENTO", 6), ("FECHA_EVENTO", 8)):
            if col in df_str.columns:
                leading_zero_checks.append(
                    _check_leading_zero_loss(df_str[col], expected_len, col)
                )

        # Unique event timestamps & stations if ACELDAT
        extra_summary = {}
        if "COD_ESTACION" in df_str.columns:
            extra_summary["unique_stations"] = int(df_str["COD_ESTACION"].nunique())
        if "UBIGEO" in df_str.columns:
            extra_summary["unique_ubigeos_raw"] = int(df_str["UBIGEO"].nunique())
        if "FECHA_EVENTO" in df_str.columns and "HORA_EVENTO" in df_str.columns:
            evt_keys = df_str["FECHA_EVENTO"].str.strip() + "_" + df_str["HORA_EVENTO"].str.strip().str.zfill(6)
            extra_summary["unique_event_timestamps"] = int(evt_keys.nunique())

        results.append(
            {
                "relative_path": csv_path.relative_to(base_dir).as_posix(),
                "detected_encoding": encoding,
                "detected_delimiter": sep,
                "rows": int(len(df_inferred)),
                "cols": int(len(df_inferred.columns)),
                "columns": cols_meta,
                "date_range_utc": date_range,
                "distributions": distributions,
                "leading_zero_checks": leading_zero_checks,
                "extra_summary": extra_summary,
            }
        )
    return results


def audit_pdfs(base_dir: Path) -> list[dict[str, Any]]:
    """Inventory PDFs with page count and selectable-text extraction check via pdfplumber."""
    results: list[dict[str, Any]] = []
    for pdf_path in sorted(base_dir.rglob("*.pdf")):
        size_mb = round(pdf_path.stat().st_size / (1024 * 1024), 4)
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            sample_pages = min(3, page_count)
            extracted_chars = 0
            sample_tables_count = 0
            first_page_preview = ""
            for idx in range(sample_pages):
                txt = pdf.pages[idx].extract_text() or ""
                extracted_chars += len(txt.strip())
                if idx == 0:
                    first_page_preview = " ".join(txt.split())[:300]
                tables = pdf.pages[idx].extract_tables() or []
                sample_tables_count += len(tables)

            # For FONDES specifically, check all pages text availability
            full_text_pages = None
            total_chars_all = None
            if "fondes" in pdf_path.name.lower():
                pages_with_text = 0
                total_chars_all = 0
                for page in pdf.pages:
                    t = (page.extract_text() or "").strip()
                    if t:
                        pages_with_text += 1
                    total_chars_all += len(t)
                full_text_pages = pages_with_text

        results.append(
            {
                "relative_path": pdf_path.relative_to(base_dir).as_posix(),
                "size_mb": size_mb,
                "page_count": page_count,
                "sample_pages_checked": sample_pages,
                "sample_extracted_chars": extracted_chars,
                "sample_tables_detected": sample_tables_count,
                "has_selectable_text": extracted_chars > 0,
                "fondes_pages_with_text": full_text_pages,
                "fondes_total_chars": total_chars_all,
                "first_page_preview": first_page_preview,
            }
        )
    return results


def audit_images(base_dir: Path) -> list[dict[str, Any]]:
    """Inventory image files (.jpg, .jpeg, .png, .webp) with dimensions and mode."""
    results: list[dict[str, Any]] = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    for img_path in sorted(base_dir.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in exts:
            size_mb = round(img_path.stat().st_size / (1024 * 1024), 4)
            with Image.open(img_path) as im:
                width, height = im.size
                fmt = im.format
                mode = im.mode
            results.append(
                {
                    "relative_path": img_path.relative_to(base_dir).as_posix(),
                    "size_mb": size_mb,
                    "format": fmt,
                    "mode": mode,
                    "width_px": width,
                    "height_px": height,
                }
            )
    return results


def audit_texts(base_dir: Path) -> list[dict[str, Any]]:
    """Inventory plain `.txt` files in `base_dir`."""
    results: list[dict[str, Any]] = []
    for txt_path in sorted(base_dir.rglob("*.txt")):
        content = txt_path.read_text(encoding="utf-8", errors="replace")
        results.append(
            {
                "relative_path": txt_path.relative_to(base_dir).as_posix(),
                "size_mb": round(txt_path.stat().st_size / (1024 * 1024), 4),
                "line_count": len(content.splitlines()),
            }
        )
    return results


def main() -> None:
    """Execute complete Phase 0 audit and persist JSON summary."""
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {BASE_DIR}")

    report = {
        "base_dir": str(BASE_DIR),
        "file_inventory": inventory_files(BASE_DIR),
        "zip_archives": audit_zips(BASE_DIR),
        "excel_datasets": audit_excel_files(BASE_DIR),
        "csv_datasets": audit_csv_files(BASE_DIR),
        "pdf_documents": audit_pdfs(BASE_DIR),
        "images": audit_images(BASE_DIR),
        "text_files": audit_texts(BASE_DIR),
    }

    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_report = _clean_json(report)
    AUDIT_JSON_PATH.write_text(
        json.dumps(clean_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== AUDITORIA COMPLETADA SIN ERRORES ===")
    print(f"Total archivos en base_informativa/: {len(clean_report['file_inventory'])}")
    print(f"Archivos ZIP auditados y extraidos a data/raw/: {len(clean_report['zip_archives'])}")
    print(f"Archivos Excel auditados: {len(clean_report['excel_datasets'])}")
    print(f"Archivos CSV auditados: {len(clean_report['csv_datasets'])}")
    print(f"Documentos PDF auditados: {len(clean_report['pdf_documents'])}")
    print(f"Imagenes auditadas: {len(clean_report['images'])}")
    print(f"Resumen JSON guardado en: {AUDIT_JSON_PATH}")


if __name__ == "__main__":
    main()
