import os
import json
import zipfile
import traceback
import difflib
import re
import unicodedata
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from shapely.geometry import Point, shape
import requests
import uvicorn
import numpy as np

APP_TITLE = "Geospatial Mapping Tool"
DEVELOPER_FOOTER = (
    "Developed by Nahiduzzaman, URA, Dpt of Microbiology and Hygiene, BAU, "
    "mail: nahiduzzaman.2001055@bau.edu.bd"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "runtime_data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI(title=APP_TITLE)

STATE: Dict[str, Any] = {
    "df_path": None,
    "shape_path": None,
    "shape_kind": None,
    "latest_result": None,
    "latest_columns": [],
    "latest_data_preview": [],
    "cached_world_geojson": None,
    "cached_world_time": None
}

# =========================================================
# CACHED WORLD COUNTRIES
# =========================================================
def get_world_countries_geojson(force_refresh: bool = False) -> str:
    cache_file = os.path.join(CACHE_DIR, "countries.geojson")
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"

    # Check cache age (refresh if older than 7 days)
    if not force_refresh and os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mod_time < timedelta(days=7):
            return cache_file

    # Download
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(cache_file, "wb") as f:
            f.write(resp.content)
        return cache_file
    except Exception as e:
        if os.path.exists(cache_file):
            return cache_file  # fallback to old cache
        raise RuntimeError(f"Failed to download world boundaries: {e}")

# =========================================================
# BASIC UTILITIES
# =========================================================
def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

def detect_file_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return "excel"
    if lower.endswith(".zip"):
        return "zip"
    raise ValueError("Unsupported file type")

def read_tabular_file(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError("Only CSV and Excel are supported")

def write_tabular_file(df: pd.DataFrame, path: str) -> None:
    lower = path.lower()
    if lower.endswith(".csv"):
        df.to_csv(path, index=False)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        df.to_excel(path, index=False)
    else:
        raise ValueError("Only CSV and Excel are supported")

def extract_shapefile_zip(zip_path: str, target_dir: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    shp_files = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith(".shp"):
                shp_files.append(os.path.join(root, f))

    if not shp_files:
        raise ValueError("No .shp file found inside uploaded ZIP")
    return shp_files[0]

# =========================================================
# ENHANCED COLUMN DETECTION
# =========================================================
def guess_region_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    preferred = [
        "country", "country_name", "nation", "admin0", "admin_0",
        "area", "region", "district", "division", "state", "province",
        "location", "county", "adm1", "adm2", "name"
    ]
    for candidate in preferred:
        if candidate in lower_map:
            return lower_map[candidate]

    for c in cols:
        lc = c.lower()
        if any(k in lc for k in
               ["country", "area", "region", "district", "division", "state", "province", "location", "adm"]):
            return c
    return None

def guess_lat_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for candidate in ["lat", "latitude", "y", "latitud", "ycoord"]:
        if candidate in lower_map:
            return lower_map[candidate]
    for c in cols:
        if "lat" in c.lower():
            return c
    return None

def guess_lon_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for candidate in ["lon", "longitude", "long", "x", "lng", "xcoord"]:
        if candidate in lower_map:
            return lower_map[candidate]
    for c in cols:
        if "lon" in c.lower() or "long" in c.lower():
            return c
    return None

def guess_value_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for candidate in ["prevalence", "spread", "incidence", "rate", "value", "percent", "percentage", "cases", "count"]:
        if candidate in lower_map:
            return lower_map[candidate]
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in ["prevalence", "spread", "incidence", "rate", "percent", "cases"]):
            return c
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[0] if numeric_cols else None

def guess_date_column(df: pd.DataFrame) -> Optional[str]:
    """Detect columns that look like dates."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        # Try converting first non-null value
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if sample:
            try:
                pd.to_datetime(sample)
                return col
            except:
                pass
    return None

def auto_plan_fields(df: pd.DataFrame) -> Dict[str, Any]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    plan = {
        "region_col": guess_region_column(df),
        "lat_col": guess_lat_column(df),
        "lon_col": guess_lon_column(df),
        "value_col": guess_value_column(df),
        "date_col": guess_date_column(df),
        "pathogen_col": None,
        "genotype_col": None,
        "serotype_col": None
    }

    for candidate in ["pathogen", "organism", "disease_agent"]:
        if candidate in lower_map:
            plan["pathogen_col"] = lower_map[candidate]
            break

    for candidate in ["genotype", "lineage", "clade"]:
        if candidate in lower_map:
            plan["genotype_col"] = lower_map[candidate]
            break

    for candidate in ["serotype", "subtype"]:
        if candidate in lower_map:
            plan["serotype_col"] = lower_map[candidate]
            break

    return plan

# =========================================================
# NORMALIZATION / MATCHING
# =========================================================
def normalize_name(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(the|republic|state|province|district|division|county|region|area|of|union|territory)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_point_geometry(df: pd.DataFrame, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    work = df.copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lon_col]).copy()
    geometry = [Point(xy) for xy in zip(work[lon_col], work[lat_col])]
    return gpd.GeoDataFrame(work, geometry=geometry, crs="EPSG:4326")

def load_uploaded_shape(shape_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shape_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf

def auto_generate_country_polygons_from_geojson(country_names: List[str]) -> gpd.GeoDataFrame:
    cache_path = get_world_countries_geojson()
    world = gpd.read_file(cache_path).to_crs("EPSG:4326")

    candidate_cols = [
        c for c in world.columns
        if c.lower() in ["admin", "name", "name_en", "country", "sovereignt", "formal_en"]
    ]
    if not candidate_cols:
        candidate_cols = list(world.columns)

    chosen_col = candidate_cols[0]
    world["_match_name_original"] = world[chosen_col].astype(str).str.strip()
    world["_match_name_norm"] = world[chosen_col].map(normalize_name)

    wanted = [normalize_name(c) for c in country_names if pd.notna(c)]
    out = world[world["_match_name_norm"].isin(wanted)].copy()

    if out.empty:
        # Try fuzzy matching as fallback
        norm_list = world["_match_name_norm"].tolist()
        matched_norms = []
        for w in wanted:
            match = difflib.get_close_matches(w, norm_list, n=1, cutoff=0.7)
            if match:
                matched_norms.append(match[0])
        out = world[world["_match_name_norm"].isin(matched_norms)].copy()
        if out.empty:
            raise ValueError(
                "Automatic country polygon generation failed. Country names did not match the online boundary file."
            )
    return out

def fuzzy_match_name(query: str, candidates_norm: List[str], cutoff: float = 0.8) -> Optional[str]:
    if not query:
        return None
    matches = difflib.get_close_matches(query, candidates_norm, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def build_name_mapping(
        data_values: pd.Series,
        shape_values: pd.Series,
        auto_correct: bool = True,
        fuzzy_cutoff: float = 0.8
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    data_unique = pd.Series(data_values.dropna().astype(str).unique())
    shape_unique = pd.Series(shape_values.dropna().astype(str).unique())

    shape_norm_to_originals: Dict[str, List[str]] = {}
    for val in shape_unique:
        n = normalize_name(val)
        shape_norm_to_originals.setdefault(n, []).append(str(val))

    shape_norm_list = list(shape_norm_to_originals.keys())
    mapping: Dict[str, str] = {}
    unmatched_report: List[Dict[str, Any]] = []

    for raw in data_unique:
        raw_str = str(raw)
        raw_norm = normalize_name(raw_str)

        if raw_norm in shape_norm_to_originals:
            mapping[raw_norm] = raw_norm
            continue

        suggestion_norm = None
        if auto_correct:
            suggestion_norm = fuzzy_match_name(raw_norm, shape_norm_list, cutoff=fuzzy_cutoff)

        if suggestion_norm:
            mapping[raw_norm] = suggestion_norm
        else:
            suggestion_display = None
            close_any = difflib.get_close_matches(raw_norm, shape_norm_list, n=1, cutoff=0.5)
            if close_any:
                suggestion_display = shape_norm_to_originals[close_any[0]][0]

            unmatched_report.append({
                "input_name": raw_str,
                "normalized_name": raw_norm,
                "suggestion": suggestion_display
            })

    return mapping, unmatched_report

def join_data_to_polygons(
        df: pd.DataFrame,
        poly_gdf: gpd.GeoDataFrame,
        data_region_col: str,
        shape_region_col: str,
        auto_correct_names: bool = True,
        fuzzy_cutoff: float = 0.8
) -> Tuple[gpd.GeoDataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    left = poly_gdf.copy()
    right = df.copy()

    left["_shape_name_original"] = left[shape_region_col].astype(str).str.strip()
    left["_shape_name_norm"] = left[shape_region_col].map(normalize_name)

    right["_data_name_original"] = right[data_region_col].astype(str).str.strip()
    right["_data_name_norm"] = right[data_region_col].map(normalize_name)

    mapping, unmatched_report = build_name_mapping(
        data_values=right["_data_name_original"],
        shape_values=left["_shape_name_original"],
        auto_correct=auto_correct_names,
        fuzzy_cutoff=fuzzy_cutoff
    )

    right["_mapped_shape_norm"] = right["_data_name_norm"].map(mapping)

    correction_report = []
    for _, row in right[["_data_name_original", "_data_name_norm", "_mapped_shape_norm"]].drop_duplicates().iterrows():
        original = row["_data_name_original"]
        data_norm = row["_data_name_norm"]
        mapped = row["_mapped_shape_norm"]

        if pd.notna(mapped) and mapped != data_norm:
            matched_shape_rows = left[left["_shape_name_norm"] == mapped]
            matched_shape_name = None
            if not matched_shape_rows.empty:
                matched_shape_name = str(matched_shape_rows.iloc[0]["_shape_name_original"])

            correction_report.append({
                "input_name": original,
                "corrected_to": matched_shape_name
            })

    merged = left.merge(
        right,
        left_on="_shape_name_norm",
        right_on="_mapped_shape_norm",
        how="left"
    )

    return merged, unmatched_report, correction_report

# =========================================================
# PLOTTING HELPERS (enhanced with customization)
# =========================================================
def style_maps():
    color_maps = {
        "viridis": "viridis",
        "plasma": "plasma",
        "coolwarm": "coolwarm",
        "YlOrRd": "YlOrRd",
        "RdPu": "RdPu",
        "Blues": "Blues"
    }
    marker_shapes = {
        "circle": "o",
        "square": "s",
        "triangle": "^",
        "diamond": "D"
    }
    hatch_patterns = {
        "none": "",
        "forward_slash": "///",
        "back_slash": "\\\\\\",
        "cross": "xx",
        "dot": "...",
        "plus": "++"
    }
    return color_maps, marker_shapes, hatch_patterns

def get_column_type(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return "missing"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"
    return "categorical"

def empty_geodf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": pd.Series(dtype="object")},
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326"
    )

def parse_manual_shapes(geojson_text: Optional[str]) -> gpd.GeoDataFrame:
    if not geojson_text:
        return empty_geodf()

    try:
        data = json.loads(geojson_text)
        features = data.get("features", [])

        if not features:
            return empty_geodf()

        rows = []
        geometries = []

        for i, feat in enumerate(features):
            geom = feat.get("geometry")
            props = feat.get("properties", {}) or {}
            name = props.get("name", f"manual_shape_{i+1}")

            if geom is None:
                continue

            try:
                shapely_geom = shape(geom)
            except Exception:
                continue

            if shapely_geom is None or shapely_geom.is_empty:
                continue

            rows.append({"name": name})
            geometries.append(shapely_geom)

        if not rows or not geometries:
            return empty_geodf()

        df = pd.DataFrame(rows)
        return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(geometries, crs="EPSG:4326"), crs="EPSG:4326")

    except Exception as e:
        raise ValueError(f"Invalid manual shape GeoJSON: {e}")

def add_labels(ax, geodf: gpd.GeoDataFrame, label_col: str, fontsize: int = 8, fontfamily: str = 'sans-serif'):
    if label_col not in geodf.columns:
        return

    for _, row in geodf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        try:
            point = geom.representative_point()
            txt = str(row[label_col]) if pd.notna(row[label_col]) else ""
            if txt:
                ax.text(point.x, point.y, txt, fontsize=fontsize, family=fontfamily, ha="center", va="center")
        except Exception:
            continue

def add_point_labels(ax, geodf: gpd.GeoDataFrame, label_col: str, fontsize: int = 8, fontfamily: str = 'sans-serif'):
    if label_col not in geodf.columns:
        return

    for _, row in geodf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        txt = str(row[label_col]) if pd.notna(row[label_col]) else ""
        if txt:
            ax.text(geom.x, geom.y, txt, fontsize=fontsize, family=fontfamily, ha="left", va="bottom")

def encode_categories(values: pd.Series) -> Dict[str, int]:
    categories = [str(v) for v in values.dropna().astype(str).unique().tolist()]
    return {cat: i for i, cat in enumerate(categories)}

def pick_marker_size_series(df: pd.DataFrame, size_col: Optional[str]) -> pd.Series:
    if size_col and size_col in df.columns:
        s = pd.to_numeric(df[size_col], errors="coerce")
        if s.notna().sum() > 0:
            s = s.fillna(s.median())
            mn = s.min()
            mx = s.max()
            if mx > mn:
                return 40 + (s - mn) / (mx - mn) * 260
            return pd.Series([120] * len(df), index=df.index)
    return pd.Series([100] * len(df), index=df.index)

def resolve_auto_country_field(
        df: pd.DataFrame,
        auto_country_field: Optional[str],
        data_region_col: Optional[str]
) -> str:
    if auto_country_field and auto_country_field in df.columns:
        return auto_country_field

    if data_region_col and data_region_col in df.columns:
        return data_region_col

    guessed = guess_region_column(df)
    if guessed and guessed in df.columns:
        return guessed

    raise ValueError(
        "Auto country field is missing or invalid. Please select a valid country/area column, "
        "or upload a file containing a column like Country, Area, Region, District, Division, or State."
    )

# =========================================================
# SUMMARY STATISTICS
# =========================================================
def compute_summary_stats(df: pd.DataFrame, region_col: str, value_col: str) -> List[Dict]:
    if region_col not in df.columns or value_col not in df.columns:
        return []
    stats = df.groupby(region_col)[value_col].agg(['count', 'mean', 'median', 'min', 'max', 'sum']).reset_index()
    stats = stats.round(2).fillna(0)
    return stats.to_dict(orient='records')

# =========================================================
# MAIN ANALYSIS (with enhanced customizations)
# =========================================================
def run_analysis_logic(
        df: pd.DataFrame,
        shapemode: str,
        map_type: str,
        data_region_col: Optional[str],
        shape_region_col: Optional[str],
        lat_col: Optional[str],
        lon_col: Optional[str],
        value_col: Optional[str],
        label_col: Optional[str],
        show_labels: bool,
        show_legend: bool,
        dpi: int,
        cmap_name: str,
        marker_shape_fields: Optional[List[str]],   # changed to list
        hatch_fields: Optional[List[str]],          # changed to list
        size_field: Optional[str],
        shapefile_path: Optional[str],
        auto_country_field: Optional[str],
        popup_cols: Optional[List[str]],
        manual_shapes_geojson: Optional[str],
        auto_correct_names: bool = True,
        fuzzy_cutoff: float = 0.8,
        map_title: str = "Disease Prevalence / Spread Map",
        # New customization parameters:
        fig_width: float = 14.0,
        fig_height: float = 9.0,
        title_fontsize: int = 16,
        title_fontweight: str = "bold",
        axis_label_fontsize: int = 12,
        axis_label_fontweight: str = "bold",
        legend_fontsize: int = 10,
        legend_position: str = "upper left",
        colorbar_orientation: str = "vertical",
        colorbar_shrink: float = 0.75,
        colorbar_pad: float = 0.02,
        colorbar_label_fontsize: int = 10,
        colorbar_fraction: float = 0.15,           # new
        xlabel_text: str = "Longitude",
        ylabel_text: str = "Latitude",
        # Additional new parameters
        missing_color_white: bool = False,
        label_fontsize: int = 8,
        label_fontfamily: str = 'sans-serif',
        remove_spines: Optional[Dict[str, bool]] = None   # e.g., {"top": True, "bottom": False, ...}
) -> Dict[str, Any]:
    color_maps, marker_shapes, hatch_patterns = style_maps()
    cmap = plt.get_cmap(color_maps.get(cmap_name, "YlOrRd"))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    plot_gdf = None
    legends = []
    unmatched_report: List[Dict[str, Any]] = []
    correction_report: List[Dict[str, Any]] = []

    # Helper to combine multiple fields for shape/hatch
    def combine_fields(gdf, fields, sep="__"):
        if not fields:
            return None
        # Only use fields that exist
        existing = [f for f in fields if f in gdf.columns]
        if not existing:
            return None
        # Combine, converting to string and handling NaN
        combined = gdf[existing[0]].astype(str)
        for f in existing[1:]:
            combined = combined + sep + gdf[f].astype(str)
        return combined

    if map_type == "polygon":
        if shapemode == "uploaded":
            if not shapefile_path:
                raise ValueError("Uploaded shapefile was not provided")
            poly_gdf = load_uploaded_shape(shapefile_path)

        elif shapemode == "auto_country":
            resolved_auto_country_field = resolve_auto_country_field(df, auto_country_field, data_region_col)

            poly_gdf = auto_generate_country_polygons_from_geojson(
                df[resolved_auto_country_field].dropna().astype(str).tolist()
            )

            possible = [
                c for c in poly_gdf.columns
                if c.lower() in ["admin", "name", "name_en", "country", "sovereignt", "formal_en"]
            ]
            if not possible:
                raise ValueError("No suitable country-name column found in automatically generated polygons")

            shape_region_col = possible[0]
            data_region_col = resolved_auto_country_field

        else:
            raise ValueError("Invalid shapemode for polygon map")

        if not data_region_col or data_region_col not in df.columns:
            guessed = guess_region_column(df)
            if guessed:
                data_region_col = guessed
            else:
                raise ValueError("Data region column is missing or invalid")

        if not shape_region_col or shape_region_col not in poly_gdf.columns:
            raise ValueError("Shape region column is missing or invalid")

        if not value_col or value_col not in df.columns:
            guessed_val = guess_value_column(df)
            if guessed_val:
                value_col = guessed_val
            else:
                raise ValueError("Value column is missing or invalid")

        plot_gdf, unmatched_report, correction_report = join_data_to_polygons(
            df=df,
            poly_gdf=poly_gdf,
            data_region_col=data_region_col,
            shape_region_col=shape_region_col,
            auto_correct_names=auto_correct_names,
            fuzzy_cutoff=fuzzy_cutoff
        )

        plot_gdf[value_col] = pd.to_numeric(plot_gdf[value_col], errors="coerce")

        # Determine missing color
        missing_color = "white" if missing_color_white else "#f0f0f0"
        missing_hatch = None if missing_color_white else "//"
        missing_label = "No data" if not missing_color_white else None  # no label if white

        # Plot polygons
        plot_gdf.plot(
            column=value_col,
            cmap=cmap,
            linewidth=0.6,
            edgecolor="black",
            legend=False,
            ax=ax,
            missing_kwds={
                "color": missing_color,
                "edgecolor": "gray",
                "hatch": missing_hatch,
                "label": missing_label
            } if not missing_color_white else {
                "color": missing_color,
                "edgecolor": "gray",
                "hatch": None,
                "label": None
            }
        )

        # Add colorbar if numeric
        if show_legend and value_col and get_column_type(plot_gdf, value_col) == "numeric":
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=plot_gdf[value_col].min(), vmax=plot_gdf[value_col].max()))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation=colorbar_orientation,
                                shrink=colorbar_shrink, pad=colorbar_pad,
                                fraction=colorbar_fraction)
            cbar.set_label(value_col, fontsize=colorbar_label_fontsize)
            cbar.ax.tick_params(labelsize=legend_fontsize)

        # Hatch overlays (multiple fields combined)
        if hatch_fields:
            combined_hatch = combine_fields(plot_gdf, hatch_fields)
            if combined_hatch is not None:
                plot_gdf["_combined_hatch"] = combined_hatch
                hatch_map = encode_categories(plot_gdf["_combined_hatch"])
                hatch_keys = list(hatch_patterns.keys())

                for cat, idx in hatch_map.items():
                    cat_gdf = plot_gdf[plot_gdf["_combined_hatch"] == cat]
                    if not cat_gdf.empty:
                        cat_gdf.plot(
                            ax=ax,
                            facecolor="none",
                            edgecolor="black",
                            linewidth=0.2,
                            hatch=hatch_patterns[hatch_keys[idx % len(hatch_keys)]]
                        )

                if show_legend:
                    for cat, idx in hatch_map.items():
                        patt = hatch_patterns[hatch_keys[idx % len(hatch_keys)]]
                        legends.append(
                            Patch(facecolor="white", edgecolor="black", hatch=patt,
                                  label=f"Hatch: {cat}")
                        )

        if show_labels and label_col and label_col in plot_gdf.columns:
            add_labels(ax, plot_gdf, label_col, fontsize=label_fontsize, fontfamily=label_fontfamily)

    elif map_type == "point":
        if not lat_col or lat_col not in df.columns:
            guessed_lat = guess_lat_column(df)
            if guessed_lat:
                lat_col = guessed_lat
            else:
                raise ValueError("Latitude column is missing or invalid")

        if not lon_col or lon_col not in df.columns:
            guessed_lon = guess_lon_column(df)
            if guessed_lon:
                lon_col = guessed_lon
            else:
                raise ValueError("Longitude column is missing or invalid")

        plot_gdf = make_point_geometry(df, lat_col, lon_col)

        if shapemode == "uploaded" and shapefile_path:
            try:
                poly_gdf = load_uploaded_shape(shapefile_path)
                poly_gdf.boundary.plot(ax=ax, linewidth=0.7, edgecolor="gray")
            except Exception:
                pass
        elif shapemode == "auto_country":
            resolved_auto_country_field = resolve_auto_country_field(df, auto_country_field, data_region_col)
            try:
                poly_gdf = auto_generate_country_polygons_from_geojson(
                    df[resolved_auto_country_field].dropna().astype(str).tolist()
                )
                poly_gdf.boundary.plot(ax=ax, linewidth=0.7, edgecolor="gray")
            except Exception:
                pass

        # Marker shape from combined fields
        marker_combined = None
        if marker_shape_fields:
            marker_combined = combine_fields(plot_gdf, marker_shape_fields)

        size_series = pick_marker_size_series(plot_gdf, size_field)

        # Prepare color values
        color_vals = None
        if value_col and value_col in plot_gdf.columns and get_column_type(plot_gdf, value_col) == "numeric":
            plot_gdf[value_col] = pd.to_numeric(plot_gdf[value_col], errors="coerce")
            color_vals = plot_gdf[value_col]
        elif not value_col:
            guessed_val = guess_value_column(df)
            if guessed_val and guessed_val in plot_gdf.columns and get_column_type(plot_gdf, guessed_val) == "numeric":
                value_col = guessed_val
                plot_gdf[value_col] = pd.to_numeric(plot_gdf[value_col], errors="coerce")
                color_vals = plot_gdf[value_col]

        # Normalize colors for colorbar
        norm = None
        if color_vals is not None and len(color_vals.dropna()) > 0:
            norm = Normalize(vmin=color_vals.min(), vmax=color_vals.max())

        if marker_combined is not None:
            marker_map = encode_categories(marker_combined)
            marker_keys = list(marker_shapes.keys())

            for cat, idx in marker_map.items():
                sub = plot_gdf[marker_combined == cat].copy()
                mk = marker_shapes[marker_keys[idx % len(marker_keys)]]

                if color_vals is not None and value_col and norm:
                    sc = ax.scatter(
                        sub.geometry.x,
                        sub.geometry.y,
                        c=pd.to_numeric(sub[value_col], errors="coerce"),
                        s=size_series.loc[sub.index],
                        cmap=cmap,
                        norm=norm,
                        marker=mk,
                        edgecolors="black",
                        linewidths=0.5
                    )
                else:
                    ax.scatter(
                        sub.geometry.x,
                        sub.geometry.y,
                        s=size_series.loc[sub.index],
                        marker=mk,
                        edgecolors="black",
                        linewidths=0.5
                    )

                legends.append(
                    Line2D(
                        [0], [0],
                        marker=mk,
                        linestyle="None",
                        markeredgecolor="black",
                        markerfacecolor="white",
                        label=f"Marker: {cat}",
                        markersize=9
                    )
                )
        else:
            if color_vals is not None and value_col and norm:
                sc = ax.scatter(
                    plot_gdf.geometry.x,
                    plot_gdf.geometry.y,
                    c=color_vals,
                    s=size_series,
                    cmap=cmap,
                    norm=norm,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5
                )
                if show_legend:
                    cbar = fig.colorbar(sc, ax=ax, shrink=colorbar_shrink, pad=colorbar_pad,
                                        orientation=colorbar_orientation, fraction=colorbar_fraction)
                    cbar.set_label(value_col, fontsize=colorbar_label_fontsize)
                    cbar.ax.tick_params(labelsize=legend_fontsize)
            else:
                ax.scatter(
                    plot_gdf.geometry.x,
                    plot_gdf.geometry.y,
                    s=size_series,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5
                )

        if show_labels and label_col and label_col in plot_gdf.columns:
            add_point_labels(ax, plot_gdf, label_col, fontsize=label_fontsize, fontfamily=label_fontfamily)

    else:
        raise ValueError("map_type must be 'polygon' or 'point'")

    # Add manual shapes
    manual_gdf = parse_manual_shapes(manual_shapes_geojson)
    if manual_gdf is not None and not manual_gdf.empty:
        try:
            manual_gdf.boundary.plot(ax=ax, linewidth=2.0, edgecolor="purple")
        except Exception:
            manual_gdf.plot(ax=ax, color="purple", markersize=20)

        if show_labels and "name" in manual_gdf.columns:
            try:
                add_labels(ax, manual_gdf, "name", fontsize=label_fontsize, fontfamily=label_fontfamily)
            except Exception:
                pass

        if show_legend:
            legends.append(Line2D([0], [0], color="purple", linewidth=2, label="Manual edited shape"))

    ax.set_title(map_title, fontsize=title_fontsize, fontweight=title_fontweight)
    ax.set_xlabel(xlabel_text, fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
    ax.set_ylabel(ylabel_text, fontsize=axis_label_fontsize, fontweight=axis_label_fontweight)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Remove spines if requested
    if remove_spines:
        for spine, visible in remove_spines.items():
            if spine in ax.spines:
                ax.spines[spine].set_visible(not visible)   # visible=True means keep, we want to remove if True

    if show_legend and legends:
        ax.legend(handles=legends, loc=legend_position, fontsize=legend_fontsize, frameon=True)

    plt.tight_layout()

    # Save outputs
    png_path = os.path.join(EXPORT_DIR, "analysis_map.png")
    tiff_path = os.path.join(EXPORT_DIR, "analysis_map.tiff")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Preview GeoJSON
    preview_features = []
    if plot_gdf is not None and len(plot_gdf) > 0:
        popup_cols = popup_cols or []
        popup_cols = [c for c in popup_cols if c in plot_gdf.columns]

        for _, row in plot_gdf.iterrows():
            props = {}
            for c in popup_cols:
                try:
                    val = row[c]
                    props[c] = None if pd.isna(val) else str(val)
                except Exception:
                    props[c] = None

            try:
                preview_features.append({
                    "type": "Feature",
                    "geometry": row.geometry.__geo_interface__,
                    "properties": props
                })
            except Exception:
                continue

    preview_geojson = {
        "type": "FeatureCollection",
        "features": preview_features
    }

    return {
        "png_path": png_path,
        "tiff_path": tiff_path,
        "preview_geojson": preview_geojson,
        "unmatched_report": unmatched_report,
        "correction_report": correction_report
    }

# =========================================================
# ENHANCED HTML UI (with new customization fields)
# =========================================================
def build_index_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{APP_TITLE}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --accent: #e74c3c;
            --light: #ecf0f1;
            --dark: #2c3e50;
            --success: #27ae60;
            --warning: #f39c12;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            color: var(--dark);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .header {{
            background: white;
            border-bottom: 1px solid #ddd;
            padding: 12px 24px;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .header i {{ color: var(--secondary); }}
        .main-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        .sidebar {{
            width: 540px;
            background: white;
            border-right: 1px solid #ddd;
            overflow-y: auto;
            padding: 20px;
            box-shadow: 2px 0 8px rgba(0,0,0,0.02);
        }}
        .map-container {{
            flex: 1;
            position: relative;
        }}
        #map {{
            width: 100%;
            height: 100%;
        }}
        .footer {{
            background: white;
            border-top: 1px solid #ddd;
            padding: 8px 24px;
            font-size: 0.85rem;
            color: #7f8c8d;
            text-align: center;
        }}
        .section {{
            margin-bottom: 24px;
            border-bottom: 1px solid #eee;
            padding-bottom: 16px;
        }}
        .section h3 {{
            font-size: 1.1rem;
            margin-bottom: 12px;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .section h3 i {{ color: var(--secondary); }}
        .form-group {{
            margin-bottom: 12px;
        }}
        label {{
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 4px;
            color: #555;
        }}
        input, select, textarea {{
            width: 100%;
            padding: 8px 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 0.9rem;
            transition: border 0.2s;
        }}
        input:focus, select:focus, textarea:focus {{
            border-color: var(--secondary);
            outline: none;
        }}
        .btn {{
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        .btn-primary {{
            background: var(--secondary);
            color: white;
        }}
        .btn-primary:hover {{ background: #2980b9; }}
        .btn-success {{ background: var(--success); color: white; }}
        .btn-success:hover {{ background: #219653; }}
        .btn-warning {{ background: var(--warning); color: white; }}
        .btn-warning:hover {{ background: #e67e22; }}
        .btn-block {{ width: 100%; }}
        .row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .status-bar {{
            background: #f8f9fa;
            border-radius: 6px;
            padding: 10px;
            font-size: 0.85rem;
            border-left: 4px solid var(--secondary);
            margin-top: 12px;
            white-space: pre-wrap;
            max-height: 80px;
            overflow-y: auto;
        }}
        .status-bar.error {{ border-left-color: var(--accent); color: #c0392b; }}
        .data-table-container {{
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 0.8rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #f4f6f9;
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 6px;
        }}
        td, th {{
            border: 1px solid #ddd;
            padding: 4px;
        }}
        td input {{
            width: 100%;
            border: none;
            padding: 2px;
            background: transparent;
        }}
        .report-box {{
            background: #fff8e1;
            border: 1px solid #f0d98a;
            border-radius: 6px;
            padding: 10px;
            font-size: 0.8rem;
            max-height: 150px;
            overflow-y: auto;
        }}
        .summary-table {{
            font-size: 0.8rem;
            max-height: 200px;
            overflow-y: auto;
        }}
        .summary-table table {{
            width: 100%;
        }}
        .loader {{
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--secondary);
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            display: inline-block;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .hidden {{ display: none; }}
        .tooltip-icon {{
            color: #7f8c8d;
            margin-left: 5px;
            cursor: help;
        }}
        .advanced-options {{
            margin-top: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 6px;
        }}
        .about-section {{
            margin-top: 20px;
            padding: 15px;
            background: #e8f4fd;
            border-radius: 8px;
            font-size: 0.9rem;
            border-left: 4px solid var(--secondary);
        }}
        .about-section a {{
            color: var(--secondary);
            text-decoration: none;
        }}
        .about-section a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <i class="fas fa-map-marked-alt"></i> {APP_TITLE}
    </div>
    <div class="main-container">
        <div class="sidebar">
            <!-- Upload Sections -->
            <div class="section">
                <h3><i class="fas fa-upload"></i> Data Upload</h3>
                <div class="form-group">
                    <label>Disease Data (CSV/Excel) <i class="fas fa-info-circle tooltip-icon" title="Upload your dataset with columns for region, coordinates, and values."></i></label>
                    <input type="file" id="dataFile" accept=".csv,.xlsx,.xls" />
                    <button class="btn btn-success btn-block" onclick="uploadData()" style="margin-top:8px;">
                        <i class="fas fa-cloud-upload-alt"></i> Upload Data
                    </button>
                </div>
                <div class="form-group">
                    <label>Shapefile (ZIP) <i class="fas fa-info-circle tooltip-icon" title="Upload a zip containing .shp, .shx, .dbf, etc."></i></label>
                    <input type="file" id="shapeFile" accept=".zip" />
                    <button class="btn btn-success btn-block" onclick="uploadShape()" style="margin-top:8px;">
                        <i class="fas fa-draw-polygon"></i> Upload Shapefile
                    </button>
                </div>
            </div>

            <!-- Data Editor -->
            <div class="section">
                <h3><i class="fas fa-edit"></i> Data Editor <span style="font-weight:normal; font-size:0.8rem;">(first 50 rows)</span></h3>
                <div class="data-table-container" id="dataEditorWrap">
                    <table id="dataEditorTable"><tr><td>No data loaded.</td></tr></table>
                </div>
                <button class="btn btn-warning btn-block" onclick="saveEditedData()" style="margin-top:8px;">
                    <i class="fas fa-save"></i> Save Edited Data
                </button>
            </div>

            <!-- Map Settings -->
            <div class="section">
                <h3><i class="fas fa-cog"></i> Map Settings</h3>
                <div class="row">
                    <div class="form-group">
                        <label>Shape Source</label>
                        <select id="shapeMode">
                            <option value="uploaded">Uploaded shapefile</option>
                            <option value="auto_country">Auto country boundaries</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Map Type</label>
                        <select id="mapType">
                            <option value="polygon">Polygon (area)</option>
                            <option value="point">Point</option>
                        </select>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>Data Region Col</label>
                        <select id="dataRegionCol"></select>
                    </div>
                    <div class="form-group">
                        <label>Shape Region Col</label>
                        <select id="shapeRegionCol"></select>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>Latitude Col</label>
                        <select id="latCol"></select>
                    </div>
                    <div class="form-group">
                        <label>Longitude Col</label>
                        <select id="lonCol"></select>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>Value Col</label>
                        <select id="valueCol"></select>
                    </div>
                    <div class="form-group">
                        <label>Label Col</label>
                        <select id="labelCol"></select>
                    </div>
                </div>
                <div class="form-group">
                    <label>Auto Country Field (for auto boundaries)</label>
                    <select id="autoCountryField"></select>
                </div>
                <div class="advanced-options">
                    <h4 style="font-size:0.9rem; margin-top:0;">Customization</h4>
                    <div class="row">
                        <div class="form-group">
                            <label>Figure Width (in)</label>
                            <input type="number" id="figWidth" value="14.0" step="0.1" min="5" max="30" />
                        </div>
                        <div class="form-group">
                            <label>Figure Height (in)</label>
                            <input type="number" id="figHeight" value="9.0" step="0.1" min="5" max="30" />
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label>Title Font Size</label>
                            <input type="number" id="titleFontSize" value="16" min="8" max="36" />
                        </div>
                        <div class="form-group">
                            <label>Title Font Weight</label>
                            <select id="titleFontWeight">
                                <option value="bold">Bold</option>
                                <option value="normal">Normal</option>
                                <option value="light">Light</option>
                            </select>
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label>Axis Label Font Size</label>
                            <input type="number" id="axisLabelFontSize" value="12" min="6" max="24" />
                        </div>
                        <div class="form-group">
                            <label>Axis Label Font Weight</label>
                            <select id="axisLabelFontWeight">
                                <option value="bold">Bold</option>
                                <option value="normal">Normal</option>
                            </select>
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label>X Axis Label</label>
                            <input type="text" id="xlabel" value="Longitude" />
                        </div>
                        <div class="form-group">
                            <label>Y Axis Label</label>
                            <input type="text" id="ylabel" value="Latitude" />
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label>Legend Font Size</label>
                            <input type="number" id="legendFontSize" value="10" min="6" max="20" />
                        </div>
                        <div class="form-group">
                            <label>Legend Position</label>
                            <select id="legendPosition">
                                <option value="upper left">Upper Left</option>
                                <option value="upper right">Upper Right</option>
                                <option value="lower left">Lower Left</option>
                                <option value="lower right">Lower Right</option>
                                <option value="best">Best</option>
                            </select>
                        </div>
                    </div>
                </div>
                <div class="advanced-options">
                    <h4 style="font-size:0.9rem; margin-top:0;">Styling</h4>
                    <div class="row">
                        <div class="form-group">
                            <label>Color Map</label>
                            <select id="cmapName">
                                <option value="YlOrRd">YlOrRd</option>
                                <option value="viridis">Viridis</option>
                                <option value="plasma">Plasma</option>
                                <option value="coolwarm">Coolwarm</option>
                                <option value="RdPu">RdPu</option>
                                <option value="Blues">Blues</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>DPI</label>
                            <input type="number" id="dpi" value="300" min="72" max="1200" />
                        </div>
                    </div>
                    <!-- New: Marker Shape Fields (multi) -->
                    <div class="form-group">
                        <label>Marker Shape Fields (multi)</label>
                        <select id="markerShapeFields" multiple size="4"></select>
                        <small>Select one or more fields; combined values define marker shape.</small>
                    </div>
                    <!-- New: Hatch Fields (multi) -->
                    <div class="form-group">
                        <label>Hatch Fields (multi)</label>
                        <select id="hatchFields" multiple size="4"></select>
                        <small>Select one or more fields; combined values define hatch pattern.</small>
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label>Marker Size Field</label>
                            <select id="sizeField"></select>
                        </div>
                        <div class="form-group">
                            <label>Popup Fields</label>
                            <select id="popupCols" multiple size="4"></select>
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label>Show Labels</label>
                            <select id="showLabels">
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Show Legend</label>
                            <select id="showLegend">
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label>Auto Correct Names</label>
                            <select id="autoCorrectNames">
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Fuzzy Cutoff</label>
                            <input type="number" id="fuzzyCutoff" value="0.80" min="0.60" max="0.95" step="0.01" />
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Map Title</label>
                        <input type="text" id="mapTitle" value="Disease Prevalence / Spread Map" />
                    </div>
                    <!-- New: White missing polygons -->
                    <div class="form-group">
                        <label><input type="checkbox" id="missingColorWhite" /> White missing polygons (instead of gray hatch)</label>
                    </div>
                </div>
                <!-- New: Label styling -->
                <div class="advanced-options">
                    <h4 style="font-size:0.9rem; margin-top:0;">Label Styling</h4>
                    <div class="row">
                        <div class="form-group">
                            <label>Label Font Size</label>
                            <input type="number" id="labelFontSize" value="8" min="6" max="24" />
                        </div>
                        <div class="form-group">
                            <label>Label Font Family</label>
                            <select id="labelFontFamily">
                                <option value="sans-serif">Sans-serif</option>
                                <option value="serif">Serif</option>
                                <option value="monospace">Monospace</option>
                                <option value="Arial">Arial</option>
                                <option value="Times New Roman">Times New Roman</option>
                                <option value="Courier New">Courier New</option>
                            </select>
                        </div>
                    </div>
                </div>
                <!-- New: Border removal -->
                <div class="advanced-options">
                    <h4 style="font-size:0.9rem; margin-top:0;">Remove Borders (Spines)</h4>
                    <div style="display:flex; gap:10px; flex-wrap:wrap;">
                        <label><input type="checkbox" id="removeTop" /> Top</label>
                        <label><input type="checkbox" id="removeBottom" /> Bottom</label>
                        <label><input type="checkbox" id="removeLeft" /> Left</label>
                        <label><input type="checkbox" id="removeRight" /> Right</label>
                    </div>
                </div>
                <!-- New: Colorbar finer control -->
                <div class="advanced-options">
                    <h4 style="font-size:0.9rem; margin-top:0;">Colorbar</h4>
                    <div class="row">
                        <div class="form-group">
                            <label>Orientation</label>
                            <select id="colorbarOrientation">
                                <option value="vertical">Vertical</option>
                                <option value="horizontal">Horizontal</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Shrink</label>
                            <input type="number" id="colorbarShrink" value="0.75" step="0.05" min="0.2" max="1.0" />
                        </div>
                    </div>
                    <div class="row">
                        <div class="form-group">
                            <label>Pad</label>
                            <input type="number" id="colorbarPad" value="0.02" step="0.01" min="0.0" max="0.5" />
                        </div>
                        <div class="form-group">
                            <label>Fraction</label>
                            <input type="number" id="colorbarFraction" value="0.15" step="0.01" min="0.05" max="0.5" />
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Label Font Size</label>
                        <input type="number" id="colorbarLabelFontSize" value="10" min="6" max="20" />
                    </div>
                </div>
            </div>

            <!-- Time Filter (if date column exists) -->
            <div class="section" id="timeSection" style="display:none;">
                <h3><i class="fas fa-clock"></i> Time Filter</h3>
                <div class="form-group">
                    <label>Date Column</label>
                    <select id="dateCol"></select>
                </div>
                <div class="form-group">
                    <label>Time Range</label>
                    <div id="timeSliderContainer">
                        <input type="range" id="timeSlider" min="0" max="100" value="100" style="width:100%;">
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
                        <span id="timeMinLabel">Min</span>
                        <span id="timeMaxLabel">Max</span>
                    </div>
                </div>
                <button class="btn btn-primary btn-block" onclick="applyTimeFilter()">
                    <i class="fas fa-filter"></i> Apply Time Filter
                </button>
            </div>

            <!-- Summary Statistics -->
            <div class="section">
                <h3><i class="fas fa-chart-bar"></i> Summary Statistics</h3>
                <div id="summaryContainer" class="summary-table">
                    <p>Run analysis to see summary.</p>
                </div>
                <button class="btn btn-primary btn-block" onclick="fetchSummary()">
                    <i class="fas fa-calculator"></i> Compute Summary
                </button>
            </div>

            <!-- Action Buttons -->
            <div class="section">
                <button class="btn btn-primary btn-block" onclick="runAnalysis()" style="padding:14px;">
                    <i class="fas fa-play"></i> Run Analysis
                </button>
                <div style="display:flex; gap:8px; margin-top:8px;">
                    <a id="pngLink" href="#" target="_blank" class="btn btn-success" style="flex:1; text-decoration:none;">
                        <i class="fas fa-download"></i> PNG
                    </a>
                    <a id="tiffLink" href="#" target="_blank" class="btn btn-success" style="flex:1; text-decoration:none;">
                        <i class="fas fa-download"></i> TIFF
                    </a>
                </div>
            </div>

            <!-- Matching Report -->
            <div class="section">
                <h3><i class="fas fa-clipboard-list"></i> Matching Report</h3>
                <div id="matchReport" class="report-box">No report yet.</div>
            </div>

            <!-- About Section -->
            <div class="about-section">
                <h4><i class="fas fa-user"></i> About the Developer</h4>
                <p><strong>Nahiduzzaman</strong> is a Doctor of Veterinary Medicine (DVM) graduate and research assistant at the Department of Microbiology and Hygiene, Bangladesh Agricultural University. His research interests lie in bioinformatics, genomics, epidemiology, and vaccine development.</p>
                <p>ORCID: <a href="https://orcid.org/0009-0000-1970-9480" target="_blank">https://orcid.org/0009-0000-1970-9480</a></p>
            </div>

            <!-- Status Bar -->
            <div id="statusBox" class="status-bar">Ready.</div>
            <div id="loader" class="loader hidden" style="margin-top:10px;"></div>
        </div>

        <div class="map-container">
            <div id="map"></div>
        </div>
    </div>
    <div class="footer">{DEVELOPER_FOOTER}</div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
    <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>

    <script>
        // Initialize map
        let map = L.map('map').setView([20, 0], 2);
        let baseLayers = {{
            "OpenStreetMap": L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; OpenStreetMap contributors'
            }})
        }};
        baseLayers["OpenStreetMap"].addTo(map);

        let previewLayer = L.geoJSON(null);
        let editableLayers = new L.FeatureGroup();
        map.addLayer(editableLayers);
        let heatLayer = null;

        // Add draw control
        const drawControl = new L.Control.Draw({{
            edit: {{ featureGroup: editableLayers }},
            draw: {{
                polygon: true,
                rectangle: true,
                circle: true,
                marker: true,
                polyline: false,
                circlemarker: false
            }}
        }});
        map.addControl(drawControl);

        map.on(L.Draw.Event.CREATED, function (e) {{
            const layer = e.layer;
            layer.feature = layer.feature || {{ type: "Feature", properties: {{ name: "manual_shape" }} }};
            editableLayers.addLayer(layer);
        }});

        // State
        let currentColumns = [];
        let currentRows = [];
        let fullData = null;
        let dateColDetected = null;

        function setStatus(msg, isError=false) {{
            const box = document.getElementById('statusBox');
            box.textContent = msg;
            box.className = isError ? 'status-bar error' : 'status-bar';
        }}

        function showLoader(show) {{
            document.getElementById('loader').classList.toggle('hidden', !show);
        }}

        function selectedValues(selectId) {{
            const opts = document.getElementById(selectId).selectedOptions;
            return Array.from(opts).map(x => x.value).filter(Boolean);
        }}

        function fillSelect(id, cols, includeEmpty=true) {{
            const el = document.getElementById(id);
            el.innerHTML = "";
            if (includeEmpty) {{
                const op = document.createElement("option");
                op.value = "";
                op.textContent = "-- none --";
                el.appendChild(op);
            }}
            cols.forEach(c => {{
                const op = document.createElement("option");
                op.value = c;
                op.textContent = c;
                el.appendChild(op);
            }});
        }}

        function setSelectValueIfExists(id, value) {{
            if (!value) return;
            const el = document.getElementById(id);
            const options = Array.from(el.options).map(o => o.value);
            if (options.includes(value)) {{
                el.value = value;
            }}
        }}

        function renderEditorTable(columns, rows) {{
            currentColumns = columns;
            currentRows = rows;
            const table = document.getElementById("dataEditorTable");
            table.innerHTML = "";
            if (!columns.length) {{
                table.innerHTML = "<tr><td>No data loaded.</td></tr>";
                return;
            }}
            const thead = document.createElement("thead");
            const hr = document.createElement("tr");
            columns.forEach(col => {{
                const th = document.createElement("th");
                th.textContent = col;
                hr.appendChild(th);
            }});
            thead.appendChild(hr);
            table.appendChild(thead);
            const tbody = document.createElement("tbody");
            rows.forEach((row, rIdx) => {{
                const tr = document.createElement("tr");
                columns.forEach((col) => {{
                    const td = document.createElement("td");
                    const inp = document.createElement("input");
                    inp.value = row[col] ?? "";
                    inp.dataset.row = rIdx;
                    inp.dataset.col = col;
                    inp.addEventListener("input", (e) => {{
                        const rr = parseInt(e.target.dataset.row);
                        const cc = e.target.dataset.col;
                        currentRows[rr][cc] = e.target.value;
                    }});
                    td.appendChild(inp);
                    tr.appendChild(td);
                }});
                tbody.appendChild(tr);
            }});
            table.appendChild(tbody);
        }}

        function showMatchReport(unmatched, corrected) {{
            const box = document.getElementById("matchReport");
            let txt = "";
            if (corrected && corrected.length) {{
                txt += "✅ Corrected names:\\n";
                corrected.forEach(x => {{ txt += `  - ${{x.input_name}} → ${{x.corrected_to}}\\n`; }});
                txt += "\\n";
            }}
            if (unmatched && unmatched.length) {{
                txt += "❌ Unmatched names:\\n";
                unmatched.forEach(x => {{
                    txt += `  - ${{x.input_name}}`;
                    if (x.suggestion) txt += ` (suggestion: ${{x.suggestion}})`;
                    txt += "\\n";
                }});
            }}
            if (!txt.trim()) txt = "✅ All names matched successfully.";
            box.textContent = txt;
        }}

        // Upload handlers
        async function uploadData() {{
            const fileInput = document.getElementById('dataFile');
            if (!fileInput.files.length) {{
                setStatus("Please choose a data file first.", true);
                return;
            }}
            showLoader(true);
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            try {{
                const res = await fetch("/upload-data", {{ method: "POST", body: formData }});
                const out = await res.json();
                if (!res.ok) throw new Error(out.detail || "Upload failed");
                const cols = out.columns || [];
                fullData = out.full_data;
                dateColDetected = out.plan?.date_col;
                ["dataRegionCol","latCol","lonCol","valueCol","labelCol","autoCountryField",
                 "sizeField","popupCols","dateCol"].forEach(id => fillSelect(id, cols, true));
                // For multi-select fields
                fillSelect("markerShapeFields", cols, true);
                fillSelect("hatchFields", cols, true);
                if (out.plan) {{
                    setSelectValueIfExists("dataRegionCol", out.plan.region_col);
                    setSelectValueIfExists("autoCountryField", out.plan.region_col);
                    setSelectValueIfExists("latCol", out.plan.lat_col);
                    setSelectValueIfExists("lonCol", out.plan.lon_col);
                    setSelectValueIfExists("valueCol", out.plan.value_col);
                    setSelectValueIfExists("labelCol", out.plan.region_col);
                    setSelectValueIfExists("dateCol", out.plan.date_col);
                }}
                renderEditorTable(out.columns || [], out.preview_rows || []);
                if (out.plan?.date_col) {{
                    document.getElementById('timeSection').style.display = 'block';
                    setupTimeSlider(out.full_data, out.plan.date_col);
                }} else {{
                    document.getElementById('timeSection').style.display = 'none';
                }}
                setStatus(`Data uploaded. ${{out.n_rows}} rows, ${{out.n_cols}} cols.`);
            }} catch (err) {{
                setStatus("Data upload error: " + err.message, true);
            }} finally {{
                showLoader(false);
            }}
        }}

        function setupTimeSlider(data, dateCol) {{
            // Placeholder
            document.getElementById('timeMinLabel').innerText = 'Min';
            document.getElementById('timeMaxLabel').innerText = 'Max';
        }}

        async function applyTimeFilter() {{
            setStatus("Time filter applied (not fully implemented in demo)");
        }}

        async function fetchSummary() {{
            const valueCol = document.getElementById('valueCol').value;
            const regionCol = document.getElementById('dataRegionCol').value;
            if (!valueCol || !regionCol) {{
                setStatus("Please select region and value columns first.", true);
                return;
            }}
            showLoader(true);
            try {{
                const res = await fetch(`/summary?value_col=${{valueCol}}&region_col=${{regionCol}}`);
                const out = await res.json();
                if (!res.ok) throw new Error(out.detail);
                let html = "<table><tr><th>Region</th><th>Count</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Sum</th></tr>";
                out.stats.forEach(row => {{
                    html += `<tr><td>${{row[regionCol]}}</td><td>${{row.count}}</td><td>${{row.mean}}</td><td>${{row.median}}</td><td>${{row.min}}</td><td>${{row.max}}</td><td>${{row.sum}}</td></tr>`;
                }});
                html += "</table>";
                document.getElementById('summaryContainer').innerHTML = html;
            }} catch (err) {{
                setStatus("Summary error: " + err.message, true);
            }} finally {{
                showLoader(false);
            }}
        }}

        async function saveEditedData() {{
            showLoader(true);
            try {{
                const res = await fetch("/save-edited-data", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ columns: currentColumns, rows: currentRows }})
                }});
                const out = await res.json();
                if (!res.ok) throw new Error(out.detail || "Failed to save");
                setStatus("Edited data saved.");
            }} catch (err) {{
                setStatus("Save error: " + err.message, true);
            }} finally {{
                showLoader(false);
            }}
        }}

        async function uploadShape() {{
            const fileInput = document.getElementById('shapeFile');
            if (!fileInput.files.length) {{
                setStatus("Please choose a shapefile ZIP first.", true);
                return;
            }}
            showLoader(true);
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            try {{
                const res = await fetch("/upload-shape", {{ method: "POST", body: formData }});
                const out = await res.json();
                if (!res.ok) throw new Error(out.detail || "Shape upload failed");
                fillSelect("shapeRegionCol", out.columns || [], true);
                setStatus(`Shape uploaded: ${{out.n_rows}} features.`);
            }} catch (err) {{
                setStatus("Shape upload error: " + err.message, true);
            }} finally {{
                showLoader(false);
            }}
        }}

        async function runAnalysis() {{
            const manualGeoJSON = editableLayers.toGeoJSON();
            // Collect border removal checkboxes
            const removeSpines = {{
                top: document.getElementById("removeTop").checked,
                bottom: document.getElementById("removeBottom").checked,
                left: document.getElementById("removeLeft").checked,
                right: document.getElementById("removeRight").checked
            }};
            const payload = {{
                shape_mode: document.getElementById("shapeMode").value,
                map_type: document.getElementById("mapType").value,
                data_region_col: document.getElementById("dataRegionCol").value || null,
                shape_region_col: document.getElementById("shapeRegionCol").value || null,
                lat_col: document.getElementById("latCol").value || null,
                lon_col: document.getElementById("lonCol").value || null,
                value_col: document.getElementById("valueCol").value || null,
                label_col: document.getElementById("labelCol").value || null,
                auto_country_field: document.getElementById("autoCountryField").value || null,
                cmap_name: document.getElementById("cmapName").value,
                marker_shape_fields: selectedValues("markerShapeFields"),  // array
                hatch_fields: selectedValues("hatchFields"),              // array
                size_field: document.getElementById("sizeField").value || null,
                show_labels: document.getElementById("showLabels").value === "true",
                show_legend: document.getElementById("showLegend").value === "true",
                auto_correct_names: document.getElementById("autoCorrectNames").value === "true",
                fuzzy_cutoff: parseFloat(document.getElementById("fuzzyCutoff").value || "0.80"),
                dpi: parseInt(document.getElementById("dpi").value || "300"),
                popup_cols: selectedValues("popupCols"),
                manual_shapes_geojson: JSON.stringify(manualGeoJSON),
                map_title: document.getElementById("mapTitle").value || "Disease Map",
                // Basic dimensions
                fig_width: parseFloat(document.getElementById("figWidth").value || "14.0"),
                fig_height: parseFloat(document.getElementById("figHeight").value || "9.0"),
                title_fontsize: parseInt(document.getElementById("titleFontSize").value || "16"),
                title_fontweight: document.getElementById("titleFontWeight").value,
                axis_label_fontsize: parseInt(document.getElementById("axisLabelFontSize").value || "12"),
                axis_label_fontweight: document.getElementById("axisLabelFontWeight").value,
                legend_fontsize: parseInt(document.getElementById("legendFontSize").value || "10"),
                legend_position: document.getElementById("legendPosition").value,
                xlabel_text: document.getElementById("xlabel").value || "Longitude",
                ylabel_text: document.getElementById("ylabel").value || "Latitude",
                // New parameters
                missing_color_white: document.getElementById("missingColorWhite").checked,
                label_fontsize: parseInt(document.getElementById("labelFontSize").value || "8"),
                label_fontfamily: document.getElementById("labelFontFamily").value,
                remove_spines: removeSpines,
                colorbar_orientation: document.getElementById("colorbarOrientation").value,
                colorbar_shrink: parseFloat(document.getElementById("colorbarShrink").value || "0.75"),
                colorbar_pad: parseFloat(document.getElementById("colorbarPad").value || "0.02"),
                colorbar_fraction: parseFloat(document.getElementById("colorbarFraction").value || "0.15"),
                colorbar_label_fontsize: parseInt(document.getElementById("colorbarLabelFontSize").value || "10")
            }};
            showLoader(true);
            try {{
                const res = await fetch("/run-analysis", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify(payload)
                }});
                const out = await res.json();
                if (!res.ok) throw new Error(out.detail || "Analysis failed");
                if (previewLayer) previewLayer.remove();
                if (heatLayer) map.removeLayer(heatLayer);

                previewLayer = L.geoJSON(out.preview_geojson, {{
                    onEachFeature: function(feature, layer) {{
                        const props = feature.properties || {{}};
                        if (Object.keys(props).length) {{
                            let html = "<div>" + Object.entries(props).map(([k,v]) => `<b>${{k}}:</b> ${{v}}`).join("<br>") + "</div>";
                            layer.bindPopup(html);
                        }}
                    }}
                }}).addTo(map);

                try {{
                    const bounds = previewLayer.getBounds();
                    if (bounds.isValid()) map.fitBounds(bounds, {{padding: [20,20]}});
                }} catch(e){{}}

                document.getElementById("pngLink").href = out.png_download;
                document.getElementById("tiffLink").href = out.tiff_download;

                showMatchReport(out.unmatched_report || [], out.correction_report || []);
                setStatus("Analysis completed.");
            }} catch (err) {{
                setStatus("Analysis error: " + err.message, true);
            }} finally {{
                showLoader(false);
            }}
        }}
    </script>
</body>
</html>
"""

# =========================================================
# ROUTES (updated)
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(build_index_html())

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        fname = safe_filename(file.filename)
        ftype = detect_file_type(fname)

        if ftype not in {"csv", "excel"}:
            raise ValueError("Data file must be CSV or Excel")

        save_path = os.path.join(UPLOAD_DIR, fname)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        df = read_tabular_file(save_path)
        if df.empty:
            raise ValueError("Uploaded data file is empty")

        STATE["df_path"] = save_path
        STATE["latest_columns"] = df.columns.tolist()
        STATE["latest_data_preview"] = df.head(50).fillna("").to_dict(orient="records")

        plan = auto_plan_fields(df)

        return JSONResponse({
            "message": "Data uploaded",
            "columns": df.columns.tolist(),
            "plan": plan,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "preview_rows": df.head(50).fillna("").to_dict(orient="records"),
            "full_data": df.fillna("").to_dict(orient="records")  # careful with size
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/save-edited-data")
async def save_edited_data(request: Request):
    try:
        if not STATE["df_path"]:
            raise ValueError("No uploaded data file found")

        payload = await request.json()
        columns = payload.get("columns", [])
        rows = payload.get("rows", [])

        if not columns:
            raise ValueError("No columns provided")
        if rows is None:
            raise ValueError("No rows provided")

        df = pd.DataFrame(rows, columns=columns)
        write_tabular_file(df, STATE["df_path"])

        STATE["latest_columns"] = df.columns.tolist()
        STATE["latest_data_preview"] = df.head(50).fillna("").to_dict(orient="records")

        return JSONResponse({
            "message": "Edited data saved successfully",
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1])
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-shape")
async def upload_shape(file: UploadFile = File(...)):
    try:
        fname = safe_filename(file.filename)
        ftype = detect_file_type(fname)

        if ftype != "zip":
            raise ValueError("Shapefile upload must be a ZIP file")

        zip_save = os.path.join(UPLOAD_DIR, fname)
        with open(zip_save, "wb") as f:
            f.write(await file.read())

        extract_dir = os.path.join(UPLOAD_DIR, os.path.splitext(fname)[0])
        os.makedirs(extract_dir, exist_ok=True)

        shp_path = extract_shapefile_zip(zip_save, extract_dir)
        gdf = load_uploaded_shape(shp_path)

        STATE["shape_path"] = shp_path
        STATE["shape_kind"] = "uploaded"

        return JSONResponse({
            "message": "Shapefile uploaded",
            "columns": gdf.columns.tolist(),
            "n_rows": int(gdf.shape[0])
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/run-analysis")
async def run_analysis(request: Request):
    try:
        if not STATE["df_path"]:
            raise ValueError("Please upload a disease data file first")

        payload = await request.json()
        df = read_tabular_file(STATE["df_path"])

        # Extract new customization parameters with defaults
        result = run_analysis_logic(
            df=df,
            shapemode=payload.get("shape_mode"),
            map_type=payload.get("map_type"),
            data_region_col=payload.get("data_region_col"),
            shape_region_col=payload.get("shape_region_col"),
            lat_col=payload.get("lat_col"),
            lon_col=payload.get("lon_col"),
            value_col=payload.get("value_col"),
            label_col=payload.get("label_col"),
            show_labels=payload.get("show_labels", True),
            show_legend=payload.get("show_legend", True),
            dpi=int(payload.get("dpi", 300)),
            cmap_name=payload.get("cmap_name", "YlOrRd"),
            marker_shape_fields=payload.get("marker_shape_fields"),  # list
            hatch_fields=payload.get("hatch_fields"),                # list
            size_field=payload.get("size_field"),
            shapefile_path=STATE["shape_path"],
            auto_country_field=payload.get("auto_country_field"),
            popup_cols=payload.get("popup_cols", []),
            manual_shapes_geojson=payload.get("manual_shapes_geojson"),
            auto_correct_names=payload.get("auto_correct_names", True),
            fuzzy_cutoff=float(payload.get("fuzzy_cutoff", 0.8)),
            map_title=payload.get("map_title", "Disease Prevalence / Spread Map"),
            # Existing parameters
            fig_width=float(payload.get("fig_width", 14.0)),
            fig_height=float(payload.get("fig_height", 9.0)),
            title_fontsize=int(payload.get("title_fontsize", 16)),
            title_fontweight=payload.get("title_fontweight", "bold"),
            axis_label_fontsize=int(payload.get("axis_label_fontsize", 12)),
            axis_label_fontweight=payload.get("axis_label_fontweight", "bold"),
            legend_fontsize=int(payload.get("legend_fontsize", 10)),
            legend_position=payload.get("legend_position", "upper left"),
            colorbar_orientation=payload.get("colorbar_orientation", "vertical"),
            colorbar_shrink=float(payload.get("colorbar_shrink", 0.75)),
            colorbar_pad=float(payload.get("colorbar_pad", 0.02)),
            colorbar_label_fontsize=int(payload.get("colorbar_label_fontsize", 10)),
            xlabel_text=payload.get("xlabel_text", "Longitude"),
            ylabel_text=payload.get("ylabel_text", "Latitude"),
            # New parameters
            missing_color_white=payload.get("missing_color_white", False),
            label_fontsize=int(payload.get("label_fontsize", 8)),
            label_fontfamily=payload.get("label_fontfamily", "sans-serif"),
            remove_spines=payload.get("remove_spines", {}),
            colorbar_fraction=float(payload.get("colorbar_fraction", 0.15))
        )

        STATE["latest_result"] = result

        return JSONResponse({
            "message": "Analysis completed",
            "png_download": "/download/png",
            "tiff_download": "/download/tiff",
            "preview_geojson": result["preview_geojson"],
            "unmatched_report": result["unmatched_report"],
            "correction_report": result["correction_report"]
        })
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/summary")
async def get_summary(value_col: str, region_col: str):
    try:
        if not STATE["df_path"]:
            raise ValueError("No data uploaded")
        df = read_tabular_file(STATE["df_path"])
        stats = compute_summary_stats(df, region_col, value_col)
        return JSONResponse({"stats": stats})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/download/png")
async def download_png():
    result = STATE.get("latest_result")
    if not result:
        raise HTTPException(status_code=404, detail="No analysis result found")
    return FileResponse(result["png_path"], filename="analysis_map.png", media_type="image/png")

@app.get("/download/tiff")
async def download_tiff():
    result = STATE.get("latest_result")
    if not result:
        raise HTTPException(status_code=404, detail="No analysis result found")
    return FileResponse(result["tiff_path"], filename="analysis_map.tiff", media_type="image/tiff")

if __name__ == "__main__":
    uvicorn.run("map:app", host="0.0.0.0", port=8000, reload=True)