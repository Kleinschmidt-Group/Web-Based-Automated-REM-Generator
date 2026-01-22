#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# app.py

from __future__ import annotations
import os

# Fix PROJ unicode errors with Python 3.13 (must be before other imports)
os.environ['PROJ_SKIP_READ_USER_WRITABLE_DIRECTORY'] = 'YES'
os.environ['PROJ_DEBUG'] = '0'
os.environ['PROJ_NETWORK'] = 'OFF'

import warnings
warnings.filterwarnings('ignore', category=UnicodeWarning)
warnings.filterwarnings('ignore', message='.*utf-8.*')
warnings.filterwarnings('ignore', module='pyproj')

import time
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import array_bounds
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent thread crashes
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw, Fullscreen, MeasureControl, Geocoder
import glob as _glob
from PIL import Image
from joblib import Parallel, delayed
from shapely.geometry import box as viewport_box
import requests
from shapely.geometry import box as shp_box
from shapely.ops import unary_union
import gc
import multiprocessing
import tempfile
import rem_utils as utils
import rem_config as cfg
import REM_Calcs as rem_calcs_module
import rem_hillshade_colorramp as style_rem_module
import hillshade as hillshade_module
import aerial_tiles
from rem_utils import (
    limit_threadpools,
    clip_dem_to_aoi,
    feet_to_m,
    _auto_chunk_tile_size_for_rem,
    _call_rem_main_with_filtered_kwargs,
    _call_style_rem_with_filtered_kwargs,
)
from rem_config import (
    CLIP_DEM,
    COLOR_RAMP_OPTIONS,
    BACKGROUND_OPTIONS,
    DEFAULT_BACKGROUND,
    DOWNLOADS_FOLDER,
    EXPORT_DPI,
    EXPORT_SCALE,
    _FIXED_N_LEVELS,
)

# Allow massive images (1m DEMs) without crashing
Image.MAX_IMAGE_PIXELS = None


import data_collections as dc

# STOP PROCESS FLAG - Global mechanism for immediate process termination
STOP_FLAG_FILE = os.path.join(tempfile.gettempdir(), "rem_stop_process.flag")

def set_stop_flag():
    """Create stop flag file to signal all processes to halt immediately"""
    with open(STOP_FLAG_FILE, 'w') as f:
        f.write('STOP')
    print("\n" + "="*60)
    print(" STOP PROCESSES KILLED - All processing will halt immediately!")
    print("="*60 + "\n")

def clear_stop_flag():
    """Remove stop flag file when starting new processing"""
    if os.path.exists(STOP_FLAG_FILE):
        os.remove(STOP_FLAG_FILE)

def check_stop_flag():
    """Check if stop has been requested - call this frequently in loops!"""
    if os.path.exists(STOP_FLAG_FILE):
        print("\n STOP FLAG DETECTED - Halting process immediately!")
        return True
    return False


# App Configuration
st.set_page_config(
    page_title="Automated REM Generator",
    layout="wide",
)

st.title("Automated Relative Elevation Model (REM) Generator")
st.write(
    "Draw an Area of Interest (AOI), configure DEM/REM options, and run the full "
    "pipeline to generate REM GeoTIFFs and styled PNGs. "
)

# Initialize Session State
for key, default in {
    "run_requested": False,
    "prep_done": False,
    "dem_acquired": False,  # Track DEM download separately from river scan
    "hillshade_qa_generated": False,  # Track QA hillshade generation separately
    "dem_validated": False,  # Track DEM validation to avoid redundant checks
    "memory_checked": False,  # Track memory safety check
    "qa_approved": False,
    "rem_done": False,
    "dem_file": None,
    "dem_stats": None,  # Cache DEM validation results
    "memory_safe": None,  # Cache memory check result
    "dem_folder_for_run": None,
    "river_vector_path": None,
    "aoi_geojson_path": None,
    "hillshade_output": None,
    "aerial_output": None,
    "reprojected_river_path": None,
    "dem_folder": None,
    "rem_folder": None,
    "visuals_folder": None,
    "extra_data_folder": None,
    "scanned_river_list": [],
    "available_resolutions": [],
    "pipeline_start_time": 0.0,
    "qa_png": None,
    "coverage_data": None,
    "custom_dem_validated": False,  # Track if custom DEM has been validated
    "custom_river_validated": False,  # Track if custom river has been validated
    "custom_dem_info": None,  # Cache custom DEM validation results
    "custom_river_info": None,  # Cache custom river validation results
    "uploaded_dem_path": None,  # Track uploaded DEM file path
    "uploaded_river_path": None,  # Track uploaded river file path
    "dem_original_crs": None,  # Track original DEM CRS
    "dem_output_crs": None,  # Track output DEM CRS (after any reprojection)
    "river_original_crs": None,  # Track original river CRS
    "river_output_crs": None,  # Track output river CRS (after reprojection)
    "show_coverage_layers": False,
    "aoi_gdf": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Remove leftover .tmp files from interrupted downloads
def cleanup_tmp_files(downloads_folder="./Downloads"):
    """Remove .tmp files from interrupted downloads on startup."""
    if os.path.exists(downloads_folder):
        tmp_files = _glob.glob(os.path.join(downloads_folder, "**/*.tmp"), recursive=True)
        for tmp_file in tmp_files:
            try:
                # Use safe_remove for Windows compatibility
                if utils.safe_remove(tmp_file, verbose=False):
                    print(f"Cleaned up: {tmp_file}")
                else:
                    print(f"Could not remove {tmp_file} (file may be locked)")
            except Exception as e:
                print(f"Could not remove {tmp_file}: {e}")

# Run cleanup once on app startup
if "cleanup_done" not in st.session_state:
    cleanup_tmp_files()
    st.session_state["cleanup_done"] = True


# Clean up intermediate files
# Removes temporary cache files after REM processing is complete
#All other files remain in the project folder
def cleanup_project_outputs(project_folder: str):
    try:
        # Delete only true temporary files (cache, temp data)
        files_to_remove = [
            "mosaic_*_dem.tif",           # Original raw downloaded DEM (large, pre-clip)
            "temp_scan.geojson",          # Temporary scan data
            "hyriver_cache.sqlite",       # Cache
        ]

        for pattern in files_to_remove:
            matches = _glob.glob(os.path.join(project_folder, pattern))
            for fpath in matches:
                try:
                    fname = os.path.basename(fpath)

                    # Do not delete the clipped mosaic
                    if "mosaic_clipped.tif" in fname:
                        continue

                    # Use safe_remove for Windows compatibility
                    if utils.safe_remove(fpath, verbose=False):
                        print(f"Cleaned up: {fname}")
                    else:
                        print(f"Could not remove {fname} (file may be locked)")
                except Exception as e:
                    print(f"Could not remove {fname}: {e}")

        print(" Cleanup complete!")

        # Return None (no inputs folder created)
        return None

    except Exception as e:
        print(f"WARNING: Cleanup encountered an error: {e}")


# Module Mapping
# Map the functions/modules to the names expected by the rest of the app
find_aoi_geojsons             = dc.find_aoi_geojsons
download_and_mosaic_dems      = dc.download_and_mosaic_dems
choose_and_save_nhd_river     = dc.choose_and_save_nhd_river
scan_nhd_rivers               = dc.scan_nhd_rivers
get_available_project_resolutions = dc.get_available_project_resolutions
get_available_dem_resolutions = dc.get_available_dem_resolutions
reproject_vector_to_match_dem = utils.reproject_vector_to_match_dem
create_hillshade              = hillshade_module.create_hillshade
style_rem                     = style_rem_module
REM_Calcs                     = rem_calcs_module


# Performance Timer
@contextmanager
def task_timer(label: str, status_box=None):
    start_time = time.time()
    msg_start = f"STARTED: {label}..."
    print(f"\n{msg_start}")
    if status_box:
        status_box.info(msg_start)
    yield
    end_time = time.time()
    duration = end_time - start_time
    msg_end = f"FINISHED: {label} in {duration:.2f} seconds"
    print(f"{msg_end}\n")
    if status_box:
        status_box.success(msg_end)


# DEM Health Checker
#Opens the DEM and returns critical stats
def inspect_dem_metadata(dem_path: str):
    try:
        with rasterio.open(dem_path) as src:
            res_x, res_y = src.res
            
            # Quick Statistics (on a 10% subsample for speed)
            df = max(1, int(max(src.width, src.height) / 1000))
            data = src.read(
                1, 
                out_shape=(1, int(src.height/df), int(src.width/df)),
                resampling=Resampling.bilinear
            )
            
            if src.nodata is not None:
                valid = data[data != src.nodata]
            else:
                valid = data.flatten()
            
            valid = valid[np.isfinite(valid)]
                
            if valid.size == 0:
                return {"status": "WARNING", "res": (res_x, res_y), "crs": str(src.crs), "msg": "Only NoData found in preview (might be sparse data)"}
            
            min_val = float(np.min(valid))
            max_val = float(np.max(valid))
            
            return {
                "status": "OK",
                "res": (res_x, res_y),
                "crs": str(src.crs),
                "min": min_val,
                "max": max_val,
                "width": src.width,
                "height": src.height
            }
    except Exception as e:
        return {"status": "ERROR", "msg": str(e)}


# Fast QA/QC Plotting
#Generates PNG for QA/QC in Streamlit
def generate_qa_png(hillshade_path: str, river_path: str, output_png: str, aoi_path: str = None):
    # Read Hillshade
    with rasterio.open(hillshade_path) as src:
        df = max(1, int(max(src.width, src.height) / 4000))
        out_shape = (1, int(src.height / df), int(src.width / df))
        hs = src.read(1, out_shape=out_shape, resampling=Resampling.lanczos)

        if src.nodata is not None:
            hs = np.ma.masked_equal(hs, src.nodata)

        xmin, ymin, xmax, ymax = src.bounds
        hs_crs = src.crs

        if np.ma.is_masked(hs):
             valid_data = hs.compressed()
        else:
             valid_data = hs.flatten()

        valid_data = valid_data[np.isfinite(valid_data)]

        if valid_data.size > 0:
            vmin, vmax = np.nanpercentile(valid_data, (2, 98))
        else:
            vmin, vmax = 0, 255

    # Read River
    gdf = gpd.read_file(river_path)
    if not gdf.empty:
        gdf = gdf.to_crs(hs_crs)

    # Plot to file
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.imshow(hs, extent=(xmin, xmax, ymin, ymax), cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation='bilinear')

    # Plot river
    if not gdf.empty:
        gdf.plot(ax=ax, color="red", linewidth=1.5, alpha=0.9)

    ax.set_title("QA/QC: DEM Hillshade + River", fontsize=14, fontweight='bold')
    ax.set_axis_off()

    plt.savefig(output_png, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)
    return output_png


# Map & UI Helpers
#Validates a DEM file and returns detailed information or errors
def path_exists(p: str) -> bool:
    return bool(p) and os.path.exists(os.path.expanduser(p))

def validate_dem_file(dem_path: str) -> dict:
    result = {'valid': False, 'error': None, 'info': {}}

    try:
        # Expand path
        dem_path = os.path.expanduser(dem_path)

        # Check exists
        if not os.path.exists(dem_path):
            result['error'] = f" DEM file not found: {dem_path}"
            return result

        # Check readable
        if not os.access(dem_path, os.R_OK):
            result['error'] = f" DEM file not readable (permission denied): {dem_path}"
            return result

        # Try to open with rasterio
        try:
            with rasterio.open(dem_path) as src:
                # Check has valid CRS
                if src.crs is None:
                    result['error'] = " DEM has no coordinate reference system (CRS). Please ensure the file has valid projection information."
                    return result

                # Check dimensions
                if src.width <= 0 or src.height <= 0:
                    result['error'] = f" DEM has invalid dimensions: {src.width} x {src.height}"
                    return result

                # Check has data (sample center)
                try:
                    sample = src.read(1, window=((src.height//2, src.height//2 + 10), (src.width//2, src.width//2 + 10)))
                    if src.nodata is not None:
                        valid_data = sample[sample != src.nodata]
                    else:
                        valid_data = sample.flatten()

                    valid_data = valid_data[np.isfinite(valid_data)]

                    if valid_data.size == 0:
                        result['error'] = " DEM appears to contain only NoData values in sampled area. File may be empty or corrupted."
                        return result
                except Exception as e:
                    result['error'] = f" Could not read DEM data: {str(e)}"
                    return result

                # Success - extract info
                bounds = src.bounds
                res_x, res_y = src.res

                # Check CRS type
                is_geographic = utils.is_geographic_crs(src.crs)
                crs_type = "Geographic (degrees)" if is_geographic else "Projected (meters)"

                result['valid'] = True
                result['info'] = {
                    'width': src.width,
                    'height': src.height,
                    'resolution': (abs(res_x), abs(res_y)),
                    'crs': str(src.crs),
                    'crs_type': crs_type,
                    'is_geographic': is_geographic,
                    'bounds': {
                        'west': bounds.left,
                        'south': bounds.bottom,
                        'east': bounds.right,
                        'north': bounds.top
                    },
                    'nodata': src.nodata,
                    'dtype': str(src.dtypes[0]),
                    'file_size_mb': os.path.getsize(dem_path) / (1024 * 1024)
                }

        except rasterio.errors.RasterioIOError as e:
            result['error'] = f" Not a valid raster file or unsupported format: {str(e)}"
            return result

    except Exception as e:
        result['error'] = f" Unexpected error validating DEM: {str(e)}"
        return result

    return result


# Validates a river vector file and returns detailed information or errors
def validate_river_file(river_path: str) -> dict:
    result = {'valid': False, 'error': None, 'info': {}}

    try:
        # Expand path
        river_path = os.path.expanduser(river_path)

        # Check exists
        if not os.path.exists(river_path):
            result['error'] = f" River file not found: {river_path}"
            return result

        # Check readable
        if not os.access(river_path, os.R_OK):
            result['error'] = f" River file not readable (permission denied): {river_path}"
            return result

        # For shapefiles, check companion files
        if river_path.lower().endswith('.shp'):
            base_path = river_path[:-4]  # Remove .shp
            required_files = ['.shx', '.dbf']
            missing_files = []

            for ext in required_files:
                if not os.path.exists(base_path + ext):
                    missing_files.append(base_path + ext)

            if missing_files:
                result['error'] = f" Shapefile missing required companion files: {', '.join(missing_files)}"
                return result

        # Try to open with geopandas
        try:
            gdf = gpd.read_file(river_path)

            # Check not empty
            if len(gdf) == 0:
                result['error'] = " River file contains no features (empty dataset)"
                return result

            # Check has valid CRS
            if gdf.crs is None:
                result['error'] = " River file has no coordinate reference system (CRS). Please ensure the file has valid projection information."
                return result

            # Check geometry types
            geom_types = gdf.geometry.geom_type.unique()
            if 'LineString' not in geom_types and 'MultiLineString' not in geom_types:
                result['error'] = f" River file must contain LineString or MultiLineString geometries. Found: {', '.join(geom_types)}"
                return result

            # Check for null geometries
            null_count = gdf.geometry.isna().sum()
            if null_count == len(gdf):
                result['error'] = " All features have null/invalid geometries"
                return result

            # Success - extract info
            bounds = gdf.total_bounds

            result['valid'] = True
            result['info'] = {
                'feature_count': len(gdf),
                'crs': str(gdf.crs),
                'bounds': {
                    'west': bounds[0],
                    'south': bounds[1],
                    'east': bounds[2],
                    'north': bounds[3]
                },
                'geometry_types': ', '.join(geom_types),
                'null_geometries': null_count,
                'columns': list(gdf.columns),
                'file_size_mb': os.path.getsize(river_path) / (1024 * 1024)
            }

        except Exception as e:
            result['error'] = f" Not a valid vector file or unsupported format: {str(e)}"
            return result

    except Exception as e:
        result['error'] = f" Unexpected error validating river file: {str(e)}"
        return result

    return result


# Fetches DEM tile footprints from USGS TNM API and ScienceBase (for Alaska IfSAR) for visualization.
# If aoi_gem is provided, only returns tiles that intersect with the AOI.
# Bounds: south, west, north, east. Exact viewport bounds
def fetch_dem_coverage_footprints(bounds: tuple, radius_km: float = 100, aoi_geom=None) -> dict:

    # Use exact viewport bounds - don't expand with radius
    # bounds format: (south, west, north, east)
    south, west, north, east = bounds

    # Convert to API format: (minx, miny, maxx, maxy) = (west, south, east, north)
    bbox_str = f"{west},{south},{east},{north}"

    # For center calculation (used for region detection)
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    # Alaska vs CONUS detection
    is_alaska_region = center_lat > 50.0

    # Build resolution keywords based on location
    # Alaska has unique "13 arc-second" product that is 5m
    # 1/3 arc-second is 10m everywhere (including Alaska)
    if is_alaska_region:
        resolution_keywords = {
            1: ["1 meter", "1m", "one meter"],
            3: ["3 meter", "3m", "1/9 arc-second", "1/9 arc second"],
            5: ["5 meter", "5m", "alaska 5 meter", "ak_ifsar", "ifsar", "5m ifsar", "13 arc-second", "13 arc second", "13 arc"],
            10: ["10 meter", "10m", "1/3 arc-second", "1/3 arc second", "0.33 arc"],
            30: ["30 meter", "30m", "1 arc-second", "1 arc second"],
        }
    else:
        resolution_keywords = {
            1: ["1 meter", "1m", "one meter"],
            3: ["3 meter", "3m", "1/9 arc-second", "1/9 arc second"],
            5: ["5 meter", "5m"],
            10: ["10 meter", "10m", "1/3 arc-second", "1/3 arc second", "0.33 arc"],
            30: ["30 meter", "30m", "1 arc-second", "1 arc second"],
        }

    tiles_by_resolution = {res: [] for res in resolution_keywords.keys()}

    # Query USGS TNM API
    try:
        api_url = "https://tnmaccess.nationalmap.gov/api/v1/products"
        params = {"bbox": bbox_str, "prodFormats": "GeoTIFF", "max": 1000}

        r = requests.get(api_url, params=params, timeout=30)
        if r.status_code == 200:
            items = r.json().get("items", [])

            for item in items:
                title = item.get("title", "").lower()
                bbox_item = item.get("boundingBox", {})
                pub_date = item.get("publicationDate", "Unknown")[:10]

                if not bbox_item:
                    continue

                # Skip S1M tiles
                is_s1m = "s1m" in title or "standard 1-meter" in title
                if is_s1m:
                    continue

                for res, keywords in resolution_keywords.items():
                    if any(kw in title for kw in keywords):
                        try:
                            tgeom = shp_box(
                                float(bbox_item["minX"]), float(bbox_item["minY"]),
                                float(bbox_item["maxX"]), float(bbox_item["maxY"])
                            )
                            tiles_by_resolution[res].append({
                                "geometry": tgeom,
                                "title": item.get("title", "")[:50] + "...",
                                "date": pub_date
                            })
                        except:
                            pass
                        break
    except Exception as e:
        st.warning(f"USGS TNM API error: {e}")

    # Query ScienceBase for Alaska IfSAR 5m tiles
    if is_alaska_region:
        try:
            viewport_geom = viewport_box(west, south, east, north)

            sb_url = "https://www.sciencebase.gov/catalog/items"
            sb_params = {
                "parentId": "5641fe98e4b0831b7d62e758",  # Alaska IfSAR collection
                "max": 1000,
                "format": "json",
                "fields": "title,spatial,webLinks",
                "bbox": bbox_str
            }

            r_sb = requests.get(sb_url, params=sb_params, timeout=30)

            if r_sb.status_code == 200:
                sb_items = r_sb.json().get("items", [])

                for item in sb_items:
                    try:
                        title = item.get("title", "")
                        spatial = item.get("spatial", {})
                        bbox_item = spatial.get("boundingBox", {})

                        if not bbox_item:
                            continue

                        tgeom = shp_box(
                            float(bbox_item["minX"]), float(bbox_item["minY"]),
                            float(bbox_item["maxX"]), float(bbox_item["maxY"])
                        )

                        # Only include if tile actually intersects with viewport
                        if tgeom.intersects(viewport_geom):
                            tiles_by_resolution[5].append({
                                "geometry": tgeom,
                                "title": title[:50] + "..." if len(title) > 50 else title,
                                "date": "IfSAR"
                            })
                    except:
                        continue
        except:
            pass

    # Filter tiles by AOI intersection if AOI is provided
    if aoi_geom is not None:
        filtered_tiles = {res: [] for res in tiles_by_resolution.keys()}
        total_before = sum(len(tiles) for tiles in tiles_by_resolution.values())

        for res, tiles in tiles_by_resolution.items():
            for tile in tiles:
                if tile["geometry"].intersects(aoi_geom):
                    filtered_tiles[res].append(tile)

        total_after = sum(len(tiles) for tiles in filtered_tiles.values())
        if total_before > total_after:
            st.success(f"✂️ Filtered from {total_before} to {total_after} tiles (only showing tiles that would be downloaded for your AOI)")
        else:
            st.info(f" All {total_after} tiles intersect your AOI")

        return filtered_tiles

    return tiles_by_resolution


# Build foliurm map with optional DEM coverage overlay.
def build_aoi_map(center: List[float], zoom_start: int = 8, coverage_data: dict = None, show_coverage_layers: bool = True) -> folium.Map:
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Aerial", control=True, show=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Topo", control=True, show=False
    ).add_to(m)

    # Add DEM coverage layers if data exists and layers are enabled
    if coverage_data and show_coverage_layers:
        # Color scheme
        colors = {
            1: "#0066CC",  # Blue - 1m 
            3: "#00CC66",  # Green - 3m
            5: "#00CCCC",  # Cyan - 5m 
            10: "#FFCC00", # Yellow - 10m
            30: "#FF6600"  # Orange - 30m
        }

        # Add each resolution as a toggleable layer
        for res in sorted(colors.keys()):
            tiles = coverage_data.get(res, [])
            if not tiles:
                continue

            # Show all coverage layers by default
            layer = folium.FeatureGroup(name=f"{res}m DEM ({len(tiles)} tiles)", show=True)

            for tile in tiles:
                # Convert shapely geometry to GeoJSON
                geom_json = {
                    "type": "Polygon",
                    "coordinates": [[
                        [tile["geometry"].bounds[0], tile["geometry"].bounds[1]],
                        [tile["geometry"].bounds[2], tile["geometry"].bounds[1]],
                        [tile["geometry"].bounds[2], tile["geometry"].bounds[3]],
                        [tile["geometry"].bounds[0], tile["geometry"].bounds[3]],
                        [tile["geometry"].bounds[0], tile["geometry"].bounds[1]]
                    ]]
                }

                # Make 5m tiles more visible with thicker borders and higher opacity
                if res == 5:
                    folium.GeoJson(
                        geom_json,
                        style_function=lambda x: {
                            'fillColor': '#00FFFF',
                            'color': '#00FFFF',
                            'weight': 2,
                            'fillOpacity': 0.4,
                            'interactive': False
                        },
                        highlight_function=None,
                        tooltip=None,
                        popup=None
                    ).add_to(layer)
                else:
                    folium.GeoJson(
                        geom_json,
                        style_function=lambda x, color=colors[res]: {
                            'fillColor': color,
                            'color': color,
                            'weight': 1,
                            'fillOpacity': 0.25,
                            'interactive': False
                        },
                        highlight_function=None,
                        tooltip=None,
                        popup=None
                    ).add_to(layer)

            layer.add_to(m)

    Draw(export=False, position="topleft", draw_options={"polyline":False,"circle":False,"marker":False,"circlemarker":False}).add_to(m)
    Fullscreen().add_to(m)
    Geocoder().add_to(m)
    MeasureControl().add_to(m)
    folium.LayerControl(position='topright', collapsed=True).add_to(m)
    return m

def extract_aoi_from_map(map_data: Dict[str, Any]) -> Optional[gpd.GeoDataFrame]:
    if not map_data: return None
    features = None
    if "all_drawings" in map_data and map_data["all_drawings"]:
        raw = map_data["all_drawings"]
        features = raw.get("features") if isinstance(raw, dict) else raw
    elif "last_active_drawing" in map_data and map_data["last_active_drawing"]:
        features = [map_data["last_active_drawing"]]
    if not features: return None
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

def save_aoi_geojson(aoi_gdf: gpd.GeoDataFrame, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    aoi_gdf.to_file(out_path, driver="GeoJSON")
    return out_path



# UI Layout
col_left, col_right = st.columns([1.3, 1.0])

with col_left:
    st.subheader("Project & AOI / Input Settings")
    project_folder = st.text_input("Project folder", "./Project", disabled=st.session_state["run_requested"])
    project_folder = os.path.expanduser(project_folder)
    os.makedirs(project_folder, exist_ok=True)

    run_mode = st.radio("Run mode", ["AOI + NHD (3DEP download)", "Custom DEM & River"], disabled=st.session_state["run_requested"])
    
    map_data: Dict[str, Any] = {}
    dem_file_input = ""
    river_vector_input = ""
    dem_res_input = ""
    selected_river_name = None

    if run_mode == "AOI + NHD (3DEP download)":
        st.markdown("---")
        st.markdown("### AOI Drawing")

        # Map render (must come before controls to get bounds)
        # Use saved map position if there, otherwise use default
        map_center = st.session_state.get("map_center", [39.8283, -98.5795])
        map_zoom = st.session_state.get("map_zoom", 5)

        aoi_map = build_aoi_map(
            center=map_center,
            zoom_start=map_zoom,
            coverage_data=st.session_state["coverage_data"],
            show_coverage_layers=st.session_state["show_coverage_layers"]
        )
        map_data = st_folium(aoi_map, width=700, height=500, key="aoi_map")

        # Update AOI in session state if user has drawn/deleted it
        current_aoi_gdf = extract_aoi_from_map(map_data)
        if current_aoi_gdf is not None and not current_aoi_gdf.empty:
            st.session_state["aoi_gdf"] = current_aoi_gdf
        elif map_data and map_data.get("all_drawings") is not None:
            # User explicitly cleared drawings
            if "aoi_gdf" in st.session_state:
                del st.session_state["aoi_gdf"]

        # Persistent warning if coverage data exists but no AOI in session state
        if st.session_state["coverage_data"]:
            if "aoi_gdf" not in st.session_state or st.session_state["aoi_gdf"] is None:
                total_tiles = sum(len(tiles) for tiles in st.session_state["coverage_data"].values())
                if total_tiles > 0:
                    st.warning(f" Showing all {total_tiles} tiles in viewport. Draw an AOI polygon to see only tiles that would be downloaded.")

        # Coverage controls
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Scan Current Map Area", disabled=st.session_state["run_requested"], help="Query USGS for available DEM tiles ONLY within your current map viewport. Pan/zoom and rescan as needed!"):
                # Get current map bounds from map_data
                if map_data and map_data.get("bounds"):
                    bounds_data = map_data["bounds"]
                    # bounds_data format: {'_southWest': {'lat': x, 'lng': y}, '_northEast': {'lat': x, 'lng': y}}
                    sw = bounds_data.get("_southWest", {})
                    ne = bounds_data.get("_northEast", {})

                    if sw and ne:
                        # Create bounds tuple: (south, west, north, east)
                        bounds = (sw.get("lat"), sw.get("lng"), ne.get("lat"), ne.get("lng"))

                        # Save current map position before rerun
                        center_lat = (sw.get("lat") + ne.get("lat")) / 2
                        center_lng = (sw.get("lng") + ne.get("lng")) / 2
                        st.session_state["map_center"] = [center_lat, center_lng]
                        if map_data.get("zoom"):
                            st.session_state["map_zoom"] = map_data["zoom"]

                        # Use AOI from session state
                        aoi_gdf = st.session_state.get("aoi_gdf")
                        aoi_geom = None
                        if aoi_gdf is not None and not aoi_gdf.empty:
                            aoi_geom = unary_union(aoi_gdf.geometry)

                        # Query using exact viewport bounds
                        coverage = fetch_dem_coverage_footprints(bounds, aoi_geom=aoi_geom)
                        st.session_state["coverage_data"] = coverage
                        st.session_state["show_coverage_layers"] = True

                        total_tiles = sum(len(tiles) for tiles in coverage.values())
                        if total_tiles > 0:
                            # Show breakdown by resolution
                            breakdown = ", ".join([f"{res}m: {len(tiles)}" for res, tiles in sorted(coverage.items()) if tiles])
                            st.success(f" Found {total_tiles} DEM tiles: {breakdown}")

                                            # Highlight 5m tiles if found and show location
                            if coverage.get(5):
                                # Calculate extent of 5m coverage
                                all_bounds = [tile["geometry"].bounds for tile in coverage[5]]
                                min_lon = min(b[0] for b in all_bounds)
                                min_lat = min(b[1] for b in all_bounds)
                                max_lon = max(b[2] for b in all_bounds)
                                max_lat = max(b[3] for b in all_bounds)
                                center_5m_lat = (min_lat + max_lat) / 2
                                center_5m_lon = (min_lon + max_lon) / 2

                                st.info(f"Found {len(coverage[5])} 5m IfSAR tiles")
                                st.warning(f" 5m coverage center: Lat {center_5m_lat:.2f}, Lon {center_5m_lon:.2f} - Pan map to this area if you don't see cyan tiles")

                            st.info(" Click the layers icon in top-right to toggle resolutions. Tiles are visible on map.")
                            st.rerun()  # Force map to rerender with new coverage data
                        else:
                            st.warning("No DEM tiles found in current map area. Try zooming out or panning to a different location.")
                    else:
                        st.error("Could not determine map bounds. Try panning/zooming the map first.")
                else:
                    st.warning("Map not ready yet. Pan or zoom the map, then try scanning again.")

            st.caption(" Tip: Pan/zoom to area of interest, then scan to see ONLY tiles in current viewport")

        with col2:
            show_layers = st.checkbox(
                "Show Coverage Layers",
                value=st.session_state["show_coverage_layers"],
                disabled=st.session_state["coverage_data"] is None or st.session_state["run_requested"],
                help="Toggle visibility of DEM coverage footprints"
            )
            st.session_state["show_coverage_layers"] = show_layers

        # Coverage legend
        if st.session_state["coverage_data"] and st.session_state["show_coverage_layers"]:
            st.caption("**Coverage Legend:** 🔵1m | 🟢 3m | 🔷 5m  | 🟡 10m | 🟠 30m")

        st.markdown("### Resolution & River")

        # Button to scan for rivers AND check available resolutions
        if st.button("Scan AOI for Rivers & Resolutions", disabled=st.session_state["run_requested"]):
            # Use AOI from session state (persists across reruns)
            temp_aoi = st.session_state.get("aoi_gdf")
            if temp_aoi is not None and not temp_aoi.empty:
                tpath = os.path.join(project_folder, "temp_scan.geojson")
                save_aoi_geojson(temp_aoi, tpath)

                # Scan for rivers
                with st.spinner("Scanning for rivers..."):
                    rlist = scan_nhd_rivers(tpath)
                    st.session_state["scanned_river_list"] = rlist
                    if not rlist:
                        st.warning("No rivers found.")

                # Check available DEM resolutions
                with st.spinner("Checking available DEM resolutions..."):
                    avail_res = get_available_project_resolutions(tpath)
                    st.session_state["available_resolutions"] = avail_res
                    if not avail_res:
                        st.warning("No Project tiles found. WCS tiles (lower quality) available as fallback.")
                    else:
                        st.success(f" Available: {avail_res}")
            else:
                st.error("Draw AOI first.")


        # Resolution dropdown (only show if resolutions are available)
        if st.session_state["available_resolutions"]:
            dem_res_input = st.selectbox(
                "DEM Resolution (Project tiles - high quality)",
                options=st.session_state["available_resolutions"],
                format_func=lambda x: f"{x}m",
                disabled=st.session_state["run_requested"],
                help="Only showing resolutions where Project tiles cover >30% of AOI"
            )
        elif st.session_state["scanned_river_list"]:  # AOI scanned but no Project tiles
            st.warning("No Project tiles available. Using WCS fallback.")
            dem_res_input = st.selectbox(
                "DEM Resolution (WCS - lower quality)",
                options=[10, 30],
                format_func=lambda x: f"{x}m (WCS fallback)",
                disabled=st.session_state["run_requested"]
            )
        else:
            st.info("Scan AOI to see available resolutions")
            dem_res_input = None

        # River selection
        if st.session_state["scanned_river_list"]:
            selected_river_name = st.selectbox("Choose River:", st.session_state["scanned_river_list"], disabled=st.session_state["run_requested"])
    
    else:
        st.markdown("---")
        st.markdown("### Choose Input Method")

        # Create Tabs for the two input methods
        tab_path, tab_upload = st.tabs([" Option A: Local File Path (Unlimited Size)", "☁️ Option B: Browser Upload"])

        # TAB 1: Manual File Path
        with tab_path:
            st.info(" **Recommended for large files (e.g. >2GB)")
            
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                # Manual DEM Path Input
                local_dem_in = st.text_input("Paste Local DEM Path", 
                                           value=st.session_state.get("uploaded_dem_path", ""),
                                           placeholder=r"e.g. C:\Data\Big_DEM.tif",
                                           disabled=st.session_state["run_requested"])
                
                # Logic to handle path entry
                if local_dem_in:
                    # Remove quotes if user used "Copy as path" in Windows
                    clean_path = local_dem_in.strip('"').strip("'")
                    if os.path.exists(clean_path):
                        # Only update if changed
                        if st.session_state.get("uploaded_dem_path") != clean_path:
                            st.session_state["uploaded_dem_path"] = clean_path
                            st.session_state["custom_dem_validated"] = False # Reset validation
                            st.rerun()
                    else:
                        st.error(" File not found. Check the path.")

            with col_p2:
                # Manual River Path Input
                local_riv_in = st.text_input("Paste Local River Path",
                                           value=st.session_state.get("uploaded_river_path", ""),
                                           placeholder=r"e.g. C:\Data\River.shp",
                                           disabled=st.session_state["run_requested"])

                if local_riv_in:
                    clean_path = local_riv_in.strip('"').strip("'")
                    if os.path.exists(clean_path):
                        if st.session_state.get("uploaded_river_path") != clean_path:
                            st.session_state["uploaded_river_path"] = clean_path
                            st.session_state["custom_river_validated"] = False # Reset validation
                            st.rerun()
                    else:
                        st.error(" File not found. Check the path.")

        # TAB 2: Browser Upload
        with tab_upload:
            uploaded_dem = st.file_uploader("Upload DEM file", type=["tif", "tiff"], disabled=st.session_state["run_requested"], help="Upload a new file to replace the current one")
            uploaded_river = st.file_uploader("Upload River file", type=["shp", "gpkg", "geojson"], disabled=st.session_state["run_requested"], help="Upload a new file to replace the current one")

            # Save uploaded files to project folder if provided
            if uploaded_dem:
                upload_path = os.path.join(project_folder, uploaded_dem.name)
                # Only save and validate if it's a NEW file
                if st.session_state.get("uploaded_dem_path") != upload_path:
                    os.makedirs(project_folder, exist_ok=True)
                    with open(upload_path, "wb") as f:
                        f.write(uploaded_dem.getbuffer())
                    st.session_state["uploaded_dem_path"] = upload_path
                    st.success(f" DEM uploaded: {uploaded_dem.name}")

                    # Auto validate DEM immediately after upload
                    dem_result = validate_dem_file(upload_path)
                    if dem_result['valid']:
                        st.session_state["custom_dem_validated"] = True
                        st.session_state["custom_dem_info"] = dem_result['info']
                    else:
                        st.session_state["custom_dem_validated"] = False
                        st.session_state["custom_dem_info"] = {'error': dem_result['error']}
                    st.rerun()

            if uploaded_river:
                upload_path = os.path.join(project_folder, uploaded_river.name)
                # Only save and validate if it's a NEW file
                if st.session_state.get("uploaded_river_path") != upload_path:
                    os.makedirs(project_folder, exist_ok=True)
                    with open(upload_path, "wb") as f:
                        f.write(uploaded_river.getbuffer())
                    st.session_state["uploaded_river_path"] = upload_path
                    st.success(f" River uploaded: {uploaded_river.name}")

                    # Auto validate River immediately after upload
                    river_result = validate_river_file(upload_path)
                    if river_result['valid']:
                        st.session_state["custom_river_validated"] = True
                        st.session_state["custom_river_info"] = river_result['info']
                    else:
                        st.session_state["custom_river_validated"] = False
                        st.session_state["custom_river_info"] = {'error': river_result['error']}
                    st.rerun()

        # This section checks what is loaded (whether via Path or Upload) and shows validation status
        # Get file paths from session state
        existing_dem = st.session_state.get("uploaded_dem_path") or st.session_state.get("dem_file")
        existing_river = st.session_state.get("uploaded_river_path") or st.session_state.get("river_vector_path")

        # Auto-validate existing files if not already validated
        if existing_dem and os.path.exists(existing_dem):
            st.info(f" DEM loaded: {os.path.basename(existing_dem)}")
            
            # Ensure session state is synced
            if not st.session_state.get("uploaded_dem_path"):
                st.session_state["uploaded_dem_path"] = existing_dem

            # Validation Check
            if not st.session_state.get("custom_dem_validated"):
                dem_result = validate_dem_file(existing_dem)
                if dem_result['valid']:
                    st.session_state["custom_dem_validated"] = True
                    st.session_state["custom_dem_info"] = dem_result['info']
                else:
                    st.session_state["custom_dem_validated"] = False
                    st.session_state["custom_dem_info"] = {'error': dem_result['error']}

        if existing_river and os.path.exists(existing_river):
            st.info(f" River loaded: {os.path.basename(existing_river)}")
            
            # Ensure session state is synced
            if not st.session_state.get("uploaded_river_path"):
                st.session_state["uploaded_river_path"] = existing_river

            # Validation Check
            if not st.session_state.get("custom_river_validated"):
                river_result = validate_river_file(existing_river)
                if river_result['valid']:
                    st.session_state["custom_river_validated"] = True
                    st.session_state["custom_river_info"] = river_result['info']
                else:
                    st.session_state["custom_river_validated"] = False
                    st.session_state["custom_river_info"] = {'error': river_result['error']}

        # Update input variables for the rest of the script
        dem_file_input = st.session_state.get("uploaded_dem_path", "")
        river_vector_input = st.session_state.get("uploaded_river_path", "")

        # Validation Status Display
        st.markdown("---")

        dem_info = st.session_state.get("custom_dem_info", {})
        river_info = st.session_state.get("custom_river_info", {})

        dem_valid = st.session_state.get("custom_dem_validated", False)
        river_valid = st.session_state.get("custom_river_validated", False)

        if dem_info or river_info:
            if dem_valid and river_valid:
                st.success(" **Files are valid and ready to use**")
                
                # Check geographic CRS
                if dem_info.get('is_geographic', False):
                    st.info("ℹ️ Your DEM uses geographic coordinates (degrees). The system will automatically reproject it to meters (UTM) during processing.")
            
            elif dem_info or river_info:
                # Show specific errors
                if dem_info and 'error' in dem_info:
                    st.error(f" **DEM:** {dem_info['error']}")
                elif dem_valid:
                    st.success(" **DEM:** Valid")

                if river_info and 'error' in river_info:
                    st.error(f" **River:** {river_info['error']}")
                elif river_valid:
                    st.success(" **River:** Valid")

    st.markdown("---")
    st.markdown("### Settings")

    # Information about settings
    with st.expander("ℹ️ What do these settings mean?", expanded=False):
        st.markdown("""
        **Spacing (m):** Controls the distance between sampling points along the river. Lower values increase accuracy but significantly increase computation time and memory usage.

        **K Neighbors (IDW):** Defines how many nearby river points influence each terrain cell during interpolation. Higher values create smoother REMs but may lose fine detail. Range: 4 (sharp/detailed) to 300 (very smooth). Default 8 works well for most cases.

        **CPU Usage:** Controls how many processor cores are used for calculations. Higher values speed up processing but may slow down other applications.
        - **Low (25-50%):** Use while working on other tasks. Keeps computer responsive.
        - **Medium (60-75%):** Recommended default. Good balance of speed and responsiveness.
        - **High (80-100%):** Fastest processing. Best for dedicated/overnight runs.

        Tip: Start at 75%. Only lower if your computer feels sluggish during processing.

        **MAX Rem (m) - Comp:** Sets the maximum relative elevation value to compute (in meters). Areas higher than this threshold are clipped, reducing processing time and focusing on the floodplain area of interest.
        """)

    spacing = st.number_input("Spacing (m)", min_value=1, max_value=500, value=20, step=1, disabled=st.session_state["run_requested"])

    k_neighbors = st.number_input("K Neighbors (IDW)", min_value=4, max_value=300, value=8, step=1, disabled=st.session_state["run_requested"])

    # This automatically scales. 8 cores -> 4 threads. 16 cores -> 8 threads
    cpu_util_percent = st.slider("CPU Usage", 10, 100, 75, 5, disabled=st.session_state["run_requested"], format="%d%%", help="Recommended: 75%. Lower if computer feels sluggish.")
    cpu_util = cpu_util_percent / 100.0

    # Warning for very high CPU usage
    if cpu_util_percent >= 95 and not st.session_state["run_requested"]:
        st.warning(" **100% CPU may make your system unresponsive during processing.** Recommended: 75%")

    comp_max_m = st.text_input("Max REM (m) - Comp", "", disabled=st.session_state["run_requested"])
    st.markdown("### Visualization")

    # Information about visualization settings
    with st.expander("ℹ️ What do these visualization settings mean?", expanded=False):
        st.markdown("""
        **Color Ramps:** Choose color schemes for your REM visualization. Different ramps highlight different features - select multiple to generate several versions for comparison.

        **Background:** Layer displayed behind the REM. Hillshade shows terrain relief, aerial shows satellite imagery, and white provides a clean backdrop for presentations.

        **BG Transparency:** Controls how visible the background layer is. Lower values make the background more transparent, emphasizing the REM data.

        **REM Transparency:** Controls opacity of the REM color overlay. Lower values let more background show through, useful for seeing terrain context.

        **Max REM (m) - Vis:** Maximum elevation to display in visualizations (in meters). Values above this are shown at the maximum color, helping focus attention on lower floodplain areas.

        **Discrete Colors:** Enable distinct color bands instead of smooth gradient. When enabled, the system automatically picks 5-8 classes based on your max REM value for optimal readability. Creates clear elevation zones useful for identifying specific flood heights.
        """)

    selected_ramps = st.multiselect("Color Ramps", COLOR_RAMP_OPTIONS, default=[COLOR_RAMP_OPTIONS[0]], disabled=st.session_state["run_requested"])
    bg_type = st.selectbox("Background", BACKGROUND_OPTIONS, disabled=st.session_state["run_requested"])
    bg_alpha = st.slider("BG Transparency", 0.0, 1.0, 0.5, disabled=st.session_state["run_requested"])
    rem_alpha = st.slider("REM Transparency", 0.0, 1.0, 1.0, disabled=st.session_state["run_requested"])
    viz_max_m = st.text_input("Max REM (m) - Vis", "", disabled=st.session_state["run_requested"])
    use_discrete_colors = st.checkbox("Use Discrete Colors", value=False, disabled=st.session_state["run_requested"], help="Automatically creates 5-8 distinct color bands for clear elevation zones")

    st.markdown("---")
    if st.button("Run REM Pipeline", use_container_width=True, disabled=st.session_state["run_requested"]):
        clear_stop_flag()  # Clear any previous stop signals
        st.session_state["run_requested"] = True
        st.session_state["prep_done"] = False
        st.session_state["dem_acquired"] = False  # Reset DEM acquisition on new run
        st.session_state["hillshade_qa_generated"] = False  # Reset hillshade on new run
        st.session_state["dem_validated"] = False  # Reset validation on new run
        st.session_state["memory_checked"] = False  # Reset memory check on new run
        st.session_state["qa_approved"] = False
        st.session_state["rem_done"] = False
        st.session_state["pipeline_start_time"] = time.time()
        st.rerun()

    # Stop button - visible during active processing
    if st.session_state.get("run_requested") and not st.session_state.get("rem_done"):
        if st.button("STOP ALL PROCESSES", use_container_width=True, type="secondary", key="stop_left", help="IMMEDIATELY STOP all processing - kills current operations mid-execution!"):
            set_stop_flag()  # Signal immediate stop to all processes
            st.session_state["run_requested"] = False
            st.session_state["prep_done"] = False
            st.session_state["qa_approved"] = False
            st.session_state["rem_done"] = False
            st.warning(" **Process Stopped** - Current step will complete, then processing halts. You can modify settings and run again. Previously downloaded/generated files are preserved.")
            st.rerun()

with col_right:
    st.subheader("Status & Outputs")

    # Stop Process button - visible during active processing
    if st.session_state.get("run_requested") and not st.session_state.get("rem_done"):
        st.markdown("---")
        if st.button(" STOP ALL PROCESSES", use_container_width=True, type="secondary", key="stop_right", help="IMMEDIATELY STOP all processing - kills current operations mid-execution!"):
            set_stop_flag()  # Signal immediate stop to all processes
            # Reset all session state flags to stop processing
            st.session_state["run_requested"] = False
            st.session_state["prep_done"] = False
            st.session_state["qa_approved"] = False
            st.session_state["rem_done"] = False
            # Note: We do NOT reset dem_acquired, hillshade_qa_generated, dem_validated, memory_checked
            # This preserves work done so far if they want to retry
            st.error("**STOP PROCESSES KILLED** - All processing halted immediately! Previously generated files are preserved.")
            st.rerun()
        st.markdown("---")

    status_box = st.empty()
    qc_box = st.empty()
    rem_img_box = st.empty()
    path_box = st.empty()



# PIPELINE EXECUTION
if st.session_state.get("run_requested"):
    
    if not selected_ramps:
        st.error("Select a color ramp.")
        st.stop()

    # Split Core Logic
    total_cores = os.cpu_count() or 4
    
    # Compute Threads (Mosaicking, REM, Styling)
    # This respects the User Slider (e.g. 100% = All Cores)
    compute_threads = max(1, min(total_cores, int(round(total_cores * float(cpu_util)))))
    
    # Download Threads (Network)
    # ALWAYS cap at 50% of system cores to prevent network choking/zombies
    # (e.g. 8 cores -> 4 download threads. 16 cores -> 8 download threads)
    download_threads = max(1, total_cores // 2)
    
    status_box.info(f"Threads: {download_threads} (Download) / {compute_threads} (Compute)")

    # Set generic 'threads' variable to compute_threads for later functions (REM/Style)
    threads = compute_threads

    # Preperation
    if not st.session_state["prep_done"]:
        proj_dir = os.path.expanduser(project_folder)
        os.makedirs(proj_dir, exist_ok=True)
        st.session_state["project_folder_expanded"] = proj_dir

        dem_folder = proj_dir
        rem_folder = proj_dir
        visuals_folder = proj_dir
        extra_data_folder = proj_dir

        dem_file, river_path, aoi_geojson_path = None, None, None

        max_m_comp = float(comp_max_m) if comp_max_m.strip() else None
        max_m_vis = float(viz_max_m) if viz_max_m.strip() else None

        if run_mode == "Custom DEM & River":
            # Enhanced validation with detailed error messages and retry capability
            validation_failed = False
            error_messages = []

            # Check if files are provided
            if not dem_file_input:
                error_messages.append(" No DEM file provided")
                validation_failed = True
            if not river_vector_input:
                error_messages.append(" No river file provided")
                validation_failed = True

            # Check if files exist
            if dem_file_input and not path_exists(dem_file_input):
                error_messages.append(f" DEM file not found: {dem_file_input}")
                validation_failed = True
            if river_vector_input and not path_exists(river_vector_input):
                error_messages.append(f" River file not found: {river_vector_input}")
                validation_failed = True

            # Check if files have been validated (recommended but not required)
            if not validation_failed:
                if not st.session_state.get("custom_dem_validated") or not st.session_state.get("custom_river_validated"):
                    st.warning(" **Files not validated yet!** It's recommended to click ' Validate Files' to check compatibility before running.")
                    st.info("Proceeding anyway... If you encounter errors, go back and validate your files first.")

            # If validation failed, show errors and offer retry
            if validation_failed:
                st.error("**Cannot proceed - file validation failed:**")
                for msg in error_messages:
                    st.error(msg)

                st.markdown("---")
                st.markdown("### What to do:")
                st.markdown("1. **Go back** and check your file paths or upload the correct files")
                st.markdown("2. **Click ' Validate Files'** to verify your files are compatible")
                st.markdown("3. **Try again** after fixing the issues")

                col_retry1, col_retry2 = st.columns(2)
                with col_retry1:
                    if st.button(" Go Back & Fix Files", use_container_width=True):
                        st.session_state["run_requested"] = False
                        st.session_state["prep_done"] = False
                        st.session_state["custom_dem_validated"] = False
                        st.session_state["custom_river_validated"] = False
                        st.session_state["custom_dem_info"] = None
                        st.session_state["custom_river_info"] = None
                        st.rerun()

                with col_retry2:
                    if st.button(" Cancel Run", use_container_width=True):
                        st.session_state["run_requested"] = False
                        st.session_state["prep_done"] = False
                        st.rerun()

                st.stop()

            # Validation passed - proceed
            dem_file = os.path.expanduser(dem_file_input)
            river_path = os.path.expanduser(river_vector_input)
            dem_folder_for_run = os.path.dirname(dem_file) or proj_dir

            # Double-check files are actually accessible before proceeding
            try:
                with rasterio.open(dem_file) as _:
                    pass
            except Exception as e:
                st.error(f" **Failed to open DEM file:** {str(e)}")
                st.warning("The file exists but cannot be read. It may be corrupted or in an unsupported format.")
                if st.button(" Go Back & Fix Files", use_container_width=True):
                    st.session_state["run_requested"] = False
                    st.session_state["prep_done"] = False
                    st.rerun()
                st.stop()

            try:
                _ = gpd.read_file(river_path)
            except Exception as e:
                st.error(f" **Failed to open river file:** {str(e)}")
                st.warning("The file exists but cannot be read. It may be corrupted, missing companion files (.shx, .dbf for shapefiles), or in an unsupported format.")
                if st.button(" Go Back & Fix Files", use_container_width=True):
                    st.session_state["run_requested"] = False
                    st.session_state["prep_done"] = False
                    st.rerun()
                st.stop()

            # Smart CRS handling
            # Check if DEM is in geographic CRS (degrees) and reproject to UTM if needed
            with task_timer("Checking DEM CRS", status_box):
                with rasterio.open(dem_file) as src:
                    dem_crs = src.crs
                    dem_original_crs = str(dem_crs)  # Track original CRS
                    is_geographic = utils.is_geographic_crs(dem_crs)

                    if is_geographic:
                        status_box.warning(f" DEM is in geographic CRS ({dem_crs}). Converting to UTM for proper REM calculations...")

                        # Reproject DEM to appropriate UTM zone
                        reprojected_dem_path = os.path.join(proj_dir, "DEM_reprojected_UTM.tif")

                        with task_timer("Reprojecting DEM to UTM", status_box):
                            dem_file, target_epsg = utils.reproject_dem_to_utm(
                                dem_file,
                                reprojected_dem_path,
                                verbose=True
                            )

                        dem_output_crs = str(target_epsg)  # Track output CRS after reprojection
                        status_box.success(f" DEM converted from {dem_crs} to {target_epsg}")
                        st.info(f" **CRS Conversion Applied:** Your DEM was in geographic coordinates (degrees). It has been automatically reprojected to {target_epsg} (meters) for accurate REM calculations. Final outputs will remain in {target_epsg}.")
                    else:
                        dem_output_crs = dem_original_crs  # No reprojection, output = original
                        status_box.info(f" DEM is already in projected CRS ({dem_crs}) - no reprojection needed")

            # Store DEM CRS info in session state
            st.session_state["dem_original_crs"] = dem_original_crs
            st.session_state["dem_output_crs"] = dem_output_crs

        else:
            # Use AOI from session state (persists across reruns)
            aoi_gdf = st.session_state.get("aoi_gdf")
            if aoi_gdf is None or aoi_gdf.empty:
                st.error("No AOI.")
                st.stop()

            aoi_geojson_path = os.path.join(proj_dir, "AOI_streamlit.geojson")
            save_aoi_geojson(aoi_gdf, aoi_geojson_path)
            aoi_path = aoi_geojson_path  # Keep for backwards compatibility in this block
            
            # Resolution already selected from dropdown
            if dem_res_input is None:
                st.error("Please select a resolution")
                st.stop()

            res = int(dem_res_input)

            # Only download DEM if not already acquired
            # This prevents re-downloading when user retries river scan
            if not st.session_state["dem_acquired"]:
                with task_timer(f"Download {res}m Data", status_box):
                    mosaic = download_and_mosaic_dems(
                        [aoi_path],
                        proj_dir,
                        res,
                        n_jobs_download=download_threads, # Safe (50%)
                        n_jobs_mosaic=compute_threads     # Fast (User %)
                    )

                if not mosaic:
                    st.error(" DEM download failed after all retry attempts.")
                    st.warning("This may be due to network issues, server downtime, or no data available for this area/resolution.")

                    col1, col2, col3 = st.columns(3)
                    if col1.button(" Retry Same Settings", use_container_width=True):
                        st.session_state["prep_done"] = False
                        st.session_state["dem_acquired"] = False  # Allow DEM re-download
                        st.session_state["hillshade_qa_generated"] = False  # Regenerate hillshade for new DEM
                        st.session_state["dem_validated"] = False  # Re-validate new DEM
                        st.session_state["memory_checked"] = False  # Re-check memory for new DEM
                        st.rerun()

                    if col2.button(" Try Different Resolution", use_container_width=True):
                        st.session_state["run_requested"] = False
                        st.session_state["prep_done"] = False
                        st.session_state["dem_acquired"] = False  # Allow new DEM download
                        st.session_state["hillshade_qa_generated"] = False  # Regenerate hillshade for new resolution
                        st.session_state["dem_validated"] = False  # Re-validate new DEM
                        st.session_state["memory_checked"] = False  # Re-check memory for new DEM
                        st.rerun()

                    if col3.button(" Use Custom DEM Instead", use_container_width=True):
                        st.session_state["run_requested"] = False
                        st.session_state["prep_done"] = False
                        st.session_state["dem_acquired"] = False
                        st.session_state["hillshade_qa_generated"] = False
                        st.session_state["dem_validated"] = False
                        st.session_state["memory_checked"] = False
                        st.info("Switch to 'Custom DEM & River' mode above to upload your own files.")
                        st.rerun()

                    st.stop()

                dem_file = mosaic

                if CLIP_DEM:
                    clip_out = os.path.join(proj_dir, "mosaic_clipped.tif")
                    try:
                        with task_timer("Clipping DEM", status_box):
                            clipped_dem = clip_dem_to_aoi(dem_file, aoi_path, clip_out)
                            if clipped_dem and os.path.exists(clipped_dem):
                                dem_file = clipped_dem
                                status_box.success(f" DEM clipped to AOI")
                            else:
                                st.warning("DEM clipping failed - using full mosaic")
                    except Exception as e:
                        st.warning(f"DEM clipping failed ({e}) - using full mosaic")
                        print(f"Clipping error details: {e}")

                # Track DEM CRS (downloaded DEMs are typically already in UTM/projected CRS)
                with rasterio.open(dem_file) as src:
                    dem_crs = str(src.crs)
                    st.session_state["dem_original_crs"] = dem_crs
                    st.session_state["dem_output_crs"] = dem_crs  # No reprojection for downloaded DEMs

                # Mark DEM as acquired and save to session state
                st.session_state["dem_acquired"] = True
                st.session_state["dem_file"] = dem_file
                status_box.success(" DEM acquisition complete")
            else:
                # DEM already acquired, reuse it
                dem_file = st.session_state.get("dem_file")
                if dem_file and os.path.exists(dem_file):
                    status_box.info("ℹ️ Using previously downloaded DEM (skipping re-download)")
                else:
                    st.error(" Previously downloaded DEM not found. Please restart the workflow.")
                    st.session_state["dem_acquired"] = False
                    st.session_state["prep_done"] = False
                    st.stop()
            
            with task_timer("Fetching River", status_box):
                river_path = choose_and_save_nhd_river([aoi_path], proj_dir, 1, selected_river_name)

            if not river_path:
                st.error(" No river found in the selected area.")
                st.warning("This may be due to: (1) No rivers in NHD database for this AOI, (2) Network timeout, or (3) API service unavailable.")

                col1, col2 = st.columns(2)
                if col1.button(" Retry River Scan", use_container_width=True):
                    # Only reset river scan, NOT DEM acquisition or hillshade
                    # This prevents re-downloading DEMs and re-generating hillshades when retrying river scan
                    st.session_state["prep_done"] = False
                    st.session_state["scanned_river_list"] = []
                    # dem_acquired stays True - DEM is reused!
                    # hillshade_qa_generated stays True - hillshade is reused!
                    st.rerun()

                if col2.button(" Upload River File", use_container_width=True):
                    st.session_state["run_requested"] = False
                    st.session_state["prep_done"] = False
                    st.session_state["dem_acquired"] = False  # Reset for fresh start
                    st.session_state["hillshade_qa_generated"] = False  # Reset hillshade
                    st.session_state["dem_validated"] = False  # Reset validation
                    st.session_state["memory_checked"] = False  # Reset memory check
                    st.info("Switch to 'Custom DEM & River' mode above to upload your own river shapefile.")
                    st.rerun()

                st.stop()
            
            dem_folder_for_run = proj_dir

        # Only run validation if DEM was just acquired
        # Validation results don't change if DEM file is the same
        if not st.session_state.get("dem_validated"):
            stats = inspect_dem_metadata(dem_file)
            if stats['status'] == 'OK':
                st.success(f"DEM Verified: {stats['res'][0]:.2f}m resolution | Elev: {stats['min']:.1f}m - {stats['max']:.1f}m")
            else:
                st.warning(f"WARNING: DEM Issue Detected (Proceeding anyway): {stats.get('msg')}")
            st.session_state["dem_validated"] = True
            st.session_state["dem_stats"] = stats
        else:
            # Reuse cached validation results
            stats = st.session_state.get("dem_stats", {})
            if stats.get('status') == 'OK':
                st.info(f"ℹ️ DEM Previously Validated: {stats['res'][0]:.2f}m resolution")

        # Only run memory check if DEM was just acquired
        # Memory requirements don't change if DEM file is the same
        if not st.session_state.get("memory_checked"):
            memory_safe = utils.check_memory_safety(dem_file, operation_name="DEM Processing", raise_on_fail=False)
            st.session_state["memory_checked"] = True
            st.session_state["memory_safe"] = memory_safe
        else:
            # Reuse cached memory check result
            memory_safe = st.session_state.get("memory_safe", True)
            if memory_safe:
                st.info("ℹ️ Memory safety previously verified")
        if not memory_safe:
            st.error(" MEMORY WARNING: This DEM is too large for your system's available RAM")
            st.warning("Processing this file may cause your system to crash or freeze. Consider using a smaller area or lower resolution.")

            col1, col2 = st.columns(2)
            if col1.button(" Proceed Anyway (Risky)", use_container_width=True):
                st.session_state["memory_override"] = True
                st.rerun()

            if col2.button(" Go Back and Adjust", use_container_width=True):
                st.session_state["run_requested"] = False
                st.session_state["prep_done"] = False
                st.session_state["dem_acquired"] = False  # Reset for fresh start
                st.session_state["hillshade_qa_generated"] = False  # Reset hillshade
                st.session_state["dem_validated"] = False  # Reset validation
                st.session_state["memory_checked"] = False  # Reset memory check
                st.rerun()

            # Block unless user explicitly overrides
            if not st.session_state.get("memory_override"):
                st.stop()

        # Check if user clicked Stop before hillshade generation
        if not st.session_state.get("run_requested"):
            st.warning(" **Process stopped by user before hillshade generation**")
            st.stop()

        # Hillshade
        # Only generate hillshade if not already created
        # Hillshade doesn't change when retrying river scan
        hs_path = os.path.join(proj_dir, "hillshade.tif")
        if not st.session_state["hillshade_qa_generated"] or not os.path.exists(hs_path):
            # Check for stop before hillshade generation
            if check_stop_flag():
                status_box.error("Hillshade generation stopped by user!")
                st.stop()

            with task_timer("Computing Hillshade", status_box):
                # Use balanced downsample for both QA and visualization
                hillshade_module.create_hillshade_fast_qa(
                    dem_file,
                    hs_path,
                    downsample_factor=4,
                    z_factor=5.5
                )
            st.session_state["hillshade_qa_generated"] = True
        else:
            status_box.info("ℹ️ Using previously generated hillshade (skipping regeneration)")
            
        # Track original river CRS before reprojection
        river_gdf = gpd.read_file(river_path)
        river_original_crs = str(river_gdf.crs)
        st.session_state["river_original_crs"] = river_original_crs

        # Reproject River
        riv_rep_path = os.path.join(proj_dir, "river_reprojected.gpkg")
        reproject_vector_to_match_dem(river_path, dem_file, riv_rep_path)

        # Track output river CRS after reprojection (matches DEM)
        river_output_crs = st.session_state.get("dem_output_crs", "N/A")
        st.session_state["river_output_crs"] = river_output_crs

        # QA Generation
        qa_png_path = os.path.join(proj_dir, "hillshade_qa_qc.png")
        with task_timer("Generating QA Preview", status_box):
            generate_qa_png(hs_path, riv_rep_path, qa_png_path, aoi_geojson_path)

        qc_box.image(qa_png_path, caption="QA/QC Preview", use_container_width=True)

        # Store state (hillshade_output set to None initially - will generate full-res if needed)
        st.session_state.update({
            "dem_file": dem_file, "river_vector_path": river_path,
            "hillshade_output": None, "aerial_output": None, "reprojected_river_path": riv_rep_path,
            "rem_folder": proj_dir, "visuals_folder": proj_dir,
            "dem_folder_for_run": dem_folder_for_run,
            "qa_png": qa_png_path,
            "aoi_geojson_path": aoi_geojson_path,
            "prep_done": True
        })
        st.rerun()

    # QA gate
    if st.session_state["prep_done"]:
        # ALWAYS SHOW QA IMAGE IF IT EXISTS
        qa_img = os.path.join(st.session_state["project_folder_expanded"], "hillshade_qa_qc.png")
        if os.path.exists(qa_img):
            qc_box.image(qa_img, caption="QA/QC Preview", use_container_width=True)

    if st.session_state["prep_done"] and not st.session_state["qa_approved"]:
        with col_right:
            st.info("Is the river aligned correctly?")
            c1, c2 = st.columns(2)
            if c1.button("Yes, Continue"):
                # DO NOT DELETE THE IMAGE HERE 
                st.session_state["qa_approved"] = True
                st.rerun()
            if c2.button("No, Stop"):
                # Stop process but preserve files and cached work
                st.session_state["run_requested"] = False
                st.session_state["prep_done"] = False
                st.session_state["qa_approved"] = False
                # Note: Preserve dem_acquired, hillshade_qa_generated, dem_validated, memory_checked
                # so user can retry without re-downloading/re-validating
                st.warning(" **Process Stopped** - River alignment not approved. You can modify settings and run again. Previously downloaded/generated files are preserved.")
                st.rerun()

    # REM
    if st.session_state["qa_approved"] and not st.session_state["rem_done"]:

        # Check if user clicked Stop before starting REM calculation
        if not st.session_state.get("run_requested"):
            # User clicked Stop button - halt processing
            st.warning(" **Process stopped by user before REM calculation**")
            st.stop()

        dem_file = st.session_state["dem_file"]
        hillshade_output = st.session_state["hillshade_output"]
        reprojected_river_path = st.session_state["reprojected_river_path"]
        rem_folder = st.session_state["rem_folder"]
        visuals_folder = st.session_state["visuals_folder"]

        max_m_comp = float(comp_max_m) if comp_max_m.strip() else None

        viz_m_val = float(viz_max_m) if viz_max_m.strip() else None
        if max_m_comp is not None:
            comp_m_val = float(comp_max_m)
            if viz_m_val is None:
                viz_m_val = comp_m_val
                st.info(f"Visualization Max defaulted to {comp_m_val} m")
            elif viz_m_val > comp_m_val:
                st.warning(f"Visual Max clamped to {comp_m_val} m.")
                viz_m_val = comp_m_val

        max_m_vis = viz_m_val if viz_m_val is not None else None
        
        # Force Projection Mode
        base_mode_auto = "projection"
        k_auto = int(k_neighbors)

        # Detect data source for adaptive cross-section width
        data_source = "nhd" if run_mode == "AOI + NHD (3DEP download)" else "user_upload"

        rem_out = os.path.join(st.session_state["rem_folder"], "REM.tif")

        # Updated Block
        rem_args = dict(
            dem_folder=st.session_state["dem_folder_for_run"],
            river_shp=reprojected_river_path,
            output_rem_path=rem_out,
            spacing=int(spacing),
            tile_size=1024,
            k_neighbors=int(k_auto),
            max_value=max_m_comp,
            threads=threads,
            idw_power=None,
            base_mode=base_mode_auto,
            engine="scipy",
            data_source=data_source
        )

        # Check for stop before starting REM calculation
        if check_stop_flag():
            status_box.error(" REM calculation stopped by user!")
            st.stop()

        with task_timer("REM Calculation", status_box):
            with limit_threadpools(threads):
                _call_rem_main_with_filtered_kwargs(REM_Calcs, **rem_args)

        # Check for stop after REM calculation
        if check_stop_flag():
            status_box.error(" Processing stopped by user!")
            st.stop()
                
        rem_tif = rem_out
        aoi_path = st.session_state.get("aoi_geojson_path")
        if CLIP_DEM and aoi_path and os.path.exists(aoi_path):
            rem_clipped = os.path.join(rem_folder, "REM_clipped.tif")
            try:
                with task_timer("Clipping REM", status_box):
                    rem_tif = clip_dem_to_aoi(rem_out, aoi_path, rem_clipped)
            except Exception: pass
        
        # Check again if user clicked Stop after REM calculation
        if not st.session_state.get("run_requested"):
            st.warning(" **Process stopped by user after REM calculation**")
            st.stop()

        # Check for stop before PNG styling
        if check_stop_flag():
            status_box.error("PNG styling stopped by user!")
            st.stop()

        pngs = []
        with task_timer("Styling PNGs", status_box):
            # Reuse the single hillshade.tif for visualization
            if bg_type == "hillshade":
                hillshade_output = os.path.join(st.session_state["rem_folder"], "hillshade.tif")
                st.session_state["hillshade_output"] = hillshade_output

            # Generate aerial imagery if selected
            aerial_output = None
            if bg_type == "aerial":
                aerial_output = os.path.join(st.session_state["rem_folder"], "aerial.tif")
                if not os.path.exists(aerial_output):
                    status_box.info("Downloading aerial imagery tiles (this happens once, then cached)...")
                    try:
                        result = aerial_tiles.build_aerial_geotiff_like(
                            dem_file,  # Use the DEM as template
                            aerial_output
                        )
                        if result:
                            status_box.success(" Aerial imagery downloaded and cached")
                        else:
                            status_box.warning(" Aerial download failed. Falling back to hillshade.")
                            bg_type = "hillshade"
                    except Exception as e:
                        status_box.warning(f" Aerial download error: {e}. Falling back to hillshade.")
                        bg_type = "hillshade"
                else:
                    status_box.info(" Using cached aerial imagery")
                st.session_state["aerial_output"] = aerial_output

            # Check dataset size FIRST to determine memory strategy
            with rasterio.open(rem_tif) as src:
                total_pixels = src.width * src.height

            # Calculate safe number of parallel workers based on dataset size
            cpu_count = multiprocessing.cpu_count()

            if total_pixels > 50_000_000:
                # Large dataset (>50M pixels): serialize for memory safety and progress visibility
                optimal_workers = 2
                is_large_dataset = True
                print(f"INFO: Large dataset ({total_pixels:,} pixels)")
                print(f"      Processing {len(selected_ramps)} color ramps sequentially to prevent crashes")
                print(f"      Each PNG will show progress as it renders...")
            else:
                # Normal dataset: use 75% of cores
                optimal_workers = min(len(selected_ramps), max(1, int(cpu_count * 0.75)))
                is_large_dataset = False

            # CRITICAL FIX: Calculate global normalization ONCE before PNG loop
            # This prevents re-scanning the entire REM+hillshade for every PNG (was causing 12GB+ of redundant I/O)
            global_norm_values = None
            if is_large_dataset:
                print(f"      Pre-calculating global normalization to avoid redundant scans...")
                global_vmin, global_vmax, global_hs_min, global_hs_max = style_rem_module._calculate_global_normalization(
                    rem_path=rem_tif,
                    hillshade_path=hillshade_output if bg_type == "hillshade" else None,
                    max_value=max_m_vis,
                    min_value=None,
                    hard_cap=True,
                    viz_max_feet=None,
                    background=bg_type
                )
                global_norm_values = {
                    'global_vmin': global_vmin,
                    'global_vmax': global_vmax,
                    'global_hs_min': global_hs_min,
                    'global_hs_max': global_hs_max
                }
                print(f"      Global normalization complete (will be reused for all {len(selected_ramps)} PNGs)")

            # Pre-load hillshade ONLY for small datasets (memory efficient for large datasets to read from disk)
            hillshade_array = None
            if not is_large_dataset and bg_type == "hillshade" and hillshade_output and os.path.exists(hillshade_output):
                try:
                    with rasterio.open(rem_tif) as rem_src:
                        rem_transform = rem_src.transform
                        rem_width = rem_src.width
                        rem_height = rem_src.height
                        rem_crs = rem_src.crs
                    # Load hillshade once for small datasets only
                    hillshade_array = style_rem_module._read_aligned_hillshade(
                        hillshade_output, rem_transform, rem_width, rem_height, rem_crs
                    )
                except Exception as e:
                    print(f"WARNING: Could not pre-load hillshade: {e}")
                    hillshade_array = None

            # Helper function to generate a single color ramp PNG with better memory management
            def generate_single_ramp(cmap):
                style_kwargs = dict(
                    rem_path=rem_tif,
                    hillshade_path=hillshade_output if bg_type == "hillshade" else None,
                    aerial_path=aerial_output if bg_type == "aerial" else None,
                    out_folder=visuals_folder,
                    color_maps=[cmap],
                    rem_alpha=float(rem_alpha),
                    dpi=int(EXPORT_DPI),
                    max_value=max_m_vis,
                    background=bg_type,
                    scale=int(EXPORT_SCALE),
                    n_classes=True if use_discrete_colors else None,
                    legend_unit="meters",
                    hard_cap=True,
                    bg_alpha=float(bg_alpha),
                    hillshade_array=hillshade_array,  # Pass pre-loaded hillshade (None for large datasets)
                )
                # Add global normalization values if pre-calculated
                if global_norm_values:
                    style_kwargs.update(global_norm_values)

                with limit_threadpools(threads):
                    result = _call_style_rem_with_filtered_kwargs(style_rem, **style_kwargs)
                    # Force immediate garbage collection after each PNG
                    gc.collect()
                    return result

            if len(selected_ramps) > 1 and optimal_workers > 1:
                # Parallel with verbose progress
                print(f"      Processing {len(selected_ramps)} PNGs with {optimal_workers} workers...")
                pngs = Parallel(n_jobs=optimal_workers, backend="threading", verbose=10)(
                    delayed(generate_single_ramp)(cmap) for cmap in selected_ramps
                )
                print(f"       All {len(selected_ramps)} PNGs complete!")
            else:
                # Sequential processing for safety with detailed progress
                pngs = []
                for i, cmap in enumerate(selected_ramps, 1):
                    # Check for stop signal before each PNG
                    if check_stop_flag():
                        status_box.error(" PNG generation stopped by user!")
                        break

                    print(f"      Generating PNG {i}/{len(selected_ramps)}: {cmap}...")
                    png = generate_single_ramp(cmap)
                    pngs.append(png)
                    print(f"       Complete: {os.path.basename(png)}")

                    # Aggressive memory cleanup between PNGs (critical for large datasets)
                    if is_large_dataset and i < len(selected_ramps):
                        import time
                        gc.collect()  # Force garbage collection
                        time.sleep(0.5)  # Brief pause to allow OS to reclaim memory
                        print(f"       Memory released for next PNG...")
        
        st.session_state["rem_done"] = True
        status_box.success("Done!")
        
        # Define the path for your new stats file
        stats_txt_path = os.path.join(visuals_folder, "REM_Project_Stats.txt")

        # Package the settings into a dictionary for the report
        report_config = {
            "engine": "SciPy Optimized",
            "spacing": spacing,
            "k_neighbors": k_neighbors,
            "max_rem_comp": comp_max_m,
            "max_rem_vis": viz_max_m,
            "png_list": [os.path.basename(p) for p in pngs],
            "dem_original_crs": st.session_state.get("dem_original_crs", "N/A"),
            "dem_output_crs": st.session_state.get("dem_output_crs", "N/A"),
            "river_original_crs": st.session_state.get("river_original_crs", "N/A"),
            "river_output_crs": st.session_state.get("river_output_crs", "N/A"),
            "data_source": data_source
        }

        # CALL THE FUNCTION
        utils.generate_full_stats_report(rem_tif, dem_file, report_config, stats_txt_path)
                
        # Add to the UI display
        st.success(f" Statistics Report generated: {os.path.basename(stats_txt_path)}")

        # Get project directory before using it
        proj_dir = st.session_state.get('project_folder_expanded', project_folder)

        # Cleanup temporary files
        with task_timer("Cleaning up temporary files", status_box):
            cleanup_project_outputs(proj_dir)

        if "pipeline_start_time" in st.session_state and st.session_state["pipeline_start_time"]:
            total_time = time.time() - st.session_state["pipeline_start_time"]
            mins, secs = divmod(total_time, 60)
            st.success(f"Total Pipeline Time: {int(mins)}m {int(secs)}s")

        rem_img_box.image(pngs, caption=[os.path.basename(p) for p in pngs], use_container_width=True)

        png_list_md = "\n".join([f"- `{p}`" for p in pngs])

        path_box.markdown(f"""
        ### Output Files
        **Project:** `{proj_dir}`
        - **REM:** `{rem_tif}`
        - **DEM:** `{dem_file}`

        **Images:**
        {png_list_md}
        """)

        # Auto-unlock sidebar when finished
        st.session_state["run_requested"] = False
        st.session_state["prep_done"] = True
        st.session_state["qa_approved"] = True


# Run another REM button

if st.session_state.get("rem_done") and not st.session_state.get("run_requested"):
    with col_left:
        st.markdown("---")
        if st.button(" Run Another REM", use_container_width=True, type="primary"):
            # Reset workflow state but PRESERVE uploaded files and their validation
            # This allows users to rerun with same files without re-uploading/re-validating
            for key in ["run_requested", "prep_done", "dem_acquired", "hillshade_qa_generated",
                        "dem_validated", "memory_checked", "qa_approved", "rem_done",
                        "dem_file", "dem_stats", "memory_safe", "dem_folder_for_run", "river_vector_path",
                        "aoi_geojson_path", "hillshade_output", "aerial_output", "reprojected_river_path",
                        "dem_folder", "rem_folder", "visuals_folder", "extra_data_folder",
                        "scanned_river_list", "pipeline_start_time", "qa_png", "memory_override"]:
                if key in st.session_state:
                    st.session_state[key] = [] if key == "scanned_river_list" else (0.0 if key == "pipeline_start_time" else None)
            st.session_state["run_requested"] = False
            st.session_state["prep_done"] = False
            st.session_state["dem_acquired"] = False  # Reset DEM for new run
            st.session_state["hillshade_qa_generated"] = False  # Reset hillshade for new run
            st.session_state["dem_validated"] = False  # Reset validation for new run
            st.session_state["memory_checked"] = False  # Reset memory check for new run
            st.session_state["qa_approved"] = False
            st.session_state["rem_done"] = False
            st.rerun()