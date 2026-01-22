#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#run_rem_guided.py

# Fix PROJ unicode errors with Python 3.13 (must be before other imports)
import os
os.environ['PROJ_SKIP_READ_USER_WRITABLE_DIRECTORY'] = 'YES'
os.environ['PROJ_DEBUG'] = '0'
os.environ['PROJ_NETWORK'] = 'OFF'
import warnings
warnings.filterwarnings('ignore', category=UnicodeWarning)
warnings.filterwarnings('ignore', message='.*utf-8.*')
warnings.filterwarnings('ignore', module='pyproj')

import time
import logging
import subprocess
import webbrowser
import tempfile
import shutil
import glob as _glob
from pathlib import Path
import sys

# Ensure import of local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import project modules
import data_collections as dc
import rem_utils as utils
import rem_config as cfg
import REM_Calcs
import rem_hillshade_colorramp as style_rem
import hillshade
import folium
from folium.plugins import Draw, Fullscreen, Geocoder
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from PIL import Image

# --- CRITICAL FIX: Allow massive images (1m DEMs) without crashing ---
Image.MAX_IMAGE_PIXELS = None

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Cleanup Temporary Files

# Remove temporary cache files after REM processing is complete
# All other files remain in project folder
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

                    # SAFETY CHECK: Do not delete the clipped mosaic
                    if "mosaic_clipped.tif" in fname:
                        continue

                    # CROSS-PLATFORM FIX: Use safe_remove for Windows compatibility
                    if utils.safe_remove(fpath, verbose=False):
                        logger.info(f"Cleaned up: {fname}")
                    else:
                        logger.warning(f"Could not remove {fname} (file may be locked)")
                except Exception as e:
                    logger.warning(f"Could not remove {fname}: {e}")

        logger.info("Cleanup complete!")

    except Exception as e:
        logger.warning(f"WARNING: Cleanup encountered an error: {e}")


# Helper: DEM Health Checker (Advisory Only)

# Open DEM and return critical stats
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


# Validate DEM file and return detailed information or errors
def validate_dem_file(dem_path: str) -> dict:
    result = {'valid': False, 'error': None, 'info': {}}

    try:
        # Expand path
        dem_path = os.path.expanduser(dem_path)

        # Check exists
        if not os.path.exists(dem_path):
            result['error'] = f"DEM file not found: {dem_path}"
            return result

        # Check readable
        if not os.access(dem_path, os.R_OK):
            result['error'] = f"DEM file not readable (permission denied): {dem_path}"
            return result

        # Try to open with rasterio
        try:
            with rasterio.open(dem_path) as src:
                # Check has valid CRS
                if src.crs is None:
                    result['error'] = "DEM has no coordinate reference system (CRS). Please ensure the file has valid projection information."
                    return result

                # Check dimensions
                if src.width <= 0 or src.height <= 0:
                    result['error'] = f"DEM has invalid dimensions: {src.width} x {src.height}"
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
                        result['error'] = "DEM appears to contain only NoData values in sampled area. File may be empty or corrupted."
                        return result
                except Exception as e:
                    result['error'] = f"Could not read DEM data: {str(e)}"
                    return result

                # Success - extract info
                bounds = src.bounds
                res_x, res_y = src.res

                result['valid'] = True
                result['info'] = {
                    'width': src.width,
                    'height': src.height,
                    'resolution': (abs(res_x), abs(res_y)),
                    'crs': str(src.crs),
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
            result['error'] = f"Not a valid raster file or unsupported format: {str(e)}"
            return result

    except Exception as e:
        result['error'] = f"Unexpected error validating DEM: {str(e)}"
        return result

    return result


# Validate river vector file and return detailed information or errors
def validate_river_file(river_path: str) -> dict:
    result = {'valid': False, 'error': None, 'info': {}}

    try:
        # Expand path
        river_path = os.path.expanduser(river_path)

        # Check exists
        if not os.path.exists(river_path):
            result['error'] = f"River file not found: {river_path}"
            return result

        # Check readable
        if not os.access(river_path, os.R_OK):
            result['error'] = f"River file not readable (permission denied): {river_path}"
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
                result['error'] = f"Shapefile missing required companion files: {', '.join(missing_files)}"
                return result

        # Try to open with geopandas
        try:
            gdf = gpd.read_file(river_path)

            # Check not empty
            if len(gdf) == 0:
                result['error'] = "River file contains no features (empty dataset)"
                return result

            # Check has valid CRS
            if gdf.crs is None:
                result['error'] = "River file has no coordinate reference system (CRS). Please ensure the file has valid projection information."
                return result

            # Check geometry types
            geom_types = gdf.geometry.geom_type.unique()
            if 'LineString' not in geom_types and 'MultiLineString' not in geom_types:
                result['error'] = f"River file must contain LineString or MultiLineString geometries. Found: {', '.join(geom_types)}"
                return result

            # Check for null geometries
            null_count = gdf.geometry.isna().sum()
            if null_count == len(gdf):
                result['error'] = "All features have null/invalid geometries"
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
            result['error'] = f"Not a valid vector file or unsupported format: {str(e)}"
            return result

    except Exception as e:
        result['error'] = f"Unexpected error validating river file: {str(e)}"
        return result

    return result


# Generate QA PNG

# Generate high-quality PNG for QA/QC display with hillshade background and river overlay
def generate_qa_png(hillshade_path: str, river_path: str, output_png: str, aoi_path: str = None):
    # Read Hillshade (Higher quality - up to 4000px)
    with rasterio.open(hillshade_path) as src:
        df = max(1, int(max(src.width, src.height) / 4000))  # Increased from 1000 to 4000
        out_shape = (1, int(src.height / df), int(src.width / df))
        hs = src.read(1, out_shape=out_shape, resampling=Resampling.lanczos)  # Better quality resampling

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

    # 2. Read River
    gdf = gpd.read_file(river_path)
    if not gdf.empty:
        gdf = gdf.to_crs(hs_crs)

    # 3. Plot to file (higher quality)
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.imshow(hs, extent=(xmin, xmax, ymin, ymax), cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation='bilinear')

    # Plot river (red line)
    if not gdf.empty:
        gdf.plot(ax=ax, color="red", linewidth=1.5, alpha=0.9)

    ax.set_title("QA/QC: Hillshade + River", fontsize=14, fontweight='bold')
    ax.set_axis_off()

    plt.savefig(output_png, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)
    return output_png


# Print a nice header
def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


# Print step indicator
def print_step(step_num, total, text):
    print(f"\n[Step {step_num}/{total}] {text}")
    print("-" * 70)


# Wait for user to press Enter
def wait_for_enter(prompt="Press Enter to continue..."):
    input(f"\n{prompt}")

# Get user choice from list
def get_choice(prompt, options, default=0):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if i == default + 1 else ""
        print(f"  {i}. {opt}{marker}")

    while True:
        choice = input(f"\nChoice [1-{len(options)}]: ").strip()
        if not choice and default is not None:
            return default
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        except:
            pass
        print(f"Please enter 1-{len(options)}")


# Get text input
def get_input(prompt, default=None):
    default_text = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{default_text}: ").strip()
        if not value and default:
            return default
        if value:
            return value
        print("This field is required.")


# Create interactive map for drawing AOI. User draws on map, exports GeoJSON
def create_aoi_map_interactive(output_html, output_geojson, center=[39.8283, -98.5795], zoom=5):
    # Create map
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)

    # Add basemaps
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Aerial",
        control=True,
        show=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Topo",
        control=True,
        show=False
    ).add_to(m)

    # Add drawing tools configured to export to our specific filename
    Draw(
        export=True,
        filename=output_geojson,
        position="topleft",
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": {"shapeOptions": {"color": "#3388ff"}, "showArea": True},
            "polygon": {"shapeOptions": {"color": "#ff7800"}, "showArea": True}
        }
    ).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    Geocoder(position="topleft").add_to(m)
    folium.LayerControl().add_to(m)

    # Save map
    m.save(output_html)
    return output_html


#     Wait for user to create AOI file by exporting from map. Polls for file existence.
def wait_for_aoi_file(geojson_path, timeout=600):
    print("\nWaiting for AOI to be drawn and exported...")
    print(f"   Looking for: {geojson_path}")

    start = time.time()
    last_msg = 0

    while True:
        if os.path.exists(geojson_path):
            # Give it a moment to finish writing
            time.sleep(1)
            # Verify it's valid
            try:
                import geopandas as gpd
                gdf = gpd.read_file(geojson_path)
                if not gdf.empty:
                    print(f"   AOI received! ({len(gdf)} feature(s))")
                    return geojson_path
            except:
                pass

        elapsed = time.time() - start

        # Print reminder every 10 seconds
        if elapsed - last_msg > 10:
            remaining = int(timeout - elapsed)
            print(f"   ... still waiting ({remaining}s remaining)")
            last_msg = elapsed

        if elapsed > timeout:
            raise TimeoutError(f"Timed out waiting for AOI after {timeout}s")

        time.sleep(2)


# Main guided workflow
def main():
    print_header("AUTOMATED REM GENERATOR - Guided Workflow")
    print("\nThis script will guide you through creating a Relative Elevation Model (REM)")
    print("step-by-step with visual tools.\n")

    # Install shutdown handler
    utils.install_shutdown_handler()

    # STEP 1: Mode Selection
    print_step(1, 7, "Select Data Source")

    mode_idx = get_choice(
        "How will you provide data?",
        [
            "Draw AOI → Auto-download DEM and river (recommended for USA)",
            "I have my own DEM and river files"
        ],
        default=0
    )

    mode = "aoi_nhd" if mode_idx == 0 else "custom"


    # STEP 2: Project Setup
    print_step(2, 7, "Project Setup")

    project_name = get_input("Project name (creates ./Project/{name})", default="REM_Project")
    base_dir = os.path.join("./Project", project_name)
    os.makedirs(base_dir, exist_ok=True)
    print(f"   Project folder: {base_dir}")

    # STEP 3A: AOI Drawing (if AOI mode) or 3B: File Selection (if custom)

    if mode == "aoi_nhd":
        print_step(3, 7, "Draw Area of Interest")

        print("\nOpening interactive map in your web browser...")
        print("\nInstructions:")
        print("  1. A map will open in your browser")
        print("  2. Use the drawing tools on the left to draw a rectangle or polygon")
        print("  3. Click the 'Export' button when done")
        print("  4. Save the file as 'aoi.geojson' in your Downloads folder")
        print("  5. Come back here - the script will detect it automatically!")

        wait_for_enter("\nReady? Press Enter to open map...")

        # Create temporary map
        temp_html = os.path.join(tempfile.gettempdir(), "rem_aoi_map.html")
        downloads_folder = os.path.expanduser("~/Downloads")
        aoi_geojson = os.path.join(downloads_folder, "aoi.geojson")

        # Remove old AOI if exists
        if os.path.exists(aoi_geojson):
            print(f"\nWARNING: Found existing AOI file: {aoi_geojson}")
            if get_choice("Use this existing AOI?", ["Yes", "No, draw new one"], default=0) == 0:
                print("   Using existing AOI")
            else:
                # CROSS-PLATFORM FIX: Use safe_remove for Windows compatibility
                if utils.safe_remove(aoi_geojson, verbose=True):
                    print("   Deleted old AOI, you can draw a new one")
                else:
                    print("   Warning: Could not delete old AOI file")

        if not os.path.exists(aoi_geojson):
            # Create and open map
            create_aoi_map_interactive(temp_html, "aoi.geojson")
            # CROSS-PLATFORM FIX: Use Path.as_uri() for correct file URLs on Windows/macOS/Linux
            webbrowser.open(Path(temp_html).as_uri())

            # Wait for AOI
            try:
                wait_for_aoi_file(aoi_geojson, timeout=600)
            except TimeoutError:
                print("\nERROR: Timeout waiting for AOI. Please try again.")
                return

        # Copy to project folder
        final_aoi = os.path.join(base_dir, "AOI.geojson")
        import shutil
        shutil.copy(aoi_geojson, final_aoi)
        print(f"   AOI saved to: {final_aoi}")

        # ========================================================================
        # STEP 4: DEM Resolution
        # ========================================================================
        print_step(4, 7, "DEM Resolution Selection")

        print("\nChecking what DEM resolutions are available for your area...")
        available = dc.get_available_project_resolutions(final_aoi)

        if available:
            print(f"   Available (Project tiles - high quality): {available} meters")
            default_res = available[0]
        else:
            print("   No Project tiles found. WCS tiles (lower quality) available as fallback.")
            print("   Common WCS options: 10, 30 meters")
            default_res = 10

        res_input = get_input(f"DEM resolution (meters)", default=str(default_res))
        resolution = int(res_input)
        print(f"   Using {resolution}m resolution")

        # ========================================================================
        # STEP 5: River Selection
        # ========================================================================
        print_step(5, 7, "River Selection")

        river_choice = get_choice(
            "How do you want to select the river?",
            [
                "Scan AOI and let me choose",
                "Auto-select largest river",
                "Enter river name manually"
            ],
            default=1
        )

        river_name = None
        if river_choice == 0:  # Scan
            print("\nScanning for rivers in your AOI...")
            rivers = dc.scan_nhd_rivers(final_aoi)
            if rivers:
                print(f"   Found {len(rivers)} rivers")
                if len(rivers) > 1:
                    print("\nTop rivers by length:")
                    display_rivers = rivers[:min(15, len(rivers))]
                    for i, r in enumerate(display_rivers, 1):
                        print(f"   {i}. {r}")

                    river_idx = get_choice("Select river:", display_rivers, default=0)
                    river_name = display_rivers[river_idx]
                else:
                    river_name = rivers[0]
                print(f"   Selected: {river_name}")
            else:
                print("   WARNING: No rivers found in this area.")
                print("   This may be due to: (1) No NHD data, (2) Network timeout, or (3) API unavailable")
                retry_choice = get_choice("What would you like to do?",
                                         ["Retry river scan", "Enter river name manually", "Continue with auto-select"],
                                         default=0)
                if retry_choice == 0:  # Retry
                    print("\n   Retrying river scan...")
                    rivers = dc.scan_nhd_rivers(final_aoi)
                    if rivers:
                        print(f"   Found {len(rivers)} rivers on retry!")
                        display_rivers = rivers[:min(15, len(rivers))]
                        for i, r in enumerate(display_rivers, 1):
                            print(f"   {i}. {r}")
                        river_idx = get_choice("Select river:", display_rivers, default=0)
                        river_name = display_rivers[river_idx]
                    else:
                        print("   Still no rivers found. Will auto-select largest.")
                elif retry_choice == 1:  # Manual entry
                    river_name = get_input("River name (e.g., 'Snake River')")
                else:  # Continue with auto-select
                    print("   Will auto-select largest river in AOI")

        elif river_choice == 2:  # Manual
            river_name = get_input("River name (e.g., 'Snake River')")

        if not river_name:
            print("   Will auto-select largest river in AOI")

        dem_path = None
        river_path = None

    else:  # custom mode
        print_step(3, 7, "Provide Your Data Files")

        print("\nYou'll need:")
        print("  • A DEM (Digital Elevation Model) GeoTIFF file (.tif)")
        print("  • A river centerline vector file (.shp or .gpkg)")

        # DEM file validation loop
        while True:
            dem_input = get_input("\nPath to DEM file (.tif)")
            dem_path = os.path.abspath(os.path.expanduser(dem_input))

            print("\n" + "="*70)
            print("   VALIDATING DEM FILE...")
            print("="*70)
            dem_result = validate_dem_file(dem_path)

            if dem_result['valid']:
                info = dem_result['info']
                print("\n   ✅ ✅ ✅  DEM FILE IS VALID  ✅ ✅ ✅\n")
                print(f"      File: {os.path.basename(dem_path)}")
                print(f"      Resolution: {info['resolution'][0]:.2f}m × {info['resolution'][1]:.2f}m")
                print(f"      Dimensions: {info['width']:,} × {info['height']:,} pixels")
                print(f"      CRS: {info['crs']}")
                print(f"      File size: {info['file_size_mb']:.1f} MB")
                print("\n" + "="*70)
                break
            else:
                print("\n   DEM FILE VALIDATION FAILED\n")
                print(f"   ERROR: {dem_result['error']}")
                print("\n   NEXT STEPS:")
                print("   1. Make sure the DEM is a valid GeoTIFF (.tif) file")
                print("   2. Ensure the file has a valid coordinate reference system (CRS)")
                print("   3. Verify the file contains elevation data (not empty/corrupted)")
                print("   4. Try opening the file in QGIS or another GIS tool to verify it works")
                print("="*70 + "\n")
                retry = get_choice("Try again?", ["Yes, enter different path", "No, exit"], default=0)
                if retry != 0:
                    logger.info("User cancelled file selection.")
                    return

        # River file validation loop
        while True:
            river_input = get_input("\nPath to river file (.shp or .gpkg)")
            river_path = os.path.abspath(os.path.expanduser(river_input))

            print("\n" + "="*70)
            print("   VALIDATING RIVER FILE...")
            print("="*70)
            river_result = validate_river_file(river_path)

            if river_result['valid']:
                info = river_result['info']
                print("\n   ✅ ✅ ✅  RIVER FILE IS VALID  ✅ ✅ ✅\n")
                print(f"      File: {os.path.basename(river_path)}")
                print(f"      Features: {info['feature_count']:,}")
                print(f"      Geometry: {info['geometry_types']}")
                print(f"      CRS: {info['crs']}")
                print(f"      File size: {info['file_size_mb']:.1f} MB")
                print("\n" + "="*70)
                break
            else:
                print("\n   RIVER FILE VALIDATION FAILED\n")
                print(f"   ERROR: {river_result['error']}")
                print("\n   NEXT STEPS:")
                print("   1. Make sure the river file is a valid vector file (.shp, .gpkg, or .geojson)")
                print("   2. For shapefiles, ensure all companion files exist (.shx, .dbf, .prj)")
                print("   3. Verify the file contains LineString or MultiLineString geometries")
                print("   4. Ensure the file has a valid coordinate reference system (CRS)")
                print("   5. Try opening the file in QGIS or another GIS tool to verify it works")
                print("="*70 + "\n")
                retry = get_choice("Try again?", ["Yes, enter different path", "No, exit"], default=0)
                if retry != 0:
                    logger.info("User cancelled file selection.")
                    return

        # --- SMART CRS HANDLING ---
        # Check if DEM is in geographic CRS and reproject to UTM if needed
        print("\n   Checking DEM coordinate system...")
        import rasterio
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
            is_geographic = utils.is_geographic_crs(dem_crs)

            if is_geographic:
                print(f"   DEM is in geographic CRS ({dem_crs})")
                print(f"   Geographic coordinates (degrees) are not suitable for REM calculations.")
                print(f"   The DEM will be automatically reprojected to UTM (meters).")

                confirm = get_choice("Proceed with automatic reprojection?", ["Yes, reproject to UTM", "No, I'll provide a different DEM"], default=0)
                if confirm != 0:
                    logger.info("User declined automatic reprojection.")
                    return

                # Reproject DEM to UTM
                print("\n   Reprojecting DEM to UTM...")
                project_folder = os.path.dirname(dem_path)
                reprojected_dem_path = os.path.join(project_folder, "DEM_reprojected_UTM.tif")

                dem_path, target_epsg = utils.reproject_dem_to_utm(
                    dem_path,
                    reprojected_dem_path,
                    verbose=True
                )

                print(f"   DEM reprojected to {target_epsg}")
                print(f"   All outputs will be in {target_epsg} (meters)")
            else:
                print(f"   DEM is in projected CRS ({dem_crs}) - ready for processing")

        final_aoi = None
        resolution = None
        river_name = None

    # ============================================================================
    # STEP 6: Processing Settings
    # ============================================================================
    print_step(6, 7, "Processing Settings")

    use_defaults = get_choice(
        "Use default processing settings?",
        ["Yes (recommended for most users)", "No (customize)"],
        default=0
    )

    if use_defaults == 0:
        spacing = 20
        k_neighbors = 8
        cpu_usage = 0.75  # Updated from 0.50 to 0.75 for better performance
        max_comp = None
        max_vis = 15.0
        color_ramp = "MagmaRamp"
        background = "hillshade"
        n_classes = 0
        print("   Using defaults (75% CPU usage for good balance)")
    else:
        print("\nProcessing parameters:")
        print("\nParameter Explanations:")
        print("  • Spacing (m): Distance between river sampling points. Lower = more accurate but slower.")
        print("  • K Neighbors (IDW): Number of nearby points for interpolation. Range: 4-300.")
        print("      - Lower (4-8): Sharp detail, preserves terrain features")
        print("      - Higher (20-300): Very smooth, averages out noise")
        print("      - Default 8: Good balance for most cases")
        print("  • CPU Usage: Fraction of processor cores to use.")
        print("      - Low (0.25-0.50): Use while working. Keeps computer responsive.")
        print("      - Medium (0.60-0.75): Recommended. Good balance of speed and responsiveness.")
        print("      - High (0.80-1.00): Fastest processing. Best for dedicated/overnight runs.")
        print("    Tip: Start at 0.75. Only lower if computer feels sluggish.")
        print("  • Max REM (ft) - Comp: Maximum elevation to compute. Limits processing to floodplain.\n")

        spacing = int(get_input("  Spacing (meters)", default="20"))
        k_neighbors = int(get_input("  K Neighbors (4-300)", default="8"))

        # Validate k_neighbors range
        if k_neighbors < 4 or k_neighbors > 300:
            print(f"\n   WARNING: K Neighbors must be between 4 and 300. You entered: {k_neighbors}")
            k_neighbors = int(get_input("  K Neighbors (4-300)", default="8"))

        cpu_usage = float(get_input("  CPU usage (0.1-1.0)", default="0.75"))

        # Warning for very high CPU usage
        if cpu_usage >= 0.95:
            print("\n   WARNING: 100% CPU may make your system unresponsive during processing.")
            print("   Recommended: 0.75 for good balance of speed and responsiveness.")
            confirm = get_choice("   Continue with 100% CPU?", ["Yes, continue", "No, I'll lower it"], default=1)
            if confirm != 0:
                cpu_usage = float(get_input("  CPU usage (0.1-1.0)", default="0.75"))

        max_comp_input = get_input("  Max REM height - computation (feet, blank=none)", default="")
        max_comp = float(max_comp_input) if max_comp_input else None

        max_vis = float(get_input("  Max REM height - visualization (feet)", default="15"))

        print("\nVisualization:")
        print("\nℹ️  Visualization Parameter Explanations:")
        print("  • Color Ramps: Color schemes for your REM. Choose multiple for comparison.")
        print("  • Background: Layer behind REM (hillshade=terrain relief, aerial=satellite, white=clean).")
        print("  • Max REM (ft) - Vis: Maximum elevation displayed. Higher values capped at max color.")
        print("  • Discrete Classes: Number of color bands (0=smooth gradient). Creates clear zones.")
        print("")
        print("  Available color ramps:")
        for i, ramp in enumerate(cfg.COLOR_RAMP_OPTIONS, 1):
            print(f"    {i}. {ramp}")
        num_ramps = len(cfg.COLOR_RAMP_OPTIONS)

        ramp_input = get_input(
            f"  Select color ramp(s) [1-{num_ramps}] (comma-separated for multiple, e.g., '1,3,5')",
            default="1"
        ).strip()

        # Parse multiple selections
        selected_ramps = []
        try:
            indices = [int(x.strip()) - 1 for x in ramp_input.split(',')]
            for idx in indices:
                if 0 <= idx < num_ramps:
                    selected_ramps.append(cfg.COLOR_RAMP_OPTIONS[idx])
                else:
                    print(f"  Warning: Ignoring invalid selection {idx+1}")
        except ValueError:
            print(f"  Invalid input, using default (AquaCopper)")
            selected_ramps = [cfg.COLOR_RAMP_OPTIONS[0]]

        if not selected_ramps:
            selected_ramps = [cfg.COLOR_RAMP_OPTIONS[0]]

        print(f"  Selected: {', '.join(selected_ramps)}")

        bg_idx = get_choice("  Background type:", cfg.BACKGROUND_OPTIONS, default=0)
        background = cfg.BACKGROUND_OPTIONS[bg_idx]

        n_classes_input = get_input("  Discrete classes (0=continuous gradient)", default="0")
        n_classes = int(n_classes_input)

    # ============================================================================
    # STEP 7: Confirmation & Run
    # ============================================================================
    print_step(7, 7, "Ready to Process")

    print("\nSummary:")
    print(f"   Mode: {mode.upper()}")
    print(f"   Project: {project_name}")
    if mode == "aoi_nhd":
        print(f"   Resolution: {resolution}m")
        print(f"   River: {river_name or 'Auto-detect'}")
    else:
        print(f"   DEM: {os.path.basename(dem_path)}")
        print(f"   River: {os.path.basename(river_path)}")
    print(f"   Color ramps: {', '.join(selected_ramps)} ({len(selected_ramps)} visualization(s))")
    print(f"   Output: {base_dir}")

    proceed = get_choice(
        "\nProceed with REM generation?",
        ["Yes, start processing!", "No, cancel"],
        default=0
    )

    if proceed != 0:
        print("\nCancelled by user.")
        return

    # ============================================================================
    # PROCESSING
    # ============================================================================
    print_header("STARTING REM PIPELINE")

    start_time = time.time()

    # Setup threading
    total_cores = os.cpu_count() or 4
    compute_threads = max(1, min(total_cores, int(round(total_cores * cpu_usage))))
    download_threads = max(1, total_cores // 2)

    logger.info(f"Threads: {download_threads} (Download) / {compute_threads} (Compute)")

    # Define paths
    riv_rep_path = os.path.join(base_dir, "river_reprojected.gpkg")
    hs_path = os.path.join(base_dir, "hillshade.tif")
    rem_out = os.path.join(base_dir, "REM.tif")

    # Wrap in sleep prevention
    with utils.prevent_sleep("REM Pipeline"):

        # PHASE 1: Data Acquisition
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: DATA ACQUISITION")
        logger.info("="*70)

        if mode == "aoi_nhd":
            # Download DEM
            logger.info("Downloading DEM...")
            mosaic_path = dc.download_and_mosaic_dems(
                [final_aoi],
                base_dir,
                resolution,
                n_jobs_download=download_threads,
                n_jobs_mosaic=compute_threads
            )

            if not mosaic_path:
                logger.error("\n" + "="*70)
                logger.error("ERROR: DEM download failed after all retry attempts")
                logger.error("This may be due to: network issues, server downtime, or no data available")
                logger.error("="*70)

                retry_download = get_choice("\nWhat would you like to do?",
                                           ["Retry download", "Exit and try different settings"],
                                           default=1)
                if retry_download == 0:
                    logger.info("\nRetrying DEM download...")
                    mosaic_path = dc.download_and_mosaic_dems(
                        [final_aoi],
                        base_dir,
                        resolution,
                        n_jobs_download=download_threads,
                        n_jobs_mosaic=compute_threads
                    )
                    if not mosaic_path:
                        logger.error("ERROR: Download failed again. Please check network and try different resolution/area.")
                        return
                else:
                    logger.info("Pipeline aborted by user.")
                    return

            # Clip if enabled
            if cfg.CLIP_DEM:
                logger.info("Clipping DEM to AOI...")
                clipped = os.path.join(base_dir, "DEM_clipped.tif")
                final_dem = utils.clip_dem_to_aoi(mosaic_path, final_aoi, clipped)
            else:
                final_dem = mosaic_path

            # Get river
            logger.info("Fetching river from NHD...")
            river_path = dc.choose_and_save_nhd_river([final_aoi], base_dir, 1, river_name)

            if not river_path:
                logger.error("\n" + "="*70)
                logger.error("ERROR: No river found in the selected area")
                logger.error("This may be due to: no NHD data, network timeout, or API unavailable")
                logger.error("="*70)

                retry_river = get_choice("\nWhat would you like to do?",
                                        ["Retry river fetch", "Continue anyway (will auto-select)", "Exit"],
                                        default=0)
                if retry_river == 0:
                    logger.info("\nRetrying river fetch...")
                    river_path = dc.choose_and_save_nhd_river([final_aoi], base_dir, 1, river_name)
                    if not river_path:
                        logger.warning("WARNING: Still no river found. Continuing with auto-select.")
                elif retry_river == 2:
                    logger.info("Pipeline aborted by user.")
                    return
                # If retry_river == 1, continue with auto-select (river_path stays None)

        else:  # custom
            final_dem = dem_path
            # river_path already set

        # --- DATA HEALTH CHECK (Non-Blocking) ---
        logger.info("\n" + "="*70)
        logger.info("DEM HEALTH CHECK")
        logger.info("="*70)
        stats = inspect_dem_metadata(final_dem)
        if stats['status'] == 'OK':
            logger.info(f"DEM Verified: {stats['res'][0]:.2f}m resolution | Elev: {stats['min']:.1f}m - {stats['max']:.1f}m")
        else:
            logger.warning(f"WARNING: DEM Issue Detected (Proceeding anyway): {stats.get('msg')}")

        # --- MEMORY SAFETY CHECK (Blocking) ---
        memory_safe = utils.check_memory_safety(final_dem, operation_name="DEM Processing", raise_on_fail=False)
        if not memory_safe:
            logger.error("MEMORY WARNING: This DEM is too large for your system's available RAM")
            logger.warning("Processing this file may cause your system to crash or freeze.")
            logger.warning("Consider using a smaller area or lower resolution.")

            proceed_anyway = get_choice(
                "\nWhat would you like to do?",
                ["Proceed Anyway (Risky)", "Exit and Adjust Settings"],
                default=1
            )

            if proceed_anyway != 0:
                logger.info("Pipeline aborted by user due to memory constraints.")
                return

            logger.warning("User chose to proceed despite memory warning...")

        # Reproject river
        logger.info("Reprojecting river...")
        utils.reproject_vector_to_match_dem(river_path, final_dem, riv_rep_path)

        # Hillshade (optimized for both QA and visualization)
        logger.info("Generating hillshade...")
        hillshade.create_hillshade_fast_qa(
            final_dem,
            hs_path,
            downsample_factor=4,  # Balanced: good quality for QA and visualization
            z_factor=5.5
        )

        # --------------------------------------------------------------------
        # QA CHECK: Generate and display QA image
        # --------------------------------------------------------------------
        logger.info("Generating QA/QC preview...")
        qa_png_path = os.path.join(base_dir, "hillshade_qa_qc.png")
        generate_qa_png(hs_path, riv_rep_path, qa_png_path, final_aoi if mode == "aoi_nhd" else None)

        # Open QA image for review
        logger.info("\n" + "="*70)
        logger.info("QA/QC CHECK: Alignment Check (Hillshade + River)")
        logger.info("="*70)
        logger.info("Generating overlay preview (Raster + Vector)...")
        logger.info(f"\nOpening preview in viewer...")

        try:
            if sys.platform == "darwin":
                subprocess.run(["open", qa_png_path], check=False)
            elif sys.platform == "win32":
                os.startfile(qa_png_path)
            else:
                subprocess.run(["xdg-open", qa_png_path], check=False)
        except Exception as e:
            logger.warning(f"Could not open image automatically: {e}")
            logger.info(f"Please manually open: {qa_png_path}")

        logger.info("\nReview the opened image:")
        logger.info("  • Verify the RED river line matches the channel in the hillshade")

        user_input = input("\nDoes this Alignment Check (Hillshade + River) look correct? [Y/n]: ").strip().lower()
        if user_input in ["n", "no"]:
            logger.error("ERROR: Pipeline aborted by user during QA/QC.")
            return
        logger.info("QA/QC Passed - Continuing pipeline")
        logger.info("="*70)
        # --------------------------------------------------------------------

        # PHASE 2: REM Calculation
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: REM CALCULATION")
        logger.info("="*70)

        # Force projection mode (matching app.py)
        base_mode_auto = "projection"
        k_auto = int(k_neighbors)

        # Detect data source for adaptive cross-section width
        data_source = "nhd" if mode == "aoi_nhd" else "user_upload"

        max_m_comp = utils.feet_to_m(max_comp) if max_comp else None

        with utils.limit_threadpools(compute_threads):
            REM_Calcs.main_rem_calc(
                dem_folder=base_dir if mode == "aoi_nhd" else os.path.dirname(final_dem),
                river_shp=riv_rep_path,
                output_rem_path=rem_out,
                spacing=int(spacing),
                tile_size=2048,
                k_neighbors=int(k_auto),
                max_value=max_m_comp,
                threads=compute_threads,
                idw_power=None,
                base_mode=base_mode_auto,
                engine="scipy",
                data_source=data_source
            )

        # PHASE 3: Visualization
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: VISUALIZATION")
        logger.info("="*70)

        # Clip REM if AOI mode and clipping enabled
        rem_tif = rem_out
        aoi_path = final_aoi if mode == "aoi_nhd" else None
        if cfg.CLIP_DEM and aoi_path and os.path.exists(aoi_path):
            rem_clipped = os.path.join(base_dir, "REM_clipped.tif")
            try:
                logger.info("Clipping REM to AOI...")
                rem_tif = utils.clip_dem_to_aoi(rem_out, aoi_path, rem_clipped)
            except Exception as e:
                logger.warning(f"Could not clip REM: {e}")
                pass

        max_m_vis = utils.feet_to_m(max_vis) if max_vis else None

        # Pre-load hillshade once for optimization (reuse the single hillshade for all visualizations)
        hillshade_array = None
        if background == "hillshade" and os.path.exists(hs_path):
            try:
                logger.info("Pre-loading hillshade for reuse across color ramps...")
                with rasterio.open(rem_tif) as rem_src:
                    rem_transform = rem_src.transform
                    rem_width = rem_src.width
                    rem_height = rem_src.height
                    rem_crs = rem_src.crs

                hillshade_array = style_rem._read_aligned_hillshade(
                    hs_path, rem_transform, rem_width, rem_height, rem_crs
                )
            except Exception as e:
                logger.warning(f"Could not pre-load hillshade: {e}")
                hillshade_array = None

        # Helper function with Smart Legend logic
        def generate_single_ramp(cmap):
            with utils.limit_threadpools(compute_threads):
                return style_rem.style_rem(
                    rem_path=rem_tif,
                    hillshade_path=hs_path if background == "hillshade" else None,
                    out_folder=base_dir,
                    color_maps=[cmap],
                    rem_alpha=1.0,
                    dpi=int(cfg.EXPORT_DPI),
                    max_value=max_m_vis,
                    background=background,
                    scale=int(cfg.EXPORT_SCALE),
                    # Uses the smart legend logic from app.py
                    n_classes=True if n_classes > 0 else None,
                    legend_unit="feet",
                    hard_cap=True,
                    bg_alpha=0.5,
                    hillshade_array=hillshade_array,
                )

        # Generate visualizations in parallel
        from joblib import Parallel, delayed
        if len(selected_ramps) > 1:
            logger.info(f"Generating {len(selected_ramps)} visualizations in parallel...")
            out_pngs = Parallel(n_jobs=min(len(selected_ramps), 4), backend="threading")(
                delayed(generate_single_ramp)(cmap) for cmap in selected_ramps
            )
        else:
            logger.info(f"Generating visualization for {selected_ramps[0]}...")
            out_pngs = [generate_single_ramp(selected_ramps[0])]

        # --- GENERATE STATISTICS REPORT (NEW from app.py) ---
        stats_txt_path = os.path.join(base_dir, "REM_Project_Stats.txt")
        
        report_config = {
            "engine": "SciPy Optimized",
            "spacing": spacing,
            "k_neighbors": k_neighbors,
            "max_rem_comp": max_comp,
            "max_rem_vis": max_vis,
            "png_list": [os.path.basename(p) for p in out_pngs]
        }

        logger.info("Generating comprehensive statistics report...")
        # Note: Ensure rem_utils.py has 'generate_full_stats_report'
        utils.generate_full_stats_report(rem_tif, final_dem, report_config, stats_txt_path)
        print(f"   Statistics Report generated: {os.path.basename(stats_txt_path)}")

    # Cleanup temporary files
    print("\n" + "=" * 70)
    logger.info("Cleaning up temporary files...")
    cleanup_project_outputs(base_dir)

    # Done!
    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)

    print_header(f"SUCCESS! Completed in {int(mins)}m {int(secs)}s")

    print("\nOutput files:")
    print(f"   Project folder: {base_dir}")
    print(f"   REM (GeoTIFF): {rem_tif}")
    print(f"   Statistics Report: {stats_txt_path}")  # Show path to new report


    print(f"   DEM: {final_dem}")

    if len(out_pngs) == 1:
        print(f"   Visualization: {out_pngs[0]}")
    else:
        print(f"   Visualizations ({len(out_pngs)}):")
        for png in out_pngs:
            print(f"      - {os.path.basename(png)}")

    # Offer to open results folder
    if get_choice("\nOpen results folder?", ["Yes", "No"], default=0) == 0:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", base_dir])
        elif sys.platform == "win32":  # Windows
            subprocess.run(["explorer", base_dir])
        else:  # Linux
            subprocess.run(["xdg-open", base_dir])

    # Offer to run another REM
    if get_choice("\nRun another REM?", ["Yes", "No, exit"], default=1) == 0:
        print("\n" + "="*70)
        print("  STARTING NEW REM SESSION")
        print("="*70 + "\n")
        main()  # Restart session
    else:
        print("\nAll done! Happy mapping!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWARNING: Interrupted by user (Ctrl+C)")
        print("Partial results may be saved.")
    except Exception as e:
        logger.error(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()