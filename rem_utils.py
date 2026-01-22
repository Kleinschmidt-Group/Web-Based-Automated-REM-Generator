# rem_utils.py
import os
import sys
import signal
import contextlib
import subprocess
import time
import gc
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import rem_config as cfg

# Import matplotlib for QA plots
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Threading Helpers

# Temporarily set OMP/MKL thread limits to avoid oversubscription
@contextlib.contextmanager
def limit_threadpools(threads=1):
    old_omp = os.environ.get("OMP_NUM_THREADS")
    old_mkl = os.environ.get("MKL_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        yield
    finally:
        if old_omp: os.environ["OMP_NUM_THREADS"] = old_omp
        else: os.environ.pop("OMP_NUM_THREADS", None)

        if old_mkl: os.environ["MKL_NUM_THREADS"] = old_mkl
        else: os.environ.pop("MKL_NUM_THREADS", None)

# Cross-Platform File Operations (Windows Locking Fix)

# Safely remove file with retry logic for Windows file locking
# Retries deletion with exponential backoff and garbage collection to release handles
def safe_remove(filepath, max_retries=5, delay=0.5, verbose=False):
    if not os.path.exists(filepath):
        return True

    for attempt in range(max_retries):
        try:
            os.remove(filepath)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"File locked, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                # Force garbage collection to close any lingering file handles
                gc.collect()
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            else:
                if verbose:
                    print(f"Failed to remove {os.path.basename(filepath)} after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            if verbose:
                print(f"Unexpected error removing {os.path.basename(filepath)}: {e}")
            return False
    return False

# Safely rename/move file with retry logic for Windows file locking
def safe_rename(src, dst, max_retries=5, delay=0.5, verbose=False):
    if not os.path.exists(src):
        if verbose:
            print(f"Source file does not exist: {src}")
        return False

    for attempt in range(max_retries):
        try:
            os.rename(src, dst)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"File locked, retrying rename in {delay}s... (attempt {attempt + 1}/{max_retries})")
                gc.collect()
                time.sleep(delay)
                delay *= 2
                continue
            else:
                if verbose:
                    print(f"Failed to rename {os.path.basename(src)} after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            if verbose:
                print(f"Unexpected error renaming {os.path.basename(src)}: {e}")
            return False
    return False

# CRS Helpers

# Check if CRS is geographic (lat/lon degrees) vs projected (x/y meters)
def is_geographic_crs(crs) -> bool:
    try:
        from pyproj import CRS
        pyproj_crs = CRS.from_user_input(crs)
        return pyproj_crs.is_geographic
    except:
        # Fallback: Check if EPSG is 4326 (WGS84) or similar
        crs_str = str(crs).lower()
        return 'epsg:4326' in crs_str or 'wgs84' in crs_str or 'longlat' in crs_str

# Determine appropriate UTM zone EPSG code from longitude/latitude
def determine_utm_zone(lon: float, lat: float) -> str:
    # Calculate UTM zone number (1-60)
    zone_number = int((lon + 180) / 6) + 1

    # Determine hemisphere (North = 326XX, South = 327XX)
    if lat >= 0:
        epsg_code = f"EPSG:326{zone_number:02d}"
    else:
        epsg_code = f"EPSG:327{zone_number:02d}"

    return epsg_code

# Reproject DEM from geographic CRS (degrees) to projected UTM (meters)
# Auto-detects appropriate UTM zone if not specified
def reproject_dem_to_utm(input_dem_path: str, output_dem_path: str, target_epsg: str = None, verbose: bool = True) -> str:
    from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling
    from pyproj import CRS

    with rasterio.open(input_dem_path) as src:
        # Get source CRS
        src_crs = src.crs

        if verbose:
            print(f"  Source CRS: {src_crs}")

        # Auto-detect UTM zone if not specified
        if target_epsg is None:
            # Use center of DEM bounds
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            target_epsg = determine_utm_zone(center_lon, center_lat)

            if verbose:
                print(f"  Auto-detected UTM zone: {target_epsg}")

        dst_crs = CRS.from_string(target_epsg)

        # Calculate transform for the new CRS
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Define output metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        if verbose:
            print(f"  Reprojecting DEM: {src.width}x{src.height} -> {width}x{height}")
            print(f"  Resolution: {src.res[0]:.2f} -> {transform[0]:.2f} meters")

        # Reproject
        with rasterio.open(output_dem_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=WarpResampling.bilinear,
                num_threads=4
            )

        if verbose:
            print(f"  DEM reprojected to {target_epsg}")

        return output_dem_path, target_epsg

# Vector Utils

# Reproject vector file (river) to match CRS of DEM
def reproject_vector_to_match_dem(vector_path, dem_path, output_path):
    with rasterio.open(dem_path) as src:
        target_crs = src.crs

    gdf = gpd.read_file(vector_path)

    # Debug: Print geometry types
    geom_types = gdf.geometry.geom_type.value_counts()
    print(f"Source geometry types: {geom_types.to_dict()}")

    # Check for infinite bounds in source data
    source_bounds = gdf.total_bounds
    if np.isinf(source_bounds).any() or np.isnan(source_bounds).any():
        raise ValueError(f"ERROR: Source vector has invalid bounds: {source_bounds}")

    # Filter: Keep only LineString and MultiLineString geometries (rivers should be lines)
    from shapely.geometry import LineString, MultiLineString
    valid_types = gdf.geometry.apply(lambda g: isinstance(g, (LineString, MultiLineString)))
    if not valid_types.all():
        print(f"WARNING: Filtering out {(~valid_types).sum()} non-LineString geometries")
        gdf = gdf[valid_types]
        if gdf.empty:
            raise ValueError("ERROR: No LineString/MultiLineString geometries found in river file!")

    # Validate and fix geometries BEFORE reprojection (reprojection can make invalid geometries worse)
    if not gdf.is_valid.all():
        print(f"Found {(~gdf.is_valid).sum()} invalid geometries out of {len(gdf)} (before reprojection)")
        print("Attempting to fix invalid geometries...")

        # For LineStrings, buffer(0) can destroy geometry. Try make_valid instead.
        try:
            from shapely.validation import make_valid
            gdf.geometry = gdf.geometry.apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
            print(f"After make_valid: {len(gdf)} geometries, {(~gdf.is_valid).sum()} still invalid")

            # If still invalid, try buffer(0) as fallback
            if not gdf.is_valid.all():
                print("Trying buffer(0) as fallback...")
                gdf.geometry = gdf.geometry.apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
                print(f"After buffer(0): {(~gdf.is_valid).sum()} still invalid")
        except (ImportError, AttributeError):
            print("shapely.validation.make_valid not available, using buffer(0)...")
            gdf.geometry = gdf.geometry.buffer(0)

        # Remove any empty/null geometries after fix
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna() & gdf.is_valid]
        print(f"After cleanup: {len(gdf)} features remain")

        if gdf.empty:
            raise ValueError("ERROR: All river geometries are empty! NHD High-Res may have corrupted geometries. Try using a different area or river.")

    if gdf.crs != target_crs:
        print(f"Reprojecting Vector from {gdf.crs} to {target_crs}...")
        gdf = gdf.to_crs(target_crs)

    # Check if reprojection created any issues
    if gdf.empty or len(gdf) == 0:
        raise ValueError("ERROR: All river geometries lost during reprojection!")

    # Validate AFTER reprojection - sometimes reprojection creates invalid geometries
    if not gdf.is_valid.all():
        print(f"Found {(~gdf.is_valid).sum()} invalid geometries after reprojection")

        # Check geometry types - if they're still LineStrings, the geometry is probably fine
        geom_types_after = gdf.geometry.geom_type.value_counts()
        print(f"Geometry types after reprojection: {geom_types_after.to_dict()}")

        # Only try to fix if geometry type is still correct (LineString/MultiLineString)
        still_lines = gdf.geometry.apply(lambda g: isinstance(g, (LineString, MultiLineString)))

        if not still_lines.all():
            print(f"ERROR: {(~still_lines).sum()} geometries changed type during reprojection!")
            # Remove non-line geometries
            gdf = gdf[still_lines]
            if gdf.empty:
                raise ValueError("ERROR: All river geometries converted to non-LineString types during reprojection!")

        # For LineString/MultiLineString that are "invalid", just warn and continue
        # The REM calculation can handle minor geometric invalidity (self-intersections, etc.)
        print(f"WARNING: {(~gdf.is_valid).sum()} LineString geometries marked as 'invalid' by GIS validator")
        print("   This is usually due to minor precision issues during reprojection.")
        print("   Proceeding without modification - REM calculation can handle this.")

    # Check for infinite bounds after reprojection
    reprojected_bounds = gdf.total_bounds
    if np.isinf(reprojected_bounds).any() or np.isnan(reprojected_bounds).any():
        print("WARNING: Detected infinite/NaN bounds after reprojection.")
        print(f"Current geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
        print(f"Bounds: {reprojected_bounds}")

        # Check if individual geometries have valid bounds
        valid_bounds_mask = gdf.geometry.apply(lambda g:
            not (np.isinf(g.bounds).any() or np.isnan(g.bounds).any()) if hasattr(g, 'bounds') else False
        )

        print(f"Geometries with valid bounds: {valid_bounds_mask.sum()} / {len(gdf)}")

        if valid_bounds_mask.any():
            # Keep only geometries with valid bounds
            gdf = gdf[valid_bounds_mask]
            print(f"Filtered to {len(gdf)} geometries with valid bounds")

            # Check again
            reprojected_bounds = gdf.total_bounds
            if np.isinf(reprojected_bounds).any() or np.isnan(reprojected_bounds).any():
                raise ValueError(f"ERROR: Still have invalid bounds after filtering: {reprojected_bounds}")
        else:
            # This means the geometry itself is corrupted by the NHD API
            raise ValueError(
                f"ERROR: All river geometries have infinite/NaN bounds after reprojection.\n"
                f"This usually means the NHD data for this river is corrupted.\n"
                f"Try selecting a different river or using a custom river shapefile."
            )

    print(f"Saving {len(gdf)} features to {output_path}")

    # Use GeoPackage format to avoid shapefile field width limitations
    if output_path.endswith('.shp'):
        output_path = output_path.replace('.shp', '.gpkg')

    # Save with error handling for infinite bounds issue
    try:
        gdf.to_file(output_path, driver='GPKG')
    except Exception as e:
        # Fallback: save as Shapefile if GPKG fails
        print(f"Warning: GeoPackage save failed ({e}), falling back to Shapefile...")
        shp_path = output_path.replace('.gpkg', '.shp')
        gdf.to_file(shp_path)
        return shp_path

    return output_path

# Raster Utils

# Clip raster to bounds of GeoJSON AOI
# Enforces blockxsize/blockysize to prevent GDAL errors on small rasters
def clip_dem_to_aoi(dem_path, aoi_geojson, output_path):
    # Read and Validate AOI
    gdf = gpd.read_file(aoi_geojson)
    if not gdf.is_valid.all():
        print("Fixing invalid AOI geometries...")
        gdf.geometry = gdf.geometry.buffer(0)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
        if gdf.empty:
            raise ValueError("ERROR: Invalid AOI geometry.")

    aoi_bounds = gdf.total_bounds
    if np.isinf(aoi_bounds).any() or np.isnan(aoi_bounds).any():
        raise ValueError(f"ERROR: AOI has invalid bounds: {aoi_bounds}")

    with rasterio.open(dem_path) as src:
        print(f"DEM size before clip: {src.width}x{src.height}")
        print(f"DEM CRS: {src.crs}, AOI CRS: {gdf.crs}")

        # Reproject AOI to match DEM CRS with error handling
        if gdf.crs != src.crs:
            print(f"Reprojecting AOI from {gdf.crs} to {src.crs}...")
            try:
                # Try direct reprojection
                gdf_reproj = gdf.to_crs(src.crs)

                # Validate reprojected bounds
                reproj_bounds = gdf_reproj.total_bounds
                if np.isinf(reproj_bounds).any() or np.isnan(reproj_bounds).any():
                    raise ValueError(f"Reprojection produced invalid bounds: {reproj_bounds}")

                gdf = gdf_reproj
                print(f"AOI reprojected successfully")

            except Exception as e:
                print(f"WARNING: AOI reprojection failed ({e})")
                print("Falling back to bbox-only clip using rasterio.windows...")

                # Fallback: Use bounding box window clip instead
                from rasterio.windows import from_bounds
                from rasterio.transform import from_bounds as transform_from_bounds

                # Reproject just the bounding box coordinates
                from shapely.geometry import box
                aoi_box = box(*gdf.total_bounds)
                aoi_box_gdf = gpd.GeoDataFrame(geometry=[aoi_box], crs=gdf.crs)
                aoi_box_reproj = aoi_box_gdf.to_crs(src.crs)
                bbox = aoi_box_reproj.total_bounds

                # Create window from bounds
                window = from_bounds(*bbox, transform=src.transform)

                # Read windowed data
                out_transform = src.window_transform(window)
                out_image = src.read(window=window)
                out_meta = src.meta.copy()

                print(f"DEM size after bbox clip: {out_image.shape[2]}x{out_image.shape[1]}")

                # Write clipped output
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": None,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "bigtiff": "YES"
                })

                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                return output_path

        # Normal geometry-based clip (if reprojection succeeded)
        shapes = [feature["geometry"] for feature in gdf.iterfeatures()]

        # Clip (all_touched=False is better for rivers)
        out_image, out_transform = mask(src, shapes, crop=True, all_touched=False, nodata=src.nodata)
        out_meta = src.meta.copy()

        print(f"DEM size after clip: {out_image.shape[2]}x{out_image.shape[1]}")

    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "compress": None,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "bigtiff": "YES"
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    return output_path

# --- 4. CALCULATION HELPERS ---
def feet_to_m(ft):
    return float(ft) * 0.3048

def _auto_chunk_tile_size_for_rem(rows, cols):
    # Heuristic for determining tile size
    if rows > 20000 or cols > 20000: return 2048
    if rows > 10000 or cols > 10000: return 1024
    return 512

def _call_rem_main_with_filtered_kwargs(module_obj, **kwargs):
    # Calls REM_Calcs.main_rem_calc
    return module_obj.main_rem_calc(**kwargs)

def _call_style_rem_with_filtered_kwargs(module_obj, **kwargs):
    # Calls rem_hillshade_colorramp.style_rem
    return module_obj.style_rem(**kwargs)

# Memory Safety

# Return available system memory in GB
def get_available_memory_gb():
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except ImportError:
        print("WARNING: psutil not installed. Assuming 8GB available.")
        return 8.0
    except Exception as e:
        print(f"WARNING: Could not determine available memory: {e}. Assuming 8GB.")
        return 8.0


# Estimate memory required to load raster into RAM
def estimate_raster_memory_gb(width: int, height: int, bands: int = 1, dtype=np.float32):
    if dtype == np.float32 or dtype == 'float32': bytes_per_element = 4
    elif dtype == np.float64 or dtype == 'float64': bytes_per_element = 8
    elif dtype == np.uint8 or dtype == 'uint8': bytes_per_element = 1
    elif dtype == np.int16 or dtype == 'int16': bytes_per_element = 2
    elif dtype == np.int32 or dtype == 'int32': bytes_per_element = 4
    else: bytes_per_element = 4

    total_bytes = width * height * bands * bytes_per_element
    return total_bytes / (1024 ** 3)


# Check if loading raster would exceed memory safety thresholds
def check_memory_safety(raster_path: str, operation_name: str = "operation", raise_on_fail: bool = False):
    try:
        with rasterio.open(raster_path) as src:
            width = src.width
            height = src.height
            bands = src.count
            dtype = src.dtypes[0]

        estimated_gb = estimate_raster_memory_gb(width, height, bands, dtype)
        available_gb = get_available_memory_gb()
        threshold_gb = available_gb * cfg.MEMORY_SAFETY_THRESHOLD

        if estimated_gb > threshold_gb:
            msg = (
                f"\n{'='*70}\n"
                f"MEMORY WARNING: {operation_name}\n"
                f"Estimated memory needed: {estimated_gb:.2f} GB\n"
                f"Available memory: {available_gb:.2f} GB\n"
                f"Safety threshold: {threshold_gb:.2f} GB\n"
                f"Raster size: {width:,} x {height:,} pixels\n"
                f"{'='*70}\n"
            )
            if raise_on_fail: raise MemoryError(msg)
            else:
                print(msg)
                return False

        elif estimated_gb > cfg.MEMORY_WARNING_GB:
            print(f"\nINFO: {operation_name} will use ~{estimated_gb:.1f} GB RAM")

        return True

    except rasterio.errors.RasterioIOError as e:
        print(f"WARNING: Could not open {raster_path} for memory check: {e}")
        return True 
    except Exception as e:
        print(f"WARNING: Memory check failed: {e}")
        return True 


# Resilience: Sleep Prevention

# Context manager to prevent system sleep during long operations
@contextlib.contextmanager
def prevent_sleep(operation_name: str = "Processing"):
    keep_awake_obj = None
    if not cfg.PREVENT_SLEEP:
        yield
        return

    try:
        from wakepy import keep
        print(f"Preventing system sleep during {operation_name}...")
        keep_awake_obj = keep.running()
        keep_awake_obj.__enter__()
    except ImportError:
        pass # silently ignore if wakepy missing
    except Exception as e:
        print(f"WARNING: Could not prevent sleep: {e}")

    try:
        yield
    finally:
        if keep_awake_obj is not None:
            try: keep_awake_obj.__exit__(None, None, None)
            except: pass


# Resilience: Graceful Shutdown

_shutdown_handlers = []
_shutdown_installed = False

def register_shutdown_handler(handler_func):
    global _shutdown_handlers
    _shutdown_handlers.append(handler_func)

def _graceful_shutdown_handler(signum, frame):
    print("\n" + "="*70)
    print("WARNING: SHUTDOWN REQUESTED")
    print("="*70)
    for handler in _shutdown_handlers:
        try: handler()
        except: pass
    sys.exit(0)

def install_shutdown_handler():
    global _shutdown_installed
    if _shutdown_installed: return
    try:
        signal.signal(signal.SIGINT, _graceful_shutdown_handler)
        signal.signal(signal.SIGTERM, _graceful_shutdown_handler)
        _shutdown_installed = True
    except: pass


# QA/QC Visual Checks

# Open image in system viewer and prompt user for approval
def qa_visual_check(image_path: str, description: str = "output", show_stats: bool = True, vector_path: str = None) -> bool:
    import tempfile

    if not os.path.exists(image_path):
        print(f"WARNING: QA/QC Error: File not found: {image_path}")
        return False

    print(f"\n{'='*70}")
    print(f"QA/QC CHECK: {description}")
    print(f"{'='*70}")

    file_to_open = image_path

    # Overlay Mode
    if vector_path and MATPLOTLIB_AVAILABLE:
        try:
            with rasterio.open(image_path) as src:
                df = max(1, int(max(src.width, src.height) / 4000))
                data = src.read(1, out_shape=(1, int(src.height / df), int(src.width / df)), resampling=Resampling.lanczos)
                extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                if src.nodata is not None: data = np.ma.masked_equal(data, src.nodata)
                
                valid_data = data.compressed() if hasattr(data, 'compressed') else data.flatten()
                vmin, vmax = (np.percentile(valid_data, [2, 98]) if valid_data.size > 0 else (None, None))

            gdf = gpd.read_file(vector_path)
            fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
            ax.imshow(data, cmap='gray', extent=extent, vmin=vmin, vmax=vmax, origin='upper')
            gdf.plot(ax=ax, color='red', linewidth=1.5, alpha=0.9)
            ax.set_title(f"{description}: Raster + River Overlay", fontsize=14, fontweight='bold')
            ax.set_axis_off()

            tmp_png = os.path.join(tempfile.gettempdir(), "rem_qa_overlay.png")
            plt.savefig(tmp_png, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close(fig)
            file_to_open = tmp_png

        except Exception as e:
            print(f"WARNING: Failed to generate overlay plot: {e}")

    # Open in Viewer
    print(f"\nOpening preview in viewer...")
    try:
        if sys.platform == "darwin": subprocess.Popen(["open", file_to_open], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux"): subprocess.Popen(["xdg-open", file_to_open], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32": os.startfile(file_to_open)
    except: pass

    # Prompt
    response = input(f"\nDoes this {description} look correct? [Y/n]: ").strip().lower()
    return response in ['', 'y', 'yes']


#Stats for REM

import numpy as np
import rasterio
from datetime import datetime

# Calculate comprehensive REM and DEM statistics for quality assurance
# Generates detailed report with DEM metadata, REM statistics, spatial coverage, and processing parameters
def generate_full_stats_report(rem_path, dem_path, config_data, output_path):

    # REM Statistics

    with rasterio.open(rem_path) as src:
        rem_data = src.read(1)
        rem_nodata = src.nodata
        rem_transform = src.transform
        rem_width = src.width
        rem_height = src.height
        rem_crs = src.crs

        # Mask nodata values
        mask = (rem_data == rem_nodata) | (rem_data == -999.0) | (~np.isfinite(rem_data))
        valid_rem = np.ma.masked_array(rem_data, mask=mask).compressed()

        # Calculate pixel area
        pixel_area_m2 = abs(rem_transform.a * rem_transform.e)
        total_pixels = rem_width * rem_height
        valid_pixels = valid_rem.size
        nodata_pixels = total_pixels - valid_pixels

    # =========================================================================
    # 2. DEM STATISTICS
    # =========================================================================
    with rasterio.open(dem_path) as d_src:
        dem_data = d_src.read(1)
        dem_nodata = d_src.nodata
        dem_res_x, dem_res_y = d_src.res
        dem_width = d_src.width
        dem_height = d_src.height
        dem_bounds = d_src.bounds
        dem_crs_full = d_src.crs

        # Mask DEM nodata
        dem_mask = (dem_data == dem_nodata) if dem_nodata is not None else np.zeros_like(dem_data, dtype=bool)
        dem_mask |= ~np.isfinite(dem_data)
        valid_dem = np.ma.masked_array(dem_data, mask=dem_mask).compressed()

        dem_total_pixels = dem_width * dem_height
        dem_valid_pixels = valid_dem.size
        dem_nodata_pixels = dem_total_pixels - dem_valid_pixels

    # =========================================================================
    # 3. CALCULATE METRICS
    # =========================================================================

    # REM Statistics (meters)
    if valid_rem.size > 0:
        rem_stats_m = {
            "min": float(np.min(valid_rem)),
            "max": float(np.max(valid_rem)),
            "mean": float(np.mean(valid_rem)),
            "median": float(np.median(valid_rem)),
            "std": float(np.std(valid_rem)),
            "p05": float(np.percentile(valid_rem, 5)),
            "p25": float(np.percentile(valid_rem, 25)),
            "p75": float(np.percentile(valid_rem, 75)),
            "p95": float(np.percentile(valid_rem, 95)),
        }
    else:
        rem_stats_m = {k: 0.0 for k in ["min", "max", "mean", "median", "std", "p05", "p25", "p75", "p95"]}

    # Convert to feet for display
    m_to_ft = 3.280839895
    rem_stats_ft = {k: v * m_to_ft for k, v in rem_stats_m.items()}

    # DEM Statistics (meters)
    if valid_dem.size > 0:
        dem_stats_m = {
            "min": float(np.min(valid_dem)),
            "max": float(np.max(valid_dem)),
            "mean": float(np.mean(valid_dem)),
            "range": float(np.max(valid_dem) - np.min(valid_dem)),
        }
    else:
        dem_stats_m = {k: 0.0 for k in ["min", "max", "mean", "range"]}

    # Spatial Coverage Metrics
    rem_valid_area_km2 = (valid_pixels * pixel_area_m2) / 1_000_000
    rem_total_area_km2 = (total_pixels * pixel_area_m2) / 1_000_000
    rem_completeness_pct = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0

    dem_pixel_area_m2 = abs(dem_res_x * dem_res_y)
    dem_total_area_km2 = (dem_total_pixels * dem_pixel_area_m2) / 1_000_000
    dem_completeness_pct = (dem_valid_pixels / dem_total_pixels * 100) if dem_total_pixels > 0 else 0

    # Data Quality Metrics
    rem_data_density = valid_pixels / rem_total_area_km2 if rem_total_area_km2 > 0 else 0  # pixels per km²

    # =========================================================================
    # 4. WRITE REPORT
    # =========================================================================

    now = datetime.now()

    with open(output_path, "w", encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  REM PROJECT REPORT - ANALYSIS & QUALITY METRICS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Software: Automated REM Generator v1.0\n\n")

        # =====================================================================
        # SECTION 1: DEM SOURCE METADATA
        # =====================================================================
        f.write("─" * 70 + "\n")
        f.write("SECTION 1: SOURCE DEM METADATA\n")
        f.write("─" * 70 + "\n\n")

        f.write("Resolution & Extent:\n")
        f.write(f"  Resolution (X × Y):        {abs(dem_res_x):.2f}m × {abs(dem_res_y):.2f}m\n")
        f.write(f"  Dimensions (W × H):        {dem_width:,} × {dem_height:,} pixels\n")
        f.write(f"  Total Area:                {dem_total_area_km2:.3f} km²\n")
        f.write(f"  Bounding Box (West):       {dem_bounds.left:.6f}\n")
        f.write(f"  Bounding Box (East):       {dem_bounds.right:.6f}\n")
        f.write(f"  Bounding Box (South):      {dem_bounds.bottom:.6f}\n")
        f.write(f"  Bounding Box (North):      {dem_bounds.top:.6f}\n\n")

        f.write("Elevation Statistics:\n")
        f.write(f"  Min Elevation:             {dem_stats_m['min']:.2f}m ({dem_stats_m['min']*m_to_ft:.2f}ft)\n")
        f.write(f"  Max Elevation:             {dem_stats_m['max']:.2f}m ({dem_stats_m['max']*m_to_ft:.2f}ft)\n")
        f.write(f"  Mean Elevation:            {dem_stats_m['mean']:.2f}m ({dem_stats_m['mean']*m_to_ft:.2f}ft)\n")
        f.write(f"  Elevation Range:           {dem_stats_m['range']:.2f}m ({dem_stats_m['range']*m_to_ft:.2f}ft)\n\n")

        f.write("Data Completeness:\n")
        f.write(f"  Valid Pixels:              {dem_valid_pixels:,} ({dem_completeness_pct:.2f}%)\n")
        f.write(f"  NoData Pixels:             {dem_nodata_pixels:,} ({100-dem_completeness_pct:.2f}%)\n")
        f.write(f"  Total Pixels:              {dem_total_pixels:,}\n\n")

        # =====================================================================
        # SECTION 2: REM OUTPUT STATISTICS
        # =====================================================================
        f.write("─" * 70 + "\n")
        f.write("SECTION 2: REM OUTPUT STATISTICS\n")
        f.write("─" * 70 + "\n\n")

        f.write("Summary Statistics:\n")
        f.write(f"  Minimum REM:               {rem_stats_m['min']:.3f} m ({rem_stats_ft['min']:.3f} ft)\n")
        f.write(f"  Maximum REM:               {rem_stats_m['max']:.3f} m ({rem_stats_ft['max']:.3f} ft)\n")
        f.write(f"  Mean REM:                  {rem_stats_m['mean']:.3f} m ({rem_stats_ft['mean']:.3f} ft)\n")
        f.write(f"  Median REM:                {rem_stats_m['median']:.3f} m ({rem_stats_ft['median']:.3f} ft)\n")
        f.write(f"  Std Deviation:             {rem_stats_m['std']:.3f} m ({rem_stats_ft['std']:.3f} ft)\n\n")

        f.write("Distribution Percentiles:\n")
        f.write(f"  5th Percentile:            {rem_stats_m['p05']:.3f} m ({rem_stats_ft['p05']:.3f} ft)\n")
        f.write(f"  25th Percentile (Q1):      {rem_stats_m['p25']:.3f} m ({rem_stats_ft['p25']:.3f} ft)\n")
        f.write(f"  50th Percentile (Median):  {rem_stats_m['median']:.3f} m ({rem_stats_ft['median']:.3f} ft)\n")
        f.write(f"  75th Percentile (Q3):      {rem_stats_m['p75']:.3f} m ({rem_stats_ft['p75']:.3f} ft)\n")
        f.write(f"  95th Percentile:           {rem_stats_m['p95']:.3f} m ({rem_stats_ft['p95']:.3f} ft)\n")
        f.write(f"  Interquartile Range (IQR): {rem_stats_m['p75'] - rem_stats_m['p25']:.3f} m ({rem_stats_ft['p75'] - rem_stats_ft['p25']:.3f} ft)\n\n")

        f.write("Spatial Coverage:\n")
        f.write(f"  REM Dimensions (W × H):    {rem_width:,} × {rem_height:,} pixels\n")
        f.write(f"  Valid Data Area:           {rem_valid_area_km2:.3f} km²\n")
        f.write(f"  Total Extent Area:         {rem_total_area_km2:.3f} km²\n")
        f.write(f"  Data Completeness:         {rem_completeness_pct:.2f}%\n")
        f.write(f"  Valid Pixels:              {valid_pixels:,}\n")
        f.write(f"  NoData/Masked Pixels:      {nodata_pixels:,}\n")
        f.write(f"  Data Density:              {rem_data_density:,.0f} pixels/km²\n\n")

        # =====================================================================
        # SECTION 3: PROCESSING PARAMETERS
        # =====================================================================
        f.write("─" * 70 + "\n")
        f.write("SECTION 3: PROCESSING PARAMETERS (For Reproducibility)\n")
        f.write("─" * 70 + "\n\n")

        f.write("Algorithm Settings:\n")
        f.write(f"  Interpolation Engine:      {config_data.get('engine', 'SciPy Optimized')}\n")
        f.write(f"  River Point Spacing:       {config_data.get('spacing', 'N/A')} m\n")
        f.write(f"  K-Nearest Neighbors:       {config_data.get('k_neighbors', 'N/A')}\n")
        f.write(f"  Smoothing Window:          Adaptive (Auto-calculated based on river length)\n")
        f.write(f"  Sampling Method:           Cross-sectional quantile (10th percentile for thalweg)\n")

        # Adaptive cross-section width based on data source
        data_source = config_data.get('data_source', 'user_upload')
        if data_source == 'nhd':
            f.write(f"  Cross-Section Width:       50 m (NHD mode - robust to centerline misalignment)\n")
        else:
            f.write(f"  Cross-Section Width:       15 m (User mode - assumes verified alignment)\n")

        f.write(f"  IDW Power:                 2.0 (default)\n")
        f.write(f"  Max Search Distance:       5000 m (default)\n\n")

        f.write("Vertical Extent Limits:\n")
        max_comp = config_data.get('max_rem_comp')
        max_vis = config_data.get('max_rem_vis')
        f.write(f"  Max REM (Computation):     {max_comp + ' m' if max_comp else 'None (Full Range)'}\n")
        f.write(f"  Max REM (Visualization):   {max_vis + ' m' if max_vis else 'None (Full Range)'}\n\n")

        # =====================================================================
        # SECTION 4: COORDINATE REFERENCE SYSTEMS
        # =====================================================================
        f.write("─" * 70 + "\n")
        f.write("SECTION 4: COORDINATE REFERENCE SYSTEMS\n")
        f.write("─" * 70 + "\n\n")

        f.write("Output Coordinate Systems:\n")
        f.write(f"  DEM CRS:                   {dem_crs_full}\n")
        f.write(f"  REM CRS:                   {rem_crs}\n")
        f.write(f"  Vertical Datum:            NAVD88 (assumed for CONUS) / EGM96 (Alaska)\n\n")

        f.write("Original Input CRS (if reprojected):\n")
        f.write(f"  DEM Original CRS:          {config_data.get('dem_original_crs', 'Same as output')}\n")
        f.write(f"  River Original CRS:        {config_data.get('river_original_crs', 'N/A')}\n\n")

        # =====================================================================
        # SECTION 5: OUTPUT FILES
        # =====================================================================
        f.write("─" * 70 + "\n")
        f.write("SECTION 5: OUTPUT FILES\n")
        f.write("─" * 70 + "\n\n")

        f.write("Geospatial Outputs:\n")
        f.write(f"  REM GeoTIFF:               {os.path.basename(rem_path)}\n")
        f.write(f"  Source DEM:                {os.path.basename(dem_path)}\n")
        f.write(f"  Full REM Path:             {os.path.abspath(rem_path)}\n\n")

        f.write("Visualization Outputs:\n")
        png_list = config_data.get('png_list', [])
        if png_list:
            for png in png_list:
                f.write(f"  • {png}\n")
        else:
            f.write(f"  (No PNG visualizations generated)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")