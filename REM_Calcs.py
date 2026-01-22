# REM_Calcs.py

import os
import glob
import datetime
import time
import shutil
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import rioxarray 
import xarray as xr
from numba import njit, float64, prange
from scipy.spatial import KDTree
from scipy.signal import savgol_filter
from shapely.geometry import LineString, MultiLineString, GeometryCollection
from shapely import ops
import gc
import time

# Memory Debugging
try:
    import psutil
except ImportError:
    psutil = None

NODATA_REM = -999.0

def log_step(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DEBUG: {msg}", flush=True)


# Helpers 

def find_dem(folder: str) -> str:
    prio = sorted(glob.glob(os.path.join(folder, "mosaic*clipped*.tif")))
    if prio: return prio[0] 
    cand = sorted(glob.glob(os.path.join(folder, "mosaic*.tif")))
    if cand: return cand[0]
    any_tif = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if any_tif: return any_tif[0]
    raise FileNotFoundError("DEM GeoTIFF not found.")

# Extract DEM metadata and close file handles imediately
# Returns metadata disctionary instead of open file handles to prevent windows file locking

def load_and_match_data(dem_path: str, river_shp: str):
    log_step(f"Opening DEM: {dem_path}")

    # Open DEM, extract metadata, and CLOSE immediately (prevents Windows file locking)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        dem_shape = src.shape
        dem_nodata = src.nodata

    log_step("Loading River Shapefile...")
    rivers = gpd.read_file(river_shp).to_crs(dem_crs)

    # Return metadata dictionary instead of open file handles
    return {
        'crs': dem_crs,
        'transform': dem_transform,
        'shape': dem_shape,
        'nodata': dem_nodata
    } 


# Geometry & Filtering

def _merge_to_single_line(gdf: gpd.GeoDataFrame) -> LineString:
    if gdf.empty: raise ValueError("River vector is empty.")
    if hasattr(gdf.geometry, "union_all"): unioned = gdf.geometry.union_all()
    else: unioned = gdf.geometry.unary_union
    if isinstance(unioned, LineString): return unioned
    merged = ops.linemerge(unioned)
    if isinstance(merged, LineString): return merged
    if isinstance(merged, (MultiLineString, GeometryCollection)) and hasattr(merged, "geoms"):
        lines = [geom for geom in merged.geoms if isinstance(geom, LineString)]
        return max(lines, key=lambda g: g.length)
    raise TypeError(f"Unexpected geometry: {type(merged)}")

def _centerline_points(line: LineString, spacing=20) -> gpd.GeoDataFrame:
    npts = max(2, int(np.ceil(line.length / float(spacing))) + 1)
    dists = np.linspace(0.0, line.length, npts)
    pts = [line.interpolate(d) for d in dists]
    return gpd.GeoDataFrame({"s": dists}, geometry=pts, crs=None)

def _remove_points_near_bridges(pts_gdf, roads_path, buffer_m=25.0, crs=None):
    if not roads_path or not os.path.exists(roads_path): return pts_gdf
    log_step(f"Filtering bridges using file: {os.path.basename(roads_path)}")
    try:
        roads = gpd.read_file(roads_path)
        if crs: roads = roads.to_crs(crs)
        bridge_zones = roads.geometry.buffer(float(buffer_m))
        exclusion = bridge_zones.union_all() if hasattr(bridge_zones, "union_all") else bridge_zones.unary_union
        mask = pts_gdf.geometry.disjoint(exclusion)
        return pts_gdf[mask].copy()
    except Exception: return pts_gdf

def _auto_remove_bridge_spikes(pts_gdf, spike_threshold_m=2, window=10):
    if len(pts_gdf) < window * 2: return pts_gdf
    z = pts_gdf["elevation"].values
    trend = pts_gdf["elevation"].rolling(window=window, center=True, min_periods=1).median()
    is_spike = (z > trend + spike_threshold_m)
    if np.sum(is_spike) > 0:
        log_step(f"Auto-Bridge Detection: Removing {np.sum(is_spike)} points")
        return pts_gdf[~is_spike].copy()
    return pts_gdf

def _compute_tangents_normals(pts_gdf):
    xy = np.array([[p.x, p.y] for p in pts_gdf.geometry], dtype="float64")
    n = xy.shape[0]
    if n < 2: return np.zeros((n, 2)), np.zeros((n, 2))
    t = np.zeros_like(xy)
    t[1:-1] = xy[2:] - xy[:-2]; t[0] = xy[1] - xy[0]; t[-1] = xy[-1] - xy[-2]
    tl = np.linalg.norm(t, axis=1); tl[tl == 0] = 1.0
    t_unit = t / tl[:, None]
    return t_unit, np.column_stack((-t_unit[:, 1], t_unit[:, 0]))

def _sample_dem_points(dem_ds, points_gdf, col_name="elevation"):
    coords = [(pt.x, pt.y) for pt in points_gdf.geometry]
    vals = np.array([v[0] for v in dem_ds.sample(coords)], dtype="float32")
    if dem_ds.nodata is not None: vals = np.where(vals == dem_ds.nodata, np.nan, vals)
    out = points_gdf.copy()
    out[col_name] = vals
    return out

# Calculate adaptive cross-section widths to prevent overlaps and bowtie intersections
# Returns array of half-widths (in meters) for each point
def _calculate_adaptive_widths(pts_gdf, normals_xy, river_line, default_half_width=25.0, min_half_width=5.0):
    from shapely.geometry import LineString as ShapelyLineString

    xs = np.array([p.x for p in pts_gdf.geometry], dtype="float64")
    ys = np.array([p.y for p in pts_gdf.geometry], dtype="float64")
    n = xs.size

    # Calculate distances to nearest neighbors
    neighbor_dists = np.full(n, float('inf'), dtype="float64")
    for i in range(n):
        if i > 0:
            neighbor_dists[i] = min(neighbor_dists[i], np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2))
        if i < n-1:
            neighbor_dists[i] = min(neighbor_dists[i], np.sqrt((xs[i]-xs[i+1])**2 + (ys[i]-ys[i+1])**2))

    # Adaptive widths for each point
    adaptive_widths = np.full(n, default_half_width, dtype="float64")
    conflict_count = 0

    for i in range(n):
        nx, ny = normals_xy[i]
        if not np.isfinite(nx):
            continue

        # Start with default, constrained by neighbor distance
        max_safe_width = neighbor_dists[i] * 0.4  # 40% of neighbor distance for safety gap
        current_width = min(default_half_width, max_safe_width)

        # Test for conflicts and reduce if needed
        for attempt in range(10):  # Max 10 reduction attempts
            if current_width < min_half_width:
                break

            # Create cross-section line geometry
            p1 = (xs[i] - current_width * nx, ys[i] - current_width * ny)
            p2 = (xs[i] + current_width * nx, ys[i] + current_width * ny)
            xs_line = ShapelyLineString([p1, p2])

            # Check 1: Bowtie detection (cross-section crosses river multiple times)
            try:
                intersection = xs_line.intersection(river_line)
                # If intersection is more than a single point, we have a bowtie
                if hasattr(intersection, 'geom_type'):
                    if intersection.geom_type in ['MultiPoint', 'LineString', 'MultiLineString', 'GeometryCollection']:
                        # Multiple intersections - reduce width
                        current_width *= 0.7
                        conflict_count += 1
                        continue
            except:
                pass  # Geometry error - keep current width

            # Check 2: Overlap with adjacent cross-sections (only check immediate neighbors)
            has_overlap = False
            for j in [i-1, i+1]:
                if j < 0 or j >= n:
                    continue

                nj_x, nj_y = normals_xy[j]
                if not np.isfinite(nj_x):
                    continue

                # Create neighbor's cross-section with its current width
                neighbor_width = adaptive_widths[j] if j < i else current_width
                pj1 = (xs[j] - neighbor_width * nj_x, ys[j] - neighbor_width * nj_y)
                pj2 = (xs[j] + neighbor_width * nj_x, ys[j] + neighbor_width * nj_y)
                xs_line_j = ShapelyLineString([pj1, pj2])

                try:
                    if xs_line.intersects(xs_line_j):
                        has_overlap = True
                        break
                except:
                    pass

            if has_overlap:
                current_width *= 0.7  # Reduce by 30%
                conflict_count += 1
            else:
                # No conflicts - use this width
                break

        adaptive_widths[i] = max(current_width, min_half_width)

    if conflict_count > 0:
        log_step(f"Smart overlap prevention: Reduced width at {conflict_count} locations to prevent bowties/overlaps")

    return adaptive_widths


def _cross_section_quantile(dem_ds, pts_gdf, normals_xy, half_len_m=30.0, nsamples=21, q=0.0, col_name="elevation", adaptive_widths=None):
    xs = np.array([p.x for p in pts_gdf.geometry], dtype="float64")
    ys = np.array([p.y for p in pts_gdf.geometry], dtype="float64")
    n = xs.size
    zs = np.full(n, np.nan, dtype="float32")

    for i in range(n):
        nx, ny = normals_xy[i]
        if not np.isfinite(nx): continue

        # Use adaptive width if provided, otherwise use default
        width = adaptive_widths[i] if adaptive_widths is not None else half_len_m
        ts = np.linspace(-width, width, int(nsamples))

        coords = list(zip(xs[i] + ts * nx, ys[i] + ts * ny))
        vals = np.array([v[0] for v in dem_ds.sample(coords)], dtype="float32")
        if dem_ds.nodata is not None: vals = np.where(vals == dem_ds.nodata, np.nan, vals)
        v = vals[np.isfinite(vals)]
        if v.size >= max(3, int(0.3 * nsamples)): zs[i] = np.quantile(v, q)
        else: zs[i] = _sample_dem_points(dem_ds, pts_gdf.iloc[[i]], col_name="_tmp")["_tmp"].values[0]
    out = pts_gdf.copy()
    out[col_name] = zs
    return out

# Enforces downstream monotonic decrease with tolerance for natural pools

@njit(float64[:](float64[:], float64, float64, float64), parallel=True)
def _enforce_monotonic_with_pool_tolerance(z, spacing_m, min_slope, tolerance_m=0.5):
    drop = float(min_slope) * float(spacing_m)
    for i in prange(1, z.size):
        # Allow small backwater/pool reversals (natural feature)
        if z[i] > z[i-1] + tolerance_m:
            z[i] = z[i-1] - drop
    return z

def _sanitize_profile(points_gdf, spacing_m, window_m, poly=2, min_slope=1e-4, enforce_isotonic=True):
    z = points_gdf["elevation"].to_numpy().astype("float64")
    if enforce_isotonic:
        # 0.5m tolerance allows natural pools while preventing major reversals
        z = _enforce_monotonic_with_pool_tolerance(z, spacing_m, min_slope, tolerance_m=0.5)
    
    win = max(5, int(round(window_m / max(spacing_m, 1e-6))))
    if win % 2 == 0: win += 1
    try:
        z_m = savgol_filter(z, window_length=max(5, min(win, z.size if z.size%2==1 else z.size-1)), polyorder=2, mode="interp")
    except (ValueError, np.linalg.LinAlgError) as e:
        log_step(f"WARNING: Smoothing filter failed ({e}). Using raw elevations.")
        z_m = z
    out = points_gdf.copy(); out["elevation"] = z_m
    return out

# Calculate adaptive smoothing window based on river length and characteristics

def _calculate_adaptive_smoothing_window(pts_gdf, user_window_m, spacing_m):
    if user_window_m is not None and user_window_m > 0:
        # User specified - respect their choice
        return user_window_m
    
    # Calculate total river length
    total_length_m = len(pts_gdf) * spacing_m

    # Stream length thresholds approximate Strahler stream orders:
    # <2km: 1-2 order (headwater), <10km: 3-4 order (creek),
    # <50km: 5-6 order (small river), >50km: 7+ order (large river)
    # Window scales with stream size to match geomorphic feature scale
    # REDUCED 75% for minimal smoothing - preserves fine-scale features

    if total_length_m < 2000:
        # Small headwater stream (1-2 order) - minimal smoothing
        return max(25, spacing_m * 2)
    elif total_length_m < 10000:
        # Medium creek/stream (3-4 order) - light smoothing
        return max(100, spacing_m * 10)
    elif total_length_m < 50000:
        # Small river (5-6 order) - moderate smoothing
        return max(160, spacing_m * 16)
    else:
        # Large river (7+ order) - conservative smoothing
        return max(375, spacing_m * 37)


# Interpolation Engine

#Optimized SciPy IDW with sidtance pruning
# Skips pixels further than 'max_dist' from the river

def _engine_scipy_kdtree(tile_coords, tree, elevations, k, power, workers, max_dist=5000.0):
    # Query with distance upper bound (Fast pruning)
    dists, idxs = tree.query(tile_coords, k=k, workers=workers, distance_upper_bound=max_dist)
    
    if dists.ndim == 1: 
        dists, idxs = dists[:, None], idxs[:, None]
    
    # SciPy sets invalid indices to tree.n
    invalid_mask = (idxs == tree.n)
    safe_idxs = np.where(invalid_mask, 0, idxs)
    
    # Compute Weights
    p = float(power) if power is not None else 2.0
    
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / (np.power(dists, p) + 1e-6)
    
    w = np.where(invalid_mask, 0.0, w)
    w_sum = np.sum(w, axis=1)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        w_norm = w / w_sum[:, None]
    
    w_norm = np.nan_to_num(w_norm, copy=False)
    z_interp = np.sum(w_norm * elevations[safe_idxs], axis=1)
    
    # If a pixel had 0 total weight (no neighbors within max_dist), set it to NaN
    z_interp = np.where(w_sum > 0, z_interp, np.nan)
    
    return z_interp

def process_tile_interpolation(engine, tile_coords, tree=None, elevations=None, k=8, power=2.0, workers=1, max_dist=5000.0, **kwargs):
    return _engine_scipy_kdtree(tile_coords, tree, elevations, k, power, workers, max_dist)

def generate_base_surface_memmap(dem_path, dem_transform, shape, pts_gdf, tile_size=2048, k_neighbors=50, power=2.0, workers=None, temp_dir=".", max_dist=5000.0, absolute_cutoff=None, **kwargs):
    rows, cols = shape
    base_filename = os.path.join(temp_dir, "rem_base_surface.dat")
    out = np.memmap(base_filename, dtype='float32', mode='w+', shape=(rows, cols))
    
    a, b, c, d, e, f = dem_transform.a, dem_transform.b, dem_transform.c, dem_transform.d, dem_transform.e, dem_transform.f
    
    log_step("Building KDTree for SciPy engine...")
    river_coords = np.array([(p.x, p.y) for p in pts_gdf.geometry], dtype="float64")
    tree = KDTree(river_coords, leafsize=16)
    river_vals = pts_gdf["elevation"].values
    
    log_step(f"Starting Interpolation. Tiles: {tile_size}px. Max Search: {max_dist}m. Z-Cutoff: {absolute_cutoff}m")
    
    total_tiles = (int(np.ceil(rows/tile_size))) * (int(np.ceil(cols/tile_size)))
    tile_count = 0
    skipped_pixels = 0
    total_pixels_processed = 0
    
    # Open DEM to read Z-values for optimization
    with rasterio.open(dem_path) as dem_src:
        for row_start in range(0, rows, tile_size):
            row_end = min(row_start + tile_size, rows)
            r = np.arange(row_start, row_end, dtype=np.float64)
            h = row_end - row_start
            
            for col_start in range(0, cols, tile_size):
                tile_count += 1
                col_end = min(col_start + tile_size, cols)
                w = col_end - col_start
                
                # Log EVERY tile so you see movement immediately
                if tile_count % 10 == 0: 
                    log_step(f"Processing Tile {tile_count}/{total_tiles} (Skipped: {skipped_pixels/1e6:.1f}M px)...")
                
                # Read DEM Block
                window = Window(col_start, row_start, w, h)
                dem_data = dem_src.read(1, window=window).astype("float32")
                
                # Mask pixels strictly above cutoff (if cutoff exists)
                valid_mask = np.ones(dem_data.shape, dtype=bool)
                if absolute_cutoff is not None:
                    with np.errstate(invalid='ignore'):
                        valid_mask = (dem_data <= absolute_cutoff) | np.isnan(dem_data)
                
                # Count optimization stats
                n_valid = np.count_nonzero(valid_mask)
                skipped_pixels += (dem_data.size - n_valid)
                total_pixels_processed += dem_data.size
                
                # If tile is 100% mountains, skip entirely
                if n_valid == 0:
                    out[row_start:row_end, col_start:col_end] = np.nan
                    continue
                
                # Generate Coordinates ONLY for Valid Pixels (Vectorized Subsetting)
                cidx = np.arange(col_start, col_end, dtype=np.float64)
                X_grid = (c + a * cidx[None, :] + b * r[:, None])
                Y_grid = (f + d * cidx[None, :] + e * r[:, None])
                
                X_flat = X_grid[valid_mask]
                Y_flat = Y_grid[valid_mask]
                
                tile_coords = np.column_stack((X_flat, Y_flat))
                
                # Run Engine on subset
                if tile_coords.shape[0] > 0:
                    z_subset = process_tile_interpolation("scipy", tile_coords, tree=tree, elevations=river_vals, k=int(k_neighbors), power=power, workers=workers, max_dist=max_dist)
                    
                    # Map back to full tile
                    full_tile = np.full((h, w), np.nan, dtype=np.float32)
                    full_tile[valid_mask] = z_subset
                    
                    out[row_start:row_end, col_start:col_end] = full_tile
                else:
                    out[row_start:row_end, col_start:col_end] = np.nan
                
                if tile_count % 20 == 0: 
                    out.flush()
    
    out.flush()
    if total_pixels_processed > 0:
        log_step(f"Interpolation Done. Skipped {skipped_pixels/1e6:.1f}M / {total_pixels_processed/1e6:.1f}M pixels ({skipped_pixels/total_pixels_processed*100:.1f}%) due to Z-Filter.")
    
    return out, base_filename


# Output Streaming

def stream_rem_subtraction(dem_path, base_memmap, output_path, max_value=None):
    log_step("Streaming final REM calculation...")
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=NODATA_REM, compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256, bigtiff="YES")
        
        with rasterio.open(output_path, "w", **profile) as dst:
            block_sz = 2048
            for row_off in range(0, src.height, block_sz):
                h = min(block_sz, src.height - row_off)
                for col_off in range(0, src.width, block_sz):
                    w = min(block_sz, src.width - col_off)
                    window = Window(col_off, row_off, w, h)
                    
                    dem_data = src.read(1, window=window).astype("float32")
                    if src.nodata is not None: 
                        dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)
                    dem_data = np.where(np.isfinite(dem_data), dem_data, np.nan)
                    
                    base_data = base_memmap[row_off:row_off+h, col_off:col_off+w]
                    
                    rem_data = dem_data - base_data
                    if max_value is not None: 
                        rem_data = np.where(rem_data <= float(max_value), rem_data, NODATA_REM)
                    rem_data = np.where(np.isfinite(rem_data), rem_data, NODATA_REM)
                    
                    dst.write(rem_data, 1, window=window)
    
    log_step("Streaming complete.")


# Main


def main_rem_calc(dem_folder, river_shp, output_rem_path, spacing=20, tile_size=2048, k_neighbors=100, max_value=None,
                  threads=None, idw_power=None, roads_path=None, bridge_buffer_m=25.0,
                  enforce_isotonic=True, max_search_dist=None, engine="scipy", data_source="user_upload", **kwargs):
    
    log_step(f"--- STARTED REM CALCULATION ---")
    log_step(f"Engine: SCIPY (Optimized + Smart Z-Filter)")
    if threads is None: threads = -1
    
    # Safety Default for Power to prevent TypeError
    if idw_power is None: idw_power = 2.0

    # Adaptive cross-section width based on data source
    if data_source == "nhd":
        # Conservative - NHD lines often offset from DEM by 5-30m
        # Due to different data vintages, coordinate shifts, river migration
        half_width_m = 25.0  # 50m total width
        log_step("Cross-section sampling: 50m width (NHD mode - robust to centerline misalignment)")
    else:  # user_upload
        # Precise - user instructed to verify 97% thalweg alignment before use
        half_width_m = 7.5   # 15m total width
        log_step("Cross-section sampling: 15m width (User mode - assumes verified centerline alignment)")

    dem_path = find_dem(dem_folder)
    dem_meta = load_and_match_data(dem_path, river_shp)

    log_step("Processing Centerline...")
    rivers = gpd.read_file(river_shp).to_crs(dem_meta['crs'])
    line = _merge_to_single_line(rivers)
    pts_gdf = _centerline_points(line, spacing=spacing)
    pts_gdf.crs = rivers.crs
    if roads_path: pts_gdf = _remove_points_near_bridges(pts_gdf, roads_path, buffer_m=bridge_buffer_m, crs=rivers.crs)

    log_step(f"Using {len(pts_gdf)} river points.")
    tangents, normals = _compute_tangents_normals(pts_gdf)

    # Calculate adaptive widths to prevent cross-section overlaps and bowtie intersections
    log_step("Calculating smart adaptive cross-section widths (prevents overlaps in high sinuosity reaches)...")
    adaptive_widths = _calculate_adaptive_widths(pts_gdf, normals, line, default_half_width=half_width_m, min_half_width=5.0)

    with rasterio.open(dem_path) as temp_ds:
        # Using 10th percentile: captures thalweg (low channel) while filtering LiDAR noise
        # (min would be too sensitive to outliers, median would be too high)
        log_step("Sampling river elevations: 50th percentile across cross-sections (filters noise while capturing thalweg)")
        pts_gdf = _cross_section_quantile(temp_ds, pts_gdf, normals, half_width_m, 21, 0.50, adaptive_widths=adaptive_widths)
        pts_gdf = pts_gdf[np.isfinite(pts_gdf["elevation"])]
        if pts_gdf.empty: raise RuntimeError("River sampling failed.")
        pts_gdf = _auto_remove_bridge_spikes(pts_gdf, 1.5, 15)

        # Always use adaptive smoothing window based on river length
        adaptive_window = _calculate_adaptive_smoothing_window(pts_gdf, 50, spacing)
        log_step(f"Auto-calculated smoothing window: {adaptive_window}m (based on river length)")

        pts_gdf = _sanitize_profile(pts_gdf, spacing, adaptive_window, 2, 1e-4, enforce_isotonic)

    # Calculate absolute z cutoff
    absolute_cutoff = None
    if max_value is not None:
        max_river_elev = pts_gdf["elevation"].max()
        # 5x buffer: Performance optimization with trade-offs
        # - Skips mountains far from river (massive speedup in canyons)
        # - Captures terraces within reasonable range
        buffer_z = float(max_value) * 5.0
        absolute_cutoff = max_river_elev + buffer_z
        log_step(f"Smart Z-Filter Active. Max River: {max_river_elev:.1f}m. Buffer: {buffer_z:.1f}m (5x multiplier). Cutoff: {absolute_cutoff:.1f}m")
        log_step(f"User Max REM: {max_value:.1f}m - REM values will be capped at this limit")
    else:
        log_step("No REM cap set - calculating for entire DEM")
    
    if max_search_dist is None:
        max_search_dist = float('inf')

    # Manual temp directory management
    base_memmap = None
    base_file = None
    temp_dir = None
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        base_memmap, base_file = generate_base_surface_memmap(
            dem_path, dem_meta['transform'], dem_meta['shape'], pts_gdf, tile_size=tile_size, k_neighbors=k_neighbors,
            power=idw_power, workers=threads, temp_dir=temp_dir, max_dist=max_search_dist, absolute_cutoff=absolute_cutoff
        )
        
        stream_rem_subtraction(dem_path, base_memmap, output_rem_path, max_value=max_value)
        
    finally:
        # Explicit cleanup sequence
        # Delete memmap object (releases file handle on both platforms)
        if base_memmap is not None:
            del base_memmap
            base_memmap = None
        
        # Force garbage collection (helps both platforms, critical for Windows)
        gc.collect()
        
        # Brief pause for Windows file system to release locks
        time.sleep(0.1)
        
        # Manual file deletion with cross-platform retry logic
        if base_file and os.path.exists(base_file):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.remove(base_file)
                    break  # Success - file deleted
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.2 * (attempt + 1))  # Exponential backoff
                    else:
                        # Final attempt failed - log but don't crash
                        log_step(f"WARNING: Could not delete temp file {os.path.basename(base_file)}: {e}")
                        log_step("  This is harmless - the file will be cleaned up when Python exits")
        
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except (PermissionError, OSError) as e:
                log_step(f"WARNING: Could not remove temp directory: {e}")
                log_step("  This is harmless - OS will clean up on exit")

    log_step(f"DONE: REM saved to {output_rem_path}")