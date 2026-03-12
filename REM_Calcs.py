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
from scipy.ndimage import gaussian_filter1d
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


def _engine_perpendicular_projection(tile_coords, tree, river_coords, river_s, river_elevations, max_dist=5000.0):
    """
    For each pixel, project perpendicularly onto the river centerline and
    interpolate elevation from the 1D smoothed profile.

    Seam-free blending: instead of hard-assigning each pixel to one segment
    (which creates visible Voronoi boundary lines), we find the two closest
    segment projections and blend their elevations with inverse-distance
    weights. Pixels deep inside a zone get ~100% from the nearest segment;
    pixels near a boundary get a smooth blend — no visible lines.

    Using k=3 nearest points covers 6 candidate segments per pixel, ensuring
    the correct segment is always found even near curved river bends.
    """
    n_river = len(river_coords)
    px = tile_coords[:, 0]
    py = tile_coords[:, 1]
    n_pix = len(px)

    # k=3 → up to 6 candidate segments; catches the correct segment even near
    # Voronoi boundaries where k=2 could miss the true closest segment
    k_query = min(3, n_river)
    dists, idxs = tree.query(tile_coords, k=k_query, distance_upper_bound=max_dist, workers=1)

    if dists.ndim == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    no_neighbor  = dists[:, 0] >= max_dist
    valid_idxs   = np.where(idxs >= n_river, -1, idxs)  # -1 = invalid slot

    def _project_segment(a_idx, b_idx):
        """Vectorised projection; returns (s_proj, perp_dist_sq)."""
        valid  = (a_idx >= 0) & (b_idx >= 0) & (b_idx < n_river)
        a_safe = np.clip(a_idx, 0, n_river - 1)
        b_safe = np.clip(b_idx, 0, n_river - 1)

        ax = river_coords[a_safe, 0];  ay = river_coords[a_safe, 1]
        bx = river_coords[b_safe, 0];  by = river_coords[b_safe, 1]

        dx = bx - ax;  dy = by - ay
        seg_len_sq = np.where(dx*dx + dy*dy < 1e-10, 1e-10, dx*dx + dy*dy)

        vx = px - ax;  vy = py - ay
        t  = np.clip((vx*dx + vy*dy) / seg_len_sq, 0.0, 1.0)

        proj_x = ax + t*dx;  proj_y = ay + t*dy
        perp_d2 = (px - proj_x)**2 + (py - proj_y)**2
        s_proj  = river_s[a_safe] + t * (river_s[b_safe] - river_s[a_safe])

        perp_d2 = np.where(valid, perp_d2, np.inf)
        return s_proj, perp_d2

    # Track the two closest segment projections (best-1 and best-2) for blending
    best1_s = np.zeros(n_pix, dtype=np.float64)
    best2_s = np.zeros(n_pix, dtype=np.float64)
    best1_d = np.full(n_pix, np.inf)
    best2_d = np.full(n_pix, np.inf)

    for ki in range(k_query):
        j       = valid_idxs[:, ki]
        valid_j = j >= 0
        j_safe  = np.where(valid_j, j, 0)

        for a_off, b_off in ((-1, 0), (0, 1)):
            s_seg, d_seg = _project_segment(j_safe + a_off, j_safe + b_off)
            d_seg = np.where(valid_j, d_seg, np.inf)

            # Is this candidate better than current best-1?
            to_1 = d_seg < best1_d
            # Not better than best-1 but better than best-2?
            to_2 = (~to_1) & (d_seg < best2_d)

            # Demote best-1 → best-2 where a new best-1 is found
            best2_s = np.where(to_1, best1_s, np.where(to_2, s_seg, best2_s))
            best2_d = np.where(to_1, best1_d, np.where(to_2, d_seg, best2_d))
            best1_s = np.where(to_1, s_seg,   best1_s)
            best1_d = np.where(to_1, d_seg,   best1_d)

    # Elevations for the two best projections
    e1 = np.interp(best1_s, river_s, river_elevations).astype(np.float32)
    e2 = np.interp(best2_s, river_s, river_elevations).astype(np.float32)

    # Inverse-distance blend (actual distance, not squared)
    d1 = np.sqrt(np.maximum(best1_d, 0.0)) + 1e-6
    d2 = np.sqrt(np.maximum(best2_d, 0.0)) + 1e-6
    w1 = 1.0 / d1
    w2 = 1.0 / d2

    # If no second segment was found (river start/end), fall back to e1 only
    has_second = best2_d < np.inf
    z_out = np.where(has_second,
                     (w1 * e1 + w2 * e2) / (w1 + w2),
                     e1).astype(np.float32)

    z_out = np.where(no_neighbor, np.nan, z_out)
    return z_out

def _smooth_base_surface_inplace(memmap, sigma_pixels, strip_size=1024):
    """
    Separable NaN-safe Gaussian blur applied to the base surface in-place.

    Eliminates the faint perpendicular banding caused by discrete cross-section
    zones in the perpendicular projection method.  Uses two 1-D passes
    (horizontal then vertical) so no full-array copy is ever needed.

    Strip processing keeps peak memory to (strip_size × max_dim × 4) bytes.
    """
    rows, cols = memmap.shape
    if sigma_pixels < 0.5:
        return

    log_step(f"Smoothing base surface (sigma={sigma_pixels:.1f} px) to remove band artifacts...")

    def _nan_gauss_1d(arr, sigma, axis):
        """Weighted convolution that ignores NaN cells."""
        nan_mask = ~np.isfinite(arr)
        filled  = np.where(nan_mask, 0.0, arr).astype(np.float64)
        weights = np.where(nan_mask, 0.0, 1.0).astype(np.float64)
        b_data  = gaussian_filter1d(filled,  sigma, axis=axis, mode='nearest')
        b_wt    = gaussian_filter1d(weights, sigma, axis=axis, mode='nearest')
        result  = np.where(b_wt > 1e-6, b_data / b_wt, np.nan)
        return result.astype(np.float32)

    # Pass 1 — horizontal (axis=1): rows are independent → row-strips, no overlap
    for r0 in range(0, rows, strip_size):
        r1 = min(rows, r0 + strip_size)
        strip = memmap[r0:r1, :].copy()
        memmap[r0:r1, :] = _nan_gauss_1d(strip, sigma_pixels, axis=1)
    memmap.flush()

    # Pass 2 — vertical (axis=0): columns are independent → col-strips, no overlap
    for c0 in range(0, cols, strip_size):
        c1 = min(cols, c0 + strip_size)
        strip = memmap[:, c0:c1].copy()
        memmap[:, c0:c1] = _nan_gauss_1d(strip, sigma_pixels, axis=0)
    memmap.flush()

    log_step("Base surface smoothing complete.")


def _engine_flow_weighted(tile_coords, tree, river_coords, tangents, river_elevations, spacing, k=8, max_dist=5000.0):
    """
    Flow-weighted interpolation: weights each river point by the inverse of its
    along-channel distance to the pixel (i.e. how far up/downstream the pixel is
    from that cross-section), rather than by Euclidean distance.

    Why this eliminates bend artifacts
    -----------------------------------
    Perpendicular projection hard-assigns every pixel to its closest segment.
    At a meander the Voronoi boundary between two adjacent segments cuts across
    the floodplain at an angle — producing a visible seam.

    Here, every nearby river point contributes, weighted by
        w_j = 1 / (along_j^2 + epsilon)
    where  along_j = dot( pixel - river_j, tangent_j )
    is the signed along-channel offset.  When a pixel sits exactly across from
    cross-section j, along_j ≈ 0 → weight is 1/epsilon (maximum).  As the pixel
    moves upstream/downstream from j, the weight drops continuously.

    Because tangent_j rotates gradually around a bend, so do the weights — the
    transition is smooth everywhere, with no hard boundaries.

    epsilon = (spacing/2)^2 sets the blending half-width to half the sample
    spacing: a pixel must be within ~spacing/2 m along-channel before it gets
    appreciable weight from that section.
    """
    n_river = len(river_coords)
    n_pix   = len(tile_coords)

    k_query = min(k, n_river)
    dists, idxs = tree.query(tile_coords, k=k_query, distance_upper_bound=max_dist, workers=1)

    if dists.ndim == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    no_neighbor = dists[:, 0] >= max_dist

    # epsilon controls the along-channel blending width.
    # A pixel at the midpoint between two cross-sections (along_dist = spacing/2) has:
    #   weight_ratio = epsilon / ((spacing/2)^2 + epsilon)
    # Setting epsilon = spacing^2 gives ~80% at the midpoint — smooth enough to eliminate
    # visible zone boundaries at any spacing, while still correctly weighting by position.
    # Using (spacing/2)^2 gave only 50% at the midpoint — too sharp at tight spacings.
    epsilon = float(spacing) ** 2

    px = tile_coords[:, 0]
    py = tile_coords[:, 1]

    z_acc = np.zeros(n_pix, dtype=np.float64)
    w_acc = np.zeros(n_pix, dtype=np.float64)

    for ki in range(k_query):
        j       = idxs[:, ki]
        valid   = j < n_river
        j_safe  = np.where(valid, j, 0)

        # Vector from river point j → pixel
        vx = px - river_coords[j_safe, 0]
        vy = py - river_coords[j_safe, 1]

        # Along-channel component at cross-section j
        tx = tangents[j_safe, 0]
        ty = tangents[j_safe, 1]
        along = vx * tx + vy * ty   # signed; 0 = directly across from section j

        # Weight peaks when pixel is directly across from this section
        w = np.where(valid, 1.0 / (along * along + epsilon), 0.0)

        elev = river_elevations[j_safe].astype(np.float64)
        z_acc += w * elev
        w_acc += w

    with np.errstate(divide="ignore", invalid="ignore"):
        z_out = np.where(w_acc > 0.0, z_acc / w_acc, np.nan)

    z_out = np.where(no_neighbor, np.nan, z_out)
    return z_out.astype(np.float32)


def generate_base_surface_memmap(dem_path, dem_transform, shape, pts_gdf, tile_size=2048, k_neighbors=50, power=2.0, workers=None, temp_dir=".", max_dist=5000.0, absolute_cutoff=None, engine="projection", river_s=None, spacing=20, tangents=None, **kwargs):
    rows, cols = shape
    base_filename = os.path.join(temp_dir, "rem_base_surface.dat")
    out = np.memmap(base_filename, dtype='float32', mode='w+', shape=(rows, cols))
    
    a, b, c, d, e, f = dem_transform.a, dem_transform.b, dem_transform.c, dem_transform.d, dem_transform.e, dem_transform.f
    
    use_flow_weighted   = (engine == "projection") and (tangents is not None)
    use_projection      = (engine == "projection") and (river_s is not None) and not use_flow_weighted

    if use_flow_weighted:
        engine_label = "Flow-Weighted (smooth, bend-safe)"
    elif use_projection:
        engine_label = "Perpendicular Projection (legacy)"
    else:
        engine_label = "IDW (SciPy KDTree)"
    log_step(f"Building KDTree — Engine: {engine_label}...")

    river_coords = np.array([(p.x, p.y) for p in pts_gdf.geometry], dtype="float64")
    tree = KDTree(river_coords, leafsize=16)
    river_vals = pts_gdf["elevation"].values

    # Projection / flow-weighted engine needs the s array
    if use_projection:
        river_s_arr = np.asarray(river_s, dtype="float64")

    # Flow-weighted engine needs the tangent array
    if use_flow_weighted:
        tangents_arr = np.asarray(tangents, dtype="float64")

    log_step(f"Starting Interpolation. Engine: {engine_label}. Tiles: {tile_size}px. Max Search: {max_dist}m. Z-Cutoff: {absolute_cutoff}m")
    
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
                    if use_flow_weighted:
                        z_subset = _engine_flow_weighted(
                            tile_coords, tree, river_coords, tangents_arr, river_vals,
                            spacing=spacing, k=8, max_dist=max_dist
                        )
                    elif use_projection:
                        z_subset = _engine_perpendicular_projection(
                            tile_coords, tree, river_coords, river_s_arr, river_vals, max_dist=max_dist
                        )
                    else:
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

    # Flow-weighted engine is smooth by construction — no blur needed.
    # Legacy perpendicular projection produces discrete cross-section bands; blur
    # those out with two Gaussian passes.
    if use_projection and not use_flow_weighted:
        pixel_size = abs(float(dem_transform.a))
        sigma_px   = float(spacing) / max(pixel_size, 1e-6)
        sigma_px   = max(3.0, min(sigma_px, 60.0))
        log_step(f"Base surface smoothing (legacy projection): sigma={sigma_px:.1f} px ({sigma_px * pixel_size:.1f} m), 2 passes")
        _smooth_base_surface_inplace(out, sigma_px)
        _smooth_base_surface_inplace(out, sigma_px)

    return out, base_filename


# HAND Engine

def generate_base_surface_hand(dem_path, dem_transform, dem_shape, pts_gdf,
                                temp_dir=".", spacing=20, max_dist=5000.0,
                                absolute_cutoff=None, **kwargs):
    """
    HAND (Height Above Nearest Drainage) base surface.

    For each terrain pixel, follows the D8 flow network downhill until it
    reaches a river cell, then assigns the smoothed river profile elevation
    of that cell as the base elevation.

    This completely eliminates Voronoi seams and bend artifacts because the
    assignment follows actual water-flow paths, not geometric distance.
    River-bend pixels naturally drain to the correct reach of the river —
    no perpendicular lines, no zone boundaries.

    Pipeline
    --------
    1. Condition DEM: pit-fill → fill depressions → resolve flats (pysheds)
    2. D8 flow direction
    3. Rasterize river centerline (buffered to ensure pixel coverage)
    4. Path-doubling from river cells upstream → each pixel learns which
       river cell it ultimately drains to (O(N · log D), fully vectorized)
    5. Look up smoothed profile elevation at that river cell
    6. Write to memmap

    Requires pysheds:  pip install pysheds
    """
    try:
        from pysheds.grid import Grid
    except ImportError:
        raise ImportError(
            "pysheds is required for the HAND engine.\n"
            "Install with:  pip install pysheds"
        )

    from rasterio.features import rasterize as rio_rasterize
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import Point
    from shapely.geometry import mapping as shp_mapping

    rows, cols = dem_shape
    pixel_size = abs(float(dem_transform.a))
    n = rows * cols

    # Memory estimate (pysheds holds ~3 float32 copies of the DEM during conditioning)
    mem_gb = (n * 4 * 4) / 1e9
    log_step(f"HAND: DEM {cols}×{rows} px — estimated peak RAM ~{mem_gb:.1f} GB")
    if psutil is not None:
        avail_gb = psutil.virtual_memory().available / 1e9
        if mem_gb > avail_gb * 0.75:
            log_step(
                f"WARNING: Estimated RAM ({mem_gb:.1f} GB) is close to available "
                f"({avail_gb:.1f} GB). Consider the Flow-Weighted engine for very large DEMs."
            )

    # ── Step 1: Condition DEM ──────────────────────────────────────────────
    log_step("HAND: Conditioning DEM (pit-fill → depressions → flats)...")
    grid     = Grid.from_raster(dem_path)
    dem_data = grid.read_raster(dem_path)
    conditioned = grid.resolve_flats(
        grid.fill_depressions(
            grid.fill_pits(dem_data)
        )
    )

    # ── Step 2: D8 flow direction ──────────────────────────────────────────
    log_step("HAND: Computing D8 flow direction...")
    fdir     = grid.flowdir(conditioned)
    fdir_arr = np.array(fdir, dtype=np.int32).ravel()
    del conditioned, fdir, dem_data
    gc.collect()

    # ── Step 3: Build downstream flat-index array (vectorised) ────────────
    log_step("HAND: Building vectorised flow graph...")

    # pysheds default ESRI dirmap: N=64, NE=128, E=1, SE=2, S=4, SW=8, W=16, NW=32
    D8_OFFSETS = {
        64:  (-1,  0),   # N
        128: (-1,  1),   # NE
        1:   ( 0,  1),   # E
        2:   ( 1,  1),   # SE
        4:   ( 1,  0),   # S
        8:   ( 1, -1),   # SW
        16:  ( 0, -1),   # W
        32:  (-1, -1),   # NW
    }

    flat_idx  = np.arange(n, dtype=np.int32)
    row_of    = flat_idx // cols
    col_of    = flat_idx % cols
    downstream = np.full(n, -1, dtype=np.int32)

    for d_val, (dr, dc) in D8_OFFSETS.items():
        mask = fdir_arr == d_val
        if not np.any(mask):
            continue
        nr = row_of[mask] + dr
        nc = col_of[mask] + dc
        ok = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
        src = flat_idx[mask][ok]
        downstream[src] = (nr[ok] * cols + nc[ok]).astype(np.int32)

    del fdir_arr, flat_idx, row_of, col_of
    gc.collect()

    # ── Step 4: Rasterise river centerline ────────────────────────────────
    log_step("HAND: Rasterising river centerline...")
    buf_m = max(pixel_size * 2.0, spacing * 0.5)
    river_shapes = [
        (shp_mapping(Point(p.x, p.y).buffer(buf_m)), 1)
        for p in pts_gdf.geometry
    ]
    is_river_2d = rio_rasterize(
        river_shapes, out_shape=(rows, cols),
        transform=dem_transform, fill=0, dtype='uint8'
    ).astype(bool)
    is_river = is_river_2d.ravel()

    # ── Step 5: Path-doubling — find nearest river cell for every pixel ───
    # Each pixel starts with `parent` = its downstream neighbour.
    # River cells point to themselves (they ARE the target).
    # Non-river sinks also point to themselves (dead end; will be excluded).
    # Repeated application of  parent = parent[parent]  halves path lengths
    # each step → converges in O(log D) iterations for max path depth D.
    log_step("HAND: Tracing flow paths to nearest river cell (path-doubling)...")

    parent = downstream.copy()

    river_cells = np.where(is_river)[0].astype(np.int32)
    parent[river_cells] = river_cells          # river cells are their own root

    sink_mask = (parent < 0) & ~is_river
    parent[np.where(sink_mask)[0]] = np.where(sink_mask)[0].astype(np.int32)

    max_iters = max(25, int(np.ceil(np.log2(max(rows, cols) + 1))) + 6)
    for i in range(max_iters):
        p_safe   = np.where(parent >= 0, parent, 0).astype(np.int64)
        new_par  = parent[p_safe].astype(np.int32)
        if np.array_equal(new_par, parent):
            log_step(f"HAND: Converged in {i + 1} iterations.")
            break
        parent = new_par
        if (i + 1) % 5 == 0:
            log_step(f"HAND: Path-doubling iteration {i + 1}/{max_iters}...")
    else:
        log_step("HAND: WARNING — path-doubling reached max iterations. "
                 "Some headwater pixels may be unlabelled.")

    # ── Step 6: Assign smoothed profile elevations ────────────────────────
    log_step("HAND: Assigning smoothed river elevations to floodplain pixels...")

    p_safe       = np.where(parent >= 0, parent, 0).astype(np.int64)
    root_is_rv   = is_river[p_safe]
    nearest_rv   = np.where(root_is_rv, parent, -1).astype(np.int64)
    del parent
    gc.collect()

    # Map every river pixel → smoothed elevation (nearest centerline point)
    rv_idx  = np.where(is_river)[0]
    rp_rows = rv_idx // cols
    rp_cols = rv_idx % cols

    a_t, b_t, c_t = dem_transform.a, dem_transform.b, dem_transform.c
    d_t, e_t, f_t = dem_transform.d, dem_transform.e, dem_transform.f
    rp_x = c_t + a_t * rp_cols.astype(float) + b_t * rp_rows.astype(float)
    rp_y = f_t + d_t * rp_cols.astype(float) + e_t * rp_rows.astype(float)

    river_xy   = np.array([(p.x, p.y) for p in pts_gdf.geometry], dtype=np.float64)
    river_elev = pts_gdf["elevation"].values.astype(np.float32)

    kd = KDTree(river_xy)
    _, rp_match = kd.query(np.column_stack([rp_x, rp_y]), k=1)
    rv_pixel_elev = river_elev[rp_match]

    elev_lut = np.full(n, np.nan, dtype=np.float32)
    elev_lut[rv_idx] = rv_pixel_elev

    # Euclidean distance + nearest-river-pixel index for every pixel.
    # Used for (a) max_dist cutoff and (b) fallback for pixels the D8 routing
    # didn't connect to the river (DEM edges, flow exiting extent, flat areas).
    log_step("HAND: Computing Euclidean distance transform for coverage fallback...")
    edt_dist, (edt_row, edt_col) = distance_transform_edt(~is_river_2d, return_indices=True)
    edt_dist  *= pixel_size                                    # convert px → metres
    edt_rv_idx = (edt_row * cols + edt_col).ravel().astype(np.int64)   # flat index of nearest river px

    # ── Primary HAND assignment ────────────────────────────────────────────
    has_rv      = nearest_rv >= 0
    within_dist = edt_dist.ravel() <= max_dist
    nr_safe     = np.where(has_rv, nearest_rv, 0).astype(np.int64)
    base_flat   = np.where(has_rv & within_dist,
                           elev_lut[nr_safe], np.nan).astype(np.float32)

    # ── Euclidean fallback for unconnected pixels ─────────────────────────
    # Pixels where D8 routing didn't reach the river (DEM-edge effects, sinks
    # outside the river catchment, pysheds flat-resolution artefacts) are
    # assigned the elevation of their Euclidean-nearest river pixel.
    # This matches the coverage behaviour of the other engines while preserving
    # correct HAND assignments everywhere the flow routing succeeded.
    unconnected = (~has_rv) & within_dist
    n_fallback  = int(np.sum(unconnected))
    if n_fallback > 0:
        log_step(f"HAND: {n_fallback:,} pixels unconnected by flow routing — "
                 f"applying Euclidean nearest-river fallback for full coverage.")
        fallback_elev = elev_lut[edt_rv_idx[unconnected]]
        base_flat[unconnected] = np.where(
            np.isfinite(fallback_elev), fallback_elev, np.nan
        )

    hand_count     = int(np.sum(has_rv & within_dist))
    fallback_count = n_fallback
    del nearest_rv, elev_lut, edt_dist, edt_row, edt_col, edt_rv_idx
    del is_river, is_river_2d
    gc.collect()

    # ── Write memmap ───────────────────────────────────────────────────────
    base_filename = os.path.join(temp_dir, "rem_base_surface.dat")
    out = np.memmap(base_filename, dtype='float32', mode='w+', shape=(rows, cols))
    out[:, :] = base_flat.reshape(rows, cols)
    out.flush()

    log_step(
        f"HAND: Done. River pixels: {len(rv_idx):,}. "
        f"Flow-routed: {hand_count:,} px. "
        f"Euclidean fallback: {fallback_count:,} px."
    )
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
                  enforce_isotonic=True, max_search_dist=None, engine="projection", data_source="user_upload", **kwargs):

    log_step(f"--- STARTED REM CALCULATION ---")
    _ENGINE_LABELS = {
        "hand":       "HAND (Height Above Nearest Drainage)",
        "projection": "Flow-Weighted (smooth, bend-safe)",
        "idw":        "IDW / SciPy KDTree",
    }
    log_step(f"Engine: {_ENGINE_LABELS.get(engine, engine)}")
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

    # Compute initial tangents/normals for cross-section geometry (before elevation filtering)
    tangents_init, normals = _compute_tangents_normals(pts_gdf)

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

    # Recompute tangents on the FINAL filtered point set.
    # Bridge spike removal and NaN filtering can remove points, which would
    # misalign pre-computed tangents with the river_coords array passed to the
    # interpolation engine.  Recomputing here guarantees exact correspondence.
    final_tangents, _ = _compute_tangents_normals(pts_gdf)
    log_step(f"Final river profile: {len(pts_gdf)} points after all filtering.")

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

        if engine == "hand":
            base_memmap, base_file = generate_base_surface_hand(
                dem_path, dem_meta['transform'], dem_meta['shape'], pts_gdf,
                temp_dir=temp_dir, spacing=spacing,
                max_dist=max_search_dist, absolute_cutoff=absolute_cutoff,
            )
        else:
            base_memmap, base_file = generate_base_surface_memmap(
                dem_path, dem_meta['transform'], dem_meta['shape'], pts_gdf,
                tile_size=tile_size, k_neighbors=k_neighbors,
                power=idw_power, workers=threads, temp_dir=temp_dir,
                max_dist=max_search_dist, absolute_cutoff=absolute_cutoff,
                engine=engine, river_s=pts_gdf["s"].values, spacing=spacing,
                tangents=final_tangents if engine == "projection" else None,
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
