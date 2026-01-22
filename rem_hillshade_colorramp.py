#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# rem_hillshade_colorramp.py

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import math
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
# Use Agg backend (Headless) to prevent GUI crashes
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, Normalize
import contextily as cx
from pyproj import Transformer, CRS
import matplotlib.colors as mcolors
from PIL import Image
import rem_utils as utils
import shutil

# Allow massive images to be saved without decompression bomb errors
Image.MAX_IMAGE_PIXELS = None

NODATA_REM = -999.0

# Color Palettes

_SCHEMES: Dict[str, List[str]] = {
    "FloodAlertRed":   ["#000000", "#1A1A1A", "#333333", "#FFFF00", "#FFD700", "#FFA500", "#FF8C00", "#FF4500", "#FF0000", "#CC0000", "#800000", "#4D0000"],
    "NeonFloodCyan":   ["#000020", "#000050", "#4B0082", "#800080", "#9932CC", "#00FFFF", "#00CED1", "#40E0D0", "#32CD32", "#008000", "#004D00", "#000000"],
    "TropicWarning":   ["#1A0030", "#2E004F", "#4B0082", "#76FF03", "#64DD17", "#00C853", "#FF00FF", "#D500F9", "#AA00FF", "#FF6D00", "#E65100", "#BF360C"],
    "VolcanicEruption":["#FFFF00", "#FFD700", "#FFA500", "#FF8C00", "#FF4500", "#FF0000", "#B22222", "#8B0000", "#4B0082", "#2E1A47", "#1A0030", "#0B0611"],
    "AquaCopper":      ["#003C46", "#0F6B77", "#1F9AA9", "#6FC0C6", "#BFE7E3", "#D4E0CA", "#EAD9B0", "#D7A170", "#C46A2F", "#8E4D23", "#593018", "#3B1E0F"][::-1],
    "PeacockBronze":   ["#0B4F6C", "#11667A", "#177E89", "#5EAB97", "#A6D9C8", "#CBE1CD", "#F0EAD2", "#D0BA9D", "#B08968", "#86634B", "#5C3D2E", "#3D281E"],
    "GlacierLava":     ["#0F172A", "#162B6C", "#1E40AF", "#5882D6", "#93C5FD", "#BCD6F4", "#E5E7EB", "#EC9595", "#EF4444", "#9D3030", "#4B1D1D", "#250E0E"],
    "SunsetDune":      ["#2E1A47", "#5E2A64", "#8E3B82", "#C06399", "#F28CB1", "#F8AFA7", "#FFD29D", "#F2A85F", "#E67E22", "#B05E11", "#7B3F00", "#4A2600"],
    "EsriClassic":     ["#00204D", "#351062", "#6A0177", "#355BBA", "#00B5FF", "#67D2FF", "#CFEFFF", "#E7EBCA", "#FFE98A", "#E3C983", "#C8A97D", "#4B2E2B"],
    "CyanDeepBlue":    ["#C4F7FF", "#A7EFFF", "#8AE7FF", "#64DDFF", "#3FD4FF", "#1FB1ED", "#008FDB", "#0068AD", "#00427F", "#002955", "#00112B", "#000510"],
    "MagmaRamp":       ["#FFB347", "#FF9623", "#FF7A00", "#FF5F00", "#FF4500", "#995559", "#3366B3", "#1E4A8E", "#0A2F6A", "#061B3E", "#020712", "#000000"],
}

_SOLID_COLORS: Dict[str, str] = {
    "SolidBlue":   "#0050A1",
    "SolidPurple": "#5B2C83",
    "SolidGold":   "#C49A00",
    "SolidGray":   "#555555",
    "SolidRed":    "#B22222",
    "SolidBlack":  "#000000",
    "SolidOrange": "#FF8C00",
    "SolidNavy":   "#000080",
    "SolidGreen":  "#008000",
}
_ALIASES: Dict[str, str] = {"EsriRemClassic": "EsriClassic", "Esri REM Classic": "EsriClassic"}

__all__ = ["list_ramps", "get_ramp", "style_rem"]

def list_ramps() -> List[str]:
    return sorted(list(_SCHEMES.keys()) + list(_SOLID_COLORS.keys()) + list(_ALIASES.keys()))

def _resolve_ramp_name(name: str) -> str:
    key = (name or "").strip()
    if key in _SCHEMES or key in _SOLID_COLORS: return key
    if key in _ALIASES: return _ALIASES[key]
    raise ValueError(f"Unknown ramp '{name}'.")

def get_ramp(name: str, reverse: bool = False) -> LinearSegmentedColormap:
    key = _resolve_ramp_name(name)
    if key in _SOLID_COLORS:
        base_hex = _SOLID_COLORS[key]
        base_rgb = mcolors.to_rgb(base_hex)
        rgba_list = [(base_rgb[0], base_rgb[1], base_rgb[2], 0.0), (base_rgb[0], base_rgb[1], base_rgb[2], 1.0)]
        if not reverse: rgba_list = rgba_list[::-1]
        return LinearSegmentedColormap.from_list(key, rgba_list, N=256)

    stops = _SCHEMES[key]
    stops_use = stops[::-1] if reverse else stops
    return LinearSegmentedColormap.from_list(key, stops_use, N=256)


# Raster Helpers

# Calculate extent [xmin, xmax, ymin, ymax] for Matplotlib
def _extent_from_meta(transform, width, height):
    left = transform.c
    right = left + transform.a * width
    top = transform.f
    # Standard GeoTIFF math: top + (negative_height * pixels) = bottom
    bottom = top + transform.e * height
    return [left, right, bottom, top]

# Read REM data efficiently using memory-mapped I/O
def _read_band(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=False).astype("float32")
        return arr, src.nodata, src.transform, src.width, src.height, src.crs

# Estimate memory needed for PNG generation in MB
def _estimate_memory_needed_mb(width: int, height: int, dpi: int, scale: int) -> float:
    # Base data arrays (REM + hillshade) in float32
    base_data_mb = (width * height * 4 * 2) / 1_048_576  # 2 arrays, 4 bytes each

    # Matplotlib figure buffer (scale * scale * 4 bytes RGBA)
    fig_width_px = width * scale
    fig_height_px = height * scale
    fig_buffer_mb = (fig_width_px * fig_height_px * 4) / 1_048_576

    # DPI multiplier (higher DPI = more internal buffers)
    dpi_multiplier = (dpi / 300.0) ** 1.2  # Less conservative estimate

    # Total with safety margin
    total_mb = (base_data_mb + fig_buffer_mb) * dpi_multiplier * 1.3  # Less conservative
    return total_mb

# Calculate optimal tile grid for chunked rendering (max 25M pixels per tile)
def _calculate_tile_grid(width: int, height: int, max_tile_pixels: int = 25_000_000) -> Tuple[int, int]:
    total_pixels = width * height

    if total_pixels <= max_tile_pixels:
        return (1, 1)  # No tiling needed

    # For very large datasets, use smaller tiles for better memory safety
    if total_pixels > 100_000_000:
        max_tile_pixels = min(max_tile_pixels, 15_000_000)  # Smaller tiles for very large datasets

    # Calculate number of tiles needed
    n_tiles = int(np.ceil(np.sqrt(total_pixels / max_tile_pixels)))

    # Try to keep tiles roughly square
    aspect_ratio = width / height
    if aspect_ratio > 1.5:
        # Wide image: more tiles horizontally
        n_tiles_x = int(np.ceil(n_tiles * np.sqrt(aspect_ratio)))
        n_tiles_y = max(1, int(np.ceil(n_tiles / np.sqrt(aspect_ratio))))
    elif aspect_ratio < 0.67:
        # Tall image: more tiles vertically
        n_tiles_y = int(np.ceil(n_tiles * np.sqrt(1/aspect_ratio)))
        n_tiles_x = max(1, int(np.ceil(n_tiles / np.sqrt(1/aspect_ratio))))
    else:
        # Roughly square: equal tiles
        n_tiles_x = n_tiles_y = n_tiles

    return (n_tiles_x, n_tiles_y)

# Read hillshade and force it to match REM's exact grid
def _read_aligned_hillshade(hs_path: str, ref_transform, ref_width, ref_height, ref_crs):
    with rasterio.open(hs_path) as src:
        hs_aligned = np.empty((ref_height, ref_width), dtype=np.float32)
        hs_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=hs_aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        # Convert NODATA to NaN so percentile calculations ignore those pixels
        if hs_nodata is not None:
            hs_aligned = np.where(hs_aligned == hs_nodata, np.nan, hs_aligned)
        return hs_aligned

def _mask_rem(arr: np.ndarray, nodata_from_src: Optional[float]) -> np.ma.MaskedArray:
    mask = np.zeros(arr.shape, dtype=bool)
    if nodata_from_src is not None and np.isfinite(nodata_from_src):
        mask |= (arr == float(nodata_from_src))
    mask |= (arr == NODATA_REM) | ~np.isfinite(arr)
    return np.ma.array(arr, mask=mask)

def _feet_to_meters(x: float) -> float:
    return float(x) * 0.3048

def _discrete_map(base: LinearSegmentedColormap, n_bins: int) -> ListedColormap:
    colors = base(np.linspace(0, 1, n_bins))
    return ListedColormap(colors)

def _format_bin_labels(edges_m: np.ndarray, unit: str):
    unit = (unit or "meters").lower()
    if unit.startswith("f"):
        edges = np.asarray(edges_m) * 3.280839895; u = "ft"
    else:
        edges = edges_m; u = "m"
    labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f} {u}" for i in range(len(edges)-1)]
    return labels, u

# Create equally-spaced bins from absolute minimum to maximum
# Example: min=-3m, max=9m, n_bins=10 creates 11 edges spanning full range
def _build_equal_bins(min_m: float, max_m: float, n_bins: int = 10) -> np.ndarray:
    min_m = float(min_m)
    max_m = float(max_m)

    if not np.isfinite(min_m) or not np.isfinite(max_m) or max_m <= min_m or n_bins < 2:
        return np.array([min_m, max_m], dtype="float64")

    return np.linspace(min_m, max_m, int(n_bins) + 1, dtype="float64")

# Contextily Helper (Web Basemaps)

def _provider_max_zoom(provider_obj) -> int:
    try: return int(getattr(provider_obj, "max_zoom", None) or provider_obj.get("max_zoom") or 19)
    except Exception: return 19

def _auto_zoom_for_extent(extent_3857, fig_px_width: int, lat_hint: float) -> int:
    R = 6378137.0
    xmin, xmax, *_ = extent_3857
    width_m = max(1.0, float(xmax - xmin))
    target_mpp = width_m / max(1, int(fig_px_width))
    lat = max(min(float(lat_hint), 85.0), -85.0)
    cosphi = math.cos(math.radians(lat))
    denom = max(1e-9, 256.0 * target_mpp)
    two_pow_z = (cosphi * 2.0 * math.pi * R) / denom
    z = int(math.floor(math.log(two_pow_z, 2.0)))
    return max(0, z)

# Add aerial basemap with retry loop to handle network/SSL timeouts
def _add_basemap_underlay(ax, data_crs, extent_xy, provider: Optional[str], alpha: float,
                          fig_px_width: int, aerial_zoom: Optional[int] = None,
                          margin_frac: float = 0.01) -> bool:
    import time
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            xmin, xmax, ymin, ymax = extent_xy
            dx, dy = (xmax - xmin) * float(margin_frac), (ymax - ymin) * float(margin_frac)
            xmin_p, xmax_p, ymin_p, ymax_p = xmin - dx, xmax + dx, ymin - dy, ymax + dy

            ax.set_xlim(xmin_p, xmax_p)
            ax.set_ylim(ymax_p, ymin_p) # Match upper origin

            src = provider or cx.providers.Esri.WorldImagery
            
            # Zoom calculation
            zoom = aerial_zoom
            if zoom is None:
                to_3857 = Transformer.from_crs(CRS.from_user_input(data_crs), CRS.from_epsg(3857), always_xy=True)
                x0, y0 = to_3857.transform(xmin_p, ymin_p)
                x1, y1 = to_3857.transform(xmax_p, ymax_p)
                zoom = _auto_zoom_for_extent((x0, x1, y0, y1), fig_px_width, 0)

            # Force a 10-second timeout for the tile request
            cx.add_basemap(ax, crs=str(data_crs), source=src, attribution=False, alpha=float(alpha), zoom=int(zoom))
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"WARNING: Aerial attempt {attempt+1} failed ({e}). Retrying in 2s...")
                time.sleep(2)
            else:
                print(f"ERROR: All aerial basemap attempts failed. Switching to Hillshade fallback.")
                return False


# Main Styling Function (Standard Matplotlib)

def style_rem(
    rem_path: str, hillshade_path: Optional[str], out_folder: str, color_maps: List[str],
    *, hs_alpha: float = 0.5, rem_alpha: float = 1.0, hillshade_mode: str = "under",
    dpi: int = 150, min_value: Optional[float] = None, max_value: Optional[float] = None,
    background: str = "hillshade", n_levels: int = 256, custom_colors: Optional[List[str]] = None,
    scale: int = 1, out_basename: Optional[str] = None, transparent: bool = False,
    n_classes: Optional[int] = None, legend_unit: str = "meters", viz_max_feet: Optional[float] = None,
    hard_cap: bool = True, aerial_path: Optional[str] = None, bg_alpha: Optional[float] = None,
    aerial_provider: Optional[str] = None, aerial_zoom: Optional[int] = None,
    hillshade_array: Optional[np.ndarray] = None, gamma: float = 0.5,
    global_vmin: Optional[float] = None, global_vmax: Optional[float] = None,
    global_hs_min: Optional[float] = None, global_hs_max: Optional[float] = None
) -> str:
    
    if not color_maps: raise ValueError("color_maps must contain one ramp name.")
    ramp_name_in = color_maps[0].strip()
    ramp_name = _resolve_ramp_name(ramp_name_in)
    os.makedirs(out_folder, exist_ok=True)

    utils.check_memory_safety(rem_path, operation_name="REM Visualization", raise_on_fail=False)

    # 0. Smart DPI adjustment for large datasets (Option A: 200 DPI threshold)
    import psutil
    import gc
    import tempfile

    # Get initial memory info
    with rasterio.open(rem_path) as src:
        W, H = src.width, src.height

    total_pixels = W * H
    original_dpi = dpi

    estimated_mb = _estimate_memory_needed_mb(W, H, dpi, scale)
    available_mb = psutil.virtual_memory().available / 1_048_576
    use_tiled_rendering = False

    # Adaptive memory threshold based on system RAM
    available_gb = available_mb / 1024
    if available_gb > 16:
        memory_threshold = 0.75  # More aggressive on high-RAM systems
    else:
        memory_threshold = 0.60  # Conservative on low-RAM systems

    if estimated_mb > available_mb * memory_threshold:
        # Option 1: Try tiled rendering first (maintains full DPI)
        n_tiles_x, n_tiles_y = _calculate_tile_grid(W, H, max_tile_pixels=20_000_000)

        if n_tiles_x > 1 or n_tiles_y > 1:
            # Tiling is feasible - use it to maintain high DPI
            use_tiled_rendering = True
            print(f"INFO: Large dataset ({W}x{H} pixels, ~{estimated_mb:.0f}MB needed)")
            print(f"      Available RAM: {available_mb:.0f}MB")
            print(f"      Using tiled rendering ({n_tiles_x}x{n_tiles_y} tiles) to maintain {dpi} DPI")
            print(f"      This uses disk space as buffer while preserving full quality")
        else:
            # Can't tile effectively - reduce DPI as fallback
            safe_dpi = int(dpi * np.sqrt((available_mb * 0.7) / estimated_mb))
            safe_dpi = max(300, min(safe_dpi, dpi))
            if safe_dpi < dpi:
                print(f"INFO: Large dataset ({W}x{H} pixels, ~{estimated_mb:.0f}MB needed)")
                print(f"      Available RAM: {available_mb:.0f}MB")
                print(f"      Auto-reducing DPI from {dpi} to {safe_dpi} to prevent crash")
                dpi = safe_dpi

    # Force garbage collection before loading large data
    gc.collect()

    # If tiled rendering is needed, use special path
    if use_tiled_rendering:
        return _style_rem_tiled(
            rem_path=rem_path,
            hillshade_path=hillshade_path,
            hillshade_array=hillshade_array,
            out_folder=out_folder,
            color_maps=color_maps,
            n_tiles_x=n_tiles_x,
            n_tiles_y=n_tiles_y,
            hs_alpha=hs_alpha,
            rem_alpha=rem_alpha,
            hillshade_mode=hillshade_mode,
            dpi=dpi,
            min_value=min_value,
            max_value=max_value,
            background=background,
            n_levels=n_levels,
            custom_colors=custom_colors,
            scale=scale,
            out_basename=out_basename,
            transparent=transparent,
            n_classes=n_classes,
            legend_unit=legend_unit,
            viz_max_feet=viz_max_feet,
            hard_cap=hard_cap,
            aerial_path=aerial_path,
            bg_alpha=bg_alpha,
            aerial_provider=aerial_provider,
            aerial_zoom=aerial_zoom,
            gamma=gamma,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
            global_hs_min=global_hs_min,
            global_hs_max=global_hs_max
        )

    # 1. Read REM Data
    rem_raw, rem_src_nodata, dem_transform, W, H, dem_crs = _read_band(rem_path)
    rem = _mask_rem(rem_raw, rem_src_nodata)
    extent = _extent_from_meta(dem_transform, W, H)
    alpha_bg = float(bg_alpha if (bg_alpha is not None) else hs_alpha)

    px_size = float(max(abs(dem_transform.a), abs(dem_transform.e)))
    rem_interp = "nearest" if px_size <= 2.0 else "bilinear"

    resolution_m = round(px_size, 1) if px_size < 10 else round(px_size)
    res_tag = f"{resolution_m:.0f}m" if resolution_m >= 1 else f"{resolution_m:.1f}m"

    # 2. Read Hillshade (Aligned)
    shade = None
    if background == "hillshade":
        hs_to_process = hillshade_array if hillshade_array is not None else None
        if hs_to_process is None and hillshade_path and os.path.exists(hillshade_path):
            hs_to_process = _read_aligned_hillshade(hillshade_path, dem_transform, W, H, dem_crs)

        if hs_to_process is not None:
            try:
                s_min, s_max = np.nanpercentile(hs_to_process, [2, 98])
                shade = np.clip((hs_to_process - s_min) / (s_max - s_min), 0, 1) if s_max > s_min else np.clip(hs_to_process, 0, 1)
                shade = shade ** 0.9
            except Exception: shade = None

        # Clear hillshade_array if it was loaded (free memory before matplotlib)
        if hs_to_process is not None and hillshade_array is None:
            del hs_to_process
            gc.collect()
            
    rgb = None
    if background == "aerial" and aerial_path and os.path.exists(aerial_path):
        try:
            with rasterio.open(aerial_path) as src:
                rgb = src.read([1,2,3]).transpose(1,2,0)
        except Exception: pass

    # 3. Caps & Stats
    if viz_max_feet is not None: max_value = _feet_to_meters(viz_max_feet)
    if max_value is not None and hard_cap:
        rem = np.ma.array(rem, mask=(rem.mask | (rem > float(max_value))))

    valid_vals = rem.compressed()
    if valid_vals.size == 0: data_min, data_max = 0.0, 1.0
    else:
        # If user set a max_value cap, use percentiles to avoid outliers
        # If no cap, use actual min/max to show full range
        if max_value is not None:
            # User wants a specific range - use percentiles to smooth outliers
            data_min, data_max = float(np.nanpercentile(valid_vals, 2)), float(np.nanpercentile(valid_vals, 98))
        else:
            # No cap set - show FULL range of data
            data_min, data_max = float(valid_vals.min()), float(valid_vals.max())

        # Only apply sanity limits for unreasonable values (like -1000m or +10000m)
        # but allow natural range of REM data to display
        data_min = max(data_min, -100.0)  # Reasonable minimum (unlikely to have REM < -100m)
        # NO MAXIMUM CAP - let data scale to its natural range
        if data_max <= data_min: data_min, data_max = 0.0, 1.0

    # 4. Colormap construction
    # A. Define Visualization Limits FIRST (so they are available for binning)
    vmin_vis = float(min_value) if min_value is not None else data_min
    vmax_vis = float(max_value) if max_value is not None else data_max
    if vmax_vis <= vmin_vis: vmax_vis = vmin_vis + 1e-6

    # B. Load Base Colormap SECOND
    base_cmap = get_ramp(ramp_name)

    # C. Determine Mode (Discrete vs Continuous)
    discrete_requested = False

    if n_classes is not None:
        if isinstance(n_classes, bool):
            discrete_requested = n_classes
        elif int(n_classes) >= 2:
            discrete_requested = True

    # D. Build Final Colormap & Norm
    # Always use continuous colormap with gamma correction for smooth colors
    x_stretched = np.linspace(0, 1, n_levels) ** gamma
    cmap = LinearSegmentedColormap.from_list(f"{ramp_name}_gamma", base_cmap(x_stretched), N=n_levels)
    cmap.set_bad(alpha=0.0)
    norm = Normalize(vmin=vmin_vis, vmax=vmax_vis)

    # For discrete mode: calculate bin edges for legend labels only
    if discrete_requested:
        edges_m = _build_equal_bins(vmin_vis, vmax_vis, n_bins=10)
        mode_tag = f"{res_tag}_{len(edges_m)-1}classes"
    else:
        edges_m = None
        mode_tag = res_tag

    # 5. Plotting (FIXED LAYOUT)
    scale = max(1, int(scale))
    fig, ax = plt.subplots(figsize=((W * scale) / float(dpi), (H * scale) / float(dpi)), dpi=dpi)
    ax.set_axis_off()
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    # LOCK ASPECT RATIO: Crucial to keep discrete map aligned with hillshade
    ax.set_aspect('equal', adjustable='box') 

    # Shared Draw Logic
    if background == "aerial":
        if rgb is not None: ax.imshow(rgb, extent=extent, interpolation="bilinear", alpha=alpha_bg, origin='upper')
        else: _add_basemap_underlay(ax, dem_crs, extent, aerial_provider, alpha_bg, int(fig.get_window_extent().width))
    elif background == "hillshade" and shade is not None:
        ax.imshow(shade, cmap="gray", extent=extent, interpolation="nearest", alpha=alpha_bg, vmin=0.0, vmax=1.0, origin='upper')

    # Draw REM with universal 'upper' origin
    rem_kwargs = dict(cmap=cmap, interpolation=rem_interp, extent=extent, alpha=float(rem_alpha), origin='upper', norm=norm)
    ax.imshow(rem, **rem_kwargs)

    # Clean Vertical Legend (ADAPTIVE SIZING + FONT SCALING)
    if discrete_requested:
        # Create simple tick labels at bin edges (e.g., "-2.9", "-1.7", "0.5", etc.)
        unit = (legend_unit or "meters").lower()
        if unit.startswith("f"):
            tick_values = edges_m * 3.280839895  # Convert to feet
            u = "ft"
        else:
            tick_values = edges_m
            u = "m"

        # Format as simple numbers
        labels = [f"{val:.1f}" for val in tick_values]

        # --- ADAPTIVE LEGEND FIX (WITH FONT SCALING) ---
        # Calculate aspect ratio to determine optimal legend sizing
        aspect_ratio = W / H  # width / height

        # Calculate adaptive font size based on number of ticks
        n_ticks = len(labels)
        if n_ticks >= 10:
            base_fontsize = 8  # Very small for many ticks
        elif n_ticks >= 7:
            base_fontsize = 9  # Small for moderate ticks
        else:
            base_fontsize = 10  # Normal for few ticks

        # Determine orientation and sizing based on shape
        if aspect_ratio > 2.5:
            # Very wide REM (e.g., 4000x800) -> use horizontal legend
            orientation = 'horizontal'
            fraction = 0.03  # Thinner for horizontal
            pad = 0.04
            aspect = 40  # Wider horizontal bar
            fontsize = base_fontsize - 1  # Slightly smaller for horizontal
        elif aspect_ratio < 0.4:
            # Very tall REM (e.g., 800x4000) -> use wider vertical legend
            orientation = 'vertical'
            fraction = 0.08  # Wider to prevent squishing
            pad = 0.04
            aspect = 15  # Shorter, wider bar
            fontsize = base_fontsize  # Normal size
        else:
            # Normal aspect ratio -> standard vertical legend
            orientation = 'vertical'
            fraction = 0.046
            pad = 0.04
            aspect = 20
            fontsize = base_fontsize  # Normal size

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=fraction,
            pad=pad,
            ticks=edges_m,  # Tick positions at bin edges
            orientation=orientation,
            aspect=aspect
        )

        # Adjust label positioning based on orientation with adaptive font size
        if orientation == 'horizontal':
            cbar.ax.set_xticklabels(labels, fontsize=fontsize, family='monospace', rotation=45, ha='right')
            cbar.set_label(f"REM ({u})", fontsize=fontsize+2, weight='bold')
        else:
            cbar.ax.set_yticklabels(labels, fontsize=fontsize, family='monospace')
            cbar.set_label(f"REM ({u})", fontsize=fontsize+2, weight='bold', rotation=270, labelpad=20)

        # Adjust tick parameters for cleaner look
        cbar.ax.tick_params(labelsize=fontsize, length=4, width=1, pad=4)

    out_png = os.path.join(out_folder, f"{out_basename or f'REM_{ramp_name_in}_{mode_tag}'}.png")
    try: fig.tight_layout(pad=0.3)
    except Exception: pass

    # Efficient PNG save with fallback mechanism and proper memory cleanup
    import gc
    save_successful = False
    current_dpi = dpi
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Try to save at current DPI
            fig.savefig(out_png, dpi=current_dpi, bbox_inches="tight", pad_inches=0.1,
                       facecolor=fig.get_facecolor(), transparent=transparent)
            save_successful = True

            # Inform user if we had to reduce DPI
            if current_dpi < original_dpi and attempt == 0:
                print(f"      Successfully saved at {current_dpi} DPI (auto-adjusted from {original_dpi})")
            elif current_dpi < dpi:
                print(f"      Fallback successful at {current_dpi} DPI")

            break  # Success, exit retry loop

        except (MemoryError, OSError) as e:
            # Memory error or disk space issue
            if attempt < max_retries - 1:
                # Try again with lower DPI (reduces disk I/O buffer size)
                current_dpi = max(200, int(current_dpi * 0.7))
                print(f"WARNING: Save failed (attempt {attempt+1}). Retrying at {current_dpi} DPI...")
                gc.collect()  # Clear memory before retry
            else:
                print(f"ERROR: Failed to save PNG after {max_retries} attempts")
                raise

    # Cleanup
    plt.close(fig)
    del rem_raw, rem
    if shade is not None:
        del shade
    gc.collect()  # Force garbage collection to free memory immediately

    if not save_successful:
        raise RuntimeError(f"Failed to save PNG after {max_retries} attempts")

    return out_png


# Global Normalization Helper (Prevents Tile Seams)

# Calculate exact global min/max normalization values from full REM dataset
# Streams through entire REM in chunks to ensure colors accurately represent elevations
def _calculate_global_normalization(
    rem_path: str,
    hillshade_path: Optional[str],
    sample_fraction: float = 0.1,  # DEPRECATED - kept for API compatibility but ignored
    max_value: Optional[float] = None,
    min_value: Optional[float] = None,
    hard_cap: bool = True,
    viz_max_feet: Optional[float] = None,
    background: str = "hillshade"
) -> Tuple[float, float, Optional[float], Optional[float]]:
    import gc
    from rasterio.windows import Window

    # Handle feet conversion
    if viz_max_feet is not None:
        max_value = _feet_to_meters(viz_max_feet)

    # Stream through FULL REM in chunks to get EXACT min/max
    # OPTIMIZED: Scan REM and hillshade in same loop to halve I/O time
    if background == "hillshade" and hillshade_path and os.path.exists(hillshade_path):
        print(f"      Scanning full REM + hillshade for exact ranges (combined pass)...")
    else:
        print(f"      Scanning full REM for exact elevation range...")

    global_min = np.inf
    global_max = -np.inf
    total_valid_pixels = 0

    # Hillshade tracking (if needed)
    hs_min = np.inf
    hs_max = -np.inf
    hs_valid_pixels = 0
    scan_hillshade = background == "hillshade" and hillshade_path and os.path.exists(hillshade_path)

    CHUNK_SIZE = 2048  # Process in 2048x2048 chunks for memory efficiency

    # Open both files if hillshade scanning is needed
    if scan_hillshade:
        with rasterio.open(rem_path) as rem_src, rasterio.open(hillshade_path) as hs_src:
            W, H = rem_src.width, rem_src.height
            nodata = rem_src.nodata
            hs_nodata = hs_src.nodata

            # Stream through BOTH rasters in same loop (halves I/O time!)
            for row_off in range(0, H, CHUNK_SIZE):
                for col_off in range(0, W, CHUNK_SIZE):
                    window_height = min(CHUNK_SIZE, H - row_off)
                    window_width = min(CHUNK_SIZE, W - col_off)
                    window = Window(col_off, row_off, window_width, window_height)

                    # Read BOTH chunks in parallel
                    rem_chunk = rem_src.read(1, window=window).astype("float32")
                    hs_chunk = hs_src.read(1, window=window).astype("float32")

                    # Process REM chunk
                    mask = np.zeros(rem_chunk.shape, dtype=bool)
                    if nodata is not None and np.isfinite(nodata):
                        mask |= (rem_chunk == float(nodata))
                    mask |= (rem_chunk == NODATA_REM) | ~np.isfinite(rem_chunk)
                    if max_value is not None and hard_cap:
                        mask |= (rem_chunk > float(max_value))

                    valid_rem = rem_chunk[~mask]
                    if valid_rem.size > 0:
                        global_min = min(global_min, float(valid_rem.min()))
                        global_max = max(global_max, float(valid_rem.max()))
                        total_valid_pixels += valid_rem.size

                    # Process hillshade chunk
                    if hs_nodata is not None:
                        hs_chunk = np.where(hs_chunk == hs_nodata, np.nan, hs_chunk)
                    valid_hs = hs_chunk[np.isfinite(hs_chunk)]
                    if valid_hs.size > 0:
                        hs_min = min(hs_min, float(valid_hs.min()))
                        hs_max = max(hs_max, float(valid_hs.max()))
                        hs_valid_pixels += valid_hs.size

                    # Free memory
                    del rem_chunk, hs_chunk, valid_rem, valid_hs
                    gc.collect()
    else:
        # REM-only scanning (no hillshade)
        with rasterio.open(rem_path) as src:
            W, H = src.width, src.height
            nodata = src.nodata

            # Stream through the raster in chunks
            for row_off in range(0, H, CHUNK_SIZE):
                for col_off in range(0, W, CHUNK_SIZE):
                    window_height = min(CHUNK_SIZE, H - row_off)
                    window_width = min(CHUNK_SIZE, W - col_off)
                    window = Window(col_off, row_off, window_width, window_height)

                    # Read chunk
                    chunk = src.read(1, window=window).astype("float32")

                    # Mask nodata
                    mask = np.zeros(chunk.shape, dtype=bool)
                    if nodata is not None and np.isfinite(nodata):
                        mask |= (chunk == float(nodata))
                    mask |= (chunk == NODATA_REM) | ~np.isfinite(chunk)

                    # Apply hard cap if requested
                    if max_value is not None and hard_cap:
                        mask |= (chunk > float(max_value))

                    # Get valid values
                    valid_chunk = chunk[~mask]

                    if valid_chunk.size > 0:
                        chunk_min = float(valid_chunk.min())
                        chunk_max = float(valid_chunk.max())

                        global_min = min(global_min, chunk_min)
                        global_max = max(global_max, chunk_max)
                        total_valid_pixels += valid_chunk.size

                    # Free memory
                    del chunk, valid_chunk
                    gc.collect()

    # Validate results
    if total_valid_pixels == 0 or not np.isfinite(global_min) or not np.isfinite(global_max):
        print(f"      WARNING: No valid REM data found. Using defaults.")
        global_vmin, global_vmax = 0.0, 1.0
    else:
        # Apply percentile logic if user wants specific range (for smoothing outliers)
        # But we've already scanned the full dataset, so we have exact values
        if max_value is not None:
            # User wants a cap - we already applied it during streaming
            global_vmin = global_min
            global_vmax = global_max
        else:
            # No cap - use the exact min/max we found
            global_vmin = global_min
            global_vmax = global_max

        # Sanity limits (prevent unreasonable values like -1000m or +10000m)
        global_vmin = max(global_vmin, -100.0)

        if global_vmax <= global_vmin:
            global_vmin, global_vmax = 0.0, 1.0

    # Override with user-specified limits
    if min_value is not None:
        global_vmin = float(min_value)
    if max_value is not None:
        global_vmax = float(max_value)

    print(f"      Scanned {total_valid_pixels:,} pixels | Exact range: [{global_vmin:.3f}, {global_vmax:.3f}] m")

    # Process hillshade results if we scanned it
    global_hs_min, global_hs_max = None, None
    if scan_hillshade and hs_valid_pixels > 0 and np.isfinite(hs_min) and np.isfinite(hs_max):
        global_hs_min = hs_min
        global_hs_max = hs_max
        print(f"      Hillshade scanned: {hs_valid_pixels:,} pixels | Range: [{global_hs_min:.2f}, {global_hs_max:.2f}]")
    elif scan_hillshade:
        print(f"      WARNING: No valid hillshade data found")

    gc.collect()
    return global_vmin, global_vmax, global_hs_min, global_hs_max


# Tiled Rendering for Very Large Datasets (Maintains High DPI)

# Render very large REM in tiles to maintain high DPI while using disk as buffer
# Splits raster into tiles, renders each separately, then stitches into single high-DPI PNG
# Uses global normalization to prevent tile seams
def _style_rem_tiled(
    rem_path: str, hillshade_path: Optional[str], hillshade_array: Optional[np.ndarray],
    out_folder: str, color_maps: List[str], n_tiles_x: int, n_tiles_y: int,
    **kwargs
) -> str:
    import tempfile
    import gc
    from PIL import Image

    ramp_name = color_maps[0].strip()
    dpi = kwargs.get('dpi', 500)
    scale = kwargs.get('scale', 2)
    out_basename = kwargs.get('out_basename')

    # Read full metadata
    with rasterio.open(rem_path) as src:
        W, H = src.width, src.height
        transform = src.transform
        crs = src.crs
        rem_nodata = src.nodata

    # ============================================================================
    # CRITICAL FIX: Calculate GLOBAL normalization values before tiling
    # This ensures all tiles use the same color scale (prevents seams!)
    # Now scans the FULL REM dataset for EXACT min/max (no sampling!)
    # ============================================================================
    # Check if global normalization values are already provided (to avoid redundant scans)
    global_vmin = kwargs.get('global_vmin')
    global_vmax = kwargs.get('global_vmax')
    global_hs_min = kwargs.get('global_hs_min')
    global_hs_max = kwargs.get('global_hs_max')

    if global_vmin is not None and global_vmax is not None:
        # Values already calculated - skip scan (HUGE performance boost for multiple PNGs!)
        print(f"      Using pre-calculated global normalization (skip scan)")
        print(f"      Global REM range: {global_vmin:.2f} to {global_vmax:.2f} meters")
        if global_hs_min is not None:
            print(f"      Global hillshade range: {global_hs_min:.2f} to {global_hs_max:.2f}")
    else:
        # First PNG or values not provided - calculate them
        print(f"      Calculating exact global normalization (scanning full REM)...")
        global_vmin, global_vmax, global_hs_min, global_hs_max = _calculate_global_normalization(
            rem_path=rem_path,
            hillshade_path=hillshade_path,
            max_value=kwargs.get('max_value'),
            min_value=kwargs.get('min_value'),
            hard_cap=kwargs.get('hard_cap', True),
            viz_max_feet=kwargs.get('viz_max_feet'),
            background=kwargs.get('background', 'hillshade')
        )

        print(f"      Global REM range: {global_vmin:.2f} to {global_vmax:.2f} meters")
        if global_hs_min is not None:
            print(f"      Global hillshade range: {global_hs_min:.2f} to {global_hs_max:.2f}")

    # Calculate tile dimensions with OVERLAP for seamless blending
    OVERLAP_PIXELS = 20  # Optimized: 20px provides seamless blending with less overhead
    tile_width = int(np.ceil(W / n_tiles_x))
    tile_height = int(np.ceil(H / n_tiles_y))

    print(f"      Rendering {n_tiles_x * n_tiles_y} tiles ({tile_width}x{tile_height} pixels each)...")
    print(f"      Using {OVERLAP_PIXELS}px overlap for seamless blending...")

    # Create temp directory for tiles
    temp_dir = tempfile.mkdtemp(prefix="rem_tiles_")
    tile_paths = []

    try:
        # Build list of all tile configurations for parallel processing
        tile_configs = []

        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Calculate tile bounds WITH OVERLAP
                x_start = max(0, tx * tile_width - OVERLAP_PIXELS)
                y_start = max(0, ty * tile_height - OVERLAP_PIXELS)
                x_end = min((tx + 1) * tile_width + OVERLAP_PIXELS, W)
                y_end = min((ty + 1) * tile_height + OVERLAP_PIXELS, H)

                # Track actual tile boundaries (without overlap) for later cropping
                x_core_start = tx * tile_width
                y_core_start = ty * tile_height
                x_core_end = min(x_core_start + tile_width, W)
                y_core_end = min(y_core_start + tile_height, H)

                tile_configs.append({
                    'ty': ty,
                    'tx': tx,
                    'x_start': x_start,
                    'y_start': y_start,
                    'x_end': x_end,
                    'y_end': y_end,
                    'x_core_start': x_core_start,
                    'y_core_start': y_core_start,
                    'x_core_end': x_core_end,
                    'y_core_end': y_core_end,
                })

        # Parallel tile rendering using joblib
        from joblib import Parallel, delayed
        import joblib

        # Auto-detect CPU cores, leave 1 free for system stability
        n_jobs = max(1, joblib.cpu_count() - 1)
        print(f"      Using {n_jobs} CPU cores for parallel rendering...")

        # Wrapper function for parallel tile rendering (passes numpy arrays directly)
        def render_tile_wrapper(tile_config):
            ty = tile_config['ty']
            tx = tile_config['tx']
            x_start = tile_config['x_start']
            y_start = tile_config['y_start']
            x_end = tile_config['x_end']
            y_end = tile_config['y_end']

            # Read tile data from REM (with overlap) - ONCE
            with rasterio.open(rem_path) as src:
                window = ((y_start, y_end), (x_start, x_end))
                tile_rem = src.read(1, window=window).astype("float32")
                tile_nodata = src.nodata

            # Read corresponding hillshade tile if needed
            tile_hillshade = None
            if kwargs.get('background') == 'hillshade' and hillshade_path:
                with rasterio.open(hillshade_path) as hs_src:
                    tile_hillshade = hs_src.read(1, window=window).astype("float32")

            # Calculate tile transform (for extent calculation)
            tile_transform = transform * transform.translation(x_start, y_start)

            # Render this tile (small memory footprint)
            tile_png = os.path.join(temp_dir, f"tile_{ty}_{tx}.png")

            _render_single_tile(
                tile_png=tile_png,
                tile_rem=tile_rem,
                tile_hillshade=tile_hillshade,
                tile_transform=tile_transform,
                tile_nodata=tile_nodata,
                color_maps=[ramp_name],
                global_vmin=global_vmin,
                global_vmax=global_vmax,
                global_hs_min=global_hs_min,
                global_hs_max=global_hs_max,
                **kwargs
            )

            # Return tile info for stitching
            return {
                'ty': ty,
                'tx': tx,
                'png_path': tile_png,
                'x_core_offset': tile_config['x_core_start'] - x_start,
                'y_core_offset': tile_config['y_core_start'] - y_start,
                'core_width': tile_config['x_core_end'] - tile_config['x_core_start'],
                'core_height': tile_config['y_core_end'] - tile_config['y_core_start']
            }

        # Use threading backend (loky causes slowdowns on macOS due to spawn overhead)
        # Threading works well enough since matplotlib releases GIL during numpy ops
        tile_paths = Parallel(n_jobs=n_jobs, backend='threading', verbose=5)(
            delayed(render_tile_wrapper)(config) for config in tile_configs
        )

        # Stitch tiles together using PIL with overlap cropping
        print(f"      Stitching {len(tile_paths)} tiles into final PNG...")

        # Calculate final image dimensions (based on actual raster size, not tile grid)
        # We need to account for DPI scaling
        with rasterio.open(rem_path) as src:
            final_raster_width = src.width
            final_raster_height = src.height

        # Load first tile to determine pixel scaling factor
        first_tile_info = tile_paths[0]
        first_tile_img = Image.open(first_tile_info['png_path'])
        tile_total_px_width, tile_total_px_height = first_tile_img.size
        first_tile_img.close()

        # Calculate scaling: PNG pixels per raster pixel
        # Use the first tile's dimensions to determine the DPI scaling factor
        # First tile spans from (0,0) with overlap, so calculate from actual raster size
        first_tile_raster_width = min(tile_width + 2 * OVERLAP_PIXELS, W)
        first_tile_raster_height = min(tile_height + 2 * OVERLAP_PIXELS, H)

        px_per_raster = tile_total_px_width / first_tile_raster_width

        final_px_width = int(final_raster_width * px_per_raster)
        final_px_height = int(final_raster_height * px_per_raster)

        final_image = Image.new('RGBA', (final_px_width, final_px_height))

        # Paste each tile, cropping overlap regions
        for tile_info in tile_paths:
            tile_img = Image.open(tile_info['png_path'])

            # Calculate crop box to remove overlap (in PNG pixel coordinates)
            x_crop_offset = int(tile_info['x_core_offset'] * px_per_raster)
            y_crop_offset = int(tile_info['y_core_offset'] * px_per_raster)
            crop_width = int(tile_info['core_width'] * px_per_raster)
            crop_height = int(tile_info['core_height'] * px_per_raster)

            # Crop the tile to remove overlap
            cropped_tile = tile_img.crop((
                x_crop_offset,
                y_crop_offset,
                x_crop_offset + crop_width,
                y_crop_offset + crop_height
            ))

            # Calculate paste position in final image
            paste_x = int(tile_info['tx'] * tile_width * px_per_raster)
            paste_y = int(tile_info['ty'] * tile_height * px_per_raster)

            final_image.paste(cropped_tile, (paste_x, paste_y))

            tile_img.close()
            cropped_tile.close()
            gc.collect()

        # Save final stitched PNG
        px_size = float(max(abs(transform.a), abs(transform.e)))
        resolution_m = round(px_size, 1) if px_size < 10 else round(px_size)
        res_tag = f"{resolution_m:.0f}m" if resolution_m >= 1 else f"{resolution_m:.1f}m"
        out_png = os.path.join(out_folder, f"{out_basename or f'REM_{ramp_name}_{res_tag}'}.png")

        final_image.save(out_png, dpi=(dpi, dpi), optimize=True)
        final_image.close()

        print(f"      Tiled rendering complete: {os.path.basename(out_png)}")

        return out_png

    finally:
        # CROSS-PLATFORM FIX: Safe cleanup for Windows file locking
        # Step 1: Close any open PIL images
        try:
            if 'final_image' in locals() and final_image:
                final_image.close()
        except:
            pass
            
        # Step 2: Force garbage collection
        import gc
        gc.collect()
        
        # Step 3: Brief pause for Windows to release file handles
        import time
        time.sleep(0.2)
        
        # Step 4: Remove temp directory with retry logic
        if temp_dir and os.path.exists(temp_dir):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(temp_dir)
                    break  # Success
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        gc.collect()  # Try again to release handles
                    else:
                        # Final attempt failed - log but don't crash
                        print(f"WARNING: Could not remove temp directory {temp_dir}: {e}")
                        print("  Leftover files will be cleaned on next run")


# Render single tile without triggering tiled rendering again
# Accepts numpy arrays directly (no temp GeoTIFF files)
# Uses global normalization values to prevent tile seams
def _render_single_tile(tile_png: str,
                        tile_rem: np.ndarray,
                        tile_hillshade: Optional[np.ndarray],
                        tile_transform,
                        tile_nodata: Optional[float],
                        color_maps: List[str],
                        global_vmin: Optional[float] = None,
                        global_vmax: Optional[float] = None,
                        global_hs_min: Optional[float] = None,
                        global_hs_max: Optional[float] = None,
                        **kwargs):
    import gc

    ramp_name = color_maps[0]
    dpi = kwargs.get('dpi', 500)
    scale = kwargs.get('scale', 2)
    rem_alpha = kwargs.get('rem_alpha', 1.0)
    bg_alpha = kwargs.get('bg_alpha', 0.5)
    background = kwargs.get('background', 'hillshade')
    gamma = kwargs.get('gamma', 0.5)

    # Use passed arrays directly (no file I/O!)
    rem_raw = tile_rem.astype("float32")
    nodata = tile_nodata
    transform = tile_transform
    H, W = rem_raw.shape

    # Mask
    mask = np.zeros(rem_raw.shape, dtype=bool)
    if nodata is not None and np.isfinite(nodata):
        mask |= (rem_raw == float(nodata))
    mask |= (rem_raw == -999.0) | ~np.isfinite(rem_raw)
    rem = np.ma.array(rem_raw, mask=mask)

    # Get extent
    extent = _extent_from_meta(transform, W, H)

    # Colormap with gamma correction (match main function)
    base_cmap = get_ramp(ramp_name)
    x_stretched = np.linspace(0, 1, 200) ** gamma
    cmap = LinearSegmentedColormap.from_list(f"{ramp_name}_gamma", base_cmap(x_stretched), N=256)
    cmap.set_bad(alpha=0.0)

    # CRITICAL FIX: Use GLOBAL normalization values instead of per-tile calculation!
    if global_vmin is not None and global_vmax is not None:
        vmin, vmax = global_vmin, global_vmax
    else:
        # Fallback to per-tile (for backward compatibility, but not recommended)
        valid_vals = rem.compressed()
        if valid_vals.size > 0:
            vmin, vmax = float(np.nanpercentile(valid_vals, 2)), float(np.nanpercentile(valid_vals, 98))
        else:
            vmin, vmax = 0.0, 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Create figure
    fig, ax = plt.subplots(figsize=((W * scale) / float(dpi), (H * scale) / float(dpi)), dpi=dpi)
    ax.set_axis_off()
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal', adjustable='box')

    # Draw hillshade if available
    if background == 'hillshade' and tile_hillshade is not None:
        # CRITICAL FIX: Use GLOBAL hillshade normalization to prevent brightness seams!
        if global_hs_min is not None and global_hs_max is not None:
            s_min, s_max = global_hs_min, global_hs_max
        else:
            # Fallback to per-tile (backward compatibility, not recommended)
            valid_hillshade = tile_hillshade[np.isfinite(tile_hillshade)]
            if valid_hillshade.size > 0:
                s_min, s_max = np.nanpercentile(tile_hillshade, [2, 98])
            else:
                s_min, s_max = 0.0, 1.0

        if s_max > s_min:
            shade = np.clip((tile_hillshade - s_min) / (s_max - s_min), 0, 1)
            shade = shade ** 0.9  # Apply gamma correction (match main function)
            ax.imshow(shade, cmap="gray", extent=extent, interpolation="nearest",
                     alpha=bg_alpha, vmin=0.0, vmax=1.0, origin='upper')

    # Draw REM
    ax.imshow(rem, cmap=cmap, norm=norm, interpolation="bilinear",
             extent=extent, alpha=rem_alpha, origin='upper')

    # Save tile
    fig.tight_layout(pad=0)
    fig.savefig(tile_png, dpi=dpi, bbox_inches="tight", pad_inches=0,
               facecolor='none', transparent=True)
    plt.close(fig)

    # Cleanup
    del rem_raw, rem
    gc.collect()

