#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# hillshade.py

import math
from typing import Optional
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from matplotlib.colors import LightSource

# Helper Functions

def _safe_read(src, band: int, win: Window, pad: int = 1, fill_value: Optional[float] = None):
    # Calculate the bounds of the padded window
    row_off = max(0, int(win.row_off) - pad)
    col_off = max(0, int(win.col_off) - pad)
    row_max = min(src.height, int(win.row_off + win.height) + pad)
    col_max = min(src.width,  int(win.col_off + win.width)  + pad)
    
    # Create the new window definition
    ext = Window(col_off, row_off, col_max - col_off, row_max - row_off)

    # Read the data
    arr = src.read(band, window=ext, boundless=False, masked=False)

    # Handle NoData values (replace with fill_value if needed)
    if fill_value is not None:
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, fill_value, arr)

    # Return the array and the offset information
    pad_top  = int(win.row_off) - row_off
    pad_left = int(win.col_off) - col_off
    return arr, ext, pad_top, pad_left


# Main Processing Function
# Generates a hillshade from a DEM using a memory-efficient streaming approach.

def create_hillshade(
    dem_path: str,
    output_path: str,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    gamma: float = 1.0,
    block_size: int = 2048,
    verbose: bool = True,
    multidirectional: bool = False,
    azimuths: Optional[list] = None,
) -> None:

    # Setup multi-directional azimuths
    if multidirectional:
        if azimuths is None:
            # Default: 4 cardinal directions for balanced illumination
            azimuths = [315.0, 45.0, 225.0, 135.0]
        if verbose:
            print(f"Multi-directional hillshade mode: {len(azimuths)} directions")
            print(f"  Azimuths: {azimuths}")
    else:
        azimuths = [azimuth]
        if verbose:
            print(f"Single-direction hillshade: azimuth={azimuth}°")

    # Setup Lighting sources for each azimuth
    light_sources = [LightSource(azdeg=az, altdeg=altitude) for az in azimuths]

    with rasterio.open(dem_path) as src:
        
        # Get pixel resolution (needed for correct slope calculation)
        transform = src.transform
        resx = float(abs(transform.a))
        resy = float(abs(transform.e))

        # CRS Guardrails
        crs = src.crs
        if crs is None:
            if verbose:
                print("WARNING: DEM has no CRS. Assuming meters for dx/dy; verify before trusting hillshade.")
        elif getattr(crs, "is_geographic", False):
            if verbose:
                print("WARNING: DEM is in geographic degrees. Hillshade will use degree-based dx/dy; "
                      "reproject to a projected CRS in meters for best results.")
        else:
            if verbose:
                print(f"INFO: DEM CRS: {crs}. Pixel size ≈ {resx:.3f} m × {resy:.3f} m")

        # Prepare Output Profile
        # We write a compressed, tiled GeoTIFF using float32 data
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=-9999.0,  # Set NODATA to preserve transparent areas
            compress="deflate",
            predictor=3,
            tiled=True,
            # Force blocks to 256x256. This prevents GDAL errors when the 
            # image dimensions are not divisible by 16 (e.g. width=470).
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

        height, width = src.height, src.width
        total_blocks = math.ceil(height / block_size) * math.ceil(width / block_size)
        done = 0

        # Main Processing Loop
        with rasterio.open(output_path, "w", **profile) as dst:
            
            # Iterate over the image in chunks (windows) to save memory
            for row_off in range(0, height, block_size):
                for col_off in range(0, width, block_size):
                    
                    # Define the 'Target' window
                    h = min(block_size, height - row_off)
                    w = min(block_size, width  - col_off)
                    win = Window(col_off, row_off, w, h)

                    # Read DEM with Halo
                    dem_ext, _win_ext, pad_top, pad_left = _safe_read(
                        src, 1, win, pad=1, fill_value=np.nan
                    )

                    # Track NaN mask to preserve NODATA areas
                    nan_mask_ext = ~np.isfinite(dem_ext)

                    # Apply vertical exaggeration (z-factor)
                    dem_ext = dem_ext.astype("float32", copy=False) * float(z_factor)

                    # Compute Multi-Directional Hillshade
                    # Calculate hillshade from each azimuth and blend
                    if len(light_sources) == 1:
                        # Single direction
                        hs_ext = light_sources[0].hillshade(dem_ext, vert_exag=1.0, dx=resx, dy=resy, fraction=1.0)
                    else:
                        # Multi-directional: average multiple azimuths
                        hs_accumulator = np.zeros_like(dem_ext, dtype="float32")
                        for ls_i in light_sources:
                            hs_i = ls_i.hillshade(dem_ext, vert_exag=1.0, dx=resx, dy=resy, fraction=1.0)
                            hs_accumulator += hs_i
                        hs_ext = hs_accumulator / len(light_sources)

                    # Apply Gamma correction (brightness/contrast adjustment)
                    if gamma != 1.0:
                        hs_ext = hs_ext ** float(gamma)

                    # Restore NaN mask (preserve NODATA areas)
                    hs_ext = np.where(nan_mask_ext, np.nan, hs_ext)

                    # Crop and Write
                    # Remove the halo padding to get back to the exact 'Target' window size
                    top = pad_top
                    left = pad_left
                    inner = hs_ext[top:top + h, left:left + w]

                    # Convert NaN to NODATA value for output
                    inner = np.where(np.isfinite(inner), inner, -9999.0)

                    dst.write(inner.astype("float32", copy=False), 1, window=win)

                    # Progress logging
                    done += 1
                    if verbose and (done % 10 == 0 or done == total_blocks):
                        print(f"  hillshade progress: {done}/{total_blocks} blocks")

    if verbose:
        mode_str = f"{len(azimuths)}-directional" if len(azimuths) > 1 else "single-direction"
        print(f"{mode_str.capitalize()} hillshade written: {output_path}")


def create_hillshade_fast_qa(
    dem_path: str,
    output_path: str,
    downsample_factor: int = 10,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    verbose: bool = True,
) -> None:

    with rasterio.open(dem_path) as src:
        # Calculate downsampled dimensions
        original_height = src.height
        original_width = src.width
        new_height = max(100, original_height // downsample_factor)
        new_width = max(100, original_width // downsample_factor)

        if verbose:
            print(f"Fast QA Hillshade:")
            print(f"  Original DEM: {original_width} × {original_height} pixels")
            print(f"  Downsampled: {new_width} × {new_height} pixels")
            print(f"  Speed gain: ~{(original_width * original_height) / (new_width * new_height):.0f}x faster")

        # Read downsampled DEM
        # Uses rasterio's built-in resampling
        dem_downsampled = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=rasterio.enums.Resampling.average
        )

        # Get adjusted pixel resolution
        transform = src.transform
        resx = float(abs(transform.a)) * downsample_factor
        resy = float(abs(transform.e)) * downsample_factor

        # Create adjusted transform for downsampled raster
        new_transform = Affine(
            transform.a * downsample_factor,
            transform.b,
            transform.c,
            transform.d,
            transform.e * downsample_factor,
            transform.f
        )

        # Handle NoData
        if src.nodata is not None:
            dem_downsampled = np.where(
                dem_downsampled == src.nodata,
                np.nan,
                dem_downsampled
            )

        # Apply z-factor
        dem_downsampled = dem_downsampled.astype("float32") * float(z_factor)

        # Calculate hillshade (single pass, no blocks needed for small array)
        light_source = LightSource(azdeg=azimuth, altdeg=altitude)
        hillshade_array = light_source.hillshade(
            dem_downsampled,
            vert_exag=1.0,
            dx=resx,
            dy=resy,
            fraction=1.0
        )

        # Preserve NoData areas
        nan_mask = ~np.isfinite(dem_downsampled)
        hillshade_array = np.where(nan_mask, -9999.0, hillshade_array)

        # Write output
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            height=new_height,
            width=new_width,
            transform=new_transform,
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(hillshade_array.astype("float32"), 1)

    if verbose:
        print(f"Fast QA hillshade written: {output_path}")