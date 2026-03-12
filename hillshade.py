#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# hillshade.py

import math
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

# Fast vectorized hillshade — replaces matplotlib LightSource.hillshade().
# Uses surface-normal dot-product formula: no per-pixel arctan/arctan2 calls,
# just two np.gradient passes and a dot product. Roughly 2–4× faster than
# the matplotlib path for large arrays.

def _hillshade_fast(dem: np.ndarray, dx: float, dy: float,
                    azimuth_deg: float, altitude_deg: float) -> np.ndarray:
    az  = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)

    # Light direction unit vector
    lx = np.cos(alt) * np.cos(az)
    ly = np.cos(alt) * np.sin(az)
    lz = np.sin(alt)

    # Surface gradients (float64 for precision)
    d = dem.astype(np.float64, copy=False)
    dzdx = np.gradient(d, dx, axis=1)
    dzdy = np.gradient(d, dy, axis=0)

    # Surface normal: (-dzdx, -dzdy, 1) normalised
    inv_norm = 1.0 / np.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
    hs = (-dzdx * lx - dzdy * ly + lz) * inv_norm

    return np.clip(hs, 0.0, 1.0).astype(np.float32)


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
# Generates a hillshade from a DEM using parallel block processing.

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
    max_workers: Optional[int] = None,
) -> None:

    if multidirectional:
        if azimuths is None:
            azimuths = [315.0, 45.0, 225.0, 135.0]
        if verbose:
            print(f"Multi-directional hillshade: {len(azimuths)} directions {azimuths}")
    else:
        azimuths = [azimuth]
        if verbose:
            print(f"Single-direction hillshade: azimuth={azimuth}°")

    # Default workers: half of logical cores (leaves headroom for other work)
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) // 2)

    with rasterio.open(dem_path) as src:
        transform = src.transform
        resx = float(abs(transform.a))
        resy = float(abs(transform.e))
        crs  = src.crs
        height, width = src.height, src.width

        if verbose:
            if crs is None:
                print("WARNING: DEM has no CRS.")
            elif getattr(crs, "is_geographic", False):
                print("WARNING: DEM in geographic degrees — reproject to metres for best results.")
            else:
                print(f"INFO: DEM CRS: {crs}. Pixel size ≈ {resx:.3f} m × {resy:.3f} m")

        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

    # Build list of all block extents up-front
    blocks = []
    for row_off in range(0, height, block_size):
        for col_off in range(0, width, block_size):
            h = min(block_size, height - row_off)
            w = min(block_size, width  - col_off)
            blocks.append(Window(col_off, row_off, w, h))

    total_blocks = len(blocks)

    def _compute_only(dem_ext, pad_top, pad_left, h, w):
        """Pure numpy computation — no file I/O. Safe to run in threads."""
        nan_mask = ~np.isfinite(dem_ext)
        arr = dem_ext.astype("float32", copy=False) * float(z_factor)

        if len(azimuths) == 1:
            hs = _hillshade_fast(arr, resx, resy, azimuths[0], altitude)
        else:
            acc = np.zeros_like(arr, dtype="float32")
            for az in azimuths:
                acc += _hillshade_fast(arr, resx, resy, az, altitude)
            hs = acc / len(azimuths)

        if gamma != 1.0:
            hs = hs ** float(gamma)

        hs = np.where(nan_mask, np.nan, hs)
        inner = hs[pad_top:pad_top + h, pad_left:pad_left + w]
        return np.where(np.isfinite(inner), inner, -9999.0).astype("float32")

    if verbose:
        print(f"  Processing {total_blocks} blocks with {max_workers} workers...")

    # Pipeline: main thread reads → thread pool computes → main thread writes.
    # File I/O never enters the thread pool — avoids GDAL concurrency crashes.
    # At most (max_workers * 2) blocks are in-flight at once to cap memory use.
    done = 0
    in_flight_limit = max_workers * 2
    pending = []  # list of (future, win) in read order

    with rasterio.open(output_path, "w", **profile) as dst:
        with rasterio.open(dem_path) as src:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:

                for win in blocks:
                    h, w = int(win.height), int(win.width)
                    # Read happens here in the main thread — single file handle, no races
                    dem_ext, _, pad_top, pad_left = _safe_read(
                        src, 1, win, pad=1, fill_value=np.nan
                    )
                    future = pool.submit(_compute_only, dem_ext, pad_top, pad_left, h, w)
                    pending.append((future, win))

                    # Drain the oldest result once the pipeline is full
                    if len(pending) >= in_flight_limit:
                        f, completed_win = pending.pop(0)
                        dst.write(f.result(), 1, window=completed_win)
                        done += 1
                        if verbose and (done % 10 == 0 or done == total_blocks):
                            print(f"  hillshade progress: {done}/{total_blocks} blocks")

                # Drain any remaining results
                for f, completed_win in pending:
                    dst.write(f.result(), 1, window=completed_win)
                    done += 1
                    if verbose and (done % 10 == 0 or done == total_blocks):
                        print(f"  hillshade progress: {done}/{total_blocks} blocks")

    if verbose:
        mode_str = f"{len(azimuths)}-directional" if len(azimuths) > 1 else "single-direction"
        print(f"{mode_str.capitalize()} hillshade written: {output_path}")


def create_hillshade_fast_qa(
    dem_path: str,
    output_path: str,
    downsample_factor: Optional[int] = None,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    verbose: bool = True,
    target_pixels: int = 500,
) -> None:
    """
    Fast QA hillshade with adaptive downsampling.

    downsample_factor=None (default): automatically chosen so the longer
    axis is ~target_pixels (500px), regardless of input DEM size.
    Pass an explicit integer to override.
    """

    with rasterio.open(dem_path) as src:
        original_height = src.height
        original_width  = src.width
        transform        = src.transform

        # Adaptive factor: scale so the longer axis ≈ target_pixels
        if downsample_factor is None:
            longer_axis      = max(original_height, original_width)
            downsample_factor = max(1, longer_axis // target_pixels)

        new_height = max(64, original_height // downsample_factor)
        new_width  = max(64, original_width  // downsample_factor)

        if verbose:
            speedup = (original_width * original_height) / (new_width * new_height)
            print(f"Fast QA Hillshade:")
            print(f"  Original: {original_width} × {original_height} px")
            print(f"  QA size:  {new_width} × {new_height} px  (factor {downsample_factor}×, ~{speedup:.0f}× faster)")

        dem_ds = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=rasterio.enums.Resampling.average
        ).astype("float32")

        if src.nodata is not None:
            dem_ds = np.where(dem_ds == src.nodata, np.nan, dem_ds)

        resx = float(abs(transform.a)) * (original_width  / new_width)
        resy = float(abs(transform.e)) * (original_height / new_height)

        new_transform = Affine(
            transform.a * (original_width  / new_width),
            transform.b,
            transform.c,
            transform.d,
            transform.e * (original_height / new_height),
            transform.f,
        )

        dem_ds *= float(z_factor)
        nan_mask = ~np.isfinite(dem_ds)

        hs = _hillshade_fast(dem_ds, resx, resy, azimuth, altitude)
        hs = np.where(nan_mask, -9999.0, hs)

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
            blockxsize=min(256, new_width),
            blockysize=min(256, new_height),
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(hs.astype("float32"), 1)

    if verbose:
        print(f"Fast QA hillshade written: {output_path}")
