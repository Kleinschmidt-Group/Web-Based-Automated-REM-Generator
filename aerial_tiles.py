#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# aerial_tiles.py
# Build an aerial GeoTIFF aligned to a target DEM (same CRS, transform, width/height)

from __future__ import annotations
import os
import tempfile
from typing import Optional, Tuple
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
import contextily as ctx

WEBMERC = "EPSG:3857"
WGS84   = "EPSG:4326"

# Esri World Imagery

DEFAULT_PROVIDER = ctx.providers.Esri.WorldImagery

# Helper Functions

def _expand_bounds(bounds_wgs84: Tuple[float, float, float, float], margin_frac: float) -> Tuple[float,float,float,float]:
    
    #Adds a percentage buffer to the bounds to prevent edge gaps
    
    w, s, e, n = bounds_wgs84
    dw = (e - w) * float(margin_frac)
    dh = (n - s) * float(margin_frac)
    return (w - dw, s - dh, e + dw, n + dh)

def _estimate_zoom_from_res(pixel_size_m: float) -> int:
    
    #Matches the DEM resolution (in meters) to the nearest Web Mercator zoom level
    
    if pixel_size_m <= 0:
        return 18

    # Thresholds for zoom levels 19 down to 10
    
    for z, thr in [(19,0.40),(18,0.8),(17,1.6),(16,3.2),(15,6.4),(14,12.8),(13,25.6),(12,51.2),(11,102.4),(10,204.8)]:
        if pixel_size_m <= thr:
            return z
    return 10

# Main Processing Function

def build_aerial_geotiff_like(
    dem_path: str,
    out_aerial_tif: str,
    *,
    provider=None,              
    margin_frac: float = 0.02,  
    zoom: Optional[int] = None  
) -> Optional[str]:

    try:
        if provider is None:
            provider = DEFAULT_PROVIDER

        # Analyze the Reference DEM
        
        with rasterio.open(dem_path) as dem:
            dem_crs = dem.crs
            dem_transform = dem.transform
            dem_h = dem.height
            dem_w = dem.width

            # Calculate resolution to determine appropriate map zoom level
            
            px_m = (abs(dem_transform.a) + abs(dem_transform.e)) / 2.0
            z = zoom if zoom is not None else _estimate_zoom_from_res(max(px_m, 0.01))

            # Convert DEM bounds to WGS84 (Lat/Lon) required for tile downloading
            
            dem_bounds = dem.bounds  # (left, bottom, right, top)
            b_wgs84 = transform_bounds(dem_crs, WGS84, *dem_bounds, densify_pts=21)
            b_wgs84 = _expand_bounds(b_wgs84, margin_frac)

        # Download and Process Imagery
        
        with tempfile.TemporaryDirectory() as tdir:
            wm_tif = os.path.join(tdir, "aerial_webmerc.tif")
            
            # Download tiles from provider into a temporary Web Mercator GeoTIFF
            
            ctx.bounds2raster(*b_wgs84, wm_tif, zoom=z, source=provider, ll=True)

            # 3. Reproject Web Mercator -> Target DEM Projection
            
            with rasterio.open(wm_tif) as src:
                
                # Setup output profile to match the DEM exactly (same grid, same CRS)
                
                indexes = [1,2,3] if src.count >= 3 else [1]
                profile = {
                    "driver": "GTiff",
                    "height": dem_h,
                    "width": dem_w,
                    "count": len(indexes),
                    "dtype": rasterio.uint8,
                    "crs": dem_crs,
                    "transform": dem_transform,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "compress": "deflate",
                    "predictor": 2,
                }
                
                # Write the output file
                
                os.makedirs(os.path.dirname(out_aerial_tif) or ".", exist_ok=True)
                with rasterio.open(out_aerial_tif, "w", **profile) as dst:
                    for i, b in enumerate(indexes, start=1):
                        src_band = src.read(b)
                        dst_band = np.zeros((dem_h, dem_w), dtype=np.uint8)
                        
                        # Warp the downloaded image to align with the DEM
                        
                        reproject(
                            source=src_band,
                            destination=dst_band,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dem_transform,
                            dst_crs=dem_crs,
                            resampling=Resampling.bilinear,
                        )
                        dst.write(dst_band, i)

        return out_aerial_tif

    except Exception as e:
        print(f"WARNING: build_aerial_geotiff_like failed: {e}")
        return None
