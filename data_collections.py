#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# data_collections.py

from __future__ import annotations
import os
import glob
import math
import time
import shutil
import logging
import sys
import multiprocessing
import warnings
from typing import Optional, List, Tuple
import requests
import warnings
from rasterio.warp import transform_bounds
import traceback
import rioxarray
from rioxarray.merge import merge_arrays
import dask
import traceback
import gc

# Fix PROJ unicode errors with Python 3.13
os.environ['PROJ_SKIP_READ_USER_WRITABLE_DIRECTORY'] = 'YES'
os.environ['PROJ_DEBUG'] = '0'
os.environ['PROJ_NETWORK'] = 'OFF'

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import box
from shapely.ops import unary_union
from affine import Affine
import requests
from joblib import Parallel, delayed
import psutil
import pynhd
import rem_config as cfg

# Suppress annoying warnings
warnings.filterwarnings('ignore', category=UnicodeWarning)
warnings.filterwarnings('ignore', message='.*utf-8.*')
warnings.filterwarnings('ignore', module='pyproj')

try:
    import py3dep
    import xarray as xr
    import rioxarray
    from rioxarray.merge import merge_arrays # CRITICAL for new mosaic function
    import dask # CRITICAL for chunking
except ImportError:
    print("WARNING: 'py3dep', 'xarray', 'rioxarray', or 'dask' not found. Smart Mosaic will fail.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_JOBS = 4
CONSERVATIVE_JOBS = 2  # For rate-limited APIs like ScienceBase


# Validation Helpers

def validate_raster_integrity(file_path: str) -> bool:
    """Confidently validates a downloaded tile."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False
    try:
        with rasterio.open(file_path) as src:
            if src.width < 1 or src.height < 1: return False

            # Try strict validation first (full checksum)
            try:
                src.checksum(1)  # Force pixel decode
                return True
            except Exception as checksum_err:
                # If checksum fails due to unsupported compression, try lenient validation
                error_msg = str(checksum_err).lower()
                if "code not yet in table" in error_msg or "unsupported" in error_msg:
                    logger.info(f"  Checksum failed (unsupported compression), using lenient validation for {os.path.basename(file_path)}")

                    # Lenient validation: Try reading a small sample of pixels
                    try:
                        # Read a 100x100 sample from center
                        y_center = src.height // 2
                        x_center = src.width // 2
                        sample = src.read(1, window=((max(0, y_center-50), min(src.height, y_center+50)),
                                                      (max(0, x_center-50), min(src.width, x_center+50))))

                        # Check we got actual data (not all zeros/nodata)
                        if sample.size > 0:
                            if src.nodata is not None:
                                valid_pixels = sample[sample != src.nodata]
                            else:
                                valid_pixels = sample.flatten()

                            if valid_pixels.size > 0:
                                logger.info(f"  {os.path.basename(file_path)} validated (lenient mode)")
                                return True
                    except:
                        pass

                # If lenient validation also failed, raise original error
                raise checksum_err

    except Exception as e:
        logger.warning(f"Corrupt file {os.path.basename(file_path)}: {e}")
        return False

def verify_mosaic_content(mosaic_path: str) -> bool:
    """Validates final mosaic and filters out extreme NoData artifacts."""
    try:
        with rasterio.open(mosaic_path) as src:
            # Read a downsampled overview
            data = src.read(1, out_shape=(1, int(src.height // 10), int(src.width // 10)))
            
            # Mask NoData
            if src.nodata is not None: 
                valid_mask = (data != src.nodata)
            else: 
                valid_mask = np.isfinite(data)

            # Filter out extreme values (often artifacts of float32 limits)
            # Valid elevation on Earth is approx -500m to 9000m. 
            # We use a loose buffer: -12,000 to +12,000.
            # Anything outside this range is likely a NoData glitch.
            sanity_mask = (data > -12000) & (data < 12000)
            
            final_mask = valid_mask & sanity_mask
            valid_pixels = data[final_mask]

            if valid_pixels.size == 0:
                logger.error("Mosaic created but contains ONLY NoData or invalid values.")
                return False

            min_val = np.min(valid_pixels)
            max_val = np.max(valid_pixels)
            logger.info(f"Mosaic Validated. Elev Range: {min_val:.1f} to {max_val:.1f}")
            return True
            
    except Exception as e:
        logger.error(f"Mosaic validation failed: {e}")
        return False

#Checks what percentage of the mosaic contains actual data (not NoData)
#Returns completness percentage and warns if below threshold

def check_data_completeness(mosaic_path: str, warning_threshold: float = 85.0) -> float:
    try:
        with rasterio.open(mosaic_path) as src:
            # Sample at 10% resolution for speed
            df = 10
            data = src.read(1, out_shape=(1, int(src.height // df), int(src.width // df)))

            total_pixels = data.size
            
            # Smart mask that handles both explicit NoData and "Infinite" glitches
            if src.nodata is not None:
                valid_mask = (data != src.nodata)
            else:
                valid_mask = np.isfinite(data)
                
            # Filter out the "infinite" junk values
            sanity_mask = (data > -12000) & (data < 12000)
            final_mask = valid_mask & sanity_mask

            valid_pixels = np.sum(final_mask)
            completeness = (valid_pixels / total_pixels) * 100.0

            if completeness < warning_threshold:
                logger.warning(f"  DATA COMPLETENESS WARNING")
                logger.warning(f"   Only {completeness:.1f}% of the mosaic contains valid data")
                logger.warning(f"   {100.0 - completeness:.1f}% is NoData (missing coverage)")
                logger.warning(f"   This may be due to gaps in available lidar/DEM coverage for this area")
                logger.warning(f"   Consider:")
                logger.warning(f"     - Using a coarser resolution (10m or 30m) which has better coverage")
                logger.warning(f"     - Adjusting your AOI to avoid the gap")
                logger.warning(f"     - Checking nationalmap.gov/viewer to see actual data availability")
            else:
                logger.info(f"Data completeness: {completeness:.1f}% (good coverage)")

            return completeness

    except Exception as e:
        logger.warning(f"Could not check data completeness: {e}")
        return 100.0  # Assume OK if check fails



# Helpers: AOI & Geometry & Location


def find_aoi_geojsons(aoi_folder: str) -> List[str]:
    if not os.path.exists(aoi_folder): return []
    return sorted(glob.glob(os.path.join(aoi_folder, "*.geojson")))

def _get_unified_aoi(geojson_files: List[str]) -> gpd.GeoDataFrame:
    aoi_list = []
    for g in geojson_files:
        try:
            a = gpd.read_file(g)
            if not a.empty:
                if a.crs is None: a = a.set_crs("EPSG:4326")
                elif a.crs.to_epsg() != 4326: a = a.to_crs("EPSG:4326")
                aoi_list.append(a)
        except Exception: pass
    
    if not aoi_list: return None
    return gpd.GeoDataFrame(
        geometry=[unary_union(pd.concat(aoi_list).geometry)],
        crs="EPSG:4326"
    )

def _get_bbox_str(gdf: gpd.GeoDataFrame) -> str:
    if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs("EPSG:4326")
    b = gdf.total_bounds
    return f"{b[0]},{b[1]},{b[2]},{b[3]}"

#xhecks if the AOI is roughly within the Continental US (Lower 48)

def _is_location_conus(gdf: gpd.GeoDataFrame) -> bool:
    try:
        minx, miny, maxx, maxy = gdf.total_bounds
        # Loose bounding box for CONUS
        conus_w, conus_e = -125.0, -66.0
        conus_s, conus_n = 24.0, 50.0
        
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        return (conus_w <= cx <= conus_e) and (conus_s <= cy <= conus_n)
    except Exception:
        return True # Default to True (safe mode)



# Availability Check (Project Tiles Only)

#Query ScienceBase for Alaska 5m DEM tiles

def _query_sciencebase_alaska_5m(aoi_gdf: gpd.GeoDataFrame, data_type: str = "ifsar") -> List[dict]:
    if aoi_gdf.crs.to_epsg() != 4326: aoi_gdf = aoi_gdf.to_crs("EPSG:4326")
    aoi_geom = unary_union(aoi_gdf.geometry)
    minx, miny, maxx, maxy = aoi_geom.bounds

    api_url = "https://www.sciencebase.gov/catalog/items"

    # Different parent IDs for different datasets
    # IFSAR: 5641fe98e4b0831b7d62e758
    # Mid-Accuracy might be under different collection - try searching both
    if data_type == "ifsar":
        parent_ids = ["5641fe98e4b0831b7d62e758"]
        logger.info("Querying ScienceBase for Alaska IFSAR 5m tiles...")
    else:  # mid_accuracy
        # Try multiple possible parent IDs or broader search
        parent_ids = ["5641fe98e4b0831b7d62e758", "63d02af7d34e8c7b609d03d1"]  # Second ID might contain Mid-Accuracy
        logger.info("Querying ScienceBase for Alaska Mid-Accuracy 5m tiles...")

    all_tiles = []

    for parent_id in parent_ids:
        params = {
            "parentId": parent_id,
            "max": 1000, "format": "json", "fields": "title,spatial,webLinks",
            "bbox": f"{minx},{miny},{maxx},{maxy}"
        }

        try:
            r = requests.get(api_url, params=params, timeout=30)
            if r.status_code != 200: continue

            items = r.json().get("items", [])
            logger.info(f"DEBUG - ScienceBase returned {len(items)} items from parent {parent_id}")

            for item in items:
                title = item.get("title", "")

                # Filter by type
                title_lower = title.lower()
                if data_type == "ifsar":
                    if "mid" in title_lower and "accuracy" in title_lower:
                        continue  # Skip mid-accuracy when looking for IFSAR
                elif data_type == "mid_accuracy":
                    if not ("mid" in title_lower and "accuracy" in title_lower):
                        continue  # Skip non-mid-accuracy

                spatial = item.get("spatial", {})
                tile_bbox = spatial.get("boundingBox", {})
                if not tile_bbox: continue

                try:
                    tmx, tmy, tMx, tMy = float(tile_bbox.get("minX")), float(tile_bbox.get("minY")), float(tile_bbox.get("maxX")), float(tile_bbox.get("maxY"))
                    if (tMx < minx or tmx > maxx or tMy < miny or tmy > maxy): continue

                    download_url = None
                    for link in item.get("webLinks", []):
                        if link.get("type") == "download" and link.get("title") == "TIFF":
                            download_url = link.get("uri"); break

                    if download_url:
                        all_tiles.append({"title": title, "downloadURL": download_url, "boundingBox": tile_bbox, "source": f"sciencebase_{data_type}"})
                        if data_type == "mid_accuracy":
                            logger.info(f"DEBUG - Found Mid-Accuracy tile: {title[:60]}...")
                except: continue

        except Exception as e:
            logger.warning(f"ScienceBase API error for parent {parent_id}: {e}")
            continue

    return all_tiles

#Legacy wrapper for ISFAR tiles

def _query_sciencebase_ifsar(aoi_gdf: gpd.GeoDataFrame) -> List[dict]:
    return _query_sciencebase_alaska_5m(aoi_gdf, data_type="ifsar")

def get_available_project_resolutions(aoi_geojson_path: str, min_coverage_percent: float = 30.0) -> List[int]:
    try:
        aoi_gdf = gpd.read_file(aoi_geojson_path)
        if aoi_gdf.crs is None: aoi_gdf = aoi_gdf.set_crs("EPSG:4326")
        elif aoi_gdf.crs.to_epsg() != 4326: aoi_gdf = aoi_gdf.to_crs("EPSG:4326")
        
        is_conus = _is_location_conus(aoi_gdf)
        region = "CONUS" if is_conus else "Non-CONUS (AK/HI/PR)"
        logger.info(f"Checking DEM availability for region: {region}")
        
        aoi_geom = unary_union(aoi_gdf.geometry)
        bbox_str = _get_bbox_str(aoi_gdf)
        
        # USGS TNM API (Works everywhere)
        api_url = "https://tnmaccess.nationalmap.gov/api/v1/products"
        params = {"bbox": bbox_str, "prodFormats": "GeoTIFF", "max": 3000}
        
        resolution_keywords = {
            1: ["1 meter", "1m", "one meter"],  # Lidar projects - keep as-is
            3: ["3 meter", "3m", "1/9 arc-second", "1/9 arc second", "19 arc", "0.11 arc"],
            5: ["5 meter", "5m", "alaska 5 meter", "ak_ifsar", "ifsar", "5m ifsar", "mid-accuracy", "mid accuracy"],  # Alaska only
            10: ["10 meter", "10m", "1/3 arc-second", "1/3 arc second", "0.33 arc", "13 arc-second", "13 arc second", "13 arc"],
            30: ["30 meter", "30m", "1 arc-second", "1 arc second", "1arc"],
            60: ["60 meter", "60m", "2 arc-second", "2 arc second", "2arc"]  # Alaska only
        }

        logger.info("Querying USGS for available Project tiles...")
        r = requests.get(api_url, params=params, timeout=30)
        items = r.json().get("items", []) if r.status_code == 200 else []
        logger.info(f"Availability check: API returned {len(items)} items")

        tiles_by_resolution = {res: [] for res in resolution_keywords.keys()}

        for item in items:
            title = item.get("title", "").lower()
            durl = item.get("downloadURL", "")
            bbox_item = item.get("boundingBox", {})
            if not durl or not bbox_item: continue
            
            is_s1m = "s1m" in title or "standard 1-meter" in title or "/S1M/" in durl
            if is_s1m: continue

            for res, keywords in resolution_keywords.items():
                if any(kw in title for kw in keywords):
                    try:
                        tgeom = box(float(bbox_item["minX"]), float(bbox_item["minY"]), float(bbox_item["maxX"]), float(bbox_item["maxY"]))
                        tiles_by_resolution[res].append(tgeom)
                    except: pass
                    break

        # SCIENCEBASE (Only run if NOT in CONUS)
        if not is_conus:
            logger.info("Non-CONUS detected: Checking ScienceBase for Alaska IFSAR...")
            ifsar_tiles = _query_sciencebase_ifsar(aoi_gdf)
            for tile in ifsar_tiles:
                try:
                    bb = tile["boundingBox"]
                    tgeom = box(float(bb["minX"]), float(bb["minY"]), float(bb["maxX"]), float(bb["maxY"]))
                    tiles_by_resolution[5].append(tgeom)
                except: pass

            logger.info("Non-CONUS detected: Checking ScienceBase for Alaska Mid-Accuracy...")
            mid_acc_tiles = _query_sciencebase_alaska_5m(aoi_gdf, data_type="mid_accuracy")
            for tile in mid_acc_tiles:
                try:
                    bb = tile["boundingBox"]
                    tgeom = box(float(bb["minX"]), float(bb["minY"]), float(bb["maxX"]), float(bb["maxY"]))
                    tiles_by_resolution[5].append(tgeom)
                except: pass

        # Calculate Coverage
        logger.info(f"Tiles found per resolution: {[(res, len(geoms)) for res, geoms in tiles_by_resolution.items() if geoms]}")
        available_resolutions = []
        aoi_area = aoi_geom.area
        for res, tile_geoms in tiles_by_resolution.items():
            if not tile_geoms: continue
            coverage_pct = (unary_union(tile_geoms).intersection(aoi_geom).area / aoi_area) * 100.0
            logger.info(f"  {res}m: {len(tile_geoms)} tiles, {coverage_pct:.1f}% coverage (threshold: {min_coverage_percent}%)")
            if coverage_pct >= min_coverage_percent: available_resolutions.append(res)
            
        return sorted(available_resolutions)

    except Exception as e:
        logger.error(f"Failed to check availability: {e}")
        return []

def get_available_dem_resolutions(aoi_geojson_path: str) -> List[int]:
    return get_available_project_resolutions(aoi_geojson_path)



# STRATEGY A: Source/Project API (With Validation)

#Download a single tile with progress logging and conservative retry logic

def _download_file_worker(url, dest_path, tile_index=0, total_tiles=0):
    filename = os.path.basename(dest_path)

    # Check if already downloaded and valid
    if os.path.exists(dest_path):
        if validate_raster_integrity(dest_path):
            logger.info(f"  [{tile_index}/{total_tiles}] {filename} (already downloaded)")
            return dest_path
        else:
            os.remove(dest_path)

    temp_path = dest_path + ".tmp"

    for attempt in range(1, cfg.DOWNLOAD_MAX_RETRIES + 1):
        try:
            if attempt > 1:
                logger.info(f"  [{tile_index}/{total_tiles}] Retry {attempt-1} for {filename}...")
            else:
                logger.info(f"  [{tile_index}/{total_tiles}] Downloading {filename}...")

            with requests.get(url, stream=True, timeout=120) as r:  # Increased timeout to 120s
                r.raise_for_status()

                # Download with size tracking
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0

                with open(temp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):  # Larger chunks = fewer writes
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                # Log download size
                size_mb = downloaded / (1024 * 1024)
                logger.info(f"  [{tile_index}/{total_tiles}] {filename} ({size_mb:.1f} MB)")

            os.replace(temp_path, dest_path)

            # Validate
            if validate_raster_integrity(dest_path):
                return dest_path
            else:
                logger.warning(f"  [{tile_index}/{total_tiles}] ✗ {filename} (failed validation)")
                if os.path.exists(dest_path):
                    os.remove(dest_path)

        except requests.exceptions.Timeout:
            logger.warning(f"  [{tile_index}/{total_tiles}] Timeout on {filename}")
            time.sleep(cfg.DOWNLOAD_RETRY_BACKOFF * attempt * 2)  # Longer delay on timeout

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                logger.warning(f"  [{tile_index}/{total_tiles}] Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                logger.warning(f"  [{tile_index}/{total_tiles}] HTTP {e.response.status_code} on {filename}")
                time.sleep(cfg.DOWNLOAD_RETRY_BACKOFF * attempt)

        except Exception as e:
            logger.warning(f"  [{tile_index}/{total_tiles}] Error: {str(e)[:50]}")
            time.sleep(cfg.DOWNLOAD_RETRY_BACKOFF * attempt)

    logger.error(f"  [{tile_index}/{total_tiles}] ✗ {filename} FAILED after {cfg.DOWNLOAD_MAX_RETRIES} attempts")
    return None

def _calculate_aoi_area_km2(aoi_gdf):
    """Calculate AOI area in square kilometers using equal-area projection."""
    try:
        # Project to appropriate equal-area CRS
        if aoi_gdf.crs is None or aoi_gdf.crs.to_epsg() != 4326:
            aoi_gdf = aoi_gdf.to_crs("EPSG:4326")

        # Use Albers Equal Area for CONUS, or appropriate projection
        try:
            aoi_proj = aoi_gdf.to_crs("EPSG:5070")  # Albers
        except:
            aoi_proj = aoi_gdf.to_crs("EPSG:3857")  # Web Mercator fallback

        area_m2 = aoi_proj.geometry.area.sum()
        area_km2 = area_m2 / 1_000_000
        return area_km2
    except Exception as e:
        logger.warning(f"Could not calculate AOI area: {e}")
        return 0.0

#Strategy A: Source/Project API, queries for both project-specific tiles AND standard (S1M) tiles
# Removes artificial blocks on S1M data to ensure gap-filling
# Downloads ALL available unique tiles (Newest First)
# Relies on Mosaic function to layer correctly

def _download_via_source_api(aoi_gdf, output_folder, resolution, n_jobs=DEFAULT_JOBS, alaska_5m_strategy=None):
    is_conus = _is_location_conus(aoi_gdf)
    logger.info(f"Strategy A: Searching for {resolution}m data in {'CONUS' if is_conus else 'Non-CONUS'}...")

    bbox_str = _get_bbox_str(aoi_gdf)
    api_url = "https://tnmaccess.nationalmap.gov/api/v1/products"
    
    # Increased max results to ensure we don't miss tiles in large AOIs
    params = {"bbox": bbox_str, "prodFormats": "GeoTIFF", "max": 10000}

    target_any = []
    if resolution == 1: target_any = ["1 meter", "1m", "one meter"]
    elif resolution == 3: target_any = ["3 meter", "3m", "1/9 arc-second", "1/9 arc second", "19 arc", "0.11 arc"]
    elif resolution == 5: target_any = ["5 meter", "5m", "alaska 5 meter", "ak_ifsar", "ifsar", "5m ifsar", "mid-accuracy", "mid accuracy"]
    elif resolution == 10: target_any = ["10 meter", "10m", "1/3 arc-second", "1/3 arc second", "0.33 arc", "13 arc-second", "13 arc second", "13 arc"]
    elif resolution == 30: target_any = ["30 meter", "30m", "1 arc-second", "1 arc second", "1arc"]
    elif resolution == 60: target_any = ["60 meter", "60m", "2 arc-second", "2 arc second", "2arc"]
    else: return None

    aoi_geom = unary_union(aoi_gdf.geometry)
    s1m_candidates, project_candidates = [], []

    try:
        r = requests.get(api_url, params=params, timeout=60)
        items = r.json().get("items", [])
        logger.info(f"API returned {len(items)} total items")

        for item in items:
            title = item.get("title", "").lower()
            durl = item.get("downloadURL", "")
            bbox_item = item.get("boundingBox", {})
            pub_date = item.get("publicationDate", "")
            if not durl: continue

            # Loose intersection check to filter irrelevant tiles
            tile_geom = None
            if bbox_item:
                try:
                    tile_geom = box(float(bbox_item["minX"]), float(bbox_item["minY"]), float(bbox_item["maxX"]), float(bbox_item["maxY"]))
                    if not tile_geom.intersects(aoi_geom): continue
                except:
                    pass 

            if any(t in title for t in target_any):
                is_s1m = "s1m" in title or "standard 1-meter" in title or "/S1M/" in durl              
                candidate = {
                    "url": durl,
                    "geometry": tile_geom,
                    "pub_date": pub_date,
                    "title": item.get("title", "")
                }

                if is_s1m: s1m_candidates.append(candidate)
                else: project_candidates.append(candidate)

    except Exception as e:
        logger.error(f"API Error: {e}")
        return None

    # Alaska 5m Handling
    mid_accuracy_candidates = []
    ifsar_candidates = []

    if not is_conus and resolution == 5:
        # Split TNM results
        for candidate in project_candidates:
            title = candidate.get("title", "").lower()
            if "mid-accuracy" in title or "mid accuracy" in title:
                mid_accuracy_candidates.append(candidate)
            else:
                ifsar_candidates.append(candidate)

        # Add ScienceBase results (filling gaps)
        logger.info("Non-CONUS: Querying ScienceBase for extra coverage...")
        
        mid_acc_sb = _query_sciencebase_alaska_5m(aoi_gdf, data_type="mid_accuracy")
        for tile in mid_acc_sb:
            mid_accuracy_candidates.append({
                "url": tile.get("downloadURL"),
                "pub_date": "", 
                "title": tile.get("title", ""),
                "source": "sciencebase_mid_accuracy"
            })
            
        ifsar_sb = _query_sciencebase_ifsar(aoi_gdf)
        for tile in ifsar_sb:
            ifsar_candidates.append({
                "url": tile.get("downloadURL"),
                "pub_date": "", 
                "title": tile.get("title", ""),
                "source": "sciencebase_ifsar"
            })

        # Combine - Get ALL of them to avoid gaps
        logger.info(f"Alaska Candidates: {len(mid_accuracy_candidates)} Mid-Acc, {len(ifsar_candidates)} IFSAR")
        project_candidates = mid_accuracy_candidates + ifsar_candidates

    # Newest tiles first
    def deduplicate_urls_only(candidates, category_name="tiles"):
        if not candidates: return []
        original_count = len(candidates)
        seen_urls = set()
        unique_candidates = []
        # Sort by date (Newest First)
        for candidate in sorted(candidates, key=lambda x: x.get("pub_date", "") or "0000-00-00", reverse=True):
            if candidate["url"] not in seen_urls:
                seen_urls.add(candidate["url"])
                unique_candidates.append(candidate)
        
        logger.info(f"Downloading {len(unique_candidates)} unique {category_name} (sorted newest first)")
        return unique_candidates

    project_optimized = deduplicate_urls_only(project_candidates, "Project tiles")
    s1m_optimized = deduplicate_urls_only(s1m_candidates, "S1M tiles")

    # Download List Prep
    valid_files = []
    temp_dir = os.path.join(output_folder, "temp_source_tiles")
    os.makedirs(temp_dir, exist_ok=True)
    
    download_queue = []

    # Add Project Tiles
    for i, t in enumerate(project_optimized):
        url = t["url"]
        title = t.get("title", "").lower()
        
        # Tagging for Mosaic Priority
        prefix = "project"
        if not is_conus and resolution == 5:
            if "mid-accuracy" in title or "mid accuracy" in title: prefix = "midacc"
            else: prefix = "ifsar"
            
        download_queue.append((url, os.path.join(temp_dir, f"{prefix}_{i}.tif")))

    # Add S1M Tiles (ALWAYS add them as backup/fill)
    if s1m_optimized:
        logger.info(f"Adding {len(s1m_optimized)} S1M tiles to ensure complete coverage...")
        for i, t in enumerate(s1m_optimized):
            download_queue.append((t["url"], os.path.join(temp_dir, f"s1m_{i}.tif")))

    if not download_queue:
        return []

    # Execute Download
    total_tiles = len(download_queue)
    logger.info(f"Starting download of {total_tiles} total tiles (Project + S1M)...")

    # Conservative mode for Alaska/ScienceBase mixed loads
    current_jobs = CONSERVATIVE_JOBS if (not is_conus and resolution == 5) else n_jobs

    tasks = [
        (url, dest, i+1, total_tiles)
        for i, (url, dest) in enumerate(download_queue)
    ]

    res = Parallel(n_jobs=current_jobs, backend="threading")(
        delayed(_download_file_worker)(u, d, idx, total) for u, d, idx, total in tasks
    )
    
    valid_files = [r for r in res if r]
    return valid_files



# STRATEGY B: WCS Tiling (With Validation)

def _flip_if_needed(da):
    try:
        t = da.rio.transform()
        if t.e > 0:
            da = da.isel(y=slice(None, None, -1)).assign_coords(y=da.y[::-1])
            new_transform = Affine(t.a, t.b, t.c, t.d, -t.e, t.f)
            da.rio.write_transform(new_transform, inplace=True)
    except Exception: pass
    return da

def _calculate_safe_tile_km(resolution_m: float) -> float:
    if resolution_m <= 5.0: return 15.0
    elif resolution_m <= 10.0: return 35.0
    else: return 50.0

def _tiles_from_aoi(aoi_gdf: gpd.GeoDataFrame, tile_km: float, resolution_m: int = 30) -> List[Tuple]:
    # Ensure Standard Input CRS
    if aoi_gdf.crs is None:
        aoi_gdf = aoi_gdf.set_crs("EPSG:4326")
    elif aoi_gdf.crs.to_epsg() != 4326:
        aoi_gdf = aoi_gdf.to_crs("EPSG:4326")

    # Defensive Projection
    # We try to project to Albers (5070) for accurate metering
    # If PROJ is corrupt or AOI is outside bounds, we fallback to Web Mercator (3857)
    aoi_m = None
    projection_errors = []

    for epsg_code in ["EPSG:5070", "EPSG:3857"]:
        try:
            # Suppress pyproj warnings during projection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_projection = aoi_gdf.to_crs(epsg_code)
                # Validate the projection immediately
                test_bounds = test_projection.total_bounds
                if all(np.isfinite(test_bounds)):
                    aoi_m = test_projection
                    break
                else:
                    projection_errors.append(f"{epsg_code}: Invalid bounds")
        except Exception as e:
            projection_errors.append(f"{epsg_code}: {str(e)}")
            continue

    if aoi_m is None:
        logger.error(f" All projections failed: {'; '.join(projection_errors)}")
        return []

    geom = unary_union(aoi_m.geometry)
    if geom.is_empty:
        logger.error(" Geometry is empty after projection")
        return []

    # Calculate Bounds safely
    minx, miny, maxx, maxy = geom.bounds

    # Sanity check for "Infinite" bounds which happen on corrupt reprojections
    if not all(np.isfinite([minx, miny, maxx, maxy])):
        logger.error(" Error: AOI bounds resulted in Infinity/NaN. Check geometry.")
        return []

    width_km = (maxx - minx) / 1000.0
    height_km = (maxy - miny) / 1000.0
    
    # Soft limits
    limit = 2000 if resolution_m >= 30 else 250
    if max(width_km, height_km) > limit:
        logger.warning(f"Large AOI ({width_km:.1f}x{height_km:.1f} km). Processing may be slow.")

    step = float(tile_km) * 1000.0
    
    # Generate Tiles
    # Use numpy arange for grid generation
    try:
        xs = np.arange(math.floor(minx), math.ceil(maxx), step)
        ys = np.arange(math.floor(miny), math.ceil(maxy), step)
    except Exception as e:
        logger.error(f"Grid generation failed: {e}")
        return []

    tiles = []
    # Loop through grid
    for x0 in xs:
        for y0 in ys:
            tile_box = box(x0, y0, x0 + step, y0 + step)
            if tile_box.intersects(geom):
                # Buffer slightly to ensure overlap
                # Reproject back to 4326 for the API call
                t_gdf = gpd.GeoDataFrame(geometry=[tile_box.buffer(50)], crs=aoi_m.crs).to_crs("EPSG:4326")
                tiles.append(tuple(t_gdf.total_bounds))
                
    return tiles

def _download_wcs_worker(bbox, resolution, idx, temp_dir, prefix=""):
    tile_path = os.path.join(temp_dir, f"wcs_{prefix}_{idx}.tif")
    if os.path.exists(tile_path): 
        if validate_raster_integrity(tile_path): return tile_path
        else: os.remove(tile_path)
    
    for attempt in range(1, 4):
        try:
            da = py3dep.get_dem(bbox, resolution).squeeze()
            if da.size > 0:
                da = _flip_if_needed(da)
                da.rio.to_raster(tile_path, compress=None) 
                if validate_raster_integrity(tile_path): return tile_path
        except: time.sleep(5 * attempt)
    return None 

def _download_via_wcs(aoi_gdf, output_folder, resolution):
    logger.info(f"Strategy B (WCS): Tiling for {resolution}m data...")
    try: import py3dep
    except ImportError: return None

    start_km = _calculate_safe_tile_km(resolution)
    attempts = sorted(list(set([start_km, 16.0, 12.0])), reverse=True)

    temp_dir = os.path.join(output_folder, "temp_wcs_tiles")
    os.makedirs(temp_dir, exist_ok=True)
    
    for km in attempts:
        tiles = _tiles_from_aoi(aoi_gdf, km, resolution)
        if not tiles: continue
        
        logger.info(f"WCS Attempt: {len(tiles)} tiles (~{km}km each).")
        valid_tiles = []
        all_good = True
        
        for i, bbox in enumerate(tiles):
            sys.stdout.write(f"\r  Downloading WCS tile {i+1}/{len(tiles)}...")
            sys.stdout.flush()
            res = _download_wcs_worker(bbox, resolution, i, temp_dir, prefix=str(km))
            if res: valid_tiles.append(res)
            else: all_good = False
        
        print("") 
        if all_good and len(valid_tiles) == len(tiles): return valid_tiles
        
    return []


# Mosaicking (Smart Dispatcher with WarpedVRT Support)

#Smart Mosaic Dispatcher
#Estimates output size and available RAM
#Uses fast rioxarray method when safe

def _mosaic_with_progress(src_paths: List[str], out_path: str, resolution: int, n_jobs: int, elevation_clamp: float = None, alaska_5m_strategy=None):
    if not src_paths: return None

    try:
        # Estimate Output size
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"  RAM: {available_ram_gb:.1f}GB available / {total_ram_gb:.1f}GB total")

        # Quick scan to estimate output mosaic size
        with rasterio.open(src_paths[0]) as src:
            sample_crs = src.crs

        # Get bounds of all tiles
        all_bounds = []
        for fp in src_paths:
            with rasterio.open(fp) as src:
                # Reproject bounds to EPSG:5070 if needed
                if src.crs and src.crs.to_epsg() != 5070:
                    bounds_5070 = transform_bounds(src.crs, "EPSG:5070", *src.bounds)
                else:
                    bounds_5070 = src.bounds
                all_bounds.append(bounds_5070)

        # Calculate merged extent
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)

        # Estimate output dimensions
        est_width = int((max_x - min_x) / resolution)
        est_height = int((max_y - min_y) / resolution)
        est_pixels = est_width * est_height
        est_size_gb = (est_pixels * 4) / (1024**3)  # float32 = 4 bytes

        logger.info(f"  Estimated output: {est_height} × {est_width} pixels (~{est_size_gb:.1f}GB)")

        # Decision Logic: Fast vs Ultra-Safe
        # Switch to Safe Mode if:
        # 1. Dataset > 500M pixels (Lowered threshold to catch large Alaska tiles)
        # 2. Or if RAM isn't 2x larger than dataset
        safe_ram_needed = (est_size_gb * 2.0) + 2.0
        
        # New Strict Rules
        use_fast_method = True
        if est_pixels > 500_000_000: # 500M Pixel Threshold
            use_fast_method = False
            logger.info("  Reason: Dataset > 500M pixels (Force Safe Mode)")
        elif available_ram_gb < safe_ram_needed:
            use_fast_method = False
            logger.info(f"  Reason: Insufficient RAM (Need {safe_ram_needed:.1f}GB, Have {available_ram_gb:.1f}GB)")
            
        if use_fast_method:
            logger.info(f"  Using FAST rioxarray method")
            return _mosaic_rioxarray_fast(src_paths, out_path, resolution, alaska_5m_strategy, available_ram_gb)
        else:
            logger.warning(f"  Large dataset detected - using ULTRA-SAFE WarpedVRT method")
            logger.warning(f"     This avoids OOM crashes by streaming data in windows.")
            return _mosaic_rasterio_safe(src_paths, out_path, resolution, alaska_5m_strategy)

    except Exception as e:
        logger.error(f"Mosaic dispatcher failed: {e}")
        traceback.print_exc()
        # Last resort: try ultra-safe method
        logger.warning("  Attempting ultra-safe fallback...")
        return _mosaic_rasterio_safe(src_paths, out_path, resolution, alaska_5m_strategy)

#Uses Dask Lazy loading and merge_arrays
#Adaptive chunking based on available RAM

def _mosaic_rioxarray_fast(src_paths: List[str], out_path: str, resolution: int, alaska_5m_strategy=None, available_ram_gb=8.0):
    try:

        # Priority Sorting
        def get_priority(path):
            basename = os.path.basename(path)
            if "s1m_" in basename: return 0
            if alaska_5m_strategy == "mid_accuracy":
                if "midacc_" in basename: return 2
                if "ifsar_" in basename: return 1
            elif alaska_5m_strategy == "ifsar":
                if "ifsar_" in basename: return 2
                if "midacc_" in basename: return 1
            else:
                if "midacc_" in basename: return 2
                if "ifsar_" in basename: return 1
            return 1

        src_paths.sort(key=get_priority, reverse=True)

        logger.info("Priority Order (Top Layer First):")
        for p in src_paths[:3]: logger.info(f"  - {os.path.basename(p)}")
        if len(src_paths) > 3: logger.info(f"  ... and {len(src_paths)-3} others")

        # Adaptive Chunking
        if available_ram_gb < 4:
            chunk_size = 256
        elif available_ram_gb < 8:
            chunk_size = 512
        elif available_ram_gb < 16:
            chunk_size = 1024
        else:
            chunk_size = 2048
        logger.info(f"  Using {chunk_size}px chunks")

        # Lazy Opening and reprohection
        datasets = []
        dst_crs = "EPSG:5070"

        logger.info(f"  Reprojecting tiles to {dst_crs}...")
        for fp in src_paths:
            da = rioxarray.open_rasterio(fp, chunks={'band': 1, 'x': chunk_size, 'y': chunk_size}).squeeze()
            if da.rio.crs is None or da.rio.crs.to_epsg() != 5070:
                da = da.rio.reproject(dst_crs, resolution=resolution, nodata=-9999.0)
            da.rio.write_nodata(-9999.0, inplace=True)
            datasets.append(da)

        # Merge and write
        logger.info("  Merging arrays...")
        merged = merge_arrays(datasets, nodata=-9999.0)
        merged.attrs["long_name"] = "Elevation"

        logger.info(f"  Streaming to disk...")
        with dask.config.set(scheduler='synchronous'):
            merged.rio.to_raster(
                out_path,
                tiled=True,
                compress=None,
                windowed=True,
                bigtiff="YES",
                dtype="float32",
                lock=False,
                num_threads=1
            )

        for da in datasets: da.close()
        logger.info(f"Mosaic saved (fast method): {out_path}")
        return out_path

    except Exception as e:
        logger.error(f"Fast method failed: {e}")
        traceback.print_exc()
        try:
            for da in datasets: da.close()
        except: pass
        # Fallback to safe method
        logger.warning("  Falling back to ultra-safe method...")
        return _mosaic_rasterio_safe(src_paths, out_path, resolution, alaska_5m_strategy)


#Reproject a single tile to target CRS and resolutino
#Returns the output path on success, None on failure

def _reproject_tile_worker(src_path: str, dst_path: str, dst_crs: str, resolution: int, tile_idx: int, total_tiles: int):
    try:
        with rasterio.open(src_path) as src:
            # Determine source NoData
            src_nodata = src.nodata if src.nodata is not None else -32767.0

            # Calculate transform and dimensions for reprojection
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=resolution
            )

            # Set up output profile
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': -9999.0,
                'dtype': rasterio.float32,
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'compress': None  # Faster for temp files
            })

            # Reproject
            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=-9999.0,
                    resampling=Resampling.bilinear,
                    num_threads=2
                )

        logger.info(f"  [{tile_idx}/{total_tiles}] Reprojected {os.path.basename(src_path)}")
        return dst_path

    except Exception as e:
        logger.error(f"  [{tile_idx}/{total_tiles}] ✗ Failed to reproject {os.path.basename(src_path)}: {e}")
        return None


#Merges pre-projected tiles with increased memory limit
#Minimal RAM usage but 5-10x faster than WarpedVRT approach

def _mosaic_rasterio_safe(src_paths: List[str], out_path: str, resolution: int, alaska_5m_strategy=None):
    try:
        from rasterio.merge import merge

        # Priority Sorting
        def get_priority(path):
            basename = os.path.basename(path)
            if "s1m_" in basename: return 0
            if alaska_5m_strategy == "mid_accuracy":
                if "midacc_" in basename: return 2
                if "ifsar_" in basename: return 1
            elif alaska_5m_strategy == "ifsar":
                if "ifsar_" in basename: return 2
                if "midacc_" in basename: return 1
            else:
                if "midacc_" in basename: return 2
                if "ifsar_" in basename: return 1
            return 1

        src_paths.sort(key=get_priority, reverse=True)

        logger.info("Priority Order (Top Layer First):")
        for p in src_paths[:3]: logger.info(f"  - {os.path.basename(p)}")
        if len(src_paths) > 3: logger.info(f"  ... and {len(src_paths)-3} others")

        # Reprojhect all tiles in parrallel
        dst_crs = "EPSG:5070"

        # Create temp folder for reprojected tiles
        temp_reproj_dir = os.path.join(os.path.dirname(out_path), "temp_reprojected")
        os.makedirs(temp_reproj_dir, exist_ok=True)

        logger.info(f"  Step 1: Pre-reprojecting {len(src_paths)} tiles to {dst_crs} (parallel)...")

        # Build task list
        reproj_tasks = []
        for i, src_path in enumerate(src_paths):
            dst_path = os.path.join(temp_reproj_dir, f"reproj_{i}_{os.path.basename(src_path)}")
            reproj_tasks.append((src_path, dst_path, dst_crs, resolution, i+1, len(src_paths)))

        # Execute reprojection in parallel (conservative parallelism for memory safety)
        n_jobs = min(CONSERVATIVE_JOBS, multiprocessing.cpu_count())
        reprojected_paths = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_reproject_tile_worker)(src, dst, crs, res, idx, total)
            for src, dst, crs, res, idx, total in reproj_tasks
        )

        # Filter out failed reprojections
        reprojected_paths = [p for p in reprojected_paths if p is not None]

        if not reprojected_paths:
            logger.error("All tile reprojections failed!")
            shutil.rmtree(temp_reproj_dir, ignore_errors=True)
            return None

        logger.info(f"  Successfully reprojected {len(reprojected_paths)}/{len(src_paths)} tiles")

        # Merge Pre-Reprojected tiles
        logger.info(f"  Step 2: Merging {len(reprojected_paths)} pre-projected tiles...")

        # Open all reprojected tiles
        datasets = [rasterio.open(fp) for fp in reprojected_paths]

        # Merge with increased memory limit (user has 8.5GB available)
        # 6GB allows larger windows = fewer I/O operations = faster
        mosaic, out_transform = merge(
            datasets,
            nodata=-9999.0,
            dtype=np.float32,
            resampling=Resampling.bilinear,
            method='first',  
            mem_limit=6144  
        )

        # Write to disk
        logger.info(f"  Step 3: Writing final mosaic to disk...")
        out_crs = rasterio.crs.CRS.from_epsg(5070)

        with rasterio.open(
            out_path, 'w',
            driver='GTiff',
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=1,
            dtype=rasterio.float32,
            crs=out_crs,
            transform=out_transform,
            nodata=-9999.0,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress=None,  # Faster writes
            bigtiff='YES'   
        ) as dest:
            dest.write(mosaic[0], 1)

        # Cleanup
        for ds in datasets:
            ds.close()

        # Remove temp reprojected tiles
        shutil.rmtree(temp_reproj_dir, ignore_errors=True)

        logger.info(f"Mosaic saved (ultra-safe optimized method): {out_path}")
        return out_path

    except Exception as e:
        logger.error(f"Ultra-safe method failed: {e}")
        traceback.print_exc()

        # Cleanup on failure
        try:
            for ds in datasets:
                ds.close()
        except:
            pass

        try:
            if 'temp_reproj_dir' in locals():
                shutil.rmtree(temp_reproj_dir, ignore_errors=True)
        except:
            pass

        return None



# Orchestrator (Hybrid Strategy)


def download_and_mosaic_dems(
    geojson_files: List[str], output_folder: str, resolution: int,
    n_jobs_download: int = DEFAULT_JOBS, n_jobs_mosaic: int = DEFAULT_JOBS,
    elevation_clamp: Optional[float] = None,
    alaska_5m_strategy: Optional[str] = None  # "mid_accuracy", "ifsar", or None for auto
) -> Optional[str]:

    if not geojson_files: return None
    os.makedirs(output_folder, exist_ok=True)
    aoi_gdf = _get_unified_aoi(geojson_files)
    if aoi_gdf is None: return None

    # Check if this is Alaska 5m - determine or use provided strategy
    is_conus = _is_location_conus(aoi_gdf)

    if not is_conus and resolution == 5:
        aoi_area_km2 = _calculate_aoi_area_km2(aoi_gdf)
        logger.info(f"AOI Area: {aoi_area_km2:.1f} km²")

        # If user provided a strategy, use it
        if alaska_5m_strategy in ["mid_accuracy", "ifsar"]:
            logger.info(f"Using user-specified strategy: {alaska_5m_strategy}")
        else:
            # Automatic strategy based on AOI size
            # Threshold for automatic Mid-Accuracy priority: 500 km² or larger
            if aoi_area_km2 >= 500:
                logger.warning(f"Large AOI detected ({aoi_area_km2:.1f} km²)")
                logger.info("Automatically using Mid-Accuracy priority for faster download")
                logger.info("   Mid-Accuracy tiles will be prioritized (larger, faster to download)")
                logger.info("   IFSAR tiles will fill any coverage gaps")
                logger.info("   Both sources combined ensure complete coverage")
                alaska_5m_strategy = "mid_accuracy"
            elif aoi_area_km2 >= 100:
                logger.info(f"Medium-sized AOI ({aoi_area_km2:.1f} km²)")
                logger.info("Using balanced strategy: IFSAR priority with Mid-Accuracy gap-fill")
                alaska_5m_strategy = None  # Default balanced approach
            else:
                # Small AOI, prefer IFSAR for best detail
                logger.info("Small AOI - using IFSAR priority for highest detail")
                alaska_5m_strategy = "ifsar"

    valid_tiles = []

    # Always prioritize Project Tiles (more reliable, higher quality)
    logger.info(f"Attempting {resolution}m download via Project Tiles (Strategy A)...")
    valid_tiles = _download_via_source_api(aoi_gdf, output_folder, resolution, n_jobs_download, alaska_5m_strategy)

    if not valid_tiles:
        logger.warning(f" Project tiles not available or failed. Fallback to WCS (Strategy B)...")
        valid_tiles = _download_via_wcs(aoi_gdf, output_folder, resolution)

    if not valid_tiles:
        logger.error(f"Download failed for {resolution}m resolution.")
        return None

    mosaic_path = os.path.join(output_folder, f"mosaic_{resolution}m_dem.tif")
    clamp = elevation_clamp if elevation_clamp is not None else cfg.ELEVATION_CLAMP_THRESHOLD

    result = _mosaic_with_progress(valid_tiles, mosaic_path, resolution, n_jobs_mosaic, elevation_clamp=clamp, alaska_5m_strategy=alaska_5m_strategy)
    
    if result:
        logger.info("Validating final mosaic...")
        if verify_mosaic_content(result):
            logger.info("Mosaic Validation Passed.")

            # Check for data completeness (NoData gaps)
            check_data_completeness(result, warning_threshold=85.0)

            # Safe cleanup with retry logic for Windows
            temp_dirs = [
                os.path.join(output_folder, "temp_source_tiles"),
                os.path.join(output_folder, "temp_wcs_tiles")
            ]
            
            for temp_dir in temp_dirs:
                if not os.path.exists(temp_dir):
                    continue
                    
                try:
                    # Try direct removal first
                    shutil.rmtree(temp_dir, ignore_errors=False)
                    logger.info(f"Cleaned up: {os.path.basename(temp_dir)}")
                except (PermissionError, OSError) as e:
                    # Windows file locking - try with retry
                    logger.warning(f"Initial cleanup failed for {os.path.basename(temp_dir)}: {e}")
                    logger.info("Attempting cleanup with retry logic...")
                    
                    # Force garbage collection to release file handles
                    gc.collect()
                    time.sleep(0.5)
                    
                    try:
                        # Retry with ignore_errors=True as fallback
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        logger.info(f"Cleanup successful: {os.path.basename(temp_dir)}")
                    except Exception as e2:
                        # Final fallback - log but don't crash
                        logger.warning(f"Could not fully remove {os.path.basename(temp_dir)}: {e2}")
                        logger.warning("Leftover files will be cleaned up on next run")
                        
            return mosaic_path
            
    return None


# River Functions (Spatially Aware Smart-Scan)

#Scans for river names using spatially aware routing
#CONUS: NHDPlus V2
#Alaska: High-Res

def scan_nhd_rivers(aoi_geojson_path: str, max_retries: int = 3) -> list[str]:
    gdf = gpd.read_file(aoi_geojson_path)
    if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs("EPSG:4326")
    
    is_conus = _is_location_conus(gdf)
    region_name = "CONUS (Lower 48)" if is_conus else "Non-CONUS (AK/HI/PR)"
    logger.info(f"Detected Region: {region_name}")

    found_rivers = []

    # NHDPlus V2 (Only run if in CONUS)
    if is_conus:
        try:
            logger.info("Running Fast Scan (NHDPlus V2)...")
            wd = pynhd.WaterData("nhdflowline_network")
            res = wd.bybox(tuple(gdf.total_bounds))
            
            if not res.empty:
                res = res.to_crs("EPSG:4326")
                try: clipped = gpd.clip(res, gdf)
                except: clipped = res.cx[gdf.total_bounds[0]:gdf.total_bounds[2], gdf.total_bounds[1]:gdf.total_bounds[3]]

                if not clipped.empty:
                    stats = clipped.groupby("gnis_name").apply(lambda x: x.geometry.length.sum()).sort_values(ascending=False)
                    found_rivers = [n for n in stats.index if n and n != ""]
                    if found_rivers:
                        logger.info(f"Fast Scan found {len(found_rivers)} rivers.")
                        return found_rivers
            else:
                logger.info("Fast Scan returned no results.")
        except Exception as e:
            logger.warning(f"Fast Scan skipped/failed: {e}")

    # NHD High-Res (Fallback OR Primary for AK)
    logger.info(f"Running Deep Scan (NHD High-Res) for {region_name}...")
    
    gdf_proj = gdf.to_crs("EPSG:3857")
    gdf_proj["geometry"] = gdf_proj.buffer(1000) 
    gdf = gdf_proj.to_crs("EPSG:4326")
    bbox = gdf.total_bounds
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    
    url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query"
    params = {
        "f": "json", "geometry": bbox_str, "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelEnvelopeIntersects", "inSR": "4326",
        "outFields": "gnis_name,lengthkm", "returnGeometry": "false", "where": "gnis_name IS NOT NULL"
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code != 200: time.sleep(2); continue
            data = r.json()
            if "features" not in data: return []
            stats = {}
            for feat in data["features"]:
                name = feat["attributes"].get("gnis_name")
                length = feat["attributes"].get("lengthkm") or 0
                if name: stats[name] = stats.get(name, 0) + length
            names = [n for n, l in sorted(stats.items(), key=lambda x: x[1], reverse=True)]
            logger.info(f"Deep Scan found {len(names)} rivers.")
            return names
        except: time.sleep(2)
    return []

#Direct API dowbload: Targeted fetch for High-Res NHD
#Uses get request (instead of post) to bypass HTTP 403 blocks
#Uses Fuzzy matching for ribust river name finding

def _download_nhd_surgical(aoi_gdf, river_name=None, max_retries=3, target_crs="EPSG:5070"):
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query"

    # Robust where clause
    if river_name:
        # Escape single quotes and use fuzzy matching
        sanitized_name = river_name.replace("'", "''").upper()
        where_clause = f"UPPER(gnis_name) LIKE '%{sanitized_name}%'"
    else:
        where_clause = "gnis_name IS NOT NULL"

    logger.info(f"Surgical Fetch: Downloading High-Res vectors for '{river_name if river_name else 'ALL'}'...")
    logger.info(f"  Query: {where_clause}")

    # Extract CRS code
    target_sr = target_crs.split(":")[-1] if ":" in target_crs else target_crs

    # Headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "*/*",
        "Connection": "keep-alive"
    }

    params = {
        "f": "geojson",
        "geometry": bbox_str, "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelEnvelopeIntersects",
        "inSR": "4326",
        "outSR": target_sr,
        "outFields": "gnis_name,permanent_identifier",
        "returnGeometry": "true",
        "where": where_clause
    }

    for attempt in range(max_retries):
        try:
            # Switched from POST to GET to bypass 403 Firewall
            r = requests.get(url, params=params, headers=headers, timeout=120)
            
            if r.status_code != 200:
                logger.warning(f"Attempt {attempt+1}: HTTP {r.status_code} - {r.reason}")
                time.sleep(5)
                continue
            
            try:
                gdf = gpd.read_file(r.text, driver="GeoJSON")
                if gdf.crs is None or gdf.crs.to_epsg() != int(target_sr):
                    gdf = gdf.set_crs(target_crs, allow_override=True)
            except Exception as e:
                logger.warning(f"Failed to parse response: {e}")
                # Sometimes GET returns HTML on error instead of JSON
                if "DOCTYPE html" in r.text[:100]:
                    logger.warning("Server returned HTML instead of GeoJSON (likely a proxy error).")
                return gpd.GeoDataFrame()

            if not gdf.empty:
                logger.info(f"Downloaded river in {gdf.crs} ({len(gdf)} features)")
                return gdf
            else:
                if attempt == 0 and river_name:
                    logger.info("  No named match found. Trying broader search (all named rivers in box)...")
                    params["where"] = "gnis_name IS NOT NULL"
                else:
                    logger.warning(f"  Attempt {attempt+1}: No data found.")

        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5)
            
    return gpd.GeoDataFrame()

def choose_and_save_nhd_river(geojson_files, output_folder, river_choice=1, river_name=None):
    wd = pynhd.WaterData("nhdflowline_network")
    all_named = []
    
    first_gdf = gpd.read_file(geojson_files[0])
    if first_gdf.crs is None: first_gdf = first_gdf.set_crs(4326)
    else: first_gdf = first_gdf.to_crs(4326)
    is_conus = _is_location_conus(first_gdf)
    
    # NHDPlus V2 (Only if CONUS)
    if is_conus:
        logger.info("Region is CONUS: Attempting NHDPlus V2 download...")
        for g in geojson_files:
            try:
                aoi = gpd.read_file(g).to_crs("EPSG:4326")
                fl = wd.bybox(tuple(aoi.total_bounds))
                if not fl.empty:
                    fl = fl.to_crs("EPSG:4326")
                    try: clipped = gpd.clip(fl, aoi)
                    except: clipped = fl.cx[aoi.total_bounds[0]:aoi.total_bounds[2], aoi.total_bounds[1]:aoi.total_bounds[3]]

                    if river_name: named = clipped[clipped["gnis_name"] == river_name]
                    else: named = clipped[clipped["gnis_name"].notna() & (clipped["gnis_name"] != "")]

                    if not named.empty: all_named.append(named)
            except: continue
    else:
        logger.info("Region is Non-CONUS: Skipping NHDPlus V2.")

    # NHD High-Res (Fallback OR Primary for AK)
    if not all_named:
        logger.info("Downloading from NHD High-Res MapServer...")
        for g in geojson_files:
            aoi = gpd.read_file(g).to_crs("EPSG:4326")
            named = _download_nhd_surgical(aoi, river_name=river_name, target_crs="EPSG:5070")
            if not named.empty:
                try:
                    # Reproject AOI to match river CRS before clipping
                    aoi_reprojected = aoi.to_crs(named.crs)
                    clipped = gpd.overlay(named, aoi_reprojected, how='intersection')
                    if not clipped.empty:
                        all_named.append(clipped)
                    else:
                        # If clipping failed, just use the named river (already filtered by name)
                        logger.info("Clipping returned empty, using full river extent")
                        all_named.append(named)
                except Exception as e:
                    # If clipping fails, use unclipped river
                    logger.warning(f"Clipping failed ({e}), using full river extent")
                    all_named.append(named)

    if not all_named: 
        logger.error("Could not find any river data.")
        return None

    named_all = pd.concat(all_named, ignore_index=True)
    
    if river_name:
        chosen = river_name
        final = named_all[named_all["gnis_name"] == chosen]
    else:
        stats = named_all.groupby("gnis_name").apply(lambda x: x.geometry.length.sum()).sort_values(ascending=False)
        chosen = stats.index[river_choice - 1] if 0 <= river_choice - 1 < len(stats) else stats.index[0]
        final = named_all[named_all["gnis_name"] == chosen]

    logger.info(f"Selected River: {chosen} ({len(final)} segments)")
    
    out_path = os.path.join(output_folder, f"NHD_{chosen.replace(' ', '_')}_centerlines.gpkg")
    try: final = final.dissolve(by="gnis_name")
    except: pass
    
    final.to_file(out_path, driver="GPKG")
    return out_path