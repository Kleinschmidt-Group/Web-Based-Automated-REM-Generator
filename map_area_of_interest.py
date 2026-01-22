#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# map_area_of_interest.py
# Generates a local HTML map file allowing the user to draw an AOI (Area of Interest) and export it as a GeoJSON file

from __future__ import annotations
import os
import time
import webbrowser
from pathlib import Path
from typing import Tuple
import folium
from folium.plugins import Draw, Fullscreen, MeasureControl
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
except Exception:
    Nominatim = None
    RateLimiter = None


# Helper: Geocoding

def _geocode_region(region: str) -> Tuple[float, float]:
    
    # Geocode a region to (lat, lon). Falls back to CONUS center if geopy is unavailable or lookup fails
    
    fallback = (39.8283, -98.5795)
    if not region or Nominatim is None:
        return fallback
    try:
        geolocator = Nominatim(user_agent="aoi_draw_app")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        loc = geocode(region)
        if loc:
            print(f"Found location: {loc.address}")
            return (loc.latitude, loc.longitude)
    except Exception:
        pass
    print("Defaulting to entire United States.")
    return fallback


# Main Map Generator 

def create_aoi_map(
    region: str = "United States",
    output_html: str = "aoi_draw_map.html",
    draw_filename: str = "aoi.geojson",
    zoom: int | None = None,
    open_browser: bool = True,
    default_basemap: str = "aerial",  
) -> None:
    
    # Generates an interactive AOI map with Esri basemaps
    # Determine Map Center
    
    lat, lon = _geocode_region(region)
    zoom_start = zoom if zoom is not None else (7 if region and region != "United States" else 4)

    # Initialize Map Frame

    m = folium.Map(location=[lat, lon], zoom_start=zoom_start, tiles=None, control_scale=True)

    # Determine which layer is active by default based on user input
    
    show_aerial = default_basemap.lower().startswith("aer")
    show_topo = default_basemap.lower().startswith("top")

    # Add Basemaps (Esri Providers)
    # Esri World Imagery (Aerial)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Aerial (Esri World Imagery)",
        control=True,
        show=show_aerial,
    ).add_to(m)

    # Esri World Topo Map
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri & contributors",
        name="Topographic (Esri World Topo)",
        control=True,
        show=show_topo,
    ).add_to(m)

    # Configure Drawing Tools
    
    Draw(
        export=True,
        filename=draw_filename,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": {"shapeOptions": {"color": "#3388ff", "weight": 2}, "showArea": True},
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
                "shapeOptions": {"color": "#ff7800", "weight": 2},
            },
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # Add UI Controls
    
    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Save and Launch
    
    out_path = os.path.abspath(output_html)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m.save(out_path)

    print(f"\nAOI map saved as: {out_path}")
    if open_browser:
        print("Opening the AOI drawing map in your browser...")
        time.sleep(1)
        try:
            # Use Path.as_uri() for correct file URLs on Windows/macOS/Linux
            webbrowser.open(Path(out_path).as_uri())
        except Exception:
            pass

    print(f"Draw your AOI polygon or rectangle, then click 'Export' to download the GeoJSON as '{draw_filename}'.")