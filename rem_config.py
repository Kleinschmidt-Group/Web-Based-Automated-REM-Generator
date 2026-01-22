# rem_config.py
import os
from typing import Dict, List

# Export Settings
EXPORT_DPI = 150
EXPORT_SCALE = 1
CLIP_DEM = True  # Set to False if you want the full square DEM

# Folder Defaults
DOWNLOADS_FOLDER = "./Downloads"

# System and resilience settings
# Prevent computer from sleeping during long processing tasks
PREVENT_SLEEP = True

# Download rentry settings
DOWNLOAD_MAX_RETRIES = 5       # Number of times to retry a failed download
DOWNLOAD_RETRY_BACKOFF = 2.0   # Factor to increase wait time between retries
DOWNLOAD_BASE_TIMEOUT = 5.0    # Initial wait time in seconds

# Memory Safety Settings
# Threshold (0.0 - 1.0) of available RAM to use before warning/blocking
MEMORY_SAFETY_THRESHOLD = 0.90 
# Warn if an operation is estimated to use more than this many GB
MEMORY_WARNING_GB = 8.0
# Elevation threshold to clamp void pixels
ELEVATION_CLAMP_THRESHOLD = None

# Color Schemes
# Updated to match the High Contrast / Hue Jump logic in rem_hillshade_colorramp.py

# --- COLOR PALETTES ---
COLOR_SCHEMES: Dict[str, List[str]] = {
    "FloodAlertRed":  ["#000000", "#1A1A1A", "#333333", "#FFFF00", "#FFD700", "#FFA500", "#FF8C00", "#FF4500", "#FF0000", "#CC0000", "#800000", "#4D0000"],
    "NeonFloodCyan":  ["#000020", "#000050", "#4B0082", "#800080", "#9932CC", "#00FFFF", "#00CED1", "#40E0D0", "#32CD32", "#008000", "#004D00", "#000000"],
    "TropicWarning":  ["#1A0030", "#2E004F", "#4B0082", "#76FF03", "#64DD17", "#00C853", "#FF00FF", "#D500F9", "#AA00FF", "#FF6D00", "#E65100", "#BF360C", "#000000"],
    "VolcanicEruption":["#FFFF00", "#FFD700", "#FFA500", "#FF8C00", "#FF4500", "#FF0000", "#B22222", "#8B0000", "#4B0082", "#2E1A47", "#1A0030", "#0B0611"],
    "AquaCopper":     ["#003C46", "#1F9AA9", "#BFE7E3", "#EAD9B0", "#C46A2F", "#593018"],
    "PeacockBronze":  ["#0B4F6C", "#177E89", "#A6D9C8", "#F0EAD2", "#B08968", "#5C3D2E"],
    "GlacierLava":    ["#0F172A", "#1E40AF", "#93C5FD", "#E5E7EB", "#EF4444", "#4B1D1D"],
    "SunsetDune":     ["#2E1A47", "#8E3B82", "#F28CB1", "#FFD29D", "#E67E22", "#7B3F00"],
    "EsriClassic":    ["#00204D", "#6A0177", "#00B5FF", "#CFEFFF", "#FFE98A", "#C8A97D", "#4B2E2B"],
    "CyanDeepBlue":   ["#C4F7FF", "#8AE7FF", "#3FD4FF", "#008FDB", "#00427F", "#00112B"],
    "MagmaRamp":      ["#FFB347", "#FF7A00", "#FF4500", "#3366B3", "#0A2F6A", "#020712"],
}

SOLID_COLORS: Dict[str, str] = {
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

# Visualization Constraints
COLOR_RAMP_OPTIONS = [
    # High Contrast
    "FloodAlertRed",
    "NeonFloodCyan",
    "TropicWarning",
    "VolcanicEruption",
    # Standard
    "AquaCopper",
    "PeacockBronze",
    "GlacierLava",
    "SunsetDune",
    "EsriClassic",
    "CyanDeepBlue",
    "MagmaRamp",
    # Solids
    "SolidBlue",
    "SolidPurple",
    "SolidGold",
    "SolidGray",
    "SolidRed",
    "SolidBlack",
    "SolidOrange",
    "SolidNavy",
    "SolidGreen"
]

# Global Settings
CLIP_DEM = True  # Set to False to keep full DEM extent
BACKGROUND_OPTIONS = ["hillshade", "aerial", "white", "none"]
DEFAULT_BACKGROUND = "hillshade"
DOWNLOADS_FOLDER = "./Downloads"
EXPORT_DPI = 150
EXPORT_SCALE = 1
_FIXED_N_LEVELS = 256