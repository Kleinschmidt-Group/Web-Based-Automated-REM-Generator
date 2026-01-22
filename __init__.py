"""
Automated REM Generator

A professional tool for generating Relative Elevation Models (REMs) from DEM data.
This package provides automated workflows for river analysis and flood mapping.

Main Components:
    - REM_Calcs: Core REM calculation algorithms
    - rem_utils: Utility functions for geospatial processing
    - data_collections: Data acquisition and management
    - rem_hillshade_colorramp: Visualization and rendering
    - app: Streamlit web application interface

Usage:
    Run the web application:
        python app.py

    Run guided CLI workflow:
        python run_rem_guided.py

    Import as package:
        from automated_rem_generator import REM_Calcs, rem_utils
"""

__version__ = "1.0.0"
__author__ = "Ethan Muhlestein"
__license__ = "MIT"

# Import key modules for easier access
from . import rem_utils
from . import REM_Calcs
from . import rem_config
from . import data_collections
from . import rem_hillshade_colorramp

# Define what's available when someone does 'from automated_rem_generator import *'
__all__ = [
    'rem_utils',
    'REM_Calcs',
    'rem_config',
    'data_collections',
    'rem_hillshade_colorramp',
]
