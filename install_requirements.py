#!/usr/bin/env python3
"""
REM Processing Application - Dependency Installer
Automatically installs all required packages with specified versions using conda and pip.

Usage:
    python install_requirements.py

Requirements:
    - Must be run in an Anaconda/Miniconda environment
    - Internet connection required for package downloads
"""

import subprocess
import sys
import os
import platform

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(message):
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(message):
    """Print a success message."""
    print(f"{Colors.OKGREEN}{message}{Colors.ENDC}")

def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}{message}{Colors.ENDC}")

def print_info(message):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")

def check_conda():
    """Check if conda is available."""
    # Check if we're in a conda environment
    if os.environ.get('CONDA_DEFAULT_ENV') or os.environ.get('CONDA_PREFIX'):
        env_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        print_success(f"Conda environment detected: {env_name}")
        return True
    
    # Fallback: try to run conda command
    try:
        result = subprocess.run(['conda', '--version'],
                              capture_output=True,
                              text=True,
                              check=True,
                              shell=True)
        print_success(f"Conda detected: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Conda not found!")
        print_info("Please ensure you're running this script in an Anaconda prompt.")
        print_info("Download Anaconda from: https://www.anaconda.com/download")
        return False

def get_python_version():
    """Get current Python version."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_info(f"Python version: {version}")
    return version

def install_conda_packages():
    """Install packages using conda from conda-forge."""
    print_header("Installing Core Packages via Conda")

    # GDAL must be installed FIRST (dependency for rasterio, geopandas, etc.)
    print_info("Step 1: Installing GDAL (critical geospatial dependency)...")
    try:
        subprocess.run(['conda', 'install', '-c', 'conda-forge', '-y', 'gdal=3.10.3'],
                      check=True, capture_output=True, text=True, shell=True)
        print_success("GDAL installed successfully!")
    except subprocess.CalledProcessError as e:
        print_error("Failed to install GDAL - this is critical!")
        print_warning("Geospatial packages may fail without GDAL. Continuing anyway...")

    print_info("\nStep 2: Installing remaining packages from conda-forge...")

    # Core Scientific Computing + Geospatial Libraries
    # Versions match working trem environment
    conda_packages = [
        # Core Scientific Computing
        "numpy=2.3.5",
        "scipy=1.16.3",
        "pandas=2.3.3",
        "numba=0.63.1",
        "xarray=2025.12.0",

        # Geospatial Core Libraries (GDAL already installed above)
        "geopandas=1.1.1",
        "rasterio=1.4.3",
        "shapely=2.1.1",
        "affine=2.4.0",
        "pyogrio=0.11.0",

        # Geospatial Data Acquisition
        "py3dep=0.19.0",
        "pynhd=0.19.4",

        # Visualization
        "matplotlib=3.10.8",
        "matplotlib-inline=0.1.6",
        "pillow=12.0.0",

        # Web Mapping
        "folium=0.20.0",
        "branca=0.8.2",
        "contextily=1.7.0",

        # Parallel Processing & Utilities
        "joblib=1.5.2",
        "psutil=5.9.0",

        # HTTP & Networking
        "requests=2.32.5",

        # Documentation Tools
        "numpydoc=1.9.0",
    ]

    print_info(f"Installing {len(conda_packages)} packages from conda-forge...")
    print_info("This may take several minutes. Please be patient.\n")

    # Install all conda packages in one command for efficiency
    cmd = ['conda', 'install', '-c', 'conda-forge', '-y'] + conda_packages

    try:
        print(f"{Colors.OKCYAN}Running: conda install -c conda-forge -y {' '.join(conda_packages[:3])}...{Colors.ENDC}\n")
        result = subprocess.run(cmd, check=True, text=True, shell=True)
        print_success("All conda packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install conda packages: {e}")
        print_warning("Trying individual package installation...")
        return install_conda_packages_individually(conda_packages)

def install_conda_packages_individually(packages):
    """Install conda packages one at a time if batch install fails."""
    failed_packages = []

    for package in packages:
        try:
            print_info(f"Installing {package}...")
            subprocess.run(['conda', 'install', '-c', 'conda-forge', '-y', package],
                         check=True,
                         capture_output=True,
                         text=True,
                         shell=True)
            print_success(f"{package} installed")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {package}")
            failed_packages.append(package)

    if failed_packages:
        print_warning(f"\nFailed packages: {', '.join(failed_packages)}")
        return False
    return True

def install_pip_packages():
    """Install packages using pip."""
    print_header("Installing Additional Packages via Pip")

    # Versions match working trem environment
    pip_packages = [
        "pyproj==3.7.2",
        "rioxarray==0.20.0",
        "streamlit==1.52.1",
        "streamlit-folium==0.25.2",
        "requests-cache==1.2.1",
    ]

    print_info(f"Installing {len(pip_packages)} packages via pip...\n")

    failed_packages = []

    for package in pip_packages:
        try:
            print_info(f"Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package],
                         check=True,
                         capture_output=True,
                         text=True)
            print_success(f"{package} installed")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {package}")
            failed_packages.append(package)

    if failed_packages:
        print_warning(f"\nFailed pip packages: {', '.join(failed_packages)}")
        return False

    print_success("\nAll pip packages installed successfully!")
    return True

def verify_installation():
    """Verify critical packages are installed correctly."""
    print_header("Verifying Installation")

    critical_packages = [
        ('numpy', 'Core scientific computing'),
        ('pandas', 'Data analysis'),
        ('scipy', 'Scientific algorithms'),
        ('numba', 'JIT compilation'),
        ('xarray', 'Multi-dimensional arrays'),
        ('geopandas', 'Geospatial dataframes'),
        ('rasterio', 'Raster I/O'),
        ('pyogrio', 'Fast geospatial I/O'),
        ('shapely', 'Geometric operations'),
        ('pyproj', 'Coordinate transformations'),
        ('rioxarray', 'Rasterio + xarray integration'),
        ('streamlit', 'Web application framework'),
        ('folium', 'Web mapping'),
        ('contextily', 'Basemap tiles'),
        ('py3dep', 'DEM data acquisition'),
        ('pynhd', 'River network data'),
    ]

    failed_imports = []

    for package, description in critical_packages:
        try:
            __import__(package)
            print_success(f"{package:20s} - {description}")
        except ImportError:
            print_error(f"{package:20s} - FAILED")
            failed_imports.append(package)

    # Special check for GDAL (can be imported via osgeo.gdal)
    try:
        from osgeo import gdal
        print_success(f"{'GDAL':20s} - Geospatial abstraction library")
    except ImportError:
        print_error(f"{'GDAL':20s} - FAILED (critical!)")
        failed_imports.append('GDAL')

    if failed_imports:
        print_error(f"\nFailed to import: {', '.join(failed_imports)}")
        print_warning("You may need to restart your Python kernel/terminal.")
        return False

    print_success("\nAll critical packages verified!")
    return True

def main():
    """Main installation routine."""
    print_header("REM Processing Application - Dependency Installer")
    print_info(f"Platform: {platform.system()} {platform.release()}")
    get_python_version()

    # Check for conda
    if not check_conda():
        sys.exit(1)

    # Confirm installation
    print_warning("\nThis will install ~30 packages. Continue? [Y/n]: ")
    response = input().strip().lower()
    if response and response not in ['y', 'yes']:
        print_info("Installation cancelled.")
        sys.exit(0)

    # Install conda packages
    conda_success = install_conda_packages()

    # Install pip packages
    pip_success = install_pip_packages()

    # Verify installation
    if conda_success and pip_success:
        verify_success = verify_installation()

        if verify_success:
            print_header("Installation Complete!")
            print_success("All dependencies installed successfully.")
            print_info("\nYou can now run the REM Processing Application:")
            print(f"  {Colors.BOLD}streamlit run app.py{Colors.ENDC}")
            print(f"  {Colors.BOLD}python run_rem_guided.py{Colors.ENDC}\n")
        else:
            print_warning("\nInstallation completed with warnings.")
            print_info("Try restarting your terminal/kernel and verify again.")
    else:
        print_error("\nInstallation completed with errors.")
        print_info("Check the error messages above for details.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nInstallation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
