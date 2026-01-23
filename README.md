# **Automated Web-Based Relative Elevation Model Generator**
**Version 1.0**
**Kleinschmidt Associates**
**Developer: Ethan Muhlestein**


<img width="863" height="770" alt="image" src="https://github.com/user-attachments/assets/bd5808ec-55c8-4099-9cb0-6daa475e67b3" />




A video on the Sotware can be found on the Kleinschmidt Q drive under "Internal Data", "Project_Templates", "Relative Elevation Model Generator". A video can be requested from the developer for non-Kleinschmidt employees.
## **Overview**
The Automated Web-Based Relative Elevation Model Generator is a software tool created to streamline the creation of high-quality Relative Elevation Models (REMs) for any river in the Continuous United States and Alaska. This software encompasses a web-based user interface that allows users to select an area of interest on a map, download Digital Elevation Models (DEMs) and National Hydrography Dataset (NHD) river centerline data, and compute REMs based on the gathered data.


What used to take hours of manual GIS work can now be done in minutes through this automated interface. The tool is designed to provide ease of use and efficiency for computing REMs without requiring access to proprietary geospatial software.


## **Key Features**

**Automatic Data Retrieval:** Automates the download, clipping, and mosaicking of USGS DEM tiles and NHD river centerline data.


**Interactive Map Selection:** Users can draw an area of interest (AOI) directly on an interactive map to scan for available resolutions and rivers.


**Manual Data Upload:** Supports user uploads for custom DEMs (Geotiff) and river centerlines (Shapefile, GeoPackage, GeoJSON), allowing for worldwide use.


**Automated REM Calculation:** Automatically samples elevations, smooths river profiles, and interpolates surfaces using Inverse Distance Weighting (IDW).


**Professional Outputs:** Produces georeferenced rasters (.tif), publication-ready images (.png) with 17 color schemes, and comprehensive metadata.


**Quality Assurance:** Includes built-in QA/QC steps where users can visually verify the alignment of river centerlines over DEM hillshades before processing.


## **System Requirements**
To run this software effectively, the following system specifications are recommended:

**Operating System:**

* **Windows 10/11**
* **macOS 10.14+**
* **Linux (Ubuntu 18.04+)**

**Hardware:** Systems newer than 2013 are recommended. High RAM usage is expected for high-resolution calculations (e.g., 1-meter DEMs over large areas).

**Software:**

* **Anaconda or Miniconda** (Required for package management).
* **Python 3.11 or newer** (Python 3.13 recommended).
* **Web Browser:** Chrome, Firefox, Safari, or Edge.

**Internet:** Required for automatic data downloads (AOI+NHD mode).


## **Installation Guide**
Follow these steps to install the necessary environment and dependencies.

### **1. Install Anaconda/Miniconda**
Download and install Anaconda or Miniconda to handle geospatial packages efficiently.

**Download link:** https://www.anaconda.com/download

### **2. Create a Conda Environment**
Open your Anaconda Prompt (Windows) or Terminal (macOS/Linux) and run the following command to create an isolated environment:

conda create -n REM_Env python=3.13 -y


**Note:** Replace "REM_Env" with your preferred environment name.

### **3. Activate the Environment**
Activate the environment before installing dependencies or running the software:

conda activate REM_Env


### **4. Install Dependencies**
Navigate to the directory containing the software folder using the `cd` command. There are three methods to install dependencies:

**Method 1: Automated Installation (Recommended)**
Run the included installation script, which detects conda availability and installs geospatial packages reliably:

python install_requirements.py


**Method 2: Manual Installation**
If the automated script fails, use the requirements file:

pip install -r requirements.txt


**Method 3: Manual Package Install**
If both methods fail, manually install core geospatial libraries using conda-forge:

conda install -c conda-forge geopandas rasterio shapely pyproj py3dep pynhd -y



## **Running the Software**
There are two distinct ways to operate the software.

### **Web Interface (Recommended)**
This method uses a browser-based point-and-click interface.

1.  **Activate your environment:** `conda activate REM_Env`
2.  **Navigate to the software folder:** `cd path/to/software`
3.  **Run the application:**

python run_app.py


The interface should open automatically in your browser at `http://localhost:8501`. If it does not, copy that URL into Chrome, Edge, or Firefox.

### **Terminal Interface (Advanced Users)**
This method uses a command-line workflow with step-by-step prompts.

1.  **Activate your environment and navigate to the software folder.**
2.  **Run the guided script:**

python run_rem_guided.py



## **Usage and Workflow**

### **1. Project Setup**
* **Project Folder:** Select a folder path where all inputs and outputs will be stored.
* **Run Mode:** Choose between "AOI+NHD (3DEP download)" for automated data retrieval or "Custom DEM & River" for manual file uploads.

### **2. Data Selection**
* **Automated Mode:**
    * Draw a polygon or rectangle on the interactive map.
    * Click "Scan AOI for Rivers & Resolutions".
    * Select a DEM Resolution (1m, 3m, 10m, 30m) and a River from the dropdown list.
* **Manual Mode:**
    * Upload a DEM (.tif or .tiff).
    * Upload a River Centerline (.shp, .gpkg, or .geojson).
    * *Note:* Shapefiles must include .shp, .shx, and .dbf files.

### **3. Processing Parameters**
Adjust the following parameters to control the REM calculation:
* **Spacing (meters):** Distance between elevation sample points (Default: 20m). Lower values (10m) provide maximum detail but slower processing. Higher values (50-100m) are faster but may miss small features.
* **K Neighbors (IDW):** Number of nearest river points for interpolation (Default: 8). Increase to 50-100 for smoother, publication-quality outputs.
* **Max REM (meters):** Optional cutoff height. Pixels above this value relative to the river are excluded to save memory.

**Note:** River elevations are sampled using perpendicular cross-sections (50m width for NHD data, 15m for user data) with 50th percentile quantile extraction to capture the channel thalweg. Smoothing window (adaptive based on river length) is automatically calculated for optimal results.

### **4. Visualization Settings**
* **Color Ramps:** Select from 17 schemes (e.g., FloodAlertRed, MagmaRamp, ESRIClassic).
* **Background:** Choose between Hillshade, Aerial, or None.
* **Transparency:** Adjust transparency for both the background (default 0.5) and the REM overlay (default 1.0).
* **Discrete Colors:** Toggle for distinct color bands instead of smooth gradients.

### **5. Execution**
Click **"Run REM Pipeline"**. The software will first generate a Quality Assurance (QA/QC) hillshade image. Verify the river centerline overlaps the channel correctly before confirming to proceed with the full calculation.


## **Outputs**
All outputs are saved in the designated Project folder:

* **REM.tif:** The final Relative Elevation Model raster (in meters, EPSG: 5070).
* **REM_[colorscheme]_[resolution].png:** High-quality visualization image(s).
* **REM_Project_Stats.txt:** Comprehensive metadata file documenting processing parameters, statistics, and coordinate systems.
* **hillshade_qa_qc.png:** The QA check image used to verify data alignment.
* **mosaic_clipped.tif:** The stitched and clipped DEM used for processing.
* **river_reprojected.gpkg:** The final river centerline data projected to EPSG: 5070.
* **hillshade.tif:** Hillshade raster generated from the source DEM.


## **Troubleshooting**

### **Common Issues**

**Map not loading / Blank Map**
Restart the web interface using `Ctrl+C` in the terminal and running `python run_app.py` again. Try clearing browser cache or using a different browser.

**Out of Memory / RAM Error**
High-resolution DEMs (1m) over large areas require significant RAM.
* **Solution:** Use a coarser resolution (e.g., 10m), reduce the AOI size, or set a "Max REM" value (e.g., 10m) to limit the calculation extent.

**DEM Download Failed**
This is often due to internet connection timeouts or USGS server availability.
* **Solution:** Check internet connection, wait 15 minutes and retry, or manually download the DEM from USGS Earth Explorer and use "Manual Data Upload".

**REM is all NoData (-999)**
This occurs if the river centerline does not overlap the DEM or if Coordinate Reference Systems (CRS) are mismatched.
* **Solution:** Verify overlap in GIS software. The software attempts to auto-reproject, but manual verification may be needed for custom files.

**Lines across PNG output**
Artifacts from the DEM or low interpolation neighbors.
* **Solution:** Increase "K Neighbors" to 50-100 to blend the surface better.

**"No Rivers Found in AOI"**
The selected area may only contain unnamed tributaries not in the main NHD dataset.
* **Solution:** Expand the AOI polygon or upload a manual river centerline.


## **Citing Your REM**
When publishing results generated by this software, please document the software version and key processing parameters found in your `REM_Project_Stats.txt` file.

**Example Citation:**
> "REMs were generated using the **Automated REM Generator (Kleinschmidt Associates)** with **10m USGS 3DEP DEM** data. River elevations were sampled at **20m spacing** using perpendicular cross-sections (50m width for NHD, 15m for user data) with 50th percentile quantile extraction to capture the thalweg. Profiles were smoothed using adaptive windowing with hydraulic monotonicity enforcement. Inverse distance weighting interpolation used **50 nearest neighbors**."


## **Contact**
For further assistance, please reach out to:
**Ethan Muhlestein**
**Ethan.Muhlestein@Kleinschmidtgroup.com**
