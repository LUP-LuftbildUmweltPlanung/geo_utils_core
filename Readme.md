## Description

This package includes multiple geoprocessing tools. It contains only basic tools that are commonly used and needed repeatedly.


## Installing

There are two ways to use the repo:

1. As a stand alone repo. Therefore, follow these steps:
   * git clone https://github.com/LUP-LuftbildUmweltPlanung/geo_utils_core
   * cd path/to/repo
   * conda env create -f environment.yaml
   * conda activate geo-utils-core

   If using Linux Micomamba should be used insted of conda. To install Micromamba on Linux: 
   * curl -Ls https://micro.mamba.pm/install.sh | bash
   * source ~/.bashrc
   
   Then install the environment:
   * cd path/to/repo
   * micromamba create -f environment.yaml
   * micromamba activate geo-utils-core

2. As a package useable in a python project. 
   * open anaconda prompt and 
     * create an environment with conda create -n geo-utils-core conda-forge python=3.10 numpy pandas scikit-learn gdal=3.8.4 rasterio=1.3.9 pyproj proj proj-data fiona shapely geopandas OR
     * activate environment and install missing dependencies 
   * activate environment
   * pip install git+https://github.com/LUP-LuftbildUmweltPlanung/geo_utils_core
   * After installation, the functions can be accessed by: from geo_utils.raster_utils import *

## Current Functions

### raster_utils
* co_registration: Co-registers two raster images by reprojecting the target raster (child) to match the reference raster (parent)
* compress_raster: Compresses raster to reduce file size
* build_pyramids: Builds pyramids in file
* mosaic_rasters_windowed: mosaics a large number of GeoTIFF tiles into a single raster using block-wise (windowed) processing to avoid memory issues

### vector_utils
* rasterize_vector: Rasterize a vector layer to a (multi-band) GeoTIFF
* count_features:  Count the number of features in a Shapefile or GeoPackage. If an attribute is provided, counts are grouped by the unique values of that attribute.

### spatial_utils
* spatial_thinning: removes points that are closer than the given minimum distance
* compute_target_sample_size: Computes target sample sizes per class and generates a simplified GRTS-ready CSV table.
* run_grts: Run the GRTS (Generalized Random Tessellation Stratified) sampling method using an R script.
* run_blockcv: Run the Block Cross-Validation (BlockCV) method using an R script.

### sample_training_points
* Generate reference points over a raster and sample selected band values.
* Generate reference points over a polygon vector file.

### validate
* compare_rasters: Compare two rasters (truth vs model) over their overlapping area only.

### fractional_cover
stand alone script that computes the frational cover of a vector file within a force pixel (10 m).

## Adding/changing Functions

Everyone is welcome to contribute, update, or improve the current functions. When adding new functions, please ensure that they follow the uniform structure of existing functions, including a detailed description of the function and parameters at the beginning. Please also add the function and a short description in the readme file. 

If functions are changed ensure to run pytest to assure that the function works correct. 

## Authors

LUP GmbH

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
