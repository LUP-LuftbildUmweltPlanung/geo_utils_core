## Description

This package includes multiple geoprocessing tools. It contains only basic tools that are commonly used and needed repeatedly.


## Installing

There are two ways to use the repo:

1. As a stand alone repo. Therefore, follow these steps:
   * git clone https://github.com/LUP-LuftbildUmweltPlanung/geo_utils_core
   * conda create -n geo-utils-core -c conda-forge python=3.10 gdal=3.8 rasterio=1.3.9 fiona=1.9 geopandas shapely
   * conda activate geo-utils-core
   * conda install -c conda-forge pytest pytest-cov _Only neccessary if you want to develop/change functions_
   * pip install -e .[dev] --no-deps _Only neccessary if you want to develop/change functions_

2. As a package useable in a python project. Notice: Systemlibs GDAL, PROJ and GEOS need to be installed!
   * The package can be easily added to any project by running: pip install git+https://github.com/LUP-LuftbildUmweltPlanung/geo_utils_core
   * After installation, the functions can be accessed for example by: from geo_utils.raster_utils *

## Current Functions

### raster_utils
* co_registration: Co-registers two raster images by reprojecting the target raster (child) to match the reference raster (parent)
* compress_raster: Compresses raster to reduce file size

### vector_utils
* rasterize_vector: Rasterize a vector layer to a (multi-band) GeoTIFF

### sample_training_points
* Generate reference points over a raster and sample selected band values.

### validadte
compare_rasters: Compare two rasters (truth vs model) over their overlapping area only.

## Adding/changing Functions

Everyone is welcome to contribute, update, or improve the current functions. When adding new functions, please ensure that they follow the uniform structure of existing functions, including a detailed description of the function and parameters at the beginning. Please also add the function and a short description in the readme file. 

If functions are changed ensure to run pytest to assure that the function works correct. 

## Authors

LUP GmbH

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
