from setuptools import setup, find_packages

setup(
    name="geo_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rasterio>=1.3.0,<2.0.0",
        "geopandas>=1.0.0,<2.0.0",
        "numpy>=1.26.1,<2.0.0",
        "shapely>=2.0.0,<3.0.0"
    ],
    author="LUP",
    description="Reusable geospatial utility functions for raster and vector data processing",
    python_requires=">=3.8",
)
