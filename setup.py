# setup.py
from setuptools import setup, find_packages

setup(
    name="geo_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rasterio",
        "numpy"
    ],
    author="LUP",
    description="Wiederverwendbare Geodatenfunktionen für Raster- und Vektorverarbeitung",
    python_requires=">=3.8",
)
