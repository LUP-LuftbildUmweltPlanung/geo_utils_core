# -*- coding: utf-8 -*-
"""
FORCE-aligned polygon percent-cover workflow without GeoWombat.

This replaces the GeoWombat-dependent aggregation in
`Vector_To_Raster_Damage_Analysis.py` with a lightweight Rasterio/GeoPandas
approach. It rasterizes polygons directly onto the FORCE 10 m grid using
sub-pixel sampling to derive percent overlap per pixel.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import rasterio
from affine import Affine
from rasterio import features, windows
from shapely.geometry import box

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

# Input polygon dataset (Shapefile, GeoPackage, etc.). Coordinates can be in
# any CRS; they are reprojected to EPSG:3035 internally.
shapefile_path = "/rvt_mount/3DTests/data/deadwood/test_filter_200.gpkg"

# Output GeoTIFF containing percent cover per FORCE 10 m pixel.
percent_cover_out_path = "/rvt_mount/3DTests/data/dead_trees/test_200_raster.tif"

# Optional custom boundary (minx, miny, maxx, maxy) in EPSG:3035 to limit the
# processing extent. Set to None to use the polygon extent.
custom_boundary = None

# Sub-pixel factor used to approximate fractional area within a 10 m pixel.
# Each FORCE pixel is subdivided into factor^2 subpixels; higher values mean
# better accuracy and higher memory use.
subpixel_factor = 8

# FORCE grid definition (EPSG:3035). Expand if operating outside Germany.
force_bounds = (4016026.363042, 2654919.607965, 4676026.363042001, 3554919.607965)
force_resolution = 10.0  # meters
force_crs = "EPSG:3035"

# -----------------------------------------------------------------------------


def clamp_to_force(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Restrict requested bounds to the FORCE domain."""
    left = max(bounds[0], force_bounds[0])
    bottom = max(bounds[1], force_bounds[1])
    right = min(bounds[2], force_bounds[2])
    top = min(bounds[3], force_bounds[3])
    if left >= right or bottom >= top:
        raise ValueError("Requested area does not intersect the FORCE grid.")
    return left, bottom, right, top


def force_window(bounds: tuple[float, float, float, float]) -> tuple[windows.Window, Affine]:
    """Return a FORCE-grid window and transform covering the requested bounds."""
    force_transform = Affine(
        force_resolution,
        0.0,
        force_bounds[0],
        0.0,
        -force_resolution,
        force_bounds[3],
    )
    win = windows.from_bounds(*bounds, transform=force_transform)
    win = win.round_offsets(op=math.floor).round_lengths(op=math.ceil)
    return win, force_transform


def rasterize_percent_cover(geoms, base_transform: Affine, height: int, width: int) -> np.ndarray:
    """Rasterize polygons onto subdivisions of the FORCE grid and return percent cover."""
    factor = max(1, int(subpixel_factor))
    sub_transform = base_transform * Affine.scale(1.0 / factor, 1.0 / factor)
    sub_shape = (height * factor, width * factor)

    shapes = ((geom.__geo_interface__, 1) for geom in geoms)
    coverage = features.rasterize(
        shapes,
        out_shape=sub_shape,
        transform=sub_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    coverage = coverage.reshape(height, factor, width, factor)
    percent = coverage.mean(axis=(1, 3)).astype("float32") * 100.0
    return percent


def main():
    """Load polygons, snap to FORCE bounds, and export percent cover."""
    vector_path = Path(shapefile_path)
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector dataset not found: {vector_path}")

    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError("Vector dataset contains no features.")

    gdf = gdf.to_crs(force_crs)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError("Vector dataset has no valid geometries after cleaning.")

    bounds = custom_boundary if custom_boundary else gdf.total_bounds
    target_bounds = clamp_to_force(bounds)
    clip_geom = box(*target_bounds)
    gdf = gpd.clip(gdf, clip_geom)
    if gdf.empty:
        raise ValueError("Clipped area contains no polygons to rasterize.")

    window, force_transform = force_window(target_bounds)
    out_transform = windows.transform(window, force_transform)
    height = int(window.height)
    width = int(window.width)
    if height <= 0 or width <= 0:
        raise ValueError("Computed FORCE window has invalid dimensions.")

    percent_cover = rasterize_percent_cover(gdf.geometry, out_transform, height, width)

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": force_crs,
        "transform": out_transform,
        "nodata": -9999.0,
        "compress": "lzw",
    }

    output_path = Path(percent_cover_out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(percent_cover, 1)

    print(f"Percent-cover raster saved to {output_path}")


if __name__ == "__main__":
    main()
