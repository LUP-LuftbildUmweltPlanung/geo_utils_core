import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import transform as window_transform, bounds as window_bounds
from shapely.geometry import box
from typing import List, Optional, Union

def rasterize_vector(
    input_path: str,
    resolution: float,
    attribute: Optional[Union[str, List[str]]] = None,
    datatype: Optional[str] = None,          # 'uint8' | 'int32' | 'float32'
    nodata_value: Optional[Union[int,float]] = None,
    crs: Optional[str] = None,
    output_path: Optional[str] = None,
    # block/tile controls
    block_size: int = 512,                   # GTiff tile size (blockxsize=blockysize)
    tiled: bool = True,
    compress: Optional[str] = None,          # e.g., 'DEFLATE', 'LZW'
):
    """
    Rasterize a (potentially large) vector layer to a (multi-band) GeoTIFF,
    writing block-by-block to keep memory usage low.

    - If `attribute` is a string → single-band raster.
    - If `attribute` is a list of strings → one band per attribute.
    - If `attribute` is None → constant value 1 (single band).

    Parameters
    ----------
    input_path : str
        Path to Shapefile/GeoPackage.
    resolution : float
        Target pixel size in layer CRS units (e.g., meters).
    attribute : str | List[str] | None
        Attribute(s) to burn into raster.
    datatype : str | None
        'uint8' | 'int32' | 'float32'. Inferred if None.
    nodata_value : number | None
        NoData value; defaults by dtype if None.
    crs : str | None
        Target CRS for output; defaults to vector CRS.
    output_path : str | None
        Output GeoTIFF path. Defaults to `<input>_rasterized.tif`.
    block_size : int
        Tile size for GTiff blocks (both x and y).
    tiled : bool
        Use tiled GeoTIFF.
    compress : str | None
        GDAL compression (e.g., 'DEFLATE', 'LZW').

    Returns
    -------
    str
        Path to the created raster.
    """
    # --- Load vector ---
    gdf = gpd.read_file(input_path)

    # Ensure CRS
    if crs is None:
        crs = gdf.crs
    else:
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)

    # Attributes handling
    created_const = False
    if attribute is None:
        gdf["_const"] = 1
        attributes = ["_const"]
        created_const = True
    elif isinstance(attribute, str):
        attributes = [attribute]
    else:
        attributes = list(attribute)

    # Dtype / nodata defaults (infer from first attribute if needed)
    if datatype is None and nodata_value is None:
        sample_dtype = gdf[attributes[0]].dtype
        if np.issubdtype(sample_dtype, np.unsignedinteger):
            datatype, nodata_value = "uint8", 255
        elif np.issubdtype(sample_dtype, np.integer):
            datatype, nodata_value = "int32", -1
        else:
            datatype, nodata_value = "float32", -9999.0
    elif datatype is None and nodata_value is not None:
        # Safe fallback if only nodata provided
        datatype = "float32"
    elif datatype is not None and nodata_value is None:
        if datatype == "uint8":
            nodata_value = 255
        elif datatype == "int32":
            nodata_value = -1
        elif datatype == "float32":
            nodata_value = -9999.0
        else:
            raise ValueError("datatype must be one of: 'uint8', 'int32', 'float32'.")

    # Extent & dimensions
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    # Output path
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_rasterized.tif"

    # Profile
    profile = {
        "driver": "GTiff",
        "count": len(attributes),
        "dtype": datatype,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": nodata_value,
        "tiled": tiled,
        "blockxsize": block_size if tiled else None,
        "blockysize": block_size if tiled else None,
    }
    if compress:
        profile["compress"] = compress

    # Spatial index to get only features intersecting a window
    sindex = gdf.sindex

    # Create and write block-wise
    with rasterio.open(output_path, "w", **profile) as dst:
        # Optional: band descriptions = attribute names
        for b, attr in enumerate(attributes, start=1):
            try:
                dst.set_band_description(b, str(attr))
            except Exception:
                pass

        # Iterate over blocks of band 1 (same tiling across bands)
        for _, window in dst.block_windows(1):
            # Bounds of this window in map coords
            w_bounds = window_bounds(window, transform)
            w_poly = box(*w_bounds)

            # Find candidate features intersecting this window
            idxs = list(sindex.query(w_poly))
            if not idxs:
                # Nothing intersects → write fill blocks for all bands
                for band in range(1, len(attributes) + 1):
                    dst.write(
                        np.full((window.height, window.width), nodata_value, dtype=datatype),
                        band,
                        window=window,
                    )
                continue

            # Subset GeoDataFrame for this window
            sub = gdf.iloc[idxs]

            # Window-specific transform
            w_transform = window_transform(window, transform)

            # For each band (attribute), rasterize ONLY intersecting geometries
            for band_idx, attr in enumerate(attributes, start=1):
                shapes = ((geom, val) for geom, val in zip(sub.geometry, sub[attr]))
                arr = rasterize(
                    shapes=shapes,
                    out_shape=(window.height, window.width),
                    transform=w_transform,
                    fill=nodata_value,
                    dtype=datatype,
                )
                dst.write(arr, band_idx, window=window)

    # Clean up temp const column if we added it (no file side-effect, just clarity)
    if created_const and "_const" in gdf.columns:
        gdf.drop(columns=["_const"], inplace=True)

    return output_path


def count_features(
        input_path: str,
        attribute: str = None
):
    """
    Count the number of features in a Shapefile or GeoPackage.
    If an attribute is provided, counts are grouped by the unique values
    of that attribute. Otherwise, the total number of features is returned.

    :param input_path: str
        Path to the Shapefile or GeoPackage.
    :param attribute: str, optional
        Attribute to group the counts by. If not provided, only the total
        number of features is counted.
    :return: dict or int
        - If `attribute` is set: dictionary with counts per attribute value.
        - Otherwise: single integer representing the total number of features.
    """

    # Load Shapefile or GeoPackage
    gdf = gpd.read_file(input_path)

    # If no attribute is given, count total features
    if attribute is None:
        count = len(gdf)
        print(f"Total number of features: {count}")
        return count

    # If attribute exists, count features grouped by attribute values
    if attribute in gdf.columns:
        counts = gdf[attribute].value_counts().to_dict()
        print(f"Number of features per '{attribute}':")
        for value, count in counts.items():
            print(f"  Value '{value}': {count} features")
        return counts
    else:
        raise ValueError(f"Attribute '{attribute}' not found in the input file.")
