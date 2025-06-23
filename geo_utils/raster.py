import os
import rasterio
from rasterio.warp import reproject, Resampling


def co_registration(parent_path, child_path, output_path):

    # Open reference raster (Parent)
    with rasterio.open(parent_path) as parent:
        dst_crs = parent.crs
        dst_transform = parent.transform
        dst_shape = (parent.height, parent.width)

    # Open target raster (Child)
    with (rasterio.open(child_path) as child):
        src_crs = child.crs
        src_transform = child.transform
        src_dtype = child.dtypes[0]
        src_nodata = child.nodata

        # copy nodata-value vom child or define new one if not set
        if child.nodata is not None:
            nodata_val = child.nodata
        else:
            # dynamic setting of nodata-value depending on datatype
            if src_dtype == 'int8':
                nodata_val = -128
            elif src_dtype == 'uint8':
                nodata_val = 255
            elif src_dtype == 'float32':
                nodata_val = -9999
            else:
                nodata_val = None  # for all other types (can be changed)

        # Create new raster profile
        dst_profile = {
            "driver": "GTiff",
            "height": dst_shape[0],
            "width": dst_shape[1],
            "count": 1,
            "dtype": src_dtype,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": nodata_val
        }

        if output_path is None:
            basename, extension = os.path.splitext(child_path)
            output_path = basename + "_coregistered" + extension

        # Create new raster
        with rasterio.open(output_path, "w", **dst_profile) as dst:
            reproject(
                source=rasterio.band(child, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                src_nodata=src_nodata,
                dst_nodata=nodata_val
            )

    return output_path
