import os
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import transform as window_transform

def co_registration(
    parent_path,
    child_path,
    resampling_method,
    output_path=None,
    compression_method="ZSTD",
    tiled=True,
    blocksize=512,
    use_big_tiff=False,
    fallback_nodata=None
):
    """
    Co-registers two raster images by reprojecting the target raster (child)
    to match the reference raster (parent). The CRS, resolution, and extent
    from the reference raster are used, and the raster values from the target
    raster are interpolated.

    Nodata value is derived from the target raster if available. If not, it is
    set according to the datatype of the target raster, or provided explicitly
    via `fallback_nodata`.

    Parameters
    ----------
    parent_path : str
        Path to the reference raster (parent raster) that defines the CRS,
        resolution, and extent.
    child_path : str
        Path to the target raster (child raster) that will be reprojected
        to match the reference raster.
    resampling_method : str
        Interpolation method to use during reprojection.
        Possible values: ['nearest', 'bilinear', 'cubic', 'lanczos'].

        - 'nearest': Fast and simple. Suitable for categorical data
          (e.g., land use, classifications) as it does not interpolate new values.
        - 'bilinear': Suitable for continuous data (e.g., elevation models).
          Performs linear interpolation using the four nearest pixels, producing
          smoother output.
        - 'cubic': Smoother than bilinear, considering the 16 nearest pixels.
          Ideal for high-resolution images with soft transitions.
        - 'lanczos': Very accurate and high-quality interpolation, especially
          for large resampling factors. Good for preserving detail and sharp edges.
    output_path : str, optional
        Path to save the reprojected raster. If None, a new filename with
        "_coregistered" suffix will be created next to the child raster.
    compression_method : str, optional
        Compression algorithm for the output raster (default: "LZW").
    tiled : bool, optional
        Whether to write the output as tiled GeoTIFF (default: True).
    blocksize : int, optional
        Tile/block size in pixels for tiled GeoTIFFs (default: 512).
    use_big_tiff : bool, optional
        Whether to allow BigTIFF format if needed (default: True).
    fallback_nodata : int or float, optional
        Custom fallback NoData value, if child raster has none
        and automatic inference is not suitable.

    Returns
    -------
    str
        Path of the output raster (the reprojected raster).
    """

    if resampling_method not in ['nearest', 'bilinear', 'cubic', 'lanczos']:
        raise ValueError(f"Invalid resampling method: {resampling_method}")

    resampling_enum = getattr(Resampling, resampling_method)

    with rasterio.open(parent_path) as parent, rasterio.open(child_path) as child:
        if parent.crs is None or child.crs is None:
            raise ValueError("Both parent and child rasters must have a valid CRS.")

        # Base profile from parent, adapt band count and dtype from child
        dst_profile = parent.profile.copy()
        dst_profile.update({
            "driver": "GTiff",
            "count": child.count,
            "dtype": child.dtypes[0],
            "compress": compression_method,
            "transform": parent.transform,
            "height": parent.height,
            "width": parent.width
        })

        # Derive NoData value
        src_dtype = child.dtypes[0]
        src_nodata = child.nodata
        if src_nodata is not None:
            nodata_val = src_nodata
        elif fallback_nodata is not None:
            nodata_val = fallback_nodata
        else:
            if src_dtype in ['int8', 'byte']:
                nodata_val = -128
            elif src_dtype == 'uint8':
                nodata_val = 255
            elif src_dtype == 'int16':
                nodata_val = -32768
            elif src_dtype == 'uint16':
                nodata_val = 65535
            elif src_dtype == 'int32':
                nodata_val = -2147483648
            elif src_dtype == 'uint32':
                nodata_val = 4294967295
            elif src_dtype in ['float32', 'float64']:
                nodata_val = -9999.0
            else:
                raise ValueError(
                    f"Unknown dtype {src_dtype}; please provide fallback_nodata."
                )

        dst_profile.update({"nodata": nodata_val})

        if tiled:
            dst_profile.update({"tiled": True, "blockxsize": blocksize, "blockysize": blocksize})
        if use_big_tiff:
            dst_profile.update({"BIGTIFF": "IF_SAFER"})
        if dst_profile["compress"] in ("LZW", "ZSTD"):
            dst_profile.update({"predictor": 3 if 'float' in src_dtype else 2})

        if output_path is None:
            base, ext = os.path.splitext(child_path)
            output_path = f"{base}_coregistered{ext}"

        if ('int' in src_dtype or 'uint' in src_dtype) and resampling_method != 'nearest':
            print("Warning: Categorical/integer data should normally be resampled with 'nearest'.")

        env_opts = {"GDAL_NUM_THREADS": "ALL_CPUS"}
        with rasterio.Env(**env_opts):
            with rasterio.open(output_path, "w", **dst_profile) as dst:
                for band_idx in range(1, child.count + 1):
                    reproject(
                        source=rasterio.band(child, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=child.transform,
                        src_crs=child.crs,
                        dst_transform=dst.transform,
                        dst_crs=parent.crs,
                        resampling=resampling_enum,
                        src_nodata=src_nodata,
                        dst_nodata=nodata_val,
                        num_threads=0
                    )

    return output_path


def compress_raster(
        input_path,
        output_path=None,
        compression_method="ZSTD"
):
    """
    Compress a raster and save it to a new path.
    The raster is processed block by block to reduce memory usage.

    :param input_path: str
        Path to the input raster.
    :param output_path: str
        Path for the compressed output file.
    :param compression_method: str
        Compression method. Possible values: ['DEFLATE', 'LZW', 'ZSTD', 'JPEG', 'PACKBITS'].
        **DEFLATE**: Lossless compression, commonly used for reducing file size without quality loss.
        **LZW**: Lossless compression, often used for TIFF files, effective for images with large areas of uniform color.
        **ZSTD**: Lossless compression method that provides faster read/write performance and smaller file sizes compared to traditional options like LZW
        **JPEG**: Lossy compression, efficient for natural images but may reduce quality.
        **PACKBITS**: Run-length encoding, efficient for simple raster data with repeating values.
    :return: str
        Path of the compressed raster.
    """
    # Open the input raster
    with rasterio.open(input_path) as src:
        # Create the target profile (copy all properties except values)
        profile = src.profile
        profile.update({
            "compress": compression_method  # Apply compression method
        })

        # If no output path is provided, create a default one
        if output_path is None:
            basename, extension = os.path.splitext(input_path)
            output_path = basename + "_compressed" + extension

        # Write the compressed raster to the output file
        with rasterio.open(output_path, "w", **profile) as dst:
            # Iterate over all bands
            for i in range(1, src.count + 1):
                # Read and write the raster block by block
                for ji, window in src.block_windows(i):  # read block by block
                    data = src.read(i, window=window)    # Read the block
                    dst.write(data, i, window=window)    # Write the block

    return output_path
