import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def co_registration(parent_path, child_path, resampling_method, output_path=None):
    """
    Co-registers two raster images by reprojecting the target raster (child) to match the reference raster (parent).
    The CRS and extent from the reference raster are used, and the raster values from the target raster are interpolated.
    Nodata-value is derived by the target raster or if None it is set according to datatype of target raster.

    :param parent_path: str
        Path to the reference raster (parent raster) that defines the CRS and extent.
    :param child_path: str
        Path to the target raster (child raster) that will be reprojected to match the reference raster.
    :param resampling_method: str
        Interpolation method to use during re-projection. Possible values are ['nearest', 'bilinear', 'cubic', 'lanczos'].
        **nearest**: Fast and simple. Suitable for categorical data (e.g., land use, classifications), as it doesn’t interpolate new values.
        **bilinear**: Suitable for continuous data (e.g., elevation models), as it performs linear interpolation using the four nearest pixels, resulting in smoother output.
        **cubic**: Provides smoother results than bilinear by considering the 16 nearest pixels. Ideal for high-resolution images and when a soft transition is needed.
        **lanczos**: Very accurate and high-quality interpolation, especially for large resampling factors. Suitable for images where details and edges should remain sharp.
    :param output_path: str
        Path to save the reprojected output raster.

    :return: str
        The path of the output raster (the reprojected raster).

    """

    if resampling_method not in ['nearest', 'bilinear', 'cubic', 'lanczos']:
        raise ValueError(f"Invalid resampling method: {resampling_method}")

    with rasterio.open(parent_path) as parent:
        dst_crs = parent.crs
        dst_transform = parent.transform
        dst_shape = (parent.height, parent.width)
        dst_profile = parent.profile.copy()
        dst_profile.update({
            "driver": "GTiff",
            "count": 1,
            "compress": "LZW"
        })

        with rasterio.open(child_path) as child:
            src_crs = child.crs
            src_transform = child.transform
            src_dtype = child.dtypes[0]
            src_nodata = child.nodata

            # Nodata ableiten
            if src_nodata is not None:
                nodata_val = src_nodata
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
                    nodata_val = None

            dst_profile.update({
                "dtype": src_dtype,
                "nodata": nodata_val
            })

            if output_path is None:
                basename, extension = os.path.splitext(child_path)
                output_path = basename + "_coregistered" + extension

            resampling_enum = getattr(Resampling, resampling_method)

            with rasterio.open(output_path, "w", **dst_profile) as dst:
                # Blockweise arbeiten: entscheidend ist das Window-Transform!
                for ji, window in dst.block_windows(1):
                    window_shape = (window.height, window.width)
                    dest_array = np.full(window_shape, nodata_val, dtype=src_dtype)

                    # *** WICHTIG: Transform des aktuellen Windows berechnen ***
                    win_transform = rasterio.windows.transform(window, dst_transform)

                    reproject(
                        source=rasterio.band(child, 1),
                        destination=dest_array,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=win_transform,
                        dst_crs=dst_crs,
                        resampling=resampling_enum,
                        src_nodata=src_nodata,
                        dst_nodata=nodata_val
                    )

                    dst.write(dest_array, 1, window=window)

    return output_path


def compress_raster(input_path, output_path=None, compression_method="LZW"):
    """
    Compress a raster and save it to a new path.
    The raster is processed block by block to reduce memory usage.

    :param input_path: str
        Path to the input raster.
    :param output_path: str
        Path for the compressed output file.
    :param compression_method: str
        Compression method. Possible values: ['DEFLATE', 'LZW', 'JPEG', 'PACKBITS'].
        **DEFLATE**: Lossless compression, commonly used for reducing file size without quality loss.
        **LZW**: Lossless compression, often used for TIFF files, effective for images with large areas of uniform color.
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
