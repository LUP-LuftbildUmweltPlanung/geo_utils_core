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
            if src_dtype == 'int8' or src_dtype == 'byte':
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
            elif src_dtype == 'float32' or 'float64':
                nodata_val = -9999.0
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
            "nodata": nodata_val,
            "compress": "LZW"
        }

        if output_path is None:
            basename, extension = os.path.splitext(child_path)
            output_path = basename + "_coregistered" + extension

        # Resampling Methode definieren
        resampling_method = getattr(Resampling, resampling_method)

        # Create new raster
        with rasterio.open(output_path, "w", **dst_profile) as dst:
            # Process the image block by block
            for ji, window in child.block_windows(1):  # block-wise iteration over the raster
                data = child.read(1, window=window)  # Read the block data
                destination_data = np.empty_like(data)  # Create an empty array for the reprojected data
                reproject(
                    source=data,
                    destination=destination_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method,
                    src_nodata=src_nodata,
                    dst_nodata=nodata_val
                )
                dst.write(destination_data, 1, window=window)  # Write the processed block

    return output_path


def compress_raster(input_path, output_path=None, compression_method="LZW"):
    """
    Komprimiert ein Raster und speichert es unter einem neuen Pfad, Blockweise verarbeitet, um den Speicherbedarf zu verringern.

    :param input_path: str
        Pfad zum Eingabe-Raster.
    :param output_path: str
        Pfad für das komprimierte Ausgabedatei.
    :param compression_method: str
        die Kompressionsmethode. Mögliche Werte: ['DEFLATE', 'LZW', 'JPEG', 'PACKBITS'].
        **DEFLATE**: Lossless compression, commonly used for reducing file size without quality loss.
        **LZW**: Lossless compression, used for TIFF files, effective for images with large areas of uniform color.
    :return: str
        der Pfad des komprimierten Rasters.
    """
    # Öffne das Eingabe-Raster
    with rasterio.open(input_path) as src:
        # Erstelle das Zielprofil (kopiere alle Eigenschaften außer den Werten)
        profile = src.profile
        profile.update({
            "compress": compression_method  # Kompressionsmethode anwenden
        })

        # Wenn kein Ausgabepfad angegeben ist, erstelle einen standardmäßigen Pfad
        if output_path is None:
            basename, extension = os.path.splitext(input_path)
            output_path = basename + "_compressed" + extension

        # Schreibe das komprimierte Raster in die Ausgabedatei
        with rasterio.open(output_path, "w", **profile) as dst:
            # Iteriere über alle Bänder
            for i in range(1, src.count + 1):
                # Lese und schreibe das Raster blockweise
                for ji, window in src.block_windows(i):  # blockweise lesen
                    data = src.read(i, window=window)  # Lese den Block
                    dst.write(data, i, window=window)  # Schreibe den Block

    return output_path
