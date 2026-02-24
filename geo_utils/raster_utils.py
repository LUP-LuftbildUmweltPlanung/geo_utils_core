import os
import glob
import warnings
import rasterio
from pathlib import Path
import numpy as np
from osgeo import gdal, gdal_array
from tqdm import tqdm
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import transform as window_transform
from rasterio.enums import Resampling



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


def build_pyramids(
        input_folder
):
    """
    Builds pyramidsdirectly in file.

    :param input_path: str
        Path to the input folder, containing raster files.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # Check for TIFF files
            filepath = os.path.join(input_folder, filename)
            with rasterio.open(filepath, 'r+') as src:  # Open the file in read-write mode
                # Define pyramid levels to generate
                # overviews = [2, 4, 8, 16, 32]
                overviews = [4, 8, 16, 32]

                src.build_overviews(overviews, Resampling.nearest)
                src.update_tags(ns='rio_overview', resampling='nearest')
                print(f"Built pyramids for {filename}")


def mosaic_rasters_windowed(
    tiles,
    output_file,
    *,
    dtype_out=None,
    recursive=False,
    window_px=4096,
    nodata_value=0,
    show_progress=True,
    target_srs=None,
    resampling="near",
    build_overviews=True,
    overview_resampling="nearest"
):
    """
    Seam-safe, pixel-aligned windowed raster mosaicking.

    This function mosaics a large number of GeoTIFF tiles into a single raster
    using block-wise (windowed) processing to avoid memory issues.

    Parameters
    ----------
    tiles : str or list
        Folder path, glob pattern, or explicit list of GeoTIFF files.
    output_file : str or Path
        Output GeoTIFF mosaic path.
    dtype_out : GDAL data type, optional
        Output datatype. If None, uses datatype of first tile.
    recursive : bool
        Search subfolders for tiles.
    window_px : int
        Window size (in pixels) for block-wise processing.
    nodata_value : int or float
        Explicit NoData value (recommended: 0 for RGB/RGBI).
    show_progress : bool
        Show progress bars.
    target_srs : str or None
        Optional reprojection target CRS (e.g. "EPSG:25832").
    resampling : str
        Resampling method if reprojection is applied.
    build_overviews : bool
        Build internal overviews (pyramids) after mosaicking.
    """

    # ------------------------------------------------------------
    # Resolve tiles
    # ------------------------------------------------------------
    tiles = _resolve_tiles(tiles, recursive)
    if not tiles:
        raise FileNotFoundError("No raster tiles found.")

    # ------------------------------------------------------------
    # Read reference tile
    # ------------------------------------------------------------
    ref = gdal.Open(tiles[0], gdal.GA_ReadOnly)
    if ref is None:
        raise RuntimeError(f"Cannot open {tiles[0]}")

    gt0 = ref.GetGeoTransform()
    # gt = (
    #     gt[0],  # top-left X (origin X)
    #     gt[1],  # pixel width (X resolution)
    #     gt[2],  # row rotation (usually 0)
    #     gt[3],  # top-left Y (origin Y)
    #     gt[4],  # column rotation (usually 0)
    #     gt[5],  # pixel height (Y resolution, usually negative)
    # )
    src_srs = ref.GetProjection()
    xres = gt0[1]
    yres = gt0[5]
    bands = ref.RasterCount
    dtype_first = ref.GetRasterBand(1).DataType
    ref = None

    if yres >= 0:
        raise RuntimeError("Input rasters must be north-up.")

    if dtype_out is None:
        dtype_out = dtype_first

    out_np_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(dtype_out)

    # ------------------------------------------------------------
    # Scan full mosaic extent (world coordinates)
    # ------------------------------------------------------------
    ulx, uly, lrx, lry = [], [], [], []

    it = tiles if not show_progress else tqdm(tiles, desc="Scanning tiles", unit="tile")
    for p in it:
        ds = gdal.Open(p, gdal.GA_ReadOnly)
        if ds is None:
            warnings.warn(f"Cannot open {p}")
            continue

        gt = ds.GetGeoTransform()
        ulx.append(gt[0])
        uly.append(gt[3])
        lrx.append(gt[0] + ds.RasterXSize * gt[1])
        lry.append(gt[3] + ds.RasterYSize * gt[5])
        ds = None

    full_ulx = min(ulx)
    full_uly = max(uly)
    full_lrx = max(lrx)
    full_lry = min(lry)

    xsize = int(round((full_lrx - full_ulx) / xres))
    ysize = int(round((full_uly - full_lry) / abs(yres)))

    # ------------------------------------------------------------
    # Create output raster
    # ------------------------------------------------------------
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(
        str(output_file),
        xsize,
        ysize,
        bands,
        dtype_out,
        options=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "BIGTIFF=IF_SAFER",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
        ],
    )

    dst.SetGeoTransform((full_ulx, xres, 0.0, full_uly, 0.0, yres))
    dst.SetProjection(src_srs)

    for b in range(1, bands + 1):
        dst.GetRasterBand(b).SetNoDataValue(nodata_value)

    # ------------------------------------------------------------
    # Build pixel-based tile index (single grid snap)
    # ------------------------------------------------------------
    index = []
    for p in tiles:
        ds = gdal.Open(p, gdal.GA_ReadOnly)
        if ds is None:
            continue

        gt = ds.GetGeoTransform()
        xoff = int(round((gt[0] - full_ulx) / xres))
        yoff = int(round((full_uly - gt[3]) / abs(yres)))

        index.append({
            "path": p,
            "xoff": xoff,
            "yoff": yoff,
            "xsize": ds.RasterXSize,
            "ysize": ds.RasterYSize,
        })
        ds = None

    # ------------------------------------------------------------
    # Windowed mosaicking (GDAL-style iteration)
    # ------------------------------------------------------------
    y_range = range(0, ysize, window_px)
    x_range = range(0, xsize, window_px)

    if show_progress:
        y_range = tqdm(y_range, desc="Mosaicking", unit="row")

    for y0 in y_range:
        for x0 in x_range:
            h = min(window_px, ysize - y0)
            w = min(window_px, xsize - x0)

            buf = np.full((bands, h, w), nodata_value, dtype=out_np_dtype)

            for t in index:
                ix0 = max(x0, t["xoff"])
                iy0 = max(y0, t["yoff"])
                ix1 = min(x0 + w, t["xoff"] + t["xsize"])
                iy1 = min(y0 + h, t["yoff"] + t["ysize"])

                if ix0 >= ix1 or iy0 >= iy1:
                    continue

                wx = ix0 - x0
                wy = iy0 - y0
                tx = ix0 - t["xoff"]
                ty = iy0 - t["yoff"]
                nx = ix1 - ix0
                ny = iy1 - iy0

                ds = gdal.Open(t["path"], gdal.GA_ReadOnly)
                if ds is None:
                    continue

                arr = np.stack(
                    [ds.GetRasterBand(b + 1).ReadAsArray(tx, ty, nx, ny)
                     for b in range(bands)],
                    axis=0,
                )
                ds = None

                buf[:, wy:wy + ny, wx:wx + nx] = arr

            for b in range(bands):
                dst.GetRasterBand(b + 1).WriteArray(buf[b], xoff=x0, yoff=y0)

    dst.FlushCache()
    dst = None

    # ------------------------------------------------------------
    # Optional reprojection
    # ------------------------------------------------------------
    final_file = output_file
    if target_srs and target_srs != src_srs:
        reproj_file = output_file.with_suffix(".reproj.tif")
        gdal.Warp(
            str(reproj_file),
            str(output_file),
            dstSRS=target_srs,
            resampleAlg=resampling,
            multithread=True,
            creationOptions=[
                "TILED=YES",
                "COMPRESS=DEFLATE",
                "BIGTIFF=IF_SAFER",
            ],
        )
        final_file = reproj_file

    # ------------------------------------------------------------
    # Build overviews
    # ------------------------------------------------------------
    if build_overviews:
        with rasterio.open(final_file, "r+") as src:
            if not src.overviews(1):
                resamp = Resampling[overview_resampling]

                src.build_overviews(
                    [2, 4, 8, 16, 32, 64],
                    resamp
                )

                src.update_tags(
                    ns="rio_overview",
                    resampling=overview_resampling
                )

    if show_progress:
        print(f"✓ Final mosaic ready → {final_file}")


def _resolve_tiles(src, recursive=False):
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.is_dir():
            files = p.rglob("*.tif") if recursive else p.glob("*.tif")
            files = list(files) + list(
                p.rglob("*.tiff") if recursive else p.glob("*.tiff")
            )
        else:
            files = glob.glob(str(p))
    else:
        files = [Path(f) for f in src]

    return sorted(str(f) for f in files if Path(f).is_file())

# Example 1: simple mosaic, no reprojection
mosaic_rasters_windowed(
    tiles=r"\Path\to\img_tiles",
    output_file=r"\Path\to\output\folder\test.tif",
    nodata_value=65300,
    build_overviews=True,
    overview_resampling="nearest", # Use: nearest when RGB, RGBI / Use: average or bilinear when (DTM / DGM / DSM / NDOM) (continuous elevation data)
)

# Example 2: mosaic + reprojection
# mosaic_rasters_windowed(
#     tiles=r"\Path\to\img_tiles",
#     output_file=r"\Path\to\output\folder\test.tif",
#     nodata_value=-9999,
#     target_srs="EPSG:25832",
#     resampling="near",
#     build_overviews=True,
# )
