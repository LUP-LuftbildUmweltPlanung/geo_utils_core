import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

# Eingabepfade
raster_a_path = r"Y:\MnD\data\GroundTruth\Sealed\hamburg_regressio.tif"
raster_b_path = r"Y:\MnD\data\GroundTruth\Sealed\class_reg_70.tif"
out_path = r"Y:\MnD\data\GroundTruth\Sealed\class_reg_70.tif"

with rasterio.open(raster_a_path) as A, rasterio.open(raster_b_path) as B:
    # B auf A ausrichten, falls nötig
    same_grid = (A.crs==B.crs and A.transform==B.transform and A.width==B.width and A.height==B.height)
    if not same_grid:
        b_aligned = np.empty((B.count, A.height, A.width), dtype=np.float32)
        for i in range(B.count):
            reproject(
                rasterio.band(B, i+1), b_aligned[i],
                src_transform=B.transform, src_crs=B.crs,
                dst_transform=A.transform, dst_crs=A.crs,
                dst_width=A.width, dst_height=A.height,
                resampling=Resampling.nearest
            )
    else:
        b_aligned = B.read().astype(np.float32)

    # robuste NoData-Maske aus A
    m = (A.dataset_mask()==0)
    a1 = A.read(1, masked=False)
    if A.nodata is not None and not (isinstance(A.nodata,float) and np.isnan(A.nodata)):
        m |= (a1 == A.nodata)
    if np.issubdtype(a1.dtype, np.floating):
        m |= np.isnan(a1)

    # NaN als NoData
    profile = A.profile
    profile.update(count=B.count, dtype="float32", nodata=np.nan)

    # Maske anwenden
    for i in range(b_aligned.shape[0]):
        band = b_aligned[i]
        band[m] = np.nan
        b_aligned[i] = band

with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(b_aligned)