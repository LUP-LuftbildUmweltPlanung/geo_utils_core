import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compare_rasters(
    truth_path: str,
    model_path: str,
    resampling: str = "bilinear",
    verbose: bool = True,
):
    """
    Compare two rasters (truth vs model) over their overlapping area only.
    Handles different dimensions, resolutions, alignment, and CRS by
    reprojecting the model raster onto the truth raster's grid.
    NoData pixels (in either raster) and non-overlapping regions are ignored.

    Parameters
    ----------
    truth_path : str
        Path to raster with ground truth values.
    model_path : str
        Path to raster with model predictions.
    resampling : {"nearest","bilinear","cubic","average","mode"}, default "bilinear"
        Resampling method used when aligning the model raster to the truth grid.
        For continuous predictions, "bilinear" is usually appropriate.
    verbose : bool, default True
        Print metrics to stdout.

    Returns
    -------
    dict
        {"RMSE": float, "MAE": float, "R2": float, "n": int}
        where n is the number of overlapping valid pixels used.
    """
    # Map string to rasterio Resampling
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }
    if resampling not in resampling_map:
        raise ValueError(f"Unsupported resampling '{resampling}'. "
                         f"Choose from {list(resampling_map.keys())}")

    with rasterio.open(truth_path) as truth_ds, rasterio.open(model_path) as model_ds:
        # Read truth (band 1) and promote to float for NaN handling
        truth = truth_ds.read(1).astype("float64", copy=False)
        truth_nodata = truth_ds.nodata

        # Convert truth nodata to NaN
        if truth_nodata is not None:
            truth[truth == truth_nodata] = np.nan

        # Prepare destination array for reprojected model on truth grid
        model_reproj = np.full(truth_ds.shape, np.nan, dtype="float64")

        # Choose a numeric nodata placeholder for the reproject call
        # (reproject works best with numeric nodata; we'll convert to NaN afterward)
        src_nodata = model_ds.nodata
        if src_nodata is None:
            # pick a value unlikely to appear in real data
            src_nodata = -9.223372e18  # a large negative sentinel

        dst_fill_value = -9.223372e18  # sentinel for dst; will be turned into NaN

        # Reproject MODEL → TRUTH grid (CRS, transform, shape)
        reproject(
            source=rasterio.band(model_ds, 1),
            destination=model_reproj,
            src_transform=model_ds.transform,
            src_crs=model_ds.crs,
            src_nodata=src_nodata,
            dst_transform=truth_ds.transform,
            dst_crs=truth_ds.crs,
            dst_nodata=dst_fill_value,
            resampling=resampling_map[resampling],
        )

        # Convert sentinels and any model nodata to NaN
        model_reproj[model_reproj == dst_fill_value] = np.nan
        # (src_nodata should already be respected, but ensure anyway)
        # Note: after reprojection, exact equality to src_nodata is rare; this is just a safeguard.
        model_reproj[model_reproj == src_nodata] = np.nan

        # Build valid mask: overlapping, non-NaN pixels in both arrays
        valid = ~np.isnan(truth) & ~np.isnan(model_reproj)

        n = int(valid.sum())
        if n == 0:
            raise ValueError("No overlapping valid pixels to compare "
                             "(check extents, CRS, or NoData settings).")

        y_true = truth[valid]
        y_pred = model_reproj[valid]

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        if verbose:
            print(f"Used pixels (overlap & valid): {n}")
            print(f"RMSE: {rmse}")
            print(f"MAE:  {mae}")
            print(f"R²:   {r2}")

        return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": n}


# Example:
truth_path = r"C:\Users\frede\Desktop\Sealed\nrw_masked2.tif"
model_path = r"C:\Users\frede\Desktop\Sealed\germany_final.tif"
metrics = compare_rasters(truth_path, model_path, resampling="bilinear")
print(metrics)
