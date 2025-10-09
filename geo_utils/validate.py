from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from geo_utils.raster_utils import *

def compare_rasters(
    truth_path: str,
    model_path: str,
    resampling: str = "nearest",
    verbose: bool = True,
):
    """
    Compare two rasters (truth vs model) over their overlapping pixels only.
    Handles different dimensions, resolutions, alignment, and CRS by
    reprojecting the model raster onto the truth raster's grid.
    NoData pixels (in either raster) and non-overlapping regions are ignored.

    Parameters
    ----------
    truth_path : str
        Path to raster with ground truth values.
    model_path : str
        Path to raster with model predictions.
    resampling : {"nearest","bilinear","cubic","average","mode"}, default "nearest"
        Resampling method used when aligning the model raster to the truth grid.
        For continuous data, "bilinear" and for categorical data, "nearest" is usually appropriate.
    verbose : bool, default True
        Print metrics to stdout.

    Returns
    -------
    dict
        {"RMSE": float, "MAE": float, "R2": float, "n": int}
        where n is the number of overlapping valid pixels used.
    """
    # Check if rasters are aligned
    with rasterio.open(truth_path) as truth_ds, rasterio.open(model_path) as model_ds:
        same_shape = truth_ds.shape == model_ds.shape
        same_crs = truth_ds.crs == model_ds.crs
        same_transform = truth_ds.transform.almost_equals(model_ds.transform, precision=6)

        if same_shape and same_crs and same_transform:
            model_reproj_path = model_path
            print("Rasters are already perfectly aligned — skipping co-registration.")
        else:
            print("Rasters differ in CRS, shape, or alignment — performing co-registration.")
            model_reproj_path = co_registration(truth_path, model_path, resampling)

    with rasterio.open(truth_path) as truth_ds, rasterio.open(model_reproj_path) as model_ds:
        # Read truth (band 1) and promote to float for NaN handling
        truth = truth_ds.read(1).astype("float32", copy=False)
        truth_nodata = truth_ds.nodata
        # Convert truth nodata to NaN
        if truth_nodata is not None:
            truth[truth == truth_nodata] = np.nan

        model = model_ds.read(1).astype("float32", copy=False)
        model_nodata = model_ds.nodata
        # Convert model nodata to NaN
        if model_nodata is not None:
            model[model == model_nodata] = np.nan

        # Build valid mask: overlapping, non-NaN pixels in both arrays
        valid = ~np.isnan(truth) & ~np.isnan(model)

        # Optional: add mask to only compare pixels below or above a certain threshold
        #valid = valid & (truth >= 90)

        n = int(valid.sum())
        if n == 0:
            raise ValueError("No overlapping valid pixels to compare "
                             "(check extents, CRS, or NoData settings).")

        y_true = truth[valid]
        y_pred = model[valid]

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        if verbose:
            print(f"Used pixels (overlap & valid): {n}")
            print(f"RMSE: {rmse}")
            print(f"MAE:  {mae}")
            print(f"R²:   {r2}")

        return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": n}