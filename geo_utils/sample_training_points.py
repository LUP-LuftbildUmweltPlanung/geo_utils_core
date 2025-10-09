import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from typing import List, Union, Tuple, Optional, Literal
from math import hypot

def generate_points_from_raster(
    raster_path: str,
    bands: Optional[List[Union[int, str, Tuple[int, str]]]] = None,
    output_path: Optional[str] = None,
    mode: Literal["fishnet", "random"] = "fishnet",
    spacing: Optional[float] = None,
    n_points: Optional[int] = None,
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_batches: int = 100,
) -> gpd.GeoDataFrame:
    """
    Generate reference points over a raster and sample selected band values.

    Modes
    -----
    - "fishnet": Create a regular grid with the given `spacing`.
    - "random": Uniformly sample random points within the raster bounds until
      `n_points` valid points are collected.
      If `spacing` is given, enforce a minimum distance between points.

    Parameters
    ----------
    raster_path : str
        Path to the input raster (GeoTIFF).
    bands : list of (int | str | (int, str)), use Tuple to determine
        attribute name.
    output_path : str, optional
        If provided, write the resulting GeoDataFrame to this path.
    mode : {"fishnet", "random"}, default="fishnet"
        Point generation mode.
    spacing : float, optional
        - In "fishnet": grid spacing in raster CRS units (required).
        - In "random": minimum distance between accepted points (optional).
    n_points : int, optional
        Number of points to generate. Required if mode="random".
    random_state : int, optional
        Seed for reproducible random sampling.
    batch_size : int, optional
        Number of random candidates tested per batch (random mode).
        Default is max(n_points*2, 1000).
    max_batches : int, default=100
        Maximum number of batches for random sampling.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame of points with one attribute column per requested band.
    """
    rng = np.random.default_rng(random_state)

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        nodata = src.nodata
        descriptions = tuple(src.descriptions) if src.descriptions else tuple([None] * src.count)

        # --- Resolve bands ---
        if bands is None:
            bands = [1]
        resolved: List[Tuple[int, str]] = []
        for b in bands:
            if isinstance(b, tuple) and len(b) == 2:
                idx, name = b
                resolved.append((idx, str(name)))
            elif isinstance(b, int):
                desc = descriptions[b - 1] if (0 <= b-1 < len(descriptions)) else None
                col_name = desc if (desc and desc.strip()) else f"band{b}"
                resolved.append((b, col_name))
            elif isinstance(b, str):
                desc_lower = [d.lower() if d else "" for d in descriptions]
                if b.lower() not in desc_lower:
                    raise ValueError(f"Band name '{b}' not found in raster descriptions: {descriptions}")
                idx = desc_lower.index(b.lower()) + 1
                resolved.append((idx, b))
            else:
                raise TypeError("bands must contain int, str, or (int, str)")

        ref_band_index = resolved[0][0]

        if mode == "fishnet":
            if spacing is None:
                raise ValueError("Parameter 'spacing' is required for mode='fishnet'.")

            x_coords = np.arange(bounds.left, bounds.right + spacing, spacing)
            y_coords = np.arange(bounds.bottom, bounds.top + spacing, spacing)
            points = [Point(x, y) for x in x_coords for y in y_coords]

            coords = [(p.x, p.y) for p in points]
            ref_vals = np.array([v[0] for v in src.sample(coords, indexes=ref_band_index)])
            valid_mask = (~np.isnan(ref_vals)) & (ref_vals != nodata) if nodata is not None else ~np.isnan(ref_vals)
            valid_points = [p for p, keep in zip(points, valid_mask) if keep]
            valid_coords = [(p.x, p.y) for p in valid_points]

        elif mode == "random":
            if n_points is None:
                raise ValueError("Parameter 'n_points' is required for mode='random'.")
            if batch_size is None:
                batch_size = max(n_points * 2, 1000)

            valid_points, valid_coords = [], []
            for _ in range(max_batches):
                xs = rng.uniform(bounds.left, bounds.right, batch_size)
                ys = rng.uniform(bounds.bottom, bounds.top, batch_size)
                batch_coords = list(zip(xs, ys))
                ref_vals = np.array([v[0] for v in src.sample(batch_coords, indexes=ref_band_index)])
                keep = (~np.isnan(ref_vals)) & (ref_vals != nodata) if nodata is not None else ~np.isnan(ref_vals)

                for (x, y), k in zip(batch_coords, keep):
                    if not k:
                        continue
                    new_point = Point(x, y)
                    # Enforce min spacing if requested
                    if spacing is not None and valid_points:
                        if any(hypot(new_point.x - p.x, new_point.y - p.y) < spacing for p in valid_points):
                            continue
                    valid_points.append(new_point)
                    valid_coords.append((x, y))
                    if len(valid_points) >= n_points:
                        break
                if len(valid_points) >= n_points:
                    break
                batch_size = int(batch_size * 1.5)

            if len(valid_points) < n_points:
                raise RuntimeError(f"Only found {len(valid_points)} valid points (requested {n_points}).")

        else:
            raise ValueError("mode must be either 'fishnet' or 'random'.")

        if not valid_points:
            gdf_empty = gpd.GeoDataFrame(geometry=[], crs=crs)
            if output_path:
                gdf_empty.to_file(output_path)
            return gdf_empty

        band_indices = [idx for idx, _ in resolved]
        samples = np.array([v for v in src.sample(valid_coords, indexes=band_indices)])

    gdf = gpd.GeoDataFrame(geometry=valid_points, crs=crs)
    for j, (_, col_name) in enumerate(resolved):
        gdf[col_name] = samples[:, j]

    if output_path:
        gdf.to_file(output_path)

    return gdf
