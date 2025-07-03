import rasterio
import numpy as np
from shapely.geometry import Point
import geopandas as gpd

# Pfad zu deinem Raster
raster_path = r"Y:\MnD\data\GroundTruth\Sealed\2_upscaled_force\Mask\mask_v3\Frankfurt_xxxx_sealed_surface_fraction_10m_FORCE.tif"

# Mindestabstand in Metern
spacing = 31

# Raster öffnen
with rasterio.open(raster_path) as src:
    bounds = src.bounds
    crs = src.crs
    transform = src.transform
    mask = src.read(1)  # erstes Band lesen

# Gitterpunkte generieren
x_coords = np.arange(bounds.left, bounds.right, spacing)
y_coords = np.arange(bounds.bottom, bounds.top, spacing)
points = [Point(x, y) for x in x_coords for y in y_coords]

# In GeoDataFrame umwandeln
gdf = gpd.GeoDataFrame(geometry=points, crs=crs)

# Nur Punkte innerhalb des gültigen Rasterbereichs (nicht-NA) behalten
with rasterio.open(raster_path) as src:
    valid_points = []
    for pt in points:
        row, col = src.index(pt.x, pt.y)
        try:
            if src.read(1)[row, col] != src.nodata:
                valid_points.append(pt)
        except IndexError:
            continue

    gdf_valid = gpd.GeoDataFrame(geometry=valid_points, crs=crs)

    coord_list = [(x, y) for x, y in zip(gdf_valid["geometry"].x, gdf_valid["geometry"].y)]
    gdf_valid["value"] = [val[0] for val in src.sample(coord_list)]
    gdf_valid = gdf_valid[~np.isnan(gdf_valid["value"])]

# Optional: Speichern
gdf_valid.to_file(r"Y:\MnD\data\GroundTruth\Sealed\2_upscaled_force\Mask\mask_v3\temp\nrw_sample_points_30m.shp")
