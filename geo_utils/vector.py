import geopandas as gpd
import os
import rasterio
from rasterio.features import rasterize
import numpy as np


def rasterize_vector(input_path, resolution, attribute=None, datatype=None, nodata_value=None, crs=None, output_path=None):
    """
    Rasterisiert ein Shapefile oder GeoPackage und erstellt ein Raster mit einem spezifischen Attribut als Pixelwert.

    :param input_path: str
        Pfad zum Shapefile oder GeoPackage.
    :param output_path: str
        Pfad zum Ausgabedatei (Raster).
    :param resolution: float
        Die Auflösung des Rasters in der gleichen Einheit wie die Projektion (z.B. Meter).
    :param attribute: str
        Das Attribut, das als Pixelwert verwendet wird.
    :param datatype: str
        Datentyp für das Output Raster. Mögliche Datentypen sind ['uint8', 'int32', 'float32']. Wird kein Datentyp angegeben,
        wird dieser basierend auf den Attributen ermittelt.
    :param nodata_value: optional
        Der Wert, der für "NoData"-Pixel verwendet wird. Wenn nicht angegeben, wird dieser basierend auf dem Datentyp gesetzt.
    :param crs: str
        Das Zielkoordinatensystem des erstellten Rasters.
    :return: str
        Der Pfad des erstellten Rasters.
    """

    # Shapefile oder GeoPackage einlesen
    gdf = gpd.read_file(input_path)

    # Falls kein Attribut angegeben ist, bekommen alle Pixel den Wert 1
    if attribute is None:
        gdf[attribute] = 1

    # Datentyp bestimmen basierend auf den Attributwerten, falls kein Datentyp angegeben wurde
    if datatype is None and nodata_value is None:
        if np.issubdtype(gdf[attribute].dtype, np.integer):
            datatype = 'int32'  # Standardmäßig 32-bit Integer
            if nodata_value is None:
                nodata_value = -1  # Standardwert für Ganzzahlen
        elif np.issubdtype(gdf[attribute].dtype, np.unsignedinteger):
            datatype = 'uint8'  # Wenn es ein unsigned integer ist
            if nodata_value is None:
                nodata_value = 255  # Standardwert für unsigned integer
        else:
            datatype = 'float32'  # Für alle anderen Fälle den größten Datentyp verwenden
            if nodata_value is None:
                nodata_value = -9999
    elif datatype is None and nodata_value is not None:
        datatype = 'float32' # größter Datentyp, damit kein Fehler auftritt

    # Falls nur der Datentyp angegeben wird, wird der Nodata-Value darauf basierend gesetzt
    if nodata_value is None:
        if datatype == 'int32':
            nodata_value = -1
        elif datatype == 'uint8':
            nodata_value = 255
        elif datatype == 'float32':
            nodata_value = -9999

    # Falls kein Koordinatensystem angegeben wurde, wird das CRS der Vektor-File genutzt
    if crs is None:
        crs = gdf.crs

    # Festlegen des Extents und der Rasterauflösung
    bounds = gdf.total_bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)

    # Transformationsmatrix erstellen
    transform = rasterio.transform.from_origin(bounds[0], bounds[3], resolution, resolution)

    # Rasterisieren der Geometrien und Zuweisen des Attributs als Pixelwert
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))
    raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=nodata_value, dtype=datatype)

    # Profil für die Ausgabe-Rasterdatei erstellen
    profile = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': datatype,
        'crs': crs,
        'transform': transform,
        'width': width,
        'height': height,
        'nodata': nodata_value
    }

    if output_path is None:
        basename, extension = os.path.splitext(input_path)
        output_path = basename + "_rasterzied.tif"

    # Das Raster speichern
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(raster, 1)

    return output_path