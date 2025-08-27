import rasterio
import numpy as np
raster_a = r"Y:\MnD\data\GroundTruth\Sealed\class_prob.tif"
raster_b = r"Y:\MnD\data\GroundTruth\Sealed\hamburg_regressio.tif"
output_raster = r"Y:\MnD\data\GroundTruth\Sealed\class_reg_70.tif"

# Öffne Raster A (3 Bänder)
with rasterio.open(raster_a) as srcA:
    A = srcA.read()   # shape: (3, height, width)
    profile = srcA.profile

# Öffne Raster B (1 Band)
with rasterio.open(raster_b) as srcB:
    B = srcB.read(1)  # shape: (height, width)

# Output-Array initialisieren mit Werten von B
out = B.copy()

# Bedingung 1: Band1 > 80 -> 0
mask1 = A[0] > 70
out[mask1] = 0

# Bedingung 2: Band3 > 80 -> 100
# (wird nur gesetzt, wo Bedingung1 NICHT zutrifft)
mask2 = (A[2] > 70) & (~mask1)
out[mask2] = 100

# Profil anpassen: ein Band, Wertebereich 0-100
profile.update(
    dtype=rasterio.uint8,
    count=1,
    nodata=255,
    compress='lzw'
)

# Speichern
with rasterio.open(output_raster, "w", **profile) as dst:
    dst.write(out.astype(np.uint8), 1)

''' 
import rasterio
from rasterio.windows import Window
import numpy as np
import os

def softmax(x, axis=0):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def logits_to_probabilities(input_path, output_path, block_size=512):
    with rasterio.open(input_path) as src:
        count = src.count  # Anzahl der Klassen / Bänder
        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=count,
            compress='lzw',
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            width, height = src.width, src.height

            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    w = min(block_size, width - x)
                    h = min(block_size, height - y)
                    window = Window(x, y, w, h)

                    # Logits laden (shape: [bands, h, w])
                    logits = src.read(window=window)

                    # Softmax pixelweise anwenden (über Achse 0: Klassen)
                    probs = softmax(logits, axis=0) * 100.0  # Prozent

                    # Schreiben
                    dst.write(probs.astype(np.float32), window=window)

    print(f"Fertig: Wahrscheinlichkeiten in {output_path}")

# Beispielaufruf
input_raster = r"Y:\MnD\data\GroundTruth\Sealed\hamburg_2020_ 2.tif"  # z. B. 3-Band TIFF
output_raster = r"Y:\MnD\data\GroundTruth\Sealed\hamburg_2020_ 2_precent.tif"
logits_to_probabilities(input_raster, output_raster)
'''
