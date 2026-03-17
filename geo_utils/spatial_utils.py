import os
import subprocess
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist


def spatial_thinning(
        input_points_path: str,
        min_dis: int,
        output_path: str = None
):
    """
    Spatial thinning of points: removes points that are closer than the given minimum distance.

    Parameters:
    - input_points_path: Path to the input points (GeoPackage or Shapefile).
    - min_dis: Minimum distance between points to keep in the dataset (in meters).
    - output_path: Path where the output file will be saved.
    """

    # Load points data into GeoDataFrame
    gdf = gpd.read_file(input_points_path)

    # Ensure the CRS is set to a projected coordinate system (e.g., UTM)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3035)  # WGS84 / ETRS89 (Europe)

    # Convert geometry to numpy array of (x, y) coordinates
    points_array = np.array([point.coords[0] for point in gdf.geometry])

    # Calculate distance matrix between all points
    distance_matrix = cdist(points_array, points_array)

    # Set the diagonal to a large number to avoid considering a point's distance to itself
    np.fill_diagonal(distance_matrix, np.inf)

    # List to store points and their attributes that meet the thinning condition
    thinned_points = []
    thinned_attributes = []

    # Iterate over all points and apply spatial thinning
    for idx in range(len(gdf)):
        keep = True

        # Check if the point is too close to any of the already kept points
        if np.any(distance_matrix[idx, :len(thinned_points)] < min_dis):
            keep = False

        # If the point is not too close to any kept point, add it to the thinned list
        if keep:
            thinned_points.append(gdf.geometry[idx])
            thinned_attributes.append(gdf.iloc[idx].drop('geometry'))  # Add attributes without 'geometry'

    # Create a new GeoDataFrame for the thinned points with attributes
    thinned_gdf = gpd.GeoDataFrame(thinned_attributes, geometry=thinned_points, crs=gdf.crs)

    # Ensure output path is correctly assigned
    if output_path is None:
        output_path = os.path.splitext(input_points_path)[0] + "_thinned.shp"

    # Save the thinned points to the output file
    thinned_gdf.to_file(output_path)

    print(f"Spatial thinning completed. Output saved to {output_path}")


def compute_target_sample_size(
        num_classes: int,
        target_n: int,
        output_csv_path: str
):
    """
    Computes target sample sizes per class and generates a simplified GRTS-ready CSV table.

    Parameters:
    ----------
    num_classes : int
        Number of classes (strata).
    target_n : int
        Target number of points per class.
    output_csv_path : str
        Path to output CSV file.

    Output:
    ------
    CSV file with columns:
    ['stratum_id', 'target_n']
    """

    # Create the stratum_id from 0 to num_classes - 1
    stratum_ids = list(range(num_classes))

    # Create a DataFrame with target_n for each class
    data = {
        'stratum_id': stratum_ids,
        'target_n': target_n  # same target_n for all classes
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False, sep=";")

    print(f"CSV with target sample sizes created: {output_csv_path}")
    return output_csv_path


def run_grts(
    points_path,
    targets_csv,
    stratum_var,
    r_script=None,
    seed=42,
    output_file=None,
    rscript_exe=r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
):
    """
        Run the GRTS (Generalized Random Tessellation Stratified) sampling method using an R script.

        Parameters:
        - points_path (str): Path to the input GeoPackage or Shapefile containing the points for the GRTS sampling.
        - targets_csv (str): Path to the CSV file containing the target data for sampling.
        - stratum_var (str): column name in point file in which the stratum ID is saved. This can be the class column or a specifically created stratum column
        - r_script (str, optional): Path to the R script that performs the GRTS sampling. If None, default script is used.
        - seed (int, optional): Random seed for reproducibility of the results. Default is 42.
        - output_file (str, optional): Path to the output file where the results will be saved.
        - rscript_exe (str, optional): Path to the Rscript executable. Default is set to the typical location of Rscript on a Windows machine for R v4.5.2.

        The function runs the R script using the provided parameters and executes the GRTS sampling process.
    """
    if r_script is None:
        r_script = os.path.abspath(os.path.join('r_scripts', 'grts.R'))

    if output_file is None:
        basename, extension = os.path.splitext(points_path)
        output_file = basename + "_grts" + extension

    cmd = [
        rscript_exe,
        r_script,
        "--points", points_path,
        "--targets", targets_csv,
        "--stratum_var", stratum_var,
        "--output", output_file,
        "--seed", str(seed),
    ]

    subprocess.run(
        cmd,
        text=True
    )


def run_blockcv(
    points_path,
    r_script=None,
    folds=5,
    blocksize=5000,
    column=None,
    selection="random",
    iteration=100,
    seed=42,
    output_file=None,
    rscript_exe=r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
):
    """
       Run the Block Cross-Validation (BlockCV) method using an R script.

       Parameters:
       - points_path (str): Path to the input GeoPackage or Shapefile containing the points for spatial blocking.
       - r_script (str): Path to the R script that performs the GRTS sampling. If None, default script is used.
       - folds (int): Number of folds to use in the cross-validation. Default is 5.
       - blocksize (int, optional): The size of the blocks used in the BlockCV method (in units of your spatial data, e.g., meters). Default is 5000.
       - column (str): Indicating the name of the column in which response variable is stored to find balanced records in folds.
       - selection (str): type of assignment of blocks into folds. Can be random, systematic, checkerboard, or predefined
       - iteration (str): number of attempts to create folds with balanced records
       - seed (int): Random seed for reproducibility of the results. Default is 42.
       - output_file (str): Path to the output file where the results will be saved.
       - rscript_exe (str): Path to the Rscript executable. Default is set to the typical location of Rscript on a Windows machine for R v4.5.2.

       The function runs the R script using the provided parameters and executes the BlockCV cross-validation process.
       """

    if r_script is None:
        r_script = os.path.abspath(os.path.join('r_scripts', 'blockcv.R'))

    if output_file is None:
        basename, extension = os.path.splitext(points_path)
        output_file = basename + "_blockcv" + extension

    cmd = [
        rscript_exe,
        r_script,
        "--points", points_path,
        "--output", output_file,
        "--folds", str(folds),
        "--blocksize", str(blocksize),
        "--column", str(column),
        "--selection", str(selection),
        "--iteration", str(iteration),
        "--seed", str(seed)
    ]

    subprocess.run(cmd, check=True)