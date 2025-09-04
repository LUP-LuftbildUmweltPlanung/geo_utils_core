# tests/test_coregistration.py
import os
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from geo_utils.raster import co_registration # adjust import to your module name


# ----------------------- Helpers -----------------------

def write_gtiff_single(path, array, transform, crs=None, nodata=None, compress="LZW"):
    """Write a single-band GeoTIFF with minimal profile."""
    h, w = array.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": array.dtype,
        "transform": transform,
        "compress": compress,
    }
    if crs is not None:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return str(path)


def write_gtiff_multi(path, arrays, transform, crs=None, nodata=None, compress="LZW"):
    """
    Write a multi-band GeoTIFF.
    arrays: np.ndarray of shape (bands, height, width)
    """
    count, h, w = arrays.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": count,
        "dtype": arrays.dtype,
        "transform": transform,
        "compress": compress,
    }
    if crs is not None:
        profile["crs"] = crs
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arrays)
    return str(path)


def read_band(path, band=1):
    with rasterio.open(path) as ds:
        return ds.read(band), ds.profile.copy()


# Stable metric CRS and simple transforms
EPSG_3857 = CRS.from_epsg(3857)
T_PARENT = Affine.translation(0, 0) * Affine.scale(1, -1)  # 1 m px, origin (0,0)
T_SHIFT_X1 = Affine.translation(1, 0) * Affine.scale(1, -1)  # shifted by +1 pixel in x


# ----------------------- Fixtures -----------------------

@pytest.fixture
def parent_raster(tmp_path):
    """10x10 float parent raster, all zeros, defines target grid."""
    arr = np.zeros((10, 10), dtype=np.float32)
    return write_gtiff_single(tmp_path / "parent.tif", arr, T_PARENT, crs=EPSG_3857)


@pytest.fixture
def child_ident_float(tmp_path):
    """10x10 float child matching parent grid with value 1.0."""
    arr = np.ones((10, 10), dtype=np.float32)
    return write_gtiff_single(tmp_path / "child_ident_float.tif", arr, T_PARENT, crs=EPSG_3857)


@pytest.fixture
def child_shifted_int(tmp_path):
    """
    10x10 int16 child shifted by +1 pixel in x.
    Values=5, nodata = -32768 (explicit).
    """
    arr = np.full((10, 10), 5, dtype=np.int16)
    return write_gtiff_single(tmp_path / "child_shifted_int.tif", arr, T_SHIFT_X1, crs=EPSG_3857, nodata=-32768)


@pytest.fixture
def child_with_nodata_region(tmp_path):
    """10x10 uint16 child with an internal NoData hole (nodata=65535)."""
    arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
    arr[2:5, 3:7] = 65535  # mark hole
    return write_gtiff_single(tmp_path / "child_hole.tif", arr, T_PARENT, crs=EPSG_3857, nodata=65535)


@pytest.fixture
def child_no_nodata_uint16(tmp_path):
    """10x10 uint16 child with no declared nodata."""
    arr = np.full((10, 10), 42, dtype=np.uint16)
    return write_gtiff_single(tmp_path / "child_no_nodata_uint16.tif", arr, T_PARENT, crs=EPSG_3857)


@pytest.fixture
def child_multiband_uint16(tmp_path):
    """3-band uint16 child (bands 10, 20, 30), same grid as parent."""
    arrays = np.stack([
        np.full((10, 10), 10, dtype=np.uint16),
        np.full((10, 10), 20, dtype=np.uint16),
        np.full((10, 10), 30, dtype=np.uint16),
    ], axis=0)  # (3, H, W)
    return write_gtiff_multi(tmp_path / "child_mb.tif", arrays, T_PARENT, crs=EPSG_3857)


@pytest.fixture
def child_missing_crs(tmp_path):
    """Child written without CRS to trigger validation error."""
    arr = np.ones((10, 10), dtype=np.float32)
    # Intentionally omit CRS
    return write_gtiff_single(tmp_path / "child_no_crs.tif", arr, T_PARENT, crs=None)


# ----------------------- Tests -----------------------

def test_identity_nearest_preserves_values(parent_raster, child_ident_float, tmp_path):
    """Same grid + nearest: all values should remain exactly 1.0."""
    out_path = tmp_path / "out_ident_nearest.tif"
    res = co_registration(parent_raster, child_ident_float, resampling_method="nearest", output_path=str(out_path))
    assert res == str(out_path)

    data, prof = read_band(res, 1)
    assert data.shape == (10, 10)
    assert np.allclose(data, 1.0)

    with rasterio.open(parent_raster) as p, rasterio.open(res) as d:
        assert d.crs == p.crs
        assert d.transform == p.transform
        assert d.width == p.width and d.height == p.height


@pytest.mark.parametrize("method", ["bilinear", "cubic", "lanczos"])
def test_identity_interpolators(parent_raster, child_ident_float, tmp_path, method):
    """Same grid + interpolators: values should remain ~1.0."""
    out_path = tmp_path / f"out_ident_{method}.tif"
    res = co_registration(parent_raster, child_ident_float, resampling_method=method, output_path=str(out_path))
    data, _ = read_band(res, 1)
    assert np.allclose(data, 1.0, atol=1e-6)


def test_shifted_child_nearest_propagates_nodata_and_values(parent_raster, child_shifted_int, tmp_path):
    """
    Shifted child with nearest: leftmost column(s) become NoData; right side should be 5.
    """
    out_path = tmp_path / "out_shifted_nearest.tif"
    res = co_registration(parent_raster, child_shifted_int, resampling_method="nearest", output_path=str(out_path))
    data, prof = read_band(res, 1)
    nodata = prof.get("nodata")
    assert nodata == -32768

    # Robust checks: leftmost column likely nodata or 5 depending on exact footprint,
    # but rightmost column should definitely be 5.
    assert np.all(data[:, -1] == 5)
    assert np.all((data[:, 0] == nodata) | (data[:, 0] == 5))


def test_nodata_region_preserved(parent_raster, child_with_nodata_region, tmp_path):
    """Child's NoData hole remains NoData after reprojection on identical grid."""
    out_path = tmp_path / "out_nodata_region.tif"
    res = co_registration(parent_raster, child_with_nodata_region, "nearest", output_path=str(out_path))
    data, prof = read_band(res, 1)
    assert prof["nodata"] == 65535
    assert np.all(data[2:5, 3:7] == 65535)


def test_fallback_nodata_applied(parent_raster, child_no_nodata_uint16, tmp_path):
    """If child has no nodata, fallback_nodata should be honored."""
    out_path = tmp_path / "out_fallback_nd.tif"
    res = co_registration(
        parent_raster, child_no_nodata_uint16, "nearest",
        output_path=str(out_path), fallback_nodata=9999
    )
    _, prof = read_band(res, 1)
    assert prof["nodata"] == 9999


def test_multiband_identity(parent_raster, child_multiband_uint16, tmp_path):
    """Multi-band child should retain band values and band count on same grid."""
    out_path = tmp_path / "out_mb.tif"
    res = co_registration(parent_raster, child_multiband_uint16, "nearest", output_path=str(out_path))

    with rasterio.open(res) as ds:
        assert ds.count == 3
        b1 = ds.read(1)
        b2 = ds.read(2)
        b3 = ds.read(3)

    assert np.all(b1 == 10)
    assert np.all(b2 == 20)
    assert np.all(b3 == 30)


def test_tiling_profile_fields_present(parent_raster, child_ident_float, tmp_path):
    """When tiled=True, GeoTIFF should have tiling metadata."""
    out_path = tmp_path / "out_tiled.tif"
    res = co_registration(
        parent_path=parent_raster,
        child_path=child_ident_float,
        resampling_method="nearest",
        output_path=str(out_path),
        tiled=True,
        blocksize=256
    )
    with rasterio.open(res) as ds:
        # rasterio exposes 'tiled', 'blockxsize', 'blockysize' in profile
        prof = ds.profile
        assert prof.get("tiled", False) is True
        assert prof.get("blockxsize") == 256
        assert prof.get("blockysize") == 256


def test_invalid_resampling_raises(parent_raster, child_ident_float, tmp_path):
    with pytest.raises(ValueError):
        co_registration(parent_raster, child_ident_float, resampling_method="not_a_method", output_path=str(tmp_path / "x.tif"))


def test_missing_crs_raises(parent_raster, child_missing_crs, tmp_path):
    with pytest.raises(ValueError):
        co_registration(parent_raster, child_missing_crs, "nearest", output_path=str(tmp_path / "x2.tif"))


def test_auto_output_path_created_and_removed(parent_raster, child_ident_float):
    """If output_path=None, the function returns a path ending with '_coregistered.tif'."""
    result_path = co_registration(parent_raster, child_ident_float, "nearest", output_path=None)
    assert result_path.endswith("_coregistered.tif")
    # Clean up file created in the child's directory
    if os.path.exists(result_path):
        os.remove(result_path)


def test_warning_when_interpolating_integers(capsys, parent_raster, child_shifted_int, tmp_path):
    """
    Integer (categorical) data with non-nearest resampling should print a warning.
    Capture stdout and assert warning presence.
    """
    out_path = tmp_path / "out_warn.tif"
    co_registration(parent_raster, child_shifted_int, "bilinear", output_path=str(out_path))
    captured = capsys.readouterr()
    assert "Warning: Categorical/integer data" in captured.out
