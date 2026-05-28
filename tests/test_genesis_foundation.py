"""Phase 1 smoke tests for genesis_toolbox.pyt.

Covers only what Phase 1 delivers: foundation imports, persistence
classes, module utilities, and the new sensor abstraction (constants,
band-role map, auto-detect, parameter helper, sensor resolution).

Tool classes (01-06) are populated in later phases; their tests will
land alongside.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Foundation copied from the audited landsat_toolbox is intact
# ---------------------------------------------------------------------------

def test_persistence_classes_present(genesis):
    for cls_name in (
        "TransformStatistics",
        "MNFNoiseStatistics",
        "MNFStatistics",
        "PCAStatistics",
        "ICAStatistics",
    ):
        assert hasattr(genesis, cls_name), f"{cls_name} missing"


def test_module_utilities_present(genesis):
    for fn_name in (
        "noise_from_valid_diffs",
        "_project_with_band_means",
        "transform_pca",
        "transform_mnf",
        "transform_ica",
        "select_by_variance",
        "select_by_eigenvalue",
        "select_by_kurtosis",
        "hfc_vd",
    ):
        assert hasattr(genesis, fn_name), f"{fn_name} missing"


def test_module_constants_present(genesis):
    for name in (
        "_EIGVAL_FLOOR_ABS",
        "_EIGVAL_FLOOR_RELATIVE",
        "_ICA_WHITENING_FLOOR",
        "_HFC_SIGMA2_FLOOR",
        "_MNF_CORR_OFFDIAG_WARN",
        "_RAM_WARNING_GB",
    ):
        assert hasattr(genesis, name)


def test_audited_ica_kurtosis_fix_carried_over(genesis):
    """ISS-001 from the original audit: ICAStatistics must persist
    kurtosis_values + warnings list + description default. Verifies
    Phase 1 copy preserved those fixes."""
    s = genesis.ICAStatistics()
    assert s.description == "ICA Transform Statistics"
    assert hasattr(s, "kurtosis_values")
    assert hasattr(s, "warnings")


# ---------------------------------------------------------------------------
# Sensor abstraction (new in Phase 1)
# ---------------------------------------------------------------------------

def test_sensor_constants_distinct(genesis):
    """The four sensor identifiers must be distinct strings."""
    seen = {
        genesis.SENSOR_AUTO,
        genesis.SENSOR_LANDSAT_89,
        genesis.SENSOR_SENTINEL2,
        genesis.SENSOR_ASTER,
    }
    assert len(seen) == 4
    assert genesis.SENSOR_CHOICES == [
        genesis.SENSOR_AUTO,
        genesis.SENSOR_LANDSAT_89,
        genesis.SENSOR_SENTINEL2,
        genesis.SENSOR_ASTER,
    ]


def test_band_role_map_has_all_three_sensors(genesis):
    """SENSOR_BAND_ROLES must define a mapping for each non-Auto sensor."""
    bm = genesis.SENSOR_BAND_ROLES
    assert genesis.SENSOR_LANDSAT_89 in bm
    assert genesis.SENSOR_SENTINEL2 in bm
    assert genesis.SENSOR_ASTER in bm
    # Auto isn't a real sensor — must not be in the lookup
    assert genesis.SENSOR_AUTO not in bm


def test_universal_band_roles_present_on_all_sensors(genesis):
    """Red, NIR, SWIR1, SWIR2 must exist for ALL sensors — they're what
    universal indices (NDVI/NDWI/NDMI/NDBI) compute against."""
    for sensor in (genesis.SENSOR_LANDSAT_89,
                   genesis.SENSOR_SENTINEL2,
                   genesis.SENSOR_ASTER):
        mapping = genesis.SENSOR_BAND_ROLES[sensor]
        for role in ("Red", "NIR", "SWIR1", "SWIR2"):
            assert role in mapping, f"{role!r} missing from {sensor!r}"


def test_aster_has_no_blue_band(genesis):
    """ASTER physically lacks a blue band (lowest visible is Green/B1).
    The mapping must REFLECT this so indices needing Blue raise KeyError
    instead of silently picking a wrong band."""
    aster = genesis.SENSOR_BAND_ROLES[genesis.SENSOR_ASTER]
    assert "Blue" not in aster


def test_sentinel2_has_red_edge_bands(genesis):
    """S2's red-edge roles are what unlock NDRE / CIred-edge / IRECI."""
    s2 = genesis.SENSOR_BAND_ROLES[genesis.SENSOR_SENTINEL2]
    for role in ("RedEdge1", "RedEdge2", "RedEdge3", "NarrowNIR"):
        assert role in s2


def test_aster_has_per_wavelength_swir_bands(genesis):
    """ASTER's distinct SWIR bands (2.165, 2.205, 2.260, 2.330) are what
    unlock alunite / kaolinite / muscovite / calcite indices that the
    coarser-SWIR L8/9 and S2 sensors cannot compute."""
    aster = genesis.SENSOR_BAND_ROLES[genesis.SENSOR_ASTER]
    for role in ("SWIR2_2165", "SWIR2_2205", "SWIR2_2260", "SWIR2_2330"):
        assert role in aster


# ---------------------------------------------------------------------------
# get_band — successful resolution + clear errors
# ---------------------------------------------------------------------------

def test_get_band_returns_correct_index(genesis):
    """For S2 in the v1.0 12-band stack (B01 + B02..B12 minus B10),
    Red (B04) is at band index 4."""
    fake_bands = {i: f"band-{i}-raster" for i in range(1, 13)}
    red = genesis.get_band(fake_bands, "Red", genesis.SENSOR_SENTINEL2)
    assert red == "band-4-raster"


def test_get_band_landsat_red_is_band_4(genesis):
    fake_bands = {i: f"L{i}" for i in range(1, 8)}
    assert genesis.get_band(fake_bands, "Red", genesis.SENSOR_LANDSAT_89) == "L4"


def test_get_band_aster_red_is_band_2(genesis):
    """ASTER's Red is B2 — band index 2 in the 9-band stack."""
    fake_bands = {i: f"A{i}" for i in range(1, 10)}
    assert genesis.get_band(fake_bands, "Red", genesis.SENSOR_ASTER) == "A2"


def test_get_band_raises_on_missing_role(genesis):
    """Asking for 'Blue' on ASTER must raise KeyError with a helpful
    message listing what IS available."""
    fake_bands = {i: f"A{i}" for i in range(1, 10)}
    with pytest.raises(KeyError, match="Blue"):
        genesis.get_band(fake_bands, "Blue", genesis.SENSOR_ASTER)


def test_get_band_raises_on_unknown_sensor(genesis):
    with pytest.raises(ValueError, match="Unknown sensor"):
        genesis.get_band({}, "Red", "UnknownSat")


# ---------------------------------------------------------------------------
# detect_sensor — filename, band count, and ambiguity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected_attr", [
    # Landsat 8 — L2SP variant (legacy)
    ("D:/data/LC08_L2SP_204032_20240215_20240301_02_T1.tif", "SENSOR_LANDSAT_89"),
    # Landsat 8 — L2SR variant (current EE downloads)
    ("D:/data/LC08_L2SR_217033_20260502_20260514_02_T1.tar", "SENSOR_LANDSAT_89"),
    # Landsat 9 — L2SP
    ("/mnt/data/LC09_L2SP_205033_20240520_20240605_02_T1.tif", "SENSOR_LANDSAT_89"),
    # Landsat 9 — L2SR .tar (current EE downloads)
    ("LC09_L2SR_217033_20211222_20230504_02_T1.tar", "SENSOR_LANDSAT_89"),
    # Sentinel-2 SAFE-style
    ("S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE", "SENSOR_SENTINEL2"),
    # ASTER
    ("AST_07XT_00409282006074522_20250531042133_SRF_VNIR_B01.tif", "SENSOR_ASTER"),
])
def test_detect_sensor_from_filename(genesis, path, expected_attr):
    expected = getattr(genesis, expected_attr)
    assert genesis.detect_sensor(path) == expected


def test_detect_sensor_returns_none_when_unrecognised(genesis):
    # Filename gives no hint, and arcpy.Describe is stubbed in tests to
    # raise — so detect_sensor falls through and returns None.
    assert genesis.detect_sensor("D:/random/no_sensor_hint.tif") is None


def test_detect_sensor_handles_none_or_empty(genesis):
    assert genesis.detect_sensor(None) is None
    assert genesis.detect_sensor("") is None


# ---------------------------------------------------------------------------
# resolve_sensor — auto vs explicit, and the error message on failure
# ---------------------------------------------------------------------------

def test_resolve_sensor_uses_explicit_choice(genesis):
    """When the user picks a specific sensor, resolve_sensor must NOT
    auto-detect — the user's choice wins."""
    assert (
        genesis.resolve_sensor(genesis.SENSOR_LANDSAT_89, "random_path.tif")
        == genesis.SENSOR_LANDSAT_89
    )


def test_resolve_sensor_auto_then_detect_from_filename(genesis):
    """When Auto-detect is selected, resolve_sensor uses detect_sensor."""
    s = genesis.resolve_sensor(
        genesis.SENSOR_AUTO,
        "LC08_L2SP_204032_20240215_x.tif",
    )
    assert s == genesis.SENSOR_LANDSAT_89


def test_resolve_sensor_raises_when_auto_detect_fails(genesis):
    """If Auto-detect can't infer the sensor, the error message must
    tell the user how to fix it (pick manually)."""
    with pytest.raises(ValueError, match="set Sensor Type explicitly"):
        genesis.resolve_sensor(genesis.SENSOR_AUTO, "/data/mystery.tif")


# ---------------------------------------------------------------------------
# Toolbox class — Phase 1 ships an empty tools list (placeholder)
# ---------------------------------------------------------------------------

def test_toolbox_class_loads(genesis):
    """Toolbox class metadata. Tools list grows as build phases land —
    after Phase 2, IndicesComposites is registered."""
    tb = genesis.Toolbox()
    assert tb.label.startswith("GENESIS")
    assert tb.alias == "genesis"
    # At least one tool by Phase 2; Phase 6 will have all six.
    assert len(tb.tools) >= 1
