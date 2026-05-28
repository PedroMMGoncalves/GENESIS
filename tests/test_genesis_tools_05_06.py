"""Phase 6 tests for Tools 05 (Transformations) and 06 (SpectralAngleMapper).

Structural tests: classes present, registered in Toolbox in workflow
order, sensor parameter wired, labels match the workflow numbering.

Behavioural tests for Transformations exist in test_genesis_algorithms.py
since the algorithms are the same as the original `LandsatTransformations`
which has been audited and battle-tested.

SAM behavioural tests are limited because most of the SAM execute path
is arcpy.sa-heavy and requires ArcGIS Pro to run end-to-end. The audit
findings beyond SetNull inversion are documented but not yet fixed.
"""

from __future__ import annotations

import inspect


# ---------------------------------------------------------------------------
# All tools registered in workflow order
# ---------------------------------------------------------------------------

def test_toolbox_registers_all_tools_in_workflow_order(genesis):
    tb = genesis.Toolbox()
    names = [t.__name__ for t in tb.tools]
    assert names == [
        "Sentinel2Mosaic",
        "LandsatMosaic",
        "AsterMosaic",
        "IndicesComposites",
        "Transformations",
        "SpectralAngleMapper",
        "TemporalStatistics",
    ]


# ---------------------------------------------------------------------------
# Tool 05: Transformations
# ---------------------------------------------------------------------------

def test_transformations_class_present(genesis):
    assert hasattr(genesis, "Transformations")
    # The legacy class name must NOT survive into the active module.
    assert not hasattr(genesis, "LandsatTransformations")


def test_transformations_label_uses_05_prefix(genesis):
    tool = genesis.Transformations()
    assert tool.label.startswith("05 —")


def test_transformations_description_mentions_three_sensors(genesis):
    """Description must signal this tool works for all three sensors,
    not just Landsat."""
    desc = genesis.Transformations().description.lower()
    assert "landsat" in desc
    assert "sentinel" in desc
    assert "aster" in desc


def test_transformations_has_sensor_parameter(genesis):
    tool = genesis.Transformations()
    params = tool.getParameterInfo()
    # The sensor parameter is the LAST one in the list (Phase 6 addition).
    last = params[-1]
    assert last.name == "sensor_type"
    assert last.value == genesis.SENSOR_AUTO


def test_transformations_audit_fixes_carried_over(genesis):
    """The audited classes carry kurtosis persistence + the warnings list
    + n_iterations validation. Sanity-check by instantiating ICAStatistics
    via the toolbox module — same backing class as Tool 05 uses."""
    s = genesis.ICAStatistics()
    assert s.description == "ICA Transform Statistics"
    assert hasattr(s, "warnings")
    assert hasattr(s, "kurtosis_values")
    assert hasattr(s, "random_state")


def test_transformations_uses_make_sensor_parameter_helper(genesis):
    """Source-level: the sensor param should come from the shared helper
    so the dropdown choices stay consistent across Tools 04, 05, 06."""
    src = inspect.getsource(genesis.Transformations.getParameterInfo)
    assert "make_sensor_parameter()" in src


# ---------------------------------------------------------------------------
# Tool 06: SpectralAngleMapper
# ---------------------------------------------------------------------------

def test_sam_class_present(genesis):
    assert hasattr(genesis, "SpectralAngleMapper")
    # Legacy name retired.
    assert not hasattr(genesis, "LandsatSAM")


def test_sam_label_uses_06_prefix(genesis):
    tool = genesis.SpectralAngleMapper()
    assert tool.label.startswith("06 —")


def test_sam_input_label_is_sensor_neutral(genesis):
    """The input raster label used to say 'Input Landsat Raster' — the
    port must drop the Landsat-specific framing."""
    src = inspect.getsource(genesis.SpectralAngleMapper.getParameterInfo)
    assert "Input Landsat Raster" not in src


def test_sam_has_sensor_parameter(genesis):
    tool = genesis.SpectralAngleMapper()
    params = tool.getParameterInfo()
    last = params[-1]
    assert last.name == "sensor_type"


def test_sam_audit_fix_setnull_inversion_carried_over(genesis):
    """The original LandsatSAM had SetNull(norm > 0, ...) which nulled
    every valid pixel. The audit fix uses SetNull(norm <= 0, ...).

    Note: inspect.getsource doesn't work on class objects loaded via
    SourceFileLoader (raises TypeError), so target the specific method
    that contains the fix.
    """
    src = inspect.getsource(genesis.SpectralAngleMapper._sam_with_table)
    assert "SetNull(norm_raster <= 0" in src
    # The broken pattern must be gone.
    assert "SetNull(norm_raster > 0," not in src


def test_sam_uses_make_sensor_parameter_helper(genesis):
    src = inspect.getsource(genesis.SpectralAngleMapper.getParameterInfo)
    assert "make_sensor_parameter()" in src
