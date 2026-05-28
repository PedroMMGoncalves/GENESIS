"""Phase 2 tests for Tool 04 (IndicesComposites).

Covers the band-role abstraction the user-facing UX depends on:
  - INDICES / COMPOSITES catalogues are structurally well-formed
  - Sensor filtering: each sensor sees only what it can compute
  - Universal indices (NDVI/NDWI/NDMI/NDBI) work across all 3 sensors
  - S2-only red-edge indices hidden from other sensors
  - ASTER-only mineral indices hidden from other sensors
  - Indices needing Blue hidden from ASTER
  - label_to_index_key / label_to_composite_key roundtrip
  - GP UI category grouping order
  - Tool class registers in Toolbox

Doesn't run arcpy.sa.Divide etc. against real rasters — that's an
integration test that requires ArcGIS Pro.
"""

from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# Catalogue structure
# ---------------------------------------------------------------------------

def test_indices_catalogue_present_and_nonempty(genesis):
    assert hasattr(genesis, "INDICES")
    assert isinstance(genesis.INDICES, dict)
    assert len(genesis.INDICES) >= 15, (
        f"Expected at least 15 indices; got {len(genesis.INDICES)}"
    )


def test_composites_catalogue_present_and_nonempty(genesis):
    assert hasattr(genesis, "COMPOSITES")
    assert isinstance(genesis.COMPOSITES, dict)
    assert len(genesis.COMPOSITES) >= 5


@pytest.mark.parametrize("entry_name", ["INDICES", "COMPOSITES"])
def test_catalogue_entries_have_required_fields(genesis, entry_name):
    cat = getattr(genesis, entry_name)
    required = {"category", "required_roles", "display_labels", "output_suffix"}
    for key, meta in cat.items():
        missing = required - meta.keys()
        assert not missing, f"{entry_name}[{key!r}] missing fields: {missing}"
        assert isinstance(meta["required_roles"], list)
        assert isinstance(meta["display_labels"], dict)
        assert len(meta["display_labels"]) >= 1


def test_indices_have_compute_lambda(genesis):
    for key, meta in genesis.INDICES.items():
        assert callable(meta.get("compute")), f"INDICES[{key!r}].compute not callable"


def test_composites_have_band_spec_list(genesis):
    for key, meta in genesis.COMPOSITES.items():
        assert isinstance(meta.get("band_spec"), list)
        assert len(meta["band_spec"]) == 3, (
            f"COMPOSITES[{key!r}].band_spec must have exactly 3 channels"
        )


# ---------------------------------------------------------------------------
# Sensor filtering: which indices/composites appear per sensor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index_key", ["NDVI", "NDWI_McFeeters", "NDMI", "NDBI"])
def test_universal_indices_show_for_all_three_sensors(genesis, index_key):
    """Universal indices must be applicable to all three sensors."""
    meta = genesis.INDICES[index_key]
    for sensor in (genesis.SENSOR_LANDSAT_89, genesis.SENSOR_SENTINEL2, genesis.SENSOR_ASTER):
        labels = genesis.applicable_index_labels_flat(sensor)
        sensor_label = meta["display_labels"].get(sensor)
        assert sensor_label is not None, f"{index_key} missing display_labels[{sensor}]"
        assert sensor_label in labels, (
            f"{sensor_label!r} should appear for {sensor} but doesn't"
        )


@pytest.mark.parametrize("index_key", ["NDRE", "CIred_edge", "IRECI"])
def test_red_edge_indices_only_for_sentinel2(genesis, index_key):
    """Red-edge indices must NOT appear for L8/9 or ASTER."""
    s2_labels = genesis.applicable_index_labels_flat(genesis.SENSOR_SENTINEL2)
    l8_labels = genesis.applicable_index_labels_flat(genesis.SENSOR_LANDSAT_89)
    aster_labels = genesis.applicable_index_labels_flat(genesis.SENSOR_ASTER)

    meta = genesis.INDICES[index_key]
    s2_label = meta["display_labels"][genesis.SENSOR_SENTINEL2]

    assert s2_label in s2_labels
    # Other sensors must not have it — assert by checking no label from
    # this index's display_labels appears in their list.
    for label in meta["display_labels"].values():
        assert label not in l8_labels, f"{label!r} leaked into L8/9 list"
        assert label not in aster_labels, f"{label!r} leaked into ASTER list"


@pytest.mark.parametrize("index_key", [
    "Alunite_ASTER", "Kaolinite_ASTER", "Muscovite_ASTER", "Calcite_ASTER",
    "Hydrothermal_Cudahy_ASTER",
])
def test_aster_mineral_indices_only_for_aster(genesis, index_key):
    aster_labels = genesis.applicable_index_labels_flat(genesis.SENSOR_ASTER)
    l8_labels = genesis.applicable_index_labels_flat(genesis.SENSOR_LANDSAT_89)
    s2_labels = genesis.applicable_index_labels_flat(genesis.SENSOR_SENTINEL2)

    meta = genesis.INDICES[index_key]
    aster_label = meta["display_labels"][genesis.SENSOR_ASTER]
    assert aster_label in aster_labels
    for label in meta["display_labels"].values():
        assert label not in l8_labels
        assert label not in s2_labels


def test_iron_oxide_red_blue_hidden_from_aster(genesis):
    """ASTER has no Blue band — Iron Oxide via Red/Blue must not appear."""
    aster = genesis.applicable_index_labels_flat(genesis.SENSOR_ASTER)
    for label in genesis.INDICES["Iron_Oxide_RedBlue"]["display_labels"].values():
        assert label not in aster


def test_iron_oxide_aster_only_for_aster(genesis):
    """The ASTER-specific Red/Green Iron Oxide variant must only show on ASTER
    (despite L8 and S2 also having Red+Green — they have the better Red/Blue
    version, so sensor_filter limits this entry to ASTER)."""
    meta = genesis.INDICES["Iron_Oxide_ASTER_RG"]
    assert meta.get("sensor_filter") == [genesis.SENSOR_ASTER]
    aster_label = meta["display_labels"][genesis.SENSOR_ASTER]
    assert aster_label in genesis.applicable_index_labels_flat(genesis.SENSOR_ASTER)
    assert aster_label not in genesis.applicable_index_labels_flat(genesis.SENSOR_LANDSAT_89)
    assert aster_label not in genesis.applicable_index_labels_flat(genesis.SENSOR_SENTINEL2)


def test_natural_color_composite_hidden_from_aster(genesis):
    """Natural Color needs Blue — ASTER has none."""
    aster = genesis.applicable_composite_labels_flat(genesis.SENSOR_ASTER)
    for label in genesis.COMPOSITES["Natural_Color_RGB"]["display_labels"].values():
        assert label not in aster


# ---------------------------------------------------------------------------
# Category grouping in UI order
# ---------------------------------------------------------------------------

def test_indices_grouped_by_category_in_canonical_order(genesis):
    """applicable_indices returns dict ordered by canonical category list,
    not alphabetical / insertion order."""
    grouped = genesis.applicable_indices(genesis.SENSOR_SENTINEL2)
    cats = list(grouped.keys())
    # Vegetation must come before Geological in the canonical ordering
    assert cats.index("Vegetation") < cats.index("Geological")
    # Red-Edge must come AFTER all the universal categories
    assert cats.index("Red-Edge") > cats.index("Vegetation")


def test_aster_grouped_includes_aster_minerals_category(genesis):
    grouped = genesis.applicable_indices(genesis.SENSOR_ASTER)
    assert "ASTER Minerals" in grouped
    # Should be last (canonical order puts it after the universal ones)
    cats = list(grouped.keys())
    assert cats.index("ASTER Minerals") == len(cats) - 1


# ---------------------------------------------------------------------------
# Label <-> key roundtrip
# ---------------------------------------------------------------------------

def test_label_to_index_key_roundtrip(genesis):
    """For every (index, sensor) pair with a display label, the reverse
    lookup must return that index's key."""
    for key, meta in genesis.INDICES.items():
        for sensor, label in meta["display_labels"].items():
            roundtripped = genesis.label_to_index_key(label, sensor)
            assert roundtripped == key, (
                f"{label!r} for {sensor!r} reverse-mapped to {roundtripped!r}, "
                f"expected {key!r}"
            )


def test_label_to_composite_key_roundtrip(genesis):
    for key, meta in genesis.COMPOSITES.items():
        for sensor, label in meta["display_labels"].items():
            assert genesis.label_to_composite_key(label, sensor) == key


def test_label_to_index_key_returns_none_for_unknown(genesis):
    assert genesis.label_to_index_key("Not a real index", genesis.SENSOR_SENTINEL2) is None


# ---------------------------------------------------------------------------
# Tool class wiring
# ---------------------------------------------------------------------------

def test_indices_composites_tool_class_present(genesis):
    assert hasattr(genesis, "IndicesComposites")


def test_toolbox_registers_indices_composites(genesis):
    tb = genesis.Toolbox()
    tool_names = [t.__name__ for t in tb.tools]
    assert "IndicesComposites" in tool_names


def test_tool_label_uses_workflow_prefix(genesis):
    tool = genesis.IndicesComposites()
    assert tool.label.startswith("04 —"), (
        f"Tool label should start with '04 —' for workflow ordering; got {tool.label!r}"
    )


def test_tool_description_mentions_sensors(genesis):
    desc = genesis.IndicesComposites().description.lower()
    for keyword in ("landsat", "sentinel", "aster"):
        assert keyword in desc, f"Description doesn't mention {keyword}"


def test_tool_getparameterinfo_returns_eight_parameters(genesis):
    tool = genesis.IndicesComposites()
    params = tool.getParameterInfo()
    assert len(params) == 8, f"Expected 8 GP parameters, got {len(params)}"
    # Sensor selector must be at index 1
    assert params[1].name == "sensor_type"


def test_tool_uses_sensor_selector_helper(genesis):
    """The tool must call make_sensor_parameter (so changes there propagate
    automatically). Verified via source inspection."""
    src = inspect.getsource(genesis.IndicesComposites.getParameterInfo)
    assert "make_sensor_parameter()" in src


def test_tool_execute_resolves_sensor(genesis):
    """execute() must call resolve_sensor — that's how Auto-detect works."""
    src = inspect.getsource(genesis.IndicesComposites.execute)
    assert "resolve_sensor(" in src


def test_tool_uses_bounds_filter_audit_fix(genesis):
    """The Bug 3 audit fix (collapsed BooleanOr SetNull) must be carried
    forward — protects against Inf surviving downstream operations."""
    src = inspect.getsource(genesis.IndicesComposites._calculate_indices)
    assert "BooleanOr" in src
    assert "10000" in src  # The bound value
