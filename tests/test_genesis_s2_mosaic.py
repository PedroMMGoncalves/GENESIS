"""Phase 4 tests for Tool 01 (Sentinel2Mosaic).

Direct behaviour tests for the pure-Python helpers (zip extraction,
SAFE folder discovery, metadata parsing, temporal filter, provenance),
plus structural tests confirming the tool is wired into the toolbox.

End-to-end SCL masking and geometric median require ArcGIS Pro and are
not covered here.
"""

from __future__ import annotations

import csv
import inspect
import io
import os
import zipfile
from datetime import date

import pytest


# ---------------------------------------------------------------------------
# Class wiring & metadata
# ---------------------------------------------------------------------------

def test_sentinel2_mosaic_class_present(genesis):
    assert hasattr(genesis, "Sentinel2Mosaic")


def test_sentinel2_mosaic_registered_first_in_toolbox(genesis):
    """Workflow order: S2 mosaic should be Tool 01 (first), then Landsat."""
    tb = genesis.Toolbox()
    names = [t.__name__ for t in tb.tools]
    assert names[0] == "Sentinel2Mosaic"
    assert "LandsatMosaic" in names


def test_sentinel2_mosaic_label_uses_01_prefix(genesis):
    tool = genesis.Sentinel2Mosaic()
    assert tool.label.startswith("01 —")


def test_description_mentions_scl_and_safe(genesis):
    desc = genesis.Sentinel2Mosaic().description.lower()
    assert "scl" in desc
    assert "safe" in desc or ".safe" in desc
    # Compositor menu surfaced in v1.0 — the description should still
    # name at least one reducer so users know what they're getting.
    assert "geometric median" in desc or "compositor" in desc or "percentile" in desc


def test_getparameterinfo_full_v1_parameter_set(genesis):
    """v1.0 added the compositor menu, cloud-aggressiveness slider, and
    subprocess batch sizing. Look up by name to stay robust against UI
    re-ordering across releases."""
    tool = genesis.Sentinel2Mosaic()
    params = tool.getParameterInfo()
    names = [p.name for p in params]
    for required in (
        "gdb_path",
        "mosaic_name",
        "data_folder",
        "cloud_aggressiveness",
        "cloud_buffer_pixels",
        "compositor",
        "percentile_value",
    ):
        assert required in names, f"missing required S2 parameter: {required}"


# ---------------------------------------------------------------------------
# SAFE folder discovery
# ---------------------------------------------------------------------------

def _make_safe_folder(parent, safe_name):
    safe = parent / safe_name
    safe.mkdir()
    # Minimal structure so the tool doesn't trip on emptiness — a GRANULE
    # subfolder with an L2A_ tile dir is what _locate_band_files looks for.
    (safe / "GRANULE").mkdir()
    return safe


def test_find_safe_scenes_discovers_safe_folders_and_zips(genesis, tmp_path):
    """v1.0 contract: returns tuples (path, kind) where kind is
    'safe' for an extracted .SAFE folder or 'zip' for a Copernicus
    zip archive. Both are downstream-uniform via GDAL VSI."""
    _make_safe_folder(tmp_path, "S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE")
    _make_safe_folder(tmp_path, "S2B_MSIL2A_20240605T103021_N0500_R108_T29SQB_20240605T142319.SAFE")
    (tmp_path / "not_safe").mkdir()
    zip_name = "S2A_MSIL2A_20240701T103021_N0500_R108_T29SQB_20240701T142319.zip"
    with zipfile.ZipFile(tmp_path / zip_name, "w") as zf:
        zf.writestr(f"{zip_name[:-4]}.SAFE/MTD_MSIL2A.xml", b"<empty/>")

    result = genesis.Sentinel2Mosaic._find_safe_scenes(str(tmp_path))
    # 2 .SAFE folders + 1 .zip. The plain "not_safe" folder is ignored.
    assert len(result) == 3
    kinds = sorted(kind for _, kind in result)
    assert kinds == ["safe", "safe", "zip"]


def test_find_safe_scenes_handles_missing_folder(genesis):
    assert genesis.Sentinel2Mosaic._find_safe_scenes(None) == []
    assert genesis.Sentinel2Mosaic._find_safe_scenes("") == []
    assert genesis.Sentinel2Mosaic._find_safe_scenes("/no/such/path") == []


# ---------------------------------------------------------------------------
# SAFE metadata parsing
# ---------------------------------------------------------------------------

def test_parse_safe_metadata_from_filename(genesis, tmp_path):
    name = "S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE"
    safe = _make_safe_folder(tmp_path, name)
    meta = genesis.Sentinel2Mosaic._parse_safe_metadata(str(safe))
    assert meta is not None
    assert meta["tile_id"] == "T29SQB"
    assert meta["date_acquired"] == date(2024, 6, 1)
    assert meta["product_uri"] == name
    # Cloud cover unset (no XML present) — should be None, not crash.
    assert meta["cloud_cover"] is None


def test_parse_safe_metadata_rejects_non_s2_folder(genesis, tmp_path):
    not_s2 = tmp_path / "random_folder.SAFE"
    not_s2.mkdir()
    assert genesis.Sentinel2Mosaic._parse_safe_metadata(str(not_s2)) is None


def test_parse_safe_metadata_reads_cloud_cover_from_xml(genesis, tmp_path):
    name = "S2B_MSIL2A_20240715T103021_N0500_R108_T29SQB_20240715T142319.SAFE"
    safe = _make_safe_folder(tmp_path, name)
    mtd = safe / "MTD_MSIL2A.xml"
    # Realistic-ish XML with the namespaced tag we look for.
    mtd.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd">'
        '<n1:Quality_Indicators_Info>'
        '<Cloud_Coverage_Assessment>23.4</Cloud_Coverage_Assessment>'
        '</n1:Quality_Indicators_Info>'
        '</n1:Level-2A_User_Product>',
        encoding="utf-8",
    )
    meta = genesis.Sentinel2Mosaic._parse_safe_metadata(str(safe))
    assert meta["cloud_cover"] == 23.4


def test_parse_safe_metadata_handles_malformed_xml(genesis, tmp_path):
    """A broken MTD_MSIL2A.xml must not crash — fall back to filename only."""
    name = "S2A_MSIL2A_20240801T103021_N0500_R108_T29SQB_20240801T142319.SAFE"
    safe = _make_safe_folder(tmp_path, name)
    (safe / "MTD_MSIL2A.xml").write_text("<not valid XML", encoding="utf-8")
    meta = genesis.Sentinel2Mosaic._parse_safe_metadata(str(safe))
    assert meta is not None  # filename parse still succeeded
    assert meta["tile_id"] == "T29SQB"
    assert meta["cloud_cover"] is None  # XML parse failed → stays None


# ---------------------------------------------------------------------------
# Temporal filter
# ---------------------------------------------------------------------------

@pytest.fixture
def june_2024_scene():
    return {
        "metadata": {
            "date_acquired": date(2024, 6, 15),
            "tile_id": "T29SQB",
            "cloud_cover": 12.0,
            "product_uri": "S2A_MSIL2A_20240615...",
        }
    }


@pytest.fixture
def january_2024_scene():
    return {
        "metadata": {
            "date_acquired": date(2024, 1, 10),
            "tile_id": "T29SQB",
            "cloud_cover": 50.0,
            "product_uri": "S2A_MSIL2A_20240110...",
        }
    }


def test_temporal_filter_month_in_year(genesis, june_2024_scene, january_2024_scene):
    """v1.0 unified the time-filter dropdown: ``Month in Year`` (the
    Landsat label) replaces the old ``year_month`` snake_case value.
    _create_temporal_filter normalises both forms via
    _time_filter_key, but tests assert the canonical key."""
    tool = genesis.Sentinel2Mosaic()
    f = tool._create_temporal_filter("Month in Year", 2024, 6, None)
    assert f["type"] == "month_in_year"
    assert tool._scene_passes_filter(june_2024_scene["metadata"], f, "temperate")
    assert not tool._scene_passes_filter(january_2024_scene["metadata"], f, "temperate")


def test_temporal_filter_specific_year_landsat_parity(genesis, june_2024_scene, january_2024_scene):
    """Specific Year was a Landsat-only capability until v1.0.
    Now S2 honours it too: every scene from the chosen year passes,
    regardless of month."""
    tool = genesis.Sentinel2Mosaic()
    f = tool._create_temporal_filter("Specific Year", 2024, None, None)
    assert f["type"] == "specific_year"
    assert tool._scene_passes_filter(june_2024_scene["metadata"], f, "temperate")
    assert tool._scene_passes_filter(january_2024_scene["metadata"], f, "temperate")


def test_temporal_filter_month_all_years(genesis, june_2024_scene, january_2024_scene):
    tool = genesis.Sentinel2Mosaic()
    f = tool._create_temporal_filter("month_all_years", None, 6, None)
    assert tool._scene_passes_filter(june_2024_scene["metadata"], f, "temperate")
    assert not tool._scene_passes_filter(january_2024_scene["metadata"], f, "temperate")


def test_temporal_filter_season_summer(genesis, june_2024_scene, january_2024_scene):
    tool = genesis.Sentinel2Mosaic()
    f = tool._create_temporal_filter("season_all_years", None, None, "summer")
    assert tool._scene_passes_filter(june_2024_scene["metadata"], f, "temperate")
    assert not tool._scene_passes_filter(january_2024_scene["metadata"], f, "temperate")


def test_temporal_filter_season_winter(genesis, june_2024_scene, january_2024_scene):
    tool = genesis.Sentinel2Mosaic()
    f = tool._create_temporal_filter("season_all_years", None, None, "winter")
    # winter = [12, 1, 2]
    assert tool._scene_passes_filter(january_2024_scene["metadata"], f, "temperate")
    assert not tool._scene_passes_filter(june_2024_scene["metadata"], f, "temperate")


def test_seasonal_pattern_for_region(genesis):
    s2 = genesis.Sentinel2Mosaic
    assert s2._seasonal_pattern_for_region("Portugal Mainland") == "temperate"
    assert (
        s2._seasonal_pattern_for_region(
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)"
        ) == "temperate"
    )
    assert s2._seasonal_pattern_for_region("Mozambique") == "mozambique"
    assert s2._seasonal_pattern_for_region("Angola") == "angola"
    assert s2._seasonal_pattern_for_region(
        "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)"
    ) == "cape_verde"


# ---------------------------------------------------------------------------
# Zip archive handling — v1.0 architecture
#
# Pre-v1.0 the toolbox extracted .zip archives to disk before reading
# bands (`_extract_zip_archives` + `_is_safe_zip_member`). v1.0 reads
# zip members in-place via GDAL VSI paths (`/vsizip/...`) so:
#   * No extraction step happens before scene discovery.
#   * No path-traversal threat surface exists (no file is written).
#   * `_list_zip_members` + `_read_safe_xml_from_zip` replace the old
#     extractor for the metadata-side accesses.
# ---------------------------------------------------------------------------


def test_list_zip_members_returns_namelist_for_real_zip(genesis, tmp_path):
    safe_name = "S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE"
    zip_path = tmp_path / f"{safe_name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{safe_name}/MTD_MSIL2A.xml", b"<empty/>")
        zf.writestr(f"{safe_name}/GRANULE/L2A_T29SQB_/IMG_DATA/R10m/foo_B02_10m.jp2", b"x")
    members = genesis.Sentinel2Mosaic._list_zip_members(str(zip_path))
    assert members is not None
    assert any(m.endswith("MTD_MSIL2A.xml") for m in members)
    assert any(m.endswith("_B02_10m.jp2") for m in members)


def test_list_zip_members_returns_none_for_corrupt_zip(genesis, tmp_path):
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip" * 32)
    assert genesis.Sentinel2Mosaic._list_zip_members(str(corrupt)) is None


def test_read_safe_xml_from_zip_returns_bytes_for_present_xml(genesis, tmp_path):
    safe_name = "S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE"
    zip_path = tmp_path / f"{safe_name}.zip"
    payload = b"<n1:Level-2A_User_Product/>"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{safe_name}/MTD_MSIL2A.xml", payload)
    content = genesis.Sentinel2Mosaic._read_safe_xml_from_zip(
        str(zip_path), "MTD_MSIL2A.xml"
    )
    assert content == payload


def test_read_safe_xml_from_zip_returns_none_when_xml_absent(genesis, tmp_path):
    safe_name = "S2A_MSIL2A_xyz.SAFE"
    zip_path = tmp_path / f"{safe_name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{safe_name}/something_else.txt", b"nope")
    content = genesis.Sentinel2Mosaic._read_safe_xml_from_zip(
        str(zip_path), "MTD_MSIL2A.xml"
    )
    assert content is None


# ---------------------------------------------------------------------------
# Provenance CSV
# ---------------------------------------------------------------------------

def test_provenance_csv_includes_tile_id_and_sensor(genesis, tmp_path):
    output_raster = str(tmp_path / "Faial_Summer_2024.tif")
    scenes_used = [
        {
            "path": "D:/data/S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE",
            "metadata": {
                "date_acquired": date(2024, 6, 1),
                "tile_id": "T29SQB",
                "cloud_cover": 12.4,
                "product_uri": "S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE",
            },
        },
        {
            "path": "D:/data/S2B_MSIL2A_20240715T103021_N0500_R108_T29SQB_20240715T142319.SAFE",
            "metadata": {
                "date_acquired": date(2024, 7, 15),
                "tile_id": "T29SQB",
                "cloud_cover": 5.0,
                "product_uri": "S2B_MSIL2A_20240715T103021_N0500_R108_T29SQB_20240715T142319.SAFE",
            },
        },
    ]
    genesis.Sentinel2Mosaic._write_provenance_csv(output_raster, scenes_used)

    csv_path = genesis._sidecar_path_for_raster(output_raster, "_provenance.csv")
    assert os.path.isfile(csv_path)
    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 3  # header + 2 data rows
    header = rows[0]
    assert "tile_id" in header
    assert "scene_id" in header
    # Sensor column must distinguish S2A from S2B based on the filename.
    sensors = [row[1] for row in rows[1:]]
    assert "Sentinel-2A" in sensors
    assert "Sentinel-2B" in sensors


def test_provenance_csv_handles_empty_scenes(genesis, tmp_path):
    output_raster = str(tmp_path / "empty.tif")
    genesis.Sentinel2Mosaic._write_provenance_csv(output_raster, [])
    expected = genesis._sidecar_path_for_raster(output_raster, "_provenance.csv")
    assert not os.path.exists(expected)


def test_provenance_csv_includes_rename_audit_when_supplied(genesis, tmp_path):
    """v1.0 added a rename audit trail to the provenance CSV so a
    post-mortem of an unexpectedly-named output can be done from the
    sidecar. Single rename → header + rows visible after scene rows."""
    output_raster = str(tmp_path / "Faial_V18_PerBandP25.tif")
    scenes_used = [
        {
            "path": "D:/data/S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE",
            "metadata": {
                "date_acquired": date(2024, 6, 1),
                "tile_id": "T29SQB",
                "cloud_cover": 10.0,
                "product_uri": "S2A_MSIL2A_20240601T103021_N0500_R108_T29SQB_20240601T142319.SAFE",
            },
        },
    ]
    rename_log = [
        ("drop_tile_id",
         "Faial_V18_T29SQB_PerBandP25",
         "Faial_V18_PerBandP25"),
        ("drop_masked_suffix",
         "Faial_V18_PerBandP25_Masked",
         "Faial_V18_PerBandP25"),
    ]
    genesis.Sentinel2Mosaic._write_provenance_csv(
        output_raster, scenes_used, rename_log=rename_log,
    )
    csv_path = genesis._sidecar_path_for_raster(
        output_raster, "_provenance.csv",
    )
    with open(csv_path, "r", encoding="utf-8") as fh:
        body = fh.read()
    assert "# rename_audit" in body
    assert "drop_tile_id" in body
    assert "drop_masked_suffix" in body
    assert "Faial_V18_T29SQB_PerBandP25" in body


def test_provenance_csv_skips_rename_audit_when_empty(genesis, tmp_path):
    """No renames happened (e.g., GeometricMedian over a multi-tile
    AOI where the merge created the canonical name directly) → the
    rename_audit section is omitted entirely. Backwards compatible
    with consumers that didn't expect it."""
    output_raster = str(tmp_path / "Faial_V18_Geomedian.tif")
    scenes_used = [
        {
            "path": "D:/data/S2A.SAFE",
            "metadata": {
                "date_acquired": date(2024, 6, 1),
                "tile_id": "T29SQB",
                "cloud_cover": 5.0,
                "product_uri": "S2A_MSIL2A.SAFE",
            },
        },
    ]
    genesis.Sentinel2Mosaic._write_provenance_csv(
        output_raster, scenes_used, rename_log=[],
    )
    csv_path = genesis._sidecar_path_for_raster(
        output_raster, "_provenance.csv",
    )
    with open(csv_path, "r", encoding="utf-8") as fh:
        body = fh.read()
    assert "# rename_audit" not in body


# ---------------------------------------------------------------------------
# Structural integration with the rest of the toolbox
# ---------------------------------------------------------------------------

def test_execute_does_not_pre_extract_archives(genesis):
    """v1.0 contract: zip archives are read in-place via GDAL VSI
    (`/vsizip/...`). There is no extract-to-disk step before scene
    discovery. `_find_safe_scenes` is what execute() calls."""
    src = inspect.getsource(genesis.Sentinel2Mosaic.execute)
    assert "_extract_zip_archives(" not in src
    assert "_is_safe_zip_member(" not in src
    assert "_find_safe_scenes(" in src


def test_execute_uses_esri_geometric_median(genesis):
    """Per user decision: use Esri's arcpy.sa.GeometricMedian for now
    (NumPy implementation is a follow-up if sparse-pixel issues bite)."""
    src = inspect.getsource(genesis.Sentinel2Mosaic.execute)
    assert "GeometricMedian(" in src


def test_execute_writes_provenance_csv(genesis):
    src = inspect.getsource(genesis.Sentinel2Mosaic.execute)
    assert "_write_provenance_csv(" in src


def test_scl_cloud_classes_match_v1_design(genesis):
    """v1.0 SCL mask: 1 (saturated/defective), 3 (shadow), 7 (unclassified),
    8 (cloud med), 9 (cloud high), 10 (thin cirrus), 11 (snow/ice).
    The aggressive default exists because Faial / Azores cloud cover is
    persistent — the cloud_aggressiveness slider lets users dial back."""
    assert genesis._S2_SCL_CLOUD_CLASSES == (1, 3, 7, 8, 9, 10, 11)


def test_s2_stack_order_matches_band_role_mapping(genesis):
    """v1.0 stack carries the full L2A surface-reflectance set including
    B01 (coastal aerosol) and B09 (water vapour). The role map must
    line up 1-indexed — otherwise Tool 04 (Indices) and Tool 06 (SAM)
    would address wrong bands on S2 mosaics."""
    expected = ["B01", "B02", "B03", "B04", "B05", "B06",
                "B07", "B08", "B8A", "B09", "B11", "B12"]
    assert genesis._S2_STACK_ORDER == expected
    role_map = genesis.SENSOR_BAND_ROLES[genesis.SENSOR_SENTINEL2]
    assert role_map["Blue"] == 2   # B02
    assert role_map["Red"] == 4    # B04
    assert role_map["NIR"] == 8    # B08
    assert role_map["SWIR1"] == 11 # B11
    assert role_map["SWIR2"] == 12 # B12


def test_s2_reflectance_scale_factor(genesis):
    """L2A scale factor 0.0001 converts DN to surface reflectance [0, 1]."""
    assert genesis._S2_REFLECTANCE_SCALE == 0.0001
