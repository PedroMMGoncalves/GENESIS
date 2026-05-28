"""Phase 3 tests for Tool 02 (LandsatMosaic).

Covers structural integration (class registered, audit fixes carried
over, tar + provenance wiring) plus direct behaviour tests for the two
new pure-Python helpers (tar extraction + provenance CSV) — these don't
need arcpy.
"""

from __future__ import annotations

import csv
import inspect
import io
import os
import tarfile

import pytest


# ---------------------------------------------------------------------------
# Class wiring & metadata
# ---------------------------------------------------------------------------

def test_landsat_mosaic_class_present(genesis):
    assert hasattr(genesis, "LandsatMosaic")


def test_landsat_mosaic_registered_in_toolbox(genesis):
    tb = genesis.Toolbox()
    names = [t.__name__ for t in tb.tools]
    assert "LandsatMosaic" in names


def test_landsat_mosaic_label_uses_workflow_prefix(genesis):
    tool = genesis.LandsatMosaic()
    assert tool.label.startswith("02 —"), (
        f"Tool 02 label must use the workflow prefix; got {tool.label!r}"
    )


def test_landsat_mosaic_description_mentions_provenance(genesis):
    desc = genesis.LandsatMosaic().description.lower()
    assert "provenance" in desc
    assert "l2sr" in desc or "l2sp" in desc, (
        "Description should mention the C2L2 variant naming"
    )


# ---------------------------------------------------------------------------
# Bug 2 + Bug 7 audit fixes carried over
# ---------------------------------------------------------------------------

def test_audit_bug2_buildseamlines_linear_present(genesis):
    """Bug 2: BuildSeamlines blend_type must be 'LINEAR' (not the
    typo 'LINER')."""
    src = inspect.getsource(genesis.LandsatMosaic._merge_zone_mosaics)
    assert 'blend_type="LINEAR"' in src
    assert 'blend_type="LINER"' not in src


def test_audit_bug7_apply_mask_returns_none_on_failure(genesis):
    """Bug 7: _apply_mask must return None on failure (not the original
    unmasked path) so callers can detect that masking was skipped."""
    src = inspect.getsource(genesis.LandsatMosaic._apply_mask)
    assert "return None" in src
    # Exactly one `return mosaic_path` should survive — the legitimate
    # "no mask requested" early-return.
    assert src.count("return mosaic_path") == 1, (
        "After Bug 7 fix, only the no-mask-requested branch should "
        "`return mosaic_path`; failure branches must return None."
    )


# ---------------------------------------------------------------------------
# Archive handling — v1.0 architecture
#
# Pre-v1.0 the toolbox extracted .tar archives to disk before reading
# bands (`_extract_tar_archives` + `_is_safe_tar_member`). v1.0 reads
# archive members in-place via GDAL VSI paths (`/vsitar/...`) so:
#   * No extraction step happens before the UTM-zone loop.
#   * No path-traversal threat surface exists (no file is written).
#   * Two helpers replace the old extractor: `_validate_tar_file` for
#     integrity, `_find_band_files_vsi` for in-place band lookup.
# ---------------------------------------------------------------------------

def _make_landsat_tar(tar_path, scene_name, include_qa=True):
    """Build a minimal Landsat-shaped .tar for the in-place reader tests.

    Includes the _MTL.txt sidecar (needed by `_derive_scene_id_from_archive`)
    and, when requested, a QA_PIXEL band so `_find_band_files_vsi` returns
    a non-empty dict.
    """
    mtl = (
        b"GROUP = LANDSAT_METADATA_FILE\n"
        b"  DATE_ACQUIRED = 2024-05-02\n"
        b"END_GROUP = LANDSAT_METADATA_FILE\n"
    )
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name=f"{scene_name}_MTL.txt")
        info.size = len(mtl)
        tar.addfile(info, io.BytesIO(mtl))
        if include_qa:
            qa = b"FAKEQA"
            info2 = tarfile.TarInfo(name=f"{scene_name}_QA_PIXEL.TIF")
            info2.size = len(qa)
            tar.addfile(info2, io.BytesIO(qa))


def test_validate_tar_file_accepts_real_archive(genesis, tmp_path):
    tar_path = tmp_path / "LC09_L2SR_217033_20240502_20240514_02_T1.tar"
    _make_landsat_tar(tar_path, "LC09_L2SR_217033_20240502_20240514_02_T1")
    ok, err = genesis.LandsatMosaic._validate_tar_file(str(tar_path))
    assert ok, f"valid archive rejected: {err}"
    assert err == ""


def test_validate_tar_file_rejects_missing_file(genesis):
    ok, err = genesis.LandsatMosaic._validate_tar_file("/nope/missing.tar")
    assert not ok
    assert "does not exist" in err


def test_validate_tar_file_rejects_empty_file(genesis, tmp_path):
    empty = tmp_path / "empty.tar"
    empty.write_bytes(b"")
    ok, err = genesis.LandsatMosaic._validate_tar_file(str(empty))
    assert not ok
    assert "empty" in err


def test_validate_tar_file_rejects_garbage(genesis, tmp_path):
    """Random bytes that aren't a tar header → graceful rejection."""
    bogus = tmp_path / "bogus.tar"
    bogus.write_bytes(b"not a tar archive at all" * 64)
    ok, err = genesis.LandsatMosaic._validate_tar_file(str(bogus))
    assert not ok
    assert "tar" in err.lower() or "header" in err.lower() or "valid" in err.lower()


def test_find_band_files_vsi_returns_paths_for_real_tar(genesis, tmp_path):
    """The in-place archive reader must map present band files to
    /vsitar/... paths without extracting anything."""
    tool = genesis.LandsatMosaic()
    scene = "LC09_L2SR_217033_20240502_20240514_02_T1"
    tar_path = tmp_path / f"{scene}.tar"
    _make_landsat_tar(tar_path, scene)
    band_paths = tool._find_band_files_vsi(str(tar_path), scene)
    assert band_paths is not None
    assert "QA_PIXEL" in band_paths


def test_find_band_files_vsi_paths_use_vsitar_scheme(genesis, tmp_path):
    """Pins the GDAL VSI contract — tar members open via the /vsitar/
    prefix, zip members via /vsizip/. Forward slashes throughout."""
    tool = genesis.LandsatMosaic()
    scene = "LC08_L2SR_217033_20240215_20240301_02_T1"
    tar_path = tmp_path / f"{scene}.tar"
    _make_landsat_tar(tar_path, scene)
    band_paths = tool._find_band_files_vsi(str(tar_path), scene)
    # Every returned path must use the /vsitar/ scheme.
    for role, path in band_paths.items():
        assert path.startswith("/vsitar/"), (
            f"{role}: expected /vsitar/ prefix, got {path!r}"
        )


def test_find_band_files_vsi_returns_none_for_corrupt_tar(genesis, tmp_path):
    tool = genesis.LandsatMosaic()
    corrupt = tmp_path / "LC09_L2SR_corrupt.tar"
    corrupt.write_bytes(b"corrupt garbage" * 1024)
    band_paths = tool._find_band_files_vsi(str(corrupt), "LC09_L2SR_corrupt")
    assert band_paths is None


def test_execute_does_not_pre_extract_archives(genesis):
    """v1.0 contract: there is no archive-extraction step before the
    UTM-zone loop. Scenes are read in-place inside `_find_scenes` via
    `_find_band_files_vsi`. If a future change reintroduces a
    pre-extraction pass this test will trip — at which point think
    hard about whether the path-traversal threat needs to come back."""
    execute_src = inspect.getsource(genesis.LandsatMosaic.execute)
    assert "_extract_tar_archives(" not in execute_src
    assert "_is_safe_tar_member(" not in execute_src
    find_scenes_src = inspect.getsource(genesis.LandsatMosaic._find_scenes)
    assert "_find_band_files_vsi(" in find_scenes_src


# ---------------------------------------------------------------------------
# Provenance CSV
# ---------------------------------------------------------------------------

def test_execute_calls_provenance_writer(genesis):
    src = inspect.getsource(genesis.LandsatMosaic.execute)
    assert "_write_provenance_csv(" in src


def test_provenance_csv_columns_documented(genesis):
    """The writer must include all the documented columns. Source-grep
    keeps the column list pinned."""
    src = inspect.getsource(genesis.LandsatMosaic._write_provenance_csv)
    for col in (
        "scene_id", "sensor", "acquisition_datetime", "path_row",
        "cloud_cover_pct", "input_path", "processing_baseline",
        "toolbox_version", "processing_datetime",
    ):
        assert col in src, f"Provenance writer missing column {col!r}"


def test_write_provenance_csv_produces_valid_csv(genesis, tmp_path):
    """End-to-end on the pure-Python helper: build a scenes list, run
    the writer, parse the output, verify column count and row count."""
    tool = genesis.LandsatMosaic()
    output_raster = str(tmp_path / "Faial_Mosaic_2024.tif")

    scenes_used = [
        {
            "path": "D:/data/LC09_L2SR_217033_20240502_20240514_02_T1",
            "metadata": {
                "date_acquired": "2024-05-02",
                "path_row": "217_033",
                "cloud_cover": "12.4",
            },
        },
        {
            "path": "D:/data/LC08_L2SR_217033_20240218_20240301_02_T1",
            "metadata": {
                "date_acquired": "2024-02-18",
                "path_row": "217_033",
                "cloud_cover": "5.7",
            },
        },
    ]

    tool._write_provenance_csv(output_raster, scenes_used, {})

    csv_path = genesis._sidecar_path_for_raster(output_raster, "_provenance.csv")
    assert os.path.isfile(csv_path), "Provenance CSV not written"

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    # Header + 2 scene rows
    assert len(rows) == 3
    header = rows[0]
    assert "scene_id" in header
    assert "acquisition_datetime" in header
    assert "toolbox_version" in header

    # Sensor column must distinguish L8 from L9 based on the filename.
    scene_ids = [row[0] for row in rows[1:]]
    sensors = [row[1] for row in rows[1:]]
    assert "Landsat 9" in sensors and "Landsat 8" in sensors


def test_write_provenance_csv_handles_empty_input(genesis, tmp_path):
    """No scenes → no file written (and no exception)."""
    tool = genesis.LandsatMosaic()
    output_raster = str(tmp_path / "empty_mosaic.tif")
    tool._write_provenance_csv(output_raster, [], {})
    expected = genesis._sidecar_path_for_raster(output_raster, "_provenance.csv")
    assert not os.path.exists(expected)


def test_write_provenance_csv_failure_is_nonfatal(genesis, tmp_path):
    """Writing to an unwritable path must not raise — provenance is
    nice-to-have, not blocking."""
    tool = genesis.LandsatMosaic()
    # Unwritable target: a path inside a nonexistent directory.
    bad_output = "/nonexistent_dir_xyz/no/way/this/works.tif"
    # Should not raise.
    tool._write_provenance_csv(bad_output, [{"path": "x", "metadata": {}}], {})


# ---------------------------------------------------------------------------
# Audit-finding regression tests (Phase 3 "option 2a" leftovers)
# ---------------------------------------------------------------------------

def test_audit_a_clean_scenes_scoping_uses_accumulated_count(genesis):
    """Bug A: the multi-zone stats line used to read `clean_scenes` from
    the last loop iteration, silently dropping earlier zones' counts.
    The fix uses the accumulated `all_scenes_used` list."""
    src = inspect.getsource(genesis.LandsatMosaic.execute)
    # The broken pattern must be gone.
    assert "sum(1 for scene in clean_scenes) if 'clean_scenes' in locals()" not in src
    # The fix uses len(all_scenes_used).
    assert "len(all_scenes_used)" in src


def test_audit_b_parse_metadata_no_indexerror_on_missing_field(genesis, tmp_path):
    """Bug B: _parse_metadata used `[0]` directly on filter results,
    raising IndexError on partial MTLs (which the broad except swallowed,
    silently dropping the scene). After the fix, missing fields default
    to None / 0 and only an absent acquisition date triggers a drop."""
    tool = genesis.LandsatMosaic()

    # Intentionally incomplete MTL: no CLOUD_COVER, no UTM_ZONE.
    mtl = tmp_path / "LC09_L2SR_PARTIAL_MTL.txt"
    mtl.write_text(
        "GROUP = L1_METADATA_FILE\n"
        "    DATE_ACQUIRED = 2024-06-15\n"
        "    PROCESSING_LEVEL = \"L2SR\"\n"
        "END_GROUP = L1_METADATA_FILE\n"
    )

    info = tool._parse_metadata(str(mtl))
    assert info is not None, "Parser must NOT drop a scene just because some fields are missing"
    assert info["acquisition_date"].year == 2024
    assert info["cloud_cover"] == 0.0  # missing → default 0.0
    assert info["utm_zone"] is None     # missing → None


def test_audit_b_parse_metadata_drops_only_when_date_missing(genesis, tmp_path):
    """The one field we DO require is DATE_ACQUIRED — without it we can't
    apply the temporal filter, so dropping is correct."""
    tool = genesis.LandsatMosaic()
    mtl = tmp_path / "LC09_L2SR_NODATE_MTL.txt"
    mtl.write_text(
        "GROUP = L1_METADATA_FILE\n"
        "    CLOUD_COVER = 12.0\n"
        "    UTM_ZONE = 29\n"
        "END_GROUP = L1_METADATA_FILE\n"
    )
    assert tool._parse_metadata(str(mtl)) is None


def test_audit_b_parse_metadata_doesnt_confuse_cloud_cover_with_cloud_cover_land(genesis, tmp_path):
    """Bug B sub-finding: `'CLOUD_COVER' in line` matched both
    CLOUD_COVER and CLOUD_COVER_LAND non-deterministically. The fix
    uses exact-token key matching."""
    tool = genesis.LandsatMosaic()
    # In this MTL, CLOUD_COVER_LAND appears BEFORE CLOUD_COVER. The
    # pre-fix code would have grabbed CLOUD_COVER_LAND's value.
    mtl = tmp_path / "LC09_L2SR_DOUBLECC_MTL.txt"
    mtl.write_text(
        "GROUP = L1_METADATA_FILE\n"
        "    DATE_ACQUIRED = 2024-06-15\n"
        "    CLOUD_COVER_LAND = 99.0\n"
        "    CLOUD_COVER = 5.0\n"
        "END_GROUP = L1_METADATA_FILE\n"
    )
    info = tool._parse_metadata(str(mtl))
    assert info is not None
    assert info["cloud_cover"] == 5.0, (
        f"CLOUD_COVER must take precedence over CLOUD_COVER_LAND; "
        f"got {info['cloud_cover']}"
    )


def test_audit_c_updateparameters_caches_years_per_folder(genesis):
    """Bug C: updateParameters used to walk the data folder on every
    parameter change (every keystroke). The fix caches (folder, years)
    on self and only rescans when the folder path changes."""
    src = inspect.getsource(genesis.LandsatMosaic.updateParameters)
    assert "_years_cache" in src, (
        "updateParameters must cache the year scan to avoid re-walking "
        "the disk on every keystroke."
    )


def test_audit_d_merge_zone_mosaics_uses_first_zone_crs(genesis):
    """Bug D: the merged dataset was forced into WGS 84 while
    BuildSeamlines was called with cell_size=30 (which meant 30 *degrees*
    — global-scale). The fix uses the first zone mosaic's projected CRS
    so 30 stays in metres."""
    src = inspect.getsource(genesis.LandsatMosaic._merge_zone_mosaics)
    # The broken pattern (hardcoded 4326 as the primary coordinate_system).
    assert "coordinate_system=4326" not in src
    # The fix uses the first zone's spatial reference.
    assert "Describe(zone_mosaics[0]).spatialReference" in src
    # cell_size in BuildSeamlines stays at 30 — now correctly metres.
    assert "cell_size=30" in src


def test_audit_e_geometric_median_cleans_temps_on_failure(genesis):
    """Bug E: temp composites built before GeometricMedian were only
    deleted in the success path. Exceptions left orphans in the GDB.
    The fix moves cleanup into a try/finally. In v1.0 the cleanup loop
    was extracted into `_cleanup_scratch_folder` — verify the call
    lives inside the finally block."""
    src = inspect.getsource(genesis.LandsatMosaic._create_geometric_median_mosaic)
    assert "finally:" in src
    assert "_cleanup_scratch_folder(" in src
    finally_pos = src.rindex("finally:")
    cleanup_pos = src.rindex("_cleanup_scratch_folder(")
    assert cleanup_pos > finally_pos, (
        "_cleanup_scratch_folder must live inside the finally block, "
        "not the try success path."
    )
