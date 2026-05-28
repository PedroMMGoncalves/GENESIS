"""Phase 5 tests for Tool 03 (AsterMosaic).

Direct behaviour tests for the pure-Python helpers (filename parsing,
scene discovery, TIFF/HDF grouping, temporal filter, provenance), plus
structural tests confirming the tool is wired into the toolbox.

End-to-end raster processing (resample, QA mask, geometric median)
requires ArcGIS Pro and is not covered here.
"""

from __future__ import annotations

import csv
import inspect
import os
from datetime import date

import pytest


# ---------------------------------------------------------------------------
# Class wiring & metadata
# ---------------------------------------------------------------------------

def test_aster_mosaic_class_present(genesis):
    assert hasattr(genesis, "AsterMosaic")


def test_aster_mosaic_registered_in_toolbox(genesis):
    tb = genesis.Toolbox()
    names = [t.__name__ for t in tb.tools]
    assert "AsterMosaic" in names
    # Workflow ordering: S2, Landsat, ASTER, Indices.
    assert names.index("Sentinel2Mosaic") < names.index("AsterMosaic")
    assert names.index("LandsatMosaic") < names.index("AsterMosaic")
    assert names.index("AsterMosaic") < names.index("IndicesComposites")


def test_aster_mosaic_label_uses_03_prefix(genesis):
    tool = genesis.AsterMosaic()
    assert tool.label.startswith("03 —")


def test_description_mentions_aster_and_qa(genesis):
    desc = genesis.AsterMosaic().description.lower()
    assert "aster" in desc
    assert "ast_07xt" in desc or "v004" in desc
    assert "cloud" in desc
    # Compositor menu surfaced in v1.0 — the description must still name
    # at least one supported reducer so users know what they're getting.
    assert "geometric" in desc or "median" in desc or "percentile" in desc


def test_getparameterinfo_returns_full_aster_v1_parameter_set(genesis):
    """ASTER mosaic grew through Phases 4–6 (AROSICS, DL cloud mask,
    compositor menu, temporal cleaner). Look up by name rather than
    by index so we stay robust to UI reordering."""
    params = genesis.AsterMosaic().getParameterInfo()
    names = [p.name for p in params]

    # Must-have v1.0 surface — the compositor and temporal cleaner
    # are the lever-points users actually touch.
    for required in (
        "gdb_path",
        "mosaic_name",
        "data_folder",
        "arosics_reference",
        "use_qa_planes",
        "enable_temporal_clean",
        "compositor",
        "percentile_value",
        "dl_mask_folder",
    ):
        assert required in names, f"missing required ASTER parameter: {required}"


def test_qa_planes_default_on(genesis):
    """QA Data Plane mask defaults on. Look up by name — UI position
    has shifted across releases."""
    params = genesis.AsterMosaic().getParameterInfo()
    qa_param = next(p for p in params if p.name == "use_qa_planes")
    assert qa_param.value is True


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def test_parse_tiff_filename_vnir_band(genesis):
    """The user's example file from the Mozambique scene screenshot."""
    name = "AST_07XT_00409282006074522_20250531042133_SRF_VNIR_B01.tif"
    parsed = genesis.AsterMosaic._parse_aster_filename(name)
    assert parsed is not None
    assert parsed["scene_id"] == "00409282006074522"
    assert parsed["pass"] == "004"
    assert parsed["acquisition_date"] == date(2006, 9, 28)
    assert parsed["group"] == "VNIR"
    assert parsed["band"] == "B01"


def test_parse_tiff_filename_b03n_nadir(genesis):
    """B03N (nadir-looking band 3) — distinct from B03B (backward)."""
    name = "AST_07XT_00409282006074522_20250531042133_SRF_VNIR_B03N.tif"
    parsed = genesis.AsterMosaic._parse_aster_filename(name)
    assert parsed is not None
    assert parsed["band"] == "B03N"


def test_parse_tiff_filename_swir_band(genesis):
    name = "AST_07XT_00409282006074522_20250531042134_SRF_SWIR_B07.tif"
    parsed = genesis.AsterMosaic._parse_aster_filename(name)
    assert parsed is not None
    assert parsed["group"] == "SWIR"
    assert parsed["band"] == "B07"


def test_parse_tiff_filename_qa_data_plane(genesis):
    name = "AST_07XT_00409282006074522_20250531042133_SRF_VNIR_QA_DataPlane.tif"
    parsed = genesis.AsterMosaic._parse_aster_filename(name)
    assert parsed is not None
    assert parsed["band"] == "QA_DataPlane"


def test_parse_tiff_filename_qa_data_plane2(genesis):
    name = "AST_07XT_00409282006074522_20250531042134_SRF_SWIR_QA_DataPlane2.tif"
    parsed = genesis.AsterMosaic._parse_aster_filename(name)
    assert parsed is not None
    assert parsed["band"] == "QA_DataPlane2"


def test_parse_hdf_filename(genesis):
    name = "AST_07XT_00409282006074522_20250531042133.hdf"
    parsed = genesis.AsterMosaic._parse_aster_filename(name)
    assert parsed is not None
    assert parsed["scene_id"] == "00409282006074522"
    assert parsed["acquisition_date"] == date(2006, 9, 28)
    assert parsed["group"] is None  # HDF doesn't have a group suffix
    assert parsed["band"] is None


@pytest.mark.parametrize("name", [
    "random_file.tif",
    "AST_07XT_too_short.tif",
    "LC08_L2SR_204032_20240215.tar",   # Landsat, not ASTER
    "S2A_MSIL2A_20240601T103021.SAFE",  # Sentinel-2, not ASTER
])
def test_parse_filename_rejects_non_aster(genesis, name):
    assert genesis.AsterMosaic._parse_aster_filename(name) is None


def test_parse_filename_rejects_invalid_date(genesis):
    """The acquisition date is derived from filename digits — month 13
    must be rejected, not produce a garbage date."""
    name = "AST_07XT_00413282006074522_20250531042133_SRF_VNIR_B01.tif"
    assert genesis.AsterMosaic._parse_aster_filename(name) is None


# ---------------------------------------------------------------------------
# Scene discovery / grouping
# ---------------------------------------------------------------------------

def _write_aster_scene_tiffs(folder, scene_id="00409282006074522",
                              proc_dt_vnir="20250531042133",
                              proc_dt_swir="20250531042134"):
    """Build the standard 13-file ASTER scene (9 bands + 2 QA × 2 groups)
    in a folder. Returns the list of created filenames."""
    files = []
    # VNIR group (3 bands + 2 QA, same proc datetime)
    for tail in ("B01", "B02", "B03N", "QA_DataPlane", "QA_DataPlane2"):
        f = f"AST_07XT_{scene_id}_{proc_dt_vnir}_SRF_VNIR_{tail}.tif"
        (folder / f).write_bytes(b"")
        files.append(f)
    # SWIR group (6 bands + 2 QA)
    for tail in ("B04", "B05", "B06", "B07", "B08", "B09", "QA_DataPlane", "QA_DataPlane2"):
        f = f"AST_07XT_{scene_id}_{proc_dt_swir}_SRF_SWIR_{tail}.tif"
        (folder / f).write_bytes(b"")
        files.append(f)
    return files


def test_find_aster_scenes_groups_files_by_scene_id(genesis, tmp_path):
    """Every file with the same 17-char sceneID is one scene, regardless of
    differing processing datetimes between VNIR and SWIR groups."""
    _write_aster_scene_tiffs(tmp_path)
    scenes = genesis.AsterMosaic._find_aster_scenes(str(tmp_path))
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene["scene_id"] == "00409282006074522"
    assert scene["format"] == "tiff"
    # Must contain all 9 image bands.
    for band in ("B01", "B02", "B03N", "B04", "B05", "B06", "B07", "B08", "B09"):
        assert band in scene["files"], f"missing {band} in files dict"


def test_find_aster_scenes_keeps_two_scenes_separate(genesis, tmp_path):
    """Files from two different sceneIDs become two scenes (matches the
    user's screenshot: one folder, multiple Mozambique scenes)."""
    _write_aster_scene_tiffs(tmp_path, scene_id="00409282006074522")
    _write_aster_scene_tiffs(
        tmp_path, scene_id="00409282006074531",
        proc_dt_vnir="20250531041544", proc_dt_swir="20250531041545",
    )
    scenes = genesis.AsterMosaic._find_aster_scenes(str(tmp_path))
    assert len(scenes) == 2
    ids = {s["scene_id"] for s in scenes}
    assert ids == {"00409282006074522", "00409282006074531"}


def test_find_aster_scenes_prefers_tiff_over_hdf_for_same_scene(genesis, tmp_path):
    """If a sceneID has both TIFFs and an HDF in the folder, the TIFF
    set wins — it's pre-extracted, cheaper to read."""
    _write_aster_scene_tiffs(tmp_path, scene_id="00409282006074522")
    (tmp_path / "AST_07XT_00409282006074522_20250531042133.hdf").write_bytes(b"hdf-bytes")
    scenes = genesis.AsterMosaic._find_aster_scenes(str(tmp_path))
    assert len(scenes) == 1
    assert scenes[0]["format"] == "tiff"


def test_find_aster_scenes_handles_hdf_only(genesis, tmp_path):
    """HDF without sibling TIFFs is its own scene."""
    (tmp_path / "AST_07XT_00409282006074522_20250531042133.hdf").write_bytes(b"hdf")
    scenes = genesis.AsterMosaic._find_aster_scenes(str(tmp_path))
    assert len(scenes) == 1
    assert scenes[0]["format"] == "hdf"
    assert "hdf" in scenes[0]["files"]


def test_find_aster_scenes_handles_empty_folder(genesis, tmp_path):
    assert genesis.AsterMosaic._find_aster_scenes(str(tmp_path)) == []


def test_find_aster_scenes_handles_missing_folder(genesis):
    assert genesis.AsterMosaic._find_aster_scenes(None) == []
    assert genesis.AsterMosaic._find_aster_scenes("/no/such/path") == []


def test_find_aster_scenes_extracts_acquisition_date_per_scene(genesis, tmp_path):
    _write_aster_scene_tiffs(tmp_path, scene_id="00409282006074522")
    scenes = genesis.AsterMosaic._find_aster_scenes(str(tmp_path))
    assert scenes[0]["metadata"]["acquisition_date"] == date(2006, 9, 28)


# ---------------------------------------------------------------------------
# Temporal filter (same shape as S2/Landsat, but using ASTER's
# acquisition_date metadata)
# ---------------------------------------------------------------------------

@pytest.fixture
def aster_september_2006_meta():
    return {"acquisition_date": date(2006, 9, 28), "scene_id": "x"}


@pytest.fixture
def aster_january_2019_meta():
    return {"acquisition_date": date(2019, 1, 15), "scene_id": "y"}


def test_temporal_filter_year_month(genesis, aster_september_2006_meta, aster_january_2019_meta):
    tool = genesis.AsterMosaic()
    f = tool._create_temporal_filter("year_month", 2006, 9, None)
    assert tool._scene_passes_filter(aster_september_2006_meta, f, "temperate")
    assert not tool._scene_passes_filter(aster_january_2019_meta, f, "temperate")


def test_temporal_filter_season_summer(genesis, aster_september_2006_meta, aster_january_2019_meta):
    tool = genesis.AsterMosaic()
    f = tool._create_temporal_filter("season_all_years", None, None, "summer")
    # September is autumn in the temperate pattern — not summer.
    assert not tool._scene_passes_filter(aster_september_2006_meta, f, "temperate")
    f = tool._create_temporal_filter("season_all_years", None, None, "autumn")
    assert tool._scene_passes_filter(aster_september_2006_meta, f, "temperate")


def test_temporal_filter_season_in_mozambique(genesis):
    """Mozambique uses a tropical wet/dry season pattern, not the temperate one."""
    tool = genesis.AsterMosaic()
    july_meta = {"acquisition_date": date(2018, 7, 15)}
    f = tool._create_temporal_filter("season_all_years", None, None, "dry")
    # Mozambique dry = [4, 5, 6, 7, 8, 9] — July is dry.
    assert tool._scene_passes_filter(july_meta, f, "mozambique")
    # But it's NOT dry in the temperate calendar (no 'dry' key there).
    assert not tool._scene_passes_filter(july_meta, f, "temperate")


# ---------------------------------------------------------------------------
# Provenance CSV
# ---------------------------------------------------------------------------

def test_provenance_csv_columns(genesis, tmp_path):
    output_raster = str(tmp_path / "Faial_ASTER_2006.tif")
    scenes_used = [
        {
            "scene_id": "00409282006074522",
            "format": "tiff",
            "files": {"B01": "D:/data/AST_07XT_00409282006074522_..._SRF_VNIR_B01.tif"},
            "metadata": {
                "acquisition_date": date(2006, 9, 28),
                "pass_number": "004",
                "scene_id": "00409282006074522",
            },
        },
    ]
    genesis.AsterMosaic._write_provenance_csv(output_raster, scenes_used)
    # Sidecar lives alongside the raster with the extension stripped —
    # use the helper so the test follows the canonical convention.
    csv_path = genesis._sidecar_path_for_raster(output_raster, "_provenance.csv")
    assert os.path.isfile(csv_path)

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 2  # header + 1 data row
    header = rows[0]
    for col in ("scene_id", "sensor", "acquisition_datetime",
                "pass_number", "input_format", "processing_baseline"):
        assert col in header
    data = rows[1]
    assert data[0] == "00409282006074522"
    assert "ASTER" in data[1]
    assert data[3] == "004"  # pass number


def test_provenance_csv_hdf_picks_hdf_path(genesis, tmp_path):
    """For HDF scenes the input_path column reflects the .hdf file, not a band."""
    output_raster = str(tmp_path / "x.tif")
    scenes_used = [{
        "scene_id": "00409282006074522",
        "format": "hdf",
        "files": {"hdf": "D:/data/AST_07XT_00409282006074522_20250531042133.hdf"},
        "metadata": {
            "acquisition_date": date(2006, 9, 28),
            "pass_number": "004",
        },
    }]
    genesis.AsterMosaic._write_provenance_csv(output_raster, scenes_used)
    csv_path = genesis._sidecar_path_for_raster(output_raster, "_provenance.csv")
    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    fmt_idx = header.index("input_format")
    path_idx = header.index("input_path")
    assert rows[1][fmt_idx] == "hdf"
    assert rows[1][path_idx].endswith(".hdf")


# ---------------------------------------------------------------------------
# Module-level constants & stack-order contract
# ---------------------------------------------------------------------------

def test_aster_stack_order_matches_band_role_mapping(genesis):
    """The ASTER stack order in this tool must match the band-role mapping
    declared in Phase 1 — otherwise Tool 04 (Indices) would see wrong
    band assignments when run on ASTER mosaics."""
    expected = ["B01", "B02", "B03N", "B04", "B05", "B06", "B07", "B08", "B09"]
    assert genesis._ASTER_STACK_ORDER == expected
    role_map = genesis.SENSOR_BAND_ROLES[genesis.SENSOR_ASTER]
    assert role_map["Green"] == 1   # B01
    assert role_map["Red"] == 2     # B02
    assert role_map["NIR"] == 3     # B03N
    assert role_map["SWIR1"] == 4   # B04 (1.6μm)
    assert role_map["SWIR2_2165"] == 5  # B05 (2.165μm)
    assert role_map["SWIR2_2330"] == 8  # B08 (2.330μm)
    assert role_map["SWIR2"] == 9   # B09 (2.395μm)


def test_aster_reflectance_scale_factor(genesis):
    """AST_07XT V004 scale factor: 0.001 converts DN to reflectance [0, 1]."""
    assert genesis._ASTER_REFLECTANCE_SCALE == 0.001


def test_aster_native_vs_resampled_band_groups(genesis):
    """VNIR is 15m native, SWIR is 30m → resampled. The lookup pins which
    bands skip the resample step."""
    assert genesis._ASTER_NATIVE_15M == {"B01", "B02", "B03N"}


# ---------------------------------------------------------------------------
# Structural integration
# ---------------------------------------------------------------------------

def test_execute_wires_pipeline_helpers(genesis):
    """In v1.0 the execute() body is thin — it discovers scenes and
    delegates to _run_mosaic_pipeline. The GeometricMedian compositor
    is still wired here; provenance writing migrated to the pipeline
    helper. Verify both surfaces independently."""
    execute_src = inspect.getsource(genesis.AsterMosaic.execute)
    pipeline_src = inspect.getsource(genesis.AsterMosaic._run_mosaic_pipeline)
    assert "_find_aster_scenes(" in execute_src
    # Compositor reaches at least one of these reducers somewhere in
    # the call chain. GeometricMedian is the legacy default; the v1.0
    # compositor menu adds CellStatistics-based reducers.
    combined = execute_src + pipeline_src
    assert "GeometricMedian(" in combined or "CellStatistics(" in combined
    assert "_write_provenance_csv(" in combined


def test_execute_offers_temporal_outlier_cleaner(genesis):
    """ASTER mosaic includes a Tmask-style robust temporal cleaner
    (default ON since commit 6259eff). Look for the new name —
    _temporal_outlier_clean — anywhere on the AsterMosaic class."""
    pipeline_src = inspect.getsource(genesis.AsterMosaic._run_mosaic_pipeline)
    assert "_temporal_outlier_clean(" in pipeline_src or "enable_temporal_clean" in pipeline_src


def test_per_scene_pipeline_resamples_swir_to_15m(genesis):
    src = inspect.getsource(genesis.AsterMosaic._process_scene)
    # SWIR bands get a Resample call.
    assert "Resample(" in src
    # BILINEAR for continuous reflectance.
    assert "BILINEAR" in src
    # Scale factor applied.
    assert "_ASTER_REFLECTANCE_SCALE" in src


def test_per_scene_pipeline_skips_resample_for_native_15m(genesis):
    """VNIR (B01, B02, B03N) is 15m native — must NOT be resampled."""
    src = inspect.getsource(genesis.AsterMosaic._process_scene)
    assert "_ASTER_NATIVE_15M" in src  # the guard reference must be there
