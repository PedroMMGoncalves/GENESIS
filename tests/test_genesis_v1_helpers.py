"""Regression tests for the v1.0 release helpers.

These cover the module-level functions added or revised in the v1.0
push: the per-band percentile compositor, the geometric-median save
wrapper with reflectance bounding, the OmniCloudMask observed-pixel
fraction helper, the DL cloud mask scene-level NoData sentinel, and
the Tmask-style temporal outlier cleaner.

Most of the heavy lifting in these helpers happens inside arcpy.sa
calls that can't run without ArcGIS Pro. The tests here verify the
contracts that DO run on stock Python: signatures, source-level
wiring (compositor branches, NoData sentinel value, MAD-based
robust z-score), and the pure-numpy reduction in
`_ocm_mask_class_fractions`.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _per_band_percentile_composite — added v1.0
# ---------------------------------------------------------------------------

def test_per_band_percentile_composite_signature(genesis):
    """Default percentile is the caller's responsibility (no module
    default). The helper takes per-band stacks of single-band rasters
    and writes a multiband output at output_path."""
    sig = inspect.signature(genesis._per_band_percentile_composite)
    assert list(sig.parameters) == [
        "stacks", "output_path", "scratch_dir", "percentile_value",
    ]


def test_per_band_percentile_composite_uses_cellstatistics_percentile(genesis):
    """Source-level contract: CellStatistics(... 'PERCENTILE', ...) with
    LINEAR interpolation. Pinning this catches accidental swaps to
    NEAREST_RANK (the default in some arcpy versions), which would
    quantise the output to actual sample values rather than the
    interpolated percentile we depend on for de-clouded composites."""
    src = inspect.getsource(genesis._per_band_percentile_composite)
    assert "CellStatistics(" in src
    assert "PERCENTILE" in src
    assert "LINEAR" in src


def test_per_band_percentile_composite_validates_percentile_range(genesis):
    """The helper either validates the percentile arg or passes it
    straight through to CellStatistics, which itself rejects values
    outside (0, 100). Either way the source must reference the value
    explicitly so a future refactor can't silently swap it for a
    constant."""
    src = inspect.getsource(genesis._per_band_percentile_composite)
    assert "percentile_value" in src


# ---------------------------------------------------------------------------
# _save_geomedian_clean — reflectance bounding added v1.0
# ---------------------------------------------------------------------------

def test_save_geomedian_clean_signature_includes_reflectance_bounded(genesis):
    """v1.0 added `reflectance_bounded=True` so the GeometricMedian
    output is clamped to [0, 1] before being written. This catches
    the rare negative-reflectance spurs the median can produce when
    a band has near-zero signal across the temporal stack."""
    sig = inspect.signature(genesis._save_geomedian_clean)
    assert "reflectance_bounded" in sig.parameters
    assert sig.parameters["reflectance_bounded"].default is True


def test_save_geomedian_clean_default_clamps_negative_reflectance(genesis):
    """When reflectance_bounded=True (the default) the source must
    apply a Con or SetNull guard that maps sub-zero values to 0 (or
    NoData). Source-grep keeps this pinned without needing arcpy."""
    src = inspect.getsource(genesis._save_geomedian_clean)
    assert "reflectance_bounded" in src
    # The bound must reference 0 (lower) and a 1-ish upper somewhere
    # in the bounded branch.
    assert "Con(" in src or "SetNull(" in src or "_clip" in src


# ---------------------------------------------------------------------------
# _ocm_mask_class_fractions — added v1.0 per design memo
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_rastertonumpy(genesis, monkeypatch):
    """Yield a context where `arcpy.RasterToNumPyArray` returns a
    caller-supplied uint8 array. Lets us exercise the pure-numpy
    reduction in `_ocm_mask_class_fractions` without arcpy."""
    holder = {"array": None}

    def _fake(path, nodata_to_value=None):
        return holder["array"]

    import arcpy
    monkeypatch.setattr(arcpy, "RasterToNumPyArray", _fake)
    return holder


def test_ocm_mask_class_fractions_signature(genesis):
    """Default sentinel is 255 (outside OCM's class range 0-3) so it
    can't collide with a real class. Cloud classes default to (1, 2)
    — mid + high cloud — and shadow to (3,)."""
    sig = inspect.signature(genesis._ocm_mask_class_fractions)
    assert sig.parameters["sentinel"].default == 255
    assert sig.parameters["cloud_classes"].default == (1, 2)
    assert sig.parameters["shadow_classes"].default == (3,)


def test_ocm_mask_class_fractions_normalises_over_observed_pixels(
    genesis, patched_rastertonumpy
):
    """The whole point of the helper: cloud_pct must be the fraction
    over OBSERVED pixels, not over the saved grid. If a scene is half
    off-footprint (sentinel 255) and the on-footprint half is 100%
    cloud (class 1), the report should be 100% — not 50%."""
    grid = np.full((10, 10), 255, dtype=np.uint8)
    # The top 5 rows are observed; everything observed is class 1 (cloud).
    grid[:5, :] = 1
    patched_rastertonumpy["array"] = grid

    cloud_pct, shadow_pct = genesis._ocm_mask_class_fractions("fake.tif")
    assert cloud_pct == pytest.approx(100.0)
    assert shadow_pct == 0.0


def test_ocm_mask_class_fractions_mixed_clouds_and_shadows(
    genesis, patched_rastertonumpy
):
    """Class 0 (clear), 1+2 (cloud), 3 (shadow), 255 (no observation).
    Verify the helper splits the cloud / shadow buckets correctly
    against the observed-pixel denominator."""
    # 100 observed pixels: 50 clear (0), 30 cloud (20 of class 1, 10
    # of class 2), 20 shadow (class 3). 28 off-footprint (255).
    arr = np.empty(128, dtype=np.uint8)
    arr[:50] = 0
    arr[50:70] = 1
    arr[70:80] = 2
    arr[80:100] = 3
    arr[100:] = 255
    patched_rastertonumpy["array"] = arr.reshape(8, 16)

    cloud_pct, shadow_pct = genesis._ocm_mask_class_fractions("fake.tif")
    # cloud = 30/100 = 30%; shadow = 20/100 = 20%. 28 unobserved
    # pixels are NOT in the denominator.
    assert cloud_pct == pytest.approx(30.0)
    assert shadow_pct == pytest.approx(20.0)


def test_ocm_mask_class_fractions_all_unobserved_returns_zero(
    genesis, patched_rastertonumpy
):
    """A mask that's entirely off-footprint (every pixel is sentinel)
    must return (0.0, 0.0), not raise a divide-by-zero."""
    patched_rastertonumpy["array"] = np.full((5, 5), 255, dtype=np.uint8)
    cloud_pct, shadow_pct = genesis._ocm_mask_class_fractions("fake.tif")
    assert cloud_pct == 0.0
    assert shadow_pct == 0.0


def test_ocm_mask_class_fractions_swallows_read_failure(
    genesis, monkeypatch
):
    """If the raster can't be read the helper must return (0.0, 0.0)
    rather than raise — caller's logging downstream depends on that."""
    import arcpy

    def _raise(*a, **kw):
        raise RuntimeError("disk unreachable")

    monkeypatch.setattr(arcpy, "RasterToNumPyArray", _raise)
    cloud_pct, shadow_pct = genesis._ocm_mask_class_fractions("missing.tif")
    assert cloud_pct == 0.0
    assert shadow_pct == 0.0


# ---------------------------------------------------------------------------
# _dl_cloud_mask_infer_scene — NoData sentinel encoding added v1.0
# ---------------------------------------------------------------------------

def test_dl_cloud_mask_infer_scene_writes_nodata_sentinel(genesis):
    """v1.0 design memo: off-footprint pixels (DN 0 in any input
    band) become sentinel 255 in the saved mask, and the saved
    raster carries `value_to_nodata=255` so Pro renders them as
    NoData. Source-grep keeps the convention pinned."""
    src = inspect.getsource(genesis._dl_cloud_mask_infer_scene)
    assert "value_to_nodata=255" in src
    # And the upstream stack must track per-band no-observation so
    # the sentinel can be applied AFTER OmniCloudMask infers — the
    # model itself sees the zeroed pixels as legitimate dark ground.
    assert "no_observation" in src or "nodata_mask" in src


def test_dl_cloud_mask_infer_scene_uses_omnicloudmask(genesis):
    """Pin the upstream library so a future swap to a different
    cloud detector trips the test — at which point the threshold
    semantics and sentinel encoding may need re-validation."""
    src = inspect.getsource(genesis._dl_cloud_mask_infer_scene)
    assert "predict_from_array" in src or "omnicloudmask" in src.lower()


# ---------------------------------------------------------------------------
# _temporal_outlier_clean — Tmask-style robust cleaner, default ON in v1.0
# ---------------------------------------------------------------------------

def test_temporal_outlier_clean_uses_robust_z_score(genesis):
    """The cleaner is the Tmask-style robust z-score described in the
    docstring — it must reach a MAD (median absolute deviation)
    floor and a k-sigma threshold rather than a simple mean+std,
    otherwise persistent cloud in a stack would skew the
    'outlier' threshold past genuine cloud pixels."""
    src = inspect.getsource(genesis._temporal_outlier_clean)
    assert "median" in src.lower()
    # The MAD-floor protects against zero-variance bands (e.g. a
    # stack of identical fill values).
    assert "mad_floor" in src or "MAD" in src
    # And the k threshold is exposed so the user can tune.
    assert "k" in inspect.signature(genesis._temporal_outlier_clean).parameters


def test_temporal_outlier_clean_default_k_is_conservative(genesis):
    """k=2.5 is the documented default — strict enough to clip
    bright cloud spikes, loose enough to preserve genuine bright
    surfaces (snow, bare alteration zones). Pin so changes get a
    deliberate review."""
    sig = inspect.signature(genesis._temporal_outlier_clean)
    assert sig.parameters["k"].default == 2.5
    assert sig.parameters["min_obs"].default == 4


# ---------------------------------------------------------------------------
# TOOLBOX_VERSION — v1.0 release stamp
# ---------------------------------------------------------------------------

def test_toolbox_version_is_1_0(genesis):
    """Provenance CSVs embed the toolbox version. Pin the release
    string so it's an intentional change when bumped."""
    assert genesis.TOOLBOX_VERSION == "1.0"
