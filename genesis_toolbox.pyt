# -*- coding: utf-8 -*-

# genesis_toolbox.pyt
#
# GENESIS — Unified multisensor satellite-processing toolbox for ArcGIS Pro.
# Replaces the earlier `landsat_toolbox.pyt` and `sentinel2_toolbox.pyt`
# with a single workflow-ordered toolbox covering three sensors:
#
#   Sensor           Input format             Cloud handling
#   ───────────────  ───────────────────────  ───────────────────────────
#   Landsat 8/9      C2L2 SR/SP scene folder  QA_PIXEL bits 0-4
#                    (`LC08_L2SR_*`,           + optional temporal refinement
#                     `LC08_L2SP_*`, and L9
#                     equivalents — both
#                     L2SR and L2SP accepted;
#                     EarthExplorer ships them
#                     as `.tar` archives that
#                     the Mosaic tool extracts
#                     transparently)
#   Sentinel-2       L2A SAFE folder          SCL classes 3, 8, 9, 10
#                    (`S2A_MSIL2A_*` /        + optional temporal refinement
#                     `S2B_MSIL2A_*`)
#   ASTER            AST_07XT V004 — both     QA Data Plane non-zero
#                    HDF (`.hdf`) and TIFF    + per-scene VIS/SWIR test
#                    folder (`*_SRF_VNIR_*`,    + optional thermal (AST_08
#                            `*_SRF_SWIR_*`)    SKT, paired by scene ID)
#
# Tools (workflow-ordered, by numeric prefix in the ArcGIS Pro UI):
#   01 — Sentinel-2 L2A Mosaic
#   02 — Landsat 8/9 C2L2 Mosaic
#   03 — ASTER L2 Mosaic
#   04 — Spectral Indices & Composites (sensor-aware, grouped UI)
#   05 — Statistical Transformations (PCA / MNF / ICA — sensor-agnostic)
#   06 — Spectral Angle Mapper (sensor-aware band-count check)
#
# Every output raster from Tools 01-03 ships with a `{output}_provenance.csv`
# documenting which input scenes contributed.
#
# Reflectance output convention: float 0-1 (analysis-friendly).
# ASTER scale factor 0.001 is applied during ingestion.

import arcpy
import json
import os
import gc
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
import arcpy.sa
import arcpy.ia
import uuid
import csv
import glob
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import scipy.stats
# scikit-learn is intentionally NOT a dependency. The Noise Transform
# tool's ICA pass uses a pure-numpy FastICA (Hyvärinen parallel /
# logcosh) defined as _fast_ica_numpy below — no external ML stack
# required, so the toolbox loads on a stock ArcGIS Pro Python
# environment without any conda customisation.
from arcpy.ia import ExtractBand, TransposeBits
from arcpy.sa import Float, Divide, Times, Con, SetNull, Plus, Minus
from arcpy.management import CompositeBands

# Disable arcpy history + metadata logging at module import. Every
# arcpy tool call otherwise appends to an XML history file under
# %AppData%\Esri\ArcGISPro\ArcToolbox\History and writes per-dataset
# metadata XML into the workspace; both files grow linearly per call
# and parsing the growing XML produces super-linear loop time. This
# is the dominant documented cause of progressive slowdown in long
# arcpy loops (Esri Community: 90 s/iter at iter 50 dropped to under
# 0.5 s/iter with logging off). Matches the GENESIS pattern (89-scene
# S2 Faial: 48 to 950 s/scene; 74-scene ASTER: 30 to 823 s/scene).
# ClearWorkspaceCache (called periodically in the per-scene loop) is
# a schema-lock release, not a metadata-XML release, which is why the
# earlier scratch-cleanup commit showed no perf improvement.
arcpy.SetLogHistory(False)
arcpy.SetLogMetadata(False)

TOOLBOX_VERSION = "1.0.0-phase3"


# ---------------------------------------------------------------------------
# Shared mosaic scaffolding
# ---------------------------------------------------------------------------

def _make_mosaic_scratch_dir(gdb_path, prefix, mosaic_name):
    """Build (and create) the per-tool scratch folder beside the output GDB.

    All three mosaic tools place their scratch alongside the output
    geodatabase on the same disk — that avoids OneDrive sync churn on
    intermediates and keeps Pro's file handles local. The folder name
    is deterministic from the mosaic name so a re-run with the same
    output name finds (and may reuse) any leftover scratch from the
    previous run.

    Args:
        gdb_path: Path to the output File Geodatabase.
        prefix: Per-tool name prefix, e.g. ``_genesis_s2_scratch``.
        mosaic_name: User-supplied output mosaic name; sanitised before
            use to drop arcpy-hostile characters.

    Returns:
        Absolute path to the (now-created) scratch directory.
    """
    scratch_dir = os.path.join(
        os.path.dirname(os.path.normpath(gdb_path)),
        f"{prefix}_{_sanitize_arcpy_name(mosaic_name)}",
    )
    os.makedirs(scratch_dir, exist_ok=True)
    return scratch_dir


def _apply_aoi_mask_and_save(unmasked_path, mask_feature, gdb_path,
                              output_name, log_prefix=""):
    """Apply an optional AOI mask to a saved median output.

    Returns ``(final_path, unmasked_to_delete)``:

      - If ``mask_feature`` is missing or does not exist, returns the
        unmasked path unchanged and ``None`` for the deletion target.
      - Otherwise, runs ``arcpy.sa.ExtractByMask``, saves the result
        next to the unmasked output as ``{output_name}_Masked``, and
        returns the masked path plus the unmasked path so the caller
        can delete it during downstream cleanup.

    The helper does NOT delete the unmasked output itself — the caller
    typically batches deletions with other intermediates after
    provenance has been written.
    """
    if not (mask_feature and arcpy.Exists(mask_feature)):
        if mask_feature:
            arcpy.AddWarning(
                f"{log_prefix}  ✗ AOI mask {mask_feature!r} not found — "
                f"output is unmasked."
            )
        return unmasked_path, None

    mask_start = datetime.now()
    masked = arcpy.sa.ExtractByMask(unmasked_path, mask_feature)
    masked_path = os.path.join(gdb_path, f"{output_name}_Masked")
    masked.save(masked_path)
    arcpy.AddMessage(
        f"{log_prefix}  ✓ AOI mask applied in "
        f"{(datetime.now() - mask_start).total_seconds():.1f}s "
        f"→ {os.path.basename(masked_path)}"
    )
    return masked_path, unmasked_path


# ---------------------------------------------------------------------------
# Temporal outlier rejection (Tmask reduction)
# ---------------------------------------------------------------------------
#
# Per-pixel temporal outlier rejection for the per-scene stacks that feed
# the GeometricMedian compositor. Per-scene spectral thresholds cannot
# separate warm low cloud (Azorean marine stratocumulus over Faial) from
# warm land because their BT and reflectance distributions overlap. This
# helper sidesteps that wall by changing the unit of analysis from the
# scene to the pixel time series: a scene-pixel is flagged not by looking
# like a cloud in absolute terms but by being anomalous against that
# pixel's own clear-sky history.
#
# Method: collect a single brightness band across the co-registered scene
# stack, compute the per-pixel temporal median and a MAD-derived robust
# sigma, and flag any scene whose value deviates by more than ``k`` robust
# sigmas (bright == cloud, dark == shadow). The flag is then applied to
# every band of that scene-pixel via SetNull; the downstream
# GeometricMedian composites only the survivors.
#
# Reference: Zhu, Z. & Woodcock, C.E. (2014). Automated cloud, cloud
# shadow, and snow detection in multitemporal Landsat data. Remote Sens.
# Environ. 152, 217-234. doi:10.1016/j.rse.2014.06.012. Full Tmask uses
# RIRLS harmonic regression; this is the robust-z reduction.
#
# Why it transfers: nothing is pinned to a site. The median and MAD are
# derived per pixel from the location's own observations, so the same
# code and the same ``k`` work over Faial marine cloud and over the drier
# South Portuguese / Ossa-Morena zones without re-tuning fixed thresholds.
#
# Assumptions: all input stacks share an identical grid (extent, cell
# size, CRS). This holds when the mosaic tool runs with an AOI because
# env.extent + env.snapRaster + cellSize make every per-scene
# CompositeBands output identical. A defensive grid check raises early
# if not.

_TMASK_K = 2.5             # robust z-score threshold, in MAD-sigma units
_TMASK_MIN_OBS = 4         # min valid obs per pixel to attempt cleaning
_TMASK_BLOCK_ROWS = 1024   # row-stripe height to bound peak memory
_TMASK_MAD_FLOOR = 1e-4    # lower bound on robust sigma (stable pixels: MAD~0)


def _tmask_reference_grid(raster_path):
    """Return ``(ncols, nrows, lower_left Point, cell_w, cell_h)``."""
    r = arcpy.Raster(raster_path)
    return (
        r.width, r.height,
        arcpy.Point(r.extent.XMin, r.extent.YMin),
        r.meanCellWidth, r.meanCellHeight,
    )


def _tmask_assert_common_grid(stack_paths):
    """Raise if the stacks are not on a common grid (within sub-cell
    drift for the lower-left and 0.01 percent relative drift for the
    cell size). ncols / nrows must match exactly.

    A snap-aware tolerance accepts up to +/- 0.5 cell of drift in the
    lower-left X / Y. env.snapRaster aligns cells to integer multiples
    of the cell size, but a sub-cell origin drift across many scenes
    from different overpasses is expected even with AOI + snap active;
    the strict 1e-6 m tolerance the earlier implementation used was
    micrometer-scale and fired on routine floating-point noise from
    chained Resample + CompositeBands operations.

    On mismatch the error message identifies which dimension drifted
    and by how much, so the caller can decide between real
    misalignment (re-resample the offender) and floating-point noise
    to be tolerated. Mosaic tool runs with AOI + snap active still
    occasionally hit a stale-stack case where the scratch carries a
    stack from a previous pre-AOI run; the per-dimension diff
    surfaces that immediately.
    """
    ncols0, nrows0, ll0, cw0, ch0 = _tmask_reference_grid(stack_paths[0])
    xy_tol = 0.5 * min(cw0, ch0)
    cs_rel_tol = 1e-4
    for sp in stack_paths[1:]:
        ncols, nrows, ll, cw, ch = _tmask_reference_grid(sp)
        diffs = []
        if ncols != ncols0:
            diffs.append(f"ncols {ncols} vs {ncols0}")
        if nrows != nrows0:
            diffs.append(f"nrows {nrows} vs {nrows0}")
        dx = abs(ll.X - ll0.X)
        if dx > xy_tol:
            diffs.append(f"lower-left X drift {dx:.4g} > tol {xy_tol:.4g}")
        dy = abs(ll.Y - ll0.Y)
        if dy > xy_tol:
            diffs.append(f"lower-left Y drift {dy:.4g} > tol {xy_tol:.4g}")
        if abs(cw - cw0) > cs_rel_tol * cw0:
            diffs.append(f"cell width {cw:.6g} vs {cw0:.6g}")
        if abs(ch - ch0) > cs_rel_tol * ch0:
            diffs.append(f"cell height {ch:.6g} vs {ch0:.6g}")
        if diffs:
            raise ValueError(
                "Temporal cleaning requires all per-scene stacks on a "
                "common grid (within sub-cell drift). Most likely "
                "cause: a stale stack from a previous run without "
                "AOI / snap is still in scratch. Mismatch: "
                f"{os.path.basename(stack_paths[0])} vs "
                f"{os.path.basename(sp)}: " + "; ".join(diffs)
            )
    return ncols0, nrows0, ll0, cw0, ch0


def _tmask_setnull_multiband(stack_path, flag_raster, out_path):
    """Set every band of ``stack_path`` to NoData where flag_raster == 1."""
    r = arcpy.Raster(stack_path)
    masked = []
    for b in range(1, r.bandCount + 1):
        band = arcpy.ia.ExtractBand(r, band_ids=[b])
        masked.append(arcpy.sa.SetNull(flag_raster == 1, band))
    arcpy.management.CompositeBands(masked, out_path)
    return out_path


def _temporal_outlier_clean(
    stack_paths, brightness_band_index, scratch_dir,
    k=_TMASK_K, min_obs=_TMASK_MIN_OBS,
    block_rows=_TMASK_BLOCK_ROWS, mad_floor=_TMASK_MAD_FLOOR,
    obs_count_path=None, cloud_freq_path=None,
    clean_suffix="_clean.tif", log=None,
):
    """Clean per-scene stacks by per-pixel temporal outlier rejection.

    Parameters
    ----------
    stack_paths : list[str]
        Co-registered multiband per-scene stacks (the same list normally
        passed to GeometricMedian). Order is preserved in the return.
    brightness_band_index : int
        1-based band used as cloud/shadow proxy. For the ASTER stack
        order (B01, B02, B03N, ...) B02 (red) is band 2; the VNIR-only
        stack (B01, B02, B03N) also has B02 at band 2. Red is preferred
        over NIR: cloud is bright in red while vegetation is dark in red.
    scratch_dir : str
        Folder for the brightness extracts, flag rasters, and cleaned
        stacks.
    k : float
        Robust z-score threshold (MAD-sigma units). Lower = more
        aggressive.
    min_obs : int
        Pixels with fewer valid observations are not cleaned (left as-is).
    obs_count_path, cloud_freq_path : str or None
        Optional outputs. obs_count is the per-pixel count of valid
        observations (Int32); cloud_freq is the flagged fraction
        ``1 - (obs_count / n_scenes)`` (Float32). Pixels never observed
        (obs_count == 0) are written as NoData in BOTH layers so the
        pair is internally consistent (never-observed ≠ always-cloudy).
        Treat both as evidence-quality layers, not QA to discard.
    clean_suffix : str
        Suffix for cleaned stack filenames.
    log : callable or None
        Message sink (defaults to ``arcpy.AddMessage``).

    Returns
    -------
    list[str]
        Cleaned stack paths (feed these to GeometricMedian). With fewer
        than two scenes the inputs are returned unchanged.

    Notes
    -----
    Two-pass implementation. Pass 1 walks the AOI in row stripes,
    loading every scene's brightness extract into a single cube of shape
    ``(n, block_rows, ncols)`` and accumulating per-pixel ``med``,
    ``sigma`` and ``obs`` into full ``(nrows, ncols)`` arrays. Pass 2
    iterates over scenes; for each scene it re-walks the row stripes
    loading only that scene's brightness, derives the per-scene drop
    mask from the pass-1 statistics, accumulates it into a single
    ``(nrows, ncols)`` flag array, saves the flag raster to scratch,
    and applies it via SetNull + CompositeBands. Peak memory is
    bounded by the larger of the pass-1 cube and the pass-2 per-scene
    flag array. The previous single-pass design held an
    ``(n, nrows, ncols)`` flag tensor which becomes multi-GB at
    continental AOIs.
    """
    say = log or arcpy.AddMessage
    n = len(stack_paths)
    if n < 2:
        say("  temporal clean: <2 scenes, skipping (no time series).")
        return list(stack_paths)

    ncols, nrows, ll, cw, ch = _tmask_assert_common_grid(stack_paths)
    say(
        f"  temporal clean: {n} scenes on {ncols}x{nrows} grid, "
        f"band {brightness_band_index}, k={k}, min_obs={min_obs}"
    )

    # 1) Extract the brightness band of each scene to a single-band
    #    raster (cheaper to block-read than the full multiband stack,
    #    and persisted so pass 2 can re-read without recomputation).
    bright_paths = []
    for sp in stack_paths:
        base = os.path.splitext(os.path.basename(sp))[0]
        bp = os.path.join(scratch_dir, base + "_bright.tif")
        arcpy.ia.ExtractBand(
            arcpy.Raster(sp), band_ids=[brightness_band_index],
        ).save(bp)
        bright_paths.append(bp)

    # 2) Pass 1: walk the AOI in row stripes; accumulate per-pixel
    #    temporal median, robust sigma, and observation count into
    #    full-extent arrays. The cube allocation is bounded by
    #    ``block_rows``; obs_full is Int32 so it survives multi-decade
    #    Sentinel-2 stacks without wraparound (Int16 caps at 32767).
    med_full = np.empty((nrows, ncols), dtype=np.float32)
    sigma_full = np.empty((nrows, ncols), dtype=np.float32)
    obs_full = np.zeros((nrows, ncols), dtype=np.int32)

    for r0 in range(0, nrows, block_rows):
        rh = min(block_rows, nrows - r0)
        # Lower-left of this top-anchored stripe (RasterToNumPyArray
        # windows are anchored at the top of the source raster).
        blk_ll = arcpy.Point(ll.X, ll.Y + (nrows - r0 - rh) * ch)

        cube = np.empty((n, rh, ncols), dtype=np.float32)
        for i, bp in enumerate(bright_paths):
            cube[i] = arcpy.RasterToNumPyArray(
                arcpy.Raster(bp), blk_ll, ncols, rh,
                nodata_to_value=np.nan,
            )

        valid = ~np.isnan(cube)
        obs_full[r0:r0 + rh, :] = valid.sum(axis=0).astype(np.int32)

        # ``nanmedian`` over a pixel where every scene is NaN returns
        # NaN; the downstream z computation then yields NaN comparisons
        # which are False, so the pixel can't be flagged. Suppress the
        # RuntimeWarning that nanmedian emits for those columns.
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(cube, axis=0)
            mad = np.nanmedian(np.abs(cube - med), axis=0) * 1.4826
        sigma = np.maximum(mad, mad_floor)

        med_full[r0:r0 + rh, :] = med
        sigma_full[r0:r0 + rh, :] = sigma

    # Free pass-1 cube allocations before pass 2 ramps up.
    del cube, valid, med, mad, sigma

    # 3) Pass 2: per scene, derive the drop mask stripe by stripe from
    #    the pass-1 statistics; accumulate into one ``(nrows, ncols)``
    #    flag array; save explicitly so SetNull receives a materialised
    #    raster (no lazy raster shadowing across loop iterations);
    #    apply via SetNull + CompositeBands.
    cleaned = []
    total_flagged = 0
    total_eligible = 0
    arcpy.SetProgressor(
        "step", "Temporal clean: writing cleaned stacks", 0, n, 1,
    )
    try:
        for i, sp in enumerate(stack_paths):
            base = os.path.splitext(os.path.basename(sp))[0]
            flag_arr = np.zeros((nrows, ncols), dtype=np.uint8)

            for r0 in range(0, nrows, block_rows):
                rh = min(block_rows, nrows - r0)
                blk_ll = arcpy.Point(ll.X, ll.Y + (nrows - r0 - rh) * ch)

                scene_stripe = arcpy.RasterToNumPyArray(
                    arcpy.Raster(bright_paths[i]),
                    blk_ll, ncols, rh, nodata_to_value=np.nan,
                )

                valid = ~np.isnan(scene_stripe)
                med_s = med_full[r0:r0 + rh, :]
                sigma_s = sigma_full[r0:r0 + rh, :]
                obs_s = obs_full[r0:r0 + rh, :]
                enough = obs_s >= min_obs
                eligible = valid & enough

                with np.errstate(invalid="ignore"):
                    z = (scene_stripe - med_s) / sigma_s
                    drop = ((z > k) | (z < -k)) & eligible

                flag_arr[r0:r0 + rh, :] = drop.astype(np.uint8)
                total_eligible += int(eligible.sum())

            total_flagged += int(flag_arr.sum())

            # Save the flag raster explicitly. The earlier lazy-raster
            # idiom (NumPyArrayToRaster held only as a local) risked
            # SetNull resolving against a stale ``==1`` Raster object
            # if the GP driver ever deferred evaluation across loop
            # iterations; explicit save removes the ambiguity.
            flag_path = os.path.join(scratch_dir, base + "_flag.tif")
            arcpy.NumPyArrayToRaster(
                flag_arr, ll, cw, ch,
            ).save(flag_path)
            flag_ras = arcpy.sa.Raster(flag_path)

            out = os.path.join(scratch_dir, base + clean_suffix)
            _tmask_setnull_multiband(sp, flag_ras, out)
            cleaned.append(out)
            arcpy.SetProgressorPosition(i + 1)
    finally:
        arcpy.ResetProgressor()

    # ``total_eligible`` excludes NaN pixels and pixels where coverage
    # was below ``min_obs``; reporting against eligible pixels (not the
    # full scene-pixel count) gives the rate at which the cleaner
    # actually flagged something it had the data to flag.
    if total_eligible > 0:
        flagged_frac = 100.0 * total_flagged / total_eligible
        say(
            f"  temporal clean: {flagged_frac:.1f}% of eligible "
            f"scene-pixels flagged as cloud/shadow "
            f"({total_flagged:,} of {total_eligible:,})"
        )
    else:
        say(
            "  temporal clean: no eligible scene-pixels "
            "(coverage too sparse everywhere); nothing flagged."
        )

    # 4) Evidence-quality layers. Both rasters use a NaN sentinel at
    #    obs == 0 so the never-observed footprint is NoData in BOTH
    #    layers (matching semantics for downstream joins). obs_full is
    #    Int32 in memory but saved as a float Raster via the NaN-bearing
    #    array; the value range still fits well within float32.
    if obs_count_path is not None:
        obs_float = obs_full.astype(np.float32)
        obs_float[obs_full == 0] = np.nan
        arcpy.NumPyArrayToRaster(
            obs_float, ll, cw, ch,
        ).save(obs_count_path)
        thin = int(((obs_full > 0) & (obs_full < min_obs)).sum())
        say(
            f"  temporal clean: obs_count written; {thin} pixels below "
            f"min_obs={min_obs} (coverage too sparse to trust)"
        )
    if cloud_freq_path is not None:
        freq = 1.0 - (obs_full.astype(np.float32) / float(n))
        freq[obs_full == 0] = np.nan
        arcpy.NumPyArrayToRaster(freq, ll, cw, ch).save(cloud_freq_path)
        say(f"  temporal clean: cloud_freq written -> "
            f"{os.path.basename(cloud_freq_path)}")

    return cleaned


# ---------------------------------------------------------------------------
# Phase-logging context manager
# ---------------------------------------------------------------------------

class phase:
    """Context manager that wraps a phase of work with start/end markers.

    Standardises the ``▶ Phase N — ...`` / ``✓ Phase N in Xs`` /
    ``✗ Phase N failed after Xs: {exc}`` logging idiom that previously
    lived open-coded in every mosaic tool, and exposes the elapsed
    time as ``.elapsed`` so callers can fold it into a richer
    close-line.

    Lowercase class name is intentional — reads as a verb in
    ``with phase("..."):`` blocks (cf. ``contextlib.suppress``).

    Three usage modes:

      Auto-close (simple sites, prints
      "✓ {label} in {t:.1f}s" on successful exit)::

          with phase("Phase 5 — Merge / mask / cleanup"):
              ...

      Quiet-close (sites that print their own enriched close line —
      averaged per-scene timings, scene-count breakdowns, etc. The
      manager still prints the ▶ entry line and the ✗ failure line,
      just suppresses its own ✓ close)::

          with phase("Phase 3 — Per-scene processing", count=n,
                     quiet_close=True) as p:
              ... work ...
              arcpy.AddMessage(f"  ✓ {n} scenes in {p.elapsed:.1f}s "
                               f"(avg {avg:.1f}s/scene)")

      Silent-error (sites already wrapped in an outer ``except``
      handler that logs the failure with its own canonical message.
      Avoids the duplicate yellow-then-red icon noise in the GP
      dialog. The manager still prints the ▶ entry line and the ✓
      close line; only the ✗ failure warning is suppressed)::

          try:
              with phase("Phase 4 — GeometricMedian", quiet_close=True,
                         silent_error=True) as ph:
                  ... work ...
          except arcpy.ExecuteError as e:
              arcpy.AddError(f"GeometricMedian failed: {e}")
              return None

    Returns ``False`` from ``__exit__`` in all error modes, so
    exceptions always propagate.
    """

    def __init__(self, label, count=None, quiet_close=False,
                 silent_error=False, message_callback=None):
        self.label = label
        self.count = count
        self.quiet_close = quiet_close
        self.silent_error = silent_error
        self._msg = message_callback or arcpy.AddMessage
        self._start = None
        self.elapsed = 0.0

    def __enter__(self):
        line = f"\n▶ {self.label}"
        if self.count is not None:
            line += f" ({self.count} scene{'s' if self.count != 1 else ''})"
        self._msg(line)
        self._start = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = (datetime.now() - self._start).total_seconds()
        if exc_type is None:
            if not self.quiet_close:
                self._msg(f"  ✓ {self.label} in {self.elapsed:.1f}s")
        elif not self.silent_error:
            arcpy.AddWarning(
                f"  ✗ {self.label} failed after {self.elapsed:.1f}s: {exc_val}"
            )
        return False  # never suppress — exceptions propagate normally


# ---------------------------------------------------------------------------
# Output sanity check
# ---------------------------------------------------------------------------

# Sensor-specific lower bound on the per-band mean. Bands whose mean
# falls below this threshold across the whole output have almost always
# collapsed to near-zero — never a real signal at typical land-surface
# reflectance. Used by _sanity_check_output to flag silent-correctness
# regressions (e.g. the May 2026 ASTER dark-blob bug where Con().save()
# stripped NoData and the temporal median was pulled toward zero).
#   - Landsat Collection 2 L2 SR is stored as DN (1-65454 valid); typical
#     B5 (NIR) over vegetated land sits in the 8 000-25 000 range, so 500
#     is comfortably below any plausible mosaic.
#   - Sentinel-2 L2A and ASTER AST_07XT are converted to reflectance
#     [0, 1] inside their per-scene processors; typical land mean is
#     ~0.05-0.30, so 0.01 is well under any real signal.
_SANITY_MIN_MEAN = {
    "landsat": 500.0,
    "sentinel-2": 0.01,
    "aster": 0.01,
}


def _sanity_check_output(raster_path, sensor_hint=None, label=None,
                         message_callback=None):
    """Per-band stats canary for newly-saved median outputs.

    Reads MIN/MEAN/MAX for each band and logs a one-line summary so the
    user can spot obvious anomalies (all-zero bands, saturated bands)
    in the messages tab. If a sensor_hint is supplied and any band's
    mean falls below the expected lower bound for that sensor, a
    warning is emitted pointing at the suspicious bands.

    This is a fast post-save sniff test — not a correctness proof.
    Visual inspection of the output is still required for publication.

    Args:
        raster_path: Path to the saved raster to inspect.
        sensor_hint: One of "landsat", "sentinel-2", "aster" to enable
            sensor-specific mean checks. None disables them (stats are
            still logged).
        label: Display name shown in the messages tab. Defaults to
            the raster's basename.
        message_callback: Optional logging callback. Defaults to
            ``arcpy.AddMessage``.
    """
    msg = message_callback or arcpy.AddMessage
    label = label or os.path.basename(raster_path)
    # Newly-saved file-geodatabase rasters arrive without statistics
    # computed, which makes every GetRasterProperties call below fail
    # with ERROR 001100 ("no statistics available"). One BUILD pass
    # over the saved raster fixes it; cheap relative to the geomedian
    # we just ran. Wrapped in try / except so a failure here only
    # demotes the sanity check to "stats unavailable" instead of
    # bringing the run down.
    try:
        arcpy.management.CalculateStatistics(raster_path)
    except Exception:
        pass
    try:
        band_count = int(arcpy.Raster(raster_path).bandCount)
    except Exception as e:
        arcpy.AddWarning(f"  Sanity check [{label}]: cannot open ({e})")
        return

    threshold = _SANITY_MIN_MEAN.get(sensor_hint)
    msg(f"  Sanity check [{label}]: {band_count} band(s)")
    suspicious = []
    for i in range(1, band_count + 1):
        band_ref = f"{raster_path}/Band_{i}"
        try:
            min_v = float(arcpy.management.GetRasterProperties(
                band_ref, "MINIMUM").getOutput(0).replace(",", "."))
            mean_v = float(arcpy.management.GetRasterProperties(
                band_ref, "MEAN").getOutput(0).replace(",", "."))
            max_v = float(arcpy.management.GetRasterProperties(
                band_ref, "MAXIMUM").getOutput(0).replace(",", "."))
            msg(f"    Band {i}: mean={mean_v:.4g}, min={min_v:.4g}, max={max_v:.4g}")
            if threshold is not None and mean_v < threshold:
                suspicious.append(i)
        except Exception as e:
            arcpy.AddWarning(f"    Band {i}: stats unavailable ({e})")

    if suspicious:
        arcpy.AddWarning(
            f"  Sanity check: band(s) {suspicious} have a mean below "
            f"{threshold} for sensor {sensor_hint!r}. This is the signature "
            "of a NoData-handling regression in the compositing step "
            "(values pulled toward zero). Verify the output visually "
            "before publishing."
        )


# ---------------------------------------------------------------------------
# Per-sensor band map + sidecar writer
# ---------------------------------------------------------------------------

# Canonical band layout for each mosaic output. Used by
# ``_write_band_sidecar_csv`` to document Band_N → satellite-band
# mapping so the mosaic stays interpretable when the GDB raster format
# can't carry per-band descriptions. Order matches the stack order
# each tool writes; positions are 1-indexed (matches ArcGIS's
# ``raster/Band_N`` convention).
_BAND_LAYOUT = {
    "landsat": [
        # SR_B1..SR_B7 — Landsat 8/9 Collection 2 Level 2 SR.
        (1, "SR_B1", "Coastal", 443, 30),
        (2, "SR_B2", "Blue", 482, 30),
        (3, "SR_B3", "Green", 561, 30),
        (4, "SR_B4", "Red", 655, 30),
        (5, "SR_B5", "NIR", 865, 30),
        (6, "SR_B6", "SWIR1", 1609, 30),
        (7, "SR_B7", "SWIR2", 2201, 30),
    ],
    "sentinel-2": [
        # 12-band L2A in wavelength order — B10 absent (Sen2Cor strips
        # it during atmospheric correction). Native resolutions: 10m,
        # 20m or 60m; all bands are resampled to 10m in the mosaic.
        (1,  "B01", "Coastal",     443, 60),
        (2,  "B02", "Blue",        490, 10),
        (3,  "B03", "Green",       560, 10),
        (4,  "B04", "Red",         665, 10),
        (5,  "B05", "RedEdge1",    705, 20),
        (6,  "B06", "RedEdge2",    740, 20),
        (7,  "B07", "RedEdge3",    783, 20),
        (8,  "B08", "NIR",         842, 10),
        (9,  "B8A", "NarrowNIR",   865, 20),
        (10, "B09", "WaterVapour", 945, 60),
        (11, "B11", "SWIR1",       1610, 20),
        (12, "B12", "SWIR2",       2190, 20),
    ],
    "aster": [
        # AST_07XT V004 VNIR+SWIR — 9-band stack. Wavelengths are band
        # midpoints (the ASTER bands are bandpass-shaped, not point
        # measurements). B03 is the nadir-looking 3N variant.
        (1, "B01",  "Green",         560,  15),
        (2, "B02",  "Red",           661,  15),
        (3, "B03N", "NIR",           807,  15),
        (4, "B04",  "SWIR1",         1656, 30),
        (5, "B05",  "SWIR2_2165",    2167, 30),
        (6, "B06",  "SWIR2_2205",    2209, 30),
        (7, "B07",  "SWIR2_2260",    2262, 30),
        (8, "B08",  "SWIR2_2330",    2336, 30),
        (9, "B09",  "SWIR2",         2400, 30),
    ],
    "aster-vnir": [
        # VNIR-only ASTER (post-Apr-2008 SWIR failure). 3-band subset
        # of the AST_07XT VNIR group.
        (1, "B01",  "Green", 560, 15),
        (2, "B02",  "Red",   661, 15),
        (3, "B03N", "NIR",   807, 15),
    ],
}


def _write_band_sidecar_csv(output_raster_path, sensor_key, suffix="_bands.csv"):
    """Write a ``{output}_bands.csv`` sidecar documenting which stack
    position holds which satellite band.

    The file format is intentionally minimal — one header line + one
    row per band — so it opens cleanly in any text editor or
    spreadsheet, and so a downstream script can parse it with
    ``csv.DictReader`` without sensor-specific logic. Columns:

    - ``band_index``       Stack position (1-indexed; matches Pro's
                           ``raster/Band_N`` band navigation).
    - ``satellite_band``   Original band name from the source product
                           (e.g., ``B04``, ``B8A``, ``SR_B5``).
    - ``role``             GENESIS band-role label (``Red``, ``NIR``,
                           ``SWIR1``…) — the same key used by
                           ``SENSOR_BAND_ROLES`` and by the indices /
                           composites catalogue in Tool 04.
    - ``wavelength_nm``    Approximate band centre in nanometres.
    - ``native_res_m``     Native sensor resolution before any
                           resampling in the mosaic pipeline.

    Returns the sidecar path on success, or ``None`` if no layout is
    registered for ``sensor_key`` (in which case nothing is written).
    """
    layout = _BAND_LAYOUT.get(sensor_key)
    if not output_raster_path or not layout:
        return None
    csv_path = _sidecar_path_for_raster(output_raster_path, suffix)
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "band_index", "satellite_band", "role",
                "wavelength_nm", "native_res_m",
            ])
            for band_index, name, role, wl, native_res in layout:
                writer.writerow([band_index, name, role, wl, native_res])
        arcpy.AddMessage(f"  Band-mapping CSV: {csv_path}")
        return csv_path
    except OSError as e:
        arcpy.AddWarning(f"  Could not write band-mapping CSV ({e})")
        return None


# ---------------------------------------------------------------------------
# Arcpy-safe identifier sanitisation
# ---------------------------------------------------------------------------

_ARCPY_NAME_TRANSLATIONS = {ord(c): "_" for c in " ()[]{}'\";,$@#!%^&*+=<>?|"}


def _sanitize_arcpy_name(name):
    """Strip filesystem/arcpy-hostile characters from a scene identifier.

    arcpy.management.Resample (and several other GP tools) rejects output
    dataset names that contain spaces, parentheses, or other punctuation
    with ERROR 000354. This bites scenes whose archives were renamed by
    the OS to dedupe downloads (e.g., Windows ``Foo (1).zip``).

    The replacement uses ``_`` rather than removal so two distinct scenes
    can never collapse to the same id by accident. Empty/None input
    returns an empty string.
    """
    if not name:
        return ""
    return str(name).translate(_ARCPY_NAME_TRANSLATIONS)


def _check_scratch_schema(reused_paths, expected_band_count, sensor_label):
    """Refuse to resume from a scratch whose stacks have an
    unexpected band count. Indicates the scratch was written by an
    earlier code version with an incompatible layout (e.g., 10-band
    S2 stacks when the current code emits 12 bands, or ASTER stacks
    built before the BT scale fix).

    Returns True when resume is safe to proceed (band counts match,
    or there is nothing to reuse). Returns False when the caller
    should abort and ask the user to delete the scratch folder.
    """
    if not reused_paths:
        return True
    try:
        actual = int(arcpy.Raster(reused_paths[0]).bandCount)
    except Exception as e:
        arcpy.AddWarning(
            f"  Resume schema check could not inspect "
            f"{os.path.basename(reused_paths[0])} ({e}). "
            "Continuing — please verify the final output visually."
        )
        return True
    if actual != expected_band_count:
        arcpy.AddError(
            f"  Resume aborted: existing {sensor_label} stacks in "
            f"scratch have {actual} band(s) but this code emits "
            f"{expected_band_count}. The scratch was produced by an "
            "earlier version of the toolbox and is incompatible. "
            "Delete the scratch folder and re-run to rebuild."
        )
        return False
    return True


def _sidecar_path_for_raster(raster_path, suffix):
    """Resolve where a text sidecar (.csv / .npz / .html / .txt /
    ...) should be written for a given saved raster.

    Rasters inside a file geodatabase (or SDE) can't carry text
    sidecars in the same workspace — opening
    ``D:\\foo.gdb\\bar_provenance.csv`` is a write into the .gdb
    directory which ArcGIS hides from its catalog view, so the user
    never sees the file. The sidecar is therefore routed to the
    parent folder of the .gdb / .sde, named after the raster's
    basename. Folder workspaces (.tif outputs) put the sidecar
    alongside the raster, with the raster's extension stripped from
    the prefix so we don't end up with ``name.tif_provenance.csv``.
    """
    norm = os.path.normpath(raster_path)
    parent = os.path.dirname(norm)
    parent_lower = parent.lower()
    stem, _ = os.path.splitext(os.path.basename(norm))
    if parent_lower.endswith(".gdb") or parent_lower.endswith(".sde"):
        sidecar_dir = os.path.dirname(parent) or parent
    else:
        sidecar_dir = parent
    return os.path.join(sidecar_dir, stem + suffix)


def _build_workspace_subfolder_path(out_workspace, name, subfolder):
    """Route a raster output into a workspace-appropriate path.

    Folder workspaces default to ESRI GRID, which caps raster names at
    13 characters; appending ``.tif`` forces GeoTIFF and lifts that
    limit. For folder workspaces the output is also routed into the
    given ``subfolder`` (created on demand) so different product
    families don't visually mix in the catalog — Tool 04 uses
    ``indices/`` + ``composites/``; Tool 05 uses ``pca/`` / ``mnf/`` /
    ``ica/``.

    File geodatabases (.gdb) and SDE workspaces have no name length
    limit and don't support nested folders, so the output is saved flat
    into the workspace root with no extension.
    """
    ws_lower = (out_workspace or "").lower().rstrip("\\/")
    if ws_lower.endswith(".gdb") or ws_lower.endswith(".sde"):
        return os.path.join(out_workspace, name)
    target_dir = os.path.join(out_workspace, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, f"{name}.tif")


# ---------------------------------------------------------------------------
# Tool 07 helpers — temporal statistics over per-scene scratch stacks
# ---------------------------------------------------------------------------

# Dry-month sets per regional climate pattern. Mirrors
# _seasonal_pattern_for_region (defined on each mosaic tool) but bucketed
# into a single binary dry/wet axis suitable for Tool 07's temporal
# stratification. Wet = not dry.
_DRY_MONTHS_BY_PATTERN = {
    "temperate":  {6, 7, 8, 9},               # Azores / Portugal / Madeira summer
    "angola":     {5, 6, 7, 8, 9, 10},        # Southern Africa dry season
    "cape_verde": {12, 1, 2, 3, 4, 5, 6, 7},  # Sahel-style long dry
    "mozambique": {4, 5, 6, 7, 8, 9},
}


def _classify_season_bucket(d, seasonal_pattern):
    """Return ``"dry"`` or ``"wet"`` for the given date + regional pattern.
    Returns ``None`` when ``d`` is None (date couldn't be parsed from a
    scratch filename); the caller drops un-dated scenes from per-season
    grouping.
    """
    if d is None:
        return None
    months = _DRY_MONTHS_BY_PATTERN.get(seasonal_pattern, {6, 7, 8, 9})
    return "dry" if d.month in months else "wet"


# Per-sensor regex set for extracting the acquisition date from a
# stratched stack filename. The mosaic tools encode the date inside the
# scene_id of the saved stack TIFF (no separate sidecar manifest, so
# the filename IS the source of truth). Each pattern returns the
# acquisition date via the year/month/day capture groups.
_STACK_DATE_PATTERNS = (
    # Sentinel-2: PRODUCT_URI starts with S2[ABC]_MSIL2A_YYYYMMDDTHHMMSS_
    re.compile(
        r"^S2[A-Z]_MSIL2A_(?P<YYYY>\d{4})(?P<MM>\d{2})(?P<DD>\d{2})T\d{6}_",
        re.IGNORECASE,
    ),
    # Landsat C2L2: LC0[89]_L2S[PR]_PPPRRR_YYYYMMDD_...
    re.compile(
        r"^LC0[89]_L2S[PR]_\d{6}_(?P<YYYY>\d{4})(?P<MM>\d{2})(?P<DD>\d{2})_",
        re.IGNORECASE,
    ),
    # ASTER AST_07XT: 17-char scene_id = PPP MM DD YYYY HHMMSS (US-style),
    # used as the stem of {scene_id}_stack.tif (or _stack_vnir.tif).
    re.compile(
        r"^\d{3}(?P<MM>\d{2})(?P<DD>\d{2})(?P<YYYY>\d{4})\d{6}_stack",
        re.IGNORECASE,
    ),
)


def _scene_date_from_stack_filename(filename):
    """Parse the acquisition date out of a per-scene scratch stack
    filename. Returns a ``datetime.date`` or ``None`` if no known pattern
    matches.
    """
    base = os.path.basename(filename)
    for pat in _STACK_DATE_PATTERNS:
        m = pat.match(base)
        if not m:
            continue
        try:
            return datetime(
                int(m.group("YYYY")), int(m.group("MM")), int(m.group("DD")),
            ).date()
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Scratch folder cleanup
# ---------------------------------------------------------------------------

def _cleanup_scratch_folder(scratch_dir):
    """Robust per-file cleanup of a scratch folder.

    ArcGIS Pro / GDAL drivers (JP2, GeoTIFF, /vsitar/, /vsizip/) cache
    file handles for tens of seconds to minutes after the rasters that
    used them go out of scope. A single `shutil.rmtree(..., ignore_errors=True)`
    at the end of a run frequently leaves leftover files behind.

    Strategy: walk the folder, retry per file with growing delays. Each
    retry pass calls gc.collect() to drop Python references and
    arcpy.management.ClearWorkspaceCache() to flush arcpy's internal
    raster cache. Per-file (rather than per-folder) so partial release
    is still useful — files that ARE unlocked get deleted on each pass.

    Sequence: 0.5 s pre-sleep → try-all → 1 s sleep → retry → 2 s →
    retry → 3 s → final pass. Total max ~6.5 s. Files still locked
    after that are reported with a single warning; never raises.
    """
    if not scratch_dir or not os.path.isdir(scratch_dir):
        return

    # Pre-release window: drop refs + flush arcpy cache + give Windows
    # a moment to release handles.
    gc.collect()
    try:
        arcpy.management.ClearWorkspaceCache()
    except Exception:
        pass
    time.sleep(0.5)

    # Collect all files (recursive).
    all_files = []
    for root, _, files in os.walk(scratch_dir):
        all_files.extend(os.path.join(root, f) for f in files)

    failed = list(all_files)
    for attempt in range(3):
        still_failed = []
        for path in failed:
            try:
                os.remove(path)
            except (PermissionError, OSError):
                still_failed.append(path)
        failed = still_failed
        if not failed:
            break
        time.sleep(1.0 * (attempt + 1))  # 1 s, 2 s, 3 s
        gc.collect()

    # Tear down the (hopefully empty) directory tree.
    try:
        shutil.rmtree(scratch_dir, ignore_errors=True)
    except Exception:
        pass

    if failed or os.path.isdir(scratch_dir):
        arcpy.AddWarning(
            f"  Scratch folder partially retained ({len(failed)} file(s) "
            f"still locked by Pro/GDAL). Safe to delete manually: {scratch_dir}"
        )


def _cleanup_per_scene_intermediates(scratch_dir, scene_id, keep_basenames):
    """Delete a scene's per-scene intermediate files after its final
    stack has been written. Bounded by the ``{scene_id}*`` glob so it
    cannot touch another scene's files; ``keep_basenames`` is the set
    of basenames (final stack + resume marker) that must survive.

    Best-effort: any file that can't be removed (catalog lock, transient
    GDAL handle) stays for the end-of-run ``_cleanup_scratch_folder``
    retry pass. Safe under ``preserve_scratch=True`` because resume only
    inspects the kept files.

    Scope: scratch directory hygiene. The cleanup keeps the per-scene
    file count flat at ~2 (stack + marker) instead of ~24, which
    bounds disk usage on long runs and makes end-of-run cleanup
    faster. It does NOT fix the per-scene timing degradation observed
    on a 89-scene Sentinel-2 Faial run: a 2026-05-23 A/B against the
    pre-cleanup baseline showed identical timing
    (48 / 97 / 192 / 328 s/scene per 8-scene batch through scene 32).
    The degradation is driven by something the cleanup does not touch;
    suspected GDAL JP2 driver handle pool or Pro raster manager
    growth, neither of which arcpy.management.ClearWorkspaceCache
    reaches. Diagnosis pending.
    """
    if not scratch_dir or not scene_id or not os.path.isdir(scratch_dir):
        return
    keep_paths = {os.path.join(scratch_dir, b) for b in keep_basenames}
    for path in glob.glob(os.path.join(scratch_dir, scene_id + "*")):
        if path in keep_paths:
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except OSError:
            pass


def _periodic_arcpy_cache_flush(idx, every_n=10):
    """Flush arcpy's workspace cache and Python references every N
    scenes during a long Phase 3 loop.

    Scope: arcpy catalog cache + Python reference release. Cheap
    insurance against arcpy holding stale workspace references on
    long runs. Does NOT bound per-scene timing in the Sentinel-2 case
    where the dominant degradation source sits below
    arcpy.management.ClearWorkspaceCache (likely the GDAL JP2 driver
    handle pool or Pro's raster manager; both opaque to arcpy).
    Kept because the flush is near-zero cost and may help cases where
    the cause IS arcpy cache; the S2 case is not one of them.

    Safe to call mid-loop because the geomedian's inputs are referenced
    by file path, not by raster object; nothing the cache holds is
    load-bearing between scenes.
    """
    if idx <= 0 or idx % every_n != 0:
        return
    try:
        arcpy.management.ClearWorkspaceCache()
    except Exception:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Subprocess-per-batch perf workaround
# ---------------------------------------------------------------------------
#
# The 2026-05-22 / 23 / 24 per-scene timing diagnostics on Sentinel-2
# (89 scenes, 48s -> 950s/scene over the run) and ASTER (74 scenes,
# 30s -> 823s/scene) showed a doubling-per-batch curve that did not
# respond to scratch-file cleanup, arcpy.management.ClearWorkspaceCache,
# or arcpy.SetLogHistory(False). The remaining candidates (GDAL JP2
# driver handle pool, Pro raster manager internal state, native library
# buffer accumulations) are all opaque to arcpy.
#
# Process exit is the only mechanism that reclaims them all
# unconditionally: when a process exits the kernel reclaims memory,
# closes file handles, drops library state. The next process starts
# from zero. This pattern is endorsed by Esri's DevSummit 2017 paper
# (Clinton Dow, Parallel Python: Multiprocessing With ArcPy) and is
# the community-standard escape hatch for arcpy long-loop accumulation.
#
# Mechanism: ``_run_scene_batches`` divides a scene list into batches
# of N and for each batch spawns ``python.exe genesis_toolbox.pyt
# --worker <kind> <spec.json>``. The .pyt file is valid Python and its
# tail dispatch block (at the very end) routes to the appropriate
# ``_worker_<kind>_batch`` function. ArcGIS Pro imports the .pyt as a
# module to discover tool classes; the ``__main__`` block does not run
# from Pro. Each worker re-imports arcpy (~5-10s overhead per batch),
# re-establishes env state from the spec, processes the scenes, and
# exits. Per-scene state cannot accumulate across batches because each
# subprocess starts fresh.
#
# Cost: ~10s arcpy reimport per batch. For 89 scenes at batch=10 that
# is ~90s of overhead in exchange for keeping per-scene timing flat at
# ~48s (vs the ~24h degraded curve).

def _resolve_to_catalog_path(layer_or_path):
    """Resolve a Pro layer reference to its on-disk catalog path.

    When the user picks an AOI from Pro's layer dropdown, the GP
    parameter's valueAsText is the layer NAME ("Limite_Municipio"),
    not a path. That name resolves in Pro's map TOC but means nothing
    to a fresh python.exe subprocess worker (which has no map open),
    so the worker's ``arcpy.Exists`` returns False, env.mask stays
    unset, and every operation falls back to the full scene extent.
    Resolving here, in the parent, lets the worker re-bind the dataset
    by path.

    Returns the input unchanged on failure so the worker's existing
    Exists check still produces a sensible "no mask" fallback.
    """
    if not layer_or_path:
        return None
    try:
        return arcpy.Describe(layer_or_path).catalogPath
    except Exception:
        return layer_or_path


def _format_scene_log_line(idx, total, scene_id, elapsed_s, extras=None, fail=None):
    """One per-scene Phase 3 log line, uniformly shaped across mosaic tools.

    Layouts:
      ``  [  1/89] <scene_id>  36.9s   cloud 12.3%``      (success + extras)
      ``  [  1/89] <scene_id>  36.9s``                    (success, no extras)
      ``  ✗ [ 23/89] <scene_id>  31.2s   FAIL: <reason>`` (failure)

    ``extras`` is an optional dict of label -> value strings, joined as
    ``label value`` pairs separated by two spaces and prefixed with
    three spaces from the timing column; ignored when ``fail`` is set.

    ``scene_id`` is right-clipped to 48 characters with a trailing
    ``...`` so Pro's message panel doesn't wrap. The counter is right-
    padded to the digit-width of ``total`` so columns line up.
    """
    width = len(str(total))
    counter = f"[{idx:>{width}}/{total}]"
    sid_clip = scene_id if len(scene_id) <= 48 else scene_id[:45] + "..."
    if fail:
        return f"  ✗ {counter} {sid_clip}  {elapsed_s:5.1f}s   FAIL: {fail}"
    extras_str = ""
    if extras:
        extras_str = "   " + "  ".join(f"{k} {v}" for k, v in extras.items())
    return f"  {counter} {sid_clip}  {elapsed_s:5.1f}s{extras_str}"


def _emit_phase3_summary(ok, failed, total_elapsed_s, failures=None):
    """Emit the Phase 3 tail summary + optional Failures: block via
    ``arcpy.AddMessage`` calls.

    Average is computed over successful scenes only; failures might be
    near-instant (worker-death synthesised) or near-pathological, and
    mixing them with successes would skew the metric. ``failures`` is
    an optional iterable of ``(idx, scene_id, fail_msg)`` tuples that
    get re-listed under a ``Failures:`` header so a long run's errors
    don't scroll out of view in Pro.
    """
    avg = total_elapsed_s / max(1, ok)
    sym = "✓" if not failed else "✗"
    arcpy.AddMessage(
        f"  {sym} {ok} ok, {failed} failed in {total_elapsed_s:.0f}s "
        f"({avg:.1f}s/scene)"
    )
    if failures:
        arcpy.AddMessage("  Failures:")
        for idx, sid, msg in failures:
            sid_clip = sid if len(sid) <= 48 else sid[:45] + "..."
            arcpy.AddMessage(f"    [{idx}] {sid_clip} {msg}")


def _per_band_median_composite(stacks, output_path):
    """Per-band, per-pixel median across a list of multi-band scene
    stacks. A/B alternative to ``arcpy.ia.GeometricMedian`` for
    diagnosing whether the L1-median iteration is producing artefacts
    on NoData-asymmetric inputs.

    For each band index N (count derived from the first stack),
    extracts band N from every input scene via ``arcpy.ia.ExtractBand``
    and computes a per-pixel median via
    ``arcpy.sa.CellStatistics(..., MEDIAN, DATA)``. The per-band
    median rasters are then combined via ``CompositeBands`` into the
    final multi-band output.

    Spectral-consistency caveat: a pixel's per-band values may come
    from different scenes (band 1 from scene A, band 2 from scene B).
    For mineral-mapping use cases that pull on band ratios this is a
    small loss vs GeometricMedian's same-scene constraint; for visual
    mosaicking or per-band statistics it is fine.

    NoData handling is the explicit ``ignore_nodata="DATA"`` flag on
    CellStatistics, which is documented behaviour (unlike
    GeometricMedian's silent NoData semantics).

    AOI clipping and cell size flow from ``arcpy.env.mask`` /
    ``arcpy.env.extent`` / ``arcpy.env.cellSize`` (caller already sets
    these in AsterMosaic._run_mosaic_pipeline).
    """
    if not stacks:
        raise ValueError("Empty stacks list passed to _per_band_median_composite")
    n_bands = int(arcpy.Raster(stacks[0]).bandCount)
    scratch_dir = os.path.dirname(output_path)
    per_band_paths = []
    try:
        for band_idx in range(1, n_bands + 1):
            band_extracts = [
                arcpy.ia.ExtractBand(p, [band_idx]) for p in stacks
            ]
            per_band_median = arcpy.sa.CellStatistics(
                band_extracts, statistics_type="MEDIAN",
                ignore_nodata="DATA",
            )
            out_band_path = os.path.join(
                scratch_dir, f"_per_band_median_b{band_idx:02d}.tif",
            )
            per_band_median.save(out_band_path)
            per_band_paths.append(out_band_path)
        arcpy.management.CompositeBands(per_band_paths, output_path)
    finally:
        for p in per_band_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Deep-learning cloud mask helpers (OmniCloudMask)
# ---------------------------------------------------------------------------
#
# AsterMosaic Phase 4 runs DL cloud-mask inference per scene in the parent
# process (GPU init is amortised across the full scene set; subprocess
# workers in Phase 5 just read the saved mask TIFFs). PyTorch and
# omnicloudmask are lazy-imported inside the helper so the toolbox loads
# cleanly without the DL stack installed; users who do not opt into
# `use_dl_cloud_mask` never trigger the imports.
#
# OmniCloudMask outputs uint8 class rasters: 0=Clear, 1=Thick Cloud,
# 2=Thin Cloud, 3=Cloud Shadow. The mosaic consumer picks which classes
# to treat as "drop" via the aggressiveness preset.

_DL_CLOUD_MASK_FILENAME_FMT = "{sid}_cloudmask.tif"


def _dl_cloud_classes_for(aggressiveness):
    """Map a UI aggressiveness preset to the set of OmniCloudMask class
    IDs to treat as cloud / shadow (i.e. drop in the per-scene mosaic).

    Aggressive (default)  -> {1, 2, 3}  thick + thin + shadow
    Moderate              -> {1, 3}     thick + shadow, keep thin
    Conservative          -> {1}        thick only

    Mirrors S2's Cloud Mask Aggressiveness pattern (Aggressive /
    Moderate / Conservative) for consistency with the rest of the
    toolbox. Returns a Python ``set[int]`` so it round-trips cleanly
    through the subprocess JSON spec.
    """
    s = (aggressiveness or "").lower()
    if s.startswith("conservative"):
        return {1}
    if s.startswith("moderate"):
        return {1, 3}
    return {1, 2, 3}


def _dl_cloud_mask_infer_scene(red_path, green_path, nir_path,
                                output_path, device="cuda"):
    """Run OmniCloudMask on one ASTER VNIR scene; save the per-pixel
    cloud-class mask aligned with the source scene grid.

    ``red_path`` / ``green_path`` / ``nir_path`` are absolute paths to
    AST_07XT VNIR band TIFFs (B02 / B01 / B03N respectively). All three
    must share the same extent + cell size; the helper does not
    reproject. Output is a uint8 single-band raster (class IDs 0..3).
    Returns the saved path on success; raises on failure (caller
    formats the per-scene log line and decides whether to skip).

    PyTorch and omnicloudmask are lazy-imported here so the toolbox
    loads on machines without the DL stack. The Esri Deep Learning
    Frameworks MSI installs PyTorch into arcgispro-py3; ``pip install
    omnicloudmask`` adds the wrapper. See AsterMosaic's
    ``use_dl_cloud_mask`` parameter docstring for the full install
    sequence.
    """
    import numpy as _np
    import torch  # noqa: F401  (imported for caller's device hints)
    from omnicloudmask import predict_from_array

    arrays = []
    nodata_masks = []
    for label, path in (("R", red_path), ("G", green_path), ("NIR", nir_path)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {label} band: {path}")
        arr = arcpy.RasterToNumPyArray(
            path, nodata_to_value=0,
        ).astype(_np.float32)
        # Track per-band no-data footprint (DN 0 in AST_07XT is the
        # conventional no-data marker for out-of-footprint pixels).
        # OmniCloudMask cannot tell "no observation" from "dark clear
        # ground" when it sees 0.0 reflectance; we re-inject that
        # distinction into the output mask after inference.
        nodata_masks.append(arr == 0.0)
        # ASTER L2 DN -> reflectance (0..1). OmniCloudMask normalises
        # per-patch internally, so absolute scale is not strictly needed,
        # but scaled values match Sentinel-2's training distribution.
        arr *= _ASTER_REFLECTANCE_SCALE
        arrays.append(arr)
    ref_shape = arrays[0].shape
    for label, arr in zip(("G", "NIR"), arrays[1:]):
        if arr.shape != ref_shape:
            raise ValueError(
                f"band shape mismatch: {label} {arr.shape} vs R {ref_shape}"
            )

    stack = _np.stack(arrays, axis=0)  # (3, H, W) in (R, G, NIR) order

    # Newer OmniCloudMask releases accept a `device` kwarg; older ones
    # auto-detect. Fall back gracefully.
    try:
        mask = predict_from_array(stack, device=device)
    except TypeError:
        mask = predict_from_array(stack)
    mask = _np.asarray(mask).squeeze()
    if mask.shape != ref_shape:
        raise ValueError(
            f"OCM output shape {mask.shape} != input {ref_shape}"
        )
    mask = mask.astype(_np.uint8)

    # Mark off-footprint pixels as NoData using uint8 sentinel 255.
    # A pixel is "no observation" if any band's source value was 0
    # (the conventional AST_07XT no-data marker). OCM classified those
    # pixels as Clear (class 0) because it saw 0.0 reflectance; we
    # overwrite with 255 so the saved TIFF carries explicit NoData
    # metadata via the value_to_nodata=255 kwarg on the save below.
    # Downstream cloud-fraction reporting then normalises against
    # observed pixels, not against the saved grid that includes the
    # off-footprint stripe.
    no_observation = _np.logical_or.reduce(nodata_masks)
    mask[no_observation] = 255

    src = arcpy.Raster(red_path)
    out_raster = arcpy.NumPyArrayToRaster(
        mask,
        lower_left_corner=arcpy.Point(
            src.extent.XMin, src.extent.YMin,
        ),
        x_cell_size=src.meanCellWidth,
        y_cell_size=src.meanCellHeight,
        value_to_nodata=255,
    )
    out_raster.save(output_path)
    arcpy.management.DefineProjection(
        output_path, src.spatialReference,
    )
    return output_path


def _ocm_mask_class_fractions(mask_path,
                               cloud_classes=(1, 2),
                               shadow_classes=(3,),
                               sentinel=255):
    """Return ``(cloud_pct, shadow_pct)`` over OBSERVED pixels only.

    Reads the saved OmniCloudMask TIFF (uint8 with explicit NoData
    sentinel set by ``_dl_cloud_mask_infer_scene``) and computes
    class fractions against the count of OBSERVED pixels, never the
    raw saved-grid size. Matches the remote-sensing convention of
    reporting cloud cover over observed land (Landsat
    ``CLOUD_COVER_LAND``, Sentinel-2 SCL-normalised fractions)
    rather than diluted by the off-footprint NoData stripe each
    ASTER scene carries.

    The sentinel default (255) matches what
    ``_dl_cloud_mask_infer_scene`` writes; OCM class IDs are 0..3 so
    255 cannot collide with a real class.

    Returns ``(0.0, 0.0)`` on read failure or empty observed area.
    """
    try:
        import numpy as _np
        arr = arcpy.RasterToNumPyArray(mask_path, nodata_to_value=sentinel)
        observed = arr != sentinel
        n_observed = int(observed.sum())
        if n_observed == 0:
            return 0.0, 0.0
        cloud_mem = _np.zeros_like(observed)
        for cls in cloud_classes:
            cloud_mem |= (arr == cls)
        cloud_n = int(cloud_mem.sum())
        shadow_mem = _np.zeros_like(observed)
        for cls in shadow_classes:
            shadow_mem |= (arr == cls)
        shadow_n = int(shadow_mem.sum())
        return (100.0 * cloud_n / n_observed,
                100.0 * shadow_n / n_observed)
    except Exception:
        return 0.0, 0.0


def _compute_aoi_overlap_pct(scene_band_path, aoi_sr, aoi_ext, aoi_area):
    """Return the per-cent overlap of a scene's bounding box with the
    AOI bounding box, in AOI CRS. Used by AsterMosaic Phase 3 to drop
    zero-overlap scenes before per-scene work begins.

    ``aoi_sr`` / ``aoi_ext`` / ``aoi_area`` are pre-computed by the
    caller (one Describe + one extent calc per AOI, reused across all
    scenes) to avoid N redundant Describes on the same mask feature.

    Scene CRS may differ from AOI CRS; the scene extent is projected
    via a corner polygon (PTRA08_UTM_Zone_26N vs WGS_1984_UTM_Zone_26N
    is the common Faial case; the difference is sub-pixel at 15 m but
    enough to skew the bbox math at the edges). Returns 0.0 on
    projection failure; the caller treats that as "no overlap" and
    drops the scene.
    """
    try:
        desc = arcpy.Describe(scene_band_path)
        scene_sr = desc.spatialReference
        scene_ext = desc.extent
        if scene_sr.factoryCode != aoi_sr.factoryCode:
            corners = arcpy.Array([
                arcpy.Point(scene_ext.XMin, scene_ext.YMin),
                arcpy.Point(scene_ext.XMax, scene_ext.YMin),
                arcpy.Point(scene_ext.XMax, scene_ext.YMax),
                arcpy.Point(scene_ext.XMin, scene_ext.YMax),
            ])
            poly = arcpy.Polygon(corners, scene_sr)
            scene_in_aoi = poly.projectAs(aoi_sr).extent
        else:
            scene_in_aoi = scene_ext
        ix_min = max(scene_in_aoi.XMin, aoi_ext.XMin)
        ix_max = min(scene_in_aoi.XMax, aoi_ext.XMax)
        iy_min = max(scene_in_aoi.YMin, aoi_ext.YMin)
        iy_max = min(scene_in_aoi.YMax, aoi_ext.YMax)
        if ix_max <= ix_min or iy_max <= iy_min:
            return 0.0
        return 100.0 * (ix_max - ix_min) * (iy_max - iy_min) / max(1.0, aoi_area)
    except Exception:
        return 0.0


def _resolve_dl_mask_folder(user_folder, data_folder):
    """Resolve the DL cloud-mask cache folder. If the user provided one,
    use it; otherwise default to a ``_dl_cloud_masks/`` sibling of the
    source data folder so masks persist across mosaic re-runs and stay
    co-located with the archive they describe.
    """
    if user_folder and user_folder.strip():
        return user_folder
    return os.path.join(os.path.dirname(data_folder), "_dl_cloud_masks")


def _resolve_worker_python():
    """Return a real python.exe to spawn subprocess workers with.

    Inside ArcGIS Pro, sys.executable is ArcGISPro.exe (Pro hosts
    CPython in-process; the OS-level executable is the GUI shell).
    Spawning that with worker argv launches a new Pro GUI and never
    runs the worker, so the orchestrator pipe blocks forever.
    sys.exec_prefix points at the active Python env (Pro's bundled
    arcgispro-py3, a cloned env, or a standalone install) in every
    host, so python.exe under it is the canonical worker interpreter.
    """
    candidate = os.path.join(sys.exec_prefix, "python.exe")
    if os.path.exists(candidate):
        return candidate
    raise RuntimeError(
        f"Could not locate python.exe under sys.exec_prefix "
        f"({sys.exec_prefix!r}); refusing to fall back to "
        f"sys.executable={sys.executable!r} (under Pro that is "
        f"ArcGISPro.exe and would hang the orchestrator)."
    )


def _run_scene_batches(
    worker_kind, scenes, batch_size, scratch_dir,
    spec_extra=None, log_prefix="  ",
):
    """Run ``scenes`` through the matching worker in batches of
    ``batch_size``, one fresh python.exe per batch.

    Workers emit one JSON event per scene over stdout::

      {"kind": "scene", "sid": "<id>", "elapsed_s": 36.9, "extras": {...}}
      {"kind": "scene", "sid": "<id>", "elapsed_s": 31.2, "fail": "..."}

    The orchestrator parses each event, formats a unified per-scene log
    line via ``_format_scene_log_line``, drives ``arcpy.SetProgressor``,
    and tracks failures for the tail summary emitted via
    ``_emit_phase3_summary``. Batch boundaries are intentionally hidden
    from the visible log; the underlying processing is still one
    subprocess per batch.

    Non-JSON lines on worker stdout pass through verbatim (leading
    whitespace stripped) so stray ``arcpy.AddMessage`` / ``AddWarning``
    from inside ``_process_scene`` still surfaces without double-
    indenting under the formatted counter.

    Worker-death detection: any scene_id that the batch's spec expected
    but never produced a JSON event for is synthesised as a failure
    with reason ``worker exit <rc> before scene reported``.

    Returns True after all batches complete (per-scene failures do NOT
    stop the run; caller proceeds with whatever scenes succeeded).
    Returns False only on user cancellation or worker-spawn exception.

    Spec format (JSON file on disk, path passed as argv[3]):
      {"scenes": [scene_dict, ...], "scratch_dir": str, ...extras}
    """
    spec_extra = spec_extra or {}
    total = len(scenes)
    n_batches = (total + batch_size - 1) // batch_size
    worker_python = _resolve_worker_python()
    arcpy.AddMessage(
        f"{log_prefix}Subprocess batching: {total} scene"
        f"{'s' if total != 1 else ''} via subprocess, {batch_size}/batch"
    )
    arcpy.AddMessage(f"{log_prefix}  worker: {worker_python}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"   # immediate worker stdout, line by line

    def _scene_id_of(scene):
        # Kept in sync with the worker's sid extraction (see
        # _worker_s2_batch / _worker_aster_batch).
        if worker_kind == "s2":
            return (scene.get("metadata") or {}).get("product_uri", "?")
        return scene.get("scene_id") or "?"

    arcpy.SetProgressor("step", "Per-scene processing", 0, max(1, total), 1)
    global_idx = 0
    failures = []
    t0_phase = time.time()
    try:
        for batch_idx in range(n_batches):
            if arcpy.env.isCancelled:
                arcpy.AddWarning(f"{log_prefix}Cancelled by user.")
                return False
            batch = scenes[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            spec_path = os.path.join(
                scratch_dir,
                f"_batch_{batch_idx:03d}_{worker_kind}.json",
            )
            spec = {"scenes": batch, "scratch_dir": scratch_dir, **spec_extra}
            with open(spec_path, "w", encoding="utf-8") as fh:
                json.dump(spec, fh, default=str)
            cmd = [
                worker_python, __file__, "--worker", worker_kind, spec_path,
            ]
            expected_sids = {_scene_id_of(s) for s in batch}
            reported_sids = set()
            rc = 1
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=env,
                )
                for line in proc.stdout:
                    stripped = line.rstrip()
                    if not stripped:
                        continue
                    try:
                        evt = json.loads(stripped)
                    except (ValueError, TypeError):
                        evt = None
                    if (isinstance(evt, dict)
                            and evt.get("kind") == "scene"):
                        global_idx += 1
                        sid = evt.get("sid", "?")
                        reported_sids.add(sid)
                        elapsed = float(evt.get("elapsed_s", 0.0))
                        extras = evt.get("extras") or None
                        fail = evt.get("fail")
                        arcpy.AddMessage(_format_scene_log_line(
                            global_idx, total, sid, elapsed,
                            extras=extras, fail=fail,
                        ))
                        arcpy.SetProgressorPosition(global_idx)
                        if fail:
                            failures.append((global_idx, sid, fail))
                    else:
                        # Non-JSON or unknown event - pass through with
                        # leading whitespace stripped so it doesn't
                        # double-indent under the orchestrator's prefix.
                        arcpy.AddMessage(
                            f"{log_prefix}  {stripped.lstrip()}"
                        )
                    if arcpy.env.isCancelled:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        arcpy.AddWarning(
                            f"{log_prefix}Worker terminated mid-batch."
                        )
                        return False
                rc = proc.wait()
            except Exception as e:
                arcpy.AddError(
                    f"{log_prefix}Worker spawn failed: "
                    f"{type(e).__name__}: {e}"
                )
                return False
            finally:
                try:
                    os.remove(spec_path)
                except OSError:
                    pass

            # Worker-death detection: synthesise failures for sids the
            # batch spec expected but the worker never reported.
            missing = expected_sids - reported_sids
            for sid in missing:
                global_idx += 1
                fail_msg = f"worker exit {rc} before scene reported"
                arcpy.AddMessage(_format_scene_log_line(
                    global_idx, total, sid, 0.0, fail=fail_msg,
                ))
                failures.append((global_idx, sid, fail_msg))
                arcpy.SetProgressorPosition(global_idx)
    finally:
        arcpy.ResetProgressor()

    ok = total - len(failures)
    _emit_phase3_summary(
        ok, len(failures), time.time() - t0_phase, failures,
    )
    return True


def _worker_s2_batch(spec):
    """Subprocess entry point for a Sentinel-2 scene batch.

    Emits one JSON event per scene over stdout::

      {"kind": "scene", "sid": "<id>", "elapsed_s": 36.9, "extras": {}}
      {"kind": "scene", "sid": "<id>", "elapsed_s": 31.2, "fail": "..."}

    Fatal pre-loop errors (missing license, etc.) write a plain
    ``FATAL: ...`` line and exit non-zero; the orchestrator's worker-
    death detection then synthesises failures for every expected sid.
    Per-scene failures emit a ``fail`` event but the worker continues
    so the rest of the batch still runs.
    """
    arcpy.SetLogHistory(False)
    arcpy.SetLogMetadata(False)
    arcpy.env.overwriteOutput = True
    mask_feature = spec.get("mask_feature")
    if mask_feature and arcpy.Exists(mask_feature):
        arcpy.env.mask = mask_feature
        arcpy.env.extent = mask_feature
    for ext in ("Spatial", "ImageAnalyst"):
        if arcpy.CheckExtension(ext) != "Available":
            print(
                f"FATAL: {ext} Analyst extension not Available",
                flush=True,
            )
            sys.exit(2)
        arcpy.CheckOutExtension(ext)
    tool = Sentinel2Mosaic()
    scratch_dir = spec["scratch_dir"]
    scl_classes = tuple(spec["scl_classes"])
    cloud_buffer_pixels = spec["cloud_buffer_pixels"]
    for scene in spec["scenes"]:
        sid = (scene.get("metadata") or {}).get("product_uri", "?")
        t = time.time()
        try:
            tool._process_scene(
                scene, scratch_dir, scl_classes, cloud_buffer_pixels,
            )
            elapsed = time.time() - t
            # S2 has no per-scene cloud diagnostic at present; extras
            # stays empty. Future: count SCL cloud classes vs total
            # pixels after mask and surface as cloud %.
            print(json.dumps({
                "kind": "scene", "sid": sid,
                "elapsed_s": round(elapsed, 1),
                "extras": {},
            }), flush=True)
        except Exception as e:
            elapsed = time.time() - t
            print(json.dumps({
                "kind": "scene", "sid": sid,
                "elapsed_s": round(elapsed, 1),
                "fail": f"{type(e).__name__}: {e}",
            }), flush=True)


def _worker_aster_batch(spec, mode):
    """Subprocess entry point for an ASTER scene batch.

    Emits one JSON event per scene over stdout; see ``_worker_s2_batch``
    for the protocol. The ``extras`` dict carries any per-scene
    diagnostics that ``AsterMosaic._process_scene`` stashed on
    ``scene["metadata"]``: ``cloud_pct`` (always when the cloud test
    ran) and ``bt_stats`` (when the optional AST_08 thermal channel
    was enabled).
    """
    arcpy.SetLogHistory(False)
    arcpy.SetLogMetadata(False)
    arcpy.env.overwriteOutput = True
    arcpy.env.autoCancelling = False
    mask_feature = spec.get("mask_feature")
    if mask_feature and arcpy.Exists(mask_feature):
        arcpy.env.mask = mask_feature
        arcpy.env.extent = mask_feature
    snap_anchor = spec.get("snap_anchor")
    if snap_anchor and os.path.exists(snap_anchor):
        arcpy.env.snapRaster = snap_anchor
        arcpy.env.cellSize = 15
    for ext in ("Spatial", "ImageAnalyst"):
        if arcpy.CheckExtension(ext) != "Available":
            print(
                f"FATAL: {ext} Analyst extension not Available",
                flush=True,
            )
            sys.exit(2)
        arcpy.CheckOutExtension(ext)
    tool = AsterMosaic()
    scratch_dir = spec["scratch_dir"]
    use_qa = bool(spec.get("use_qa", True))
    bt_threshold_k = spec.get("bt_threshold_k")
    use_ast08_thermal = bool(spec.get("use_ast08_thermal", False))
    cloud_buffer_px = int(spec.get("cloud_buffer_px", _ASTER_CLOUD_BUFFER_PX))
    dl_mask_folder = spec.get("dl_mask_folder")
    dl_cloud_classes_list = spec.get("dl_cloud_classes")
    dl_cloud_classes = (
        set(dl_cloud_classes_list) if dl_cloud_classes_list else None
    )
    aster_mode = (
        _ASTER_MODE_VNIR if mode == "vnir" else _ASTER_MODE_FULL
    )
    for scene in spec["scenes"]:
        sid = scene.get("scene_id") or "?"
        t = time.time()
        # Resolve per-scene DL cloud-mask path; falls through silently
        # when DL is off or the cache miss for this scene.
        dl_mask_path = None
        if dl_mask_folder:
            candidate = os.path.join(
                dl_mask_folder,
                _DL_CLOUD_MASK_FILENAME_FMT.format(sid=sid),
            )
            if os.path.exists(candidate):
                dl_mask_path = candidate
        try:
            tool._process_scene(
                scene, scratch_dir, use_qa, aster_mode,
                bt_threshold_k=bt_threshold_k,
                use_ast08_thermal=use_ast08_thermal,
                cloud_buffer_px=cloud_buffer_px,
                dl_cloud_mask_path=dl_mask_path,
                dl_cloud_classes=dl_cloud_classes,
            )
            elapsed = time.time() - t
            meta = scene.get("metadata") or {}
            extras = {}
            cloud_pct = meta.get("cloud_pct")
            if cloud_pct is not None:
                extras["cloud"] = f"{cloud_pct:.1f}%"
            bt_stats = meta.get("bt_stats")
            if bt_stats:
                extras["BT[K]"] = bt_stats
            print(json.dumps({
                "kind": "scene", "sid": sid,
                "elapsed_s": round(elapsed, 1),
                "extras": extras,
            }), flush=True)
        except Exception as e:
            elapsed = time.time() - t
            print(json.dumps({
                "kind": "scene", "sid": sid,
                "elapsed_s": round(elapsed, 1),
                "fail": f"{type(e).__name__}: {e}",
            }), flush=True)


# ---------------------------------------------------------------------------
# Internal tunables. Hoisted from inline values so callers can read and
# adjust knobs in one place. Underscore-prefixed because they're not a
# stable public API — names or values may change between toolbox versions.
# ---------------------------------------------------------------------------

# Eigenvalues below this absolute threshold are treated as rank-deficient
# and replaced by an eigenvalue floor. Set well below float32 epsilon so
# legitimate low-variance directions in low-dynamic-range imagery survive.
# Used by noise_from_valid_diffs, the MNF subset noise path, and the MNF
# whitening matrix construction.
_EIGVAL_FLOOR_ABS = 1e-10

# When the noise covariance is rank-deficient, the floor is data-adaptive:
# lifted to this fraction of the median surviving eigenvalue (rather than
# an absolute value) so the regularisation scales with the imagery.
_EIGVAL_FLOOR_RELATIVE = 1e-4

# ICA whitening eigenvalue floor — looser than the noise floor because the
# DATA covariance (not noise covariance) typically has much larger scale.
_ICA_WHITENING_FLOOR = 1e-12

# Floor for the HFC VD sigma^2 term — guards np.sqrt(0) without altering
# the statistic in any realistic regime.
_HFC_SIGMA2_FLOOR = 1e-30

# MNF ISS-004 diagnostic: warn when the empirical component correlation
# matrix has any off-diagonal element larger than this. The MNF construction
# should yield ~identity in noise-whitened space; visible off-diagonals
# indicate a poor noise estimate or a numerically pathological covariance.
_MNF_CORR_OFFDIAG_WARN = 0.1

# ISS-010 RAM warning: cubes larger than this (estimated as float64) trigger
# a user-facing warning before the in-memory transform tries to allocate
# the working matrix. Chunked PCA/MNF is out of scope.
_RAM_WARNING_GB = 4.0

# arcpy.ia.GeometricMedian convergence parameters. Used by all three
# mosaic tools (Landsat, Sentinel-2, ASTER) when reducing the per-scene
# temporal stack to a cloud-removed composite. ``epsilon`` bounds the
# iteration-to-iteration change before convergence is declared;
# ``max_iter`` caps the cost on noisy stacks where the iteration never
# fully converges. Values match the Esri defaults for GeometricMedian.
_GEOMETRIC_MEDIAN_EPSILON = 0.001
_GEOMETRIC_MEDIAN_MAX_ITER = 20

# ASTER per-scene multi-spectral cloud test. The AST_07XT QA Data Plane
# only flags SR retrieval status (not clouds), so the per-scene path
# uses a hardened multi-spectral test that runs on VNIR alone (works
# on the post-Apr-2008 majority of the archive where SWIR is dead)
# with optional SWIR confirmation for the pre-failure subset.
#
# Primary VNIR test (all three thresholds must hit):
#   (mean(B01,B02,B03N) > _BRIGHT_MIN)                  bright
#   AND |B03N - B02| < _FLAT_TOL                        spectrally flat
#   AND |B01 - B02|  < _FLAT_TOL                        across VNIR
#   AND (B03N - B02)/(B03N + B02) < _NDVI_MAX           not vegetation
#
# Spectral flatness separates true cloud tops (flat across VNIR) from
# bright bare basalt / sand (red-dominated) without needing SWIR. The
# vegetation guard prevents canopy from tripping the brightness term
# in healthy NIR. Coupled with the new Phase 4 temporal cleaner the
# per-scene test is intentionally recall-oriented: catch gross cloud
# cheaply, let the temporal layer handle the residue. Thresholds are
# starting points calibrated against Faial; log to provenance and
# revisit if the cleaner's flagged-pixel fraction trends abnormally
# high or low on a new region.
#
# Optional SWIR confirmation (pre-Apr-2008 scenes only) ORs in
# pixels bright in both red AND SWIR: a separate, narrower bright-
# cloud signature that complements the VNIR flat-and-bright test.
_ASTER_CLOUD_B02_MIN     = 0.30   # red brightener (SWIR-confirmed path only)
_ASTER_CLOUD_B04_MIN     = 0.18   # SWIR brightener (SWIR-confirmed path only)
_ASTER_CLOUD_BRIGHT_MIN  = 0.24   # mean VNIR brightness for primary test
_ASTER_CLOUD_FLAT_TOL    = 0.06   # spectral flatness tolerance (reflectance)
_ASTER_CLOUD_NDVI_MAX    = 0.20   # vegetation guard ceiling
_ASTER_WATER_NDWI_MIN    = 0.00   # NDWI > 0 marks water (McFeeters 1996);
                                  # excluded from cloud candidacy so coastal
                                  # whitewater / surf is not flagged
_ASTER_CLOUD_BUFFER_PX   = 3      # edge dilation radius (cells); catches the
                                  # 1-3 px halo around marine cloud that the
                                  # per-pixel test misses

# Optional brightness-temperature cloud channel. When AST_08 (Surface
# Kinetic Temperature, V004) scenes are paired with the AST_07XT input
# by scene ID, pixels colder than this threshold are flagged as cloud
# regardless of VIS/SWIR brightness. The May 2026 Faial diagnostic
# showed that Hulley & Hook (2008)'s 295 K threshold, calibrated for
# tropical scenes, over-masks mid-latitude Atlantic land, where
# daytime surface temperatures legitimately sit in the 275-296 K range
# (cool forest canopy, north-facing slopes, early-morning passes).
# The earlier 270 K floor proved too cold to catch warm maritime
# stratus that sits on the Faial caldera at ~275-285 K and was leaking
# through the (B02 AND B04) bright-cloud conjunction; bumped to 280 K
# after the May 2026 visual inspection of AsterV3_VnirSwir_Masked.
# Still well below typical daytime Faial land (vegetated slopes report
# 285-296 K) and captures both clear cumulus tops (250-275 K) and warm
# low-altitude marine cloud (275-280 K).
#
# ASTER product family caveat: AST_08 (Surface Kinetic Temperature),
# AST_09 (VNIR/SWIR surface radiance), and AST_09T (TIR surface-
# leaving radiance) are ALL produced AFTER the operational L2 cloud
# mask is applied and are therefore clear-sky-only by construction.
# None of them describe a cloud and none can be used as a cloud
# detector. The only ASTER thermal quantity valid over cloud is at-
# sensor TOA brightness temperature from L1B / L1T radiance; that
# path is not implemented in GENESIS today but is the logical entry
# point if a thermal channel is ever revisited.
_ASTER_CLOUD_BT_MAX_K = 280.0

# AST_08 V004 stores Surface Kinetic Temperature as Int16 with a scale
# factor of 0.1 (DN x 0.1 = Kelvin). The earlier comment in this
# module claimed COG TIFFs export already-scaled Kelvin; the
# diagnostic on Faial scratch showed otherwise (raw values 2700-3000,
# the literal DN). Applied inside ``_load_bt_kelvin`` immediately
# after the 90 m to 15 m Resample.
_ASTER_TIR_SCALE = 0.1

# Lower validity bound for AST_08 Surface Kinetic Temperature, in
# Kelvin. AST_08 NoData is stored as 0 DN (which scales to 0 K) and
# Resample can introduce float32 NoData sentinels (~ -3.4e+38) near
# scene edges; both fall well below this floor. Daytime land
# brightness temperatures over Earth's surface are very rarely below
# 200 K.
_ASTER_TIR_VALID_K_FLOOR = 200.0

# Tool 07 — temporal-statistics tunables. ``NDVI_PERSISTENCE_THRESHOLD``
# is the cut-off used to count how many scenes had "vegetated" canopy
# at a pixel — Howard & Merrifield (2010) GDV-mapping uses 0.5 for
# temperate / Mediterranean biomes, 0.3 for arid / semi-arid; raise for
# tropical dense forest. ``NDWI_WATER_THRESHOLD`` follows McFeeters
# (1996): NDWI > 0 marks open water. ``GDV_DRY_FLOOR`` is the
# dry-season NDVI floor that Eamus / Naumburg use to flag persistent
# canopy (groundwater-dependent vegetation candidates).
_NDVI_PERSISTENCE_THRESHOLD = 0.5
_NDWI_WATER_THRESHOLD = 0.0
_GDV_DRY_FLOOR = 0.3

# Tool 07 stack discovery: auto-detect across the three per-scene
# scratch conventions emitted by tools 01/02/03. Each mosaic writes a
# different filename suffix (S2 -> ``*_stack.tif``, Landsat ->
# ``*_composite.tif``, ASTER -> ``*_stack.tif`` or ``*_stack_vnir.tif``)
# and the user shouldn't have to know which one applies to the scratch
# they just pointed at. When the GP parameter is left at the auto
# label, the tool globs all three patterns and unions the results.
_TOOL07_AUTO_LABEL = "(auto-detect)"
_TOOL07_AUTO_PATTERNS = (
    "*_stack.tif",
    "*_stack_vnir.tif",
    "*_composite.tif",
)

# Pure-numpy FastICA defaults (Hyvärinen parallel, logcosh). Match the
# previous sklearn defaults so the convergence behaviour is unchanged
# from the version that used scikit-learn.
_FAST_ICA_MAX_ITER = 1000
_FAST_ICA_TOL = 1e-4

# Symmetric-decorrelation eigenvalue floor in _sym_decorrelation.
# Clamps near-zero eigenvalues of ``W W^T`` before
# inverse-square-rooting so a rank-deficient ``W`` does not produce
# inf / NaN. Tighter than _EIGVAL_FLOOR_ABS because ICA whitening
# operates on a higher-dynamic-range data covariance.
_SYM_DECORRELATION_FLOOR = 1e-12


class TransformStatistics:
    """Base class for transformation statistics"""
    def __init__(self):
        self.creation_date = datetime.now()
        self.description = ""
        # errors: real failures (missing file, bad data, malformed npz).
        # warnings: informational notes that do NOT mean the object is invalid
        # (e.g., "loaded from a legacy .npz format"). Kept separate so
        # validate() does not flag a usable object as broken just because the
        # loader added a note.
        self.errors = []
        self.warnings = []

    def validate(self):
        """Base validation method"""
        return len(self.errors) == 0
        
    def save(self, filepath):
        """Save statistics to a .npz file.

        creation_date is serialised as an ISO 8601 string so np.savez writes
        a numpy unicode array rather than a Python-object array. This lets
        np.load read the file under the default allow_pickle=False (numpy
        1.16+ refuses to deserialise object arrays without explicit opt-in).
        """
        try:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            if isinstance(self.creation_date, datetime):
                creation_date_str = self.creation_date.isoformat()
            else:
                creation_date_str = str(self.creation_date)

            np.savez(
                filepath,
                creation_date=creation_date_str,
                description=self.description or "",
                **self._get_save_dict()
            )
            return True
        except Exception as e:
            self.errors.append(f"Error saving statistics: {str(e)}")
            return False

    def load(self, filepath):
        """Load statistics from a .npz file.

        Default path uses allow_pickle=False (safe). If the file contains
        numpy object arrays — typically legacy .npz files where
        creation_date was saved as a pickled datetime, or where None-valued
        optional fields were saved unconditionally — we fall back to
        allow_pickle=True and record a note so the user knows to re-save.

        SECURITY: the legacy-fallback path will execute pickle bytecode.
        Only load .npz files you produced or fully trust.
        """
        try:
            if not os.path.exists(filepath):
                self.errors.append(f"Statistics file not found: {filepath}")
                return False

            legacy = False
            try:
                data = np.load(filepath, allow_pickle=False)
                # Force eager evaluation of every key so an object-array
                # failure surfaces here, not deep inside _load_from_dict.
                for k in data.files:
                    _ = data[k]
            except ValueError as e:
                if "allow_pickle" not in str(e).lower():
                    raise
                try:
                    data.close()
                except Exception:
                    pass
                data = np.load(filepath, allow_pickle=True)
                legacy = True

            try:
                raw_date = data['creation_date']
                date_val = raw_date.item() if hasattr(raw_date, 'item') else raw_date
                if isinstance(date_val, datetime):
                    self.creation_date = date_val
                elif isinstance(date_val, str):
                    try:
                        self.creation_date = datetime.fromisoformat(date_val)
                    except ValueError:
                        self.creation_date = datetime.now()
                else:
                    self.creation_date = datetime.now()
            except KeyError:
                self.creation_date = datetime.now()

            self.description = (
                str(data['description']) if 'description' in data.files else ""
            )

            if legacy:
                self.warnings.append(
                    "Loaded a legacy .npz containing pickled objects. "
                    "Re-save with the current toolbox to upgrade format."
                )

            result = self._load_from_dict(data)
            try:
                data.close()
            except Exception:
                pass
            return result
        except Exception as e:
            self.errors.append(f"Error loading statistics: {str(e)}")
            return False
            
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        raise NotImplementedError
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        raise NotImplementedError

class MNFNoiseStatistics(TransformStatistics):
    """Statistics from first MNF rotation (noise statistics)"""
    def __init__(self):
        super().__init__()
        self.noise_covariance = None
        self.noise_eigenvalues = None
        self.noise_eigenvectors = None
        self.description = "MNF Noise Statistics"
        
    def validate(self):
        """Validate noise statistics"""
        if not super().validate():
            return False
            
        if self.noise_covariance is None:
            self.errors.append("Noise covariance matrix is missing")
        if self.noise_eigenvalues is None:
            self.errors.append("Noise eigenvalues are missing")
        if self.noise_eigenvectors is None:
            self.errors.append("Noise eigenvectors are missing")
            
        return len(self.errors) == 0
        
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        return {
            'noise_covariance': self.noise_covariance,
            'noise_eigenvalues': self.noise_eigenvalues,
            'noise_eigenvectors': self.noise_eigenvectors
        }
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.noise_covariance = data['noise_covariance']
            self.noise_eigenvalues = data['noise_eigenvalues']
            self.noise_eigenvectors = data['noise_eigenvectors']
            return True
        except Exception as e:
            self.errors.append(f"Error loading noise statistics: {str(e)}")
            return False

class MNFStatistics(TransformStatistics):
    """Statistics from MNF transformation"""
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.transform_matrix = None
        self.noise_covariance = None
        self.whitening_matrix = None
        self.signal_covariance = None
        self.component_correlation = None
        self.description = "MNF Transform Statistics"

    def validate(self):
        """Validate MNF statistics"""
        if not super().validate():
            return False

        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.eigenvalues is None:
            self.errors.append("Eigenvalues are missing")
        if self.eigenvectors is None:
            self.errors.append("Eigenvectors are missing")
        if self.transform_matrix is None:
            self.errors.append("Transform matrix is missing")

        return len(self.errors) == 0

    def _get_save_dict(self):
        """Get dictionary of values to save"""
        save_dict = {
            'band_means': self.band_means,
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'transform_matrix': self.transform_matrix
        }

        # Add optional fields if they exist
        if hasattr(self, 'noise_covariance') and self.noise_covariance is not None:
            save_dict['noise_covariance'] = self.noise_covariance

        if hasattr(self, 'whitening_matrix') and self.whitening_matrix is not None:
            save_dict['whitening_matrix'] = self.whitening_matrix

        if hasattr(self, 'signal_covariance') and self.signal_covariance is not None:
            save_dict['signal_covariance'] = self.signal_covariance

        if hasattr(self, 'component_correlation') and self.component_correlation is not None:
            save_dict['component_correlation'] = self.component_correlation

        return save_dict

    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.eigenvalues = data['eigenvalues']
            self.eigenvectors = data['eigenvectors']
            self.transform_matrix = data['transform_matrix']

            # Load optional fields if available
            if 'noise_covariance' in data:
                self.noise_covariance = data['noise_covariance']

            if 'whitening_matrix' in data:
                self.whitening_matrix = data['whitening_matrix']

            if 'signal_covariance' in data:
                self.signal_covariance = data['signal_covariance']

            if 'component_correlation' in data:
                self.component_correlation = data['component_correlation']

            return True
        except Exception as e:
            self.errors.append(f"Error loading MNF statistics: {str(e)}")
            return False

class PCAStatistics(TransformStatistics):
    """Statistics for PCA transformation"""
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance = None
        self.covariance_matrix = None
        self.description = "PCA Transform Statistics"
        
    def validate(self):
        """Validate PCA statistics"""
        if not super().validate():
            return False
            
        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.eigenvalues is None:
            self.errors.append("Eigenvalues are missing")
        if self.eigenvectors is None:
            self.errors.append("Eigenvectors are missing")
        if self.explained_variance is None:
            self.errors.append("Explained variance is missing")
        if self.covariance_matrix is None:
            self.errors.append("Covariance matrix is missing")
            
        return len(self.errors) == 0
        
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        return {
            'band_means': self.band_means,
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'explained_variance': self.explained_variance,
            'covariance_matrix': self.covariance_matrix
        }
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.eigenvalues = data['eigenvalues']
            self.eigenvectors = data['eigenvectors']
            self.explained_variance = data['explained_variance']
            self.covariance_matrix = data['covariance_matrix']
            return True
        except Exception as e:
            self.errors.append(f"Error loading PCA statistics: {str(e)}")
            return False

class ICAStatistics(TransformStatistics):
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.mixing_matrix = None
        self.unmixing_matrix = None
        self.whitening_matrix = None
        self.dewhitening_matrix = None
        self.n_iterations = None
        self.independence_metrics = None
        self.kurtosis_values = None
        self.random_state = None
        self.description = "ICA Transform Statistics"

    def validate(self):
        """Validate ICA statistics"""
        if not super().validate():
            return False

        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.mixing_matrix is None:
            self.errors.append("Mixing matrix is missing")
        if self.unmixing_matrix is None:
            self.errors.append("Unmixing matrix is missing")
        if self.whitening_matrix is None:
            self.errors.append("Whitening matrix is missing")
        if self.dewhitening_matrix is None:
            self.errors.append("Dewhitening matrix is missing")
        if self.n_iterations is None:
            self.errors.append("n_iterations is missing")
        if self.kurtosis_values is None:
            self.errors.append("kurtosis_values is missing")

        return len(self.errors) == 0

    def _get_save_dict(self):
        """Get dictionary of values to save"""
        out = {
            'band_means': self.band_means,
            'mixing_matrix': self.mixing_matrix,
            'unmixing_matrix': self.unmixing_matrix,
            'whitening_matrix': self.whitening_matrix,
            'dewhitening_matrix': self.dewhitening_matrix,
            'n_iterations': self.n_iterations,
        }
        # Optional fields: omit when None so np.savez does not write a
        # numpy object array (which np.load refuses without allow_pickle=True).
        if self.independence_metrics is not None:
            out['independence_metrics'] = self.independence_metrics
        if self.kurtosis_values is not None:
            out['kurtosis_values'] = self.kurtosis_values
        if self.random_state is not None:
            out['random_state'] = self.random_state
        return out

    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.mixing_matrix = data['mixing_matrix']
            self.unmixing_matrix = data['unmixing_matrix']
            self.whitening_matrix = data['whitening_matrix']
            self.dewhitening_matrix = data['dewhitening_matrix']
            self.n_iterations = data['n_iterations']
            if 'independence_metrics' in data:
                self.independence_metrics = data['independence_metrics']
            if 'kurtosis_values' in data:
                self.kurtosis_values = data['kurtosis_values']
            if 'random_state' in data:
                self.random_state = data['random_state']
            return True
        except Exception as e:
            self.errors.append(f"Error loading ICA statistics: {str(e)}")
            return False


# ---------------------------------------------------------------------------
# Module-level utilities (NoData-safe, arcpy-free, unit-testable).
# ---------------------------------------------------------------------------

def noise_from_valid_diffs(cube, valid_mask):
    """Estimate MNF noise via shift-difference, restricted to valid pixel pairs.

    Replaces the inlined shift-difference logic from _perform_mnf and fixes
    two issues:
      * NaN/NoData propagation: only neighbour pairs where BOTH endpoints
        are valid contribute. The previous implementation zero-padded the
        boundary and counted those zeros as valid samples, biasing the
        noise covariance toward smaller values.
      * Non-positive-definite covariance is regularised with an eigenvalue
        floor instead of an additive diagonal shift.

    Parameters
    ----------
    cube : np.ndarray
        (h, w, n_bands) data cube.
    valid_mask : np.ndarray
        (h, w) boolean mask, True where the pixel is fully valid.

    Returns
    -------
    (noise_mean, noise_cov, n_pairs)
        noise_mean is the per-band mean delta; noise_cov is the covariance
        of the deltas divided by 2 (since Var(X - Y) = 2 * Var(noise) when
        signal is smooth). Returns (None, None, 0) when fewer than
        2 * n_bands valid pairs are available.
    """
    h, w, nb = cube.shape

    h_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    h_deltas = (cube[:, :-1, :] - cube[:, 1:, :])[h_valid]

    v_valid = valid_mask[:-1, :] & valid_mask[1:, :]
    v_deltas = (cube[:-1, :, :] - cube[1:, :, :])[v_valid]

    all_deltas = np.concatenate([h_deltas, v_deltas], axis=0)
    n_pairs = int(len(all_deltas))

    if n_pairs < nb * 2:
        return None, None, 0

    noise_mean = np.mean(all_deltas, axis=0)
    noise_cov = np.cov(all_deltas.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(noise_cov)
    if np.any(eigvals < _EIGVAL_FLOOR_ABS):
        valid_eig = eigvals[eigvals > _EIGVAL_FLOOR_ABS]
        floor = (
            np.median(valid_eig) * _EIGVAL_FLOOR_RELATIVE
            if len(valid_eig) > 0
            else _EIGVAL_FLOOR_ABS
        )
        eigvals = np.maximum(eigvals, floor)
        noise_cov = (eigvecs * eigvals) @ eigvecs.T

    return noise_mean, noise_cov, n_pairs


def _project_with_band_means(cube, band_means, projection_matrix):
    """Internal: center, neutralise NaN, project, restore per-pixel NaN.

    Shared by transform_pca / transform_mnf / transform_ica. The math is the
    same for all three — only the projection_matrix differs.

    NaN handling: filling NaN bands with band_mean prior to centering is
    equivalent to centering first and then replacing NaN with 0, since
    (band_mean - band_mean) = 0. We use the latter (vectorised, single pass)
    via np.nan_to_num, then restore NaN in every output component for any
    pixel that had at least one non-finite band on input.
    """
    h, w, nb = cube.shape
    if nb != band_means.shape[0]:
        raise ValueError(
            f"Band count mismatch: cube has {nb} bands, statistics has "
            f"{band_means.shape[0]} bands."
        )
    # astype(copy=True) is the default; one allocation, no redundant .copy().
    flat = cube.reshape(-1, nb).astype(np.float64)
    nan_per_pixel = np.any(~np.isfinite(flat), axis=1)
    centered = flat - band_means  # NaN bands stay NaN
    # In-place zero out NaN entries — equivalent to filling with band_mean
    # pre-centering, but vectorised in one C-level pass.
    np.nan_to_num(centered, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    projected = centered @ projection_matrix
    projected[nan_per_pixel] = np.nan
    n_components = projection_matrix.shape[1]
    return projected.reshape(h, w, n_components).astype(np.float32)


def transform_pca(new_cube, stats, n_components=None):
    """Apply a fitted PCAStatistics to a new (h, w, n_bands) cube.

    Implements ISS-005 — cross-AOI re-application. The PCA projection is
    `(X - mean) @ eigenvectors[:, :n]`. Pixels with any non-finite band
    become NaN in every output component.
    """
    if stats.band_means is None or stats.eigenvectors is None:
        raise ValueError("PCAStatistics is missing band_means or eigenvectors.")
    available = stats.eigenvectors.shape[1]
    if n_components is None:
        n_components = available
    if n_components > available:
        raise ValueError(
            f"Requested {n_components} components but only {available} were saved."
        )
    return _project_with_band_means(
        new_cube,
        np.asarray(stats.band_means),
        np.asarray(stats.eigenvectors)[:, :n_components],
    )


def transform_mnf(new_cube, stats, n_components=None):
    """Apply a fitted MNFStatistics to a new (h, w, n_bands) cube.

    Uses the precomputed transform_matrix = whitening_matrix @ signal_eigenvecs
    so the projection is a single matmul. The shape is (n_bands, n_components_saved);
    passing n_components larger than that is rejected.
    """
    if stats.band_means is None or stats.transform_matrix is None:
        raise ValueError("MNFStatistics is missing band_means or transform_matrix.")
    available = stats.transform_matrix.shape[1]
    if n_components is None:
        n_components = available
    if n_components > available:
        raise ValueError(
            f"Requested {n_components} components but only {available} were saved."
        )
    return _project_with_band_means(
        new_cube,
        np.asarray(stats.band_means),
        np.asarray(stats.transform_matrix)[:, :n_components],
    )


def transform_ica(new_cube, stats):
    """Apply a fitted ICAStatistics to a new (h, w, n_bands) cube.

    Sources are recovered with `(X - mean) @ unmixing_matrix.T` — the
    sklearn FastICA convention. Always returns all stored components;
    select_by_kurtosis can be used afterwards to pick interesting ones.
    """
    if stats.band_means is None or stats.unmixing_matrix is None:
        raise ValueError("ICAStatistics is missing band_means or unmixing_matrix.")
    return _project_with_band_means(
        new_cube,
        np.asarray(stats.band_means),
        np.asarray(stats.unmixing_matrix).T,
    )


# ---------------------------------------------------------------------------
# Component selectors (ISS-008, ISS-014).
# ---------------------------------------------------------------------------

def select_by_variance(eigenvalues, threshold=0.95):
    """Smallest n such that cumulative explained variance >= threshold.

    Standard PCA component count rule. `threshold` is a fraction in (0, 1].
    """
    arr = np.asarray(eigenvalues, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 1
    cum = np.cumsum(arr) / total
    n = int(np.searchsorted(cum, threshold) + 1)
    return max(1, min(n, len(arr)))


def select_by_eigenvalue(eigenvalues, threshold=1.0):
    """Count components with eigenvalue > threshold (Kaiser criterion).

    For MNF, eigenvalues are 1 + SNR — values above 1.0 indicate signal
    dominates noise. Returns at least 1.
    """
    arr = np.asarray(eigenvalues, dtype=np.float64)
    return max(1, int(np.sum(arr > threshold)))


def select_by_kurtosis(kurtosis_values, threshold=3.0):
    """Indices of ICA components with |kurtosis| > threshold.

    High |kurtosis| ≈ sharp / non-Gaussian distribution; useful for
    isolating distinct mineral signatures or geological boundaries.
    Returns a numpy array of indices (possibly empty).
    """
    arr = np.asarray(kurtosis_values, dtype=np.float64)
    return np.flatnonzero(np.abs(arr) > threshold)


def hfc_vd(data_2d, false_alarm=1e-3, max_samples=200_000):
    """Harsanyi-Farrand-Chang Virtual Dimensionality (Chang & Du 2004).

    Automatic estimator of the number of spectrally distinct signal
    sources. Operates on a 2D pixels-by-bands matrix (drop the spatial
    dimensions first). Pixels with NaN must be filtered by the caller.

    Parameters
    ----------
    data_2d : array, shape (n_pixels, n_bands)
    false_alarm : float
        Neyman-Pearson false alarm probability (smaller → fewer components).
    max_samples : int
        Random subsample size when n_pixels is large.

    Returns
    -------
    int
        Estimated number of signal sources, >= 0.
    """
    data = np.asarray(data_2d, dtype=np.float64)
    n_pixels, n_bands = data.shape
    if n_pixels > max_samples:
        rng = np.random.default_rng(42)
        data = data[rng.choice(n_pixels, max_samples, replace=False)]
        n_pixels = max_samples

    R = (data.T @ data) / n_pixels
    mean = data.mean(axis=0, keepdims=True)
    C = ((data - mean).T @ (data - mean)) / n_pixels
    eig_R = np.sort(np.linalg.eigvalsh(R))[::-1]
    eig_C = np.sort(np.linalg.eigvalsh(C))[::-1]

    z_threshold = scipy.stats.norm.isf(false_alarm)
    sigma2 = (2.0 / n_pixels) * (eig_R ** 2 + eig_C ** 2)
    sigma = np.sqrt(np.maximum(sigma2, _HFC_SIGMA2_FLOOR))
    return int(np.sum((eig_R - eig_C) > z_threshold * sigma))


# ---------------------------------------------------------------------------
# Sensor abstraction (Phase 1).
#
# Tools 04, 05, and 06 are sensor-aware via a single "Sensor Type" GP
# parameter that takes one of four values. The band-role lookup below
# resolves role names ("Red", "NIR", "SWIR1", ...) to band indices per
# sensor, so the SAME index formula expressed in terms of roles works
# across all three sensors.
# ---------------------------------------------------------------------------

SENSOR_AUTO = "Auto-detect"
SENSOR_LANDSAT_89 = "Landsat 8/9"
SENSOR_SENTINEL2 = "Sentinel-2"
SENSOR_ASTER = "ASTER"

SENSOR_CHOICES = [SENSOR_AUTO, SENSOR_LANDSAT_89, SENSOR_SENTINEL2, SENSOR_ASTER]

# Band-role mapping per sensor. Values are 1-based band indices into the
# canonical multiband stack documented for each sensor (see file header).
#
# Universal roles (Red, NIR, SWIR1, SWIR2) exist for all three sensors so
# universal indices (NDVI, NDWI, NDMI, NDBI) compute via the same lambda.
#
# Sensor-specific roles:
#   - Coastal (Landsat only)
#   - RedEdge1/2/3 + NarrowNIR (Sentinel-2 only — enables NDRE, IRECI, ...)
#   - SWIR2_2165 / SWIR2_2205 / SWIR2_2260 / SWIR2_2330 (ASTER only —
#     each is a separate 30m SWIR band; enables alunite, kaolinite,
#     muscovite, calcite indices)
#
# Note: ASTER has NO blue band — the "Blue" role is intentionally absent
# from the ASTER mapping. Indices that need Blue (e.g., L8 Iron Oxide
# B4/B2) get an ASTER-specific reformulation (Red/Green = B2/B1).
SENSOR_BAND_ROLES = {
    SENSOR_LANDSAT_89: {
        "Coastal": 1,
        "Blue": 2,
        "Green": 3,
        "Red": 4,
        "NIR": 5,
        "SWIR1": 6,
        "SWIR2": 7,
    },
    SENSOR_SENTINEL2: {
        # 12-band L2A stack in S2 wavelength order. B01 + B09 are
        # included so the output Band_N count and order match the
        # native L2A product 1:1 (Band_N = BNN nativa) for all bands
        # except B8A which sits between B08 and B09 in wavelength.
        # B10 is excluded — Sen2Cor strips it during L1C→L2A
        # atmospheric correction so it does not exist in L2A.
        "Coastal": 1,       # B01 443nm  (60m native → resampled to 10m)
        "Blue": 2,          # B02 490nm
        "Green": 3,         # B03 560nm
        "Red": 4,           # B04 665nm
        "RedEdge1": 5,      # B05 705nm
        "RedEdge2": 6,      # B06 740nm
        "RedEdge3": 7,      # B07 783nm
        "NIR": 8,           # B08 842nm
        "NarrowNIR": 9,     # B8A 865nm
        "WaterVapour": 10,  # B09 945nm  (60m native → resampled to 10m)
        "SWIR1": 11,        # B11 1610nm
        "SWIR2": 12,        # B12 2190nm
    },
    SENSOR_ASTER: {
        # AST_07XT V004 9-band stack: VNIR B1-B3N + crosstalk-corrected SWIR B4-B9
        "Green": 1,           # B1 0.52-0.60μm
        "Red": 2,             # B2 0.63-0.69μm
        "NIR": 3,             # B3N 0.76-0.86μm (nadir-looking)
        "SWIR1": 4,           # B4 1.60-1.70μm  (≈L8 B6, S2 B11)
        "SWIR2_2165": 5,      # B5 2.145-2.185μm
        "SWIR2_2205": 6,      # B6 2.185-2.225μm
        "SWIR2_2260": 7,      # B7 2.235-2.285μm
        "SWIR2_2330": 8,      # B8 2.295-2.365μm
        "SWIR2": 9,           # B9 2.360-2.430μm (≈L8 B7, S2 B12 — generic SWIR2)
    },
}


def get_band(bands, role, sensor):
    """Resolve a band-role name to a band raster for the given sensor.

    Parameters
    ----------
    bands : dict
        {1: Raster, 2: Raster, ...} as built by the indices/SAM tool's
        per-band ExtractBand loop.
    role : str
        A band role from SENSOR_BAND_ROLES, e.g., "Red", "NIR", "SWIR1".
    sensor : str
        One of SENSOR_LANDSAT_89 / SENSOR_SENTINEL2 / SENSOR_ASTER.

    Returns
    -------
    Raster
        The arcpy Raster for that band in the stack.

    Raises
    ------
    KeyError
        If the role isn't available for the requested sensor (e.g., asking
        for "Blue" on ASTER, or "RedEdge1" on Landsat). The caller (an
        index lambda) catches this and skips the index gracefully.
    """
    mapping = SENSOR_BAND_ROLES.get(sensor)
    if mapping is None:
        raise ValueError(f"Unknown sensor: {sensor!r}")
    if role not in mapping:
        raise KeyError(
            f"Band role {role!r} is not available for {sensor!r}. "
            f"Available roles: {sorted(mapping.keys())}"
        )
    idx = mapping[role]
    if idx not in bands:
        raise KeyError(
            f"Band {idx} (role {role!r} for {sensor!r}) was not extracted; "
            f"the input raster may have fewer bands than expected."
        )
    return bands[idx]


def detect_sensor(raster_path):
    """Auto-detect sensor from raster path and (as fallback) band count.

    Returns one of SENSOR_LANDSAT_89 / SENSOR_SENTINEL2 / SENSOR_ASTER,
    or None if no confident match. The GP UI presents this as the default
    for the sensor-selector dropdown; the user can override.

    Detection cascade (most specific first):
      1. Filename prefix / token:
           LC08_, LC09_                → Landsat 8/9
           S2A_, S2B_, _MSIL2A_         → Sentinel-2
           AST_07XT                     → ASTER
      2. Band count (less reliable; only used if path/filename gives no
         hint):
           7 bands  → Landsat 8/9
           12 bands → Sentinel-2 (10-band legacy stack still accepted)
           9 bands  → ASTER
    """
    if not raster_path:
        return None
    name = os.path.basename(str(raster_path)).upper()

    if "LC08_" in name or "LC09_" in name:
        return SENSOR_LANDSAT_89
    if name.startswith("S2A_") or name.startswith("S2B_") or "_MSIL2A_" in name:
        return SENSOR_SENTINEL2
    if "AST_07XT" in name:
        return SENSOR_ASTER

    # Fall back to band count
    try:
        desc = arcpy.Describe(raster_path)
        if hasattr(desc, "bandCount"):
            count = int(desc.bandCount)
            if count == 7:
                return SENSOR_LANDSAT_89
            # 12-band current S2 (B01..B12 minus B10) plus 10-band legacy
            # output (pre-multi-spectral expansion) — both resolve to S2.
            if count in (10, 12):
                return SENSOR_SENTINEL2
            if count == 9:
                return SENSOR_ASTER
            # 3-band stacks are the ASTER VNIR-only product (B01, B02,
            # B03N), produced by Tool 03 when the SWIR detector failed
            # for that scene's epoch. Tool 07 and friends need to know
            # this is still ASTER so the band-role resolution works.
            if count == 3:
                return SENSOR_ASTER
    except Exception:
        pass
    return None


def make_sensor_parameter(name="sensor_type", display="Sensor Type"):
    """Build the standard sensor-selector GP parameter.

    Used by Tools 04 (Indices), 05 (Transformations) and 06 (SAM). The
    parameter defaults to "Auto-detect"; updateParameters in each tool
    is responsible for resolving Auto-detect against the input raster
    and either updating the displayed value or surfacing a clear error
    if detection fails.
    """
    param = arcpy.Parameter(
        displayName=display,
        name=name,
        datatype="GPString",
        parameterType="Required",
        direction="Input",
    )
    param.filter.list = SENSOR_CHOICES
    param.value = SENSOR_AUTO
    return param


def resolve_sensor(sensor_param_value, input_raster_path):
    """Resolve a sensor selection into a concrete sensor constant.

    Used at execute() time. If the user left the parameter at
    "Auto-detect", we run detect_sensor on the input raster. If auto
    detection fails, raises ValueError with a clear remediation message
    (telling the user to pick the sensor manually).
    """
    if sensor_param_value and sensor_param_value != SENSOR_AUTO:
        return sensor_param_value
    detected = detect_sensor(input_raster_path)
    if detected is None:
        raise ValueError(
            f"Could not auto-detect sensor from input {input_raster_path!r}. "
            f"Please set Sensor Type explicitly in the tool dialog "
            f"(one of: {', '.join(SENSOR_CHOICES[1:])})."
        )
    return detected


# ---------------------------------------------------------------------------
# Toolbox class (Phase 1 skeleton — tools are populated in subsequent phases).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Index & composite catalogue (Phase 2).
#
# Sensor-aware indices/composites use band ROLES (resolved at runtime via
# get_band) so the same compute lambda works across every sensor that has
# the required roles. The display_labels dict gives the user-facing string
# in the ArcGIS Pro Catalog pane — sensor-specific so the user always sees
# the formula spelled out in that sensor's band labels.
#
# An index appears in the GP UI for a given sensor iff ALL of:
#   1. Every required_role is in SENSOR_BAND_ROLES[sensor]
#   2. The index has a display_labels[sensor] entry
#   3. If sensor_filter is set, the sensor is in that list (overrides 1+2)
#
# Two small helpers keep the formula lambdas readable:
# ---------------------------------------------------------------------------


def _normalized_diff(bands, sensor, role_a, role_b):
    """(role_a - role_b) / (role_a + role_b) — the NDVI family."""
    a = get_band(bands, role_a, sensor)
    b = get_band(bands, role_b, sensor)
    return Divide(Minus(a, b), Plus(a, b))


def _ratio(bands, sensor, numerator_role, denominator_role):
    """role_a / role_b — simple band ratio."""
    return Divide(
        get_band(bands, numerator_role, sensor),
        get_band(bands, denominator_role, sensor),
    )


INDICES = {
    # =====================================================================
    # Vegetation
    # =====================================================================
    "NDVI": {
        "category": "Vegetation",
        "required_roles": ["Red", "NIR"],
        "display_labels": {
            SENSOR_LANDSAT_89: "NDVI (B5-B4)/(B5+B4)",
            SENSOR_SENTINEL2: "NDVI (B08-B04)/(B08+B04)",
            SENSOR_ASTER: "NDVI (B3N-B2)/(B3N+B2)",
        },
        "output_suffix": "NDVI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "NIR", "Red"),
    },
    "SAVI": {
        "category": "Vegetation",
        "required_roles": ["Red", "NIR"],
        "display_labels": {
            SENSOR_LANDSAT_89: "SAVI 1.5*(B5-B4)/(B5+B4+0.5)",
            SENSOR_SENTINEL2: "SAVI 1.5*(B08-B04)/(B08+B04+0.5)",
            SENSOR_ASTER: "SAVI 1.5*(B3N-B2)/(B3N+B2+0.5)",
        },
        "output_suffix": "SAVI",
        "compute": lambda bands, sensor: Times(
            Divide(
                Minus(get_band(bands, "NIR", sensor), get_band(bands, "Red", sensor)),
                Plus(Plus(get_band(bands, "NIR", sensor), get_band(bands, "Red", sensor)), 0.5),
            ),
            1.5,
        ),
    },
    # =====================================================================
    # Water
    # =====================================================================
    "NDWI_McFeeters": {
        "category": "Water",
        "required_roles": ["Green", "NIR"],
        "display_labels": {
            SENSOR_LANDSAT_89: "NDWI McFeeters (B3-B5)/(B3+B5)",
            SENSOR_SENTINEL2: "NDWI McFeeters (B03-B08)/(B03+B08)",
            SENSOR_ASTER: "NDWI McFeeters (B1-B3N)/(B1+B3N)",
        },
        "output_suffix": "NDWI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "Green", "NIR"),
    },
    "MNDWI": {
        "category": "Water",
        "required_roles": ["Green", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "MNDWI (B3-B6)/(B3+B6)",
            SENSOR_SENTINEL2: "MNDWI (B03-B11)/(B03+B11)",
            SENSOR_ASTER: "MNDWI (B1-B4)/(B1+B4)",
        },
        "output_suffix": "MNDWI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "Green", "SWIR1"),
    },
    # =====================================================================
    # Moisture / Built-up
    # =====================================================================
    "NDMI": {
        "category": "Moisture",
        "required_roles": ["NIR", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "NDMI (B5-B6)/(B5+B6)",
            SENSOR_SENTINEL2: "NDMI (B08-B11)/(B08+B11)",
            SENSOR_ASTER: "NDMI (B3N-B4)/(B3N+B4)",
        },
        "output_suffix": "NDMI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "NIR", "SWIR1"),
    },
    "NDBI": {
        "category": "Built-up",
        "required_roles": ["NIR", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "NDBI (B6-B5)/(B6+B5)",
            SENSOR_SENTINEL2: "NDBI (B11-B08)/(B11+B08)",
            SENSOR_ASTER: "NDBI (B4-B3N)/(B4+B3N)",
        },
        "output_suffix": "NDBI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "SWIR1", "NIR"),
    },
    # =====================================================================
    # Geological — universal (no Blue requirement)
    # =====================================================================
    "Clay_Minerals_SWIR": {
        "category": "Geological",
        "required_roles": ["SWIR1", "SWIR2"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Clay Minerals B6/B7 (SWIR1/SWIR2)",
            SENSOR_SENTINEL2: "Clay Minerals B11/B12 (SWIR1/SWIR2)",
            SENSOR_ASTER: "Clay Minerals B4/B9 (SWIR1/SWIR2)",
        },
        "output_suffix": "ClaySWIR",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR1", "SWIR2"),
    },
    "Silica_Index": {
        "category": "Geological",
        "required_roles": ["NIR", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Silica Index B6/B5 (SWIR1/NIR)",
            SENSOR_SENTINEL2: "Silica Index B11/B08 (SWIR1/NIR)",
            SENSOR_ASTER: "Silica Index B4/B3N (SWIR1/NIR)",
        },
        "output_suffix": "Silica",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR1", "NIR"),
    },
    "Ferrous_Iron_Ratio": {
        "category": "Geological",
        "required_roles": ["Red", "NIR"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Ferrous Iron B5/B4 (NIR/Red)",
            SENSOR_SENTINEL2: "Ferrous Iron B08/B04 (NIR/Red)",
            SENSOR_ASTER: "Ferrous Iron B3N/B2 (NIR/Red)",
        },
        "output_suffix": "FerrousIron",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "NIR", "Red"),
    },
    "AAI_Advanced_Argillic": {
        "category": "Geological",
        "required_roles": ["Red", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Adv. Argillic (B4-B6)/(B4+B6)",
            SENSOR_SENTINEL2: "Adv. Argillic (B04-B11)/(B04+B11)",
            SENSOR_ASTER: "Adv. Argillic (B2-B4)/(B2+B4)",
        },
        "output_suffix": "AAI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "Red", "SWIR1"),
    },
    # =====================================================================
    # Geological — need Blue (Landsat 8/9 + Sentinel-2 only)
    # =====================================================================
    "Iron_Oxide_RedBlue": {
        "category": "Geological",
        "required_roles": ["Red", "Blue"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Iron Oxide B4/B2 (Red/Blue)",
            SENSOR_SENTINEL2: "Iron Oxide B04/B02 (Red/Blue)",
        },
        "output_suffix": "IronOxide",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "Red", "Blue"),
    },
    "Ferric_Iron_Ratio": {
        "category": "Geological",
        "required_roles": ["Red", "Blue"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Ferric Iron B4/B2 (Red/Blue)",
            SENSOR_SENTINEL2: "Ferric Iron B04/B02 (Red/Blue)",
        },
        "output_suffix": "FerricIron",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "Red", "Blue"),
    },
    "Gossan_Index": {
        "category": "Geological",
        "required_roles": ["Red", "Blue", "Green"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Gossan (B4/B2)*(B4/B3)",
            SENSOR_SENTINEL2: "Gossan (B04/B02)*(B04/B03)",
        },
        "output_suffix": "Gossan",
        "compute": lambda bands, sensor: Times(
            _ratio(bands, sensor, "Red", "Blue"),
            _ratio(bands, sensor, "Red", "Green"),
        ),
    },
    # =====================================================================
    # Geological — ASTER analog for missing Blue (Red/Green substitute)
    # =====================================================================
    "Iron_Oxide_ASTER_RG": {
        "category": "Geological",
        "required_roles": ["Red", "Green"],
        "display_labels": {
            SENSOR_ASTER: "Iron Oxide ASTER B2/B1 (Red/Green)",
        },
        "output_suffix": "IronOxide_RG",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "Red", "Green"),
        # sensor_filter restricts this index to ASTER even though L8/9 and
        # S2 also have Red+Green — they have the better Red/Blue version.
        "sensor_filter": [SENSOR_ASTER],
    },
    # =====================================================================
    # Red-Edge — Sentinel-2 only
    # =====================================================================
    "NDRE": {
        "category": "Red-Edge",
        "required_roles": ["NIR", "RedEdge1"],
        "display_labels": {
            SENSOR_SENTINEL2: "NDRE (B08-B05)/(B08+B05)",
        },
        "output_suffix": "NDRE",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "NIR", "RedEdge1"),
    },
    "CIred_edge": {
        "category": "Red-Edge",
        "required_roles": ["RedEdge1", "RedEdge3"],
        "display_labels": {
            SENSOR_SENTINEL2: "CIred-edge (B07/B05) - 1",
        },
        "output_suffix": "CIredEdge",
        "compute": lambda bands, sensor: Minus(_ratio(bands, sensor, "RedEdge3", "RedEdge1"), 1),
    },
    "IRECI": {
        "category": "Red-Edge",
        "required_roles": ["Red", "RedEdge1", "RedEdge2", "RedEdge3"],
        "display_labels": {
            SENSOR_SENTINEL2: "IRECI (B07-B04)/(B05/B06)",
        },
        "output_suffix": "IRECI",
        "compute": lambda bands, sensor: Divide(
            Minus(get_band(bands, "RedEdge3", sensor), get_band(bands, "Red", sensor)),
            _ratio(bands, sensor, "RedEdge1", "RedEdge2"),
        ),
    },
    # =====================================================================
    # ASTER Minerals — per-wavelength SWIR (ASTER only)
    # =====================================================================
    "Alunite_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2165", "SWIR2_2260"],
        "display_labels": {
            SENSOR_ASTER: "Alunite B7/B5 (2.260/2.165 μm)",
        },
        "output_suffix": "Alunite",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR2_2260", "SWIR2_2165"),
    },
    "Kaolinite_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2165", "SWIR2_2260"],
        "display_labels": {
            SENSOR_ASTER: "Kaolinite B5/B7 (2.165/2.260 μm)",
        },
        "output_suffix": "Kaolinite",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR2_2165", "SWIR2_2260"),
    },
    "Muscovite_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2165", "SWIR2_2205"],
        "display_labels": {
            SENSOR_ASTER: "Muscovite B6/B5 (2.205/2.165 μm)",
        },
        "output_suffix": "Muscovite",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR2_2205", "SWIR2_2165"),
    },
    "Calcite_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2260", "SWIR2_2330"],
        "display_labels": {
            SENSOR_ASTER: "Calcite B8/B7 (2.330/2.260 μm)",
        },
        "output_suffix": "Calcite",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR2_2330", "SWIR2_2260"),
    },
    "Hydrothermal_Cudahy_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2165", "SWIR2_2205", "SWIR2_2260"],
        "display_labels": {
            SENSOR_ASTER: "Hydrothermal (Cudahy) (B5/B6)*(B7/B6)",
        },
        "output_suffix": "Hydrothermal",
        "compute": lambda bands, sensor: Times(
            _ratio(bands, sensor, "SWIR2_2165", "SWIR2_2205"),
            _ratio(bands, sensor, "SWIR2_2260", "SWIR2_2205"),
        ),
    },
}


COMPOSITES = {
    "Natural_Color_RGB": {
        "category": "True Color",
        "required_roles": ["Red", "Green", "Blue"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Natural Color (B4,B3,B2)",
            SENSOR_SENTINEL2: "Natural Color (B04,B03,B02)",
        },
        "output_suffix": "NaturalColor",
        "band_spec": ["Red", "Green", "Blue"],
    },
    "False_Color_IR": {
        "category": "False Color",
        "required_roles": ["NIR", "Red", "Green"],
        "display_labels": {
            SENSOR_LANDSAT_89: "False Color IR (B5,B4,B3)",
            SENSOR_SENTINEL2: "False Color IR (B08,B04,B03)",
            SENSOR_ASTER: "False Color IR (B3N,B2,B1)",
        },
        "output_suffix": "FalseColorIR",
        "band_spec": ["NIR", "Red", "Green"],
    },
    "SWIR_Geology": {
        "category": "Geology",
        "required_roles": ["SWIR2", "SWIR1", "Red"],
        "display_labels": {
            SENSOR_LANDSAT_89: "SWIR Geology (B7,B6,B4)",
            SENSOR_SENTINEL2: "SWIR Geology (B12,B11,B04)",
            SENSOR_ASTER: "SWIR Geology (B9,B4,B2)",
        },
        "output_suffix": "SWIRGeology",
        "band_spec": ["SWIR2", "SWIR1", "Red"],
    },
    "Land_Surface": {
        "category": "Geology",
        "required_roles": ["NIR", "SWIR1", "Red"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Land Surface (B5,B6,B4)",
            SENSOR_SENTINEL2: "Land Surface (B08,B11,B04)",
            SENSOR_ASTER: "Land Surface (B3N,B4,B2)",
        },
        "output_suffix": "LandSurface",
        "band_spec": ["NIR", "SWIR1", "Red"],
    },
    "Lithological_NRB": {
        "category": "Geology",
        "required_roles": ["NIR", "Red", "Blue"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Lithological (B5,B4,B2)",
            SENSOR_SENTINEL2: "Lithological (B08,B04,B02)",
        },
        "output_suffix": "Lithological",
        "band_spec": ["NIR", "Red", "Blue"],
    },
    "Vegetation_RedEdge_S2": {
        "category": "Red-Edge",
        "required_roles": ["NIR", "RedEdge1", "Red"],
        "display_labels": {
            SENSOR_SENTINEL2: "Vegetation Red-Edge (B08,B05,B04)",
        },
        "output_suffix": "VegRedEdge",
        "band_spec": ["NIR", "RedEdge1", "Red"],
    },
    "ASTER_Mineral_RGB": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2205", "SWIR2_2165", "Green"],
        "display_labels": {
            SENSOR_ASTER: "ASTER Mineral RGB (B6,B5,B1)",
        },
        "output_suffix": "ASTERMineralRGB",
        "band_spec": ["SWIR2_2205", "SWIR2_2165", "Green"],
    },
    "ASTER_Alteration_RGB": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2260", "SWIR2_2205", "SWIR2_2165"],
        "display_labels": {
            SENSOR_ASTER: "ASTER Alteration RGB (B7,B6,B5)",
        },
        "output_suffix": "ASTERAlterationRGB",
        "band_spec": ["SWIR2_2260", "SWIR2_2205", "SWIR2_2165"],
    },
}


# Category display order in the GP dropdown — controls grouping.
_INDEX_CATEGORY_ORDER = [
    "Vegetation", "Water", "Moisture", "Built-up",
    "Geological", "Red-Edge", "ASTER Minerals",
]
_COMPOSITE_CATEGORY_ORDER = [
    "True Color", "False Color", "Geology", "Red-Edge", "ASTER Minerals",
]


def _applicable_entries(catalogue, sensor, category_order):
    """Filter a catalogue (INDICES or COMPOSITES) to entries available for
    the sensor, grouped by category in canonical display order."""
    if sensor not in SENSOR_BAND_ROLES:
        return {}
    roles = SENSOR_BAND_ROLES[sensor]
    grouped = {}
    for key, meta in catalogue.items():
        if meta.get("sensor_filter") and sensor not in meta["sensor_filter"]:
            continue
        if not all(r in roles for r in meta["required_roles"]):
            continue
        label = meta["display_labels"].get(sensor)
        if label is None:
            continue
        grouped.setdefault(meta["category"], []).append(label)
    # Re-order keys by canonical category order.
    ordered = {}
    for cat in category_order:
        if cat in grouped:
            ordered[cat] = grouped[cat]
    for cat in grouped:
        if cat not in ordered:
            ordered[cat] = grouped[cat]
    return ordered


def applicable_indices(sensor):
    """Indices for a sensor, grouped by category in canonical order."""
    return _applicable_entries(INDICES, sensor, _INDEX_CATEGORY_ORDER)


def applicable_composites(sensor):
    """Composites for a sensor, grouped by category in canonical order."""
    return _applicable_entries(COMPOSITES, sensor, _COMPOSITE_CATEGORY_ORDER)


def _flat_labels(grouped):
    """Flatten a grouped dict (category -> [labels]) preserving order."""
    out = []
    for cat in grouped:
        out.extend(grouped[cat])
    return out


def applicable_index_labels_flat(sensor):
    """Flat ordered list for the GP filter list."""
    return _flat_labels(applicable_indices(sensor))


def applicable_composite_labels_flat(sensor):
    return _flat_labels(applicable_composites(sensor))


def label_to_index_key(label, sensor):
    """Reverse-map a display label to its INDICES key. None if unmatched."""
    for key, meta in INDICES.items():
        if meta["display_labels"].get(sensor) == label:
            return key
    return None


def label_to_composite_key(label, sensor):
    for key, meta in COMPOSITES.items():
        if meta["display_labels"].get(sensor) == label:
            return key
    return None


# ---------------------------------------------------------------------------
# Toolbox class. Phase 2 lands Tool 04 (IndicesComposites); the mosaic
# tools (01-03) and the trivial 05/06 ports land in subsequent phases.
# ---------------------------------------------------------------------------


class Toolbox(object):
    def __init__(self):
        self.label = "GENESIS — Satellite Analysis Toolbox"
        self.alias = "genesis"
        # Workflow order. `.pyt` files have no nested toolsets, so numeric
        # prefixes in tool labels convey ordering in the Catalog pane.
        self.tools = [
            Sentinel2Mosaic,        # 01
            LandsatMosaic,          # 02
            AsterMosaic,            # 03
            IndicesComposites,      # 04
            Transformations,        # 05
            SpectralAngleMapper,    # 06
            TemporalStatistics,     # 07
        ]


# ---------------------------------------------------------------------------
# Tool 04 — Spectral Indices & Composites
# ---------------------------------------------------------------------------


class IndicesComposites(object):
    """Sensor-aware spectral indices and RGB composites.

    Accepts a pre-stacked multiband raster from any of the three supported
    sensors. The Sensor Type parameter drives band-role resolution and
    filters the indices / composites dropdowns to only those that the
    selected sensor can compute (e.g., NDRE only shows for Sentinel-2,
    Alunite only for ASTER, Iron Oxide via Red/Blue only for L8/9 + S2).
    """

    def __init__(self):
        self.label = "04 — Spectral Indices & Composites"
        self.description = (
            "Compute spectral indices and RGB band composites on a "
            "pre-stacked Landsat 8/9, Sentinel-2 L2A, or ASTER AST_07XT "
            "multiband raster. The indices / composites dropdowns are "
            "automatically filtered to those the selected sensor can "
            "actually compute given its band roles."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        input_raster = arcpy.Parameter(
            displayName="Input Multiband Raster",
            name="input_raster",
            datatype=["DERasterDataset", "GPRasterLayer"],
            parameterType="Required",
            direction="Input",
        )

        sensor_type = make_sensor_parameter()

        indices = arcpy.Parameter(
            displayName="Select Indices",
            name="indices",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
        )
        # ``filter.type = "ValueList"`` MUST be set explicitly before
        # ``filter.list`` on a multi-value parameter. Without it Pro
        # picks a default filter type that is not fully compatible
        # with the multi-value checkbox control, and the user's tick
        # state silently disappears on every UI interaction (folder
        # browse, Run click, etc.) — the same pattern that works in
        # the sibling era5land toolbox uses this explicit assignment.
        indices.filter.type = "ValueList"
        indices.filter.list = []

        composites = arcpy.Parameter(
            displayName="Select Composites",
            name="composites",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
        )
        composites.filter.type = "ValueList"
        composites.filter.list = []

        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )

        out_prefix = arcpy.Parameter(
            displayName="Output Prefix",
            name="out_prefix",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )

        rescale = arcpy.Parameter(
            displayName="Rescale Outputs to [0, 255]",
            name="rescale",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        rescale.value = False

        mask_feature = arcpy.Parameter(
            displayName="Optional AOI Mask (polygon)",
            name="mask_feature",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
        )
        mask_feature.filter.list = ["Polygon"]

        return [
            input_raster, sensor_type, indices, composites,
            out_workspace, out_prefix, rescale, mask_feature,
        ]

    def updateParameters(self, parameters):
        """Rebuild the indices / composites filter lists only when the
        desired content differs from what is currently in the dialog.

        Same pattern as the sibling ``era5land-arcgis-tools`` toolbox:
        a direct ``filter.list != desired`` comparison with no
        wrapping, no ``str()`` coercion, no instance / class cache.
        Combined with the explicit ``filter.type = "ValueList"``
        declarations in ``getParameterInfo``, Pro keeps the user's
        tick state intact through every dialog interaction.

        Identical-list re-assignment of a multi-value ``filter.list``
        is what causes Pro to wipe ``.value`` underneath the user.
        The conditional re-assignment below is the single defence
        against that — when the resolved sensor has not changed the
        method is a no-op and the ticks survive untouched.
        """
        try:
            input_raster = parameters[0]
            sensor_param = parameters[1]
            indices = parameters[2]
            composites = parameters[3]

            # Resolve the effective sensor: honour an explicit choice;
            # otherwise auto-detect from the input raster.
            effective_sensor = sensor_param.valueAsText
            if effective_sensor == SENSOR_AUTO and input_raster.valueAsText:
                detected = detect_sensor(input_raster.valueAsText)
                if detected is not None:
                    effective_sensor = detected

            # Desired filter content for this sensor.
            if effective_sensor and effective_sensor != SENSOR_AUTO:
                desired_idx = applicable_index_labels_flat(effective_sensor)
                desired_cmp = applicable_composite_labels_flat(effective_sensor)
            else:
                desired_idx = []
                desired_cmp = []

            # Only touch ``filter.list`` when its content genuinely
            # needs to change. arcpy's own equality on the ``ValueList``
            # wrapper works correctly when ``filter.type`` is the
            # canonical ``"ValueList"`` declared up in
            # ``getParameterInfo``.
            if indices.filter.list != desired_idx:
                indices.filter.list = desired_idx
            if composites.filter.list != desired_cmp:
                composites.filter.list = desired_cmp
        except Exception:
            # updateParameters must never raise — it runs on every
            # keystroke and an exception would freeze the dialog.
            pass

    def updateMessages(self, parameters):
        """Lightweight validation. Heavy work happens in execute."""
        try:
            input_raster = parameters[0]
            sensor_param = parameters[1]
            indices = parameters[2]
            composites = parameters[3]

            if not input_raster.valueAsText:
                return
            if not arcpy.Exists(input_raster.valueAsText):
                input_raster.setErrorMessage(
                    f"Input raster does not exist: {input_raster.valueAsText}"
                )
                return

            # If user picked Auto and we can't detect, surface that early.
            if sensor_param.valueAsText == SENSOR_AUTO:
                if detect_sensor(input_raster.valueAsText) is None:
                    sensor_param.setErrorMessage(
                        "Could not auto-detect sensor from input filename. "
                        "Please set Sensor Type explicitly."
                    )
                    return

            # Require at least one output: an index or a composite.
            sel_indices = indices.values or []
            sel_composites = composites.values or []
            if not sel_indices and not sel_composites:
                indices.setWarningMessage(
                    "Select at least one index or composite to compute."
                )
        except Exception:
            pass

    def execute(self, parameters, messages):
        try:
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required.")
                return None

            arcpy.env.overwriteOutput = True

            input_raster = parameters[0].valueAsText
            sensor_choice = parameters[1].valueAsText
            sel_indices = parameters[2].values or []
            sel_composites = parameters[3].values or []
            out_workspace = parameters[4].valueAsText
            out_prefix = (parameters[5].valueAsText or "").strip()
            rescale = bool(parameters[6].value)
            mask_feature = parameters[7].valueAsText

            sensor = resolve_sensor(sensor_choice, input_raster)
            arcpy.AddMessage(f"Sensor resolved to: {sensor}")

            mask_obj = mask_feature if mask_feature and arcpy.Exists(mask_feature) else None
            if mask_feature and not mask_obj:
                arcpy.AddWarning(
                    f"Mask {mask_feature!r} does not exist — proceeding unmasked."
                )

            bands = self._extract_bands(input_raster, sensor)

            n_indices_written = self._calculate_indices(
                bands, sel_indices, sensor, out_workspace, out_prefix, rescale, mask_obj,
            )
            n_composites_written = self._create_composites(
                bands, sel_composites, sensor, out_workspace, out_prefix, mask_obj,
            )

            arcpy.AddMessage(
                f"\nDone. {n_indices_written} index/indices + "
                f"{n_composites_written} composite(s) written to "
                f"{out_workspace}."
            )
            return None

        except Exception as e:
            arcpy.AddError(f"Tool 04 failed: {e}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
        finally:
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    # ----- helpers -----

    def _extract_bands(self, input_raster, sensor):
        """Extract bands 1..N into a {idx: Raster} dict for INDICES /
        COMPOSITES lookups.

        N is the minimum of the input raster's actual ``bandCount`` and
        the number of roles defined for ``sensor``. Truncating to the
        input's band count handles partial-spec products (e.g. the
        3-band VNIR-only ASTER mosaic from a post-Apr-2008 archive)
        without producing a flood of "Band X extraction failed"
        warnings — indices that reference missing bands raise KeyError
        in get_band() and are skipped gracefully by the caller.
        """
        role_mapping = SENSOR_BAND_ROLES[sensor]
        n_expected = len(role_mapping)
        try:
            n_available = int(arcpy.Raster(input_raster).bandCount)
        except Exception:
            n_available = n_expected  # fall back to sensor expectation
        n_to_extract = min(n_available, n_expected)

        arcpy.AddMessage(
            f"Extracting {n_to_extract} band(s) for {sensor} "
            f"(input has {n_available}, sensor defines {n_expected})..."
        )
        bands = {}
        for i in range(1, n_to_extract + 1):
            try:
                bands[i] = Float(ExtractBand(input_raster, [i]))
            except Exception as e:
                arcpy.AddWarning(f"  Band {i} extraction failed: {e}")

        if n_available < n_expected:
            missing_roles = sorted(
                r for r, idx in role_mapping.items() if idx > n_available
            )
            arcpy.AddMessage(
                f"  Input has fewer bands than {sensor} defines — "
                f"indices/composites requiring {missing_roles} will be "
                f"skipped automatically."
            )
        return bands

    def _calculate_indices(self, bands, selected_labels, sensor,
                           out_workspace, out_prefix, rescale, mask_obj):
        """Compute and save each selected index. Returns count written."""
        if not selected_labels:
            return 0

        from arcpy.sa import ExtractByMask, SetNull, Float

        # Self-mask the implicit NoData footprint of the input. Mosaic
        # rasters in a file geodatabase commonly carry no explicit
        # NoData metadata; the fill pixels outside the AOI sit at
        # value 0 (the U16 default fill written by Pro on save). Using
        # the first available band as a sentinel ("data wherever
        # band 1 > 0"), we drop those fill pixels from every index
        # output — eliminates the green/pink rectangles that
        # otherwise appear at the corners of the result extent.
        any_band_key = next(iter(bands)) if bands else None
        self_mask = bands[any_band_key] > 0 if any_band_key is not None else None

        written = 0
        for label in selected_labels:
            clean = str(label).strip("'\"")
            key = label_to_index_key(clean, sensor)
            if key is None:
                arcpy.AddWarning(f"Index {clean!r} not recognised for {sensor}; skipped.")
                continue
            meta = INDICES[key]
            try:
                arcpy.AddMessage(f"\nComputing index: {clean}")
                result = meta["compute"](bands, sensor)

                # Apply AOI mask if provided.
                if mask_obj is not None:
                    result = ExtractByMask(result, mask_obj)

                # Sanity bounds (same as the Landsat audit fix): arcpy.sa.Divide
                # already returns NoData on /0; Inf > 10000 is also caught.
                result_f = Float(result)
                result = SetNull(
                    arcpy.sa.BooleanOr(result_f > 10000, result_f < -10000),
                    result,
                )

                # Drop the input's implicit NoData footprint (pixels
                # where the source bands sit at 0 — outside-AOI fill).
                if self_mask is not None:
                    result = SetNull(~self_mask, result)

                if rescale:
                    result = self._rescale_to_0_255(result)

                name = f"{out_prefix}{meta['output_suffix']}" if out_prefix else meta["output_suffix"]
                out_path = self._build_out_path(out_workspace, name, "index")
                result.save(out_path)
                arcpy.AddMessage(f"  Saved: {out_path}")
                written += 1
            except KeyError as e:
                arcpy.AddWarning(f"Skipping {clean}: missing band ({e})")
            except Exception as e:
                arcpy.AddWarning(f"Error computing {clean}: {e}")
        return written

    def _create_composites(self, bands, selected_labels, sensor,
                           out_workspace, out_prefix, mask_obj):
        if not selected_labels:
            return 0

        from arcpy.sa import ExtractByMask, SetNull

        # Self-mask the implicit NoData footprint (see _calculate_indices
        # docstring for the same reasoning — composites suffer the same
        # corner-rectangle artefact from outside-AOI fill pixels).
        any_band_key = next(iter(bands)) if bands else None
        self_mask = bands[any_band_key] > 0 if any_band_key is not None else None

        written = 0
        for label in selected_labels:
            clean = str(label).strip("'\"")
            key = label_to_composite_key(clean, sensor)
            if key is None:
                arcpy.AddWarning(f"Composite {clean!r} not recognised for {sensor}; skipped.")
                continue
            meta = COMPOSITES[key]
            try:
                arcpy.AddMessage(f"\nCreating composite: {clean}")
                channels = []
                for role in meta["band_spec"]:
                    b = get_band(bands, role, sensor)
                    if mask_obj is not None:
                        b = ExtractByMask(b, mask_obj)
                    if self_mask is not None:
                        b = SetNull(~self_mask, b)
                    channels.append(b)

                name = f"{out_prefix}{meta['output_suffix']}" if out_prefix else meta["output_suffix"]
                out_path = self._build_out_path(out_workspace, name, "composite")
                arcpy.management.CompositeBands(channels, out_path)
                arcpy.AddMessage(f"  Saved: {out_path}")
                written += 1
            except KeyError as e:
                arcpy.AddWarning(f"Skipping {clean}: missing band ({e})")
            except Exception as e:
                arcpy.AddWarning(f"Error creating {clean}: {e}")
        return written

    def _build_out_path(self, out_workspace, name, kind):
        """Build the output path for an index or composite raster.

        Thin wrapper around ``_build_workspace_subfolder_path`` that
        chooses the subfolder name from this tool's two product
        families. The underlying helper handles the .gdb / folder
        dispatch and the .tif extension.
        """
        subfolder = "indices" if kind == "index" else "composites"
        return _build_workspace_subfolder_path(out_workspace, name, subfolder)

    def _rescale_to_0_255(self, raster):
        """Linear rescale a raster to [0, 255] using its own min/max."""
        try:
            min_val = float(arcpy.GetRasterProperties_management(raster, "MINIMUM").getOutput(0)
                            .replace(",", "."))
            max_val = float(arcpy.GetRasterProperties_management(raster, "MAXIMUM").getOutput(0)
                            .replace(",", "."))
            if max_val == min_val:
                arcpy.AddWarning("  Rescale skipped — constant raster.")
                return raster
            rng = max_val - min_val
            return Times(Divide(Minus(Float(raster), min_val), rng), 255)
        except Exception as e:
            arcpy.AddWarning(f"  Rescale failed: {e}")
            return raster

class LandsatMosaic(object):
    """Tool 02 — Landsat 8/9 C2L2 cloud-removed mosaic.

    Accepts either:
      a) A folder of EarthExplorer `.tar` archives (LC08_/LC09_*.tar) —
         the tool extracts each archive into a sibling folder before
         processing. Both L2SR and L2SP variants are accepted.
      b) A folder of already-extracted scene subfolders, each containing
         the per-band SR TIFFs and the _MTL.txt.

    Cloud removal uses QA_PIXEL bits 0-4 (fill, dilated cloud, cirrus,
    cloud, cloud shadow). Geometric median across the temporal stack
    builds the cloud-free composite. A `_provenance.csv` is written
    alongside the output documenting every scene that contributed.

    Audit fixes carried over: Bug 2 (BuildSeamlines blend_type LINEAR)
    and Bug 7 (_apply_mask returns None on failure so callers can
    detect that masking was skipped).
    """

    def __init__(self):
        self.label = "02 — Landsat 8/9 C2L2 Mosaic"
        self.description = (
            "Build a cloud-removed mosaic from Landsat 8/9 Collection 2 "
            "Level 2 (L2SR or L2SP) scenes. Accepts a folder of .tar "
            "archives (auto-extracted) or already-extracted scene folders. "
            "QA_PIXEL bits 0-4 are masked; the temporal stack is reduced "
            "to a geometric median. A provenance CSV is written alongside "
            "the output documenting every contributing scene."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        # Output Geodatabase
        gdb = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="gdb_path",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        gdb.filter.list = ["Local Database"]

        # Output Mosaic Name
        mosaic_name = arcpy.Parameter(
            displayName="Output Mosaic Name",
            name="mosaic_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )

        # Landsat Data Folder
        data_folder = arcpy.Parameter(
            displayName="Landsat Data Folder",
            name="data_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # Region
        region = arcpy.Parameter(
            displayName="Region",
            name="region",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        region.filter.list = ["Portugal Mainland", 
                      "Azores Western (Flores, Corvo)", 
                      "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)", 
                      "Azores Eastern (São Miguel, Santa Maria)", 
                      "Madeira", 
                      "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)", 
                      "Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)",
                      "Angola", 
                      "Mozambique"]

        # Time Filter Type
        time_type = arcpy.Parameter(
            displayName="Time Filter Type",
            name="time_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        time_type.filter.list = ["All Images", "Specific Year", "Month in Year", 
                                "Month All Years", "Season in Year", "Season All Years"]

        # Year (optional, enabled for year-specific options)
        year = arcpy.Parameter(
            displayName="Year",
            name="year",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )

        # Month (optional)
        month = arcpy.Parameter(
            displayName="Month",
            name="month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        month.filter.list = list(range(1, 13))

        # Season (optional)
        season = arcpy.Parameter(
            displayName="Season",
            name="season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )

        # Mask Feature
        mask = arcpy.Parameter(
            displayName="Mask Feature (Optional, polygon)",
            name="mask_feature",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input"
        )
        # Restrict to polygon layers/feature classes only. The
        # GPFeatureLayer datatype renders as a dropdown of polygon
        # layers currently in the map AND keeps the "Browse" button
        # for navigating to a feature class on disk — best of both.
        mask.filter.list = ["Polygon"]

        # Save Statistics
        save_stats = arcpy.Parameter(
            displayName="Save Processing Statistics",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        save_stats.value = True

        preserve_scratch = arcpy.Parameter(
            displayName="Preserve Scratch & Resume on Re-run",
            name="preserve_scratch",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        preserve_scratch.value = False

        params = [gdb, mosaic_name, data_folder, region, time_type,
                 year, month, season, mask, save_stats, preserve_scratch]
        return params
    
    def updateParameters(self, parameters):
        """Modify parameter values and properties.

        Performance: updateParameters fires on every keystroke / dialog
        change. The data-folder year scan previously walked the disk on
        every call — a 50k-scene archive could freeze the ArcGIS Pro UI.
        We now cache (folder_path → sorted_years) on `self` and only
        rescan when the folder path actually changes.
        """
        if not parameters[2].altered:  # If data folder not set
            return

        if parameters[2].value and parameters[2].altered:
            folder_path = parameters[2].valueAsText
            cache = getattr(self, "_years_cache", None)
            if cache and cache[0] == folder_path:
                years_sorted = cache[1]
            else:
                try:
                    years = set()
                    for root, _, files in os.walk(folder_path):
                        for fn in files:
                            # Accept either extracted MTL filenames OR raw
                            # EarthExplorer .tar archives. Both encode the
                            # acquisition date in the 4th underscore-token
                            # (LC08_L2SR_204032_20240215_..._MTL.txt or
                            # LC08_L2SR_204032_20240215_..._T1.tar).
                            fl = fn.lower()
                            is_mtl = fn.endswith('_MTL.txt')
                            is_landsat_tar = fl.endswith('.tar') and (
                                fl.startswith('lc08_') or fl.startswith('lc09_')
                            )
                            if not (is_mtl or is_landsat_tar):
                                continue
                            parts = fn.split('_')
                            if len(parts) >= 4:
                                try:
                                    years.add(int(parts[3][:4]))
                                except ValueError:
                                    continue
                    years_sorted = sorted(years)
                    self._years_cache = (folder_path, years_sorted)
                except Exception as e:
                    arcpy.AddWarning(f"Error scanning years: {str(e)}")
                    years_sorted = []
            if years_sorted:
                parameters[5].filter.list = years_sorted

        # Update season list based on region
        if parameters[3].value:  # If region is selected
            region = parameters[3].valueAsText
            if "Angola" in region:
                parameters[7].filter.list = ["Rainy", "Dry", "Rainy Peak", "Dry Peak"]
            elif "Mozambique" in region:
                parameters[7].filter.list = ["Rainy", "Dry", "Rainy Peak", "Dry Peak"]
            elif "Cape Verde" in region:
                parameters[7].filter.list = ["Dry", "Rainy", "Transition Dry-Wet", "Transition Wet-Dry"]    
            else:  # Temperate regions
                parameters[7].filter.list = ["Spring", "Summer", "Autumn", "Winter"]

        # Enable/disable time-based parameters
        if parameters[4].value:  # Time Filter Type
            time_type = parameters[4].valueAsText
            parameters[5].enabled = time_type in ["Specific Year", "Month in Year", "Season in Year"]
            parameters[6].enabled = time_type in ["Month in Year", "Month All Years"]
            parameters[7].enabled = time_type in ["Season in Year", "Season All Years"]

    def updateMessages(self, parameters):
            """Modify messages created by internal validation"""
            if parameters[0].altered:  # Geodatabase validation
                gdb_path = parameters[0].valueAsText
                arcpy.AddMessage(f"\nValidating geodatabase: {gdb_path}")
                
                try:
                    # Check if it's a geodatabase
                    if not gdb_path.endswith('.gdb'):
                        parameters[0].setErrorMessage("Output must be a File Geodatabase (.gdb)")
                        return

                    # Check if it exists
                    exists = arcpy.Exists(gdb_path)
                    arcpy.AddMessage(f"Geodatabase exists: {exists}")
                    if not exists:
                        parameters[0].setErrorMessage("Geodatabase does not exist")
                        return

                    # Verify it's a valid workspace
                    desc = arcpy.Describe(gdb_path)
                    arcpy.AddMessage(f"Workspace type: {desc.dataType}")
                    if desc.dataType != "Workspace":
                        parameters[0].setErrorMessage("Not a valid geodatabase workspace")
                        return

                    # Test write permissions
                    try:
                        test_name = "delete_me_test"
                        test_path = os.path.join(gdb_path, test_name)
                        arcpy.management.CreateFeatureclass(gdb_path, test_name, "POINT")
                        arcpy.management.Delete(test_path)
                        arcpy.AddMessage("Write permission test: Passed")
                    except Exception as e:
                        parameters[0].setErrorMessage(f"No write permissions: {str(e)}")
                        return

                except Exception as e:
                    parameters[0].setErrorMessage(f"Workspace validation error: {str(e)}")

            # Validate data folder. Accept either:
            #  (a) already-extracted scene folders with _MTL.txt files, OR
            #  (b) EarthExplorer .tar archives (LC08_*, LC09_*) which
            #      execute() will auto-extract before processing.
            if parameters[2].altered:
                folder_path = parameters[2].valueAsText
                if not os.path.exists(folder_path):
                    parameters[2].setErrorMessage("Data folder does not exist")
                    return

                found_scene = False
                for root, _, files in os.walk(folder_path):
                    for f in files:
                        if f.endswith('_MTL.txt'):
                            found_scene = True
                            break
                        fl = f.lower()
                        if (fl.endswith('.tar') or fl.endswith('.zip')) and (
                            fl.startswith('lc08_') or fl.startswith('lc09_')
                        ):
                            found_scene = True
                            break
                    if found_scene:
                        break

                if not found_scene:
                    parameters[2].setErrorMessage(
                        "No Landsat scenes found in folder. Expected either "
                        "EarthExplorer .tar / .zip archives (LC08_* / LC09_*) "
                        "read on the fly via GDAL, or already-extracted scene "
                        "folders containing *_MTL.txt files."
                    )
                    
    def remove_cloud(self, scenes, stats):
        """Validate scenes and pass band_paths through to the composite stage.

        Previously this method built lazy `Con(value_mask, band)` expressions
        per scene and stored them on the scene dict for `_create_geometric_median_mosaic`
        to materialise. That design had a hidden cost: each scene's
        `value_mask = ~TransposeBits(qa_raster, ...)` was referenced
        seven times (once per band Con), and arcpy's map-algebra evaluator
        doesn't cache the result — so when CompositeBands later
        evaluated the 7 lazy Cons it re-decoded the QA band from
        /vsitar/ seven times per scene. On the 173-scene Faial run
        that meant ~1200 redundant QA decodes.

        The fix moved into `_create_geometric_median_mosaic`: materialise
        the value_mask exactly once per scene as a scratch GeoTIFF, then
        either reference it from 7 lazy Cons (no-AOI case) or materialise
        each cleaned band individually at AOI extent (AOI-active case).

        This method now just validates band availability and forwards the
        VSI band paths intact. No /vsitar/ opens happen here — those are
        deferred to the composite phase where they happen exactly once
        per band per scene.
        """
        try:
            start_time = datetime.now()
            self._update_processing_stats(stats, stage="cloud_removal")
            clean_scenes = []
            total_scenes = len(scenes)

            arcpy.AddMessage(f"\n▶ Phase 2 — Validating {total_scenes} scenes")

            required = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "QA_PIXEL"]
            arcpy.SetProgressor(
                "step", "Validating scenes", 0, total_scenes, 1
            )

            for idx, scene in enumerate(scenes, 1):
                # Validation is fast (~10ms); check cancel every 25.
                if idx % 25 == 0 and arcpy.env.isCancelled:
                    arcpy.ResetProgressor()
                    arcpy.AddWarning(
                        f"  ✗ Cancelled by user after {idx-1}/{total_scenes} scenes."
                    )
                    return None
                try:
                    scene_path = scene.get('path', '')
                    scene_id = scene.get('scene_id') or os.path.basename(
                        scene_path.rstrip(os.sep)
                    )
                    band_paths = scene.get('band_paths') or {}

                    arcpy.SetProgressorLabel(f"[{idx}/{total_scenes}] {scene_id}")

                    missing = [k for k in required if k not in band_paths]
                    if missing:
                        arcpy.AddWarning(
                            f"  ✗ {scene_id}: missing {missing}"
                        )
                        stats['failed_scenes'] = stats.get('failed_scenes', 0) + 1
                        continue

                    clean_scenes.append({
                        'path': scene_path,
                        'scene_id': scene_id,
                        'band_paths': band_paths,
                        'metadata': scene.get('metadata', {}),
                        'is_archive': scene.get('is_archive', False),
                    })
                    arcpy.SetProgressorPosition(idx)

                except Exception as e:
                    arcpy.AddWarning(f"  ✗ {scene.get('scene_id', '?')}: {str(e)}")
                    stats['failed_scenes'] = stats.get('failed_scenes', 0) + 1
                    stats['errors'].append(str(e))
                    continue

            arcpy.ResetProgressor()
            arcpy.AddMessage(
                f"  ✓ {len(clean_scenes)}/{total_scenes} ready "
                f"({(datetime.now() - start_time).total_seconds():.1f}s)"
            )

            stats['cloud_removal'] = {
                'scenes_processed': total_scenes,
                'scenes_cleaned': len(clean_scenes),
                'processing_time': (datetime.now() - start_time).total_seconds(),
            }

            if not clean_scenes:
                arcpy.AddWarning("No scenes were successfully validated")
                return None
            return clean_scenes

        except Exception as e:
            arcpy.AddError(f"Scene validation failed: {str(e)}")
            return None

    @staticmethod
    def _composite_path_for(scene, scratch_dir):
        """Resolve the per-scene composite stack path in scratch.
        Static so the subprocess worker can use it without needing the
        full Phase 3 context."""
        sid = scene.get('scene_id') or os.path.basename(
            (scene.get('path') or '').rstrip(os.sep)
        ) or "scene"
        return os.path.join(
            scratch_dir,
            f"{_sanitize_arcpy_name(sid)}_composite.tif",
        )

    def _process_scene(self, scene, scratch_dir):
        """Build one per-scene composite from a Landsat C2L2 scene.

        Extracted from the inline body of _create_geometric_median_mosaic
        (2026-05-24) so the subprocess-per-batch worker can call this
        directly on a fresh python.exe per batch. The legacy single-
        process loop in _create_geometric_median_mosaic also calls
        this method; behaviour is unchanged.

        Returns the composite path on success. Raises on failure (the
        caller decides whether to AddWarning + skip or propagate).
        """
        band_paths = scene['band_paths']
        scene_id = scene.get('scene_id') or os.path.basename(
            (scene.get('path') or '').rstrip(os.sep)
        ) or "scene"

        # Per-scene subfolder so vmask.tif + Bn.tif don't collide
        # between scenes. _cleanup_scratch_folder() handles the lot.
        scene_scratch = os.path.join(
            scratch_dir, _sanitize_arcpy_name(scene_id),
        )
        os.makedirs(scene_scratch, exist_ok=True)

        # 1. value_mask materialised ONCE per scene -> kills the
        #    7x QA decode redundancy of the previous lazy chain.
        #    Honours env.mask + env.extent when AOI is active.
        qa_raster = arcpy.Raster(band_paths["QA_PIXEL"])
        cloud_mask = TransposeBits(
            qa_raster, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 0, None
        )
        value_mask = ~cloud_mask
        value_mask_path = os.path.join(scene_scratch, "vmask.tif")
        value_mask.save(value_mask_path)
        value_mask_raster = arcpy.Raster(value_mask_path)

        # Lazy Con chain. The per-band .save() materialisation tried
        # in commit 4f4728b was reverted because it produced visible
        # dark blobs over high-cloud areas (NoData semantics on
        # cloudy pixels were lost when .save() ran with env.mask
        # active; cloudy ended up as zero, not NoData, and
        # GeometricMedian pulled the median toward zero).
        composite_inputs = [
            Con(value_mask_raster, arcpy.Raster(band_paths[f"B{n}"]))
            for n in range(1, 8)
        ]

        temp_composite = self._composite_path_for(scene, scratch_dir)
        arcpy.management.CompositeBands(composite_inputs, temp_composite)

        # Resume sentinel: only written after CompositeBands returns,
        # so its presence guarantees the composite is complete.
        try:
            with open(temp_composite + ".complete", "w", encoding="utf-8") as fh:
                fh.write(datetime.now().isoformat(timespec="seconds") + "\n")
        except OSError:
            pass

        # Per-scene scratch cleanup: rmtree the {scene_id}/ subfolder
        # that held vmask.tif. Bounds scratch growth so NTFS directory
        # ops + GDAL handle caches stay cheap on long runs.
        sid_safe = _sanitize_arcpy_name(scene_id)
        _cleanup_per_scene_intermediates(
            scratch_dir, sid_safe,
            keep_basenames=(
                f"{sid_safe}_composite.tif",
                f"{sid_safe}_composite.tif.complete",
            ),
        )
        return temp_composite

    def _create_geometric_median_mosaic(self, clean_scenes, gdb_path, mosaic_name,
                                         preserve_scratch=False,
                                         mask_feature=None):
        """Create geometric median mosaic preserving multi-band structure.

        Cleanup fix: previously the temp composites (one per scene) were
        deleted ONLY on success. Any exception during GeometricMedian or
        .save() left orphan temp rasters in the production geodatabase;
        re-runs accumulated more. We now build the list under a
        try/finally so cleanup ALWAYS runs.

        Instrumentation: this phase used to run silently for tens of
        minutes — each `CompositeBands` call forces the full lazy
        pipeline (tar read via VSI → QA decode → TransposeBits → Con
        mask), and `GeometricMedian` then iterates up to 20 passes
        across all scenes without any internal progress hook. We log
        per-scene composite progress and announce the median phase so
        the user knows the tool is making progress, not hung.
        """
        multiband_rasters = []
        output_path = None
        scratch_dir = None
        try:
            start_time = datetime.now()
            total = sum(
                1 for s in clean_scenes
                if 'band_paths' in s and all(
                    f"B{n}" in s['band_paths'] for n in range(1, 8)
                ) and 'QA_PIXEL' in s['band_paths']
            )

            scratch_dir = _make_mosaic_scratch_dir(
                gdb_path, "_genesis_landsat_composites", mosaic_name,
            )

            aoi_active = bool(arcpy.env.mask)
            arcpy.AddMessage(f"\n▶ Phase 3 — Per-scene composites ({total} scenes)")
            arcpy.AddMessage(f"  Scratch: {scratch_dir}")
            arcpy.AddMessage(
                "  Mode: lazy Cons → CompositeBands (full-scene); value_mask "
                "materialised once per scene to avoid 7× QA decode"
            )

            # Resume scan — any scene whose composite + .complete marker
            # both survive in scratch is reused as-is. A composite file
            # without a marker is partial (Phase 3 died mid-CompositeBands)
            # and gets rebuilt below.
            eligible_scenes = [
                s for s in clean_scenes
                if (s.get('band_paths') and
                    all(f"B{n}" in s['band_paths'] for n in range(1, 8)) and
                    'QA_PIXEL' in s['band_paths'])
            ]
            resumed_count = 0
            to_process = []
            for scene in eligible_scenes:
                comp_path = self._composite_path_for(scene, scratch_dir)
                if os.path.exists(comp_path) and os.path.exists(comp_path + ".complete"):
                    multiband_rasters.append(comp_path)
                    resumed_count += 1
                else:
                    to_process.append(scene)
            # Schema defence: refuse to resume incompatible scratch
            # (e.g., from a code version with a different band layout).
            if not _check_scratch_schema(multiband_rasters, 7, "Landsat"):
                return None
            if resumed_count:
                arcpy.AddMessage(
                    f"  ✓ Resume: {resumed_count}/{len(eligible_scenes)} "
                    f"composite(s) reused from previous run"
                )

            if to_process:
                arcpy.SetProgressor(
                    "step", "Per-scene processing",
                    0, max(1, len(to_process)), 1,
                )
                t0_phase = time.time()
                scene_times = []
                failures = []
                composite_idx = 0
                for scene in to_process:
                    if arcpy.env.isCancelled:
                        arcpy.ResetProgressor()
                        arcpy.AddWarning(
                            f"  ✗ Cancelled by user after {composite_idx}/{len(to_process)} composites."
                        )
                        return None
                    composite_idx += 1
                    scene_id = scene.get('scene_id') or os.path.basename(
                        (scene.get('path') or '').rstrip(os.sep)
                    ) or f"scene_{composite_idx}"
                    arcpy.SetProgressorLabel(
                        f"[{composite_idx}/{len(to_process)}] {scene_id}"
                    )
                    scene_start = time.time()
                    try:
                        temp_composite = self._process_scene(scene, scratch_dir)
                        elapsed = time.time() - scene_start
                        arcpy.AddMessage(_format_scene_log_line(
                            composite_idx, len(to_process),
                            scene_id, elapsed,
                        ))
                        multiband_rasters.append(temp_composite)
                        scene_times.append(elapsed)
                    except Exception as e:
                        elapsed = time.time() - scene_start
                        fail_msg = f"{type(e).__name__}: {e}"
                        arcpy.AddMessage(_format_scene_log_line(
                            composite_idx, len(to_process),
                            scene_id, elapsed, fail=fail_msg,
                        ))
                        failures.append((composite_idx, scene_id, fail_msg))
                    arcpy.SetProgressorPosition(composite_idx)
                    _periodic_arcpy_cache_flush(composite_idx)
                arcpy.ResetProgressor()
                _emit_phase3_summary(
                    len(scene_times), len(failures),
                    time.time() - t0_phase, failures,
                )

            # Phase 4 — GeometricMedian
            output_path = os.path.join(gdb_path, f"{mosaic_name}_Geomedian")
            with phase(
                f"Phase 4 — GeometricMedian over {len(multiband_rasters)} stacks",
                quiet_close=True,
                # Outer except in _create_geometric_median_mosaic logs the
                # canonical failure message — suppress phase's own warning
                # to avoid the double yellow+red icons in the GP dialog.
                silent_error=True,
            ) as ph:
                arcpy.AddMessage(
                    "  Silent phase (arcpy.ia.GeometricMedian has no per-iteration hook)."
                )
                arcpy.SetProgressor("default", "Computing GeometricMedian...")
                geomedian = arcpy.ia.GeometricMedian(
                    multiband_rasters,
                    epsilon=_GEOMETRIC_MEDIAN_EPSILON,
                    max_iteration=_GEOMETRIC_MEDIAN_MAX_ITER,
                    extent_type="UnionOf",
                    cellsize_type="FirstOf",
                )
                geomedian.save(output_path)
                arcpy.ResetProgressor()
            arcpy.AddMessage(
                f"  ✓ GeometricMedian complete in {ph.elapsed:.1f}s "
                f"→ {os.path.basename(output_path)}"
            )

            _sanity_check_output(
                output_path, sensor_hint="landsat",
                label=os.path.basename(output_path),
            )

            return output_path

        except Exception as e:
            arcpy.AddError(f"Error creating geometric median: {str(e)}")
            return None

        finally:
            if preserve_scratch and scratch_dir:
                arcpy.AddMessage(
                    f"  Scratch preserved at: {scratch_dir}\n"
                    f"  Re-run with the same Output Mosaic Name to resume "
                    f"from completed composites."
                )
            else:
                if multiband_rasters:
                    arcpy.AddMessage(
                        f"  Cleaning up scratch folder "
                        f"({len(multiband_rasters)} composite(s))..."
                    )
                if 'scratch_dir' in locals():
                    _cleanup_scratch_folder(scratch_dir)


    def execute(self, parameters, messages):
        try:
            # Check out necessary extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            if arcpy.CheckExtension("ImageAnalyst") == "Available":
                arcpy.CheckOutExtension("ImageAnalyst")

            # Enable overwrite
            arcpy.env.overwriteOutput = True

            # Take control of cancellation so we can clean up scratch +
            # extensions instead of Pro hard-killing the process. Cancel
            # checks at the loop boundaries below return None with a
            # warning when the user clicks the Cancel button in the GP
            # dialog.
            arcpy.env.autoCancelling = False

            # Get parameters
            gdb_path = parameters[0].valueAsText
            mosaic_name = parameters[1].valueAsText
            data_folder = parameters[2].valueAsText
            region = parameters[3].valueAsText
            time_type = parameters[4].valueAsText
            year = parameters[5].value
            month = parameters[6].value
            season = parameters[7].valueAsText
            mask_feature = parameters[8].valueAsText
            save_stats = parameters[9].value
            preserve_scratch = bool(parameters[10].value)

            # ----------------------------------------------------------------
            # AOI-first scoping. Set arcpy.env.mask + arcpy.env.extent BEFORE
            # any raster work begins so every downstream operation (cloud
            # mask via TransposeBits + Con, per-scene CompositeBands, the
            # 20-iteration GeometricMedian, the inter-zone MosaicToNewRaster)
            # is restricted to AOI pixels only.
            #
            # For Faial-sized AOIs vs. a full Landsat scene footprint this
            # is roughly a 190x compute reduction on every per-pixel
            # operation. The trailing _apply_mask call is left in place as
            # defence-in-depth — it becomes a no-op when the env scope has
            # already constrained the mosaic to the AOI.
            #
            # CRS mismatch (AOI in WGS84, scenes in UTM, etc.) is handled
            # automatically by ArcGIS, paying a one-time projection cost
            # the first time each scene's CRS is touched.
            # ----------------------------------------------------------------
            # Initialize statistics
            stats = {
                'start_time': datetime.now(),
                'total_scenes': 0,
                'processed_scenes': 0,
                'failed_scenes': 0,
                'cloud_coverage': [],
                'processing_time': [],
                'errors': []
            }
            stats['cloud_removal'] = {
                'scenes_processed': 0, 'scenes_cleaned': 0, 'processing_time': 0
            }
            stats['geometric_median'] = {
                'batches_processed': 0, 'total_batches': 0, 'processing_time': 0
            }

            # Header — one block of run context, no more "Initializing
            # processing:" prefix on every line.
            arcpy.AddMessage("=" * 60)
            arcpy.AddMessage(f"LANDSAT 8/9 MOSAIC — {region}")
            arcpy.AddMessage("=" * 60)
            arcpy.AddMessage(f"  Output:     {gdb_path}\\{mosaic_name}")
            arcpy.AddMessage(f"  Source:     {data_folder}")

            if mask_feature and arcpy.Exists(mask_feature):
                arcpy.env.mask = mask_feature
                arcpy.env.extent = mask_feature
                arcpy.AddMessage(f"  AOI:        {mask_feature} (env.mask + env.extent active)")
            elif mask_feature:
                arcpy.AddWarning(
                    f"  AOI:        {mask_feature!r} NOT FOUND — running over full scene footprint"
                )

            # Create temporal filter and get region info
            temporal_filter = self._create_temporal_filter(time_type, year, month, season)
            region_info = self._get_region_info(region)
            arcpy.AddMessage(f"  UTM zones:  {region_info['utm_zones']}")

            # Track scenes that actually fed the final mosaic for provenance.
            all_scenes_used = []

            # Process each UTM zone
            final_mosaics = []
            for utm_zone in region_info['utm_zones']:
                if arcpy.env.isCancelled:
                    arcpy.AddWarning("\n✗ Cancelled by user between UTM zones.")
                    return None
                try:
                    arcpy.AddMessage(
                        f"\n▶ Phase 1 — UTM zone {utm_zone}{region_info['hemisphere']}: scene discovery"
                    )

                    scenes = self._find_scenes(
                        data_folder=data_folder,
                        utm_zone=utm_zone,
                        temporal_filter=temporal_filter,
                        seasonal_pattern=region_info['seasonal_pattern'],
                        stats=stats
                    )

                    if not scenes:
                        arcpy.AddWarning(f"  ✗ No scenes for UTM {utm_zone}")
                        continue

                    clean_scenes = self.remove_cloud(scenes, stats)

                    if not clean_scenes:
                        arcpy.AddWarning(
                            f"  ✗ No valid scenes after validation for UTM {utm_zone}"
                        )
                        continue

                    zone_mosaic = self._create_geometric_median_mosaic(
                        clean_scenes,
                        gdb_path,
                        f"{mosaic_name}_UTM{utm_zone}{region_info['hemisphere']}",
                        preserve_scratch=preserve_scratch,
                        mask_feature=mask_feature,
                    )

                    if zone_mosaic:
                        final_mosaics.append(zone_mosaic)
                        all_scenes_used.extend(clean_scenes)

                except Exception as e:
                    arcpy.AddWarning(f"  ✗ UTM {utm_zone}: {str(e)}")
                    stats['errors'].append(f"Zone {utm_zone} error: {str(e)}")
                    continue

            # Check if any mosaics were created
            if not final_mosaics:
                arcpy.AddError("No valid mosaics were created")
                return None

            # Track intermediates created during this run so we can delete
            # them after the final output is established. The user wants
            # ONE raster in the output GDB, not a trail of per-zone /
            # pre-mask intermediates.
            intermediates_to_delete = []

            with phase(
                "Phase 5 — Merge / mask / cleanup",
                quiet_close=True,
                silent_error=True,
            ) as ph5:
                # Merge zones if needed
                if len(final_mosaics) > 1:
                    merged = self._merge_zone_mosaics(
                        gdb_path, mosaic_name, final_mosaics, region_info
                    )
                    if merged:
                        intermediates_to_delete.extend(final_mosaics)
                        final_mosaic = merged
                        arcpy.AddMessage(f"  ✓ Merged {len(final_mosaics)} zones → {os.path.basename(merged)}")
                    else:
                        final_mosaic = final_mosaics[0]
                        arcpy.AddWarning("  ✗ Merge failed; keeping single-zone output")
                else:
                    final_mosaic = final_mosaics[0]

                # Apply mask if specified
                if final_mosaic and mask_feature:
                    masked = self._apply_mask(
                        final_mosaic, mask_feature, gdb_path, mosaic_name
                    )
                    if masked and masked != final_mosaic:
                        intermediates_to_delete.append(final_mosaic)
                        final_mosaic = masked
                        arcpy.AddMessage(f"  ✓ AOI mask applied → {os.path.basename(masked)}")

                # Clean up superseded intermediates so the GDB ends with
                # one raster, not a trail.
                cleaned_count = 0
                for path in intermediates_to_delete:
                    if path and path != final_mosaic:
                        try:
                            if arcpy.Exists(path):
                                arcpy.management.Delete(path)
                                cleaned_count += 1
                        except arcpy.ExecuteError as e:
                            arcpy.AddWarning(f"  Could not delete {os.path.basename(path)}: {e}")
                if cleaned_count:
                    arcpy.AddMessage(f"  ✓ Cleaned up {cleaned_count} intermediate(s)")

            # Save statistics
            if save_stats:
                stats['processed_scenes'] = len(all_scenes_used)
                stats['failed_scenes'] = stats.get('failed_scenes', 0)
                stats['end_time'] = datetime.now()
                stats['total_duration'] = stats['end_time'] - stats['start_time']
                self._save_statistics(gdb_path, mosaic_name, stats)
                self._save_enhanced_statistics(gdb_path, mosaic_name, stats)

            if final_mosaic:
                try:
                    self._write_provenance_csv(final_mosaic, all_scenes_used, stats)
                except Exception as e:
                    arcpy.AddWarning(f"  Provenance CSV write failed (non-fatal): {e}")
                _write_band_sidecar_csv(final_mosaic, "landsat")

                total_elapsed = (datetime.now() - stats['start_time']).total_seconds()
                mins, secs = divmod(int(total_elapsed), 60)
                hrs, mins = divmod(mins, 60)
                time_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"

                arcpy.AddMessage("\n" + "=" * 60)
                arcpy.AddMessage(f"DONE — {os.path.basename(final_mosaic)}")
                arcpy.AddMessage(f"Total: {time_str}  |  Scenes: {len(all_scenes_used)}")
                arcpy.AddMessage("=" * 60)
                return final_mosaic
            else:
                arcpy.AddError("Failed to create final mosaic")
                return None

        except Exception as e:
            arcpy.AddError(f"Critical error in processing: {str(e)}")
            raise
        
        finally:
            # Check in extensions
            try:
                for ext in ["Spatial", "ImageAnalyst"]:
                    if arcpy.CheckExtension(ext) == "Available":
                        arcpy.CheckInExtension(ext)
            except:
                pass
        
    def _create_temporal_filter(self, time_type, year, month, season):
        """Create temporal filter dictionary"""
        filter_dict = {'type': time_type.lower().replace(' ', '_')}
        
        if year:
            filter_dict['year'] = year
        if month:
            filter_dict['month'] = month
        if season:
            filter_dict['season'] = season
            
        arcpy.AddMessage(f"Temporal filter: {filter_dict}")
        return filter_dict
        
    def _get_region_info(self, region):
        """Get region UTM zones and other information"""
        region_info = {
            'Portugal Mainland': {
                'utm_zones': [29],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Western (Flores, Corvo)': {
                'utm_zones': [25],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Eastern (São Miguel, Santa Maria)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Madeira': {
                'utm_zones': [28],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Cape Verde Western (Santo Antão, São Vicente, São Nicolau)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'cape_verde'
            },
            'Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)': {
                'utm_zones': [27],
                'hemisphere': 'N',
                'seasonal_pattern': 'cape_verde'
            },
            'Angola': {
                'utm_zones': [32, 33, 34],
                'hemisphere': 'S',
                'seasonal_pattern': 'angola'
            },
            'Mozambique': {
                'utm_zones': [36, 37],
                'hemisphere': 'S',
                'seasonal_pattern': 'mozambique'
            }
        }
        
        if region not in region_info:
            raise ValueError(f"Unknown region: {region}")
            
        return region_info[region]
        
    def _create_zone_mosaic(self, gdb_path, name, utm_zone, hemisphere):
        """Create mosaic dataset for specific UTM zone"""
        try:
            mosaic_path = os.path.join(gdb_path, name)
            epsg = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone
            
            arcpy.AddMessage(f"Creating mosaic dataset: {name}")
            arcpy.AddMessage(f"EPSG: {epsg}")
            
            if arcpy.Exists(mosaic_path):
                arcpy.AddMessage("Removing existing mosaic dataset")
                arcpy.Delete_management(mosaic_path)
                
            # Create base mosaic dataset
            arcpy.management.CreateMosaicDataset(
                in_workspace=gdb_path,
                in_mosaicdataset_name=name,
                coordinate_system=epsg,
                num_bands=7,
                pixel_type="16_BIT_UNSIGNED",
                product_definition="NONE"
            )
            
            # Add time field
            arcpy.AddMessage("Adding time field...")
            arcpy.management.AddField(
                in_table=mosaic_path,
                field_name="acquisitionDate",
                field_type="DATE",
                field_alias="Acquisition Date"
            )
            
            # Configure mosaic properties with proper time settings
            arcpy.AddMessage("Configuring mosaic properties...")
            arcpy.management.SetMosaicDatasetProperties(
                in_mosaic_dataset=mosaic_path,
                rows_maximum_imagesize=15000,
                columns_maximum_imagesize=15000,
                allowed_compressions="NONE",
                default_compression_type="NONE",
                resampling_type="BILINEAR",
                clip_to_footprints="CLIP",
                footprints_may_contain_nodata="FOOTPRINTS_MAY_CONTAIN_NODATA",
                clip_to_boundary="CLIP",
                color_correction="NOT_APPLY",
                allowed_mensuration_capabilities="BASIC",
                default_mensuration_capabilities="BASIC",
                allowed_mosaic_methods="Center;NorthWest;Nadir;LockRaster;ByAttribute;Seamline;None",
                default_mosaic_method="ByAttribute",
                order_field="acquisitionDate",
                order_base="1/1/1900 12:00:00 AM",
                sorting_order="Ascending",
                mosaic_operator="FIRST",
                blend_width=10,
                view_point_x=0,
                view_point_y=0,
                max_num_per_mosaic=50,
                cell_size_tolerance=0.8,
                cell_size=30,
                metadata_level="BASIC",
                transmission_fields="acquisitionDate",
                use_time="ENABLED"
            )
            
            
            arcpy.AddMessage(f"Mosaic dataset created successfully: {name}")
            return mosaic_path
            
        except Exception as e:
            arcpy.AddError(f"Error creating zone mosaic: {str(e)}")
            return None
        
          
    def _add_scenes_to_mosaic(self, mosaic_path, scenes, stats):
        """Add scenes to mosaic dataset"""
        try:
            arcpy.AddMessage("\nAdding scenes to mosaic...")
            
            if not scenes:
                arcpy.AddError("No scenes to add to mosaic")
                return False
            
            # Group scenes by satellite type
            scenes_by_type = {}
            for scene in scenes:
                scene_type = 'Landsat 8' if 'LC08' in scene['path'] else 'Landsat 9'
                if scene_type not in scenes_by_type:
                    scenes_by_type[scene_type] = []
                scenes_by_type[scene_type].append(scene)
            
            # Add scenes by satellite type
            for sat_type, type_scenes in scenes_by_type.items():
                try:
                    arcpy.AddMessage(f"\nProcessing {sat_type} scenes")
                    
                    # Add each scene folder
                    for scene in type_scenes:
                        arcpy.AddMessage(f"Adding scene: {scene['path']}")
                        
                        arcpy.management.AddRastersToMosaicDataset(
                            in_mosaic_dataset=mosaic_path,
                            raster_type=sat_type,
                            input_path=scene['path'],
                            aux_inputs="ProcessingTemplate Multiband"
                        )
                    
                    arcpy.AddMessage(f"Successfully added {len(type_scenes)} {sat_type} scenes")
                    
                    stats['processed_scenes'] += len(type_scenes)
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error processing {sat_type} scenes: {str(e)}")
                    stats['failed_scenes'] += len(type_scenes)
                    stats['errors'].append(str(e))
            
            # Check number of rasters in mosaic dataset
            raster_count = int(arcpy.management.GetCount(mosaic_path).getOutput(0))
            arcpy.AddMessage(f"Number of rasters in mosaic dataset: {raster_count}")
            
            if raster_count == 0:
                arcpy.AddError("No rasters were successfully added to the mosaic dataset")
                return False
            
            # Build seamlines
            arcpy.AddMessage("Building seamlines...")
            arcpy.management.BuildSeamlines(
                in_mosaic_dataset=mosaic_path,
                sort_order="ASCENDING",
                computation_method="GEOMETRY",
                blend_width=10,
                blend_type="BOTH"
            )
            
            return True
            
        except Exception as e:
            arcpy.AddError(f"Error adding scenes to mosaic: {str(e)}")
            return False
            
    def _merge_zone_mosaics(self, gdb_path, mosaic_name, zone_mosaics, region_info):
        """Merge multiple UTM zone mosaics.

        Fix: previously the merged dataset was forced into WGS 84 (degrees)
        while BuildSeamlines was called with cell_size=30. cell_size=30
        degrees is global-scale — meaningless for Landsat. We now inherit
        the CRS from the first zone mosaic (projected, metres) so the
        30 m cell_size is in the right units. AddRastersToMosaicDataset
        reprojects subsequent zones into that CRS on the fly.
        """
        try:
            if len(zone_mosaics) == 1:
                return zone_mosaics[0]

            arcpy.AddMessage("\nMerging zone mosaics...")
            merged_name = f"{mosaic_name}_Merged"
            merged_path = os.path.join(gdb_path, merged_name)

            # Use the first zone mosaic's spatial reference (projected, in metres).
            try:
                first_sr = arcpy.Describe(zone_mosaics[0]).spatialReference
            except Exception:
                first_sr = arcpy.SpatialReference(4326)  # last-resort fallback
                arcpy.AddWarning(
                    "Could not read first zone mosaic CRS; falling back to "
                    "WGS 84. BuildSeamlines cell_size will be approximate."
                )

            arcpy.management.CreateMosaicDataset(
                in_workspace=gdb_path,
                in_mosaicdataset_name=merged_name,
                coordinate_system=first_sr,
                num_bands=7,
                pixel_type="16_BIT_UNSIGNED",
            )
            
            # Add each zone mosaic
            for zone_mosaic in zone_mosaics:
                arcpy.AddMessage(f"Adding {os.path.basename(zone_mosaic)}")
                arcpy.management.AddRastersToMosaicDataset(
                    in_mosaic_dataset=merged_path,
                    raster_type="Raster Dataset",
                    input_path=zone_mosaic
                )
                
            # Build seamlines for merged result
            arcpy.management.BuildSeamlines(
                in_mosaic_dataset=merged_path,
                cell_size=30,
                sort_order="Closest_To_Center",
                computation_method="GEOMETRY",
                blend_width=10,
                blend_type="LINEAR"
            )
            
            return merged_path
            
        except Exception as e:
            arcpy.AddError(f"Error merging zone mosaics: {str(e)}")
            return None
        
    def _apply_mask(self, mosaic_path, mask_feature, gdb_path, mosaic_name):
        """Apply mask to mosaic dataset.

        Returns:
            str: path to the masked mosaic on success.
            str: the unchanged input path when no mask was requested
                 (legitimate "skip" — caller asked for no masking).
            None: on failure (mask feature missing, ExtractByMask raised,
                 save failed). Caller MUST treat None as failure rather
                 than silently using the unmasked mosaic. Previously this
                 method returned the unmasked path on failure, hiding the
                 problem from the caller (bug 7 from the 2026 audit).
        """
        try:
            if not mask_feature:
                arcpy.AddMessage("No mask feature provided. Skipping masking.")
                return mosaic_path

            arcpy.AddMessage("\nApplying mask...")
            masked_name = f"{mosaic_name}_Masked"
            masked_path = os.path.join(gdb_path, masked_name)

            if not arcpy.Exists(mask_feature):
                arcpy.AddError(
                    f"Mask feature {mask_feature} does not exist. "
                    f"Masking step failed; the unmasked mosaic remains at "
                    f"{mosaic_path}."
                )
                return None

            from arcpy.sa import ExtractByMask
            extracted = ExtractByMask(mosaic_path, mask_feature)
            extracted.save(masked_path)

            arcpy.AddMessage(f"Masked mosaic saved as: {masked_path}")
            return masked_path

        except Exception as e:
            arcpy.AddError(
                f"Error applying mask: {str(e)}. The unmasked mosaic "
                f"remains at {mosaic_path}."
            )
            return None
            
    def _save_statistics(self, gdb_path, mosaic_name, stats):
        """Save processing statistics to file"""
        try:
            # Create statistics folder if it doesn't exist
            stats_folder = os.path.join(os.path.dirname(gdb_path), "statistics")
            if not os.path.exists(stats_folder):
                os.makedirs(stats_folder)
                
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(stats_folder, f"{mosaic_name}_stats_{timestamp}.txt")
            
            with open(stats_file, 'w') as f:
                f.write("Landsat Mosaic Processing Statistics\n")
                f.write("===================================\n\n")
                
                f.write(f"Processing Start: {stats['start_time']}\n")
                f.write(f"Processing End: {stats['end_time']}\n")
                f.write(f"Total Duration: {stats['total_duration']}\n\n")
                
                f.write("Scene Statistics:\n")
                f.write(f"Total Scenes Found: {stats['total_scenes']}\n")
                f.write(f"Successfully Processed: {stats['processed_scenes']}\n")
                f.write(f"Failed Scenes: {stats['failed_scenes']}\n\n")
                
                if stats['cloud_coverage']:
                    avg_cloud = sum(stats['cloud_coverage']) / len(stats['cloud_coverage'])
                    f.write(f"Average Cloud Coverage: {avg_cloud:.2f}%\n\n")
                    
                if stats['processing_time']:
                    avg_time = sum(stats['processing_time']) / len(stats['processing_time'])
                    f.write(f"Average Processing Time per Scene: {avg_time:.2f} seconds\n\n")
                    
                if stats['errors']:
                    f.write("Errors Encountered:\n")
                    for error in stats['errors']:
                        f.write(f"- {error}\n")
                        
            arcpy.AddMessage(f"\nStatistics saved to: {stats_file}")
            return stats_file
            
        except Exception as e:
            arcpy.AddError(f"Error saving statistics: {str(e)}")
            return None
        
    def _update_processing_stats(self, stats, stage="general"):
        """Enhanced statistics tracking"""
        try:
            # Add new statistics categories if not present
            if 'cloud_removal' not in stats:
                stats['cloud_removal'] = {
                    'scenes_processed': 0,
                    'scenes_failed': 0,
                    'average_cloud_coverage_before': 0,
                    'average_cloud_coverage_after': 0,
                    'processing_time': 0
                }
                
            if 'geometric_median' not in stats:
                stats['geometric_median'] = {
                    'batches_processed': 0,
                    'total_batches': 0,
                    'memory_usage': [],
                    'processing_time': 0
                }
                
            if 'memory_tracking' not in stats:
                stats['memory_tracking'] = {
                    'peak_memory': 0,
                    'average_memory': 0,
                    'timestamps': []
                }
                
            # Update memory tracking
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            stats['memory_tracking']['timestamps'].append({
                'time': datetime.now(),
                'memory': current_memory,
                'stage': stage
            })
            stats['memory_tracking']['peak_memory'] = max(
                stats['memory_tracking']['peak_memory'],
                current_memory
            )
            
            return stats
            
        except Exception as e:
            arcpy.AddWarning(f"Error updating statistics: {str(e)}")
            return stats

    def _save_enhanced_statistics(self, gdb_path, mosaic_name, stats):
        """Save enhanced processing statistics"""
        try:
            stats_folder = os.path.join(os.path.dirname(gdb_path), "statistics")
            if not os.path.exists(stats_folder):
                os.makedirs(stats_folder)
                    
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(stats_folder, f"{mosaic_name}_detailed_stats_{timestamp}.txt")
            
            with open(stats_file, 'w') as f:
                f.write("Landsat Processing Detailed Statistics\n")
                f.write("====================================\n\n")
                
                # Basic processing info
                f.write("Processing Duration:\n")
                f.write(f"Start: {stats.get('start_time', 'N/A')}\n")
                f.write(f"End: {stats.get('end_time', datetime.now())}\n")
                f.write(f"Total Duration: {stats.get('total_duration', 'N/A')}\n\n")
                
                # Cloud removal statistics
                cloud_removal = stats.get('cloud_removal', {})
                f.write("Cloud Removal Statistics:\n")
                f.write(f"Scenes Processed: {cloud_removal.get('scenes_processed', 0)}\n")
                f.write(f"Scenes Cleaned: {cloud_removal.get('scenes_cleaned', 0)}\n")
                f.write(f"Failed Scenes: {stats.get('failed_scenes', 0)}\n")
                f.write(f"Processing Time: {cloud_removal.get('processing_time', 0):.2f} seconds\n\n")
                
                # Geometric median statistics
                geo_median = stats.get('geometric_median', {})
                f.write("Geometric Median Statistics:\n")
                f.write(f"Batches Processed: {geo_median.get('batches_processed', 0)}\n")
                f.write(f"Total Batches: {geo_median.get('total_batches', 0)}\n")
                f.write(f"Processing Time: {geo_median.get('processing_time', 0):.2f} seconds\n\n")
                
                # Errors
                if stats.get('errors'):
                    f.write("Errors Encountered:\n")
                    for error in stats.get('errors', []):
                        f.write(f"- {error}\n")
                        
            arcpy.AddMessage(f"\nDetailed statistics saved to: {stats_file}")
            return stats_file
            
        except Exception as e:
            arcpy.AddError(f"Error saving enhanced statistics: {str(e)}")
            return None
            
    def _parse_metadata(self, mtl_path):
        """Parse Landsat MTL file from a filesystem path."""
        try:
            with open(mtl_path) as f:
                content = f.read()
            return self._parse_metadata_content(content, mtl_path)
        except Exception as e:
            arcpy.AddWarning(f"Error parsing metadata {mtl_path}: {str(e)}")
            return None

    def _parse_metadata_content(self, content, source_label):
        """Parse Landsat MTL content provided as a string.

        Factored out of _parse_metadata so MTL text read from inside a
        .tar / .zip archive (via _read_mtl_from_archive) can reuse the
        same logic without first writing to disk.
        """
        try:
            # Extract key metadata. Each field is best-effort: a missing
            # tag is logged and left as None / 0, never propagated as an
            # IndexError. Pre-fix the `[0]` access would silently drop the
            # whole scene when ANY field was missing (common on partial
            # downloads, Collection-1 MTLs, or pre-C2 scenes).
            lines = content.split('\n')
            scene_info = {}

            def _first_match(needle, parser=None, default=None, exact_token=False):
                """Return parser(first MTL line's value), or default.

                Lines look like `    KEY = VALUE`. We split on the first
                `=`, strip the key, and compare. exact_token=True ensures
                'CLOUD_COVER' doesn't also match 'CLOUD_COVER_LAND' — the
                pre-fix code used a bare `in` substring check which would
                grab whichever variant appeared first.
                """
                for ln in lines:
                    if "=" not in ln:
                        continue
                    key, _, value = ln.partition("=")
                    key = key.strip()
                    if exact_token:
                        if key != needle:
                            continue
                    else:
                        if needle not in key:
                            continue
                    try:
                        raw = value.strip().strip('"')
                        return parser(raw) if parser else raw
                    except (IndexError, ValueError):
                        return default
                return default

            scene_info['acquisition_date'] = _first_match(
                'DATE_ACQUIRED',
                parser=lambda s: datetime.strptime(s, '%Y-%m-%d'),
            )
            scene_info['cloud_cover'] = _first_match(
                'CLOUD_COVER',
                parser=float,
                default=0.0,
                exact_token=True,  # don't pick up CLOUD_COVER_LAND
            )
            scene_info['utm_zone'] = _first_match('UTM_ZONE', parser=int)
            scene_info['processing_level'] = _first_match('PROCESSING_LEVEL')

            if scene_info['acquisition_date'] is None:
                arcpy.AddWarning(
                    f"  Scene {os.path.basename(source_label)} has no parseable "
                    f"DATE_ACQUIRED; dropping."
                )
                return None
            return scene_info

        except Exception as e:
            arcpy.AddWarning(f"Error parsing metadata {source_label}: {str(e)}")
            return None
        
    def _find_scenes(self, data_folder, utm_zone, temporal_filter, seasonal_pattern, stats):
        """Discover Landsat scenes for the given UTM zone.

        Two scene sources are accepted:

          1. EarthExplorer **.tar / .zip archives** sitting directly in
             `data_folder` (preferred). Band files are NEVER extracted —
             we list members with the stdlib, then build GDAL VSI paths
             (`/vsitar/...` or `/vsizip/...`) that arcpy.Raster can open
             directly through the GDAL drivers.
          2. **Already-extracted scene folders** (any depth under
             `data_folder`) containing the `*_MTL.txt` sidecar. Kept for
             back-compatibility with users who pre-extract.

        Scenes from both sources are de-duplicated by scene_id — if both
        an archive AND its extracted twin sit in the folder, the archive
        wins (cheaper to re-open, no risk of a stale partial extract).
        """
        try:
            scenes = []
            ls8_count = 0
            ls9_count = 0
            seen_scene_ids = set()
            arcpy.AddMessage("\nScanning for Landsat scenes...")

            # ---- 1) Archives (.tar / .zip) at the top of data_folder ----
            try:
                entries = os.listdir(data_folder)
            except OSError as e:
                arcpy.AddWarning(f"Could not list data folder: {e}")
                entries = []

            archive_entries = [
                fn for fn in entries
                if (fn.lower().endswith('.tar') or fn.lower().endswith('.zip'))
                and (fn.lower().startswith('lc08_') or fn.lower().startswith('lc09_'))
            ]

            for fn in archive_entries:
                stats['total_scenes'] += 1
                archive_path = os.path.join(data_folder, fn)

                # Derive the canonical scene_id from INSIDE the archive
                # rather than the outer filename. Required because:
                #   * Windows duplicate-download renames produce filenames
                #     like `LC08_..._T1 (1).tar`; the ` (1)` suffix breaks
                #     filename-based scene_id derivation but the MTL
                #     inside the archive still uses the canonical name.
                #   * Users sometimes rename archives manually.
                #   * Empty / corrupt tars are caught here too — they
                #     return None and get a clean skip with a warning.
                canonical_scene_id = self._derive_scene_id_from_archive(archive_path)
                if not canonical_scene_id:
                    arcpy.AddWarning(
                        f"  Could not derive scene_id from {fn} "
                        f"(archive unreadable or no *_MTL.txt inside)."
                    )
                    stats['failed_scenes'] += 1
                    continue
                scene_id = canonical_scene_id
                if scene_id in seen_scene_ids:
                    continue

                try:
                    mtl_content = self._read_mtl_from_archive(archive_path, scene_id)
                    if not mtl_content:
                        arcpy.AddWarning(f"  No MTL found inside archive {fn}; skipping.")
                        stats['failed_scenes'] += 1
                        continue

                    scene_info = self._parse_metadata_content(mtl_content, archive_path)
                    if not scene_info:
                        stats['failed_scenes'] += 1
                        continue

                    band_paths = self._find_band_files_vsi(archive_path, scene_id)
                    if not band_paths or 'QA_PIXEL' not in band_paths:
                        arcpy.AddWarning(
                            f"  Archive {fn} is missing required bands "
                            f"(found: {sorted((band_paths or {}).keys())}); skipping."
                        )
                        stats['failed_scenes'] += 1
                        continue

                    if scene_id.startswith('LC08'):
                        ls8_count += 1
                    elif scene_id.startswith('LC09'):
                        ls9_count += 1

                    if scene_info['utm_zone'] == utm_zone:
                        if self._apply_temporal_filter(scene_info, temporal_filter, seasonal_pattern):
                            scenes.append({
                                'path': archive_path,
                                'scene_id': scene_id,
                                'band_paths': band_paths,
                                'metadata': scene_info,
                                'is_archive': True,
                            })
                            stats['cloud_coverage'].append(scene_info['cloud_cover'])
                            seen_scene_ids.add(scene_id)
                except Exception as e:
                    stats['failed_scenes'] += 1
                    stats['errors'].append(f"{fn}: {e}")
                    continue

            # ---- 2) Already-extracted scene folders (back-compat) ----
            for root, _, files in os.walk(data_folder):
                for file in files:
                    if not file.endswith('_MTL.txt'):
                        continue
                    scene_id = file[:-len('_MTL.txt')]
                    if scene_id in seen_scene_ids:
                        continue  # archive form already discovered

                    stats['total_scenes'] += 1
                    try:
                        mtl_path = os.path.join(root, file)
                        scene_info = self._parse_metadata(mtl_path)
                        if not scene_info:
                            stats['failed_scenes'] += 1
                            continue

                        # Map filesystem band files by role.
                        band_paths = {}
                        for entry in os.listdir(root):
                            if not entry.startswith(scene_id):
                                continue
                            matched = False
                            for n in range(1, 8):
                                if entry == f"{scene_id}_SR_B{n}.TIF":
                                    band_paths[f"B{n}"] = os.path.join(root, entry)
                                    matched = True
                                    break
                            if not matched and entry == f"{scene_id}_QA_PIXEL.TIF":
                                band_paths["QA_PIXEL"] = os.path.join(root, entry)

                        required = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "QA_PIXEL"]
                        if any(k not in band_paths for k in required):
                            missing = [k for k in required if k not in band_paths]
                            arcpy.AddWarning(
                                f"  Extracted scene {scene_id} is missing {missing}; skipping."
                            )
                            stats['failed_scenes'] += 1
                            continue

                        if 'LC08' in file:
                            ls8_count += 1
                        elif 'LC09' in file:
                            ls9_count += 1

                        if scene_info['utm_zone'] == utm_zone:
                            if self._apply_temporal_filter(scene_info, temporal_filter, seasonal_pattern):
                                scenes.append({
                                    'path': root,
                                    'scene_id': scene_id,
                                    'band_paths': band_paths,
                                    'metadata': scene_info,
                                    'is_archive': False,
                                })
                                stats['cloud_coverage'].append(scene_info['cloud_cover'])
                                seen_scene_ids.add(scene_id)
                    except Exception as e:
                        stats['failed_scenes'] += 1
                        stats['errors'].append(str(e))
                        continue

            arcpy.AddMessage(f"Total Landsat 8 scenes: {ls8_count}")
            arcpy.AddMessage(f"Total Landsat 9 scenes: {ls9_count}")
            arcpy.AddMessage(f"Found {len(scenes)} valid scenes for UTM zone {utm_zone}")
            return scenes

        except Exception as e:
            arcpy.AddError(f"Error finding scenes: {str(e)}")
            return []
                
    def _apply_temporal_filter(self, scene_info, temporal_filter, seasonal_pattern):
        """Apply temporal filter to scene"""
        try:
            filter_type = temporal_filter['type']
            date = scene_info['acquisition_date']
            
            if filter_type == 'all_images':
                return True
                
            elif filter_type == 'specific_year':
                return date.year == temporal_filter['year']
                
            elif filter_type == 'month_in_year':
                return (date.year == temporal_filter['year'] and 
                        date.month == temporal_filter['month'])
                
            elif filter_type == 'month_all_years':
                return date.month == temporal_filter['month']
                
            elif filter_type == 'season_in_year':
                if 'year' in temporal_filter:
                    if date.year != temporal_filter['year']:
                        return False
                return self._is_in_season(date, seasonal_pattern, temporal_filter['season'])
                
            elif filter_type == 'season_all_years':
                return self._is_in_season(date, seasonal_pattern, temporal_filter['season'])
                
            return False
            
        except Exception as e:
            arcpy.AddWarning(f"Error applying temporal filter: {str(e)}")
            return False
            
    def _is_in_season(self, date, pattern, season):
        """Check if date falls within specified season"""
        season_months = {
            'temperate': {
                'spring': [3, 4, 5],
                'summer': [6, 7, 8],
                'autumn': [9, 10, 11],
                'winter': [12, 1, 2]
            },
            'angola': {
                'rainy': [11, 12, 1, 2, 3, 4],
                'dry': [5, 6, 7, 8, 9, 10],
                'rainy_peak': [1, 2, 3],
                'dry_peak': [6, 7, 8]
            },
            'cape_verde': {
                'dry': [12, 1, 2, 3, 4, 5, 6],
                'rainy': [8, 9, 10],
                'transition_dry_wet': [7],
                'transition_wet_dry': [11]
            },
            'mozambique': {
                'rainy': [10, 11, 12, 1, 2, 3],
                'dry': [4, 5, 6, 7, 8, 9],
                'rainy_peak': [12, 1, 2],
                'dry_peak': [7, 8, 9]
            }
        }
        
        return date.month in season_months[pattern][season.lower()]

    # ------------------------------------------------------------------
    # Archive reading via GDAL Virtual File System (no extraction to disk)
    # ------------------------------------------------------------------
    #
    # EarthExplorer ships Landsat C2L2 scenes as multi-hundred-MB `.tar`
    # archives. The old approach extracted each archive into a sibling
    # folder before processing; the disk overhead doubled the data
    # footprint and the cleanup story for stale extracts was fragile.
    #
    # The new approach reads bands directly from inside the archive via
    # GDAL's virtual file system: `/vsitar/{path}/{member}` for tar and
    # `/vsizip/{path}/{member}` for zip. arcpy.Raster opens these strings
    # through GDAL's drivers — no extraction, no cleanup, no double disk.
    # MTL.txt sidecars are read in-memory with the stdlib tarfile/zipfile
    # modules.

    @staticmethod
    def _derive_scene_id_from_archive(archive_path):
        """Read the canonical Landsat scene_id from inside an archive.

        Looks for a `*_MTL.txt` member and returns the part before the
        suffix. This is more robust than deriving from the outer
        filename because:

          * Windows duplicate downloads add ` (1)` to the filename
            (e.g., `LC08_..._T1 (1).tar`) while the MTL inside stays
            canonical. The outside-derived scene_id wouldn't match
            the inside member names.
          * Users sometimes rename archives manually.
          * Empty/corrupt archives that can't be opened return None,
            giving callers a clean way to skip them with a warning.

        Returns the canonical scene_id (e.g.,
        `LC08_L2SR_217033_20210707_20210713_02_T1`) or None.
        """
        sl = archive_path.lower()
        try:
            if sl.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        basename = member.name.rsplit('/', 1)[-1]
                        if basename.endswith('_MTL.txt'):
                            return basename[:-len('_MTL.txt')]
            elif sl.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    for name in zf.namelist():
                        basename = name.rsplit('/', 1)[-1]
                        if basename.endswith('_MTL.txt'):
                            return basename[:-len('_MTL.txt')]
        except (tarfile.TarError, zipfile.BadZipFile, OSError):
            # Empty / corrupt / unreadable archive — caller logs.
            pass
        return None

    @staticmethod
    def _validate_tar_file(tar_path):
        """Quick integrity check for a .tar archive.

        Returns (is_valid, error_message). A tar is considered valid when
        it exists, is non-empty, is recognised by `tarfile.is_tarfile`,
        and can be walked to its first member without raising. We
        intentionally do NOT decompress the whole archive — GDAL will do
        that lazily, member-by-member, as bands are read.
        """
        try:
            if not os.path.isfile(tar_path):
                return False, "file does not exist"
            if os.path.getsize(tar_path) == 0:
                return False, "file is empty"
            if not tarfile.is_tarfile(tar_path):
                return False, "not a valid tar archive"
            with tarfile.open(tar_path, 'r') as tf:
                # getmembers() walks the header chain; if any header is
                # corrupt the call raises here rather than later inside
                # the GDAL read path.
                tf.getmembers()
            return True, ""
        except (tarfile.TarError, OSError) as e:
            return False, str(e)

    def _find_band_files_vsi(self, archive_path, scene_id):
        """Build GDAL VSI paths for every required Landsat band inside an archive.

        Lists members with the stdlib (tarfile / zipfile) — never
        extracts — then maps each Landsat band filename to its VSI path.
        Returns `{band_role: vsi_path}` on success, or None if the
        archive is unreadable or holds none of the required bands.

        Band roles returned: 'B1'..'B7' (Surface Reflectance) and
        'QA_PIXEL' (the bit-packed QA band used for cloud masking).

        Notes on path syntax:
          * GDAL VSI uses forward slashes throughout. On Windows the
            drive-letter form `/vsitar/C:/path/to/archive.tar/member`
            is the supported convention — backslashes in the archive
            path are normalised to '/' before prepending the VSI scheme.
          * Members may include a leading directory (`{scene}/{band}.TIF`)
            on some downloads and live at the archive root on others.
            We match on basename so both layouts work.
        """
        sl = archive_path.lower()
        if sl.endswith('.tar'):
            ok, err = self._validate_tar_file(archive_path)
            if not ok:
                arcpy.AddWarning(f"  Invalid tar {archive_path}: {err}")
                return None
            try:
                with tarfile.open(archive_path, 'r') as tf:
                    namelist = [m.name for m in tf.getmembers() if m.isfile()]
            except (tarfile.TarError, OSError) as e:
                arcpy.AddWarning(f"  Could not list tar {archive_path}: {e}")
                return None
            scheme = "vsitar"
        elif sl.endswith('.zip'):
            try:
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    namelist = [n for n in zf.namelist() if not n.endswith('/')]
            except (zipfile.BadZipFile, OSError) as e:
                arcpy.AddWarning(f"  Could not list zip {archive_path}: {e}")
                return None
            scheme = "vsizip"
        else:
            return None

        base_path = f"/{scheme}/{archive_path.replace(os.sep, '/')}"
        band_paths = {}
        for member in namelist:
            basename = member.rsplit('/', 1)[-1]
            if not basename.startswith(scene_id):
                continue
            matched = False
            for n in range(1, 8):
                if basename == f"{scene_id}_SR_B{n}.TIF":
                    band_paths[f"B{n}"] = f"{base_path}/{member}"
                    matched = True
                    break
            if not matched and basename == f"{scene_id}_QA_PIXEL.TIF":
                band_paths["QA_PIXEL"] = f"{base_path}/{member}"
        return band_paths or None

    def _read_mtl_from_archive(self, archive_path, scene_id):
        """Read `{scene_id}_MTL.txt` content from inside a .tar or .zip.

        Returns the MTL text as a string, or None if the file is absent
        or the archive can't be opened. No disk extraction.
        """
        mtl_name = f"{scene_id}_MTL.txt"
        sl = archive_path.lower()
        try:
            if sl.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        if member.name.rsplit('/', 1)[-1] == mtl_name:
                            fobj = tf.extractfile(member)
                            if fobj is None:
                                return None
                            return fobj.read().decode('utf-8', errors='replace')
            elif sl.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    for name in zf.namelist():
                        if name.rsplit('/', 1)[-1] == mtl_name:
                            with zf.open(name) as fobj:
                                return fobj.read().decode('utf-8', errors='replace')
        except (tarfile.TarError, zipfile.BadZipFile, OSError, UnicodeError) as e:
            arcpy.AddWarning(f"  Could not read MTL from {archive_path}: {e}")
        return None

    def _write_provenance_csv(self, output_raster_path, scenes_used, stats):
        """Write `{output_raster}_provenance.csv` documenting every scene
        that fed into the mosaic.

        Columns (RFC 4180-quoted UTF-8):
            scene_id, sensor, acquisition_datetime, path_row,
            cloud_cover_pct, input_path, processing_baseline,
            toolbox_version, processing_datetime

        Failures are non-fatal — a missing provenance file is annoying
        but doesn't invalidate the mosaic itself.
        """
        if not output_raster_path or not scenes_used:
            return

        try:
            csv_path = _sidecar_path_for_raster(output_raster_path, "_provenance.csv")
            now_iso = datetime.now().isoformat(timespec="seconds")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "scene_id", "sensor", "acquisition_datetime",
                    "path_row", "cloud_cover_pct", "input_path",
                    "processing_baseline", "toolbox_version",
                    "processing_datetime",
                ])
                for scene in scenes_used:
                    meta = scene.get("metadata", {}) or {}
                    scene_path = scene.get("path", "") or ""
                    # New scene dicts carry an explicit scene_id; older
                    # ones derived it from the path basename (which for
                    # archives includes the `.tar` / `.zip` suffix).
                    scene_id = scene.get("scene_id") or os.path.basename(
                        scene_path.rstrip(os.sep)
                    ) or ""
                    if scene_id.lower().endswith((".tar", ".zip")):
                        scene_id = scene_id[:-4]
                    sensor = "Landsat 9" if scene_id.startswith("LC09") else "Landsat 8"
                    writer.writerow([
                        scene_id,
                        sensor,
                        meta.get("date_acquired", "") or meta.get("acquisition_date", ""),
                        meta.get("path_row", "") or "",
                        meta.get("cloud_cover", "") or "",
                        scene_path,
                        meta.get("collection_number", "") or "C2",
                        TOOLBOX_VERSION,
                        now_iso,
                    ])
            arcpy.AddMessage(f"Provenance CSV: {csv_path}")
        except OSError as e:
            arcpy.AddWarning(f"Failed to write provenance CSV: {e}")


# ---------------------------------------------------------------------------
# Tool 01 — Sentinel-2 L2A Mosaic
# ---------------------------------------------------------------------------


# Sen2Cor Scene Classification Layer (SCL) class meanings:
#   0  = No data           (NoData already)
#   1  = Saturated/defective
#   2  = Dark area pixels  (can be legitimate dark terrain — e.g. basalt)
#   3  = Cloud shadow
#   4  = Vegetation        (keep)
#   5  = Bare soil         (keep)
#   6  = Water             (keep)
#   7  = Unclassified      (Sen2Cor often catches cloud edges here)
#   8  = Cloud medium probability
#   9  = Cloud high probability
#   10 = Thin cirrus
#   11 = Snow / ice        (false-positive over terrain without permanent snow)
_S2_SCL_PRESETS = {
    # Original Sen2Cor default — leaves unclassified pixels and any snow
    # misclassifications in the output. Misses cloud edges Sen2Cor doesn't
    # tag.
    "Standard": (3, 8, 9, 10),
    # Recommended for most AOIs: adds saturated, unclassified (cloud edges),
    # and snow/ice (false positives anywhere without permanent snow). Used
    # by default.
    "Aggressive": (1, 3, 7, 8, 9, 10, 11),
    # Adds dark-area pixels — useful where cloud shadows aren't fully
    # tagged, but can over-mask legitimate dark terrain (volcanic basalt,
    # burned areas). Use with care.
    "Maximum": (1, 2, 3, 7, 8, 9, 10, 11),
}
# Backwards compatibility — kept so tests/scripts referencing the old name
# don't break. The active mask is now selected by the GP parameter and
# defaults to Aggressive.
_S2_SCL_CLOUD_CLASSES = _S2_SCL_PRESETS["Aggressive"]

# Sentinel-2 L2A scale factor: DN * 0.0001 = surface reflectance.
_S2_REFLECTANCE_SCALE = 0.0001

# 12-band L2A stack order matching SENSOR_BAND_ROLES[SENSOR_SENTINEL2].
# Wavelength-ordered: B01 (Coastal, 60m) → B12 (SWIR2, 20m).  B10 is
# absent because L2A does not carry it (Sen2Cor strips B10 from L1C
# during atmospheric correction). The atmospheric bands B01 and B09
# are included for native band-numbering parity with the S2 user
# guide and traceability when the mosaic is opened in a third-party
# tool — surface-analysis pipelines (Tool 04 indices, Tool 05
# transforms) resolve bands by role name via SENSOR_BAND_ROLES and
# so do not need to know which physical bands are present.
_S2_STACK_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]
_S2_NATIVE_10M = {"B02", "B03", "B04", "B08"}  # the rest are 20m or 60m → resampled


class Sentinel2Mosaic(object):
    """Tool 01 — Sentinel-2 L2A cloud-removed mosaic.

    Accepts a folder of S2 L2A products, either as:
      a) Copernicus `.zip` archives (e.g., `S2A_MSIL2A_*.zip`) — extracted
         transparently before processing.
      b) Already-extracted `.SAFE` folders.

    For each scene the tool reads the 10m bands (B02, B03, B04, B08) at
    native resolution, the 20m bands (B05, B06, B07, B8A, B11, B12)
    resampled to 10m via BILINEAR, and the 60m bands (B01, B09)
    aggressively upsampled to 10m via BILINEAR. Cloud masking uses the
    SCL layer (classes 3, 8, 9, 10). Surface reflectance is scaled to
    [0, 1] via the 0.0001 factor.

    The cloud-masked, scaled 12-band stack from each scene is then fed
    to arcpy.ia.GeometricMedian per MGRS tile. Multi-tile regions are
    merged after per-tile median composites are built. A provenance CSV
    and a band-mapping CSV are written alongside the output — the
    band-mapping documents which stack position corresponds to each S2
    native band name, role and wavelength, so the mosaic stays
    interpretable when the GDB raster format can't carry band
    descriptions.
    """

    def __init__(self):
        self.label = "01 — Sentinel-2 L2A Mosaic"
        self.description = (
            "Build a cloud-removed mosaic from Sentinel-2 L2A scenes "
            "(Sen2Cor / Copernicus L2A). Accepts a folder of .zip "
            "archives (auto-extracted) or already-extracted .SAFE "
            "folders. SCL classes 3/8/9/10 are masked; the temporal "
            "stack is reduced to a geometric median per MGRS tile. "
            "20m bands are resampled to 10m. A provenance CSV is "
            "written alongside the output."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # GP parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        gdb = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="gdb_path",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        gdb.filter.list = ["Local Database"]

        mosaic_name = arcpy.Parameter(
            displayName="Output Mosaic Name",
            name="mosaic_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        data_folder = arcpy.Parameter(
            displayName="Sentinel-2 Data Folder",
            name="data_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )

        region = arcpy.Parameter(
            displayName="Region",
            name="region",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        # Same region list as LandsatMosaic — provides the seasonal
        # pattern for temporal filtering. S2 uses MGRS tiles rather than
        # UTM zones; tile assignment is done by scanning the data folder.
        region.filter.list = [
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
            "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)",
            "Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)",
            "Angola",
            "Mozambique",
        ]

        time_type = arcpy.Parameter(
            displayName="Time Filter Type",
            name="time_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        # `all_images` (default) processes every scene that parsed,
        # matching the behaviour of LandsatMosaic. The previous default
        # of `year_month` silently required year+month to be filled in
        # for any scene to pass the filter — a usability footgun that
        # produced an immediate "No scenes match" crash when the user
        # accepted defaults.
        time_type.filter.list = [
            "all_images",
            "year_month", "month_all_years",
            "season_in_year", "season_all_years",
        ]
        time_type.value = "all_images"

        year = arcpy.Parameter(
            displayName="Year (when applicable)",
            name="year",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        month = arcpy.Parameter(
            displayName="Month (when applicable, 1-12)",
            name="month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        season = arcpy.Parameter(
            displayName="Season (when applicable)",
            name="season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )

        # SCL aggressiveness preset. The Aggressive default (1, 3, 7, 8,
        # 9, 10, 11) catches the cloud-edge halo Sen2Cor leaves as
        # "Unclassified" + the snow/ice false positives the original
        # default (3, 8, 9, 10) missed. See _S2_SCL_PRESETS docstring
        # near the top of this file for the full class table.
        cloud_aggressiveness = arcpy.Parameter(
            displayName="Cloud Mask Aggressiveness",
            name="cloud_aggressiveness",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        cloud_aggressiveness.filter.list = list(_S2_SCL_PRESETS.keys())
        cloud_aggressiveness.value = "Aggressive"

        # Cloud-mask buffer — dilates the SCL mask by N pixels via
        # FocalStatistics MAXIMUM. Catches the 1-2 pixel halo around
        # clouds that Sen2Cor classifies as Vegetation/Bare-soil but is
        # in fact thin cloud / partial cover.
        cloud_buffer = arcpy.Parameter(
            displayName="Cloud Mask Buffer (pixels)",
            name="cloud_buffer_pixels",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        cloud_buffer.value = 2
        cloud_buffer.filter.type = "Range"
        cloud_buffer.filter.list = [0, 10]

        mask_feature = arcpy.Parameter(
            displayName="Optional AOI Mask Feature (polygon)",
            name="mask_feature",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
        )
        mask_feature.filter.list = ["Polygon"]

        save_stats = arcpy.Parameter(
            displayName="Save Provenance CSV",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        save_stats.value = True

        preserve_scratch = arcpy.Parameter(
            displayName="Preserve Scratch & Resume on Re-run",
            name="preserve_scratch",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        preserve_scratch.value = False

        # Subprocess-per-batch perf workaround. The 2026-05 in-process
        # per-scene loop showed doubling-per-batch timing degradation
        # (48s to 950s per scene over 89 scenes) that no in-process
        # cleanup could bound. Splitting the run into batches and
        # spawning a fresh python.exe per batch reclaims accumulated
        # arcpy / GDAL state by OS guarantee.
        subprocess_batch_size = arcpy.Parameter(
            displayName=(
                "Subprocess Batch Size (advanced; controls the per-"
                "scene-loop memory cycling. Each batch of N scenes "
                "runs in a fresh python.exe subprocess to reclaim "
                "accumulated arcpy / GDAL state. Default 10 trades "
                "~10s arcpy reimport per batch for flat per-scene "
                "timing. Set 0 to disable subprocess batching (legacy "
                "single-process loop, for A/B comparison or debugging)."
            ),
            name="subprocess_batch_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        subprocess_batch_size.value = 10
        subprocess_batch_size.filter.type = "Range"
        subprocess_batch_size.filter.list = [0, 100]

        return [
            gdb, mosaic_name, data_folder, region, time_type,
            year, month, season,
            cloud_aggressiveness, cloud_buffer,
            mask_feature, save_stats, preserve_scratch,
            subprocess_batch_size,
        ]

    def updateParameters(self, parameters):
        """Enable/disable time-detail parameters based on time_type."""
        try:
            time_type = parameters[4]
            year = parameters[5]
            month = parameters[6]
            season = parameters[7]
            if time_type.valueAsText == "all_images":
                year.enabled = False
                month.enabled = False
                season.enabled = False
            elif time_type.valueAsText == "year_month":
                year.enabled = True
                month.enabled = True
                season.enabled = False
            elif time_type.valueAsText == "month_all_years":
                year.enabled = False
                month.enabled = True
                season.enabled = False
            elif time_type.valueAsText == "season_in_year":
                year.enabled = True
                month.enabled = False
                season.enabled = True
            elif time_type.valueAsText == "season_all_years":
                year.enabled = False
                month.enabled = False
                season.enabled = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        # Always-defined for the finally cleanup.
        scratch_dir = None
        composite_temp_paths = []

        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                arcpy.AddError("Spatial Analyst extension is required.")
                return None
            arcpy.CheckOutExtension("Spatial")
            arcpy.env.overwriteOutput = True
            # Take control of cancellation so loop-boundary checks can
            # abort cleanly instead of Pro hard-killing the process.
            arcpy.env.autoCancelling = False

            gdb_path = parameters[0].valueAsText
            mosaic_name = parameters[1].valueAsText
            data_folder = parameters[2].valueAsText
            region = parameters[3].valueAsText
            time_type = parameters[4].valueAsText
            year = parameters[5].value
            month = parameters[6].value
            season = parameters[7].valueAsText
            cloud_aggressiveness = parameters[8].valueAsText or "Aggressive"
            cloud_buffer_pixels = parameters[9].value if parameters[9].value is not None else 2
            mask_feature = parameters[10].valueAsText
            save_stats = bool(parameters[11].value)
            preserve_scratch = bool(parameters[12].value)
            # Subprocess batch size for the per-scene-loop perf
            # workaround (Advanced Options). Default 10. Set to 0 to
            # use the legacy single-process loop (A/B comparison or
            # debugging only).
            subprocess_batch_size = (
                int(parameters[13].value)
                if len(parameters) > 13 and parameters[13].value is not None
                else 10
            )

            # Resolve aggressiveness preset → SCL class tuple. The dropdown
            # constrains the value to the preset keys; fall back defensively
            # to Aggressive if somehow null.
            scl_classes = _S2_SCL_PRESETS.get(
                cloud_aggressiveness, _S2_SCL_PRESETS["Aggressive"]
            )

            # ----------------------------------------------------------------
            # AOI-first scoping. See LandsatMosaic.execute() for the full
            # rationale. For Sentinel-2 specifically the win is significant:
            # the per-band 20m→10m resample (the dominant cost per scene)
            # now writes only AOI pixels, the per-band SCL cloud mask is
            # only evaluated over the AOI, and GeometricMedian iterates
            # over the AOI extent. For a Faial-sized AOI vs. a full S2
            # tile (110×110 km) that's ~70x fewer pixels per scene through
            # the JP2 decode pipeline.
            # ----------------------------------------------------------------
            # Header — one block of run context.
            arcpy.AddMessage("=" * 60)
            arcpy.AddMessage(f"SENTINEL-2 L2A MOSAIC — {region}")
            arcpy.AddMessage("=" * 60)
            arcpy.AddMessage(f"  Output:     {gdb_path}\\{mosaic_name}")
            arcpy.AddMessage(f"  Source:     {data_folder}")
            arcpy.AddMessage(
                f"  Cloud mask: {cloud_aggressiveness} "
                f"(SCL classes {list(scl_classes)}, buffer {cloud_buffer_pixels}px)"
            )

            if mask_feature and arcpy.Exists(mask_feature):
                arcpy.env.mask = mask_feature
                arcpy.env.extent = mask_feature
                arcpy.AddMessage(f"  AOI:        {mask_feature} (env.mask + env.extent active)")
            elif mask_feature:
                arcpy.AddWarning(
                    f"  AOI:        {mask_feature!r} NOT FOUND — running over full tile footprint"
                )

            scratch_dir = _make_mosaic_scratch_dir(
                gdb_path, "_genesis_s2_scratch", mosaic_name,
            )
            arcpy.AddMessage(f"  Scratch:    {scratch_dir}")
            if preserve_scratch:
                arcpy.AddMessage(
                    "  Resume:     Preserve Scratch is ON — completed "
                    "per-scene stacks from previous runs will be reused"
                )

            # Phase 1 — discover scenes
            arcpy.AddMessage("\n▶ Phase 1 — Scene discovery")
            sources = self._find_safe_scenes(data_folder)
            if not sources:
                arcpy.AddError(
                    "  ✗ No Sentinel-2 scenes found. Expected Copernicus .zip "
                    "archives or extracted .SAFE folders."
                )
                return None
            n_zip = sum(1 for _, k in sources if k == "zip")
            n_safe = sum(1 for _, k in sources if k == "safe")
            arcpy.AddMessage(
                f"  ✓ {len(sources)} scene(s) — {n_zip} archive(s), {n_safe} SAFE folder(s)"
            )

            # Phase 2 — temporal filter
            arcpy.AddMessage("\n▶ Phase 2 — Temporal filter")
            seasonal_pattern = self._seasonal_pattern_for_region(region)
            temporal_filter = self._create_temporal_filter(
                time_type, year, month, season,
            )
            kept_scenes = []
            skipped_no_meta = 0
            seen_product_uris = set()
            duplicate_sources = []
            for path, kind in sources:
                meta = self._parse_safe_metadata(path, kind)
                if meta is None:
                    skipped_no_meta += 1
                    arcpy.AddWarning(f"  ✗ {os.path.basename(path)}: no metadata")
                    continue
                if not self._scene_passes_filter(meta, temporal_filter, seasonal_pattern):
                    continue
                product_uri = meta.get("product_uri")
                # Drop duplicate downloads ("Foo.zip" + "Foo (1).zip" both
                # resolve to the same PRODUCT_URI). Keeping both costs ~10
                # minutes per scene and produces redundant scratch outputs.
                if product_uri in seen_product_uris:
                    duplicate_sources.append(os.path.basename(path))
                    continue
                seen_product_uris.add(product_uri)
                kept_scenes.append({
                    "path": path,
                    "source_kind": kind,
                    "metadata": meta,
                })
            if duplicate_sources:
                arcpy.AddMessage(
                    f"  ✓ {len(duplicate_sources)} duplicate download(s) "
                    f"dropped: {', '.join(duplicate_sources[:5])}"
                    + ("..." if len(duplicate_sources) > 5 else "")
                )
            if not kept_scenes:
                arcpy.AddError("  ✗ No scenes match the temporal filter.")
                return None
            arcpy.AddMessage(
                f"  ✓ {len(kept_scenes)}/{len(sources)} scenes kept "
                f"({temporal_filter.get('type', 'all_images')})"
            )

            # Step 3: process each scene (mask + scale + stack), grouped by tile.
            arcpy.AddMessage(
                f"\n▶ Phase 3 — Cloud-mask + stack ({len(kept_scenes)} scenes)"
            )
            scenes_by_tile = {}
            all_scenes_used = []
            stack_start = datetime.now()

            # Resume scan: any scene whose stack file AND its .complete
            # marker survive in scratch is treated as already processed.
            # Both files must be present — the marker is written only
            # after CompositeBands returns successfully, so a stack
            # without a marker is partial and must be rebuilt.
            resumed_count = 0
            to_process = []
            for scene in kept_scenes:
                meta = scene["metadata"]
                tile = meta.get("tile_id")
                product_uri = meta.get("product_uri")
                if not (tile and product_uri):
                    to_process.append(scene)
                    continue
                expected_stack = os.path.join(
                    scratch_dir, f"{product_uri}_stack.tif"
                )
                expected_marker = expected_stack + ".complete"
                if (os.path.exists(expected_stack)
                        and os.path.exists(expected_marker)):
                    scenes_by_tile.setdefault(tile, []).append(expected_stack)
                    composite_temp_paths.append(expected_stack)
                    all_scenes_used.append(scene)
                    resumed_count += 1
                else:
                    to_process.append(scene)
            # Schema defence: refuse to resume from a scratch built by
            # an earlier code version with a different band layout (the
            # May 2026 expansion from 10 to 12 bands is the classic
            # trigger — old 10-band stacks would silently mix into the
            # geomedian and fail Phase 4).
            if not _check_scratch_schema(
                composite_temp_paths, len(_S2_STACK_ORDER), "Sentinel-2",
            ):
                return None
            if resumed_count:
                arcpy.AddMessage(
                    f"  ✓ Resume: {resumed_count}/{len(kept_scenes)} scene "
                    f"stack(s) reused from previous run"
                )
                if not to_process:
                    arcpy.AddMessage(
                        "  ✓ All scenes already processed — skipping to "
                        "Phase 4"
                    )

            if to_process and subprocess_batch_size > 0:
                # Subprocess-per-batch path. Each batch runs in a fresh
                # python.exe to flush accumulated arcpy / GDAL state.
                # See module-level _run_scene_batches docstring.
                spec_extra = {
                    "mask_feature": (
                        _resolve_to_catalog_path(mask_feature)
                        if mask_feature and arcpy.Exists(mask_feature)
                        else None
                    ),
                    "scl_classes": list(scl_classes),
                    "cloud_buffer_pixels": cloud_buffer_pixels,
                }
                ok = _run_scene_batches(
                    worker_kind="s2",
                    scenes=to_process,
                    batch_size=subprocess_batch_size,
                    scratch_dir=scratch_dir,
                    spec_extra=spec_extra,
                    log_prefix="  ",
                )
                if not ok:
                    return None
                # Rescan scratch for newly-completed scenes; append to
                # the accumulators that the resume scan already primed
                # with previously-completed work. The user-facing tally
                # is already emitted by _run_scene_batches' tail
                # summary; this rescan only populates internal state.
                for scene in to_process:
                    meta = scene["metadata"]
                    tile = meta.get("tile_id")
                    product_uri = meta.get("product_uri")
                    if not (tile and product_uri):
                        continue
                    stack = os.path.join(
                        scratch_dir, f"{product_uri}_stack.tif",
                    )
                    marker = stack + ".complete"
                    if os.path.exists(stack) and os.path.exists(marker):
                        scenes_by_tile.setdefault(tile, []).append(stack)
                        composite_temp_paths.append(stack)
                        all_scenes_used.append(scene)
            elif to_process:
                # Legacy single-process loop, kept for A/B comparison
                # and as a fallback when subprocess_batch_size = 0.
                arcpy.SetProgressor(
                    "step", "Per-scene processing",
                    0, max(1, len(to_process)), 1,
                )
                t0_phase = time.time()
                scene_times = []
                failures = []
                for idx, scene in enumerate(to_process, 1):
                    if arcpy.env.isCancelled:
                        arcpy.ResetProgressor()
                        arcpy.AddWarning(
                            f"  ✗ Cancelled after {idx-1}/{len(to_process)} scenes."
                        )
                        return None
                    meta = scene["metadata"]
                    tile = meta.get("tile_id")
                    sid = meta.get("product_uri") or os.path.basename(scene["path"])
                    if not tile:
                        fail_msg = "no tile ID"
                        arcpy.AddMessage(_format_scene_log_line(
                            idx, len(to_process), sid, 0.0, fail=fail_msg,
                        ))
                        failures.append((idx, sid, fail_msg))
                        arcpy.SetProgressorPosition(idx)
                        continue
                    src_tag = "zip" if scene.get("source_kind") == "zip" else "safe"
                    arcpy.SetProgressorLabel(
                        f"[{idx}/{len(to_process)}] [{tile}/{src_tag}] {sid}"
                    )
                    scene_start = time.time()
                    try:
                        stacked_path = self._process_scene(
                            scene, scratch_dir, scl_classes, cloud_buffer_pixels,
                        )
                        elapsed = time.time() - scene_start
                    except Exception as e:
                        elapsed = time.time() - scene_start
                        fail_msg = f"{type(e).__name__}: {e}"
                        arcpy.AddMessage(_format_scene_log_line(
                            idx, len(to_process), sid, elapsed, fail=fail_msg,
                        ))
                        failures.append((idx, sid, fail_msg))
                        arcpy.SetProgressorPosition(idx)
                        continue
                    if stacked_path:
                        scenes_by_tile.setdefault(tile, []).append(stacked_path)
                        composite_temp_paths.append(stacked_path)
                        all_scenes_used.append(scene)
                        scene_times.append(elapsed)
                        arcpy.AddMessage(_format_scene_log_line(
                            idx, len(to_process), sid, elapsed,
                        ))
                    arcpy.SetProgressorPosition(idx)
                    _periodic_arcpy_cache_flush(idx)
                arcpy.ResetProgressor()
                _emit_phase3_summary(
                    len(scene_times), len(failures),
                    time.time() - t0_phase, failures,
                )
            if not scenes_by_tile:
                arcpy.AddError("  ✗ No scenes survived cloud masking + stacking.")
                return None
            arcpy.AddMessage(
                f"  Tiles: {sorted(scenes_by_tile.keys())}"
            )

            # Step 4: per-tile geometric median.
            tile_mosaics = []
            with phase(
                f"Phase 4 — GeometricMedian over {len(scenes_by_tile)} tile(s)",
                quiet_close=True,
                # Outer except in Sentinel2Mosaic.execute logs the canonical
                # "Tool 01 failed" message + traceback — suppress phase's own
                # warning to avoid the double yellow+red icons.
                silent_error=True,
            ) as ph:
                arcpy.SetProgressor(
                    "step", "Per-tile GeometricMedian", 0, len(scenes_by_tile), 1
                )
                for ti, (tile, stacked_paths) in enumerate(scenes_by_tile.items(), 1):
                    if arcpy.env.isCancelled:
                        arcpy.ResetProgressor()
                        arcpy.AddWarning(
                            f"  ✗ Cancelled after {ti-1}/{len(scenes_by_tile)} tiles."
                        )
                        return None
                    arcpy.SetProgressorLabel(
                        f"[{ti}/{len(scenes_by_tile)}] {tile} ({len(stacked_paths)} scenes)"
                    )
                    tile_start = datetime.now()
                    try:
                        tile_mosaic_name = f"{mosaic_name}_{tile}"
                        tile_mosaic_path = os.path.join(gdb_path, tile_mosaic_name)
                        median = arcpy.ia.GeometricMedian(
                            stacked_paths,
                            epsilon=_GEOMETRIC_MEDIAN_EPSILON,
                            max_iteration=_GEOMETRIC_MEDIAN_MAX_ITER,
                            extent_type="UnionOf",
                            cellsize_type="FirstOf",
                        )
                        median.save(tile_mosaic_path)
                        tile_mosaics.append(tile_mosaic_path)
                        arcpy.AddMessage(
                            f"  ✓ [{tile}] {len(stacked_paths)} scenes → "
                            f"{(datetime.now() - tile_start).total_seconds():.1f}s"
                        )
                    except arcpy.ExecuteError as e:
                        arcpy.AddError(f"  ✗ [{tile}] GeometricMedian failed: {e}")
                        continue
                    arcpy.SetProgressorPosition(ti)
                arcpy.ResetProgressor()
            arcpy.AddMessage(
                f"  ✓ Phase 4 in {ph.elapsed:.1f}s "
                f"({len(tile_mosaics)} tile mosaic(s))"
            )

            if not tile_mosaics:
                arcpy.AddError("  ✗ No tile mosaics were created.")
                return None

            # Phase 5 — merge + mask + cleanup
            with phase(
                "Phase 5 — Merge / mask / cleanup",
                quiet_close=True,
                silent_error=True,
            ) as ph5:
                intermediates_to_delete = []

                if len(tile_mosaics) > 1:
                    final_path = os.path.join(gdb_path, mosaic_name)
                    merge_start = datetime.now()
                    arcpy.management.MosaicToNewRaster(
                        input_rasters=tile_mosaics,
                        output_location=gdb_path,
                        raster_dataset_name_with_extension=mosaic_name,
                        coordinate_system_for_the_raster="",
                        pixel_type="32_BIT_FLOAT",
                        cellsize=10,
                        number_of_bands=len(_S2_STACK_ORDER),
                        mosaic_method="MEAN",
                    )
                    arcpy.AddMessage(
                        f"  ✓ Merged {len(tile_mosaics)} tiles in "
                        f"{(datetime.now() - merge_start).total_seconds():.1f}s "
                        f"→ {os.path.basename(final_path)}"
                    )
                    intermediates_to_delete.extend(tile_mosaics)
                    final_mosaic = final_path
                else:
                    final_mosaic = tile_mosaics[0]
                    arcpy.AddMessage(f"  ✓ Single tile → {os.path.basename(final_mosaic)}")

                # Apply AOI mask if requested.
                final_mosaic, unmasked_to_delete = _apply_aoi_mask_and_save(
                    final_mosaic, mask_feature, gdb_path, mosaic_name,
                )
                if unmasked_to_delete:
                    intermediates_to_delete.append(unmasked_to_delete)

                # Delete superseded intermediates so only the final mosaic
                # remains in the output GDB.
                cleaned_count = 0
                for path in intermediates_to_delete:
                    if path and path != final_mosaic:
                        try:
                            if arcpy.Exists(path):
                                arcpy.management.Delete(path)
                                cleaned_count += 1
                        except arcpy.ExecuteError as e:
                            arcpy.AddWarning(
                                f"  Could not delete {os.path.basename(path)}: {e}"
                            )
                if cleaned_count:
                    arcpy.AddMessage(f"  ✓ Cleaned up {cleaned_count} intermediate(s)")

                if save_stats:
                    self._write_provenance_csv(final_mosaic, all_scenes_used)
                _write_band_sidecar_csv(final_mosaic, "sentinel-2")

            _sanity_check_output(
                final_mosaic, sensor_hint="sentinel-2",
                label=os.path.basename(final_mosaic),
            )

            total_elapsed = (datetime.now() - stack_start).total_seconds()
            mins, secs = divmod(int(total_elapsed), 60)
            hrs, mins = divmod(mins, 60)
            time_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"

            arcpy.AddMessage("\n" + "=" * 60)
            arcpy.AddMessage(f"DONE — {os.path.basename(final_mosaic)}")
            arcpy.AddMessage(f"Total: {time_str}  |  Scenes: {len(all_scenes_used)}")
            arcpy.AddMessage("=" * 60)
            return final_mosaic

        except Exception as e:
            arcpy.AddError(f"Tool 01 failed: {e}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None

        finally:
            if locals().get("preserve_scratch") and scratch_dir:
                arcpy.AddMessage(
                    f"  Scratch preserved at: {scratch_dir}\n"
                    "  Re-run with the same Output Mosaic Name to resume "
                    "from completed scenes."
                )
            else:
                _cleanup_scratch_folder(scratch_dir)
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    # ------------------------------------------------------------------
    # Archive reading via GDAL Virtual File System (no extraction)
    # ------------------------------------------------------------------
    #
    # Copernicus L2A products ship as ~1 GB `.zip` archives. The previous
    # implementation extracted each one into a sibling `.SAFE/` folder
    # before processing — at the scale we work (hundreds of scenes)
    # the disk doubling is prohibitive.
    #
    # The new approach reads JP2 bands and MTD_MSIL2A.xml directly from
    # inside the zip via GDAL's `/vsizip/{path}/{member}` virtual file
    # system. arcpy.Raster opens these strings through the GDAL JP2
    # driver; XML is read in-memory with the stdlib `zipfile` module.
    # No extraction step, no scratch SAFE folders, no cleanup.

    @staticmethod
    def _list_zip_members(zip_path):
        """List file-only members of a zip archive (skip directories).

        Returns the member list, or None if the archive can't be opened
        (corrupt download, empty file, etc.).
        """
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                return [n for n in zf.namelist() if not n.endswith("/")]
        except (zipfile.BadZipFile, OSError) as e:
            arcpy.AddWarning(f"  Cannot open zip {zip_path}: {e}")
            return None

    @staticmethod
    def _read_safe_xml_from_zip(zip_path, xml_basename):
        """Return the bytes of `xml_basename` (e.g., 'MTD_MSIL2A.xml') from
        anywhere inside the zip, or None if absent / unreadable.

        Match by basename so the Copernicus on-disk layout drift (different
        baselines occasionally nest XML differently) is tolerated.
        """
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if name.rsplit("/", 1)[-1] == xml_basename:
                        with zf.open(name) as fobj:
                            return fobj.read()
        except (zipfile.BadZipFile, OSError) as e:
            arcpy.AddWarning(f"  Cannot read {xml_basename} from {zip_path}: {e}")
        return None

    @staticmethod
    def _locate_band_files_vsi(zip_path, member_names):
        """Build GDAL VSI band paths from a Copernicus L2A zip's member list.

        Mirrors the shape of `_locate_band_files` (SAFE-folder variant):
        returns `{band_name: vsi_path}` for the 10 stack bands + SCL,
        each path of the form
        `/vsizip/{zip_path}/{SAFE}/GRANULE/L2A_*/IMG_DATA/R{10,20}m/...jp2`.
        Returns None if no recognisable bands are found.

        Resolution is inferred from the filename suffix (`_10m.jp2` vs
        `_20m.jp2`), not the folder name — that lets us tolerate the
        occasional Copernicus zip that ships without the R10m/R20m
        subfolders.
        """
        base = f"/vsizip/{zip_path.replace(os.sep, '/')}"
        out = {}
        for member in member_names:
            basename = member.rsplit("/", 1)[-1]
            matched = False
            for band in ("B02", "B03", "B04", "B08"):
                if basename.endswith(f"_{band}_10m.jp2"):
                    out[band] = f"{base}/{member}"
                    matched = True
                    break
            if matched:
                continue
            for band in ("B05", "B06", "B07", "B8A", "B11", "B12"):
                if basename.endswith(f"_{band}_20m.jp2"):
                    out[band] = f"{base}/{member}"
                    matched = True
                    break
            if matched:
                continue
            # B01 (Coastal Aerosol) and B09 (Water Vapour) only exist
            # in the 60m product. Pulled from R60m so the output stack
            # preserves native S2 band parity with the L2A user guide.
            for band in ("B01", "B09"):
                if basename.endswith(f"_{band}_60m.jp2"):
                    out[band] = f"{base}/{member}"
                    matched = True
                    break
            if matched:
                continue
            if basename.endswith("_SCL_20m.jp2"):
                out["SCL"] = f"{base}/{member}"
        return out if out else None

    # ------------------------------------------------------------------
    # SAFE discovery + metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _find_safe_scenes(data_folder):
        """Return a list of scene sources discovered in data_folder.

        Two source types, both treated uniformly downstream:

          * `(path, 'zip')` — a Copernicus `.zip` archive sitting at the
            top of data_folder. Bands and metadata are read via GDAL VSI
            (`/vsizip/...`) — no extraction.
          * `(path, 'safe')` — an already-extracted `.SAFE` folder
            (kept for users who pre-extract).

        A zip wins over an extracted twin with the same basename — the
        zip is always a complete product, while a `.SAFE/` next to it
        could be a stale partial extract from a previous run.
        """
        if not data_folder or not os.path.isdir(data_folder):
            return []
        try:
            entries = os.listdir(data_folder)
        except OSError as e:
            arcpy.AddWarning(f"  Cannot list data folder: {e}")
            return []

        sources = []
        seen_safe_basenames = set()

        # Pass 1: zip archives.
        for e in sorted(entries):
            if not e.lower().endswith(".zip"):
                continue
            # Copernicus convention: "<safe_basename>.zip" where
            # safe_basename ends with ".SAFE". Tolerate the rare cases
            # where the .zip wraps a SAFE differently named — the
            # metadata reader handles both.
            safe_basename = e[:-4]
            sources.append((os.path.join(data_folder, e), "zip"))
            seen_safe_basenames.add(safe_basename)

        # Pass 2: extracted SAFE folders not already covered by a zip.
        for e in sorted(entries):
            if not e.endswith(".SAFE"):
                continue
            if e in seen_safe_basenames:
                continue
            full = os.path.join(data_folder, e)
            if os.path.isdir(full):
                sources.append((full, "safe"))

        return sources

    _SAFE_NAME_RE = re.compile(
        # Note the captured tile group includes the "T" prefix so callers
        # get the canonical MGRS tile ID (e.g., "T29SQB", not "29SQB").
        # S2A (launched 2015), S2B (2017), S2C (2024 — adds C to the
        # platform code). Future S2D would need extending; for now S2[ABC]
        # covers everything Copernicus is currently flying.
        r"^S2[ABC]_MSIL2A_(\d{8})T(\d{6})_N\d{4}_R\d{3}_(T\w{5})_",
    )

    @classmethod
    def _parse_safe_metadata(cls, source_path, source_kind="safe"):
        """Extract acquisition date, tile ID, and cloud cover for a scene.

        `source_kind` selects how MTD_MSIL2A.xml is read:
          * 'safe' — disk path to an extracted `.SAFE/` folder
          * 'zip'  — disk path to a Copernicus `.zip` archive (XML is
                     read in-memory via the stdlib `zipfile`).

        Returns dict with keys: date_acquired (date), tile_id, cloud_cover,
        product_uri. Falls back to filename parsing if XML is missing or
        unparseable (only `cloud_cover` is XML-derived). Returns None if
        even the filename doesn't match the canonical S2 L2A pattern.
        """
        raw_basename = os.path.basename(source_path.rstrip(os.sep))
        # For zip we want the SAFE basename (strip .zip).
        if source_kind == "zip" and raw_basename.lower().endswith(".zip"):
            safe_basename = raw_basename[:-4]
        else:
            safe_basename = raw_basename

        m = cls._SAFE_NAME_RE.match(safe_basename)
        if not m:
            return None
        date_str, _, tile_id = m.groups()
        try:
            acquired = datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            return None

        meta = {
            "date_acquired": acquired,
            "tile_id": tile_id,
            "cloud_cover": None,
            "product_uri": safe_basename,
        }

        # Read MTD_MSIL2A.xml. Namespaces drift across baselines, so we
        # match tag names by local name only and don't fail if the tag
        # is missing.
        xml_bytes = None
        if source_kind == "zip":
            xml_bytes = cls._read_safe_xml_from_zip(source_path, "MTD_MSIL2A.xml")
        else:
            mtd_path = os.path.join(source_path, "MTD_MSIL2A.xml")
            if os.path.isfile(mtd_path):
                try:
                    with open(mtd_path, "rb") as fh:
                        xml_bytes = fh.read()
                except OSError:
                    xml_bytes = None

        if xml_bytes:
            try:
                root = ET.fromstring(xml_bytes)
                for elem in root.iter():
                    tag = elem.tag.split("}", 1)[-1] if "}" in elem.tag else elem.tag
                    if tag in ("Cloud_Coverage_Assessment", "CLOUDY_PIXEL_PERCENTAGE"):
                        try:
                            meta["cloud_cover"] = float(elem.text)
                        except (TypeError, ValueError):
                            pass
                    # PRODUCT_URI inside the XML is the canonical SAFE name
                    # without the Windows "(1)" rename suffix that browsers
                    # apply to duplicate downloads. Prefer this over the
                    # filename whenever it's present.
                    elif tag == "PRODUCT_URI" and elem.text:
                        canonical = elem.text.strip()
                        if canonical.lower().endswith(".safe"):
                            canonical = canonical[:-5]
                        if canonical:
                            meta["product_uri"] = canonical
            except ET.ParseError:
                pass

        # Defensive sanitisation: arcpy.management.Resample rejects spaces
        # and parentheses in output dataset names (ERROR 000354). If the
        # canonical PRODUCT_URI wasn't available, strip the offending
        # characters from the filename-derived id so the scratch paths we
        # build downstream don't trip the validator.
        meta["product_uri"] = _sanitize_arcpy_name(meta["product_uri"])
        return meta

    # ------------------------------------------------------------------
    # Per-scene processing
    # ------------------------------------------------------------------

    @classmethod
    def _locate_band_files(cls, source_path, source_kind="safe"):
        """Find the JP2 file (or VSI path) for each band + SCL.

        `source_kind`:
          * 'safe' — disk paths under {source_path}/GRANULE/L2A_*/IMG_DATA/...
          * 'zip'  — `/vsizip/...` paths into the Copernicus archive

        Returns dict {band_name: path} including all 10 stack bands and
        the SCL. Returns None on missing GRANULE structure or an
        unreadable zip.
        """
        if source_kind == "zip":
            members = cls._list_zip_members(source_path)
            if members is None:
                return None
            return cls._locate_band_files_vsi(source_path, members)

        # Extracted SAFE folder.
        granule_dirs = glob.glob(os.path.join(source_path, "GRANULE", "L2A_*"))
        if not granule_dirs:
            return None
        img_data = os.path.join(granule_dirs[0], "IMG_DATA")
        r10 = os.path.join(img_data, "R10m")
        r20 = os.path.join(img_data, "R20m")
        r60 = os.path.join(img_data, "R60m")

        out = {}
        for band in ("B02", "B03", "B04", "B08"):
            matches = glob.glob(os.path.join(r10, f"*_{band}_10m.jp2"))
            if matches:
                out[band] = matches[0]
        for band in ("B05", "B06", "B07", "B8A", "B11", "B12"):
            matches = glob.glob(os.path.join(r20, f"*_{band}_20m.jp2"))
            if matches:
                out[band] = matches[0]
        # B01 (Coastal Aerosol) and B09 (Water Vapour) — 60m natives.
        for band in ("B01", "B09"):
            matches = glob.glob(os.path.join(r60, f"*_{band}_60m.jp2"))
            if matches:
                out[band] = matches[0]
        scl = glob.glob(os.path.join(r20, "*_SCL_20m.jp2"))
        if scl:
            out["SCL"] = scl[0]
        return out

    def _process_scene(self, scene, scratch_dir,
                       scl_classes=None, buffer_pixels=0):
        """Apply SCL mask + scale + resample-to-10m + stack into a single
        12-band float32 raster. Returns the saved raster path or None.

        Reads bands from either an extracted `.SAFE/` folder (disk JP2)
        or a Copernicus `.zip` archive via GDAL VSI (`/vsizip/...`).
        Either way, the per-band resample step writes a 10 m GeoTIFF to
        `scratch_dir`, so the (relatively expensive) JP2 decode is paid
        exactly once per band — downstream operations (mask, scale,
        composite, GeometricMedian) read from the cheap scratch GeoTIFFs.

        Args:
            scl_classes: Tuple of SCL class integers to mask. Defaults
                to the Aggressive preset (saturated + cloud shadow +
                unclassified + medium/high cloud + cirrus + snow/ice).
            buffer_pixels: Dilate the cloud mask by N pixels via
                FocalStatistics MAXIMUM. Catches the 1-2 pixel halo
                around clouds that Sen2Cor misclassifies as Vegetation
                or Bare-soil. 0 = no dilation.
        """
        if scl_classes is None:
            scl_classes = _S2_SCL_CLOUD_CLASSES  # back-compat default

        source_path = scene["path"]
        source_kind = scene.get("source_kind", "safe")
        scene_id = scene["metadata"]["product_uri"]
        bands = self._locate_band_files(source_path, source_kind)
        if not bands or "SCL" not in bands:
            arcpy.AddWarning("      Missing SCL or band data; scene skipped.")
            return None
        if not all(b in bands for b in _S2_STACK_ORDER):
            arcpy.AddWarning("      Missing one or more required bands; scene skipped.")
            return None

        # Build the cloud mask from SCL resampled to 10m (NEAREST so we
        # don't blur the class boundaries).
        scl_10m_path = os.path.join(scratch_dir, f"{scene_id}_SCL_10m.tif")
        arcpy.management.Resample(bands["SCL"], scl_10m_path, 10, "NEAREST")
        scl_10m = arcpy.sa.Raster(scl_10m_path)
        cloud_expr = None
        for klass in scl_classes:
            term = (scl_10m == klass)
            cloud_expr = term if cloud_expr is None else (cloud_expr | term)

        # Optional buffer: dilate the SCL cloud mask so the 1-2 pixel
        # halo Sen2Cor misclassifies as Vegetation/Bare-soil also gets
        # masked. Implemented via FocalStatistics MAXIMUM over a
        # circular neighbourhood; the binary mask treats any non-zero
        # focal max as "near cloud" → mask.
        if buffer_pixels and buffer_pixels > 0:
            buffered_path = os.path.join(
                scratch_dir, f"{scene_id}_cloud_buffer.tif"
            )
            buffered = arcpy.sa.FocalStatistics(
                cloud_expr,
                arcpy.sa.NbrCircle(buffer_pixels, "CELL"),
                "MAXIMUM",
                "DATA",
            )
            buffered.save(buffered_path)
            cloud_expr = arcpy.sa.Raster(buffered_path) > 0

        # Process each band: resample if 20m or 60m, scale to reflectance,
        # apply cloud mask, save. ``SetNull(< 0)`` after Resample
        # neutralises the float32 NoData sentinel (~-3.4e+38) that
        # BILINEAR can introduce near scene edges — without it the
        # sentinel flows through ``Float() * scale`` and contaminates
        # downstream statistics. The aggressive upsample on the 60m
        # bands (B01/B09) is the main beneficiary of this guard.
        masked_paths = []
        for band in _S2_STACK_ORDER:
            band_src = bands[band]
            if band not in _S2_NATIVE_10M:
                # 20m or 60m → 10m via BILINEAR (continuous reflectance).
                resampled_path = os.path.join(
                    scratch_dir, f"{scene_id}_{band}_10m.tif"
                )
                arcpy.management.Resample(band_src, resampled_path, 10, "BILINEAR")
                resampled_raster = arcpy.sa.Raster(resampled_path)
                band_raster = arcpy.sa.SetNull(
                    resampled_raster < 0, resampled_raster,
                )
            else:
                band_raster = arcpy.sa.Raster(band_src)

            reflectance = Float(band_raster) * _S2_REFLECTANCE_SCALE
            masked = arcpy.sa.SetNull(cloud_expr, reflectance)
            out_path = os.path.join(scratch_dir, f"{scene_id}_{band}_masked.tif")
            masked.save(out_path)
            masked_paths.append(out_path)

        # Composite into a single 12-band raster (B01..B12 minus B10).
        stacked_path = os.path.join(scratch_dir, f"{scene_id}_stack.tif")
        arcpy.management.CompositeBands(masked_paths, stacked_path)

        # Resume sentinel — written only after CompositeBands returns,
        # so its presence guarantees the stack file is complete. A failure
        # mid-CompositeBands leaves the stack without a marker, which the
        # Phase 3 resume scan treats as partial and rebuilds.
        try:
            with open(stacked_path + ".complete", "w", encoding="utf-8") as fh:
                fh.write(datetime.now().isoformat(timespec="seconds") + "\n")
        except OSError:
            pass  # Non-fatal — without the marker the scene rebuilds next run

        # Per-scene scratch cleanup: drop the SCL extract, cloud buffer,
        # the eight per-band 10m resamples, and the twelve per-band
        # masked rasters. Keeps the final stack + resume marker. Bounds
        # scratch growth so NTFS directory ops + GDAL handle caches
        # stay cheap as the Phase 3 loop runs.
        _cleanup_per_scene_intermediates(
            scratch_dir, scene_id,
            keep_basenames=(
                f"{scene_id}_stack.tif",
                f"{scene_id}_stack.tif.complete",
            ),
        )

        return stacked_path

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    @staticmethod
    def _write_provenance_csv(output_raster_path, scenes_used):
        if not output_raster_path or not scenes_used:
            return
        try:
            csv_path = _sidecar_path_for_raster(output_raster_path, "_provenance.csv")
            now_iso = datetime.now().isoformat(timespec="seconds")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "scene_id", "sensor", "acquisition_datetime",
                    "tile_id", "cloud_cover_pct", "input_path",
                    "processing_baseline", "toolbox_version",
                    "processing_datetime",
                ])
                for scene in scenes_used:
                    meta = scene.get("metadata", {}) or {}
                    p = scene.get("path", "") or ""
                    scene_id = meta.get("product_uri") or os.path.basename(p.rstrip(os.sep))
                    sensor_tag = "Sentinel-2A" if "S2A_" in scene_id else (
                        "Sentinel-2B" if "S2B_" in scene_id else "Sentinel-2"
                    )
                    date_acq = meta.get("date_acquired", "")
                    if hasattr(date_acq, "isoformat"):
                        date_acq = date_acq.isoformat()
                    writer.writerow([
                        scene_id,
                        sensor_tag,
                        date_acq,
                        meta.get("tile_id", ""),
                        meta.get("cloud_cover", "") if meta.get("cloud_cover") is not None else "",
                        p,
                        "L2A",
                        TOOLBOX_VERSION,
                        now_iso,
                    ])
            arcpy.AddMessage(f"Provenance CSV: {csv_path}")
        except OSError as e:
            arcpy.AddWarning(f"Failed to write provenance CSV: {e}")

    # ------------------------------------------------------------------
    # Temporal-filter helpers (duplicated from LandsatMosaic; shared
    # extraction is a refactor for Phase 6 once all tools have settled).
    # ------------------------------------------------------------------

    @staticmethod
    def _seasonal_pattern_for_region(region):
        if region in (
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
        ):
            return "temperate"
        if "Cape Verde" in (region or ""):
            return "cape_verde"
        if region == "Angola":
            return "angola"
        if region == "Mozambique":
            return "mozambique"
        return "temperate"

    @staticmethod
    def _create_temporal_filter(time_type, year, month, season):
        return {
            "type": time_type,
            "year": year,
            "month": month,
            "season": season,
        }

    def _scene_passes_filter(self, meta, temporal_filter, seasonal_pattern):
        d = meta.get("date_acquired")
        if d is None:
            return False
        ftype = temporal_filter.get("type")
        if ftype == "all_images":
            return True
        try:
            if ftype == "year_month":
                return (
                    d.year == temporal_filter["year"]
                    and d.month == temporal_filter["month"]
                )
            if ftype == "month_all_years":
                return d.month == temporal_filter["month"]
            if ftype == "season_in_year":
                if d.year != temporal_filter["year"]:
                    return False
                return self._is_in_season(d, seasonal_pattern, temporal_filter["season"])
            if ftype == "season_all_years":
                return self._is_in_season(d, seasonal_pattern, temporal_filter["season"])
        except (TypeError, KeyError):
            return False
        return False

    @staticmethod
    def _is_in_season(date, pattern, season):
        if season is None:
            return False
        season_months = {
            "temperate": {
                "spring": [3, 4, 5],
                "summer": [6, 7, 8],
                "autumn": [9, 10, 11],
                "winter": [12, 1, 2],
            },
            "angola": {
                "rainy": [11, 12, 1, 2, 3, 4],
                "dry": [5, 6, 7, 8, 9, 10],
                "rainy_peak": [1, 2, 3],
                "dry_peak": [6, 7, 8],
            },
            "cape_verde": {
                "dry": [12, 1, 2, 3, 4, 5, 6],
                "rainy": [8, 9, 10],
                "transition_dry_wet": [7],
                "transition_wet_dry": [11],
            },
            "mozambique": {
                "rainy": [10, 11, 12, 1, 2, 3],
                "dry": [4, 5, 6, 7, 8, 9],
                "rainy_peak": [12, 1, 2],
                "dry_peak": [7, 8, 9],
            },
        }
        try:
            return date.month in season_months[pattern][season.lower()]
        except KeyError:
            return False


# ---------------------------------------------------------------------------
# Tool 03 — ASTER L2 Mosaic
# ---------------------------------------------------------------------------

# ASTER AST_07XT V004 conventions:
_ASTER_REFLECTANCE_SCALE = 0.001
_ASTER_VNIR_BANDS = ["B01", "B02", "B03N"]                # 15m native
_ASTER_SWIR_BANDS = ["B04", "B05", "B06", "B07", "B08", "B09"]  # 30m → resample to 15m
_ASTER_STACK_ORDER = _ASTER_VNIR_BANDS + _ASTER_SWIR_BANDS  # 9-band stack
_ASTER_NATIVE_15M = set(_ASTER_VNIR_BANDS)
_ASTER_QA_NAMES = ["VNIR_QA_DataPlane", "SWIR_QA_DataPlane"]

# Per-scene processing modes — the tool can build either a full 9-band stack
# (only viable for pre-Apr-2008 scenes that still carry SWIR) or a 3-band
# VNIR-only stack (works for the full archive, since VNIR has never failed).
_ASTER_MODE_FULL = "vnir_swir"
_ASTER_MODE_VNIR = "vnir_only"

# Filename pattern for AST_07XT V004 TIFF exports:
#   AST_07XT_<17-char-sceneID>_<14-char-procDT>_SRF_<VNIR|SWIR>_<band-or-QA>.tif
# The sceneID encodes the acquisition date as <3-digit-pass><MM><DD><YYYY><HHMMSS>
# (US-style MMDDYYYY, then HHMMSS — both UTC).
_ASTER_TIFF_RE = re.compile(
    r"^AST_07XT_"
    r"(?P<pass>\d{3})(?P<MM>\d{2})(?P<DD>\d{2})(?P<YYYY>\d{4})(?P<HMS>\d{6})_"
    r"(?P<proc>\d{14})_SRF_"
    r"(?P<group>VNIR|SWIR)_"
    r"(?P<band>B0[1-9]N?|QA_DataPlane2?)\.tif$",
    re.IGNORECASE,
)

# AST_08 V004 (Surface Kinetic Temperature) — optional companion
# product, joined to AST_07XT by scene ID. The SKT COG TIFF naming is
#   AST_08_<17-char-sceneID>_<14-char-procDT>_SKT.tif
# and the matching HDF-EOS5 archive is
#   AST_08_<17-char-sceneID>_<14-char-procDT>.hdf
# (with subdataset "SurfaceKineticTemperature"). 90 m native; LP DAAC
# COG exports come already in Kelvin so no scale factor is applied
# downstream. We accept the SKT band only; the AST_08 QA Data Plane
# is not consumed.
_AST08_TIFF_RE = re.compile(
    r"^AST_08_"
    r"(?P<pass>\d{3})(?P<MM>\d{2})(?P<DD>\d{2})(?P<YYYY>\d{4})(?P<HMS>\d{6})_"
    r"(?P<proc>\d{14})_SKT\.tif$",
    re.IGNORECASE,
)
_AST08_HDF_RE = re.compile(
    r"^AST_08_(?P<pass>\d{3})(?P<MM>\d{2})(?P<DD>\d{2})(?P<YYYY>\d{4})"
    r"(?P<HMS>\d{6})_(?P<proc>\d{14})\.hdf$",
    re.IGNORECASE,
)
# HDF-EOS5 SDS names accepted for the surface kinetic temperature
# layer. Modern V004 emits "SurfaceKineticTemperature"; older revisions
# (V003 / legacy SwathName 'TIR') used "KineticTemperature". We accept
# either to keep older archives readable.
_AST08_BT_SDS_NAMES = ("SurfaceKineticTemperature", "KineticTemperature")


def _aster_bt_kelvin_from_path(path, scratch_dir, target_cellsize=None, scene_id=""):
    """Load AST_08 Surface Kinetic Temperature as a lazy Kelvin Raster.

    Accepts a path to either an AST_08 GeoTIFF or HDF (the BT subdataset
    is extracted to scratch via ``AsterMosaic._extract_ast08_bt_tiff``
    when the path ends in ``.hdf``). Optionally resamples to
    ``target_cellsize`` metres via BILINEAR; default ``None`` keeps the
    native 90 m grid (the right choice for thermal statistics; the
    AsterMosaic cloud test passes 15 to match the SR grid). Applies
    ``_ASTER_TIR_SCALE`` (DN x 0.1 -> Kelvin) and screens NoData via
    ``SetNull`` below ``_ASTER_TIR_VALID_K_FLOOR``.

    Returns a lazy ``arcpy.sa.Raster`` in Kelvin. Raises on failure;
    callers attach scene-level context to their warnings.
    """
    if path.lower().endswith(".hdf"):
        bt_src = AsterMosaic._extract_ast08_bt_tiff(path, scratch_dir)
        if bt_src is None:
            raise RuntimeError("AST_08 HDF extraction failed")
    else:
        bt_src = path
    if target_cellsize is not None:
        resampled = os.path.join(
            scratch_dir,
            f"{scene_id or 'ast08'}_BT_{int(target_cellsize)}m.tif",
        )
        arcpy.management.Resample(bt_src, resampled, target_cellsize, "BILINEAR")
        bt_dn = Float(arcpy.sa.Raster(resampled))
    else:
        bt_dn = Float(arcpy.sa.Raster(bt_src))
    bt_kelvin = bt_dn * _ASTER_TIR_SCALE
    return arcpy.sa.SetNull(bt_kelvin < _ASTER_TIR_VALID_K_FLOOR, bt_kelvin)


class AsterMosaic(object):
    """Tool 03 — ASTER AST_07XT V004 mineral-mapping mosaic.

    Accepts a folder of ASTER L2 Surface Reflectance products, either as:
      a) Per-band TIFFs following the standard naming convention
         (`AST_07XT_<sceneID>_<procDT>_SRF_<VNIR|SWIR>_<band>.tif`) —
         3 VNIR + 6 SWIR + 2 QA per scene.
      b) HDF-EOS `.hdf` archives (best-effort via osgeo.gdal — extracted
         to TIFFs in a scratch folder before processing).

    Optionally also accepts paired AST_08 (Surface Kinetic Temperature,
    V004) files (`AST_08_<sceneID>_<procDT>_SKT.tif` or the equivalent
    HDF) — either co-located in the main data folder, or in a separate
    folder pointed to by the optional "ASTER Thermal Data Folder"
    parameter (the natural LP DAAC by-product download layout). When
    present, the thermal channel is folded into the per-scene cloud test
    — pixels colder than ~280 K (below typical mid-latitude land BT but
    above warm maritime stratus on Atlantic island summits) are flagged
    as cloud regardless of VIS/SWIR brightness, which catches thin
    cirrus and warm low cloud and reduces false positives over warm
    bare ground.

    Per-scene processing:
      1. Load 3 VNIR bands at native 15m + 6 SWIR bands resampled to 15m
         (BILINEAR).
      2. Apply scale factor 0.001 to convert DN → surface reflectance [0, 1].
      3. Apply QA cloud mask from the QA Data Plane layers (best-effort
         bit decoding — exact layout per ASTER User Handbook V004).
      4. Build a multi-spectral cloud mask
         ``(B02 > 0.45 AND B04 > 0.25) OR (BT < 280 K)`` — Hulley & Hook
         (2008)-style ACCA-on-ASTER with a mid-latitude-adjusted thermal
         threshold tuned for warm maritime stratus. The thermal term is
         dropped when no AST_08 is paired; the SWIR term is dropped for
         VNIR-only scenes.
      5. Stack into a 9-band float32 raster (3-band for VNIR-only mode).

    Mosaicking: Esri arcpy.ia.GeometricMedian across the cloud-masked
    stack, then optional AOI clip and provenance CSV.
    """

    def __init__(self):
        self.label = "03 — ASTER L2 Mosaic"
        self.description = (
            "Build a mineral-mapping mosaic from ASTER AST_07XT V004 "
            "Surface Reflectance scenes (VNIR + crosstalk-corrected SWIR). "
            "Accepts per-band TIFFs (the common LP DAAC export) or HDF-EOS "
            "archives. SWIR (30m) is resampled to 15m to match VNIR. Cloud "
            "handling combines the QA Data Plane non-zero flag with a "
            "per-scene multi-spectral test on B02 (red) + B04 (SWIR1); "
            "paired AST_08 Surface Kinetic Temperature scenes (either "
            "co-located, or supplied through the optional thermal folder "
            "parameter) add a brightness-temperature channel that catches "
            "thin cirrus and warm bare ground that VIS/SWIR alone would "
            "miss. The temporal stack is reduced to a geometric median. A "
            "provenance CSV is written alongside the output."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # GP parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        gdb = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="gdb_path",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        gdb.filter.list = ["Local Database"]

        mosaic_name = arcpy.Parameter(
            displayName="Output Mosaic Name",
            name="mosaic_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        data_folder = arcpy.Parameter(
            displayName="ASTER Data Folder (AST_07XT)",
            name="data_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )

        thermal_folder = arcpy.Parameter(
            displayName=(
                "Optional ASTER Thermal Data Folder (AST_08, Surface "
                "Kinetic Temperature). Used by the optional thermal "
                "cloud test in Advanced Options (default OFF, so this "
                "field is hidden by default). A separate LST temporal "
                "statistics tool, planned but not yet shipped, will "
                "have its own thermal-folder parameter."
            ),
            name="thermal_folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input",
        )
        # Initially hidden; updateParameters reveals it when
        # use_ast08_thermal is checked. Keeps the default dialog clean.
        thermal_folder.enabled = False

        region = arcpy.Parameter(
            displayName="Region",
            name="region",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        region.filter.list = [
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
            "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)",
            "Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)",
            "Angola",
            "Mozambique",
        ]

        time_type = arcpy.Parameter(
            displayName="Time Filter Type",
            name="time_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        # `all_images` (default) processes every scene that parsed,
        # matching the behaviour of LandsatMosaic. The previous default
        # of `year_month` silently required year+month to be filled in
        # for any scene to pass the filter — a usability footgun that
        # produced an immediate "No scenes match" crash when the user
        # accepted defaults.
        time_type.filter.list = [
            "all_images",
            "year_month", "month_all_years",
            "season_in_year", "season_all_years",
        ]
        time_type.value = "all_images"

        year = arcpy.Parameter(
            displayName="Year (when applicable)",
            name="year",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        month = arcpy.Parameter(
            displayName="Month (when applicable, 1-12)",
            name="month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        season = arcpy.Parameter(
            displayName="Season (when applicable)",
            name="season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )

        use_qa_planes = arcpy.Parameter(
            displayName="Apply QA Data Plane Quality Mask",
            name="use_qa_planes",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        use_qa_planes.value = True

        mask_feature = arcpy.Parameter(
            displayName="Optional AOI Mask Feature (polygon)",
            name="mask_feature",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
        )
        mask_feature.filter.list = ["Polygon"]

        save_stats = arcpy.Parameter(
            displayName="Save Provenance CSV",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        save_stats.value = True

        preserve_scratch = arcpy.Parameter(
            displayName="Preserve Scratch & Resume on Re-run",
            name="preserve_scratch",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        preserve_scratch.value = False

        # Cloud-mask edge dilation. The per-scene reflectance + NDVI
        # test misses a 1-3 pixel halo around marine cloud (subpixel
        # mixing softens the edge). FocalStatistics MAXIMUM over a
        # circular neighbourhood pulls those pixels into the mask.
        # Mirrors the S2 tool's "Cloud Mask Buffer (pixels)" idiom.
        # Sits with the per-scene cloud-test family at the top of the
        # Advanced Options group; always active (no toggle gates it).
        cloud_buffer_px = arcpy.Parameter(
            displayName="Cloud Mask Buffer (pixels)",
            name="cloud_buffer_px",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        cloud_buffer_px.value = _ASTER_CLOUD_BUFFER_PX
        cloud_buffer_px.filter.type = "Range"
        cloud_buffer_px.filter.list = [0, 10]

        # AST_08 (Surface Kinetic Temperature) is produced by the TES
        # algorithm AFTER the operational L2 cloud mask is already
        # applied; over a cloud it is NoData or a corrupted retrieval,
        # so it fails on precisely the pixels a cloud test needs. The
        # warm-low-cloud / warm-land BT distributions also overlap, so
        # no scalar threshold separates them. The path is preserved
        # behind this opt-in switch (default OFF); prefer the temporal
        # cleaner below for the actual cloud removal. Sits BEFORE the
        # threshold field it gates so the dialog reads top-down.
        use_ast08_thermal = arcpy.Parameter(
            displayName=(
                "Use AST_08 thermal cloud test (NOT recommended). "
                "AST_08 is produced by the TES algorithm AFTER the "
                "operational cloud mask is applied, so over a cloud "
                "it is NoData or a corrupted retrieval. The Phase 4 "
                "temporal cleaner handles cloud removal. Enable ONLY "
                "for A/B comparison runs."
            ),
            name="use_ast08_thermal",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        use_ast08_thermal.value = False

        # Per-scene thermal cloud threshold. Greyed out by default
        # because ``use_ast08_thermal`` is OFF by default; updateParameters
        # toggles the .enabled state when the master switch changes.
        # 280 K catches warm maritime stratus on Atlantic island summits
        # (Faial caldera at 275-285 K). Raise to ~285-295 K for hot /
        # arid AOIs (Cape Verde, Angola) where surface BT runs warmer;
        # lower to ~255-265 K for very cold / high-altitude AOIs where
        # ground can be cool enough to false-flag.
        bt_threshold_k = arcpy.Parameter(
            displayName=(
                "Thermal Cloud Threshold (K). Default 280 K (Faial / "
                "mid-latitude maritime). Raise for hot AOIs "
                "(Cape Verde, Angola; ~285-295 K), lower for high-"
                "altitude AOIs (~255-265 K)."
            ),
            name="bt_threshold_k",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        bt_threshold_k.value = _ASTER_CLOUD_BT_MAX_K
        bt_threshold_k.enabled = False

        # Temporal outlier cleaner master toggle. Default flipped to OFF
        # in 2026-05-26 alongside the DL cloud-mask integration: with
        # OmniCloudMask (Phase 4) catching the persistent orographic
        # cloud over Faial that the Tmask-reduction layer cannot flag
        # (cloud is modal at those pixels, so robust-z does not see it
        # as outlier), the cleaner's net value drops. Kept as an opt-in
        # for archives where DL is unavailable or as a second line of
        # defence; reconsider default after V8 comparison evidence.
        enable_temporal_clean = arcpy.Parameter(
            displayName=(
                "Enable temporal outlier cleaner (default OFF since "
                "DL cloud masking became the primary path). Tmask-"
                "reduction layer; opt in for archives without DL cloud "
                "masks or to A/B against DL-only output."
            ),
            name="enable_temporal_clean",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        enable_temporal_clean.value = False

        temporal_k = arcpy.Parameter(
            displayName=(
                "Temporal cleaner: robust z-score threshold (k, in "
                "MAD-sigma units). Lower is more aggressive. Default "
                "2.5 matches the Tmask paper convention."
            ),
            name="temporal_k",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        temporal_k.value = _TMASK_K

        temporal_min_obs = arcpy.Parameter(
            displayName=(
                "Temporal cleaner: minimum valid observations per pixel "
                "to attempt cleaning. Pixels with fewer observations "
                "are left as-is (their cleaning would be unreliable) "
                "and are reported in the obs_count sidecar."
            ),
            name="temporal_min_obs",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        temporal_min_obs.value = _TMASK_MIN_OBS

        # Subprocess-per-batch perf workaround, same shape as the one
        # on Sentinel2Mosaic. Each batch of N scenes runs in a fresh
        # python.exe so accumulated arcpy / GDAL state is reclaimed
        # by OS guarantee at process exit.
        subprocess_batch_size = arcpy.Parameter(
            displayName=(
                "Subprocess Batch Size (advanced; controls the per-"
                "scene-loop memory cycling. Each batch of N scenes "
                "runs in a fresh python.exe subprocess to reclaim "
                "accumulated arcpy / GDAL state. Default 10 trades "
                "~10s arcpy reimport per batch for flat per-scene "
                "timing. Set 0 to disable subprocess batching (legacy "
                "single-process loop, for A/B comparison or debugging)."
            ),
            name="subprocess_batch_size",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        subprocess_batch_size.value = 10
        subprocess_batch_size.filter.type = "Range"
        subprocess_batch_size.filter.list = [0, 100]

        compositor = arcpy.Parameter(
            displayName=(
                "Compositor (advanced; the per-pixel reducer over the "
                "cleaned multi-scene stack. GeometricMedian (default) "
                "computes the L1-median across the multi-band spectral "
                "signature, preserving same-scene consistency across "
                "bands at the cost of opaque NoData handling. Per-band "
                "median computes each band's median independently via "
                "arcpy.sa.CellStatistics with explicit ignore_nodata "
                "semantics; a pixel's band-1 value can come from a "
                "different scene than its band-2 value. Use Per-band "
                "median as an A/B for diagnosing GeometricMedian "
                "artefacts on NoData-asymmetric inputs."
            ),
            name="compositor",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        compositor.filter.list = [
            "GeometricMedian (default)",
            "Per-band median",
        ]
        compositor.value = "GeometricMedian (default)"

        # DL cloud-mask family. ASTER AST_07XT ships without a cloud
        # mask (unlike S2 SCL / Landsat QA_PIXEL); OmniCloudMask
        # (Wright et al. 2025) is a sensor-agnostic R/G/NIR U-Net
        # ensemble whose outputs (0=Clear, 1=Thick, 2=Thin, 3=Shadow)
        # replace or augment the hand-rolled spectral cloud test.
        # Phase 4 runs inference per scene in the parent (GPU init
        # amortised); Phase 5 subprocess workers consume the cached
        # mask TIFFs. Default OFF so the toolbox still runs without
        # the optional PyTorch + omnicloudmask install.
        use_dl_cloud_mask = arcpy.Parameter(
            displayName=(
                "Use DL cloud mask via OmniCloudMask (default OFF). "
                "When ON, Phase 4 runs U-Net inference per scene "
                "(parent process, GPU when available) and Phase 5 "
                "consumes the cached masks alongside the existing "
                "spectral cloud test. Requires PyTorch (Esri Deep "
                "Learning Frameworks MSI) and omnicloudmask (pip / "
                "Pro Package Manager into a cloned arcgispro-py3 env)."
            ),
            name="use_dl_cloud_mask",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        use_dl_cloud_mask.value = False

        dl_mask_aggressiveness = arcpy.Parameter(
            displayName=(
                "DL mask aggressiveness (mirrors S2's Cloud Mask "
                "Aggressiveness). Aggressive drops classes "
                "{Thick, Thin, Shadow}; Moderate drops {Thick, "
                "Shadow} (keeps Thin to recover observations in "
                "haze-prone scenes); Conservative drops {Thick} only."
            ),
            name="dl_mask_aggressiveness",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        dl_mask_aggressiveness.filter.list = [
            "Aggressive (Thick + Thin + Shadow)",
            "Moderate (Thick + Shadow, keep Thin)",
            "Conservative (Thick only)",
        ]
        dl_mask_aggressiveness.value = "Aggressive (Thick + Thin + Shadow)"

        dl_mask_folder = arcpy.Parameter(
            displayName=(
                "DL cloud-mask cache folder (optional; defaults to a "
                "'_dl_cloud_masks/' sibling of the ASTER Data Folder "
                "so masks persist across mosaic re-runs). One uint8 "
                "TIFF per scene named '{scene_id}_cloudmask.tif'."
            ),
            name="dl_mask_folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )

        # Return order: master toggles BEFORE the values they gate so
        # the dialog reads top-down. Per-scene cloud-mask family (always
        # active) sits with the AST_08 thermal family (opt-in) above the
        # temporal-cleaner family. Resume safety stays first under
        # Advanced Options.
        return [
            gdb, mosaic_name, data_folder, thermal_folder,
            region, time_type, year, month, season,
            use_qa_planes, mask_feature, save_stats, preserve_scratch,
            cloud_buffer_px,
            use_ast08_thermal, bt_threshold_k,
            enable_temporal_clean, temporal_k, temporal_min_obs,
            subprocess_batch_size,
            compositor,
            use_dl_cloud_mask, dl_mask_aggressiveness, dl_mask_folder,
        ]

    def updateParameters(self, parameters):
        try:
            # Existing time-filter logic: time_type at index 5 gates
            # year (6), month (7), season (8).
            time_type = parameters[5]
            year = parameters[6]
            month = parameters[7]
            season = parameters[8]
            if time_type.valueAsText == "all_images":
                year.enabled = False; month.enabled = False; season.enabled = False
            elif time_type.valueAsText == "year_month":
                year.enabled = True; month.enabled = True; season.enabled = False
            elif time_type.valueAsText == "month_all_years":
                year.enabled = False; month.enabled = True; season.enabled = False
            elif time_type.valueAsText == "season_in_year":
                year.enabled = True; month.enabled = False; season.enabled = True
            elif time_type.valueAsText == "season_all_years":
                year.enabled = False; month.enabled = False; season.enabled = True

            # AST_08 thermal family gate: use_ast08_thermal (idx 14)
            # controls visibility of thermal_folder (idx 3) AND the BT
            # threshold (idx 15). Default OFF keeps the dialog clean.
            thermal_folder = parameters[3]
            use_ast08_thermal = parameters[14]
            bt_threshold_k = parameters[15]
            thermal_on = bool(use_ast08_thermal.value)
            thermal_folder.enabled = thermal_on
            bt_threshold_k.enabled = thermal_on

            # Temporal cleaner family gate: enable_temporal_clean
            # (idx 16) controls temporal_k (17) and temporal_min_obs
            # (18). Default OFF since DL cloud masking became primary.
            enable_temporal_clean = parameters[16]
            temporal_k = parameters[17]
            temporal_min_obs = parameters[18]
            cleaner_on = bool(enable_temporal_clean.value)
            temporal_k.enabled = cleaner_on
            temporal_min_obs.enabled = cleaner_on

            # DL cloud-mask family gate: use_dl_cloud_mask (idx 21)
            # controls dl_mask_aggressiveness (22) and dl_mask_folder
            # (23). Default OFF keeps the toolbox loadable without the
            # PyTorch + omnicloudmask dependency stack.
            use_dl_cloud_mask = parameters[21]
            dl_mask_aggressiveness = parameters[22]
            dl_mask_folder = parameters[23]
            dl_on = bool(use_dl_cloud_mask.value)
            dl_mask_aggressiveness.enabled = dl_on
            dl_mask_folder.enabled = dl_on
        except Exception:
            pass

    def updateMessages(self, parameters):
        """Surface a yellow warning whenever the AST_08 thermal cloud
        test is enabled. The toggle defaults OFF and is documented as
        NOT recommended; this nudge keeps the structural caveat
        visible in the dialog any time the user opts in, so an
        accidental check doesn't sail through to a multi-hour run."""
        try:
            use_ast08_thermal = parameters[14]
            if use_ast08_thermal.value:
                use_ast08_thermal.setWarningMessage(
                    "AST_08 is clear-sky-only by construction. Over a "
                    "cloud it is NoData or a corrupted retrieval, so "
                    "it fails on the very pixels a cloud test needs. "
                    "The Phase 4 temporal cleaner handles cloud "
                    "removal. Leave this OFF unless you are running "
                    "an explicit A/B comparison."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        scratch_dir = None
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                arcpy.AddError("Spatial Analyst extension is required.")
                return None
            arcpy.CheckOutExtension("Spatial")
            arcpy.env.overwriteOutput = True
            # Take control of cancellation so loop-boundary checks can
            # abort cleanly instead of Pro hard-killing the process.
            arcpy.env.autoCancelling = False

            gdb_path = parameters[0].valueAsText
            mosaic_name = parameters[1].valueAsText
            data_folder = parameters[2].valueAsText
            thermal_folder = parameters[3].valueAsText
            region = parameters[4].valueAsText
            time_type = parameters[5].valueAsText
            year = parameters[6].value
            month = parameters[7].value
            season = parameters[8].valueAsText
            use_qa = bool(parameters[9].value)
            mask_feature = parameters[10].valueAsText
            save_stats = bool(parameters[11].value)
            preserve_scratch = bool(parameters[12].value)
            # Advanced Options reads. Indices match the new return
            # order in getParameterInfo: cloud_buffer_px (13) is the
            # always-active per-scene edge dilation; use_ast08_thermal
            # (14) gates bt_threshold_k (15); enable_temporal_clean
            # (16) gates temporal_k (17) and temporal_min_obs (18).
            # bt_threshold_k stays float, not int, since 295.5 K etc.
            # are valid thresholds.
            cloud_buffer_px = (
                int(parameters[13].value)
                if len(parameters) > 13 and parameters[13].value is not None
                else _ASTER_CLOUD_BUFFER_PX
            )
            use_ast08_thermal = (
                bool(parameters[14].value)
                if len(parameters) > 14 and parameters[14].value is not None
                else False
            )
            bt_threshold_k = (
                float(parameters[15].value)
                if len(parameters) > 15 and parameters[15].value is not None
                else _ASTER_CLOUD_BT_MAX_K
            )
            # Affirmative phrasing: enable_temporal_clean defaults True
            # so the cleaner runs on default. Replaces the older
            # disable_temporal_clean (double negative); the call sites
            # below consume the affirmative value directly.
            enable_temporal_clean = (
                bool(parameters[16].value)
                if len(parameters) > 16 and parameters[16].value is not None
                else True
            )
            temporal_k = (
                float(parameters[17].value)
                if len(parameters) > 17 and parameters[17].value is not None
                else _TMASK_K
            )
            temporal_min_obs = (
                int(parameters[18].value)
                if len(parameters) > 18 and parameters[18].value is not None
                else _TMASK_MIN_OBS
            )
            subprocess_batch_size = (
                int(parameters[19].value)
                if len(parameters) > 19 and parameters[19].value is not None
                else 10
            )
            compositor = (
                parameters[20].valueAsText
                if len(parameters) > 20 and parameters[20].valueAsText
                else "GeometricMedian (default)"
            )
            use_dl_cloud_mask = (
                bool(parameters[21].value)
                if len(parameters) > 21 and parameters[21].value is not None
                else False
            )
            dl_mask_aggressiveness = (
                parameters[22].valueAsText
                if len(parameters) > 22 and parameters[22].valueAsText
                else "Aggressive (Thick + Thin + Shadow)"
            )
            dl_mask_folder_param = (
                parameters[23].valueAsText
                if len(parameters) > 23 else None
            )

            # ----------------------------------------------------------------
            # AOI-first scoping. See LandsatMosaic.execute() for the full
            # rationale. For ASTER the win includes: the SWIR 30m→15m
            # resample for six bands per scene (the heaviest per-scene
            # cost) now operates only on AOI pixels; the QA Data Plane
            # non-zero mask is only built over the AOI; the per-scene
            # multi-spectral cloud test (B02 reflectance + optional B04
            # reflectance + optional BT) operates on AOI-clipped rasters;
            # and the AST_08 thermal resample also runs only over the AOI.
            # ----------------------------------------------------------------
            # Header — one block of run context.
            arcpy.AddMessage("=" * 60)
            arcpy.AddMessage(f"ASTER L2 MOSAIC ({region})")
            arcpy.AddMessage("=" * 60)
            arcpy.AddMessage(f"  Output:     {gdb_path}\\{mosaic_name}")
            arcpy.AddMessage(f"  Source:     {data_folder}")
            if thermal_folder:
                if os.path.isdir(thermal_folder):
                    arcpy.AddMessage(f"  Thermal:    {thermal_folder}")
                else:
                    arcpy.AddWarning(
                        f"  Thermal:    {thermal_folder!r} NOT FOUND. "
                        "Falling back to scanning the main data folder for AST_08."
                    )
                    thermal_folder = None
            arcpy.AddMessage(
                f"  Options:    QA mask = {use_qa}, "
                f"per-scene cloud test = hardened VNIR "
                f"(brightness > {_ASTER_CLOUD_BRIGHT_MIN}, "
                f"flat tol < {_ASTER_CLOUD_FLAT_TOL}, "
                f"NDVI < {_ASTER_CLOUD_NDVI_MAX}; "
                f"+ SWIR confirmation B02>{_ASTER_CLOUD_B02_MIN} & "
                f"B04>{_ASTER_CLOUD_B04_MIN} on pre-failure scenes; "
                f"+ NDWI water guard > {_ASTER_WATER_NDWI_MIN}; "
                f"+ {cloud_buffer_px}px edge dilation"
                + (f"; + BT < {bt_threshold_k:.0f} K when AST_08 paired"
                   if use_ast08_thermal else "")
                + ")"
            )
            if use_ast08_thermal:
                arcpy.AddMessage(
                    "  AST_08:     thermal cloud test ENABLED. Note: "
                    "AST_08 is clear-sky-only by construction and may "
                    "give misleading results over cloud."
                )
            else:
                arcpy.AddMessage(
                    "  AST_08:     thermal cloud test disabled "
                    "(default; AST_08 is clear-sky-only)."
                )
            arcpy.AddMessage(f"  Compositor: {compositor}")
            if enable_temporal_clean:
                arcpy.AddMessage(
                    f"  Temporal:   outlier cleaner ON "
                    f"(k={temporal_k}, min_obs={temporal_min_obs})"
                )
            else:
                arcpy.AddMessage(
                    "  Temporal:   outlier cleaner DISABLED "
                    "(A/B comparison mode)."
                )

            if mask_feature and arcpy.Exists(mask_feature):
                arcpy.env.mask = mask_feature
                arcpy.env.extent = mask_feature
                arcpy.AddMessage(f"  AOI:        {mask_feature} (env.mask + env.extent active)")
            elif mask_feature:
                arcpy.AddWarning(
                    f"  AOI:        {mask_feature!r} NOT FOUND — running over full scene footprint"
                )

            scratch_dir = _make_mosaic_scratch_dir(
                gdb_path, "_genesis_aster_scratch", mosaic_name,
            )
            arcpy.AddMessage(f"  Scratch:    {scratch_dir}")
            if preserve_scratch:
                arcpy.AddMessage(
                    "  Resume:     Preserve Scratch is ON — completed "
                    "per-scene stacks from previous runs will be reused"
                )

            # Snap raster anchors every per-scene Resample to a single
            # 15m pixel grid covering the AOI. Without it, scenes from
            # different overpasses drift by a fraction of a cell on
            # save, which causes downstream alignment issues for the
            # GeometricMedian temporal reduction. Only set when an AOI
            # is active — without env.extent the constant-raster anchor
            # has nothing to cover.
            if arcpy.env.mask:
                try:
                    snap_anchor = os.path.join(scratch_dir, "_snap_anchor.tif")
                    if not os.path.exists(snap_anchor):
                        prev_cell = arcpy.env.cellSize
                        arcpy.env.cellSize = 15
                        arcpy.sa.CreateConstantRaster(
                            1, "INTEGER", 15
                        ).save(snap_anchor)
                        arcpy.env.cellSize = prev_cell
                    arcpy.env.snapRaster = snap_anchor
                    arcpy.AddMessage(
                        "  Snap:       15m grid anchored to AOI "
                        "(eliminates per-scene cell drift)"
                    )
                except Exception as e:
                    arcpy.AddWarning(
                        f"  Snap raster setup failed ({e}); per-scene "
                        "stacks may not share a single cell origin"
                    )

            # Phase 1 — discover scenes (TIFF and HDF)
            arcpy.AddMessage("\n▶ Phase 1 — Scene discovery")
            scenes = self._find_aster_scenes(data_folder, thermal_folder)
            if not scenes:
                arcpy.AddError("  ✗ No ASTER scenes found.")
                return None
            arcpy.AddMessage(f"  ✓ {len(scenes)} scene(s)")

            # Phase 2 — temporal filter
            arcpy.AddMessage("\n▶ Phase 2 — Temporal filter")
            seasonal_pattern = self._seasonal_pattern_for_region(region)
            temporal_filter = self._create_temporal_filter(time_type, year, month, season)
            kept_scenes = [
                s for s in scenes
                if s.get("metadata") and self._scene_passes_filter(
                    s["metadata"], temporal_filter, seasonal_pattern,
                )
            ]
            if not kept_scenes:
                arcpy.AddError("  ✗ No scenes match the temporal filter.")
                return None
            arcpy.AddMessage(
                f"  ✓ {len(kept_scenes)}/{len(scenes)} scenes kept "
                f"({temporal_filter.get('type', 'all_images')})"
            )

            # Classify scenes by available SWIR coverage. Scenes acquired
            # before April 2008 carry the full nine bands; post-failure
            # scenes are VNIR-only. We always emit both products so the
            # full archive contributes to vegetation/water indices while
            # the pre-failure subset still yields a mineral-grade 9-band
            # mosaic.
            full_scenes = [s for s in kept_scenes if self._has_swir_bands(s)]
            vnir_only_count = len(kept_scenes) - len(full_scenes)
            arcpy.AddMessage(
                f"  Scene split: {len(full_scenes)} VNIR+SWIR "
                f"(pre-Apr-2008) + {vnir_only_count} VNIR-only "
                f"(post-Apr-2008 SWIR failure)"
            )
            paired = sum(1 for s in kept_scenes if s.get("thermal"))
            if paired:
                arcpy.AddMessage(
                    f"  Thermal:    {paired}/{len(kept_scenes)} scenes paired with "
                    f"AST_08 — thermal cloud channel active for those scenes"
                )
            else:
                arcpy.AddMessage(
                    "  Thermal:    no AST_08 (Surface Kinetic Temperature) "
                    "files found alongside AST_07XT — "
                    "cloud test runs on VIS/SWIR only"
                )

            # ----------------------------------------------------------------
            # Phase 3 — AOI mask + scene intersection filter
            # ----------------------------------------------------------------
            # Promoted from the silent env.mask setup at execute() startup
            # to an explicit phase so the workflow visibility matches the
            # cost. Two jobs: (1) set env.mask + env.extent so every
            # downstream raster op is auto-clipped to AOI (already
            # happening); (2) drop scenes whose bounding box has zero
            # overlap with the AOI bbox so Phase 4 (DL inference) and
            # Phase 5 (per-scene processing) don't waste compute on
            # scenes that contribute nothing. For Faial's 257-scene
            # archive this typically drops 3 scenes (~1%) but the win
            # scales with AOI tightness.
            arcpy.AddMessage("\n▶ Phase 3 — AOI intersection")
            if mask_feature and arcpy.Exists(mask_feature):
                aoi_desc = arcpy.Describe(mask_feature)
                aoi_sr = aoi_desc.spatialReference
                aoi_ext = aoi_desc.extent
                aoi_area = max(
                    1.0,
                    (aoi_ext.XMax - aoi_ext.XMin)
                    * (aoi_ext.YMax - aoi_ext.YMin),
                )
                arcpy.AddMessage(
                    f"  AOI bbox area: {aoi_area / 1e6:.1f} km^2 "
                    f"({aoi_sr.name})"
                )
                # Per-scene overlap requires a representative band path.
                # B01 is present on every AST_07XT scene; HDF format
                # scenes use the .hdf as the descriptor target.
                survivors = []
                dropped_zero = 0
                for s in kept_scenes:
                    band_path = None
                    files = s.get("files") or {}
                    if s.get("format") == "tiff":
                        band_path = files.get("B01") or files.get("B02")
                    else:
                        band_path = files.get("hdf")
                    if not band_path:
                        # Scene with unknown geometry: keep, let
                        # downstream phases handle the missing-band case.
                        survivors.append(s)
                        continue
                    cov_pct = _compute_aoi_overlap_pct(
                        band_path, aoi_sr, aoi_ext, aoi_area,
                    )
                    s["metadata"] = s.get("metadata") or {}
                    s["metadata"]["aoi_overlap_pct"] = round(cov_pct, 2)
                    if cov_pct <= 0.0:
                        dropped_zero += 1
                    else:
                        survivors.append(s)
                if dropped_zero:
                    arcpy.AddMessage(
                        f"  ✓ Filtered scenes: kept {len(survivors)} / "
                        f"{len(kept_scenes)} (dropped {dropped_zero} "
                        f"with 0% AOI bbox overlap)"
                    )
                else:
                    arcpy.AddMessage(
                        f"  ✓ All {len(kept_scenes)} scenes intersect AOI"
                    )
                kept_scenes = survivors
                full_scenes = [s for s in kept_scenes if self._has_swir_bands(s)]
            else:
                arcpy.AddMessage(
                    "  No AOI provided — processing all scenes at full extent"
                )

            # ----------------------------------------------------------------
            # Phase 4 — DL cloud masking (OmniCloudMask)
            # ----------------------------------------------------------------
            # Sensor-agnostic R/G/NIR U-Net ensemble (Wright et al. 2025,
            # Remote Sensing of Environment). Runs in the parent process
            # (GPU init amortised across all scenes); Phase 5 subprocess
            # workers consume the cached mask TIFFs alongside source
            # bands. Cache folder defaults to a sibling of the data
            # folder so masks persist across mosaic re-runs.
            dl_mask_folder = None
            dl_cloud_classes = None
            if use_dl_cloud_mask:
                dl_mask_folder = _resolve_dl_mask_folder(
                    dl_mask_folder_param, data_folder,
                )
                dl_cloud_classes = _dl_cloud_classes_for(dl_mask_aggressiveness)
                arcpy.AddMessage(
                    f"\n▶ Phase 4 — DL Cloud Masking ({len(kept_scenes)} scenes)"
                )
                arcpy.AddMessage(f"  Aggressiveness: {dl_mask_aggressiveness}")
                arcpy.AddMessage(
                    f"  Classes dropped: {sorted(dl_cloud_classes)} "
                    f"(0=Clear, 1=Thick, 2=Thin, 3=Shadow)"
                )
                arcpy.AddMessage(f"  Cache folder:   {dl_mask_folder}")
                try:
                    os.makedirs(dl_mask_folder, exist_ok=True)
                except OSError as e:
                    arcpy.AddError(
                        f"  ✗ Cannot create DL mask folder "
                        f"{dl_mask_folder!r}: {e}"
                    )
                    return None
                # Lazy-import the DL stack only when the user opted in.
                # ImportError surfaces install instructions for the
                # Esri Deep Learning Frameworks MSI + the Pro Package
                # Manager omnicloudmask install.
                try:
                    import torch
                    import omnicloudmask  # noqa: F401  (presence check)
                except ImportError as e:
                    arcpy.AddError(
                        f"  ✗ DL cloud-mask path requires PyTorch + "
                        f"omnicloudmask: {e}.\n"
                        f"  Install: (1) Esri Deep Learning Frameworks "
                        f"MSI (https://github.com/Esri/deep-learning-"
                        f"frameworks) into arcgispro-py3; (2) Pro: "
                        f"clone arcgispro-py3 and activate the clone; "
                        f"(3) Pro Package Manager -> Add Packages -> "
                        f"omnicloudmask; (4) close and reopen Pro."
                    )
                    return None
                device = "cuda" if torch.cuda.is_available() else "cpu"
                device_name = (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "CPU"
                )
                arcpy.AddMessage(f"  Device:         {device} ({device_name})")
                arcpy.SetProgressor(
                    "step", "DL cloud-mask inference",
                    0, max(1, len(kept_scenes)), 1,
                )
                t0_dl = time.time()
                ok_dl = 0
                skipped_dl = 0
                failures_dl = []
                total_dl = len(kept_scenes)
                try:
                    for idx, scene in enumerate(kept_scenes, 1):
                        if arcpy.env.isCancelled:
                            arcpy.AddWarning(
                                f"  ✗ Cancelled after {idx - 1}/{total_dl} scenes."
                            )
                            return None
                        sid = scene.get("scene_id") or "?"
                        arcpy.SetProgressorLabel(f"[{idx}/{total_dl}] {sid}")
                        out_path = os.path.join(
                            dl_mask_folder,
                            _DL_CLOUD_MASK_FILENAME_FMT.format(sid=sid),
                        )
                        if os.path.exists(out_path):
                            arcpy.AddMessage(_format_scene_log_line(
                                idx, total_dl, sid, 0.0,
                                extras={"status": "skipped(exists)"},
                            ))
                            skipped_dl += 1
                            arcpy.SetProgressorPosition(idx)
                            continue
                        files = scene.get("files") or {}
                        if scene.get("format") != "tiff" or not all(
                            b in files for b in ("B01", "B02", "B03N")
                        ):
                            fail_msg = "TIFF B01/B02/B03N required for DL inference"
                            arcpy.AddMessage(_format_scene_log_line(
                                idx, total_dl, sid, 0.0, fail=fail_msg,
                            ))
                            failures_dl.append((idx, sid, fail_msg))
                            arcpy.SetProgressorPosition(idx)
                            continue
                        t_scene = time.time()
                        try:
                            _dl_cloud_mask_infer_scene(
                                red_path=files["B02"],
                                green_path=files["B01"],
                                nir_path=files["B03N"],
                                output_path=out_path,
                                device=device,
                            )
                            elapsed = time.time() - t_scene
                            # Per-scene class fractions over OBSERVED
                            # pixels only (excludes the off-footprint
                            # stripe encoded as NoData sentinel 255 by
                            # _dl_cloud_mask_infer_scene). See
                            # ``_ocm_mask_class_fractions`` docstring
                            # for the remote-sensing convention.
                            cloud_pct, shadow_pct = _ocm_mask_class_fractions(
                                out_path,
                            )
                            arcpy.AddMessage(_format_scene_log_line(
                                idx, total_dl, sid, elapsed,
                                extras={
                                    "cloud": f"{cloud_pct:.1f}%",
                                    "shadow": f"{shadow_pct:.1f}%",
                                },
                            ))
                            ok_dl += 1
                        except Exception as e:
                            elapsed = time.time() - t_scene
                            fail_msg = f"{type(e).__name__}: {e}"
                            arcpy.AddMessage(_format_scene_log_line(
                                idx, total_dl, sid, elapsed, fail=fail_msg,
                            ))
                            failures_dl.append((idx, sid, fail_msg))
                        arcpy.SetProgressorPosition(idx)
                finally:
                    arcpy.ResetProgressor()
                _emit_phase3_summary(
                    ok_dl, len(failures_dl), time.time() - t0_dl, failures_dl,
                )
                if skipped_dl:
                    arcpy.AddMessage(
                        f"  Skipped (cached on disk): {skipped_dl} scene(s)"
                    )

            outputs = []
            total_start = datetime.now()

            # Mosaic 1 — 9-band VNIR+SWIR from pre-failure scenes only.
            if full_scenes:
                output_full = self._run_mosaic_pipeline(
                    full_scenes, gdb_path, f"{mosaic_name}_VnirSwir",
                    scratch_dir, use_qa, mask_feature,
                    save_stats, _ASTER_MODE_FULL, "VNIR+SWIR (9-band)",
                    bt_threshold_k=bt_threshold_k,
                    use_ast08_thermal=use_ast08_thermal,
                    enable_temporal_clean=enable_temporal_clean,
                    temporal_k=temporal_k,
                    temporal_min_obs=temporal_min_obs,
                    cloud_buffer_px=cloud_buffer_px,
                    subprocess_batch_size=subprocess_batch_size,
                    compositor=compositor,
                    dl_mask_folder=dl_mask_folder,
                    dl_cloud_classes=dl_cloud_classes,
                )
                if output_full:
                    outputs.append(output_full)
            else:
                arcpy.AddWarning(
                    "\n  ✗ No pre-Apr-2008 scenes — skipping VNIR+SWIR (9-band) mosaic"
                )

            # Mosaic 2 — 3-band VNIR-only from the full archive (post-
            # failure scenes contribute here; pre-failure scenes also
            # contribute their VNIR bands for a longer temporal stack).
            output_vnir = self._run_mosaic_pipeline(
                kept_scenes, gdb_path, f"{mosaic_name}_Vnir",
                scratch_dir, use_qa, mask_feature,
                save_stats, _ASTER_MODE_VNIR, "VNIR-only (3-band)",
                bt_threshold_k=bt_threshold_k,
                use_ast08_thermal=use_ast08_thermal,
                enable_temporal_clean=enable_temporal_clean,
                temporal_k=temporal_k,
                temporal_min_obs=temporal_min_obs,
                cloud_buffer_px=cloud_buffer_px,
                subprocess_batch_size=subprocess_batch_size,
                compositor=compositor,
                dl_mask_folder=dl_mask_folder,
                dl_cloud_classes=dl_cloud_classes,
            )
            if output_vnir:
                outputs.append(output_vnir)

            total_elapsed = (datetime.now() - total_start).total_seconds()
            mins, secs = divmod(int(total_elapsed), 60)
            hrs, mins = divmod(mins, 60)
            time_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
            arcpy.AddMessage("\n" + "=" * 60)
            arcpy.AddMessage(f"DONE — {len(outputs)} mosaic(s) written")
            for p in outputs:
                arcpy.AddMessage(f"  • {os.path.basename(p)}")
            arcpy.AddMessage(f"Total: {time_str}")
            arcpy.AddMessage("=" * 60)
            return outputs[0] if outputs else None

        except Exception as e:
            arcpy.AddError(f"Tool 03 failed: {e}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None

        finally:
            if locals().get("preserve_scratch") and scratch_dir:
                arcpy.AddMessage(
                    f"  Scratch preserved at: {scratch_dir}\n"
                    "  Re-run with the same Output Mosaic Name to resume "
                    "from completed scene stacks."
                )
            else:
                _cleanup_scratch_folder(scratch_dir)
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    # ------------------------------------------------------------------
    # Per-mosaic pipeline (Phases 3-6 for one mode + scene subset)
    # ------------------------------------------------------------------

    def _run_mosaic_pipeline(
        self, scenes, gdb_path, output_name, scratch_dir,
        use_qa, mask_feature, save_stats, mode, label,
        bt_threshold_k=None, use_ast08_thermal=False,
        enable_temporal_clean=False,
        temporal_k=_TMASK_K, temporal_min_obs=_TMASK_MIN_OBS,
        cloud_buffer_px=_ASTER_CLOUD_BUFFER_PX,
        subprocess_batch_size=10,
        compositor="GeometricMedian (default)",
        dl_mask_folder=None,
        dl_cloud_classes=None,
    ):
        """Run Phases 5-8 over a scene list in one of the supported modes.

        Phases 1-2 (scene discovery, temporal filter), 3 (AOI mask +
        intersection filter) and 4 (DL cloud mask inference) all happen
        once at AsterMosaic.execute() level since they are shared
        across the VNIR+SWIR (9-band) and VNIR-only (3-band) mosaic
        modes. This method picks up at Phase 5 with the per-scene
        processing and runs through Phase 8 cleanup/provenance per
        mode.

        Returns the final raster path on success, or None if no scenes
        survived per-scene processing (in which case a warning has
        already been emitted).
        """
        n_scenes = len(scenes)
        arcpy.AddMessage(
            f"\n▶ Phase 5 [{label}] — Per-scene processing ({n_scenes} scenes)"
        )
        stacked_paths = []
        scenes_used = []
        scene_times = []

        # Resume scan — reuse any per-scene stack that survives in scratch
        # alongside its .complete marker. The stack suffix is mode-specific
        # so the VNIR+SWIR and VNIR-only mosaics maintain independent
        # markers and do not collide.
        stack_suffix = "_stack_vnir.tif" if mode == _ASTER_MODE_VNIR else "_stack.tif"
        resumed_count = 0
        to_process = []
        for scene in scenes:
            sid = scene.get("scene_id", "")
            expected = os.path.join(scratch_dir, f"{sid}{stack_suffix}")
            if (sid and os.path.exists(expected)
                    and os.path.exists(expected + ".complete")):
                stacked_paths.append(expected)
                scenes_used.append(scene)
                resumed_count += 1
            else:
                to_process.append(scene)
        # Schema defence: 9 bands for VNIR+SWIR mode, 3 for VNIR-only.
        # Scratch from before the May 2026 BT scale fix has the correct
        # band count but a silently-broken cloud mask (thermal channel
        # was disabled); the band-count check at least catches the
        # cross-mode and cross-version layout mismatches.
        expected_bands = (
            len(_ASTER_VNIR_BANDS) if mode == _ASTER_MODE_VNIR
            else len(_ASTER_STACK_ORDER)
        )
        if not _check_scratch_schema(
            stacked_paths, expected_bands, f"ASTER [{label}]",
        ):
            return None
        if resumed_count:
            arcpy.AddMessage(
                f"  ✓ Resume: {resumed_count}/{n_scenes} stack(s) reused "
                f"from previous run"
            )

        if to_process and subprocess_batch_size > 0:
            # Subprocess path. Each batch runs in a fresh python.exe
            # to flush accumulated arcpy / GDAL state. See module-level
            # _run_scene_batches docstring.
            snap_anchor = os.path.join(scratch_dir, "_snap_anchor.tif")
            spec_extra = {
                "mask_feature": (
                    _resolve_to_catalog_path(mask_feature)
                    if mask_feature and arcpy.Exists(mask_feature)
                    else None
                ),
                "snap_anchor": (
                    snap_anchor if os.path.exists(snap_anchor) else None
                ),
                "use_qa": use_qa,
                "bt_threshold_k": bt_threshold_k,
                "use_ast08_thermal": use_ast08_thermal,
                "cloud_buffer_px": cloud_buffer_px,
                "dl_mask_folder": dl_mask_folder,
                "dl_cloud_classes": (
                    sorted(dl_cloud_classes) if dl_cloud_classes else None
                ),
            }
            worker_kind = (
                "aster_vnir" if mode == _ASTER_MODE_VNIR
                else "aster_vnir_swir"
            )
            ok = _run_scene_batches(
                worker_kind=worker_kind,
                scenes=to_process,
                batch_size=subprocess_batch_size,
                scratch_dir=scratch_dir,
                spec_extra=spec_extra,
                log_prefix="  ",
            )
            if not ok:
                return None
            # Rescan scratch for newly-completed stacks via the same
            # marker convention the resume scan uses at the top of
            # this method. The user-facing tally is already emitted by
            # _run_scene_batches' tail summary; this rescan only
            # populates internal state.
            for scene in to_process:
                sid = scene.get("scene_id", "")
                expected = os.path.join(scratch_dir, f"{sid}{stack_suffix}")
                if (sid and os.path.exists(expected)
                        and os.path.exists(expected + ".complete")):
                    stacked_paths.append(expected)
                    scenes_used.append(scene)
        elif to_process:
            # Legacy single-process loop, kept for A/B comparison and
            # as a fallback when subprocess_batch_size = 0.
            arcpy.SetProgressor(
                "step", f"Per-scene processing [{label}]",
                0, max(1, len(to_process)), 1,
            )
            t0_phase = time.time()
            failures = []
            for idx, scene in enumerate(to_process, 1):
                if arcpy.env.isCancelled:
                    arcpy.ResetProgressor()
                    arcpy.AddWarning(
                        f"  ✗ Cancelled after {idx-1}/{len(to_process)} scenes."
                    )
                    return None
                sid = scene.get("scene_id") or "?"
                arcpy.SetProgressorLabel(
                    f"[{idx}/{len(to_process)}] [{scene.get('format','?')}] {sid}"
                )
                scene_start = time.time()
                # Resolve per-scene DL cloud mask path if the cache
                # exists; falls through silently when DL is off or the
                # mask for this scene was not generated.
                dl_mask_path = None
                if dl_mask_folder:
                    candidate = os.path.join(
                        dl_mask_folder,
                        _DL_CLOUD_MASK_FILENAME_FMT.format(sid=sid),
                    )
                    if os.path.exists(candidate):
                        dl_mask_path = candidate
                try:
                    stacked = self._process_scene(
                        scene, scratch_dir, use_qa, mode,
                        bt_threshold_k=bt_threshold_k,
                        use_ast08_thermal=use_ast08_thermal,
                        cloud_buffer_px=cloud_buffer_px,
                        dl_cloud_mask_path=dl_mask_path,
                        dl_cloud_classes=dl_cloud_classes,
                    )
                    elapsed = time.time() - scene_start
                except Exception as e:
                    elapsed = time.time() - scene_start
                    fail_msg = f"{type(e).__name__}: {e}"
                    arcpy.AddMessage(_format_scene_log_line(
                        idx, len(to_process), sid, elapsed, fail=fail_msg,
                    ))
                    failures.append((idx, sid, fail_msg))
                    arcpy.SetProgressorPosition(idx)
                    continue
                if stacked:
                    stacked_paths.append(stacked)
                    scenes_used.append(scene)
                    scene_times.append(elapsed)
                    meta = scene.get("metadata") or {}
                    extras = {}
                    cloud_pct = meta.get("cloud_pct")
                    if cloud_pct is not None:
                        extras["cloud"] = f"{cloud_pct:.1f}%"
                    bt_stats = meta.get("bt_stats")
                    if bt_stats:
                        extras["BT[K]"] = bt_stats
                    arcpy.AddMessage(_format_scene_log_line(
                        idx, len(to_process), sid, elapsed,
                        extras=extras or None,
                    ))
                arcpy.SetProgressorPosition(idx)
                _periodic_arcpy_cache_flush(idx)
            arcpy.ResetProgressor()
            _emit_phase3_summary(
                len(scene_times), len(failures),
                time.time() - t0_phase, failures,
            )
        if not stacked_paths:
            arcpy.AddWarning(
                f"  ✗ [{label}] No scenes survived per-scene processing."
            )
            return None

        # Output path is resolved BEFORE Phase 4 so the evidence-layer
        # publish can be tied to the mosaic's GDB stem from the start.
        # If Phase 5 (GeometricMedian) fails downstream, the sidecars
        # published in Phase 4 still survive at their final location.
        output_path = os.path.join(gdb_path, output_name)

        # Phase 4: Temporal outlier cleaner (Tmask reduction).
        # Runs per-pixel robust z-score against each pixel's own clear-
        # sky history; flags warm marine cloud that no per-scene
        # spectral threshold can separate from warm land. Default ON;
        # disabled only for A/B comparison runs. Emits two evidence-
        # quality sidecar rasters tied to this mosaic's output name:
        # ``{name}_obs_count`` (valid clear observations per pixel)
        # and ``{name}_cloud_freq`` (flagged fraction per pixel).
        # Treat both as inputs to downstream uncertainty propagation,
        # not as QA to discard. Sidecars are CopyRaster-published from
        # scratch to the GDB sidecar folder inside this phase so they
        # survive both scratch cleanup AND a Phase 5 failure. The
        # cleaner output is the diagnostic users need to triage a
        # GeometricMedian crash.
        composite_inputs = stacked_paths
        obs_count_pub = None
        cloud_freq_pub = None
        if not enable_temporal_clean:
            arcpy.AddMessage(
                f"\n▶ Phase 6 [{label}]: Temporal cleaner DISABLED"
            )
        else:
            scratch_obs = os.path.join(
                scratch_dir, f"{output_name}_obs_count.tif",
            )
            scratch_freq = os.path.join(
                scratch_dir, f"{output_name}_cloud_freq.tif",
            )
            cleaner_ran = False
            try:
                with phase(
                    f"Phase 6 [{label}]: Temporal outlier cleaner "
                    f"(k={temporal_k}, min_obs={temporal_min_obs})",
                    quiet_close=True,
                ) as ph4:
                    composite_inputs = _temporal_outlier_clean(
                        stacked_paths,
                        brightness_band_index=2,   # B02 (red) in both modes
                        scratch_dir=scratch_dir,
                        k=temporal_k,
                        min_obs=temporal_min_obs,
                        obs_count_path=scratch_obs,
                        cloud_freq_path=scratch_freq,
                    )
                arcpy.AddMessage(
                    f"  ✓ Temporal cleaner in {ph4.elapsed:.1f}s"
                )
                cleaner_ran = True
            except ValueError as e:
                arcpy.AddWarning(
                    f"  ✗ Temporal cleaner skipped: {e}"
                )
                composite_inputs = stacked_paths
            except Exception as e:
                arcpy.AddWarning(
                    f"  ✗ Temporal cleaner failed ({e}); proceeding "
                    "with uncleaned stacks."
                )
                composite_inputs = stacked_paths

            # Publish evidence layers from scratch to the GDB sidecar
            # folder. Independent of Phase 5 outcome; runs as long as
            # the cleaner produced files. Delete-then-CopyRaster is
            # tolerant of catalog locks from a previous run that
            # ``.save()`` overwrite would trip on.
            if cleaner_ran:
                for src, suffix in (
                    (scratch_obs, "_obs_count.tif"),
                    (scratch_freq, "_cloud_freq.tif"),
                ):
                    if not arcpy.Exists(src):
                        continue
                    dst = _sidecar_path_for_raster(output_path, suffix)
                    try:
                        if arcpy.Exists(dst):
                            arcpy.management.Delete(dst)
                        arcpy.management.CopyRaster(src, dst)
                        arcpy.AddMessage(
                            f"  ✓ Evidence layer -> {os.path.basename(dst)}"
                        )
                        if suffix == "_obs_count.tif":
                            obs_count_pub = dst
                        else:
                            cloud_freq_pub = dst
                    except arcpy.ExecuteError as e:
                        arcpy.AddWarning(
                            f"  Could not publish "
                            f"{os.path.basename(src)}: {e}"
                        )

        # Phase 7 — compositor
        compositor_tag = (
            "PerBandMedian" if compositor.startswith("Per-band")
            else "GeometricMedian"
        )
        arcpy.SetProgressor(
            "default", f"Computing {compositor_tag} [{label}]...",
        )
        try:
            with phase(
                f"Phase 7 [{label}] — {compositor_tag} over "
                f"{len(composite_inputs)} stacks",
                quiet_close=True,
            ) as ph:
                if compositor.startswith("Per-band"):
                    _per_band_median_composite(composite_inputs, output_path)
                else:
                    median = arcpy.ia.GeometricMedian(
                        composite_inputs,
                        epsilon=_GEOMETRIC_MEDIAN_EPSILON,
                        max_iteration=_GEOMETRIC_MEDIAN_MAX_ITER,
                        extent_type="UnionOf",
                        cellsize_type="FirstOf",
                    )
                    median.save(output_path)
                arcpy.ResetProgressor()
            arcpy.AddMessage(
                f"  ✓ {compositor_tag} in {ph.elapsed:.1f}s "
                f"→ {os.path.basename(output_path)}"
            )
            _sanity_check_output(
                output_path, sensor_hint="aster",
                label=f"{os.path.basename(output_path)} [{label}]",
            )
        except arcpy.ExecuteError:
            # phase manager already warned with the canonical "✗ Phase 5
            # ... failed after Xs" line; no need for a duplicate AddError.
            arcpy.ResetProgressor()
            return None
        except Exception:
            # Non-arcpy failure (memory, logic bug). phase's warning
            # is the canonical record here too.
            arcpy.ResetProgressor()
            return None

        # Phase 8 — cleanup + provenance
        # The redundant ExtractByMask that used to live here was
        # dropped 2026-05-26: env.mask + env.extent are set in Phase 3
        # and honoured by every downstream op including GeometricMedian,
        # so output_path is already AOI-clipped at save time. There's
        # no merge step (unlike S2's MosaicToNewRaster across MGRS
        # tiles) that would break the AOI envelope. One intersection
        # at Phase 3 suffices.
        with phase(f"Phase 8 [{label}] — Cleanup / provenance") as ph6:
            final_path = output_path

            if save_stats:
                run_config = {
                    "cloud_bright_min": _ASTER_CLOUD_BRIGHT_MIN,
                    "cloud_flat_tol": _ASTER_CLOUD_FLAT_TOL,
                    "cloud_ndvi_max": _ASTER_CLOUD_NDVI_MAX,
                    "water_ndwi_min": _ASTER_WATER_NDWI_MIN,
                    "swir_confirm_b02_min": _ASTER_CLOUD_B02_MIN,
                    "swir_confirm_b04_min": _ASTER_CLOUD_B04_MIN,
                    "cloud_buffer_px": cloud_buffer_px,
                    "use_ast08_thermal": use_ast08_thermal,
                    "bt_threshold_k": (
                        f"{bt_threshold_k:g}" if use_ast08_thermal
                        else "n/a"
                    ),
                    "temporal_clean": (
                        f"on(k={temporal_k:g},min_obs={temporal_min_obs})"
                        if enable_temporal_clean else "off"
                    ),
                    "dl_cloud_mask": (
                        f"on(classes={sorted(dl_cloud_classes)})"
                        if dl_cloud_classes else "off"
                    ),
                    "dl_mask_folder": dl_mask_folder or "n/a",
                    "compositor": compositor,
                    "toolbox_version": TOOLBOX_VERSION,
                }
                self._write_provenance_csv(
                    final_path, scenes_used, run_config=run_config,
                )
            _write_band_sidecar_csv(
                final_path,
                "aster-vnir" if mode == _ASTER_MODE_VNIR else "aster",
            )

            # Evidence layers were published to the GDB sidecar folder
            # in Phase 4, using ``output_path``'s stem. If AOI mask was
            # applied, ``final_path`` carries a ``_Masked`` suffix; the
            # sidecars are renamed to match so downstream pairing on
            # basename works unambiguously. When AOI was not applied
            # (final_path == output_path), the rename is a no-op.
            if final_path != output_path:
                for src_pub, suffix in (
                    (obs_count_pub, "_obs_count.tif"),
                    (cloud_freq_pub, "_cloud_freq.tif"),
                ):
                    if not src_pub or not arcpy.Exists(src_pub):
                        continue
                    dst = _sidecar_path_for_raster(final_path, suffix)
                    if dst == src_pub:
                        continue
                    try:
                        if arcpy.Exists(dst):
                            arcpy.management.Delete(dst)
                        arcpy.management.Rename(src_pub, dst)
                        arcpy.AddMessage(
                            f"  ✓ Evidence layer renamed -> "
                            f"{os.path.basename(dst)}"
                        )
                    except arcpy.ExecuteError as e:
                        arcpy.AddWarning(
                            f"  Could not rename "
                            f"{os.path.basename(src_pub)}: {e}"
                        )

        return final_path

    # ------------------------------------------------------------------
    # Scene discovery (TIFF + HDF)
    # ------------------------------------------------------------------

    @classmethod
    def _find_aster_scenes(cls, data_folder, thermal_folder=None):
        """Discover ASTER scenes in data_folder. Groups per-band TIFFs by
        sceneID; for HDF, each .hdf is one scene (extracted lazily by
        _process_scene). If ``thermal_folder`` is supplied, AST_08 files
        are sourced from it (so AST_07XT and AST_08 can live in sibling
        folders, the LP DAAC default); otherwise the main data folder is
        scanned for AST_08 too.

        Returns a list of dicts with keys:
            scene_id:   17-char ASTER granule identifier
            format:     "tiff" or "hdf"
            files:      dict {band_name: path} (for tiff) or {hdf: path}
            metadata:   {acquisition_date: date, pass_number: str, ...}
            thermal:    optional {"format", "path"} when an AST_08 file
                        was paired by scene ID.
        """
        if not data_folder or not os.path.isdir(data_folder):
            return []

        entries = os.listdir(data_folder)
        scenes_by_id = {}

        # First pass: TIFF inputs. Dispatch by extension so we don't
        # accidentally classify an .hdf as a TIFF — _parse_aster_filename
        # matches both formats but only TIFFs carry a band name.
        for entry in entries:
            full = os.path.join(data_folder, entry)
            if not os.path.isfile(full):
                continue
            if not (entry.lower().endswith(".tif") or entry.lower().endswith(".tiff")):
                continue
            parsed = cls._parse_aster_filename(entry)
            if not parsed or parsed.get("band") is None:
                continue
            scene_id = parsed["scene_id"]
            if scene_id not in scenes_by_id:
                scenes_by_id[scene_id] = {
                    "scene_id": scene_id,
                    "format": "tiff",
                    "files": {},
                    "metadata": {
                        "acquisition_date": parsed["acquisition_date"],
                        "pass_number": parsed["pass"],
                        "scene_id": scene_id,
                    },
                }
            scenes_by_id[scene_id]["files"][parsed["band"]] = full

        # Second pass: HDF inputs. If we already have a TIFF set for the
        # same sceneID, keep the TIFF (pre-extracted, cheaper to read).
        for entry in entries:
            if not entry.lower().endswith(".hdf"):
                continue
            full = os.path.join(data_folder, entry)
            parsed = cls._parse_aster_filename(entry)
            if not parsed:
                continue
            scene_id = parsed["scene_id"]
            if scene_id in scenes_by_id and scenes_by_id[scene_id]["format"] == "tiff":
                continue
            scenes_by_id[scene_id] = {
                "scene_id": scene_id,
                "format": "hdf",
                "files": {"hdf": full},
                "metadata": {
                    "acquisition_date": parsed["acquisition_date"],
                    "pass_number": parsed["pass"],
                    "scene_id": scene_id,
                },
            }

        # Third pass: AST_08 (Surface Kinetic Temperature) — optional
        # thermal companion. Attached to the matching AST_07XT scene by
        # scene_id; AST_08 files with no AST_07XT counterpart are
        # ignored. TIFF is preferred over HDF when both exist (the COG
        # TIFF is already in Kelvin; HDF needs gdal subdataset extraction).
        # When ``thermal_folder`` is supplied AND exists, scan it for
        # AST_08; otherwise look in the main data folder. The thermal
        # folder also matches sibling-of-main when the user only set the
        # main folder but co-located both products on disk.
        thermal_folder = thermal_folder or None
        if thermal_folder and os.path.isdir(thermal_folder):
            scan_root = thermal_folder
            scan_entries = os.listdir(thermal_folder)
        else:
            scan_root = data_folder
            scan_entries = entries
        for entry in scan_entries:
            full = os.path.join(scan_root, entry)
            if not os.path.isfile(full):
                continue
            scene_id = cls._parse_ast08_filename(entry)
            if not scene_id or scene_id not in scenes_by_id:
                continue
            existing = scenes_by_id[scene_id].get("thermal")
            is_tiff = entry.lower().endswith((".tif", ".tiff"))
            is_hdf = entry.lower().endswith(".hdf")
            if not (is_tiff or is_hdf):
                continue
            # Prefer TIFF; do not overwrite a TIFF entry with an HDF.
            if existing and existing.get("format") == "tiff" and is_hdf:
                continue
            scenes_by_id[scene_id]["thermal"] = {
                "format": "tiff" if is_tiff else "hdf",
                "path": full,
            }

        return sorted(scenes_by_id.values(), key=lambda s: s["scene_id"])

    @staticmethod
    def _parse_aster_filename(filename):
        """Parse an ASTER AST_07XT filename (TIFF or HDF) into its parts.

        Returns dict with scene_id, pass, acquisition_date, band (None for
        HDF), group (None for HDF), proc (None for HDF). Returns None on
        non-matching names.
        """
        name = os.path.basename(filename)
        # Try the TIFF pattern first.
        m = _ASTER_TIFF_RE.match(name)
        if m:
            scene_id = (
                m.group("pass") + m.group("MM") + m.group("DD")
                + m.group("YYYY") + m.group("HMS")
            )
            try:
                acq = datetime(
                    int(m.group("YYYY")), int(m.group("MM")), int(m.group("DD")),
                ).date()
            except ValueError:
                return None
            return {
                "scene_id": scene_id,
                "pass": m.group("pass"),
                "acquisition_date": acq,
                "proc": m.group("proc"),
                "group": m.group("group").upper(),
                "band": m.group("band"),
            }

        # HDF pattern is similar but without the _SRF_<group>_<band> tail.
        hdf_re = re.compile(
            r"^AST_07XT_(?P<pass>\d{3})(?P<MM>\d{2})(?P<DD>\d{2})(?P<YYYY>\d{4})"
            r"(?P<HMS>\d{6})_(?P<proc>\d{14})\.hdf$",
            re.IGNORECASE,
        )
        m = hdf_re.match(name)
        if m:
            scene_id = (
                m.group("pass") + m.group("MM") + m.group("DD")
                + m.group("YYYY") + m.group("HMS")
            )
            try:
                acq = datetime(
                    int(m.group("YYYY")), int(m.group("MM")), int(m.group("DD")),
                ).date()
            except ValueError:
                return None
            return {
                "scene_id": scene_id,
                "pass": m.group("pass"),
                "acquisition_date": acq,
                "proc": m.group("proc"),
                "group": None,
                "band": None,
            }
        return None

    @staticmethod
    def _parse_ast08_filename(filename):
        """Match an AST_08 (Surface Kinetic Temperature) file. Returns the
        17-char scene_id used to pair the file with its AST_07XT
        counterpart, or None on non-matching names.
        """
        name = os.path.basename(filename)
        for regex in (_AST08_TIFF_RE, _AST08_HDF_RE):
            m = regex.match(name)
            if m:
                return (
                    m.group("pass") + m.group("MM") + m.group("DD")
                    + m.group("YYYY") + m.group("HMS")
                )
        return None

    # ------------------------------------------------------------------
    # Per-scene processing
    # ------------------------------------------------------------------

    @staticmethod
    def _has_swir_bands(scene):
        """True if the scene has all six SWIR bands B04..B09.

        ASTER's SWIR detector failed in April 2008; scenes acquired after
        that date arrive as VNIR-only TIFF sets and lack any of B04..B09.
        HDF granules always carry the full subdataset table (any missing
        SWIR planes inside the HDF are detected later, on read).
        """
        if scene.get("format") == "hdf":
            return True
        files = scene.get("files") or {}
        return all(b in files for b in _ASTER_SWIR_BANDS)

    def _process_scene(self, scene, scratch_dir, use_qa, mode,
                       bt_threshold_k=None, use_ast08_thermal=False,
                       cloud_buffer_px=_ASTER_CLOUD_BUFFER_PX,
                       dl_cloud_mask_path=None,
                       dl_cloud_classes=None):
        """Apply scale + resample + QA mask + stack → single multi-band raster.

        Parameters
        ----------
        mode : str
            "vnir_swir" — build the full 9-band stack (B01, B02, B03N,
            B04..B09). Requires all six SWIR bands.
            "vnir_only" — build the 3-band VNIR stack (B01, B02, B03N).
            Skips SWIR resample entirely; usable for post-Apr-2008 scenes.

        Returns the stacked raster path, or None on failure (missing bands,
        unreadable QA, etc.).
        """
        if scene["format"] == "hdf":
            # Try GDAL subdataset URIs first — arcpy in Pro 3.x can read
            # `HDF4_EOS:EOS_SWATH:"foo.hdf":...:ImageData4` directly with
            # windowed reads, avoiding the per-band gdal.Translate
            # materialisation that the legacy path needs. Falls back to
            # scratch TIFFs if arcpy can't open the URIs on this install.
            extracted = self._get_hdf_subdataset_uris(scene["files"]["hdf"])
            if extracted is not None:
                try:
                    test_uri = next(iter(extracted.values()))
                    _ = arcpy.sa.Raster(test_uri).bandCount
                except Exception:
                    extracted = None  # arcpy can't read URIs; fall back below
            if extracted is None:
                extracted = self._extract_hdf_to_tiffs(scene["files"]["hdf"], scratch_dir)
                if extracted is None:
                    arcpy.AddWarning(f"  ✗ {scene['scene_id']}: HDF extraction failed")
                    return None
            scene = dict(scene, format="tiff", files=extracted)

        files = scene["files"]
        scene_id = scene["scene_id"]

        if mode == _ASTER_MODE_VNIR:
            required_bands = _ASTER_VNIR_BANDS
            preferred_qa = ["VNIR_QA_DataPlane"]
            stack_suffix = "_stack_vnir.tif"
        else:
            required_bands = _ASTER_STACK_ORDER
            preferred_qa = _ASTER_QA_NAMES
            stack_suffix = "_stack.tif"

        missing = [b for b in required_bands if b not in files]
        if missing:
            if mode == _ASTER_MODE_FULL and any(
                b in _ASTER_SWIR_BANDS for b in missing
            ):
                arcpy.AddWarning(
                    f"  ✗ {scene_id}: VNIR-only scene "
                    f"(post-Apr-2008 SWIR failure) — skipped from 9-band mosaic"
                )
            else:
                arcpy.AddWarning(f"  ✗ {scene_id}: missing bands {missing}")
            return None

        # Optional QA mask. Conservative implementation: treat any non-zero
        # value in either QA Data Plane as a quality issue (fill / bad
        # pixel / retrieval failure). Note that the AST_07XT QA Data Plane
        # is NOT a cloud mask — it flags surface-reflectance retrieval
        # status only. Cloud detection is handled separately below via
        # the per-scene multi-spectral test.
        qa_mask = None
        if use_qa and any(qa in files for qa in preferred_qa):
            qa_path = next((files[q] for q in preferred_qa if q in files), None)
            if qa_path:
                try:
                    qa_resampled = os.path.join(scratch_dir, f"{scene_id}_QA_15m.tif")
                    arcpy.management.Resample(qa_path, qa_resampled, 15, "NEAREST")
                    qa_raster = arcpy.sa.Raster(qa_resampled)
                    qa_mask = qa_raster != 0  # non-zero == flagged
                except Exception as e:
                    arcpy.AddWarning(f"  ✗ {scene_id}: QA mask build failed ({e}); continuing without")
                    qa_mask = None

        # First pass: build scaled-reflectance rasters for every required
        # band. SWIR bands are resampled from 30m to 15m first. The
        # ``SetNull(<= 0)`` after read/resample is applied SYMMETRICALLY
        # to VNIR and SWIR. Two failure modes it catches:
        #   - Zero: AST_07XT L2 source TIFFs use 0 as the conventional
        #     no-data marker for out-of-footprint pixels and
        #     unrecoverable retrievals. Without this guard, those zeros
        #     leak through ``Float() * scale = 0`` into the per-scene
        #     composite. Across many scenes the GeometricMedian then
        #     pulls toward zero in proportion to how many scenes do not
        #     cover the pixel; the bias is visible as scene-footprint-
        #     shaped dark patches in the final composite (the 2026-05-25
        #     VNIR regression on Faial; AOI mask = 1373x964, NIR mean
        #     0.10 instead of expected 0.3-0.5).
        #   - Negative: the float32 NoData sentinel (~-3.4e+38) that
        #     BILINEAR resample can introduce near scene edges; if it
        #     survives into Float() * scale it propagates as huge-
        #     magnitude values that destroy band statistics (the
        #     symmetric SWIR symptom: bands 4-9 mean ~-2.6e+17 in the
        #     same V6 run).
        # Previously the guard was only on the SWIR path; the VNIR
        # native-15m read bypassed it because no Resample step was
        # involved. The 2026-05-25 AOI catalog-path fix made env.mask
        # actually reach the worker, which surfaced both failure modes
        # because the saved per-scene rasters now span the AOI rather
        # than the scene's natural footprint.
        scaled = {}
        for band in required_bands:
            src = files[band]
            if band not in _ASTER_NATIVE_15M:
                resampled = os.path.join(scratch_dir, f"{scene_id}_{band}_15m.tif")
                arcpy.management.Resample(src, resampled, 15, "BILINEAR")
                src_raster = arcpy.sa.Raster(resampled)
            else:
                src_raster = arcpy.sa.Raster(src)
            src_raster = arcpy.sa.SetNull(src_raster <= 0, src_raster)
            scaled[band] = Float(src_raster) * _ASTER_REFLECTANCE_SCALE

        # Per-scene multi-spectral cloud test (hardened, VNIR-driven).
        # Primary test runs on VNIR alone so the post-Apr-2008 majority
        # of the archive (no SWIR) gets a real test, not a single-
        # threshold red brightener. Cloud over a vegetated basaltic
        # island is bright, spectrally flat across VNIR, and low-NDVI;
        # vegetation is high-NDVI; bare basalt is bright but red-
        # dominated (not flat); water and shadow are dark. Optional
        # SWIR confirmation tightens the test on pre-failure scenes.
        # Coupled with the temporal cleaner in Phase 4, the per-scene
        # path is deliberately recall-oriented: catch gross cloud
        # cheaply, let the temporal layer handle the residue.
        b1 = scaled["B01"]      # green
        b2 = scaled["B02"]      # red
        b3 = scaled["B03N"]     # NIR

        brightness = (b1 + b2 + b3) / 3.0
        ndvi = (b3 - b2) / (b3 + b2 + 1e-6)

        # Spectral flatness proxy: |B03N - B02| AND |B01 - B02| both
        # small. Two pairs of inequalities instead of arcpy.sa.Abs so
        # we don't import a third name just for the absolute value.
        nir_red = b3 - b2
        grn_red = b1 - b2
        flat = (
            (nir_red < _ASTER_CLOUD_FLAT_TOL)
            & (nir_red > -_ASTER_CLOUD_FLAT_TOL)
            & (grn_red < _ASTER_CLOUD_FLAT_TOL)
            & (grn_red > -_ASTER_CLOUD_FLAT_TOL)
        )

        cloud_vis_swir = (
            (brightness > _ASTER_CLOUD_BRIGHT_MIN)
            & flat
            & (ndvi < _ASTER_CLOUD_NDVI_MAX)
        )

        # SWIR confirmation. ORs in pixels bright in BOTH red AND SWIR:
        # a narrower bright-cloud signature complementary to the VNIR
        # flat-and-bright test. Skipped on VNIR-only stacks.
        if "B04" in scaled:
            cloud_vis_swir = cloud_vis_swir | (
                (b2 > _ASTER_CLOUD_B02_MIN)
                & (scaled["B04"] > _ASTER_CLOUD_B04_MIN)
            )

        # NDWI water guard (McFeeters 1996). NDWI > 0 marks open water;
        # coastal whitewater / surf can be bright and spectrally
        # flat-ish, tripping the cloud test. Excluded from cloud
        # candidacy so the temporal median composites the water
        # without per-scene whitewater holes.
        ndwi = (b1 - b3) / (b1 + b3 + 1e-6)
        water = ndwi > _ASTER_WATER_NDWI_MIN
        cloud_vis_swir = cloud_vis_swir & ~water

        # Optional thermal channel. AST_08 (Surface Kinetic Temperature)
        # is produced by the TES algorithm AFTER the operational L2
        # cloud mask is already applied, so over cloud it is NoData or a
        # corrupted retrieval; it fails on precisely the pixels a cloud
        # test needs, and the warm-cloud / warm-land BT distributions
        # overlap besides. The path is preserved for experimentation
        # but is OFF by default; the temporal cleaner in Phase 4
        # handles cloud removal instead.
        bt_threshold = (
            float(bt_threshold_k) if bt_threshold_k is not None
            else _ASTER_CLOUD_BT_MAX_K
        )
        bt_kelvin = (
            self._load_bt_kelvin(scene.get("thermal"), scratch_dir, scene_id)
            if use_ast08_thermal else None
        )
        if bt_kelvin is not None:
            cloud_mask = cloud_vis_swir | (bt_kelvin < bt_threshold)
        else:
            cloud_mask = cloud_vis_swir

        # OmniCloudMask DL output, when generated by Phase 4. Classes
        # are uint8 (0=Clear, 1=Thick, 2=Thin, 3=Shadow); the caller
        # picks which to drop via ``dl_cloud_classes`` (see
        # _dl_cloud_classes_for). OR-ing into ``cloud_mask`` BEFORE
        # the FocalStatistics dilation below means the cloud-edge
        # buffer applies uniformly across spectral and DL detections.
        if dl_cloud_mask_path and dl_cloud_classes:
            try:
                dl_mask_raster = arcpy.sa.Raster(dl_cloud_mask_path)
                dl_drop = None
                for cls in dl_cloud_classes:
                    term = (dl_mask_raster == cls)
                    dl_drop = term if dl_drop is None else (dl_drop | term)
                if dl_drop is not None:
                    cloud_mask = cloud_mask | dl_drop
            except Exception as e:
                arcpy.AddWarning(
                    f"    DL cloud mask load failed ({e}); proceeding "
                    "with spectral test only"
                )

        # Edge dilation. Marine cloud has a soft 1-3 pixel halo from
        # subpixel mixing that the per-pixel test misses; a circular
        # FocalStatistics MAXIMUM pulls those pixels into the mask.
        # Mirrors the S2 tool's cloud-buffer idiom. Materialised to
        # scratch so the diagnostic below + downstream SetNull both
        # consume a real raster (no lazy-expression re-evaluation).
        if cloud_buffer_px and cloud_buffer_px > 0:
            try:
                buffered_path = os.path.join(
                    scratch_dir, f"{scene_id}_cloudbuf.tif",
                )
                cloud_bin = arcpy.sa.Con(cloud_mask, 1, 0)
                buffered = arcpy.sa.FocalStatistics(
                    cloud_bin,
                    arcpy.sa.NbrCircle(cloud_buffer_px, "CELL"),
                    "MAXIMUM",
                    "DATA",
                )
                buffered.save(buffered_path)
                cloud_mask = arcpy.sa.Raster(buffered_path) > 0
            except arcpy.ExecuteError as e:
                arcpy.AddWarning(
                    f"    cloud dilation failed ({e}); using "
                    "undilated mask"
                )

        # Per-scene cloud diagnostic. Materialises the lazy cloud mask
        # once and reads it via numpy to get the flagged-pixel fraction;
        # also surfaces the AST_08 BT min/median/max so threshold tuning
        # can be done from log evidence rather than guesswork. Saving
        # ``cloud_mask`` to disk first avoids re-evaluating the
        # raster-math expression downstream when it's used in SetNull.
        cloud_mask_path = os.path.join(scratch_dir, f"{scene_id}_cloudmask.tif")
        try:
            cloud_mask.save(cloud_mask_path)
            cloud_mask = arcpy.sa.Raster(cloud_mask_path)
            # Report flagged fraction over OBSERVED pixels only, not
            # over the saved-grid size. The per-band SetNull(<= 0)
            # guard up-stream makes ``scaled[required_bands[0]]``
            # NoData where the source was off-footprint; load that
            # band with NaN sentinels and use ``~isnan`` as the
            # observed-pixel mask. Matches the convention applied to
            # the DL mask in Phase 4 via _ocm_mask_class_fractions.
            cm_arr = arcpy.RasterToNumPyArray(cloud_mask, nodata_to_value=0)
            try:
                ref_band_arr = arcpy.RasterToNumPyArray(
                    scaled[required_bands[0]],
                    nodata_to_value=np.nan,
                )
                observed = ~np.isnan(ref_band_arr)
                n_observed = int(observed.sum())
            except Exception:
                # Fall back to "all pixels observed" if the reference
                # band can't be reduced to a NaN-bearing array; the
                # number is then approximate but never zero-divides.
                n_observed = int(cm_arr.size)
                observed = None
            if n_observed:
                if observed is not None:
                    cm_flagged = int(((cm_arr == 1) & observed).sum())
                else:
                    cm_flagged = int(cm_arr.sum())
                cm_pct = 100.0 * cm_flagged / n_observed
            else:
                cm_pct = 0.0
            # Attach to the shared metadata dict so the provenance CSV
            # row for this scene can surface it without re-deriving.
            # ``dict(scene, ...)`` on line 7213 creates a new outer dict
            # but the metadata reference is shared with the caller.
            scene["metadata"]["cloud_pct"] = round(cm_pct, 2)
            # Stash BT stats on metadata too (when AST_08 thermal is on)
            # so the subprocess worker can surface them via the per-scene
            # log line's extras dict. The orchestrator's unified Phase 3
            # log replaces the inline arcpy.AddMessage diagnostic that
            # previously lived here.
            if bt_kelvin is not None:
                try:
                    bt_arr = arcpy.RasterToNumPyArray(bt_kelvin)
                    bt_valid = bt_arr[bt_arr > _ASTER_TIR_VALID_K_FLOOR]
                    if bt_valid.size:
                        scene["metadata"]["bt_stats"] = (
                            f"min={float(np.min(bt_valid)):.1f}"
                            f"/med={float(np.median(bt_valid)):.1f}"
                            f"/max={float(np.max(bt_valid)):.1f}"
                            f" (cut {bt_threshold:.0f})"
                        )
                except Exception:
                    pass
        except Exception as e:
            arcpy.AddWarning(
                f"    cloud diagnostic failed ({e}); proceeding"
            )

        # Combine QA mask (non-cloud quality issues) with the cloud mask.
        # Either condition is enough to drop the pixel.
        if qa_mask is not None:
            drop_mask = qa_mask | cloud_mask
        else:
            drop_mask = cloud_mask

        # Second pass: apply the combined drop mask to every band and save.
        masked_paths = []
        for band in required_bands:
            masked = arcpy.sa.SetNull(drop_mask, scaled[band])
            out = os.path.join(scratch_dir, f"{scene_id}_{band}_masked.tif")
            masked.save(out)
            masked_paths.append(out)

        stacked = os.path.join(scratch_dir, f"{scene_id}{stack_suffix}")
        arcpy.management.CompositeBands(masked_paths, stacked)

        # Resume sentinel — written only after CompositeBands succeeds,
        # so its presence guarantees the stack file is complete. A
        # crash mid-CompositeBands leaves the stack without a marker,
        # which the Phase 3 resume scan treats as partial and rebuilds.
        try:
            with open(stacked + ".complete", "w", encoding="utf-8") as fh:
                fh.write(datetime.now().isoformat(timespec="seconds") + "\n")
        except OSError:
            pass

        # Per-scene scratch cleanup: drop QA_15m, per-band 15m resamples,
        # per-band masked rasters, cloudmask diagnostic save, cloudbuf
        # dilation save, and any HDF subdataset extracts. Keeps the
        # final stack + resume marker; the Phase 4 temporal cleaner's
        # brightness extracts and cleaned stacks are created LATER (in
        # ``_temporal_outlier_clean``) and therefore aren't touched.
        _cleanup_per_scene_intermediates(
            scratch_dir, scene_id,
            keep_basenames=(
                f"{scene_id}{stack_suffix}",
                f"{scene_id}{stack_suffix}.complete",
            ),
        )

        return stacked

    # Subdataset name → band label mapping used by both the URI helper
    # (no materialisation) and the legacy gdal.Translate path.
    _HDF_SUBDATASET_LOOKUP = {
        "ImageData1": "B01", "ImageData2": "B02", "ImageData3": "B03N",
        "ImageData4": "B04", "ImageData5": "B05", "ImageData6": "B06",
        "ImageData7": "B07", "ImageData8": "B08", "ImageData9": "B09",
        "QA_DataPlane": "VNIR_QA_DataPlane",
        "QA_DataPlane2": "SWIR_QA_DataPlane",
    }

    @classmethod
    def _get_hdf_subdataset_uris(cls, hdf_path):
        """Map ASTER HDF subdatasets to GDAL URIs without materialisation.

        Returns dict {band_label: subdataset_uri} or None on failure.
        The URIs are GDAL-format strings like
        `HDF4_EOS:EOS_SWATH:"foo.hdf":SwathName:ImageData4` that arcpy
        can read directly on Pro 3.x. arcpy then issues windowed reads
        through the GDAL HDF4 driver as the downstream `Resample` /
        `CompositeBands` need pixels — avoiding the ~10 gdal.Translate
        calls per scene that `_extract_hdf_to_tiffs` does.

        Returns None if osgeo.gdal isn't available, the HDF can't be
        opened, or no recognised subdatasets are present. Callers
        should fall back to `_extract_hdf_to_tiffs` in that case.
        """
        try:
            from osgeo import gdal
        except ImportError:
            return None
        try:
            ds = gdal.Open(hdf_path)
            if ds is None:
                return None
            outputs = {}
            for sub_path, _ in ds.GetSubDatasets():
                tail = sub_path.rsplit(":", 1)[-1]
                if tail in cls._HDF_SUBDATASET_LOOKUP:
                    outputs[cls._HDF_SUBDATASET_LOOKUP[tail]] = sub_path
            return outputs if outputs else None
        except Exception:
            return None

    @classmethod
    def _extract_hdf_to_tiffs(cls, hdf_path, scratch_dir):
        """Extract ASTER AST_07XT subdatasets from an HDF-EOS file to a
        per-band TIFF dict matching the standard TIFF convention.

        Used as a fallback when arcpy cannot read HDF subdataset URIs
        directly on the current install (see `_get_hdf_subdataset_uris`
        which avoids this materialisation step entirely).

        Uses osgeo.gdal (bundled with arcpy) to read the subdatasets.
        Returns None if gdal isn't available or if the HDF structure
        doesn't match the expected AST_07XT V004 layout.
        """
        try:
            from osgeo import gdal
        except ImportError:
            arcpy.AddWarning(
                "    osgeo.gdal not available — cannot read HDF inputs. "
                "Pre-extract to TIFF (e.g., via ENVI or gdal_translate)."
            )
            return None

        try:
            ds = gdal.Open(hdf_path)
            if ds is None:
                return None
            # The exact subdataset names vary by hdf_eos library version;
            # we match by suffix using the shared class-level lookup.
            outputs = {}
            scene_stem = os.path.splitext(os.path.basename(hdf_path))[0]
            for sub_path, _ in ds.GetSubDatasets():
                # sub_path looks like 'HDF4_EOS:EOS_SWATH:"foo.hdf":SwathName:ImageData4'
                tail = sub_path.rsplit(":", 1)[-1]
                if tail not in cls._HDF_SUBDATASET_LOOKUP:
                    continue
                band_label = cls._HDF_SUBDATASET_LOOKUP[tail]
                out_tiff = os.path.join(scratch_dir, f"{scene_stem}_{band_label}.tif")
                gdal.Translate(out_tiff, sub_path)
                outputs[band_label] = out_tiff
            return outputs if outputs else None
        except Exception as e:
            arcpy.AddWarning(f"    HDF extraction error: {e}")
            return None

    # ------------------------------------------------------------------
    # AST_08 (Surface Kinetic Temperature) — thermal cloud channel
    # ------------------------------------------------------------------

    @classmethod
    def _extract_ast08_bt_tiff(cls, hdf_path, scratch_dir):
        """Materialise the AST_08 kinetic temperature SDS to a scratch
        TIFF. Returns the TIFF path, or None on failure (gdal missing,
        no recognised SDS, etc.).
        """
        try:
            from osgeo import gdal
        except ImportError:
            return None
        try:
            ds = gdal.Open(hdf_path)
            if ds is None:
                return None
            scene_stem = os.path.splitext(os.path.basename(hdf_path))[0]
            for sub_path, _ in ds.GetSubDatasets():
                tail = sub_path.rsplit(":", 1)[-1]
                if tail in _AST08_BT_SDS_NAMES:
                    out_tiff = os.path.join(scratch_dir, f"{scene_stem}_BT.tif")
                    gdal.Translate(out_tiff, sub_path)
                    return out_tiff
            return None
        except Exception as e:
            arcpy.AddWarning(f"    AST_08 HDF extraction error: {e}")
            return None

    def _load_bt_kelvin(self, thermal_meta, scratch_dir, scene_id):
        """Load the paired AST_08 surface kinetic temperature at the 15 m
        grid used by the per-scene cloud test. Returns ``None`` when no
        thermal pairing exists or the load fails (the caller falls back
        to VIS/SWIR-only cloud detection). The actual load logic lives
        in module-level ``_aster_bt_kelvin_from_path``; this method
        preserves the per-scene warning convention of the cloud-test
        path.
        """
        if not thermal_meta:
            return None
        fmt = thermal_meta.get("format")
        path = thermal_meta.get("path")
        if not (fmt and path):
            return None
        try:
            return _aster_bt_kelvin_from_path(
                path, scratch_dir, target_cellsize=15, scene_id=scene_id,
            )
        except Exception as e:
            arcpy.AddWarning(
                f"  ✗ {scene_id}: AST_08 BT load failed ({e}); "
                "continuing without thermal cloud channel"
            )
            return None

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    @staticmethod
    def _write_provenance_csv(output_raster_path, scenes_used, run_config=None):
        """Write the per-scene provenance CSV. Optional ``run_config``
        is serialised as a leading ``# run config: k=v; ...`` comment
        line so consumers with ``comment='#'`` skip it cleanly. The
        per-scene rows include a ``cloud_pct`` column carrying the
        flagged-pixel fraction from the per-scene cloud diagnostic
        (populated when ``_process_scene`` ran the diagnostic block,
        which is the default path).
        """
        if not output_raster_path or not scenes_used:
            return
        try:
            csv_path = _sidecar_path_for_raster(output_raster_path, "_provenance.csv")
            now_iso = datetime.now().isoformat(timespec="seconds")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                if run_config:
                    config_str = "; ".join(
                        f"{k}={v}" for k, v in run_config.items()
                    )
                    fh.write(f"# run config: {config_str}\n")
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "scene_id", "sensor", "acquisition_datetime",
                    "pass_number", "input_format", "input_path",
                    "cloud_pct",
                    "processing_baseline", "toolbox_version",
                    "processing_datetime",
                ])
                for scene in scenes_used:
                    meta = scene.get("metadata", {}) or {}
                    acq = meta.get("acquisition_date", "")
                    if hasattr(acq, "isoformat"):
                        acq = acq.isoformat()
                    # Prefer the HDF path for HDF scenes, else the first file path.
                    files = scene.get("files", {})
                    if scene.get("format") == "hdf":
                        input_path = files.get("hdf", "")
                    else:
                        # Pick the path of B01 as the canonical reference.
                        input_path = files.get("B01") or next(iter(files.values()), "")
                    writer.writerow([
                        scene.get("scene_id", ""),
                        "ASTER (AST_07XT V004)",
                        acq,
                        meta.get("pass_number", ""),
                        scene.get("format", ""),
                        input_path,
                        meta.get("cloud_pct", ""),
                        "AST_07XT V004",
                        TOOLBOX_VERSION,
                        now_iso,
                    ])
            arcpy.AddMessage(f"Provenance CSV: {csv_path}")
        except OSError as e:
            arcpy.AddWarning(f"Failed to write provenance CSV: {e}")

    # ------------------------------------------------------------------
    # Temporal-filter helpers (duplicated from S2/Landsat — Phase 6 extracts)
    # ------------------------------------------------------------------

    @staticmethod
    def _seasonal_pattern_for_region(region):
        if region in (
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
        ):
            return "temperate"
        if "Cape Verde" in (region or ""):
            return "cape_verde"
        if region == "Angola":
            return "angola"
        if region == "Mozambique":
            return "mozambique"
        return "temperate"

    @staticmethod
    def _create_temporal_filter(time_type, year, month, season):
        return {"type": time_type, "year": year, "month": month, "season": season}

    def _scene_passes_filter(self, meta, temporal_filter, seasonal_pattern):
        d = meta.get("acquisition_date")
        if d is None:
            return False
        ftype = temporal_filter.get("type")
        if ftype == "all_images":
            return True
        try:
            if ftype == "year_month":
                return d.year == temporal_filter["year"] and d.month == temporal_filter["month"]
            if ftype == "month_all_years":
                return d.month == temporal_filter["month"]
            if ftype == "season_in_year":
                if d.year != temporal_filter["year"]:
                    return False
                return self._is_in_season(d, seasonal_pattern, temporal_filter["season"])
            if ftype == "season_all_years":
                return self._is_in_season(d, seasonal_pattern, temporal_filter["season"])
        except (TypeError, KeyError):
            return False
        return False

    @staticmethod
    def _is_in_season(date, pattern, season):
        if season is None:
            return False
        season_months = {
            "temperate": {"spring": [3, 4, 5], "summer": [6, 7, 8],
                          "autumn": [9, 10, 11], "winter": [12, 1, 2]},
            "angola": {"rainy": [11, 12, 1, 2, 3, 4], "dry": [5, 6, 7, 8, 9, 10],
                       "rainy_peak": [1, 2, 3], "dry_peak": [6, 7, 8]},
            "cape_verde": {"dry": [12, 1, 2, 3, 4, 5, 6], "rainy": [8, 9, 10],
                           "transition_dry_wet": [7], "transition_wet_dry": [11]},
            "mozambique": {"rainy": [10, 11, 12, 1, 2, 3], "dry": [4, 5, 6, 7, 8, 9],
                           "rainy_peak": [12, 1, 2], "dry_peak": [7, 8, 9]},
        }
        try:
            return date.month in season_months[pattern][season.lower()]
        except KeyError:
            return False


# ---------------------------------------------------------------------------
# Pure-numpy FastICA (Hyvärinen parallel algorithm, symmetric
# decorrelation, ``logcosh`` non-linearity). Replaces the previous
# ``sklearn.decomposition.FastICA`` dependency so the toolbox loads on
# Pro environments that don't have ``scikit-learn`` installed.
#
# Matches the sklearn API the Noise Transform tool relies on:
#   - Input ``X`` is shape ``(n_samples, n_features)`` and assumed
#     already whitened (the caller does the MNF whitening upstream).
#   - Returns ``(S, W, n_iter)`` where ``S`` has shape
#     ``(n_samples, n_features)`` (all components — caller truncates
#     to the requested ``n_components``), ``W`` has shape
#     ``(n_features, n_features)`` (unmixing matrix), and ``n_iter``
#     is the iteration count at convergence (for logging).
# ---------------------------------------------------------------------------

def _sym_decorrelation(W):
    """Symmetric decorrelation: ``W' = (W W^T)^(-1/2) W``.

    Clamps the eigenvalues at a small floor before inverting their
    square root so a rank-deficient ``W`` (or one that drifts toward
    rank-deficiency over many iterations) does not produce inf / NaN.
    """
    s, u = np.linalg.eigh(W @ W.T)
    inv_sqrt_s = 1.0 / np.sqrt(np.maximum(s, _SYM_DECORRELATION_FLOOR))
    return (u * inv_sqrt_s) @ u.T @ W


def _fast_ica_numpy(X, max_iter=_FAST_ICA_MAX_ITER, tol=_FAST_ICA_TOL,
                    random_state=None):
    """FastICA (Hyvärinen, parallel) on already-whitened ``X``.

    Drop-in replacement for ``sklearn.decomposition.FastICA`` with
    ``whiten=False``, ``algorithm='parallel'``, and the default
    ``logcosh`` non-linearity. No external dependencies.
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    X_T = X.T  # (n_features, n_samples) — Hyvärinen convention

    W = rng.normal(size=(n_features, n_features)).astype(X.dtype, copy=False)
    W = _sym_decorrelation(W)

    n_iter = max_iter  # default if loop exits via max_iter without converging
    for n_iter in range(1, max_iter + 1):
        # g(u) = tanh(u), g'(u) = 1 - tanh(u)^2  (logcosh non-linearity)
        WX = W @ X_T
        g_wx = np.tanh(WX)
        gp_mean = (1.0 - g_wx ** 2).mean(axis=1)

        # FastICA update: W_new = E[X g(W X)^T] - diag(E[g'(W X)]) W
        W_new = (g_wx @ X_T.T) / n_samples - gp_mean[:, None] * W
        W_new = _sym_decorrelation(W_new)

        # Convergence: each row of W_new should be parallel (cos=1 or -1)
        # to the corresponding row of W. ``np.diag(W_new @ W.T)`` gives
        # the per-row cosines.
        delta = float(np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0)))
        W = W_new
        if delta < tol:
            break

    S = (W @ X_T).T  # (n_samples, n_features)
    return S, W, n_iter


class Transformations(object):
    """Tool 05 — Statistical transformations (PCA / MNF / ICA).

    Sensor-agnostic at the algorithm level: PCA/MNF/ICA operate on column
    vectors and don't care which spectral band each column represents.
    The Sensor Type parameter is used for (a) sensible default
    `num_components` per sensor, (b) sanity-checking the requested
    component count against the expected band count, and (c) tagging the
    saved *Statistics .npz for provenance.

    Inherits the full set of Phase 1 audit fixes: kurtosis_values
    persistence, NoData-safe noise estimator, sensor-agnostic transform
    re-apply functions, etc.
    """

    def __init__(self):
        self.label = "05 — Statistical Transformations"
        self.description = (
            "Run PCA, MNF, or ICA on a pre-stacked multiband raster from "
            "any of the three supported sensors (Landsat 8/9, Sentinel-2 "
            "L2A, ASTER AST_07XT). The algorithm is sensor-agnostic — the "
            "Sensor Type parameter selects sensible defaults and is "
            "recorded on the saved *Statistics .npz for provenance."
        )
        self.canRunInBackground = True
        
    def getParameterInfo(self):
        # Input raster bands
        input_rasters = arcpy.Parameter(
            displayName="Input Raster Bands",
            name="input_rasters",
            datatype=["DERasterDataset", "GPRasterLayer"],
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        
        # Transformation type
        transform_type = arcpy.Parameter(
            displayName="Transformation Type",
            name="transform_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        transform_type.filter.list = ["MNF", "PCA", "ICA"]
        
        # Number of components
        num_components = arcpy.Parameter(
            displayName="Number of Output Components",
            name="num_components",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        
        # Optional parameters for MNF
        noise_stats_file = arcpy.Parameter(
            displayName="Input Noise Statistics (Optional, MNF only)",
            name="noise_stats_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        noise_subset = arcpy.Parameter(
            displayName="Spatial Subset for Noise (Optional, MNF only)",
            name="noise_subset",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        # Output workspace
        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        
        # Output name
        out_name = arcpy.Parameter(
            displayName="Output Name",
            name="out_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        
        # Save statistics
        save_stats = arcpy.Parameter(
            displayName="Save Transform Statistics",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        save_stats.value = True
        
        # Statistics folder. Empty by default — the tool co-locates the
        # .npz / .txt sidecar with the output raster (same folder for
        # folder workspaces, GDB's parent folder for .gdb / .sde). Set
        # explicitly only to override that placement.
        stats_folder = arcpy.Parameter(
            displayName=(
                "Statistics Folder (optional — leave empty to co-locate "
                "the .npz / .txt with the output raster)"
            ),
            name="stats_folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        
        # Preserve Input Mask
        preserve_mask = arcpy.Parameter(
            displayName="Preserve Input Mask",
            name="preserve_mask",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        preserve_mask.value = True

        # ISS-007: random_state for FastICA reproducibility. Only relevant
        # for ICA; ignored by PCA / MNF.
        random_state = arcpy.Parameter(
            displayName="ICA Random Seed (Optional, ICA only)",
            name="random_state",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            enabled=False,
        )
        random_state.value = 42

        # Sensor selector (Phase 6 addition) — informational for the
        # algorithm (PCA/MNF/ICA don't care about band identity) but
        # drives sensible defaults for num_components and records the
        # sensor on the saved *Statistics .npz.
        sensor_type = make_sensor_parameter()

        return [input_rasters, transform_type, num_components,
                noise_stats_file, noise_subset,
                out_workspace, out_name, save_stats, stats_folder,
                preserve_mask, random_state, sensor_type]
        
    def updateParameters(self, parameters):
        """Modify parameter values and properties"""
        try:
            # Get references to parameters for easier access
            input_rasters = parameters[0]
            transform_type = parameters[1]
            num_components = parameters[2]
            noise_stats_file = parameters[3]
            noise_subset = parameters[4]
            save_stats = parameters[7]
            stats_folder = parameters[8]
            
            # Enable/disable MNF-specific parameters based on transformation type
            if transform_type.altered:
                is_mnf = transform_type.valueAsText == "MNF"
                is_ica = transform_type.valueAsText == "ICA"
                noise_stats_file.enabled = is_mnf
                noise_subset.enabled = is_mnf

                # If noise statistics file is provided, disable noise subset
                if is_mnf and noise_stats_file.altered and noise_stats_file.valueAsText:
                    noise_subset.enabled = False

                # ISS-007: random_state only matters for ICA.
                if len(parameters) > 10:
                    parameters[10].enabled = is_ica
            
            # Enable/disable statistics folder parameter based on save_stats
            if save_stats.altered:
                stats_folder.enabled = save_stats.value
            
            # If input rasters are set, update number of components
            if input_rasters.altered and input_rasters.value:
                try:
                    raster_paths = input_rasters.valueAsText.split(";")
                    total_bands = 0
                    
                    # Calculate total number of bands across all inputs
                    for raster_path in raster_paths:
                        raster = arcpy.Raster(raster_path)
                        total_bands += raster.bandCount
                    
                    # Update number of components range
                    num_components.filter.type = "Range"
                    num_components.filter.list = [1, total_bands]
                    
                    # Set default number of components if not already set
                    if not num_components.altered:
                        num_components.value = min(3, total_bands)
                    
                    arcpy.AddMessage(f"Total bands available: {total_bands}")
                except Exception as e:
                    arcpy.AddWarning(f"Error counting bands: {str(e)}")
            
            # Statistics Folder: left empty by default. The execute() path
            # derives a sensible default — same folder as the output raster
            # (folder workspace) or the GDB's parent folder (GDB
            # workspace) — so the .npz / .txt always lands next to the
            # data it describes. The user can still override by filling
            # in the parameter explicitly.

        except Exception as e:
            arcpy.AddWarning(f"Error updating parameters: {str(e)}")
            
    def updateMessages(self, parameters):
        """Validate parameters"""
        if parameters[0].altered:
            try:
                # Get input raster paths
                input_rasters = parameters[0].valueAsText.split(";")
                
                # For PCA, we'll be more flexible with inputs
                if parameters[1].valueAsText == "PCA":
                    # Just check that inputs exist and are rasters
                    for raster_path in input_rasters:
                        if not arcpy.Exists(raster_path):
                            parameters[0].setErrorMessage(f"Input raster does not exist: {raster_path}")
                            return
                        
                        desc = arcpy.Describe(raster_path)
                        if not hasattr(desc, "bandCount"):
                            parameters[0].setErrorMessage(f"Input is not a valid raster: {raster_path}")
                            return
                else:
                    # For MNF or ICA, be more permissive but still check format
                    for raster_path in input_rasters:
                        if not arcpy.Exists(raster_path):
                            parameters[0].setErrorMessage(f"Input raster does not exist: {raster_path}")
                            return
                        
                        desc = arcpy.Describe(raster_path)
                        if not hasattr(desc, "bandCount"):
                            parameters[0].setErrorMessage(f"Input is not a valid raster: {raster_path}")
                            return
                
                # Check that number of components doesn't exceed total bands
                if parameters[2].altered and parameters[0].altered:
                    total_bands = 0
                    for raster_path in input_rasters:
                        raster = arcpy.Raster(raster_path)
                        total_bands += raster.bandCount
                    
                    if parameters[2].value > total_bands:
                        parameters[2].setErrorMessage(
                            f"Number of components ({parameters[2].value}) " +
                            f"cannot exceed total bands ({total_bands})"
                        )
                    
            except Exception as e:
                parameters[0].setErrorMessage(f"Error validating input: {str(e)}")
                
    def execute(self, parameters, messages):
        """Execute the tool"""
        # Declared at function scope so the finally block can clean it up
        # regardless of where execution fails.
        composite_temp_path = None
        try:
            # Check out Spatial Analyst extension
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required but not available")
                return None

            # Enable overwrite output
            arcpy.env.overwriteOutput = True
            
            # Get parameters with updated indices
            input_rasters = parameters[0].valueAsText.split(";")
            transform_type = parameters[1].valueAsText
            num_components = parameters[2].value
            noise_stats_file = parameters[3].valueAsText
            noise_subset = parameters[4].valueAsText
            out_workspace = parameters[5].valueAsText
            out_name = parameters[6].valueAsText
            save_stats = parameters[7].value
            stats_folder_param = parameters[8].valueAsText if parameters[8].altered else None
            preserve_mask = parameters[9].value  # New preserve mask parameter
            # ISS-007: random_state for ICA (optional, defaults to 42).
            random_state = (
                int(parameters[10].value)
                if len(parameters) > 10 and parameters[10].value is not None
                else 42
            )
            # parameters[11] is the sensor selector (read inline via
            # ``resolve_sensor`` later). parameters[12] is the optional
            # AOI mask feature applied to the final multiband output
            # via ``ExtractByMask`` once the eigendecomposition + save
            # have finished.
            aoi_mask_feature = (
                parameters[12].valueAsText
                if len(parameters) > 12 and parameters[12].value is not None
                else None
            )
            
            # Initialize statistics
            stats = {
                'start_time': datetime.now(),
                'transform_type': transform_type,
                'num_components': num_components,
                'errors': []
            }
            
            try:
                # Output path for result. Folder workspaces route the
                # raster into a per-transform subfolder (``pca/`` /
                # ``mnf/`` / ``ica/``) with a forced ``.tif`` extension;
                # .gdb / .sde workspaces save flat at the workspace root.
                out_path = _build_workspace_subfolder_path(
                    out_workspace, out_name, transform_type.lower(),
                )

                # Statistics file destination. Honour an explicit
                # ``Statistics Folder`` if the user supplied one;
                # otherwise co-locate with the output raster — same
                # folder as the .tif for folder workspaces, GDB's
                # parent folder for .gdb / .sde (GDBs can't store
                # .npz / .txt sidecars). Named after the output raster
                # so the pairing is obvious in the file browser.
                if save_stats:
                    if stats_folder_param:
                        stats_folder = stats_folder_param
                    else:
                        ws_lower = (out_workspace or "").lower().rstrip("\\/")
                        if ws_lower.endswith(".gdb") or ws_lower.endswith(".sde"):
                            stats_folder = os.path.dirname(os.path.normpath(out_workspace))
                        else:
                            stats_folder = os.path.dirname(out_path)
                    os.makedirs(stats_folder, exist_ok=True)
                    # All three transforms emit THREE artifacts:
                    #   .npz  — binary numpy archive, machine-reloadable
                    #           so the fitted transform can be re-applied
                    #           to a new AOI without refitting (see the
                    #           module-level transform_pca / transform_mnf
                    #           / transform_ica helpers — they accept a
                    #           loaded {PCA,MNF,ICA}Statistics object).
                    #   .txt  — human-readable summary (eigenvalues,
                    #           variance explained, mixing matrices,
                    #           independence metrics — whichever applies).
                    #   .html — self-contained dashboard with embedded
                    #           matplotlib PNGs (scree / SNR / kurtosis
                    #           plots + loadings heatmap labelled with
                    #           satellite band names when available).
                    stats_file_npz = os.path.join(
                        stats_folder, f"{out_name}_{transform_type}_stats.npz",
                    )
                    stats_file_txt = os.path.join(
                        stats_folder, f"{out_name}_{transform_type}_stats.txt",
                    )
                    stats_file_html = os.path.join(
                        stats_folder, f"{out_name}_{transform_type}_report.html",
                    )
                else:
                    stats_file_npz = None
                    stats_file_txt = None
                    stats_file_html = None
                    stats_folder = None
                
                # All transforms (PCA, MNF, ICA) share the in-tree NumPy implementation.
                arcpy.AddMessage("Using custom implementation...")

                # Multi-raster input: composite into a single multi-band raster
                # via arcpy.management.CompositeBands so the downstream loader
                # gets one cube. Restores the multi-input behaviour of the
                # deleted arcpy.sa.PrincipalComponents call. The temp file is
                # cleaned up in the finally block.
                if len(input_rasters) > 1:
                    arcpy.AddMessage(
                        f"Combining {len(input_rasters)} input rasters via CompositeBands..."
                    )
                    # Scratch pinned to output workspace's parent (avoids
                    # OneDrive sync overhead from arcpy.env.scratchFolder).
                    scratch_dir = os.path.dirname(os.path.normpath(out_workspace))
                    composite_temp_path = os.path.join(
                        scratch_dir, f"_genesis_compose_{uuid.uuid4().hex}.tif"
                    )
                    try:
                        arcpy.management.CompositeBands(input_rasters, composite_temp_path)
                        arcpy.AddMessage(f"  Combined raster: {composite_temp_path}")
                        raster_path = composite_temp_path
                    except Exception as e:
                        arcpy.AddError(f"Failed to combine input rasters: {e}")
                        return None
                else:
                    raster_path = input_rasters[0]

                arcpy.AddMessage(f"Processing input: {os.path.basename(raster_path)}")
                
                # Verify raster exists
                if not arcpy.Exists(raster_path):
                    arcpy.AddError(f"Input raster does not exist: {raster_path}")
                    return None
                
                # Get raster properties
                raster_obj = arcpy.Raster(raster_path)
                band_count = raster_obj.bandCount
                arcpy.AddMessage(f"  Has {band_count} bands")
                arcpy.AddMessage(f"  Format: {raster_obj.format}")
                arcpy.AddMessage(f"  Width: {raster_obj.width}, Height: {raster_obj.height}")
                arcpy.AddMessage(f"  Data type: {raster_obj.pixelType}")
                
                # Get reference information
                extent = raster_obj.extent
                cell_size = (raster_obj.meanCellWidth, raster_obj.meanCellHeight)
                spatial_ref = raster_obj.spatialReference
                
                # Check if raster has a mask
                has_mask = False
                mask = None
                
                # Try different methods to find mask
                if hasattr(raster_obj, 'mask'):
                    mask = raster_obj.mask
                    has_mask = True
                    arcpy.AddMessage("  Raster has an explicit mask")
                elif hasattr(raster_obj, 'noDataValue'):
                    arcpy.AddMessage(f"  Raster has NoData value: {raster_obj.noDataValue}")
                    # We'll handle NoData during array processing
                else:
                    arcpy.AddMessage("  No mask or NoData value detected")
                
                # Store mask info in raster_info
                raster_info = {
                    'extent': extent,
                    'cell_size': cell_size,
                    'spatial_ref': spatial_ref,
                    'mask': mask if preserve_mask and has_mask else None
                }
                        
                # Load entire raster at once
                arcpy.AddMessage(
                    "  Loading full raster into NumPy array via "
                    "arcpy.RasterToNumPyArray (this step is silent until "
                    "the array materialises in memory)..."
                )
                load_start = datetime.now()

                try:
                    # Load the entire raster at once
                    data_array = arcpy.RasterToNumPyArray(raster_obj)
                    arcpy.AddMessage(
                        f"  Full array loaded in "
                        f"{(datetime.now() - load_start).total_seconds():.1f}s, "
                        f"shape: {data_array.shape}"
                    )
                    
                    # For multiband raster, the dimensions should be (bands, height, width)
                    # or (height, width, bands) depending on how ArcGIS returns it
                    if len(data_array.shape) == 3:
                        # Check if bands are in the first dimension
                        if data_array.shape[0] == band_count:
                            arcpy.AddMessage("  Transposing array from (bands, height, width) to (height, width, bands)")
                            data_array = np.transpose(data_array, (1, 2, 0))
                        else:
                            arcpy.AddMessage("  Array already in (height, width, bands) format")
                    else:
                        # Single band - reshape to add band dimension
                        arcpy.AddMessage("  Reshaping single band array")
                        data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)
                    
                    arcpy.AddMessage(f"  Final array shape: {data_array.shape}")
                    
                    # Handle NoData values via three layered defences,
                    # because file-geodatabase rasters expose NoData
                    # inconsistently:
                    #
                    #   1. ``raster_obj.noDataValue`` — the easy case;
                    #      when arcpy reports an explicit fill value
                    #      we mask exactly those pixels.
                    #   2. ``arcpy.sa.IsNull`` on band 1 — the canonical
                    #      Pro way to detect NoData, works regardless
                    #      of whether the value is stored in band-level
                    #      metadata, raster-level metadata, or via the
                    #      raster's mask layer. The (1) branch above
                    #      misses many GDB configs where
                    #      ``raster_obj.noDataValue`` returns ``None``
                    #      even though the band carries a real NoData
                    #      mask — ``IsNull`` finds those.
                    #   3. All-band-zero numpy check — catches the
                    #      remaining case where the input was
                    #      AOI-masked but the outside-AOI pixels were
                    #      written at value 0 with no NoData metadata
                    #      at all.
                    #
                    # Each defence sets the corresponding pixels to
                    # ``np.nan`` in ``data_array`` so the PCA / MNF /
                    # ICA eigendecomposition skips them and the
                    # NoData footprint propagates to every component
                    # via ``NumPyArrayToRaster(value_to_nodata=np.nan)``.
                    arcpy.AddMessage("Handling NoData values...")
                    data_array = data_array.astype(float)

                    # Defence 1 — explicit raster-level noDataValue.
                    try:
                        if hasattr(raster_obj, 'noDataValue') and raster_obj.noDataValue is not None:
                            no_data = raster_obj.noDataValue
                            for i in range(data_array.shape[2]):
                                data_array[:, :, i][data_array[:, :, i] == no_data] = np.nan
                            arcpy.AddMessage(f"Applied NoData value: {no_data}")
                        else:
                            arcpy.AddMessage("No explicit raster-level NoData value.")
                    except Exception as e:
                        arcpy.AddWarning(f"Error applying explicit NoData: {str(e)}")

                    # Defence 2 — ``arcpy.sa.IsNull`` on the whole
                    # multi-band raster, then OR across bands to get
                    # a single 2D "any band is NoData" mask. This is
                    # Pro's native NoData detection: it sees through
                    # band-level masks, GDB raster fill values and
                    # any other storage form that ``Raster.noDataValue``
                    # might fail to surface. Calling on the multi-band
                    # raster directly avoids the ``{path}/Band_1``
                    # syntax that fails for layer-name inputs
                    # (``ERROR 000732`` we hit earlier in this
                    # session).
                    try:
                        is_null_raster = arcpy.sa.IsNull(arcpy.Raster(raster_path))
                        null_arr_raw = arcpy.RasterToNumPyArray(is_null_raster)
                        # Collapse the multi-band IsNull result into
                        # a single (h, w) "any band is NoData" mask.
                        if null_arr_raw.ndim == 3:
                            # Bands axis is the leading one in arcpy's
                            # ``RasterToNumPyArray`` multi-band output.
                            null_2d = null_arr_raw.any(axis=0).astype(bool)
                        else:
                            null_2d = null_arr_raw.astype(bool)
                        n_null = int(null_2d.sum())
                        if n_null:
                            data_array[null_2d] = np.nan
                            n_total = null_2d.size
                            arcpy.AddMessage(
                                f"IsNull NoData: {n_null} "
                                f"({100*n_null/n_total:.1f}%) pixel(s) masked."
                            )
                    except Exception as e:
                        arcpy.AddWarning(f"IsNull NoData detection failed: {e}")

                    # Defence 3 — all-band-zero numpy check. Last
                    # resort for rasters with no NoData metadata at
                    # all but with outside-AOI fill at value 0 (the
                    # U16 default Pro writes when there is no
                    # ``noDataValue`` declared).
                    try:
                        all_zero = np.all(np.nan_to_num(data_array) == 0, axis=-1)
                        n_zero = int(all_zero.sum())
                        if n_zero:
                            data_array[all_zero] = np.nan
                            n_total = all_zero.size
                            arcpy.AddMessage(
                                f"All-band-zero NoData: {n_zero} "
                                f"({100*n_zero/n_total:.1f}%) pixel(s) masked."
                            )
                    except Exception as e:
                        arcpy.AddWarning(f"All-band-zero detection failed: {e}")

                    arcpy.AddMessage("Handled NoData values")
                    
                    # Perform transformation
                    arcpy.AddMessage(
                        f"\nPerforming {transform_type} transformation "
                        f"(eigendecomposition + projection — duration scales "
                        f"with valid-pixel count × band count²)..."
                    )
                    transform_start = datetime.now()

                    # ISS-010: warn before allocating the (n_pixels, n_bands)
                    # working matrix on country-scale rasters. Full chunked
                    # PCA/MNF is a separate research project; this surfaces
                    # the risk so the user can tile the AOI instead.
                    est_gb = (data_array.shape[0] * data_array.shape[1]
                              * data_array.shape[2] * 8) / 1e9
                    if est_gb > _RAM_WARNING_GB:
                        arcpy.AddWarning(
                            f"Input cube is ~{est_gb:.1f} GB as float64 "
                            f"(shape {data_array.shape}); the transform may "
                            f"exhaust RAM. Consider tiling the AOI or reducing "
                            f"the band count."
                        )

                    if transform_type == "PCA":
                        transformed_data, transform_stats = self._perform_pca(
                            data_array,
                            num_components,
                            stats,
                            message_callback=arcpy.AddMessage,
                            warning_callback=arcpy.AddWarning,
                        )
                    elif transform_type == "MNF":
                        # Process noise subset if provided
                        noise_data = None
                        if noise_subset:
                            arcpy.AddMessage(f"Extracting noise statistics from subset: {noise_subset}")
                            try:
                                # Create temporary raster from first band for extraction
                                temp_raster_path = os.path.join(arcpy.env.scratchWorkspace or "memory", "temp_raster")
                                temp_raster = arcpy.NumPyArrayToRaster(
                                    data_array[:,:,0],
                                    lower_left_corner=arcpy.Point(extent.XMin, extent.YMin),
                                    x_cell_size=cell_size[0],
                                    y_cell_size=cell_size[1]
                                )
                                temp_raster.save(temp_raster_path)
                                
                                # Extract by mask
                                subset = arcpy.sa.ExtractByMask(temp_raster_path, noise_subset)
                                
                                # Convert to numpy array
                                subset_array = arcpy.RasterToNumPyArray(subset)
                                
                                # If subset extraction was successful, prepare noise data
                                if subset_array.size > 0:
                                    noise_data = data_array.copy()
                                    arcpy.AddMessage(f"Using noise subset")
                                else:
                                    arcpy.AddWarning("Subset extraction yielded no valid pixels, using full image for noise estimation")
                                    
                                # Clean up
                                if arcpy.Exists(temp_raster_path) and "memory" not in temp_raster_path:
                                    arcpy.management.Delete(temp_raster_path)
                                    
                            except Exception as e:
                                arcpy.AddWarning(f"Error processing noise subset: {str(e)}")
                                arcpy.AddWarning("Using full image for noise estimation")
                                stats['errors'].append(f"Noise subset error: {str(e)}")
                        
                        transformed_data, transform_stats = self._perform_mnf(
                            data_array,
                            num_components,
                            noise_stats_file,
                            noise_data,
                            stats,
                            message_callback=arcpy.AddMessage,
                            warning_callback=arcpy.AddWarning,
                        )
                    else:  # ICA
                        transformed_data, transform_stats = self._perform_ica(
                            data_array,
                            num_components,
                            stats,
                            random_state=random_state,
                            message_callback=arcpy.AddMessage,
                            warning_callback=arcpy.AddWarning,
                        )
                    
                    arcpy.AddMessage(
                        f"  {transform_type} computed in "
                        f"{(datetime.now() - transform_start).total_seconds():.1f}s."
                    )

                    # Create multiband output
                    arcpy.AddMessage(
                        "\nWriting multiband output raster "
                        "(NumPyArrayToRaster + CompositeBands per component)..."
                    )
                    output_start = datetime.now()
                    output_path = self._create_multiband_output(
                        transformed_data,
                        raster_info,
                        out_path,
                        preserve_mask,
                    )
                    arcpy.AddMessage(
                        f"  Output written in "
                        f"{(datetime.now() - output_start).total_seconds():.1f}s."
                    )
                    
                    # Define data_info for statistics files
                    data_info = {
                        'input_rasters': input_rasters,
                        'output_path': output_path
                    }
                    
                    # Save statistics if requested. Always write BOTH a
                    # reloadable .npz (so the fitted transform can be
                    # re-applied to a new AOI) and a human-readable .txt
                    # summary (for at-a-glance inspection in any editor).
                    if save_stats:
                        arcpy.AddMessage(f"Reloadable statistics: {stats_file_npz}")
                        if not transform_stats.save(stats_file_npz):
                            for err in transform_stats.errors:
                                arcpy.AddWarning(
                                    f"{transform_type}Statistics.save: {err}"
                                )

                        arcpy.AddMessage(f"Human-readable summary: {stats_file_txt}")
                        if transform_type == "PCA":
                            self._save_pca_statistics_txt(
                                stats_file_txt, data_info, transform_stats,
                            )
                        elif transform_type == "MNF":
                            self._save_mnf_statistics_txt(
                                stats_file_txt, data_info, transform_stats,
                            )
                        else:  # ICA
                            self._save_ica_statistics_txt(
                                stats_file_txt, data_info, transform_stats,
                            )

                        # HTML dashboard. Detect the sensor from the first
                        # input raster so the loadings heatmap can label
                        # rows with satellite band names ("B04 (Red)")
                        # rather than generic "Band 4".
                        try:
                            sensor_const = detect_sensor(input_rasters[0]) if input_rasters else None
                            sensor_to_layout = {
                                SENSOR_LANDSAT_89: "landsat",
                                SENSOR_SENTINEL2: "sentinel-2",
                                SENSOR_ASTER: "aster",
                            }
                            sensor_key = sensor_to_layout.get(sensor_const)
                            # 3-band ASTER VNIR mosaics use a separate layout.
                            if sensor_key == "aster" and (
                                stats_obj_bands := getattr(transform_stats, "band_means", None)
                            ) is not None and len(stats_obj_bands) == 3:
                                sensor_key = "aster-vnir"
                            self._generate_html_report(
                                stats_file_html, transform_type, transform_stats,
                                data_info, sensor_key=sensor_key,
                                transformed_data=transformed_data,
                            )
                        except Exception as e:
                            arcpy.AddWarning(
                                f"HTML report generation failed (non-fatal): {e}"
                            )
                    
                    # Update final statistics
                    stats['end_time'] = datetime.now()
                    stats['processing_time'] = (
                        stats['end_time'] - stats['start_time']
                    ).total_seconds()
                    
                    arcpy.AddMessage("\nProcessing completed successfully!")
                    arcpy.AddMessage(f"Total processing time: {stats['processing_time']:.2f} seconds")
                    
                    return output_path
                    
                except Exception as e:
                    arcpy.AddError(f"Error loading or processing raster: {str(e)}")
                    import traceback
                    arcpy.AddError(traceback.format_exc())
                    return None
                    
            except Exception as e:
                arcpy.AddError(f"Error in transformation: {str(e)}")
                stats['errors'].append(str(e))
                # Print detailed traceback for debugging
                import traceback
                arcpy.AddError(traceback.format_exc())
                raise
                
        finally:
            # Clean up the composite temp raster (Bug 2 fix). Wrapped in its
            # own try so a delete failure does not mask a real exception
            # propagating from the body of execute().
            if composite_temp_path:
                try:
                    if arcpy.Exists(composite_temp_path):
                        arcpy.management.Delete(composite_temp_path)
                except Exception:
                    pass

            # Check in extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    def _perform_mnf(self, data: np.ndarray, n_components: int,
                     noise_stats_file: str, noise_data: np.ndarray,
                     stats: dict,
                     message_callback=None, warning_callback=None) -> tuple[np.ndarray, MNFStatistics]:
        """
        Perform MNF transformation. Logging is routed through the optional
        callbacks (ISS-011) so this can be exercised without arcpy.
        """
        log = message_callback or (lambda msg: None)
        warn = warning_callback or (lambda msg: None)
        try:
            # Initialize statistics objects
            noise_stats = MNFNoiseStatistics()
            mnf_stats = MNFStatistics()

            # Load noise statistics if provided
            if noise_stats_file:
                log(f"Loading noise statistics from: {noise_stats_file}")
                if not noise_stats.load(noise_stats_file):
                    warn("Failed to load noise statistics file, estimating noise instead")
                    noise_stats_file = None

            # If noise statistics weren't loaded, estimate them
            if not noise_stats_file:
                log("Estimating noise statistics...")

                # Use noise subset data if provided, otherwise use full data
                noise_source = noise_data if noise_data is not None else data

                # Handle shape for noise data
                if len(noise_source.shape) == 3:
                    noise_flat = noise_source.reshape(-1, noise_source.shape[-1])
                else:
                    noise_flat = noise_source

                # Remove NaN values
                valid_mask = ~np.isnan(noise_flat).any(axis=1)
                valid_noise = noise_flat[valid_mask]

                if valid_noise.shape[0] < noise_flat.shape[1] * 10:
                    warn(f"Low sample count for noise estimation: {valid_noise.shape[0]} samples "
                         f"for {noise_flat.shape[1]} bands. Results may be unstable.")

                # Estimate noise covariance.
                if noise_data is None:
                    log("Using shift difference method for noise estimation...")
                    pixel_valid = ~np.isnan(data).any(axis=-1)
                    _, noise_covariance, n_pairs = noise_from_valid_diffs(data, pixel_valid)
                    if noise_covariance is None:
                        raise RuntimeError(
                            f"Noise estimation failed: insufficient valid neighbour pairs "
                            f"(need at least {data.shape[2] * 2}). Check input mask coverage."
                        )
                    log(f"  Estimated noise from {n_pairs} valid neighbour pairs.")
                else:
                    log("Calculating noise statistics from subset data...")
                    noise_covariance = np.cov(valid_noise, rowvar=False)
                    # Match the regularisation behaviour of noise_from_valid_diffs.
                    eigvals, eigvecs = np.linalg.eigh(noise_covariance)
                    if np.any(eigvals < _EIGVAL_FLOOR_ABS):
                        valid_eig = eigvals[eigvals > _EIGVAL_FLOOR_ABS]
                        floor = (
                            np.median(valid_eig) * _EIGVAL_FLOOR_RELATIVE
                            if len(valid_eig) > 0
                            else _EIGVAL_FLOOR_ABS
                        )
                        eigvals = np.maximum(eigvals, floor)
                        noise_covariance = (eigvecs * eigvals) @ eigvecs.T

                noise_eigenvals, noise_eigenvecs = np.linalg.eigh(noise_covariance)
                noise_stats.noise_covariance = noise_covariance
                noise_stats.noise_eigenvalues = noise_eigenvals
                noise_stats.noise_eigenvectors = noise_eigenvecs
                mnf_stats.noise_covariance = noise_covariance

            # Flatten input data for transformation
            shape = data.shape
            flat_data = data.reshape(-1, shape[2])

            # Handle NaN values
            valid_mask = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[valid_mask]

            log(f"Processing {valid_data.shape[0]} valid pixels for MNF...")

            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean

            log("Applying noise whitening transformation...")
            whitening_matrix = (noise_stats.noise_eigenvectors @
                                np.diag(1.0 / np.sqrt(np.maximum(noise_stats.noise_eigenvalues, _EIGVAL_FLOOR_ABS))) @
                                noise_stats.noise_eigenvectors.T)

            whitened_data = centered_data @ whitening_matrix
            whitened_cov = np.cov(whitened_data.T)
            signal_eigenvals, signal_eigenvecs = np.linalg.eigh(whitened_cov)

            # Sort in descending order
            idx = signal_eigenvals.argsort()[::-1]
            signal_eigenvals = signal_eigenvals[idx]
            signal_eigenvecs = signal_eigenvecs[:, idx]

            # Truncate to requested number of components
            signal_eigenvals = signal_eigenvals[:n_components]
            signal_eigenvecs = signal_eigenvecs[:, :n_components]

            transformed_valid = whitened_data @ signal_eigenvecs

            # ISS-004: report the empirical component correlation. A successful
            # MNF should yield ~identity, since MNF components are orthogonal
            # in the noise-whitened space. Off-diagonal max > 0.1 signals a
            # poor noise estimate or a numerical issue.
            component_correlation = np.corrcoef(transformed_valid.T)
            if component_correlation.ndim == 2 and component_correlation.shape[0] > 1:
                n_comp = component_correlation.shape[0]
                off_diag_max = float(np.max(np.abs(component_correlation - np.eye(n_comp))))
                if off_diag_max > _MNF_CORR_OFFDIAG_WARN:
                    warn(f"MNF component correlation off-diagonal max = {off_diag_max:.3f} "
                         f"(target < {_MNF_CORR_OFFDIAG_WARN}). Components may not be cleanly separated.")
                else:
                    log(f"MNF component correlation OK (off-diagonal max = {off_diag_max:.3f}).")

            log("\nMNF Component Statistics (Signal-to-Noise Ratios):")
            for i in range(n_components):
                log(f"  Component {i+1}: {signal_eigenvals[i]:.4f}")

            # Store MNF statistics
            mnf_stats.band_means = data_mean
            mnf_stats.eigenvalues = signal_eigenvals
            mnf_stats.eigenvectors = signal_eigenvecs
            mnf_stats.transform_matrix = whitening_matrix @ signal_eigenvecs
            mnf_stats.whitening_matrix = whitening_matrix
            mnf_stats.signal_covariance = whitened_cov
            mnf_stats.component_correlation = component_correlation

            # Reconstruct full data array. Initialise with NaN so the
            # outside-mask pixels (where the input had NoData) carry
            # forward as NoData through the save path
            # (``NumPyArrayToRaster(value_to_nodata=np.nan)``); the
            # earlier ``np.zeros`` initialisation produced a hard
            # zero sentinel that survived the save and rendered as
            # "0.0000" in the corners of the result extent.
            transformed_data = np.full(
                (flat_data.shape[0], n_components), np.nan, dtype=float,
            )
            transformed_data[valid_mask] = transformed_valid
            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)

            return transformed_data, mnf_stats

        except Exception as e:
            stats['errors'].append(f"MNF Error: {str(e)}")
            raise

    def _perform_pca(self, data: np.ndarray, n_components: int, stats: dict,
                     message_callback=None, warning_callback=None) -> tuple[np.ndarray, PCAStatistics]:
        """
        Perform PCA transformation.

        The arcpy logging calls are routed through `message_callback` /
        `warning_callback` so this method is unit-testable without ArcGIS
        Pro. Callers running inside the toolbox must pass arcpy.AddMessage
        / arcpy.AddWarning explicitly; tests can pass no-op callables.
        """
        log = message_callback or (lambda msg: None)
        try:
            # Initialize statistics object
            pca_stats = PCAStatistics()

            # Flatten data and center
            shape = data.shape
            flat_data = data.reshape(-1, shape[2])

            # Handle NaN values
            is_not_nan = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[is_not_nan]

            log(f"Computing PCA on {valid_data.shape[0]} valid pixels...")
            
            # Center the data
            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean
            
            # Calculate covariance matrix
            covariance_matrix = np.cov(centered_data, rowvar=False)
            
            # Calculate eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
            
            # Sort in descending order
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Truncate to requested number of components
            eigenvals = eigenvals[:n_components]
            eigenvecs = eigenvecs[:, :n_components]
            
            # Calculate explained variance
            total_var = np.sum(eigenvals)
            explained_variance = eigenvals / total_var
            
            # Transform valid data
            transformed_valid = centered_data @ eigenvecs

            # Reconstruct full data array. NaN-initialised so
            # ``NumPyArrayToRaster(value_to_nodata=np.nan)`` later
            # turns the out-of-mask pixels into proper NoData on
            # save (rather than the hard-zero sentinel a
            # ``np.zeros`` init would leave behind).
            transformed_data = np.full(
                (flat_data.shape[0], n_components), np.nan, dtype=float,
            )
            transformed_data[is_not_nan] = transformed_valid

            # Store PCA statistics
            pca_stats.band_means = data_mean
            pca_stats.eigenvalues = eigenvals
            pca_stats.eigenvectors = eigenvecs
            pca_stats.explained_variance = explained_variance
            pca_stats.covariance_matrix = covariance_matrix
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance)
            log("\nPCA Component Statistics:")
            for i in range(n_components):
                log(f"  Component {i+1}: {explained_variance[i]*100:.2f}% variance " +
                    f"(Cumulative: {cumulative_variance[i]*100:.2f}%)")
            
            # Reshape back to original dimensions
            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)
            
            return transformed_data, pca_stats
            
        except Exception as e:
            stats['errors'].append(f"PCA Error: {str(e)}")
            raise

    def _perform_ica(self, data: np.ndarray, n_components: int, stats: dict,
                     random_state: int = 42,
                     message_callback=None, warning_callback=None) -> tuple[np.ndarray, ICAStatistics]:
        """
        Perform ICA transformation with kurtosis metrics.

        ISS-007: random_state is a parameter (default 42) and is persisted on
        ICAStatistics so a fresh fit is reproducible across runs of the same
        code version. Bit-identical reproduction across toolbox versions is
        not guaranteed (ICA results are unique only up to sign + permutation
        of components), but stored unmixing/mixing matrices replay the
        original fit exactly via _apply_ica_transform.
        ISS-011: logging is routed through the optional callbacks.
        """
        log = message_callback or (lambda msg: None)
        try:
            ica_stats = ICAStatistics()

            shape = data.shape
            flat_data = data.reshape(-1, shape[2])

            is_not_nan = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[is_not_nan]

            log(f"Processing {valid_data.shape[0]} valid pixels for ICA...")

            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean

            cov_matrix = np.cov(centered_data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            whitening = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, _ICA_WHITENING_FLOOR))) @ eigenvectors.T
            dewhitening = eigenvectors @ np.diag(np.sqrt(eigenvalues))

            whitened = centered_data @ whitening.T

            log(f"Performing FastICA (random_state={random_state})...")

            # Pure-numpy FastICA (logcosh, parallel, symmetric decorrelation).
            # Returns all components; we truncate to n_components below so
            # downstream shapes match the original sklearn-based path.
            transformed_valid, W_full, ica_n_iter = _fast_ica_numpy(
                whitened,
                max_iter=_FAST_ICA_MAX_ITER,
                tol=_FAST_ICA_TOL,
                random_state=random_state,
            )
            transformed_valid = transformed_valid[:, :n_components]
            W = W_full[:n_components, :]

            # Kurtosis (peakedness) per component — high |kurtosis| ≈ more
            # independent / non-Gaussian, used by select_by_kurtosis.
            kurtosis_values = np.array([
                scipy.stats.kurtosis(transformed_valid[:, i], fisher=True)
                for i in range(n_components)
            ])

            independence_metrics = np.array([
                (np.mean((transformed_valid[:, i] - np.mean(transformed_valid[:, i])) ** 4)
                 / (np.var(transformed_valid[:, i]) ** 2)) - 3
                for i in range(n_components)
            ])

            ica_stats.kurtosis_values = kurtosis_values
            ica_stats.independence_metrics = independence_metrics

            # unmixing maps centred original-space data to source space.
            unmixing = W @ whitening
            mixing = np.linalg.pinv(unmixing)
            
            # Store statistics
            ica_stats.band_means = data_mean
            ica_stats.mixing_matrix = mixing
            ica_stats.unmixing_matrix = unmixing
            ica_stats.whitening_matrix = whitening
            ica_stats.dewhitening_matrix = dewhitening
            ica_stats.n_iterations = ica_n_iter
            ica_stats.random_state = random_state

            # NaN-init for proper NoData propagation on save — see
            # the matching comment in ``_perform_pca`` / ``_perform_mnf``.
            transformed_data = np.full(
                (flat_data.shape[0], n_components), np.nan, dtype=float,
            )
            transformed_data[is_not_nan] = transformed_valid

            log(f"ICA completed in {ica_n_iter} iterations")

            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)

            return transformed_data, ica_stats

        except Exception as e:
            stats['errors'].append(f"ICA Error: {str(e)}")
            raise
        
    def _create_multiband_output(self, component_arrays, raster_info, out_path, preserve_mask=True):
        """
        Create a multiband raster from component arrays

        Parameters:
        -----------
        component_arrays : ndarray
            Component arrays of shape (height, width, components)
        raster_info : dict
            Raster information containing extent, cell size, etc.
        out_path : str
            Final destination for the multiband raster (already resolved
            by the caller via ``_build_workspace_subfolder_path`` —
            includes the ``pca`` / ``mnf`` / ``ica`` subfolder + ``.tif``
            for folder workspaces, or the flat workspace path for .gdb).
        preserve_mask : bool
            Whether to preserve the input mask

        Returns:
        --------
        str
            Path to the output multiband raster (same as ``out_path``).
        """
        try:
            arcpy.AddMessage("\nCreating multiband output...")

            # Extract raster information
            extent = raster_info['extent']
            cell_size = raster_info['cell_size']
            spatial_ref = raster_info['spatial_ref']
            mask = raster_info.get('mask', None)

            # Paths for temporary component rasters
            temp_component_paths = []

            # Scratch dir for per-component temp TIFFs. We anchor it to
            # the parent of the final output path's *workspace* (one
            # level up from the .tif's containing subfolder, or the .gdb
            # / folder workspace itself when saving flat) — same-disk-as-
            # output, avoids OneDrive sync overhead from
            # arcpy.env.scratchFolder.
            out_dir = os.path.dirname(os.path.normpath(out_path))
            temp_dir = os.path.join(
                os.path.dirname(out_dir) or out_dir,
                f"_genesis_components_{uuid.uuid4().hex[:8]}",
            )
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # Save each component as a temporary raster
            num_components = component_arrays.shape[2]
            for i in range(num_components):
                temp_path = os.path.join(temp_dir, f"temp_comp_{i+1}_{uuid.uuid4().hex}.tif")
                
                # Create raster from array
                out_raster = arcpy.NumPyArrayToRaster(
                    in_array=component_arrays[:,:,i],
                    lower_left_corner=arcpy.Point(
                        extent.XMin, 
                        extent.YMin
                    ),
                    x_cell_size=cell_size[0],
                    y_cell_size=cell_size[1],
                    value_to_nodata=np.nan
                )
                
                # Apply mask if requested and available
                if preserve_mask and mask is not None:
                    arcpy.AddMessage(f"  Applying mask to component {i+1}")
                    try:
                        # Convert mask to raster if it's not already
                        if not isinstance(mask, arcpy.Raster):
                            mask_raster = arcpy.Raster(mask)
                        else:
                            mask_raster = mask
                            
                        # Create SetNull expression
                        masked_raster = arcpy.sa.SetNull(mask_raster == 0, out_raster)
                        masked_raster.save(temp_path)
                    except Exception as e:
                        arcpy.AddWarning(f"  Error applying mask to component {i+1}: {str(e)}")
                        out_raster.save(temp_path)
                else:
                    # Save without masking
                    out_raster.save(temp_path)
                
                # Set spatial reference
                arcpy.DefineProjection_management(
                    temp_path,
                    spatial_ref
                )
                
                temp_component_paths.append(temp_path)
                arcpy.AddMessage(f"  Created component {i+1}")
            
            # Create multiband raster using Composite Bands at the
            # caller-resolved destination.
            arcpy.AddMessage(f"Creating final multiband raster: {out_path}")
            arcpy.management.CompositeBands(temp_component_paths, out_path)
            output_path = out_path
            
            # Clean up temporary files + the per-run scratch dir
            # itself. ``arcpy.management.Delete`` removes each .tif
            # along with its auxiliary sidecars (.aux.xml, .tfw, ...),
            # but leaves the parent directory empty. ``os.rmdir``
            # closes it out cleanly — fails silently if the directory
            # still has unexpected residue (kept defensive so a
            # CompositeBands edge case can't break the whole tool).
            arcpy.AddMessage("Cleaning up temporary files...")
            for temp_path in temp_component_paths:
                try:
                    arcpy.management.Delete(temp_path)
                except arcpy.ExecuteError:
                    pass
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

            return output_path
            
        except Exception as e:
            arcpy.AddError(f"Error creating multiband output: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    @staticmethod
    def _get_pyplot():
        """Lazy headless-matplotlib import. Returns ``pyplot`` on
        success or ``None`` if matplotlib is missing (in which case
        callers skip HTML report generation with a warning). Re-imports
        are cheap — Python caches the module after the first call.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            return plt
        except Exception:
            return None

    @staticmethod
    def _plot_to_base64_png(plt, fig, dpi=120):
        """Render a Matplotlib figure to a base64-encoded PNG data URI.

        Used by ``_generate_html_report`` to embed every plot directly
        in the HTML output — no sidecar PNGs to ship, the report is one
        self-contained file. Closes the figure to release backend
        resources so a long ICA/MNF run does not accumulate handles.
        """
        import base64
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _generate_html_report(self, html_file, transform_type, stats_obj,
                               data_info, sensor_key=None, transformed_data=None):
        """Write a self-contained HTML dashboard summarising a fitted
        PCA / MNF / ICA result.

        The report is a static HTML page with embedded base64 PNGs —
        renders in any browser, works offline, no external assets. It
        complements (does not replace) the ``.npz`` reloadable archive
        and the ``.txt`` numerical summary that the tool also writes.
        When ``sensor_key`` is supplied, loadings heatmap rows are
        labelled with the satellite band names from ``_BAND_LAYOUT``
        (``B04 (Red)`` rather than ``Band 4``) — the same UX win the
        ``_bands.csv`` sidecar gives outside this report. When
        ``transformed_data`` is supplied (the (h, w, n_components)
        array produced by the transform), the report also embeds
        per-component value histograms.
        """
        plt = self._get_pyplot()
        if plt is None:
            arcpy.AddWarning(
                "HTML report skipped — matplotlib not available in this "
                "Python environment."
            )
            return

        layout = _BAND_LAYOUT.get(sensor_key or "") if sensor_key else None
        input_band_labels = (
            [f"{row[1]} ({row[2]})" for row in layout] if layout else None
        )

        # ---- Build the per-transform plots ----
        plot_blocks = []  # list of (title, data_uri, optional caption)
        cut_text = None   # one-line "where to cut" summary, surfaced
                          # into the run-metadata block

        if transform_type == "PCA":
            ev = np.asarray(stats_obj.eigenvalues, dtype=float)
            evec = np.asarray(stats_obj.eigenvectors, dtype=float)
            total = ev.sum() if ev.sum() else 1.0
            var_pct = 100.0 * ev / total
            cumvar = np.cumsum(var_pct)

            # Where to cut: number of components needed to cross
            # 90 / 95 / 99 % of total variance. Reports a concise
            # recommendation in the run-metadata block and drops a
            # dashed line on the scree plot.
            cuts = {}
            for thr in (90.0, 95.0, 99.0):
                idxs = np.where(cumvar >= thr)[0]
                cuts[thr] = (int(idxs[0]) + 1) if idxs.size else len(ev)
            cut95 = cuts[95.0]
            cut_text = (
                f"{cuts[90]} component(s) capture 90% of variance, "
                f"{cuts[95]} reach 95%, {cuts[99]} reach 99% "
                f"(Jolliffe rule of thumb: keep enough components for "
                f"~95% of the variance)."
            )

            # Scree plot + the 95% recommended-cut line
            n_show = min(20, len(ev))
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.bar(range(1, n_show + 1), var_pct[:n_show],
                    color="#3498db", label="Variance per component (%)")
            ax1.set_xlabel("Component"); ax1.set_ylabel("Variance (%)", color="#3498db")
            ax1.tick_params(axis="y", labelcolor="#3498db")
            ax2 = ax1.twinx()
            ax2.plot(range(1, n_show + 1), cumvar[:n_show],
                     color="#e74c3c", marker="o", label="Cumulative (%)")
            ax2.set_ylabel("Cumulative (%)", color="#e74c3c")
            ax2.set_ylim(0, 105)
            ax2.tick_params(axis="y", labelcolor="#e74c3c")
            if cut95 <= n_show:
                ax1.axvline(cut95 + 0.5, color="#27ae60", linestyle="--",
                            linewidth=1.5, alpha=0.7)
                ax2.annotate(
                    f"95% at PC{cut95}", xy=(cut95 + 0.5, 95),
                    xytext=(min(cut95 + 1.5, n_show), 80),
                    fontsize=9, color="#27ae60",
                    arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1),
                )
            fig.suptitle(f"PCA Scree Plot — top {n_show} components")
            fig.tight_layout()
            plot_blocks.append((
                "Scree plot",
                self._plot_to_base64_png(plt, fig),
                f"Variance per component + cumulative ({cumvar[n_show-1]:.1f}% at PC{n_show}). "
                f"Dashed green line marks the 95% cut.",
            ))

            # Cumulative-variance curve with 90/95/99 % cuts. Sits
            # alongside the scree plot because users routinely pick
            # the component count from this view.
            plot_blocks.append(
                self._cumulative_variance_block(plt, ev, cuts)
            )

            # Loadings heatmap
            plot_blocks.append(
                self._loadings_heatmap_block(
                    plt, "PCA loadings", evec, input_band_labels, n_components=n_show,
                )
            )

        elif transform_type == "MNF":
            ev = np.asarray(stats_obj.eigenvalues, dtype=float)
            evec = np.asarray(stats_obj.eigenvectors, dtype=float)
            snr = ev - 1.0
            n_show = min(20, len(ev))
            n_signal = int(np.sum(snr >= 1.0))
            cut_text = (
                f"{n_signal} of {len(snr)} components have SNR ≥ 1 "
                f"(carry more signal than noise). Components with "
                f"SNR < 1 are predominantly noise and can usually be "
                f"discarded for downstream analysis."
            )

            # SNR (signal-to-noise) plot with the SNR = 1 cut line
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#27ae60" if s >= 1.0 else "#bdc3c7" for s in snr[:n_show]]
            ax.bar(range(1, n_show + 1), snr[:n_show], color=colors)
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="SNR = 1 (signal floor)")
            if n_signal and n_signal <= n_show:
                ax.axvline(n_signal + 0.5, color="#27ae60", linestyle="--",
                           linewidth=1.5, alpha=0.7,
                           label=f"Last signal component (MNF{n_signal})")
            ax.set_xlabel("MNF component"); ax.set_ylabel("SNR (eigenvalue − 1)")
            ax.set_title(f"MNF SNR — top {n_show} components")
            ax.legend(loc="upper right")
            fig.tight_layout()
            plot_blocks.append((
                "Signal-to-noise plot",
                self._plot_to_base64_png(plt, fig),
                "Components above the red line (SNR ≥ 1) carry more signal than noise — "
                f"{n_signal} component(s) qualify here.",
            ))

            plot_blocks.append(
                self._loadings_heatmap_block(
                    plt, "MNF loadings", evec, input_band_labels, n_components=n_show,
                )
            )

        else:  # ICA
            kurt = (np.asarray(stats_obj.kurtosis_values, dtype=float)
                    if stats_obj.kurtosis_values is not None else np.array([]))
            mixing = np.asarray(stats_obj.mixing_matrix, dtype=float)
            n_show = mixing.shape[1] if mixing.size else 0
            n_iter = getattr(stats_obj, "n_iterations", None)
            n_indep = int(np.sum(np.abs(kurt) > 1.0)) if kurt.size else 0
            cut_text = (
                f"{n_indep} of {len(kurt) if kurt.size else 0} components have "
                f"|kurtosis| > 1 — reliably non-Gaussian, the ICA target. "
                f"Convergence: {n_iter} iteration(s)."
                if kurt.size else "No kurtosis data available."
            )

            if kurt.size:
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ["#27ae60" if abs(k) > 1.0 else "#bdc3c7" for k in kurt]
                ax.bar(range(1, len(kurt) + 1), kurt, color=colors)
                ax.axhline(0, color="black", linewidth=0.5)
                ax.axhline(1.0, color="red", linestyle="--", linewidth=1,
                           label="|kurtosis| = 1 (non-Gaussian threshold)")
                ax.axhline(-1.0, color="red", linestyle="--", linewidth=1)
                ax.set_xlabel("ICA component")
                ax.set_ylabel("Kurtosis (Gaussian → 0)")
                ax.set_title("ICA kurtosis — non-Gaussianity per component")
                ax.legend(loc="upper right")
                fig.tight_layout()
                plot_blocks.append((
                    "Kurtosis",
                    self._plot_to_base64_png(plt, fig),
                    f"Components with |kurtosis| > 1 (green) are reliably non-Gaussian — "
                    f"the ICA target. {n_indep} of {len(kurt)} qualify here.",
                ))

            if mixing.size:
                plot_blocks.append(
                    self._loadings_heatmap_block(
                        plt, "ICA mixing matrix (A)", mixing,
                        input_band_labels, n_components=n_show,
                    )
                )

        # ---- Cross-transform diagnostics ----

        # Input-band correlation matrix (heatmap). Derived from the
        # covariance matrix stored on the stats object when available
        # (PCA / MNF), otherwise skipped. Surfaces strong inter-band
        # correlations which explain why one PC tends to dominate.
        cov = getattr(stats_obj, "covariance_matrix", None)
        if cov is None:
            cov = getattr(stats_obj, "signal_covariance", None)
        if cov is not None:
            try:
                plot_blocks.insert(0, self._input_correlation_block(
                    plt, np.asarray(cov, dtype=float), input_band_labels,
                ))
            except Exception as _e:
                arcpy.AddWarning(f"Correlation heatmap skipped: {_e}")

        # Per-component value histograms (small-multiples grid). Pulls
        # the values straight from ``transformed_data`` so the
        # distribution shape comes from real pixels, not from a
        # gaussian approximation. Skipped when transformed_data is
        # not passed in.
        if transformed_data is not None:
            try:
                plot_blocks.append(self._component_histograms_block(
                    plt, transformed_data, transform_type,
                ))
            except Exception as _e:
                arcpy.AddWarning(f"Histograms grid skipped: {_e}")

        # ---- Compose the HTML ----
        run_ts = datetime.now().isoformat(timespec="seconds")
        n_components = (
            len(stats_obj.eigenvalues) if hasattr(stats_obj, "eigenvalues")
            and stats_obj.eigenvalues is not None else "-"
        )
        n_input_bands = (
            len(stats_obj.band_means) if stats_obj.band_means is not None else "-"
        )
        inputs_html = "".join(
            f"<li><code>{p}</code></li>" for p in data_info.get("input_rasters", [])
        ) or "<li>(none recorded)</li>"

        # Extra run-metadata derived from transformed_data + the stats
        # object. Stays defensive — anything that can't be computed
        # falls back to the dash placeholder used by the earlier
        # version of the report.
        if transformed_data is not None:
            try:
                td = np.asarray(transformed_data)
                h, w = td.shape[:2]
                n_total = int(h * w)
                # A valid pixel is one that has no NaN across components
                # — matches the (valid_mask / is_not_nan) gate that the
                # transforms apply on the input.
                if td.ndim == 3:
                    valid_mask_2d = ~np.isnan(td).any(axis=2)
                else:
                    valid_mask_2d = ~np.isnan(td)
                n_valid = int(valid_mask_2d.sum())
                shape_card = f"{h} × {w}"
                pct_valid = (100.0 * n_valid / n_total) if n_total else 0.0
                valid_card = f"{n_valid:,} ({pct_valid:.1f}%)"
            except Exception:
                shape_card = "—"
                valid_card = "—"
        else:
            shape_card = "—"
            valid_card = "—"

        # Transform-specific extra metadata (ICA convergence, etc.).
        extra_meta_rows = []
        if transform_type == "ICA":
            n_iter = getattr(stats_obj, "n_iterations", None)
            rs = getattr(stats_obj, "random_state", None)
            if n_iter is not None:
                extra_meta_rows.append(("ICA convergence", f"{n_iter} iteration(s)"))
            if rs is not None:
                extra_meta_rows.append(("ICA random seed", f"{rs}"))

        extra_meta_html = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in extra_meta_rows
        )

        # "Where to cut" recommendation card. Each transform branch
        # populated ``cut_text`` above with a one-line summary derived
        # from its own diagnostics (PCA: 95% variance, MNF: SNR ≥ 1,
        # ICA: |kurtosis| > 1).
        recommendation_html = (
            f'<div class="recommend"><b>Recommended cut:</b> {cut_text}</div>'
            if cut_text else ""
        )

        html_chunks = []
        html_chunks.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{transform_type} report — {os.path.basename(data_info.get('output_path', 'output'))}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #2c3e50; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                  gap: 12px; margin: 18px 0; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 14px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08); text-align: center; }}
  .stat-value {{ font-size: 1.9em; font-weight: 600; color: #2980b9; }}
  .stat-label {{ color: #7f8c8d; font-size: 0.85em; margin-top: 4px; }}
  .plot-card {{ background: white; border-radius: 8px; padding: 16px; margin: 18px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
  .plot-card img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
  .caption {{ color: #7f8c8d; font-size: 0.9em; margin-top: 8px; text-align: center; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0;
           background: white; border-radius: 8px; overflow: hidden;
           box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
  th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f0f7ff; }}
  .meta {{ font-size: 0.9em; color: #5d6d7e; }}
  .recommend {{ background: #e8f5e9; border-left: 4px solid #27ae60;
                padding: 12px 16px; border-radius: 4px; margin: 18px 0;
                font-size: 0.95em; }}
  code {{ font-family: Consolas, monospace; font-size: 0.9em; color: #2c3e50; }}
</style></head><body><div class="container">
<h1>{transform_type} report</h1>
<div class="meta">
  Output raster: <code>{data_info.get('output_path', '?')}</code><br>
  Generated: {run_ts}
</div>
<div class="stats-grid">
  <div class="stat-card"><div class="stat-value">{transform_type}</div>
    <div class="stat-label">Transform</div></div>
  <div class="stat-card"><div class="stat-value">{n_input_bands}</div>
    <div class="stat-label">Input bands</div></div>
  <div class="stat-card"><div class="stat-value">{n_components}</div>
    <div class="stat-label">Components fitted</div></div>
  <div class="stat-card"><div class="stat-value">{sensor_key or "?"}</div>
    <div class="stat-label">Sensor</div></div>
  <div class="stat-card"><div class="stat-value">{shape_card}</div>
    <div class="stat-label">Raster shape (h × w)</div></div>
  <div class="stat-card"><div class="stat-value">{valid_card}</div>
    <div class="stat-label">Valid pixels (in mask)</div></div>
</div>
{recommendation_html}
<h2>Input rasters</h2>
<ul>{inputs_html}</ul>
""")

        if extra_meta_rows:
            html_chunks.append(
                "<h2>Run details</h2><table>"
                "<tr><th>Property</th><th>Value</th></tr>"
                + extra_meta_html + "</table>"
            )

        # Top-N component table (eigenvalue / variance / cumulative)
        if hasattr(stats_obj, "eigenvalues") and stats_obj.eigenvalues is not None:
            ev = np.asarray(stats_obj.eigenvalues, dtype=float)
            total = ev.sum() if ev.sum() else 1.0
            var_pct = 100.0 * ev / total
            cumvar = np.cumsum(var_pct)
            n_top = min(20, len(ev))
            row_html = []
            for i in range(n_top):
                row_html.append(
                    f"<tr><td>{transform_type}{i+1}</td>"
                    f"<td>{ev[i]:.6f}</td>"
                    f"<td>{var_pct[i]:.4f}</td>"
                    f"<td>{cumvar[i]:.2f}</td></tr>"
                )
            html_chunks.append(
                "<h2>Top components</h2><table>"
                "<tr><th>Component</th><th>Eigenvalue</th><th>Variance (%)</th>"
                "<th>Cumulative (%)</th></tr>"
                + "".join(row_html) + "</table>"
            )

        for title, img_uri, caption in plot_blocks:
            html_chunks.append(
                f'<div class="plot-card"><h2>{title}</h2>'
                f'<img src="{img_uri}" alt="{title}">'
                f'<div class="caption">{caption}</div></div>'
            )

        html_chunks.append("</div></body></html>")
        try:
            with open(html_file, "w", encoding="utf-8") as fh:
                fh.write("".join(html_chunks))
            arcpy.AddMessage(f"HTML report: {html_file}")
        except OSError as e:
            arcpy.AddWarning(f"Could not write HTML report ({e})")

    def _cumulative_variance_block(self, plt, eigenvalues, cuts):
        """Render the cumulative variance curve as a PNG with vertical
        marker lines at the 90 / 95 / 99 % cuts. ``cuts`` is the dict
        produced upstream — ``{threshold_pct: component_count}``. The
        caption summarises the cuts in plain text.
        """
        ev = np.asarray(eigenvalues, dtype=float)
        total = ev.sum() if ev.sum() else 1.0
        var_pct = 100.0 * ev / total
        cumvar = np.cumsum(var_pct)
        n_show = min(20, len(ev))
        x = np.arange(1, n_show + 1)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x, cumvar[:n_show], color="#e74c3c", marker="o",
                linewidth=2, label="Cumulative variance (%)")
        ax.fill_between(x, 0, cumvar[:n_show], color="#e74c3c", alpha=0.08)

        cut_colors = {90.0: "#f39c12", 95.0: "#27ae60", 99.0: "#2980b9"}
        for thr, n_comp in cuts.items():
            if n_comp <= n_show:
                color = cut_colors.get(thr, "#7f8c8d")
                ax.axhline(thr, color=color, linestyle=":", linewidth=1, alpha=0.7)
                ax.axvline(n_comp, color=color, linestyle=":", linewidth=1, alpha=0.7)
                ax.scatter([n_comp], [thr], color=color, s=70, zorder=5)
                ax.annotate(
                    f"{int(thr)}% at PC{n_comp}",
                    xy=(n_comp, thr),
                    xytext=(n_comp + 0.4, thr - 6),
                    fontsize=9, color=color,
                )

        ax.set_xlabel("Number of components retained")
        ax.set_ylabel("Cumulative variance (%)")
        ax.set_ylim(0, 105)
        ax.set_xlim(0.5, n_show + 0.5)
        ax.set_title("Cumulative variance — where to cut")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        return (
            "Cumulative variance",
            self._plot_to_base64_png(plt, fig),
            f"Component counts to capture 90 / 95 / 99% of the input "
            f"variance: PC{cuts[90.0]} / PC{cuts[95.0]} / PC{cuts[99.0]} "
            f"(Jolliffe rule of thumb: retain enough components for ~95%).",
        )

    def _input_correlation_block(self, plt, cov_matrix, input_band_labels):
        """Render the input-band Pearson correlation matrix as a
        diverging heatmap. Derived from the covariance matrix stored
        on the stats object: ``corr[i,j] = cov[i,j] / sqrt(cov[i,i] *
        cov[j,j])``. Strong off-diagonal correlations (which the
        Landsat / Sentinel-2 visible+NIR bands always have) explain
        why a single principal component tends to dominate the
        variance budget.
        """
        cov = np.asarray(cov_matrix, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
            return ("Input band correlation", "", "(insufficient data)")

        n = cov.shape[0]
        std = np.sqrt(np.diag(cov))
        # Guard zero-variance bands so we don't divide by zero.
        denom = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, 0.0)
        corr = np.clip(corr, -1.0, 1.0)

        labels = (input_band_labels[:n] if input_band_labels
                  else [f"Band {i+1}" for i in range(n)])
        side = max(4.5, 0.45 * n + 2.0)
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        # Annotate cells with their value so the magnitude is readable
        # without sampling the colorbar by eye.
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{corr[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "#2c3e50",
                    fontsize=max(6, min(10, 80 // n)),
                )
        ax.set_title("Input band correlation matrix (Pearson)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="correlation")
        fig.tight_layout()

        # Strongest off-diagonal magnitude — surfaces the "all bands
        # are basically the same" pattern that explains a dominant PC1.
        off = corr.copy()
        np.fill_diagonal(off, 0.0)
        peak = float(np.abs(off).max()) if off.size else 0.0

        return (
            "Input band correlation",
            self._plot_to_base64_png(plt, fig),
            f"Pearson correlation between input bands. Strongest "
            f"off-diagonal magnitude = {peak:.2f}. Values near ±1 "
            f"indicate redundancy across bands, which is what drives "
            f"the eigendecomposition to concentrate variance into a "
            f"single dominant component.",
        )

    def _component_histograms_block(self, plt, transformed_data, transform_type):
        """Render a small-multiples grid of per-component value
        histograms. Skips NaN (the outside-mask pixels). One panel
        per component, up to 12 (a 3x4 grid).
        """
        td = np.asarray(transformed_data, dtype=float)
        if td.ndim != 3:
            return ("Per-component histograms", "", "(unexpected data shape)")
        n_comp = td.shape[2]
        n_show = min(12, n_comp)
        ncols = 4
        nrows = (n_show + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.6 * nrows + 0.3))
        axes = np.atleast_1d(axes).ravel()
        for i in range(n_show):
            ax = axes[i]
            col = td[:, :, i].ravel()
            col = col[~np.isnan(col)]
            if col.size == 0:
                ax.set_visible(False)
                continue
            ax.hist(col, bins=60, color="#3498db", alpha=0.8)
            ax.set_title(f"{transform_type}{i+1}", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.set_yticks([])
        # Hide unused axes in the bottom row when n_show is not a
        # multiple of ncols.
        for j in range(n_show, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Per-component value distribution (valid pixels only)",
                     fontsize=12, y=1.0)
        fig.tight_layout()

        return (
            "Per-component histograms",
            self._plot_to_base64_png(plt, fig),
            "Distribution of pixel values for each component over the "
            "valid-data footprint. Sharp peaks at zero usually mean "
            "the component is dominated by ground; long tails point to "
            "rare features (anomalies, outliers — the candidates for "
            "downstream analysis).",
        )

    def _loadings_heatmap_block(self, plt, title, matrix, input_band_labels,
                                 n_components=20):
        """Render the input-band → component loadings as a heatmap PNG
        and return a ``(title, data_uri, caption)`` tuple ready for
        ``_generate_html_report`` to drop into the HTML.

        ``matrix`` is the eigenvector / mixing matrix with shape
        ``(input_bands, components)``. When ``input_band_labels`` is
        supplied (from ``_BAND_LAYOUT``), the y-axis is labelled with
        the satellite band names — otherwise generic ``Band N``.
        """
        m = np.asarray(matrix, dtype=float)
        if m.size == 0:
            return (title, "", "(no data)")
        n_comp = min(int(n_components), m.shape[1])
        m_show = m[:, :n_comp]
        n_input = m_show.shape[0]
        labels = (input_band_labels[:n_input] if input_band_labels
                  else [f"Band {i+1}" for i in range(n_input)])

        height = max(3.0, 0.35 * n_input + 1.2)
        fig, ax = plt.subplots(figsize=(min(12, 0.6 * n_comp + 3), height))
        vmax = float(np.abs(m_show).max() or 1.0)
        im = ax.imshow(m_show, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(n_comp))
        ax.set_xticklabels([f"C{i+1}" for i in range(n_comp)])
        ax.set_yticks(range(n_input))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Component")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85, label="loading")
        fig.tight_layout()
        return (
            title,
            self._plot_to_base64_png(plt, fig),
            "Diverging palette — blue = negative loading, red = positive. "
            "Strong colours mark which input bands drive each component.",
        )

    def _save_pca_statistics_txt(self, stats_file, data_info, pca_stats):
        """Save PCA statistics as a human-readable .txt summary.

        Mirror of ``_save_mnf_statistics_txt`` / ``_save_ica_statistics_txt`` —
        the fitted matrices live alongside this in the ``.npz``
        reloadable archive; the ``.txt`` is the at-a-glance summary that
        opens in any text editor and lists the band means, covariance
        matrix, eigenvalues + eigenvectors and the percent / cumulative
        variance explained.
        """
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Data file produced by Principal Component Analysis (PCA) Transform\n")
                f.write("#\tInput raster(s):\n")
                for raster_path in data_info.get('input_rasters', []):
                    f.write(f"#\t\t{raster_path}\n")
                f.write(f"#\tThe number of components = {len(pca_stats.eigenvalues)}\n")
                f.write(f"#\tOutput raster(s):\n")
                f.write(f"#\t\t{data_info.get('output_path', 'Not specified')}\n\n")

                num_bands = len(pca_stats.band_means)

                # Band Means
                f.write("#                    BAND MEANS\n\n")
                f.write("# Input Band   Mean\n")
                f.write("#  --------------------------------------------------------------------------\n")
                for i, mean in enumerate(pca_stats.band_means):
                    f.write(f"        {i+1:d}       {mean:14.6e}\n")
                f.write("#  ==========================================================================\n\n")

                # Covariance Matrix
                if pca_stats.covariance_matrix is not None:
                    f.write("#                    COVARIANCE MATRIX\n\n")
                    f.write("#    Layer         " + "".join([f"{i+1:14d}" for i in range(num_bands)]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(pca_stats.covariance_matrix.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in pca_stats.covariance_matrix[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")

                # Eigenvalues + Eigenvectors
                f.write("#                    EIGENVALUES + EIGENVECTORS\n\n")
                f.write("# Number of Input Layers     Number of PCA Component Layers\n")
                f.write(f"            {num_bands:d}                              {len(pca_stats.eigenvalues):d}\n")
                f.write("# PCA Layer        " + "".join([f"{i+1:14d}" for i in range(len(pca_stats.eigenvalues))]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                f.write("# Eigenvalues\n")
                f.write("               " + "".join([f"{x:14.5f}" for x in pca_stats.eigenvalues]) + "\n")

                f.write("# Eigenvectors (input-band loadings; columns = components)\n")
                f.write("# Input Layer\n")
                for i in range(pca_stats.eigenvectors.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.5f}" for x in pca_stats.eigenvectors[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")

                # Percent and Accumulative Variance Explained
                f.write("#                 PERCENT AND ACCUMULATIVE VARIANCE EXPLAINED\n\n")
                f.write("# PCA Layer   EigenValue   Percent of Variance   Accumulative of Variance\n")

                total_var = sum(pca_stats.eigenvalues)
                cumulative = 0.0
                for i, ev in enumerate(pca_stats.eigenvalues):
                    percent = (ev / total_var) * 100.0 if total_var else 0.0
                    cumulative += percent
                    f.write(f"        {i+1:d} {ev:14.5f}          {percent:7.4f}               {cumulative:7.4f}\n")
                f.write("#  ==========================================================================\n")
        except Exception as e:
            arcpy.AddWarning(f"Error saving PCA statistics to text file: {str(e)}")

    def _save_mnf_statistics_txt(self, stats_file, data_info, mnf_stats):
        """
        Save MNF statistics in text format
        
        Parameters:
        -----------
        stats_file : str
            Path to output statistics file
        data_info : dict
            Information about the input data
        mnf_stats : MNFStatistics
            MNF statistics object
        """
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Data file produced by Minimum Noise Fraction (MNF) Transform\n")
                f.write("#\tInput raster(s):\n")
                for raster_path in data_info.get('input_rasters', []):
                    f.write(f"#\t\t{raster_path}\n")
                f.write(f"#\tThe number of components = {len(mnf_stats.eigenvalues)}\n")
                f.write(f"#\tOutput raster(s):\n")
                f.write(f"#\t\t{data_info.get('output_path', 'Not specified')}\n\n")
                
                # Noise Covariance Matrix
                f.write("#                    NOISE COVARIANCE MATRIX\n\n")
                num_bands = len(mnf_stats.band_means)
                f.write("#    Layer         " + "".join([f"{i+1:14d}" for i in range(num_bands)]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                # Include noise covariance if available
                noise_cov = mnf_stats.noise_covariance if hasattr(mnf_stats, 'noise_covariance') else None
                if noise_cov is not None:
                    for i in range(noise_cov.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in noise_cov[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Noise Whitening Matrix
                if hasattr(mnf_stats, 'whitening_matrix') and mnf_stats.whitening_matrix is not None:
                    f.write("#                    NOISE WHITENING MATRIX\n\n")
                    f.write("#    Output Band   " + "".join([f"{i+1:14d}" for i in range(num_bands)]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.whitening_matrix.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in mnf_stats.whitening_matrix[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Signal Covariance Matrix (after whitening)
                if hasattr(mnf_stats, 'signal_covariance') and mnf_stats.signal_covariance is not None:
                    f.write("#                    SIGNAL COVARIANCE MATRIX (AFTER WHITENING)\n\n")
                    f.write("#    Band          " + "".join([f"{i+1:14d}" for i in range(mnf_stats.signal_covariance.shape[0])]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.signal_covariance.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in mnf_stats.signal_covariance[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Signal-to-Noise Ratio (Eigenvalues)
                f.write("#                 SIGNAL-TO-NOISE RATIOS (EIGENVALUES)\n\n")
                f.write("# Number of Input Layers     Number of MNF Component Layers\n")
                f.write(f"            {len(mnf_stats.band_means):d}                              {len(mnf_stats.eigenvalues):d}\n")
                f.write("# MNF Layer        " + "".join([f"{i+1:14d}" for i in range(len(mnf_stats.eigenvalues))]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                f.write("# Eigenvalues\n")
                f.write("               " + "".join([f"{x:14.5f}" for x in mnf_stats.eigenvalues]) + "\n")
                
                # Eigenvectors
                f.write("# Eigenvectors\n")
                f.write("# Input Layer\n")
                for i in range(mnf_stats.eigenvectors.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.5f}" for x in mnf_stats.eigenvectors[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Component Correlation Matrix
                if hasattr(mnf_stats, 'component_correlation') and mnf_stats.component_correlation is not None:
                    f.write("#                 COMPONENT CORRELATION MATRIX\n")
                    f.write("# This matrix should be close to an identity matrix (diagonal = 1, off-diagonal ≈ 0)\n\n")
                    f.write("# Component      " + "".join([f"{i+1:14d}" for i in range(len(mnf_stats.eigenvalues))]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.component_correlation.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.5f}" for x in mnf_stats.component_correlation[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Percent and Accumulative Eigenvalues
                f.write("#                 PERCENT AND ACCUMULATIVE EIGENVALUES\n\n")
                f.write("# MNF Layer   EigenValue   Percent of EigenValues   Accumulative of EigenValues\n")
                
                total_eigenvalue = sum(mnf_stats.eigenvalues)
                cumulative = 0.0
                for i, eigenvalue in enumerate(mnf_stats.eigenvalues):
                    percent = (eigenvalue / total_eigenvalue) * 100.0
                    cumulative += percent
                    f.write(f"        {i+1:d} {eigenvalue:14.5f}          {percent:7.4f}               {cumulative:7.4f}\n")
                f.write("#  ==========================================================================\n")
                
        except Exception as e:
            arcpy.AddWarning(f"Error saving MNF statistics to text file: {str(e)}")

    def _save_ica_statistics_txt(self, stats_file, data_info, ica_stats):
        """
        Save ICA statistics in text format
        
        Parameters:
        -----------
        stats_file : str
            Path to output statistics file
        data_info : dict
            Information about the input data
        ica_stats : ICAStatistics
            ICA statistics object
        """
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Data file produced by Independent Component Analysis (ICA) Transform\n")
                f.write("#\tInput raster(s):\n")
                for raster_path in data_info.get('input_rasters', []):
                    f.write(f"#\t\t{raster_path}\n")
                f.write(f"#\tThe number of components = {ica_stats.mixing_matrix.shape[1]}\n")
                f.write(f"#\tOutput raster(s):\n")
                f.write(f"#\t\t{data_info.get('output_path', 'Not specified')}\n")
                f.write(f"#\tConverged in {ica_stats.n_iterations} iterations\n\n")
                
                # Mixing Matrix
                f.write("#                    MIXING MATRIX (A)\n")
                f.write("# Independent components are reconstructed as X = AS, where A is the mixing matrix and S are the sources\n\n")
                f.write("#    Component    " + "".join([f"{i+1:14d}" for i in range(ica_stats.mixing_matrix.shape[1])]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                for i in range(ica_stats.mixing_matrix.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.6f}" for x in ica_stats.mixing_matrix[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Unmixing Matrix
                f.write("#                    UNMIXING MATRIX (W)\n")
                f.write("# Sources are computed as S = WX, where W is the unmixing matrix and X is the data\n\n")
                f.write("#    Source       " + "".join([f"{i+1:14d}" for i in range(ica_stats.unmixing_matrix.shape[1])]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                for i in range(ica_stats.unmixing_matrix.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.6f}" for x in ica_stats.unmixing_matrix[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Component Independence (approximate mutual information)
                f.write("#                 COMPONENT INDEPENDENCE MEASURES\n\n")
                f.write("# Component   Mutual Information   Independence Score\n")
                
                # Calculate simple independence metrics
                # In actual ICA implementation, mutual information would be calculated
                # Here we're providing placeholder values
                if hasattr(ica_stats, 'independence_metrics') and ica_stats.independence_metrics is not None:
                    for i, metric in enumerate(ica_stats.independence_metrics):
                        # 0-100 score driven by ABSOLUTE departure from
                        # Gaussianity. The raw metric (excess kurtosis) is
                        # negative for sub-Gaussian components (common in
                        # smooth-terrain remote sensing) — taking abs keeps
                        # the user-facing score in [0, 100] regardless of sign.
                        indep_score = 100 * (1 - np.exp(-abs(metric)))
                        f.write(f"        {i+1:d}         {metric:8.4f}           {indep_score:8.4f}\n")
                else:
                    # Fallback if metrics aren't available
                    for i in range(ica_stats.mixing_matrix.shape[1]):
                        f.write(f"        {i+1:d}         {'N/A':8s}           {'N/A':8s}\n")
                f.write("#  ==========================================================================\n")
                              
                # Kurtosis Values
                f.write("# COMPONENT KURTOSIS VALUES\n")
                f.write("# Component   Kurtosis\n")
                for i, kurt in enumerate(ica_stats.kurtosis_values):
                    f.write(f"        {i+1:d}         {kurt:8.4f}\n")
                f.write("#  ==========================================================================\n")
                
        except Exception as e:
            arcpy.AddWarning(f"Error saving ICA statistics to text file: {str(e)}")

    def _extract_subset_data(self, data: np.ndarray, subset_feature: str) -> np.ndarray:
        """
        Extract data from specified spatial subset
        """
        try:
            # Create temporary raster from numpy array
            temp_raster = arcpy.NumPyArrayToRaster(data[:,:,0])
            
            # Extract by mask
            subset = arcpy.sa.ExtractByMask(temp_raster, subset_feature)
            
            # Convert back to numpy array
            subset_data = arcpy.RasterToNumPyArray(subset)
            
            return subset_data
            
        except Exception as e:
            arcpy.AddWarning(f"Error extracting subset: {str(e)}")
            return None

# Tool 4: Spectral Angle Mapper
class SpectralAngleMapper(object):
    """Tool 06 — Spectral Angle Mapper classification.

    Sensor-aware via the Sensor Type parameter, which drives the
    expected input band count and validates reference-spectra column
    counts. The SAM algorithm itself is sensor-agnostic.

    Carries the Phase 5 SAM SetNull inversion fix (was: classifier
    initialised NoData over valid pixels, leaving the final raster
    blank). Other SAM audit findings (color-map application, endmember
    GPLong precision, class-index determinism, class-field heuristic
    fallback) are documented but NOT yet fixed — they're queued as a
    follow-up phase.
    """

    def __init__(self):
        self.label = "06 — Spectral Angle Mapper"
        self.description = (
            "Spectral Angle Mapper classification on a pre-stacked "
            "multiband raster from any of the three supported sensors. "
            "Reference spectra can be supplied as a table, training "
            "samples, or endmember pixels. The Sensor Type parameter "
            "drives band-count validation."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        # Input raster
        input_raster = arcpy.Parameter(
            displayName="Input Multiband Raster",
            name="input_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )
        
        # Reference spectra source
        ref_source = arcpy.Parameter(
            displayName="Reference Spectra Source",
            name="ref_source",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        ref_source.filter.list = ["Table", "ROIs/Training Samples", "Endmember Pixels"]
        ref_source.value = "Table"
        
        # Reference spectra table (for Table option)
        ref_table = arcpy.Parameter(
            displayName="Reference Spectra Table",
            name="ref_table",
            datatype="DETable",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        
        # Training samples (for ROIs option)
        training_samples = arcpy.Parameter(
            displayName="Training Samples/ROIs",
            name="training_samples",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        # Endmember pixels (for Endmember option)
        endmember_pixels = arcpy.Parameter(
            displayName="Endmember Pixels (x,y coordinates)",
            name="endmember_pixels",
            datatype="GPValueTable",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        endmember_pixels.columns = [['GPString', 'Class Name'], ['GPLong', 'X'], ['GPLong', 'Y']]
        
        # Class names field (for Table option)
        class_field = arcpy.Parameter(
            displayName="Class Names Field",
            name="class_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        class_field.parameterDependencies = [ref_table.name]
        class_field.filter.list = ['Text']
        
        # Band fields (for Table option)
        band_fields = arcpy.Parameter(
            displayName="Band Value Fields",
            name="band_fields",
            datatype="Field",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
            enabled=True
        )
        band_fields.parameterDependencies = [ref_table.name]
        band_fields.filter.list = ['Double', 'Float', 'Long', 'Short']
        
        # Maximum angle
        max_angle = arcpy.Parameter(
            displayName="Maximum Angle (degrees)",
            name="max_angle",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        max_angle.value = 10.0
        max_angle.filter.type = "Range"
        max_angle.filter.list = [0.1, 90.0]
        
        # Threshold
        threshold = arcpy.Parameter(
            displayName="Classification Threshold",
            name="threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        threshold.value = 0.1
        threshold.filter.type = "Range"
        threshold.filter.list = [0.01, 1.0]
        
        # Output workspace
        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        
        # Output classification raster
        out_raster = arcpy.Parameter(
            displayName="Output Classification Raster",
            name="out_raster",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        
        # Output SAM raster
        out_sam = arcpy.Parameter(
            displayName="Output SAM Angle Raster",
            name="out_sam",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        
        # Color scheme
        color_scheme = arcpy.Parameter(
            displayName="Apply Color Scheme",
            name="color_scheme",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        color_scheme.value = True

        # Sensor selector (Phase 6 addition) — drives the expected band
        # count for the input raster and the reference spectra columns.
        sensor_type = make_sensor_parameter()

        return [input_raster, ref_source, ref_table, training_samples, endmember_pixels,
                class_field, band_fields, max_angle, threshold, out_workspace,
                out_raster, out_sam, color_scheme, sensor_type]
    
    def updateParameters(self, parameters):
        """Modify parameter values and properties"""
        # Get references to parameters
        ref_source = parameters[1]
        ref_table = parameters[2]
        training_samples = parameters[3]
        endmember_pixels = parameters[4]
        class_field = parameters[5]
        band_fields = parameters[6]
        
        # Enable/disable parameters based on reference source
        if ref_source.altered:
            if ref_source.value == "Table":
                ref_table.enabled = True
                training_samples.enabled = False
                endmember_pixels.enabled = False
                class_field.enabled = True
                band_fields.enabled = True
            elif ref_source.value == "ROIs/Training Samples":
                ref_table.enabled = False
                training_samples.enabled = True
                endmember_pixels.enabled = False
                class_field.enabled = False
                band_fields.enabled = False
            elif ref_source.value == "Endmember Pixels":
                ref_table.enabled = False
                training_samples.enabled = False
                endmember_pixels.enabled = True
                class_field.enabled = False
                band_fields.enabled = False
        
        # Update output name based on input
        if parameters[0].altered and not parameters[10].altered:
            try:
                input_name = os.path.basename(parameters[0].valueAsText)
                name_parts = os.path.splitext(input_name)
                if name_parts[0]:
                    parameters[10].value = f"{name_parts[0]}_SAM_class"
                    parameters[11].value = f"{name_parts[0]}_SAM_angles"
            except:
                pass
        
    def updateMessages(self, parameters):
        """Modify messages created by internal validation"""
        # Validate reference spectra table
        if parameters[1].value == "Table" and parameters[2].altered:
            try:
                table_path = parameters[2].valueAsText
                if not arcpy.Exists(table_path):
                    parameters[2].setErrorMessage("Reference spectra table does not exist")
                else:
                    # Check for required fields
                    if parameters[5].value and parameters[6].value:
                        # Check class field
                        class_field = parameters[5].valueAsText
                        
                        # Check band fields (ensure they match input raster bands)
                        band_fields = parameters[6].valueAsText.split(";")
                        if len(band_fields) < 2:
                            parameters[6].setWarningMessage("At least 2 band fields should be specified")
            except Exception as e:
                parameters[2].setErrorMessage(f"Error validating reference table: {str(e)}")
        
        # Validate training samples
        if parameters[1].value == "ROIs/Training Samples" and parameters[3].altered:
            try:
                training_path = parameters[3].valueAsText
                if not arcpy.Exists(training_path):
                    parameters[3].setErrorMessage("Training samples do not exist")
                else:
                    # Check for class field
                    desc = arcpy.Describe(training_path)
                    if not any(field.name.lower() in ["class", "classname", "class_name"] for field in desc.fields):
                        parameters[3].setWarningMessage("No class field found. Expected fields: Class, ClassName, or Class_Name")
            except Exception as e:
                parameters[3].setErrorMessage(f"Error validating training samples: {str(e)}")
    
    def execute(self, parameters, messages):
        """Execute the tool"""
        try:
            # Check out extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required but not available")
                return
                
            # Enable overwrite
            arcpy.env.overwriteOutput = True
            
            # Get parameters
            input_raster = parameters[0].valueAsText
            ref_source = parameters[1].valueAsText
            ref_table = parameters[2].valueAsText if parameters[1].value == "Table" else None
            training_samples = parameters[3].valueAsText if parameters[1].value == "ROIs/Training Samples" else None
            endmember_pixels = parameters[4].value if parameters[1].value == "Endmember Pixels" else None
            class_field = parameters[5].valueAsText if parameters[1].value == "Table" and parameters[5].value else None
            band_fields = parameters[6].valueAsText.split(";") if parameters[1].value == "Table" and parameters[6].value else []
            max_angle = parameters[7].value
            threshold = parameters[8].value
            out_workspace = parameters[9].valueAsText
            out_raster = parameters[10].valueAsText
            out_sam = parameters[11].valueAsText if parameters[11].value else None
            apply_color = parameters[12].value
            
            # Convert max angle to radians
            max_angle_rad = max_angle * (3.14159265359 / 180.0)
            
            # Process based on reference source
            if ref_source == "Table":
                # Perform SAM with reference table
                self._sam_with_table(
                    input_raster=input_raster,
                    ref_table=ref_table,
                    class_field=class_field,
                    band_fields=band_fields,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            elif ref_source == "ROIs/Training Samples":
                # Perform SAM with training samples
                self._sam_with_training(
                    input_raster=input_raster,
                    training_samples=training_samples,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            elif ref_source == "Endmember Pixels":
                # Perform SAM with endmember pixels
                self._sam_with_endmembers(
                    input_raster=input_raster,
                    endmember_pixels=endmember_pixels,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            
            # Return output path
            out_path = os.path.join(out_workspace, out_raster)
            arcpy.SetParameterAsText(10, out_path)
            
            if out_sam:
                sam_path = os.path.join(out_workspace, out_sam)
                arcpy.SetParameterAsText(11, sam_path)
                
            return out_path
            
        except Exception as e:
            arcpy.AddError(f"Error executing SAM: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
            
        finally:
            # Check in extensions
            arcpy.CheckInExtension("Spatial")
    
    def _sam_with_table(self, input_raster, ref_table, class_field, band_fields,
                        max_angle_rad, threshold, out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using reference spectra from a table"""
        import numpy as np
        import math
        
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage(f"Reference spectra table: {ref_table}")
            
            # Load the input raster
            raster_obj = arcpy.Raster(input_raster)
            band_count = raster_obj.bandCount
            
            # Check if band fields match available bands
            if len(band_fields) > band_count:
                arcpy.AddWarning(f"Reference table has {len(band_fields)} bands, but input raster has {band_count} bands")
                arcpy.AddWarning("Using only the available bands from the reference table")
                band_fields = band_fields[:band_count]
            elif len(band_fields) < band_count:
                arcpy.AddWarning(f"Reference table has {len(band_fields)} bands, but input raster has {band_count} bands")
                arcpy.AddWarning("Some raster bands will not be used in the analysis")
            
            # Load band data into memory
            arcpy.AddMessage("Loading raster bands...")
            bands = {}
            for i in range(1, band_count + 1):
                if i <= len(band_fields):  # Only load bands that have reference data
                    arcpy.AddMessage(f"Loading band {i}...")
                    bands[i] = arcpy.Raster(f"{input_raster}/{i}")
            
            # Read reference spectra from the table
            arcpy.AddMessage("Reading reference spectra...")
            ref_spectra = {}
            class_names = []
            
            with arcpy.da.SearchCursor(ref_table, [class_field] + band_fields) as cursor:
                for row in cursor:
                    class_name = row[0]
                    band_values = [float(row[i+1]) for i in range(len(band_fields))]
                    
                    if class_name in ref_spectra:
                        arcpy.AddWarning(f"Duplicate class name '{class_name}' found in reference table")
                        continue
                    
                    # Normalize reference spectrum
                    magnitude = math.sqrt(sum(v*v for v in band_values))
                    if magnitude > 0:
                        normalized = [v/magnitude for v in band_values]
                        ref_spectra[class_name] = normalized
                        class_names.append(class_name)
                    else:
                        arcpy.AddWarning(f"Reference spectrum for '{class_name}' has zero magnitude and will be ignored")
            
            arcpy.AddMessage(f"Found {len(ref_spectra)} reference spectra:")
            for name in class_names:
                arcpy.AddMessage(f"  - {name}")
            
            if not ref_spectra:
                arcpy.AddError("No valid reference spectra found in the reference table")
                return
            
            # Create output paths
            out_class_path = os.path.join(out_workspace, out_raster)
            out_sam_path = os.path.join(out_workspace, out_sam) if out_sam else None
            
            # Perform SAM calculation
            arcpy.AddMessage(
                f"Building SAM map-algebra expressions for {len(ref_spectra)} "
                f"reference spectra over {len(band_fields)} bands "
                f"(expressions are lazy — pixels evaluate on .save below)..."
            )
            sam_build_start = datetime.now()

            # Create lists for map algebra expressions
            band_expressions = [f"Float('{bands[i+1]}')" for i in range(len(band_fields))]

            # Create SAM rasters for each reference spectrum
            sam_rasters = {}

            # 1. Compute normalization factor for pixel vectors
            norm_expr = " + ".join([f"Power({expr}, 2)" for expr in band_expressions])
            norm_raster = arcpy.sa.SquareRoot(arcpy.sa.Raster(arcpy.sa.Float(norm_expr)))

            # 2. Compute SAM for each reference spectrum
            for ci, (class_name, ref_vector) in enumerate(ref_spectra.items(), 1):
                arcpy.AddMessage(
                    f"  [{ci}/{len(ref_spectra)}] SAM expression for class '{class_name}'"
                )
                # Calculate dot product
                dot_expr = " + ".join([f"{expr} * {ref_vector[i]}" for i, expr in enumerate(band_expressions)])
                dot_raster = arcpy.sa.Float(dot_expr)

                # Calculate SAM angle (arccos of dot product divided by magnitudes)
                # Since reference vectors are already normalized, we only need to normalize the pixel vector
                sam_angle = arcpy.sa.ACos(dot_raster / norm_raster)

                # Convert from radians to degrees
                sam_angle_deg = sam_angle * (180.0 / math.pi)

                # Store the SAM raster
                sam_rasters[class_name] = sam_angle_deg
            arcpy.AddMessage(
                f"  Expression chain built in "
                f"{(datetime.now() - sam_build_start).total_seconds():.1f}s."
            )

            # 3. Create classification raster
            arcpy.AddMessage("Building final classification expression (per-class minimum-angle reduction)...")

            # Initialise the classification/min-angle rasters over valid pixels
            # only. SetNull(cond, false_val) returns NoData where `cond` is True
            # and `false_val` elsewhere; we want NoData on invalid pixels
            # (norm_raster <= 0) and a seed value on the valid ones.
            class_raster = arcpy.sa.SetNull(norm_raster <= 0, 0)
            min_angle_raster = arcpy.sa.SetNull(norm_raster <= 0, 90)

            # Loop through classes to find minimum angle
            for idx, class_name in enumerate(class_names, 1):
                arcpy.AddMessage(
                    f"  [{idx}/{len(class_names)}] folding '{class_name}' into "
                    f"min-angle reduction"
                )
                sam_raster = sam_rasters[class_name]

                # Update classification where this class has smaller angle
                class_raster = arcpy.sa.Con(
                    arcpy.sa.BooleanAnd(
                        sam_raster < min_angle_raster,
                        sam_raster <= max_angle_rad * (180.0 / math.pi)
                    ),
                    idx,
                    class_raster
                )

                # Update minimum angle raster
                min_angle_raster = arcpy.sa.Con(
                    sam_raster < min_angle_raster,
                    sam_raster,
                    min_angle_raster
                )

            # Save output classification raster — THIS is the moment the entire
            # lazy expression chain is evaluated against pixels (read all bands,
            # compute norm, dot products, ACos, minimum reduction). Expect this
            # call to dominate the wall-clock time of the whole tool.
            arcpy.AddMessage(
                f"Materialising classification raster — this evaluates the "
                f"entire SAM expression chain against every pixel. Save target: "
                f"{out_class_path}"
            )
            save_start = datetime.now()
            class_raster.save(out_class_path)
            arcpy.AddMessage(
                f"  Saved in "
                f"{(datetime.now() - save_start).total_seconds():.1f}s."
            )

            # Save SAM angle raster if requested
            if out_sam_path:
                arcpy.AddMessage(f"Saving SAM angle raster to: {out_sam_path}")
                sam_save_start = datetime.now()
                min_angle_raster.save(out_sam_path)
                arcpy.AddMessage(
                    f"  Saved in "
                    f"{(datetime.now() - sam_save_start).total_seconds():.1f}s."
                )
            
            # Apply color map if requested
            if apply_color:
                arcpy.AddMessage("Applying color map to classification raster...")
                
                # Create color map
                color_map = []
                for idx, class_name in enumerate(class_names, 1):
                    # Generate a color based on index
                    hue = (idx * 137) % 360  # Use golden ratio to distribute colors
                    rgb = self._hsv_to_rgb(hue/360.0, 0.8, 0.9)
                    color_map.append([idx, class_name, rgb[0], rgb[1], rgb[2]])
                
                # Apply color map
                try:
                    arcpy.AddMessage("Setting classification symbology...")
                    result = arcpy.management.ApplySymbologyFromLayer(out_class_path, "")
                    
                    # Get the output raster layer
                    layer = result.getOutput(0)
                    
                    # Apply custom color map
                    for entry in color_map:
                        idx, name, r, g, b = entry
                        # Add code to apply color to layer
                        arcpy.AddMessage(f"  Class {idx}: {name} - RGB({r},{g},{b})")
                except Exception as e:
                    arcpy.AddWarning(f"Could not apply color map: {str(e)}")
            
            # Return output paths
            return out_class_path
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with table: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _sam_with_training(self, input_raster, training_samples, max_angle_rad, threshold,
                          out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using training samples/ROIs"""
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage(f"Training samples: {training_samples}")
            
            # Look for class field in training samples
            class_field = None
            desc = arcpy.Describe(training_samples)
            
            # Check possible field names
            possible_fields = ["class", "classname", "class_name", "category", "label"]
            for field in desc.fields:
                if field.type == "String" and field.name.lower() in possible_fields:
                    class_field = field.name
                    break
            
            if not class_field:
                arcpy.AddWarning("No suitable class field found in training samples")
                arcpy.AddWarning("Using first string field as class field")
                
                # Use first string field
                for field in desc.fields:
                    if field.type == "String":
                        class_field = field.name
                        break
            
            if not class_field:
                arcpy.AddError("No string field found in training samples for classification")
                return None
            
            arcpy.AddMessage(f"Using field '{class_field}' for class names")
            
            # Extract values to points
            arcpy.AddMessage("Extracting spectral values from training samples...")
            
            # Create a temporary table to store extracted values
            temp_table = arcpy.CreateUniqueName("sam_training", arcpy.env.scratchGDB)
            arcpy.sa.ExtractValuesToTable(
                training_samples,
                input_raster,
                temp_table,
                "NONE",
                "ALL"
            )
            
            # Add class field to the temporary table
            arcpy.management.AddJoin(
                temp_table,
                "OBJECTID",
                training_samples,
                "OBJECTID",
                "KEEP_ALL"
            )
            
            # Get class names and band fields
            fields = [f.name for f in arcpy.ListFields(temp_table)]
            training_class_field = f"{os.path.basename(training_samples)}.{class_field}"
            
            # Get band fields (BAND_1, BAND_2, etc.)
            band_fields = [f for f in fields if f.startswith("BAND_")]
            
            # Sort band fields numerically
            band_fields.sort(key=lambda x: int(x.split("_")[1]))
            
            # Now use the table-based SAM with our extracted training data
            self._sam_with_table(
                input_raster=input_raster,
                ref_table=temp_table,
                class_field=training_class_field,
                band_fields=band_fields,
                max_angle_rad=max_angle_rad,
                threshold=threshold,
                out_workspace=out_workspace,
                out_raster=out_raster,
                out_sam=out_sam,
                apply_color=apply_color
            )
            
            # Clean up
            try:
                arcpy.management.Delete(temp_table)
            except arcpy.ExecuteError:
                pass
            
            # Return output path
            return os.path.join(out_workspace, out_raster)
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with training samples: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _sam_with_endmembers(self, input_raster, endmember_pixels, max_angle_rad, threshold,
                            out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using endmember pixels"""
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage("Using endmember pixels for reference spectra")
            
            # Create temporary points feature class
            temp_points = arcpy.CreateUniqueName("endmember_points", arcpy.env.scratchGDB)
            
            # Create points feature class
            arcpy.management.CreateFeatureclass(
                os.path.dirname(temp_points),
                os.path.basename(temp_points),
                "POINT",
                spatial_reference=arcpy.Describe(input_raster).spatialReference
            )
            
            # Add CLASS field
            arcpy.management.AddField(temp_points, "CLASS", "TEXT")
            
            # Add points
            with arcpy.da.InsertCursor(temp_points, ["SHAPE@XY", "CLASS"]) as cursor:
                for row in endmember_pixels:
                    class_name = row[0]
                    x = row[1]
                    y = row[2]
                    cursor.insertRow([(x, y), class_name])
            
            # Now use the training samples approach with our temporary points
            self._sam_with_training(
                input_raster=input_raster,
                training_samples=temp_points,
                max_angle_rad=max_angle_rad,
                threshold=threshold,
                out_workspace=out_workspace,
                out_raster=out_raster,
                out_sam=out_sam,
                apply_color=apply_color
            )
            
            # Clean up
            try:
                arcpy.management.Delete(temp_points)
            except arcpy.ExecuteError:
                pass
            
            # Return output path
            return os.path.join(out_workspace, out_raster)
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with endmember pixels: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB"""
        if s == 0.0:
            return (int(v * 255), int(v * 255), int(v * 255))
        
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        
        if i == 0:
            return (int(v * 255), int(t * 255), int(p * 255))
        elif i == 1:
            return (int(q * 255), int(v * 255), int(p * 255))
        elif i == 2:
            return (int(p * 255), int(v * 255), int(t * 255))
        elif i == 3:
            return (int(p * 255), int(q * 255), int(v * 255))
        elif i == 4:
            return (int(t * 255), int(p * 255), int(v * 255))
        else:
            return (int(v * 255), int(p * 255), int(q * 255))


# ---------------------------------------------------------------------------
# Tool 07 — Temporal Composites & Statistics
# ---------------------------------------------------------------------------


class TemporalStatistics(object):
    """Tool 07 — Per-pixel temporal statistics over a mosaic scratch.

    Takes a preserved scratch folder from Tool 01 / 02 / 03 (the
    per-scene cloud-masked multiband stacks) and produces standalone
    single-band rasters summarising the temporal behaviour of each
    pixel. Targeted at the Faial demonstrator's groundwater workflow:
    persistent-canopy mapping (phreatophyte / GDV indicators) plus
    water-occurrence statistics inspired by the JRC Global Surface
    Water mapping framework — but built purely on ``arcpy.sa``
    CellStatistics + Con (cleanroom Apache-licensed implementation).

    Each output indicator maps to a remote-sensing GDV reference so the
    products are bibliographically defensible:

      - ``NDVI_min``, ``NDVI_max``, ``NDVI_mean``, ``NDVI_std``:
        per-pixel temporal moments. Underpin every GDV mapping
        workflow (Pérez Hoyos et al., 2018 review).
      - ``NDVI_persistence``: fraction of scenes with NDVI above a
        biome-tunable threshold. Howard & Merrifield (2010) — California
        GDE mapping.
      - ``NDWI_freq``: fraction of scenes with NDWI > 0. Equivalent to
        the JRC GSW Water Occurrence Frequency (Pekel et al., 2016)
        but per-AOI and computed locally.
      - ``NDWI_max``: extent of any-time open water (max NDWI).
      - ``obs_count``: number of cloud-mask-survived scenes per pixel.
        Confidence map for all the other indicators.

    Per-season mode adds two cross-group composites:

      - ``GDV_ratio = NDVI_mean_dry / NDVI_mean_wet`` (Lv et al., 2013
        — values ≈ 1 indicate no seasonal stress, the GDV signature).
      - ``GDV_dry_floor = NDVI_mean_dry > 0.3`` (Eamus / Naumburg
        framing — sustained dry-season canopy).
    """

    def __init__(self):
        self.label = "07 — Temporal Composites & Statistics"
        self.description = (
            "Per-pixel temporal statistics over a mosaic-tool scratch "
            "folder. Emits NDVI moments (min/max/mean/std + biome-tunable "
            "persistence), NDWI water occurrence and an observation count "
            "map; with Per-season stratification, also emits the Lv (2013) "
            "GDV ratio and the Eamus/Naumburg dry-season NDVI floor. All "
            "outputs are single-band rasters routed into a `temporal/` "
            "subfolder for folder workspaces, or flat at the workspace "
            "root for `.gdb` / `.sde`. Built on arcpy.sa.CellStatistics."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # GP parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        scratch = arcpy.Parameter(
            displayName=(
                "Mosaic Scratch Folder (the `_genesis_*_scratch_*` "
                "directory written by Tools 01-03 when Preserve Scratch "
                "is on)"
            ),
            name="scratch_dir",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )

        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )

        out_prefix = arcpy.Parameter(
            displayName="Output Name Prefix",
            name="out_prefix",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        sensor_type = make_sensor_parameter()

        region = arcpy.Parameter(
            displayName=(
                "Region (only consulted when stratification = "
                "Per season — drives the dry / wet month buckets)"
            ),
            name="region",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )
        region.filter.list = [
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
            "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)",
            "Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)",
            "Angola",
            "Mozambique",
        ]

        stratification = arcpy.Parameter(
            displayName="Temporal Stratification",
            name="stratification",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        stratification.filter.list = ["All scenes", "Per season"]
        stratification.value = "All scenes"

        stack_pattern = arcpy.Parameter(
            displayName=(
                "Stack Filename Pattern (advanced; default auto-detects "
                "across the three mosaic conventions: `*_stack.tif` "
                "(S2 / ASTER 9-band), `*_stack_vnir.tif` (ASTER VNIR), "
                "and `*_composite.tif` (Landsat). Pick a single pattern "
                "to restrict the scan."
            ),
            name="stack_pattern",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        stack_pattern.filter.list = [
            _TOOL07_AUTO_LABEL,
            "*_stack.tif",
            "*_stack_vnir.tif",
            "*_composite.tif",
        ]
        stack_pattern.value = _TOOL07_AUTO_LABEL

        mask_feature = arcpy.Parameter(
            displayName="Optional AOI Mask Feature (polygon)",
            name="mask_feature",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
        )
        mask_feature.filter.list = ["Polygon"]

        save_stats = arcpy.Parameter(
            displayName="Save Provenance CSV",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        save_stats.value = True

        stat_source = arcpy.Parameter(
            displayName="Statistics Source",
            name="stat_source",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        stat_source.filter.list = [
            "NDVI/NDWI (multispectral stacks)",
            "LST (AST_08 thermal)",
            "LST (Landsat ST_B10 thermal)",
        ]
        stat_source.value = "NDVI/NDWI (multispectral stacks)"

        ast08_folder = arcpy.Parameter(
            displayName=(
                "AST_08 Folder (LST AST_08 mode only; walked recursively "
                "for AST_08 TIFFs and HDFs; TIFF preferred when both "
                "formats coexist for the same scene_id)"
            ),
            name="ast08_folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input",
        )

        lst_cool_delta_k = arcpy.Parameter(
            displayName=(
                "LST cool-persistence delta (K below per-scene spatial "
                "mean; default 2.0)"
            ),
            name="lst_cool_delta_k",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        lst_cool_delta_k.value = 2.0

        lst_warm_delta_k = arcpy.Parameter(
            displayName=(
                "LST warm-persistence delta (K above per-scene spatial "
                "mean; default 4.0)"
            ),
            name="lst_warm_delta_k",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input",
            category="Advanced Options",
        )
        lst_warm_delta_k.value = 4.0

        return [
            scratch, out_workspace, out_prefix, sensor_type, region,
            stratification, stack_pattern, mask_feature, save_stats,
            stat_source, ast08_folder, lst_cool_delta_k, lst_warm_delta_k,
        ]

    def updateParameters(self, parameters):
        """Mode-aware enable/disable.

        Region is only consulted under Per-season stratification. The
        Statistics Source choice drives which input set is live:
        NDVI/NDWI needs Mosaic Scratch Folder + Stack Filename Pattern;
        the LST modes need AST_08 Folder + LST persistence deltas.
        """
        try:
            stratification = parameters[5]
            region = parameters[4]
            region.enabled = (stratification.valueAsText == "Per season")

            stat_source = (parameters[9].valueAsText or "")
            scratch = parameters[0]
            stack_pattern = parameters[6]
            ast08_folder = parameters[10]
            lst_cool = parameters[11]
            lst_warm = parameters[12]
            is_lst = stat_source.startswith("LST")
            scratch.enabled = not is_lst
            stack_pattern.enabled = not is_lst
            ast08_folder.enabled = is_lst
            lst_cool.enabled = is_lst
            lst_warm.enabled = is_lst
        except Exception:
            pass

    def updateMessages(self, parameters):
        """Mode-aware required-input validation. ``scratch`` is declared
        Required in getParameterInfo for NDVI/NDWI; for the LST modes we
        forgive the missing scratch and instead require ast08_folder.
        Landsat ST_B10 is parameter-flagged as not yet implemented."""
        try:
            stat_source = (parameters[9].valueAsText or "")
            scratch = parameters[0]
            ast08_folder = parameters[10]
            if stat_source == "LST (AST_08 thermal)":
                scratch.clearMessage()
                if not ast08_folder.valueAsText:
                    ast08_folder.setErrorMessage(
                        "AST_08 Folder is required for the LST (AST_08 "
                        "thermal) mode."
                    )
            elif stat_source == "LST (Landsat ST_B10 thermal)":
                scratch.clearMessage()
                parameters[9].setErrorMessage(
                    "LST (Landsat ST_B10 thermal) mode is not yet "
                    "implemented. Pick NDVI/NDWI or LST (AST_08 thermal)."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                arcpy.AddError("Spatial Analyst extension is required.")
                return None
            arcpy.CheckOutExtension("Spatial")
            arcpy.env.overwriteOutput = True

            scratch_dir = parameters[0].valueAsText
            out_workspace = parameters[1].valueAsText
            out_prefix = (parameters[2].valueAsText or "").strip()
            sensor_param = parameters[3].valueAsText
            region = parameters[4].valueAsText
            stratification = parameters[5].valueAsText or "All scenes"
            stack_pattern = parameters[6].valueAsText or _TOOL07_AUTO_LABEL
            mask_feature = parameters[7].valueAsText
            save_stats = bool(parameters[8].value)
            stat_source = (
                parameters[9].valueAsText
                if len(parameters) > 9 and parameters[9].valueAsText
                else "NDVI/NDWI (multispectral stacks)"
            )
            ast08_folder = (
                parameters[10].valueAsText
                if len(parameters) > 10 else None
            )
            cool_delta_k = (
                float(parameters[11].value)
                if len(parameters) > 11 and parameters[11].value is not None
                else 2.0
            )
            warm_delta_k = (
                float(parameters[12].value)
                if len(parameters) > 12 and parameters[12].value is not None
                else 4.0
            )

            if stack_pattern == _TOOL07_AUTO_LABEL:
                discovery_patterns = list(_TOOL07_AUTO_PATTERNS)
                pattern_display = (
                    f"{_TOOL07_AUTO_LABEL} -> "
                    + ", ".join(_TOOL07_AUTO_PATTERNS)
                )
            else:
                discovery_patterns = [stack_pattern]
                pattern_display = stack_pattern

            arcpy.AddMessage("=" * 60)
            arcpy.AddMessage(f"TEMPORAL STATISTICS — prefix: {out_prefix}")
            arcpy.AddMessage("=" * 60)
            arcpy.AddMessage(f"  Source mode:    {stat_source}")
            arcpy.AddMessage(f"  Output:         {out_workspace}")
            arcpy.AddMessage(f"  Stratification: {stratification}")

            if mask_feature and arcpy.Exists(mask_feature):
                arcpy.env.mask = mask_feature
                arcpy.env.extent = mask_feature
                arcpy.AddMessage(f"  AOI:            {mask_feature}")
            elif mask_feature:
                arcpy.AddWarning(
                    f"  AOI: {mask_feature!r} NOT FOUND — full extent"
                )

            # Mode dispatch. The LST modes share AOI handling and
            # provenance shape with NDVI/NDWI but discover scenes from
            # an AST_08 / Landsat folder rather than a mosaic scratch
            # of multi-band stacks.
            if stat_source == "LST (AST_08 thermal)":
                self._execute_thermal_ast08(
                    ast08_folder=ast08_folder,
                    out_workspace=out_workspace,
                    out_prefix=out_prefix,
                    region=region,
                    stratification=stratification,
                    cool_delta_k=cool_delta_k,
                    warm_delta_k=warm_delta_k,
                    save_stats=save_stats,
                )
                arcpy.AddMessage("\n" + "=" * 60)
                arcpy.AddMessage("DONE")
                arcpy.AddMessage("=" * 60)
                return None
            if stat_source == "LST (Landsat ST_B10 thermal)":
                arcpy.AddError(
                    "LST (Landsat ST_B10 thermal) mode is not yet "
                    "implemented. Pick NDVI/NDWI or LST (AST_08 thermal)."
                )
                return None

            # NDVI/NDWI path. Restore the mode-specific header lines
            # that don't apply to the LST modes.
            arcpy.AddMessage(f"  Scratch:        {scratch_dir}")
            arcpy.AddMessage(f"  Pattern:        {pattern_display}")

            # Discover stacks across all selected patterns and union.
            found = []
            for pat in discovery_patterns:
                found.extend(glob.glob(os.path.join(scratch_dir, pat)))
            stacks = sorted(set(found))
            if not stacks:
                arcpy.AddError(
                    f"No stacks matching {discovery_patterns!r} in "
                    f"{scratch_dir}."
                )
                return None
            arcpy.AddMessage(f"  Scenes found:   {len(stacks)}")

            # Detect sensor from band count of first stack (or honour the
            # user's explicit selection).
            sensor = resolve_sensor(sensor_param, stacks[0])
            if sensor is None:
                arcpy.AddError(
                    "Could not detect sensor. Pick one explicitly via the "
                    "Sensor Type parameter."
                )
                return None
            roles = SENSOR_BAND_ROLES.get(sensor, {})
            if "NIR" not in roles or "Red" not in roles:
                arcpy.AddError(
                    f"Sensor {sensor!r} has no NIR / Red band roles; "
                    "cannot compute NDVI."
                )
                return None
            arcpy.AddMessage(f"  Sensor:         {sensor}")

            # Parse acquisition dates per scene (None when filename
            # doesn't match a known sensor pattern).
            dated = [(p, _scene_date_from_stack_filename(p)) for p in stacks]
            n_dated = sum(1 for _, d in dated if d is not None)
            if n_dated < len(dated):
                arcpy.AddWarning(
                    f"  Dated:          {n_dated}/{len(dated)} scenes "
                    "have a parseable date — the rest will be excluded "
                    "from per-season grouping."
                )

            # Build groups: {"all": [...]} or {"dry": [...], "wet": [...]}.
            seasonal_pattern = self._seasonal_pattern_for_region(region or "")
            groups = self._group_scenes(dated, stratification, seasonal_pattern)
            for g, paths in groups.items():
                arcpy.AddMessage(f"  Group {g!r}: {len(paths)} scene(s)")
                if len(paths) < 2:
                    arcpy.AddWarning(
                        f"  Group {g!r} has only {len(paths)} scene(s); "
                        "stats need ≥ 2 — group will be skipped."
                    )

            # Per-group outputs
            arcpy.AddMessage("\n▶ Computing temporal statistics per group...")
            group_outputs = {}
            for group_name, paths in groups.items():
                if len(paths) < 2:
                    continue
                group_start = datetime.now()
                arcpy.SetProgressor(
                    "default",
                    f"Computing temporal statistics [{group_name}]...",
                )
                outs = self._compute_group_outputs(
                    group_name, paths, sensor, out_prefix, out_workspace,
                    stratified=(stratification == "Per season"),
                )
                arcpy.ResetProgressor()
                group_outputs[group_name] = outs
                arcpy.AddMessage(
                    f"  ✓ Group {group_name!r}: {len(outs)} raster(s) in "
                    f"{(datetime.now() - group_start).total_seconds():.1f}s"
                )

            # Cross-group GDV composites (Per season only)
            if stratification == "Per season" and {"dry", "wet"} <= set(group_outputs.keys()):
                arcpy.AddMessage("\n▶ Computing cross-season GDV composites...")
                gdv_outs = self._compute_gdv_compounds(
                    out_prefix, out_workspace,
                    group_outputs["dry"].get("NDVI_mean"),
                    group_outputs["wet"].get("NDVI_mean"),
                )
                if gdv_outs:
                    arcpy.AddMessage(
                        f"  ✓ {len(gdv_outs)} GDV composite(s) written"
                    )

            # Provenance
            if save_stats:
                self._write_provenance_csv(
                    out_workspace, out_prefix, sensor, stratification,
                    groups, region,
                )

            arcpy.AddMessage("\n" + "=" * 60)
            arcpy.AddMessage("DONE")
            arcpy.AddMessage("=" * 60)

        except Exception as e:
            arcpy.AddError(f"Tool 07 failed: {e}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
        finally:
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seasonal_pattern_for_region(region):
        """Same dispatch as the mosaic tools."""
        if region in (
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
        ):
            return "temperate"
        if "Cape Verde" in (region or ""):
            return "cape_verde"
        if region == "Angola":
            return "angola"
        if region == "Mozambique":
            return "mozambique"
        return "temperate"

    @staticmethod
    def _group_scenes(dated_pairs, stratification, seasonal_pattern):
        """Bucket ``(path, date)`` pairs into groups."""
        if stratification == "All scenes":
            return {"all": [p for p, _ in dated_pairs]}
        groups = {"dry": [], "wet": []}
        for path, date in dated_pairs:
            bucket = _classify_season_bucket(date, seasonal_pattern)
            if bucket is not None:
                groups[bucket].append(path)
        return groups

    def _compute_group_outputs(self, group_name, scene_paths, sensor,
                                prefix, out_workspace, stratified):
        """Build and save the per-group output rasters.

        Returns a dict mapping the indicator name to its on-disk path,
        so the caller (Per-season GDV composites) can look up the dry /
        wet ``NDVI_mean`` paths without re-resolving the names.
        """
        roles = SENSOR_BAND_ROLES[sensor]
        nir_idx = roles["NIR"]
        red_idx = roles["Red"]
        green_idx = roles.get("Green")

        # Suffix only when stratified: "" in All-scenes mode keeps the
        # output names short and unambiguous.
        suffix = f"_{group_name}" if stratified else ""

        # Build lazy NDVI rasters (one per scene). Con guards the
        # division by zero where NIR + Red is null/zero.
        ndvi_list = []
        ndwi_list = []
        valid_mask = []  # for obs_count + as denominator of frequencies
        for path in scene_paths:
            nir = arcpy.sa.Float(arcpy.ia.ExtractBand(path, [nir_idx]))
            red = arcpy.sa.Float(arcpy.ia.ExtractBand(path, [red_idx]))
            denom_v = nir + red
            ndvi = arcpy.sa.SetNull(denom_v == 0, (nir - red) / denom_v)
            ndvi_list.append(ndvi)
            valid_mask.append(
                arcpy.sa.Con(arcpy.sa.IsNull(ndvi), 0, 1)
            )
            if green_idx is not None:
                green = arcpy.sa.Float(arcpy.ia.ExtractBand(path, [green_idx]))
                denom_w = green + nir
                ndwi = arcpy.sa.SetNull(denom_w == 0, (green - nir) / denom_w)
                ndwi_list.append(ndwi)

        outputs = {}

        # ---- NDVI moments ----
        for stat_tag, method in (
            ("NDVI_min", "MINIMUM"),
            ("NDVI_max", "MAXIMUM"),
            ("NDVI_mean", "MEAN"),
            ("NDVI_std", "STD"),
        ):
            out_path = _build_workspace_subfolder_path(
                out_workspace, f"{prefix}_{stat_tag}{suffix}", "temporal",
            )
            try:
                arcpy.sa.CellStatistics(
                    ndvi_list, statistics_type=method, ignore_nodata="DATA",
                ).save(out_path)
                outputs[stat_tag] = out_path
                arcpy.AddMessage(f"    → {os.path.basename(out_path)}")
            except arcpy.ExecuteError as e:
                arcpy.AddWarning(f"    ✗ {stat_tag} failed: {e}")

        # ---- NDVI persistence ----
        above_list = [
            arcpy.sa.Con(ndvi > _NDVI_PERSISTENCE_THRESHOLD, 1, 0)
            for ndvi in ndvi_list
        ]
        n_above = arcpy.sa.CellStatistics(
            above_list, statistics_type="SUM", ignore_nodata="DATA",
        )
        n_valid = arcpy.sa.CellStatistics(
            valid_mask, statistics_type="SUM", ignore_nodata="DATA",
        )
        # obs_count is reused below; save the persistence ratio.
        persistence = arcpy.sa.Con(n_valid > 0, n_above / n_valid)
        persistence_path = _build_workspace_subfolder_path(
            out_workspace, f"{prefix}_NDVI_persistence{suffix}", "temporal",
        )
        try:
            persistence.save(persistence_path)
            outputs["NDVI_persistence"] = persistence_path
            arcpy.AddMessage(f"    → {os.path.basename(persistence_path)}")
        except arcpy.ExecuteError as e:
            arcpy.AddWarning(f"    ✗ NDVI_persistence failed: {e}")

        # ---- obs_count ----
        obs_count_path = _build_workspace_subfolder_path(
            out_workspace, f"{prefix}_obs_count{suffix}", "temporal",
        )
        try:
            n_valid.save(obs_count_path)
            outputs["obs_count"] = obs_count_path
            arcpy.AddMessage(f"    → {os.path.basename(obs_count_path)}")
        except arcpy.ExecuteError as e:
            arcpy.AddWarning(f"    ✗ obs_count failed: {e}")

        # ---- NDWI products (when Green role exists) ----
        if ndwi_list:
            try:
                ndwi_max = arcpy.sa.CellStatistics(
                    ndwi_list, statistics_type="MAXIMUM", ignore_nodata="DATA",
                )
                ndwi_max_path = _build_workspace_subfolder_path(
                    out_workspace, f"{prefix}_NDWI_max{suffix}", "temporal",
                )
                ndwi_max.save(ndwi_max_path)
                outputs["NDWI_max"] = ndwi_max_path
                arcpy.AddMessage(f"    → {os.path.basename(ndwi_max_path)}")
            except arcpy.ExecuteError as e:
                arcpy.AddWarning(f"    ✗ NDWI_max failed: {e}")

            try:
                ndwi_above = [
                    arcpy.sa.Con(ndwi > _NDWI_WATER_THRESHOLD, 1, 0)
                    for ndwi in ndwi_list
                ]
                n_water = arcpy.sa.CellStatistics(
                    ndwi_above, statistics_type="SUM", ignore_nodata="DATA",
                )
                ndwi_freq = arcpy.sa.Con(n_valid > 0, n_water / n_valid)
                ndwi_freq_path = _build_workspace_subfolder_path(
                    out_workspace, f"{prefix}_NDWI_freq{suffix}", "temporal",
                )
                ndwi_freq.save(ndwi_freq_path)
                outputs["NDWI_freq"] = ndwi_freq_path
                arcpy.AddMessage(f"    → {os.path.basename(ndwi_freq_path)}")
            except arcpy.ExecuteError as e:
                arcpy.AddWarning(f"    ✗ NDWI_freq failed: {e}")
        else:
            arcpy.AddMessage(
                "    (NDWI products skipped — sensor has no Green band role)"
            )

        return outputs

    def _compute_gdv_compounds(self, prefix, out_workspace,
                                ndvi_mean_dry_path, ndvi_mean_wet_path):
        """GDV-family cross-season composites (Lv 2013 + Eamus/Naumburg)."""
        outputs = []
        if not (ndvi_mean_dry_path and ndvi_mean_wet_path):
            arcpy.AddWarning(
                "  Missing dry or wet NDVI_mean — GDV composites skipped."
            )
            return outputs

        dry = arcpy.sa.Raster(ndvi_mean_dry_path)
        wet = arcpy.sa.Raster(ndvi_mean_wet_path)

        # Lv 2013 GDV ratio: NDVI_mean_dry / NDVI_mean_wet. Cap the
        # denominator at 0.1 so pixels with near-zero wet-season NDVI
        # (water, bare soil) don't blow up the ratio.
        try:
            gdv_ratio = arcpy.sa.Con(wet > 0.1, dry / wet)
            gdv_ratio_path = _build_workspace_subfolder_path(
                out_workspace, f"{prefix}_GDV_ratio", "temporal",
            )
            gdv_ratio.save(gdv_ratio_path)
            outputs.append(gdv_ratio_path)
            arcpy.AddMessage(f"    → {os.path.basename(gdv_ratio_path)}")
        except arcpy.ExecuteError as e:
            arcpy.AddWarning(f"    ✗ GDV_ratio failed: {e}")

        # Eamus / Naumburg dry-season NDVI floor → binary candidate mask.
        try:
            gdv_dry_floor = arcpy.sa.Con(dry > _GDV_DRY_FLOOR, 1, 0)
            gdv_floor_path = _build_workspace_subfolder_path(
                out_workspace, f"{prefix}_GDV_dry_floor", "temporal",
            )
            gdv_dry_floor.save(gdv_floor_path)
            outputs.append(gdv_floor_path)
            arcpy.AddMessage(f"    → {os.path.basename(gdv_floor_path)}")
        except arcpy.ExecuteError as e:
            arcpy.AddWarning(f"    ✗ GDV_dry_floor failed: {e}")

        return outputs

    @staticmethod
    def _write_provenance_csv(out_workspace, prefix, sensor,
                               stratification, groups, region):
        """One CSV per Tool 07 run listing the scenes that contributed
        to each group, plus the run metadata. Lives next to the first
        temporal output (or alongside the workspace for .gdb)."""
        # Anchor the CSV to the first written output path so it lands
        # in the same folder / sibling location as the rasters.
        anchor = _build_workspace_subfolder_path(
            out_workspace, prefix, "temporal",
        )
        csv_path = _sidecar_path_for_raster(anchor, "_temporal_provenance.csv")
        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "group", "scene_path", "scene_date",
                ])
                for group_name, paths in groups.items():
                    for p in paths:
                        d = _scene_date_from_stack_filename(p)
                        writer.writerow([
                            group_name, p,
                            d.isoformat() if d else "",
                        ])
                writer.writerow([])
                writer.writerow(["run_metadata"])
                writer.writerow(["prefix", prefix])
                writer.writerow(["sensor", sensor])
                writer.writerow(["stratification", stratification])
                writer.writerow(["region", region or ""])
                writer.writerow(["toolbox_version", TOOLBOX_VERSION])
                writer.writerow([
                    "ndvi_persistence_threshold",
                    f"{_NDVI_PERSISTENCE_THRESHOLD}",
                ])
                writer.writerow([
                    "ndwi_water_threshold",
                    f"{_NDWI_WATER_THRESHOLD}",
                ])
                writer.writerow([
                    "gdv_dry_floor",
                    f"{_GDV_DRY_FLOOR}",
                ])
                writer.writerow([
                    "generated", datetime.now().isoformat(timespec="seconds"),
                ])
            arcpy.AddMessage(f"  Provenance CSV: {csv_path}")
        except OSError as e:
            arcpy.AddWarning(f"  Could not write provenance CSV ({e})")

    # ------------------------------------------------------------------
    # LST (AST_08) thermal mode
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_ast08(folder):
        """Walk ``folder`` recursively for AST_08 files. Pair TIFF and
        HDF on the 17-char scene_id (TIFF wins when both formats exist
        for the same scene, matching the AsterMosaic discovery
        convention). Returns ``(paths, metas)`` aligned by index; each
        meta is ``{"scene_id", "acquisition_date", "format"}``.
        """
        by_sid = {}  # scene_id -> list[(fmt, path)]
        for root, _, files in os.walk(folder):
            for name in files:
                scene_id = AsterMosaic._parse_ast08_filename(name)
                if not scene_id:
                    continue
                ext = os.path.splitext(name)[1].lower()
                fmt = "hdf" if ext == ".hdf" else "tiff"
                by_sid.setdefault(scene_id, []).append(
                    (fmt, os.path.join(root, name))
                )

        paths = []
        metas = []
        for sid, candidates in by_sid.items():
            # Prefer TIFF when both formats coexist; fall back to HDF.
            chosen = next(
                ((f, p) for (f, p) in candidates if f == "tiff"),
                candidates[0],
            )
            fmt, path = chosen
            # scene_id layout (17 chars): PPP MM DD YYYY HHMMSS
            try:
                acq = datetime(
                    int(sid[7:11]), int(sid[3:5]), int(sid[5:7]),
                ).date()
            except (ValueError, IndexError):
                acq = None
            paths.append(path)
            metas.append({
                "scene_id": sid,
                "acquisition_date": acq,
                "format": fmt,
            })

        # Sort by acquisition date for deterministic order; undated
        # entries (corrupt filenames) go to the end.
        paired = sorted(
            zip(paths, metas),
            key=lambda pm: pm[1]["acquisition_date"] or datetime.max.date(),
        )
        if paired:
            paths, metas = map(list, zip(*paired))
        return paths, metas

    @staticmethod
    def _materialise_kelvin_group(paths, meta_by_path, scratch, group_name):
        """For each AST_08 path in the group, build the Kelvin raster
        via ``_aster_bt_kelvin_from_path`` at native 90 m, materialise
        to a scratch TIFF, and collect the saved path. AOI clipping
        falls out of the active ``arcpy.env.mask`` / ``env.extent``,
        which are set by the caller. Returns list of saved paths;
        scenes that fail to load are skipped with a warning.
        """
        saved = []
        for p in paths:
            meta = meta_by_path.get(p, {})
            sid = meta.get("scene_id") or "scene"
            try:
                bt = _aster_bt_kelvin_from_path(
                    p, scratch, target_cellsize=None, scene_id=sid,
                )
                out_k = os.path.join(scratch, f"{group_name}_{sid}_K.tif")
                bt.save(out_k)
                saved.append(out_k)
            except Exception as e:
                arcpy.AddWarning(f"    ✗ {sid}: BT load failed ({e}); skipped")
        return saved

    def _compute_thermal_stats(self, group_name, kelvin_paths, out_workspace,
                                prefix, stratified, cool_delta_k, warm_delta_k):
        """CellStatistics min/max/mean/std + LST_obs_count +
        LST_persistence_cool/warm over a group's materialised Kelvin
        rasters. Returns ``{stat_tag: out_path}``.

        Persistence is per-scene-relative: each input scene contributes
        a binary raster (1 where the pixel is colder than the scene's
        spatial mean minus cool_delta_k, or warmer than the mean plus
        warm_delta_k), summed across the group. Per-scene means are
        read from the saved TIFFs via GetRasterProperties, which honours
        env.mask so the scalar is AOI-restricted.
        """
        suffix = f"_{group_name}" if stratified else ""
        outputs = {}

        bt_rasters = [arcpy.sa.Raster(p) for p in kelvin_paths]
        valid_masks = [
            arcpy.sa.Con(arcpy.sa.IsNull(r), 0, 1) for r in bt_rasters
        ]

        # Per-scene spatial mean for the persistence thresholds.
        scene_means = []
        for p in kelvin_paths:
            try:
                m = float(
                    arcpy.management.GetRasterProperties(p, "MEAN").getOutput(0)
                )
            except Exception:
                m = None
            scene_means.append(m)

        # Moments
        for stat_tag, method in (
            ("LST_min", "MINIMUM"),
            ("LST_max", "MAXIMUM"),
            ("LST_mean", "MEAN"),
            ("LST_std", "STD"),
        ):
            out_path = _build_workspace_subfolder_path(
                out_workspace, f"{prefix}_{stat_tag}{suffix}", "temporal",
            )
            try:
                arcpy.sa.CellStatistics(
                    kelvin_paths, statistics_type=method, ignore_nodata="DATA",
                ).save(out_path)
                outputs[stat_tag] = out_path
                arcpy.AddMessage(f"    → {os.path.basename(out_path)}")
            except arcpy.ExecuteError as e:
                arcpy.AddWarning(f"    ✗ {stat_tag} failed: {e}")

        # obs_count
        try:
            obs = arcpy.sa.CellStatistics(
                valid_masks, statistics_type="SUM", ignore_nodata="DATA",
            )
            obs_path = _build_workspace_subfolder_path(
                out_workspace, f"{prefix}_LST_obs_count{suffix}", "temporal",
            )
            obs.save(obs_path)
            outputs["LST_obs_count"] = obs_path
            arcpy.AddMessage(f"    → {os.path.basename(obs_path)}")
        except arcpy.ExecuteError as e:
            arcpy.AddWarning(f"    ✗ LST_obs_count failed: {e}")

        # Persistence (cool, warm)
        cool_list = []
        warm_list = []
        skipped = 0
        for bt, m in zip(bt_rasters, scene_means):
            if m is None:
                skipped += 1
                continue
            cool_list.append(arcpy.sa.Con(bt < (m - cool_delta_k), 1, 0))
            warm_list.append(arcpy.sa.Con(bt > (m + warm_delta_k), 1, 0))
        if skipped:
            arcpy.AddWarning(
                f"    {skipped} scene(s) had no readable spatial mean; "
                "excluded from persistence."
            )
        for stat_tag, plist in (
            ("LST_persistence_cool", cool_list),
            ("LST_persistence_warm", warm_list),
        ):
            if not plist:
                arcpy.AddWarning(
                    f"    ✗ {stat_tag}: no scenes with valid spatial mean"
                )
                continue
            try:
                persistence = arcpy.sa.CellStatistics(
                    plist, statistics_type="SUM", ignore_nodata="DATA",
                )
                out_path = _build_workspace_subfolder_path(
                    out_workspace, f"{prefix}_{stat_tag}{suffix}", "temporal",
                )
                persistence.save(out_path)
                outputs[stat_tag] = out_path
                arcpy.AddMessage(f"    → {os.path.basename(out_path)}")
            except arcpy.ExecuteError as e:
                arcpy.AddWarning(f"    ✗ {stat_tag} failed: {e}")

        return outputs

    @staticmethod
    def _write_thermal_provenance_csv(out_workspace, prefix, mode_tag,
                                       stratification, region, groups,
                                       meta_by_path, cool_delta_k,
                                       warm_delta_k):
        """Provenance CSV for the thermal modes. Header is a leading
        ``# run config: k=v; ...`` comment so consumers with
        ``comment='#'`` skip it cleanly; per-scene rows carry scene_id,
        acquisition_date_iso, source_format, source_path.
        """
        anchor = _build_workspace_subfolder_path(
            out_workspace, prefix, "temporal",
        )
        csv_path = _sidecar_path_for_raster(anchor, "_temporal_provenance.csv")
        config = {
            "stat_source": mode_tag,
            "stratification": stratification,
            "region": region or "",
            "bt_scale_factor": _ASTER_TIR_SCALE,
            "valid_floor_K": _ASTER_TIR_VALID_K_FLOOR,
            "cool_delta_K": cool_delta_k,
            "warm_delta_K": warm_delta_k,
            "native_resolution_m": 90,
            "toolbox_version": TOOLBOX_VERSION,
            "generated": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(
                    "# run config: "
                    + "; ".join(f"{k}={v}" for k, v in config.items())
                    + "\n"
                )
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "group", "scene_id", "acquisition_date_iso",
                    "source_format", "source_path",
                ])
                for group_name, paths in groups.items():
                    for p in paths:
                        m = meta_by_path.get(p, {})
                        d = m.get("acquisition_date")
                        writer.writerow([
                            group_name,
                            m.get("scene_id", ""),
                            d.isoformat() if d else "",
                            m.get("format", ""),
                            p,
                        ])
            arcpy.AddMessage(f"  Provenance CSV: {csv_path}")
        except OSError as e:
            arcpy.AddWarning(f"  Could not write provenance CSV ({e})")

    def _execute_thermal_ast08(self, ast08_folder, out_workspace, out_prefix,
                                region, stratification, cool_delta_k,
                                warm_delta_k, save_stats):
        """LST (AST_08 thermal) mode entry point. Discovers AST_08
        files under ``ast08_folder``, groups by season (or "all"),
        materialises per-scene Kelvin rasters to a temp scratch,
        computes the seven stat layers per group, and writes provenance.
        AOI clipping flows from the active ``arcpy.env.mask``."""
        if not (ast08_folder and os.path.isdir(ast08_folder)):
            arcpy.AddError(
                f"AST_08 Folder {ast08_folder!r} does not exist or is "
                "not a directory."
            )
            return

        arcpy.AddMessage(f"  AST_08 folder:  {ast08_folder}")
        arcpy.AddMessage(f"  Cool delta:     {cool_delta_k} K")
        arcpy.AddMessage(f"  Warm delta:     {warm_delta_k} K")

        paths, metas = self._discover_ast08(ast08_folder)
        if not paths:
            arcpy.AddError(
                f"No AST_08 files matched under {ast08_folder} "
                "(expecting AST_08_*.tif or *.hdf)."
            )
            return
        arcpy.AddMessage(f"  AST_08 scenes:  {len(paths)}")

        # Per-season grouping reuses the helpers used by the NDVI path.
        seasonal_pattern = self._seasonal_pattern_for_region(region or "")
        dated = [(p, m["acquisition_date"]) for p, m in zip(paths, metas)]
        n_dated = sum(1 for _, d in dated if d is not None)
        if n_dated < len(dated):
            arcpy.AddWarning(
                f"  Dated:          {n_dated}/{len(dated)} parseable "
                "filenames; rest excluded from per-season grouping."
            )
        groups = self._group_scenes(dated, stratification, seasonal_pattern)
        for g, ps in groups.items():
            arcpy.AddMessage(f"  Group {g!r}: {len(ps)} scene(s)")
            if len(ps) < 2:
                arcpy.AddWarning(
                    f"  Group {g!r} has only {len(ps)} scene(s); stats "
                    "need >= 2 — group will be skipped."
                )

        meta_by_path = {p: m for p, m in zip(paths, metas)}

        scratch = tempfile.mkdtemp(prefix="genesis_lst_ast08_")
        arcpy.AddMessage(f"  Scratch:        {scratch}")
        try:
            arcpy.AddMessage("\n▶ Computing thermal statistics per group...")
            for group_name, group_paths in groups.items():
                if len(group_paths) < 2:
                    continue
                group_start = datetime.now()
                arcpy.SetProgressor(
                    "default",
                    f"Materialising Kelvin rasters [{group_name}]...",
                )
                kelvin_saved = self._materialise_kelvin_group(
                    group_paths, meta_by_path, scratch, group_name,
                )
                if len(kelvin_saved) < 2:
                    arcpy.AddWarning(
                        f"  Group {group_name!r}: only {len(kelvin_saved)} "
                        "valid Kelvin raster(s); group skipped."
                    )
                    arcpy.ResetProgressor()
                    continue
                arcpy.SetProgressorLabel(
                    f"Computing thermal statistics [{group_name}]..."
                )
                self._compute_thermal_stats(
                    group_name, kelvin_saved, out_workspace, out_prefix,
                    stratified=(stratification == "Per season"),
                    cool_delta_k=cool_delta_k, warm_delta_k=warm_delta_k,
                )
                arcpy.ResetProgressor()
                arcpy.AddMessage(
                    f"  ✓ Group {group_name!r}: stats written in "
                    f"{(datetime.now() - group_start).total_seconds():.1f}s"
                )

            if save_stats:
                self._write_thermal_provenance_csv(
                    out_workspace, out_prefix, "LST_AST08", stratification,
                    region, groups, meta_by_path, cool_delta_k, warm_delta_k,
                )
        finally:
            try:
                shutil.rmtree(scratch, ignore_errors=True)
            except Exception:
                pass


# ===========================================================================
# Subprocess worker entry point.
# ===========================================================================
#
# ArcGIS Pro imports this .pyt as a module to discover tool classes; the
# block below does not execute under Pro. It only fires when a mosaic
# tool's execute() spawns this same file as a standalone Python script
# via subprocess.Popen with `--worker <kind> <spec.json>` argv. See
# _run_scene_batches for the orchestration that drives this.

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _worker_kind = sys.argv[2]
        with open(sys.argv[3], encoding="utf-8") as _fh:
            _spec = json.load(_fh)
        if _worker_kind == "s2":
            _worker_s2_batch(_spec)
        elif _worker_kind == "aster_vnir_swir":
            _worker_aster_batch(_spec, mode="vnir_swir")
        elif _worker_kind == "aster_vnir":
            _worker_aster_batch(_spec, mode="vnir")
        else:
            raise SystemExit(f"Unknown worker kind: {_worker_kind!r}")
        sys.exit(0)
    else:
        raise SystemExit(
            "genesis_toolbox.pyt is an ArcGIS Pro toolbox; load it via "
            "Pro's Catalog (right-click Toolboxes > Add Toolbox). Direct "
            "invocation is only supported in --worker mode by the "
            "subprocess-batching orchestration; see _run_scene_batches."
        )
