# Changelog

Notable changes to GENESIS. Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/) — `TOOLBOX_VERSION` in [`genesis_toolbox.pyt`](genesis_toolbox.pyt) tracks the same value as `version:` in [`CITATION.cff`](CITATION.cff).

## [Unreleased] — UX-consistency sweep

These changes will land in the next tagged release. They tighten consistency across the three mosaic tools and Tool 07 without changing any output-file format or sensor-specific algorithm.

### Changed

- **`Time Filter Type` dropdown is now identical across Landsat / Sentinel-2 / ASTER.** Single TitleCase label set: `All Images`, `Specific Year`, `Month in Year`, `Month All Years`, `Season in Year`, `Season All Years`. Pre-v1.0 Sentinel-2 and ASTER used a snake_case dropdown (`all_images`, `year_month`, …) and lacked the `Specific Year` option entirely. **Breaking for saved Pro workflows holding the lowercase values** — but a coercion layer in each tool's `updateParameters` rewrites pre-v1.0 values to their new canonical labels on load, so workflows round-trip. Scripted callers and arcpy.toolbox calls passing the old snake_case key still work too: `_time_filter_key` remaps `year_month` → `month_in_year` so `_scene_passes_filter` keeps matching.
- **Landsat per-zone phase markers** carry a clearer per-zone banner (`─── PROCESSING UTM 26N ───`) and dropped the redundant `UTM zone X:` prefix from the Phase 1 label. The cycling Phase 1/2/3 numbers no longer read as the tool restarting.
- **Tool 07 LST mode** now emits Phase 1 (AST_08 scene discovery) and Phase 2 (Scene grouping) markers, matching the NDVI path's full 5-phase structure.
- **Tool 07 provenance sidecar** renamed from `_temporal_provenance.csv` to `_provenance.csv` — the `_temporal_` infix duplicated context already encoded in the parent `temporal/` subfolder, and the new name matches the mosaic tools' `{output}_provenance.csv` convention. **Compat shim** writes both filenames this release so scripts reading the old name keep working; the legacy filename will be dropped in the release after next.

### Added

- **`Specific Year` time-filter mode is now available on Sentinel-2 and ASTER mosaics**, matching Landsat. Selects every scene from a chosen year regardless of month — useful for annual mosaics on archives that span many years.
- **Required-companion validation** for every time-filter mode. `Specific Year` with no year value (and analogous gaps for `Month in Year`, `Season in Year`, etc.) now surface an inline error in the GP dialog before Run, instead of silently dropping every scene at the filter step.
- **Per-biome NDVI persistence threshold** (Tool 07). Howard & Merrifield (2010) per-biome value: `temperate = 0.5`, `mozambique / angola / cape_verde = 0.3`. Pre-v1.0 the threshold was hardcoded to 0.5 across all biomes, under-reporting persistent vegetation in arid regions.
- **Per-biome GDV thresholds** (Tool 07). Eamus / Naumburg dry-season floor and Lv 2013 ratio denominator guard both shift downward for arid patterns.
- **AROSICS detection retries 3× with 2 s backoff** so a transient concurrent-Pro-tool flake (PROJ database lock, GDAL driver init race) no longer aborts the run.

### Fixed

- **`_sanity_check_output` deadlock** that hung Phase 9 indefinitely on certain `CalculateStatistics` calls. Sanity check now runs after both ASTER mosaics are committed to the GDB, never during the per-mosaic pipeline. The earlier daemon-thread fix corrupted Aster_V16_Vnir as a stub table when the daemon held an arcpy handle into the next mosaic's Phase 8 write; this has been superseded by moving the call to a quiescent workspace.
- **Dead `LST (Landsat ST_B10)` dropdown option** (Tool 07) removed. Validation already rejected it as "not yet implemented"; the dropdown no longer offers it.

## [1.0] — 2026-05-28

First public release of the unified `genesis_toolbox.pyt`. Seven tools in workflow order: Sentinel-2, Landsat 8/9, and ASTER cloud-masked mosaics; sensor-aware spectral indices and composites; PCA/MNF/ICA statistical transformations; Spectral Angle Mapper classification; per-pixel temporal composites and statistics. See [`README.md`](README.md) for the full feature list and dependency matrix.

### ASTER pipeline highlights

Tool 03 went through a substantial redesign in the final week before this release to handle the AST_07XT's specific challenges (no native cloud mask, L1B-grade geolocation):

- **Phase 4 AROSICS sub-pixel co-registration** against a Sentinel-2 NIR reference corrects AST_07XT's ~20-50 m per-scene geolocation drift that would otherwise blur a multi-scene median to ~30-50 m effective resolution. Applied as a global geotransform translation (mean shift in metres) so the deshift is robust at any per-band resolution. New mandatory dependency: [`arosics`](https://github.com/GFZ/arosics) (Scheffler et al. 2017).
- **Phase 5 OmniCloudMask DL cloud + shadow segmentation** is the mandatory primary cloud detection layer for ASTER (AST_07XT ships without a native mask, unlike S2 SCL / Landsat QA_PIXEL). U-Net ensemble from Wright et al. 2025 ([`omnicloudmask`](https://github.com/DPIRD-DMA/OmniCloudMask)). A hardened VNIR spectral cloud test runs as a second-line union with the DL mask.
- **Dual output** (`{name}_VnirSwir` 9-band pre-Apr-2008 + `{name}_Vnir` 3-band full archive). The ASTER SWIR detector failed in April 2008; the dual output keeps the longer temporal baseline available where appropriate.
- **Evidence-quality sidecars** (`obs_count.tif`, `cloud_freq.tif`) for downstream uncertainty propagation, plus per-scene `cloud_pct` in the provenance CSV.

### Compositor menu (all three mosaic tools)

- **GeometricMedian (default)** — L1 multivariate median, preserves same-scene consistency across bands.
- **Per-band median** — band-by-band reduction via `arcpy.sa.CellStatistics(MEDIAN, ignore_nodata="DATA")`; explicit NoData semantics.
- **Per-band percentile** *(new)* — band-by-band reduction via `arcpy.sa.CellStatistics(PERCENTILE, ...)` with a `percentile_value` parameter (default 25, range 5-50). p25 biases toward the darker quartile to discard cloud-bright residuals that survived the cloud mask — useful for persistently cloudy AOIs like Faial's caldera.

Numerically verified against `numpy.nanpercentile(method='linear')` to floating-point precision including NoData handling.

### Defaults that may differ from the dialog you remember

- **ASTER temporal outlier cleaner: ON.** The 2026-05-26 redesign briefly flipped it OFF on the assumption that DL cloud masking would replace it; A/B evidence (cleaner ON vs OFF, everything else identical) disproved that and the default is now ON. DL and the cleaner are complementary: DL catches whole-region cloud, the cleaner catches per-pixel time-stack outliers DL doesn't see (cloud-shadow values that survived the mask, sensor glitches, residual haze).
- **Sentinel-2 cloud mask buffer: 3 px** (was 2). Three-pixel dilation catches the cloud-edge halo Sen2Cor classifies as Vegetation/Bare-soil but is in fact thin cloud / partial cover.

### Dependencies

Tools 01, 02, 04, 05, 06, 07 have no dependencies beyond the ArcGIS Pro base (`numpy`, `scipy`, `matplotlib` are bundled). Tool 03 adds three Python packages (`omnicloudmask`, `arosics`, `rasterio`) and PyTorch via the [Esri Deep Learning Frameworks](https://github.com/Esri/deep-learning-frameworks) MSI. All install into a single Pro-cloned `arcgispro-py3` env. See `README.md` § "Requirements" for the full install sequence and a two-env fallback for Pro versions whose pinned `gdal`/`pyproj` reject the `arosics` install in the clone.
