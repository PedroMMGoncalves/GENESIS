# Changelog

Notable changes to GENESIS. Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/) — `TOOLBOX_VERSION` in [`genesis_toolbox.pyt`](genesis_toolbox.pyt) tracks the same value as `version:` in [`CITATION.cff`](CITATION.cff).

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
