# GENESIS

> Multisensor Satellite Analysis Toolbox for ArcGIS Pro

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ArcGIS Pro](https://img.shields.io/badge/ArcGIS_Pro-3.0%2B-green.svg)](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](https://zenodo.org/)
<!-- Once the Zenodo DOI is minted, replace the badge above with:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER) -->

GENESIS is an ArcGIS Pro toolbox for analysing **Sentinel-2 L2A**,
**Landsat 8/9 Collection 2 Level 2**, and **ASTER L2 (AST_07XT)** imagery.
It has six tools: a cloud-removed mosaicker for each sensor, plus three
sensor-agnostic tools for spectral indices and composites, PCA / MNF / ICA
statistical transformations, and Spectral Angle Mapper classification.
A `Sensor Type` parameter on the analysis tools selects the correct band
roles, indices, and reflectance scaling for the input data.

The toolbox was developed during the author's work on the **GENESIS**
Horizon Europe project (grant 101157447,
[genesisnbs.eu](https://genesisnbs.eu/faial/)) to produce cloud-removed
multisensor satellite mosaics for the **Faial Island** demonstrator site
in the Azores.

---

## Tools

The toolbox exposes six tools in workflow order:

| # | Tool | Purpose |
| --- | --- | --- |
| **01** | Sentinel-2 L2A Mosaic | Cloud-masked mosaic from S2 `.SAFE` archives or `.zip` downloads (SCL classes 3/8/9/10 masked; 20m bands resampled to 10m) |
| **02** | Landsat 8/9 C2L2 Mosaic | Cloud-masked mosaic from EarthExplorer `.tar` archives (QA_PIXEL bits 0-4 masked; multi-UTM-zone merging) |
| **03** | ASTER L2 Mosaic | Mineral-mapping mosaic from AST_07XT V004 (TIFF + HDF input; SWIR 30m → 15m; QA Data Plane + multitemporal cloud refinement) |
| **04** | Spectral Indices & Composites | ~25 indices + ~8 RGB composites, sensor-filtered to what each sensor can compute (red-edge for S2; per-wavelength SWIR minerals for ASTER) |
| **05** | Statistical Transformations | PCA, MNF, and ICA — sensor-agnostic algorithm with persisted `.npz` statistics for cross-AOI re-application |
| **06** | Spectral Angle Mapper | Reference-spectra classification (table / training samples / endmember pixels) |

All three mosaic tools write a `_provenance.csv` next to the output documenting every scene that contributed (scene ID, acquisition date, cloud cover, input path, processing baseline, toolbox version).

### Resume on re-run

All three mosaic tools expose a **Preserve Scratch & Resume on Re-run** option under *Advanced Options*. When the option is on:

- Per-scene intermediate stacks are kept in a deterministically-named scratch folder beside the output geodatabase (e.g. `_genesis_s2_scratch_{mosaic_name}`) instead of being cleaned up at the end of the run.
- Each successfully written stack also writes a one-line `{scene_id}_stack.tif.complete` sentinel.
- Re-running with the same output name causes Phase 3 to skip any scene whose stack file *and* sentinel are both present — the per-scene processing is reused as-is, the run jumps straight to the geometric-median stage.

A late-phase failure (e.g. a transient GP error during the median or AOI-mask step) therefore costs only the time of the failed phase, not the hours of per-scene QA / mask / resample / composite work that preceded it.

---

## Sensor-aware design

Tools 04, 05, and 06 take a `Sensor Type` parameter (Auto-detect / Landsat 8/9 / Sentinel-2 / ASTER). The auto-detection reads the input raster's filename pattern or band count and resolves to the right sensor. Indices and composites are filtered at runtime to only what the selected sensor can compute — NDRE only appears for Sentinel-2, Alunite / Kaolinite / Muscovite / Calcite only for ASTER, Iron Oxide via Red/Blue for L8/9 and S2 (ASTER gets the Red/Green variant since it lacks a blue band).

| Sensor | Canonical band stack | Specific indices |
| --- | --- | --- |
| **Landsat 8/9** | 7 bands: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2 | NDVI, NDWI, NDMI, NDBI, geological mineral ratios with Blue |
| **Sentinel-2** | 10 bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | All of the above + NDRE, CIred-edge, IRECI |
| **ASTER** | 9 bands: B01, B02, B03N, B04, B05, B06, B07, B08, B09 ¹ | Geological staples + Alunite, Kaolinite, Muscovite, Calcite, Hydrothermal Alteration (Cudahy) |

¹ Tool 03 emits **two outputs**: `{name}_VnirSwir` is the canonical 9-band mosaic built from pre-April-2008 scenes only (full VNIR + crosstalk-corrected SWIR), and `{name}_Vnir` is a 3-band (B01, B02, B03N) mosaic built from the full archive — the ASTER SWIR detector failed in April 2008 and any post-failure scene contributes only to the VNIR-only product. Tool 04 detects the band count of its input automatically and skips indices that need SWIR roles when handed the 3-band variant.

---

## Requirements

- **ArcGIS Pro 3.0 or higher**
- **Spatial Analyst** extension (required for all six tools)
- **Image Analyst** extension (used by Mosaic and Transformations)
- Python 3.9+ with `numpy` and `scipy` — both included with ArcGIS Pro's bundled Python (no other ML or scientific-computing dependencies; the FastICA used in Tool 05 is a pure-numpy in-house implementation)
- `osgeo.gdal` (also bundled with ArcGIS Pro) — used by Tool 03 for HDF input

---

## Installation

1. Clone or download this repository.
2. Open **ArcGIS Pro**.
3. In the **Catalog** pane, right-click **Toolboxes** → **Add Toolbox**.
4. Navigate to `genesis_toolbox.pyt` and select it.
5. The six tools appear under "GENESIS — Satellite Analysis Toolbox" in workflow order (01–06).

---

## Input formats

| Sensor | Accepted | Notes |
| --- | --- | --- |
| Sentinel-2 L2A | `.SAFE` folders or `.zip` Copernicus archives | Archives read on the fly via GDAL VSI (no extraction needed); browser-style duplicate-download suffixes (`(1).zip`) are deduped by canonical `PRODUCT_URI` from the archive's MTD XML |
| Landsat 8/9 | EarthExplorer `.tar` or `.zip` archives, or extracted scene folders | Both L2SR and L2SP accepted; archives read on the fly via GDAL VSI (no extraction needed) |
| ASTER | AST_07XT V004 per-band TIFFs or `.hdf` archives | TIFFs grouped by 17-char scene ID; HDF read via `osgeo.gdal` |

---

## Output convention

- Reflectance: **float 0–1** (analysis-friendly; Sentinel-2 ×0.0001, ASTER ×0.001 applied during ingestion)
- Mosaic CRS: inherited from input (UTM for Landsat / S2; resample-aligned for ASTER)
- Each mosaic: **multi-band GeoTIFF or geodatabase raster + `_provenance.csv`**
- Transformation statistics: **`.npz` (NumPy archive)** for re-applying to new AOIs
- Tool 04 outputs: when the workspace is a **folder**, indices and composites are written to sibling `indices/` and `composites/` subfolders (forced GeoTIFF, no 13-character ESRI-GRID name limit). When the workspace is a **`.gdb`**, both products are saved flat at the workspace root with no extension (geodatabases have no name-length limit and don't support nested folders).
- Each mosaic's median output is sanity-checked after save: per-band MIN / MEAN / MAX are logged, and a warning is emitted if any band's mean is suspiciously close to zero for that sensor's expected reflectance range — catches the class of NoData-handling regression that would otherwise pass type-checking and structural validation while producing visually-broken outputs.

---

## Citation

If you use GENESIS in your research, please cite it via the metadata in
[`CITATION.cff`](CITATION.cff) (GitHub renders this in the sidebar as
"Cite this repository"). A versioned DOI is minted via Zenodo for each
release.

---

## License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE)
for the full text. In short: you can use, modify, and redistribute this
work commercially or otherwise, provided attribution is preserved; explicit
patent grant and patent-retaliation clauses apply per the Apache 2.0 terms.

---

## Acknowledgements

### Funding context

This toolbox was developed to support the author's analytical work on the **GENESIS** project ([genesisnbs.eu](https://genesisnbs.eu)), funded by the European Union's Horizon Europe research and innovation programme under grant agreement Nº **101157447**.

The author's contribution within GENESIS is the production of cloud-removed multisensor satellite mosaics and the organisation of derived information (spectral indices and statistical transformations were added as analytically useful extensions) for the [Faial demonstrator site](https://genesisnbs.eu/faial/). The Faial demonstrator is designing and building an aquifer storage and recovery system, using excess potable water accumulated in existing earth dams/watering ponds, to re-establish the freshwater supply capacity of *Furo das Cancelas* — the sole water source for ~3,000 inhabitants — and to counter saltwater intrusion in the coastal aquifer.

> Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or CINEA. Neither the European Union nor the granting authority can be held responsible for them.

### References

#### Cloud masking

- *Landsat C2L2 QA_PIXEL.* USGS, 2020. *Landsat Collection 2 Level-2 Science Product Guide* (LSDS-1619). U.S. Geological Survey. Bit definitions for cloud, cloud-shadow, cirrus, dilated-cloud and fill flags.
- *Sentinel-2 SCL (Sen2Cor).* Main-Knorn, M., Pflug, B., Louis, J., Debaecker, V., Müller-Wilm, U., & Gascon, F. (2017). Sen2Cor for Sentinel-2. In *Image and Signal Processing for Remote Sensing XXIII* (Vol. 10427, 1042704). SPIE. <https://doi.org/10.1117/12.2278218>
- *ASTER QA Data Plane.* NASA LP DAAC. *AST_07XT: ASTER L2 Surface Reflectance VNIR and Crosstalk Corrected SWIR V004* product specification. <https://lpdaac.usgs.gov/products/ast_07xtv004/>

#### Statistical transformations

- *PCA.* Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559–572. <https://doi.org/10.1080/14786440109462720>
- *MNF.* Green, A. A., Berman, M., Switzer, P., & Craig, M. D. (1988). A transformation for ordering multispectral data in terms of image quality with implications for noise removal. *IEEE Transactions on Geoscience and Remote Sensing*, 26(1), 65–74. <https://doi.org/10.1109/36.3001>
- *ICA / FastICA.* Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4–5), 411–430. <https://doi.org/10.1016/S0893-6080(00)00026-5> — the Hyvärinen parallel algorithm with symmetric decorrelation and the *logcosh* non-linearity is implemented in pure NumPy inside the toolbox (see `_fast_ica_numpy` in `genesis_toolbox.pyt`); no `scikit-learn` dependency.

#### Spectral indices — vegetation, water, built-up

- *NDVI.* Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1973). Monitoring vegetation systems in the Great Plains with ERTS. In *Third ERTS Symposium*, NASA SP-351 I, 309–317.
- *NDWI (water).* McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing*, 17(7), 1425–1432. <https://doi.org/10.1080/01431169608948714>
- *NDMI / NDWI (vegetation moisture).* Gao, B.-C. (1996). NDWI — A normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment*, 58(3), 257–266. <https://doi.org/10.1016/S0034-4257(96)00067-3>
- *NDBI.* Zha, Y., Gao, J., & Ni, S. (2003). Use of normalized difference built-up index in automatically mapping urban areas from TM imagery. *International Journal of Remote Sensing*, 24(3), 583–594. <https://doi.org/10.1080/01431160304987>
- *Red-edge indices (NDRE, CIred-edge, IRECI — Sentinel-2 only).* Gitelson, A. A., Viña, A., Ciganda, V., Rundquist, D. C., & Arkebauer, T. J. (2005). Remote estimation of canopy chlorophyll content in crops. *Geophysical Research Letters*, 32, L08403. <https://doi.org/10.1029/2005GL022688>; Frampton, W. J., Dash, J., Watmough, G., & Milton, E. J. (2013). Evaluating the capabilities of Sentinel-2 for quantitative estimation of biophysical variables in vegetation. *ISPRS Journal of Photogrammetry and Remote Sensing*, 82, 83–92. <https://doi.org/10.1016/j.isprsjprs.2013.04.007>

#### Spectral indices — geological / mineralogical

- *Iron oxide (Red/Blue and Red/Green ratios).* Sabins, F. F. (1999). Remote sensing for mineral exploration. *Ore Geology Reviews*, 14(3–4), 157–183. <https://doi.org/10.1016/S0169-1368(99)00007-4>
- *ASTER hydrothermal alteration indices (Cudahy method).* Cudahy, T. (2012). *Satellite ASTER Geoscience Product Notes for Australia* (version 1, CSIRO Report EP-30-07-12-44). CSIRO. Defines per-band ratios for alteration, ferric iron, ferrous iron, AlOH, MgOH, FeOH and quartz indices.
- *Clay / Alunite / Kaolinite / Muscovite (ASTER SWIR ratios).* Mars, J. C., & Rowan, L. C. (2006). Regional mapping of phyllic- and argillic-altered rocks in the Zagros magmatic arc, Iran, using Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) data and logical operator algorithms. *Geosphere*, 2(3), 161–186. <https://doi.org/10.1130/GES00044.1>
- *Carbonate / Calcite (ASTER B7/B8 ratio).* Rowan, L. C., & Mars, J. C. (2003). Lithologic mapping in the Mountain Pass, California area using Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) data. *Remote Sensing of Environment*, 84(3), 350–366. <https://doi.org/10.1016/S0034-4257(02)00127-X>

#### Compositing

- *Geometric median for temporal pixel reduction.* Roberts, D., Mueller, N., & McIntyre, A. (2017). High-dimensional pixel composites from earth observation time series. *IEEE Transactions on Geoscience and Remote Sensing*, 55(11), 6254–6264. <https://doi.org/10.1109/TGRS.2017.2723896> (the algorithm Esri's `arcpy.ia.GeometricMedian` implements, called by all three mosaic tools)

#### Classification

- *Spectral Angle Mapper.* Kruse, F. A., Lefkoff, A. B., Boardman, J. W., Heidebrecht, K. B., Shapiro, A. T., Barloon, P. J., & Goetz, A. F. H. (1993). The Spectral Image Processing System (SIPS) — interactive visualization and analysis of imaging spectrometer data. *Remote Sensing of Environment*, 44(2–3), 145–163. <https://doi.org/10.1016/0034-4257(93)90013-N>

---

> **Provenance note.** This toolbox is independently authored work used in support of GENESIS analytical tasks at the Faial demonstrator site. It is **not an official deliverable** of the GENESIS Horizon Europe consortium. For formal project deliverables, see [genesisnbs.eu](https://genesisnbs.eu) and the [Faial demonstrator page](https://genesisnbs.eu/faial/).
