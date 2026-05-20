# GENESIS

> Multisensor Satellite Analysis Toolbox for ArcGIS Pro

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ArcGIS Pro](https://img.shields.io/badge/ArcGIS_Pro-3.0%2B-green.svg)](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](https://zenodo.org/)
<!-- Once the Zenodo DOI is minted, replace the badge above with:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER) -->

GENESIS is a unified ArcGIS Pro Python Toolbox for remote-sensing analysis
of **Sentinel-2 L2A**, **Landsat 8/9 Collection 2 Level 2**, and **ASTER L2
AST_07XT** imagery. Six tools cover the full workflow — from raw scene
folders to cloud-removed multi-band mosaics, spectral indices, PCA/MNF/ICA
transforms, and Spectral Angle Mapper classification — across all three
sensors via a single sensor-aware design.

Developed at **LNEG** (Laboratório Nacional de Energia e Geologia) for
geological mapping, mineral exploration, and hydrothermal alteration
detection across Portuguese and African study areas (Iberian Peninsula,
Azores, Madeira, Cape Verde, Angola, Mozambique).

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

---

## Sensor-aware design

Tools 04, 05, and 06 take a `Sensor Type` parameter (Auto-detect / Landsat 8/9 / Sentinel-2 / ASTER). The auto-detection reads the input raster's filename pattern or band count and resolves to the right sensor. Indices and composites are filtered at runtime to only what the selected sensor can compute — NDRE only appears for Sentinel-2, Alunite / Kaolinite / Muscovite / Calcite only for ASTER, Iron Oxide via Red/Blue for L8/9 and S2 (ASTER gets the Red/Green variant since it lacks a blue band).

| Sensor | Canonical band stack | Specific indices |
| --- | --- | --- |
| **Landsat 8/9** | 7 bands: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2 | NDVI, NDWI, NDMI, NDBI, geological mineral ratios with Blue |
| **Sentinel-2** | 10 bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 | All of the above + NDRE, CIred-edge, IRECI |
| **ASTER** | 9 bands: B01, B02, B03N, B04, B05, B06, B07, B08, B09 | Geological staples + Alunite, Kaolinite, Muscovite, Calcite, Hydrothermal Alteration (Cudahy) |

---

## Requirements

- **ArcGIS Pro 3.0 or higher**
- **Spatial Analyst** extension (required for all six tools)
- **Image Analyst** extension (used by Mosaic and Transformations)
- Python 3.9+ with `numpy`, `scipy`, `scikit-learn` — all included with ArcGIS Pro's bundled Python
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
| Sentinel-2 L2A | `.SAFE` folders or `.zip` Copernicus archives | Zips auto-extracted on first run; idempotent |
| Landsat 8/9 | EarthExplorer `.tar` archives or extracted scene folders | Both L2SR and L2SP accepted; tars auto-extracted |
| ASTER | AST_07XT V004 per-band TIFFs or `.hdf` archives | TIFFs grouped by 17-char scene ID; HDF read via `osgeo.gdal` |

---

## Output convention

- Reflectance: **float 0–1** (analysis-friendly; Sentinel-2 ×0.0001, ASTER ×0.001 applied during ingestion)
- Mosaic CRS: inherited from input (UTM for Landsat / S2; resample-aligned for ASTER)
- Each mosaic: **multi-band GeoTIFF or geodatabase raster + `_provenance.csv`**
- Transformation statistics: **`.npz` (NumPy archive)** for re-applying to new AOIs

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

- Developed at **LNEG** (Laboratório Nacional de Energia e Geologia)
- Spectral index formulations from the published remote-sensing literature; mineral indices for ASTER follow Cudahy (alteration), Mars & Rowan (clays), and Sabins (iron oxides)
- PCA / MNF / ICA implementations follow Green et al. (1988) for MNF and use scikit-learn's FastICA
- Cloud-masking conventions: Landsat C2L2 QA_PIXEL (Vermote et al.), Sentinel-2 SCL (Sen2Cor), ASTER QA Data Plane (LP DAAC)
