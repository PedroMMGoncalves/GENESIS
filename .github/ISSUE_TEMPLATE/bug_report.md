---
name: Bug report
about: Report a problem with a GENESIS tool
title: "[BUG] "
labels: bug
assignees: ''
---

## Which tool?

<!-- Pick one -->
- [ ] 01 — Sentinel-2 L2A Mosaic
- [ ] 02 — Landsat 8/9 C2L2 Mosaic
- [ ] 03 — ASTER L2 Mosaic
- [ ] 04 — Spectral Indices & Composites
- [ ] 05 — Statistical Transformations
- [ ] 06 — Spectral Angle Mapper
- [ ] 07 — Temporal Composites & Statistics

## Environment

- **GENESIS version** (from `TOOLBOX_VERSION` in `genesis_toolbox.pyt`, or commit hash if running on `main`):
- **ArcGIS Pro version**:
- **OS** (Windows 10 / 11; build):
- **Active Python env** (Pro's default `arcgispro-py3` or a clone? path?):
- **GPU** (model + driver version, if Tool 03 / OmniCloudMask is involved):

## Tool 03 (ASTER) only

If your bug is on Tool 03, please confirm:

- [ ] `omnicloudmask` import succeeds (`python -c "import omnicloudmask"`).
- [ ] `arosics` import succeeds in the env the toolbox uses for AROSICS (`python -c "from arosics import COREG_LOCAL"`).
- [ ] `rasterio` is installed.
- [ ] Esri Deep Learning Frameworks MSI matching your Pro version is installed.

## Input data scale

- **Sensor + product**: (e.g. AST_07XT V004, S2 L2A, Landsat 9 C2L2)
- **Scene count**:
- **AOI extent** (km², or rough bbox):
- **Total input size on disk**:

## What did you do?

Step-by-step. Include the parameter values you set in the tool dialog (or the `_run_*.py` runner you used). Paste them as a list so we can reproduce.

## What did you expect?

## What actually happened?

Paste the full error message + the **Phase** the failure occurred in (visible in the geoprocessing pane log).

## Relevant log lines

<!-- Paste the geoprocessing log around the failure. Redact any
     paths / scene IDs you don't want public. -->

```
(log lines here)
```

## Have you checked?

- [ ] The relevant section of [`README.md`](../../README.md).
- [ ] The [`CHANGELOG.md`](../../CHANGELOG.md) — your version may pre-date a fix.
- [ ] Existing issues for similar reports.
