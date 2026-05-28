# tests/ — public regression suite

These tests protect the pure-Python helpers in `genesis_toolbox.pyt`
from regression. They run without ArcGIS Pro: a small `arcpy` stub
in [`conftest.py`](conftest.py) covers the surface the `.pyt` touches
at import time, and the tests target functions that don't need a
real raster engine (parsers, filename regexes, sensor / band-role
lookups, archive metadata readers, the pure-numpy compositor and
ICA helpers, the OmniCloudMask observed-pixel reducer, etc.).

## What's in scope

Anything that can be exercised through stock Python:

- Sensor / band-role detection and mapping
- ASTER / Landsat / Sentinel-2 filename parsers
- Scene discovery against in-place `.tar` / `.zip` archives via GDAL
  VSI paths (`/vsitar/...`, `/vsizip/...`) — no extraction
- Provenance CSV writers
- Pure-numpy ICA (`_fast_ica_numpy`) plumbing and statistics
- Per-band percentile compositor wiring
- OmniCloudMask cloud-fraction reduction over observed pixels
- Temporal outlier cleaner (MAD-based robust z-score) wiring
- Sidecar-path resolution (`.gdb` workspace vs folder workspace)

Anything that genuinely needs arcpy raster ops (CellStatistics,
GeometricMedian, SetRasterProperties, etc.) is verified through the
end-to-end mosaic runs in real workflows, not here.

## How to run

The tests load `genesis_toolbox.pyt` via `importlib`. The bundled
`conftest.py` stubs `arcpy` and its submodules so a CI runner without
ArcGIS Pro can still exercise the pure-Python surface. Run from any
environment with `pytest`, `numpy`, and `scipy`:

```bash
python -m pytest tests -q
```

If you have ArcGIS Pro locally you can also run from its bundled
Python — the real `arcpy` shadows the stub and the suite still
passes:

```powershell
& "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" -m pytest tests -q
```

## Author-side scratch (`tests/_dev/`)

The `tests/_dev/` subfolder is gitignored. It holds one-off probes,
A/B drivers, and audit scripts (e.g., AROSICS reliability sweeps,
ASTER cloud-mask audits, V12/V13/V14 run drivers) that depend on
local scene data and external hardware (RTX 4090 / Faial scene
archive). Those are not portable and not regression-style.

## Adding tests

Drop a `test_*.py` file under `tests/`. Use the `genesis` pytest
fixture (defined in `conftest.py`) to get a handle on the toolbox
module:

```python
def test_thing(genesis):
    assert genesis._sanitize_arcpy_name("a b") == "a_b"
```

If your test needs a numpy array back from `arcpy.RasterToNumPyArray`,
patch it with `monkeypatch` — see
[`test_genesis_v1_helpers.py`](test_genesis_v1_helpers.py) for the
`_ocm_mask_class_fractions` tests as a pattern.
