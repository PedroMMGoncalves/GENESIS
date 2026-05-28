"""Tests for ``AsterMosaic._has_swir_bands``.

The classifier is the gate that decides whether a scene contributes
to the 9-band VNIR+SWIR mosaic (pre-April-2008) or only to the
3-band VNIR-only mosaic (post-April-2008 SWIR failure).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_toolbox


class HasSwirBandsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gt = load_toolbox()
        cls.fn = staticmethod(cls.gt.AsterMosaic._has_swir_bands)

    def test_hdf_always_assumed_full(self):
        # HDF granules carry the full subdataset table; partial SWIR
        # is detected later on read, not at classification time.
        scene = {"format": "hdf", "files": {"hdf": "fake.hdf"}}
        self.assertTrue(self.fn(scene))

    def test_tiff_with_all_swir_bands_passes(self):
        files = {b: f"/tmp/{b}.tif" for b in (
            "B01", "B02", "B03N", "B04", "B05", "B06", "B07", "B08", "B09",
        )}
        scene = {"format": "tiff", "files": files}
        self.assertTrue(self.fn(scene))

    def test_tiff_with_vnir_only_fails(self):
        files = {b: f"/tmp/{b}.tif" for b in ("B01", "B02", "B03N")}
        scene = {"format": "tiff", "files": files}
        self.assertFalse(self.fn(scene))

    def test_tiff_missing_one_swir_band_fails(self):
        # Missing B07 alone is enough to disqualify the scene from
        # the 9-band mosaic.
        files = {b: f"/tmp/{b}.tif" for b in (
            "B01", "B02", "B03N", "B04", "B05", "B06", "B08", "B09",
        )}
        scene = {"format": "tiff", "files": files}
        self.assertFalse(self.fn(scene))

    def test_empty_files_dict_fails(self):
        scene = {"format": "tiff", "files": {}}
        self.assertFalse(self.fn(scene))

    def test_missing_files_key_fails(self):
        # Defensive: scene dict without a "files" entry should be
        # treated as VNIR-only, not crash.
        scene = {"format": "tiff"}
        self.assertFalse(self.fn(scene))


if __name__ == "__main__":
    unittest.main()
