"""Tests for ``AsterMosaic._parse_aster_filename``.

The regex underpins scene discovery in ``_find_aster_scenes`` — it
extracts the acquisition date (MMDDYYYY embedded in the sceneID),
pass number, band group, and band name from filenames following the
LP DAAC ``AST_07XT`` V004 convention.

Catches a class of subtle regressions: if the regex starts rejecting
clean filenames or accepting malformed ones, the whole pipeline
falls back to "no scenes found" without a useful error.
"""

import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_toolbox


class ParseAsterFilenameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gt = load_toolbox()
        cls.fn = staticmethod(cls.gt.AsterMosaic._parse_aster_filename)

    def test_valid_vnir_b01_tiff(self):
        name = "AST_07XT_00302012003124657_20250305062641_SRF_VNIR_B01.tif"
        parsed = self.fn(name)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["band"], "B01")
        self.assertEqual(parsed["group"], "VNIR")
        # sceneID decode: pass=003, MM=02, DD=01, YYYY=2003, HMS=124657
        # i.e. 1 Feb 2003, 12:46:57 (US-style MMDDYYYY in the spec)
        self.assertEqual(parsed["acquisition_date"], date(2003, 2, 1))

    def test_valid_swir_b09_tiff(self):
        name = "AST_07XT_00302012003124657_20250305062642_SRF_SWIR_B09.tif"
        parsed = self.fn(name)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["band"], "B09")
        self.assertEqual(parsed["group"], "SWIR")

    def test_post_failure_scene_still_parses(self):
        # Feb 29, 2012 — a SWIR-failed scene must still parse so the
        # tool can decide what to do; rejection happens later.
        name = "AST_07XT_00402292012125215_20250715032348_SRF_VNIR_B01.tif"
        parsed = self.fn(name)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["acquisition_date"], date(2012, 2, 29))

    def test_vnir_b03n_special_band_name(self):
        # B03N (nadir-looking) carries the trailing "N" — easy to
        # miss in the regex if it's tightened to ``B0[1-9]``.
        name = "AST_07XT_00302012003124657_20250305062641_SRF_VNIR_B03N.tif"
        parsed = self.fn(name)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["band"], "B03N")

    def test_qa_dataplane_recognised(self):
        name = "AST_07XT_00302012003124657_20250305062641_SRF_VNIR_QA_DataPlane.tif"
        parsed = self.fn(name)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["band"], "QA_DataPlane")

    def test_qa_dataplane2_recognised(self):
        name = "AST_07XT_00302012003124657_20250305062642_SRF_SWIR_QA_DataPlane2.tif"
        parsed = self.fn(name)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["band"], "QA_DataPlane2")

    def test_non_matching_name_returns_none(self):
        self.assertIsNone(self.fn("totally_not_aster.tif"))
        self.assertIsNone(self.fn(""))

    def test_truncated_scene_id_rejected(self):
        # Missing digits in the sceneID block — must fail rather than
        # silently misinterpret the date.
        bad = "AST_07XT_004029220_20250715032348_SRF_VNIR_B01.tif"
        self.assertIsNone(self.fn(bad))

    def test_case_insensitive_extension(self):
        # The pattern is anchored at lowercase ``.tif`` via re.IGNORECASE.
        upper = "AST_07XT_00302012003124657_20250305062641_SRF_VNIR_B01.TIF"
        self.assertIsNotNone(self.fn(upper))


if __name__ == "__main__":
    unittest.main()
