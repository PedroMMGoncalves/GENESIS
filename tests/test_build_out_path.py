"""Tests for ``IndicesComposites._build_out_path``.

The helper decides where each Tool 04 output lands and whether to
append ``.tif``. Two responsibilities to keep stable:

  1. Plain folder workspaces split outputs into ``indices/`` and
     ``composites/`` subfolders with ``.tif`` appended (ESRI GRID
     would otherwise cap raster names at 13 chars).
  2. ``.gdb`` and ``.sde`` workspaces save flat into the workspace
     root with no extension (geodatabases have no name length limit
     and don't support nested folders).
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_toolbox


class BuildOutPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gt = load_toolbox()
        cls.tool = gt.IndicesComposites()

    def setUp(self):
        # Real temp dir so the os.makedirs side-effect can be checked.
        self.tmp = tempfile.mkdtemp(prefix="genesis_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_folder_index_appends_tif_and_creates_subfolder(self):
        out = self.tool._build_out_path(self.tmp, "LSNDVI", "index")
        self.assertTrue(out.endswith(".tif"))
        self.assertIn(os.path.join("indices", "LSNDVI.tif"), out)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "indices")))

    def test_folder_composite_appends_tif_and_creates_subfolder(self):
        out = self.tool._build_out_path(self.tmp, "NatColor", "composite")
        self.assertTrue(out.endswith(".tif"))
        self.assertIn(os.path.join("composites", "NatColor.tif"), out)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "composites")))

    def test_subfolders_are_independent(self):
        idx = self.tool._build_out_path(self.tmp, "X", "index")
        comp = self.tool._build_out_path(self.tmp, "X", "composite")
        self.assertNotEqual(idx, comp)

    def test_geodatabase_no_extension_no_subfolder(self):
        # No need to actually create a .gdb on disk — the helper is
        # pure string logic for the gdb branch.
        gdb = "D:/some/path/my.gdb"
        out = self.tool._build_out_path(gdb, "LSNDVI", "index")
        self.assertFalse(out.endswith(".tif"))
        self.assertNotIn("indices", out)
        self.assertNotIn("composites", out)
        self.assertTrue(out.endswith("LSNDVI"))

    def test_sde_workspace_treated_like_gdb(self):
        sde = "D:/connections/work.sde"
        out = self.tool._build_out_path(sde, "X", "index")
        self.assertFalse(out.endswith(".tif"))
        self.assertNotIn("indices", out)

    def test_gdb_case_insensitive(self):
        # Catalog paths sometimes come back with mixed case; the gdb
        # branch must trigger regardless.
        gdb = "D:/some/path/MY.GDB"
        out = self.tool._build_out_path(gdb, "X", "index")
        self.assertFalse(out.endswith(".tif"))

    def test_trailing_separator_in_gdb_path(self):
        # Defensive: a path that ends with a slash should still be
        # recognised as a gdb.
        gdb = "D:/some/path/my.gdb/"
        out = self.tool._build_out_path(gdb, "X", "index")
        self.assertFalse(out.endswith(".tif"))


if __name__ == "__main__":
    unittest.main()
