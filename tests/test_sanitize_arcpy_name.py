"""Tests for the module-level ``_sanitize_arcpy_name`` helper.

Covers the cases that motivated its introduction (Windows duplicate
download suffixes, embedded spaces) plus the standard "leave the
input alone" path for clean scene IDs.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _loader import load_toolbox


class SanitizeArcpyNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gt = load_toolbox()
        # staticmethod wrapper prevents Python from auto-binding the
        # module-level function as a method when accessed via ``self``.
        cls.fn = staticmethod(cls.gt._sanitize_arcpy_name)

    def test_empty_inputs_return_empty_string(self):
        self.assertEqual(self.fn(""), "")
        self.assertEqual(self.fn(None), "")

    def test_clean_id_passes_through(self):
        clean = "S2C_MSIL2A_20260122T125031_N0511_R095_T26SLH_20260122T135913"
        self.assertEqual(self.fn(clean), clean)

    def test_windows_duplicate_suffix_collapses_to_underscores(self):
        raw = "S2C_MSIL2A_20260122T125031_N0511_R095_T26SLH_20260122T135913 (1)"
        result = self.fn(raw)
        # The duplicate suffix becomes "__1_": one underscore for the space,
        # one for "(", one for ")".
        self.assertTrue(result.endswith("__1_"))
        # No arcpy-hostile characters survive.
        for ch in " ()":
            self.assertNotIn(ch, result)

    def test_distinct_inputs_never_collapse(self):
        # Replacement (not removal) preserves uniqueness across scenes.
        a = self.fn("scene a")
        b = self.fn("scene-b")  # hyphen is not in the hostile set
        self.assertNotEqual(a, b)

    def test_all_hostile_chars_get_replaced(self):
        # Every character in the translation table maps to "_".
        hostile = " ()[]{}'\";,$@#!%^&*+=<>?|"
        result = self.fn(hostile)
        self.assertEqual(result, "_" * len(hostile))

    def test_non_string_input_coerced_to_string(self):
        # Defensive: the helper documents handling None; integers etc.
        # should also coerce cleanly.
        self.assertEqual(self.fn(42), "42")


if __name__ == "__main__":
    unittest.main()
