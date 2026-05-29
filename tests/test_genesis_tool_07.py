"""Regression tests for Tool 07 (TemporalStatistics).

The tool is registered + structurally tested in test_genesis_tools_05_06.py;
this file covers the pure-Python helpers that drive the per-biome
threshold dispatch and the season grouping. Raster math
(CellStatistics, Con, SetNull) runs only under ArcGIS Pro and is
not exercised here.
"""

from __future__ import annotations

import datetime as _dt


# ---------------------------------------------------------------------------
# Per-biome NDVI persistence threshold (Defect A)
# ---------------------------------------------------------------------------

def test_persistence_threshold_temperate_preserves_historical_value(genesis):
    """The temperate value (0.5) must not change — Faial / Azores
    runs would otherwise see a behaviour regression."""
    assert genesis._resolve_persistence_threshold("temperate") == 0.5


def test_persistence_threshold_arid_biomes_drop_to_0_3(genesis):
    """Howard & Merrifield (2010): arid / semi-arid biomes use 0.3,
    not the temperate 0.5. Pre-fix the tool silently applied 0.5 to
    Mozambique / Angola / Cape Verde, under-reporting persistent
    vegetation."""
    for pattern in ("mozambique", "angola", "cape_verde"):
        assert genesis._resolve_persistence_threshold(pattern) == 0.3, (
            f"{pattern!r} should resolve to 0.3 per the literature"
        )


def test_persistence_threshold_unknown_pattern_falls_back_to_temperate(genesis):
    """Future-proofing: an unrecognised pattern key must not raise
    KeyError — it falls back to the temperate default."""
    assert genesis._resolve_persistence_threshold("future_biome_xyz") == 0.5
    assert genesis._resolve_persistence_threshold(None) == 0.5
    assert genesis._resolve_persistence_threshold("") == 0.5


# ---------------------------------------------------------------------------
# Per-biome GDV thresholds (Defect B)
# ---------------------------------------------------------------------------

def test_gdv_thresholds_temperate_preserves_historical_values(genesis):
    """Temperate keeps dry_floor=0.3 (Eamus / Naumburg) and
    wet_min=0.1 (Lv 2013 denominator guard) — pre-fix values."""
    dry_floor, wet_min = genesis._resolve_gdv_thresholds("temperate")
    assert dry_floor == 0.3
    assert wet_min == 0.1


def test_gdv_thresholds_arid_biomes_shift_downward(genesis):
    """Arid patterns drop both thresholds to match the lower
    vegetation baseline. dry_floor 0.3 -> 0.2, wet_min 0.1 -> 0.05."""
    for pattern in ("mozambique", "angola", "cape_verde"):
        dry_floor, wet_min = genesis._resolve_gdv_thresholds(pattern)
        assert dry_floor == 0.2, f"{pattern!r} dry_floor should be 0.2"
        assert wet_min == 0.05, f"{pattern!r} wet_min should be 0.05"


def test_gdv_thresholds_unknown_pattern_falls_back_to_temperate(genesis):
    """Same fallback semantics as the persistence threshold resolver."""
    assert genesis._resolve_gdv_thresholds("future_biome_xyz") == (0.3, 0.1)
    assert genesis._resolve_gdv_thresholds(None) == (0.3, 0.1)


def test_threshold_dicts_share_key_set_with_dry_months(genesis):
    """The per-biome threshold dicts must mirror the seasonal-pattern
    keys exactly — otherwise a region resolves to a pattern that has
    seasonal months but no threshold (or vice versa)."""
    months_keys = set(genesis._DRY_MONTHS_BY_PATTERN)
    persistence_keys = set(genesis._PERSISTENCE_THRESHOLD_BY_PATTERN)
    gdv_keys = set(genesis._GDV_THRESHOLDS_BY_PATTERN)
    assert persistence_keys == months_keys, (
        "persistence threshold dict drifted from DRY_MONTHS pattern set"
    )
    assert gdv_keys == months_keys, (
        "GDV threshold dict drifted from DRY_MONTHS pattern set"
    )


# ---------------------------------------------------------------------------
# Region -> seasonal_pattern dispatch (sanity check that the inputs
# the resolvers consume are reachable from the actual region dropdown)
# ---------------------------------------------------------------------------

def test_seasonal_pattern_for_faial_is_temperate(genesis):
    """Faial / Azores Central uses the temperate seasonal pattern;
    Tool 07 baseline Faial runs must keep emitting persistence
    threshold 0.5."""
    pattern = genesis.TemporalStatistics._seasonal_pattern_for_region(
        "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
    )
    assert pattern == "temperate"
    assert genesis._resolve_persistence_threshold(pattern) == 0.5


def test_seasonal_pattern_for_mozambique_routes_to_arid_thresholds(genesis):
    """Mozambique routes through the resolver to the arid 0.3
    threshold — proves the end-to-end dispatch works."""
    pattern = genesis.TemporalStatistics._seasonal_pattern_for_region(
        "Mozambique",
    )
    assert pattern == "mozambique"
    assert genesis._resolve_persistence_threshold(pattern) == 0.3
    assert genesis._resolve_gdv_thresholds(pattern) == (0.2, 0.05)


# ---------------------------------------------------------------------------
# stat_source dropdown — Landsat ST_B10 removal (Defect C)
# ---------------------------------------------------------------------------

def test_stat_source_dropdown_excludes_dead_landsat_st_b10(genesis):
    """The Landsat ST_B10 option was a dead UI entry — validation
    rejected it. It must be gone from the dropdown after the fix."""
    tool = genesis.TemporalStatistics()
    params = tool.getParameterInfo()
    stat_source = next(p for p in params if p.name == "stat_source")
    choices = stat_source.filter.list
    assert "NDVI/NDWI (multispectral stacks)" in choices
    assert "LST (AST_08 thermal)" in choices
    assert all("ST_B10" not in opt for opt in choices), (
        f"dead Landsat ST_B10 option still present: {choices!r}"
    )


# ---------------------------------------------------------------------------
# Group dispatch reaches per-biome resolver via real region names
# ---------------------------------------------------------------------------

def test_classify_season_bucket_temperate_dry_months(genesis):
    """Temperate dry season is Jun-Sep (Azores summer)."""
    july = _dt.date(2024, 7, 15)
    february = _dt.date(2024, 2, 15)
    assert genesis._classify_season_bucket(july, "temperate") == "dry"
    assert genesis._classify_season_bucket(february, "temperate") == "wet"
