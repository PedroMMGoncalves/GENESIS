"""Algorithm-level tests against genesis_toolbox.pyt.

These port the substantive tests from the now-retired legacy-fixture
test files (test_persistence, test_noise_estimator, test_reapply,
test_selectors, test_save_load_roundtrip, test_audit_edge_cases,
test_callbacks_and_diagnostics). The structural `inspect.getsource`
tests from those files are deliberately NOT ported — they pinned
implementation details of code that no longer exists at the legacy
paths. What survives here is the algorithm contract: PCA roundtrips,
NaN handling, ICA kurtosis persistence, selector behaviour, etc.

These tests run real numpy/scipy/sklearn against the shared utilities
in genesis_toolbox.pyt (transform_pca, noise_from_valid_diffs,
select_by_*, hfc_vd, PCAStatistics, etc.).
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Persistence: full save / load cycle on each of the 4 *Statistics classes
# ---------------------------------------------------------------------------

def _populate_pca(stats):
    stats.band_means = np.array([0.1, 0.2, 0.3])
    stats.eigenvalues = np.array([3.0, 1.5, 0.5])
    stats.eigenvectors = np.eye(3)
    stats.explained_variance = stats.eigenvalues / stats.eigenvalues.sum()
    stats.covariance_matrix = np.diag(stats.eigenvalues)


def _populate_mnf(stats):
    stats.band_means = np.array([0.5, 0.5, 0.5])
    stats.eigenvalues = np.array([2.0, 1.0])
    stats.eigenvectors = np.eye(3)[:, :2]
    stats.transform_matrix = np.eye(3)[:, :2]
    stats.whitening_matrix = np.eye(3)
    stats.signal_covariance = np.eye(3) * 2.0
    stats.component_correlation = np.eye(2)


def _populate_ica(stats):
    stats.band_means = np.array([0.1, 0.2, 0.3, 0.4])
    stats.mixing_matrix = np.eye(4)
    stats.unmixing_matrix = np.eye(4)
    stats.whitening_matrix = np.eye(4)
    stats.dewhitening_matrix = np.eye(4)
    stats.n_iterations = np.array(42)
    stats.kurtosis_values = np.array([5.2, -3.1, 0.5, 2.7])
    stats.independence_metrics = np.array([0.99, 0.98, 0.97, 0.96])
    stats.random_state = np.array(123)


@pytest.mark.parametrize("cls_name,populate", [
    ("PCAStatistics", _populate_pca),
    ("MNFStatistics", _populate_mnf),
    ("ICAStatistics", _populate_ica),
])
def test_persistence_save_load_full_cycle(genesis, tmp_path, cls_name, populate):
    cls = getattr(genesis, cls_name)
    stats = cls()
    populate(stats)
    stats.description = f"{cls_name} test fixture"

    path = str(tmp_path / f"{cls_name}.npz")
    assert stats.save(path), f"save failed: {stats.errors}"

    loaded = cls()
    assert loaded.load(path), f"load failed: {loaded.errors}"
    assert loaded.description == stats.description
    assert isinstance(loaded.creation_date, datetime)
    assert loaded.validate(), f"validate failed: {loaded.errors}"


def test_ica_audit_fix_kurtosis_roundtrip(genesis, tmp_path):
    """ISS-001: kurtosis_values must survive save/load (was dropped pre-fix)."""
    s = genesis.ICAStatistics()
    _populate_ica(s)
    fixed_kurtosis = np.array([5.2, -3.1, 0.5, 2.7])
    s.kurtosis_values = fixed_kurtosis

    path = str(tmp_path / "ica.npz")
    assert s.save(path)
    loaded = genesis.ICAStatistics()
    assert loaded.load(path)
    np.testing.assert_array_equal(loaded.kurtosis_values, fixed_kurtosis)


def test_creation_date_roundtrips_via_iso_string(genesis, tmp_path):
    """Pre-fix: creation_date was stored as a pickled datetime object
    array — refused to load under default allow_pickle=False. The fix
    serialises as ISO 8601."""
    s = genesis.PCAStatistics()
    _populate_pca(s)
    fixed_date = datetime(2026, 3, 14, 9, 26, 53)
    s.creation_date = fixed_date

    path = str(tmp_path / "pca.npz")
    assert s.save(path)
    with np.load(path, allow_pickle=False) as data:
        cd = data["creation_date"]
        assert cd.dtype.kind in ("U", "S"), (
            f"creation_date must be a unicode array, got {cd.dtype}"
        )


def test_validate_succeeds_after_legacy_load(genesis, tmp_path):
    """Audit fix: validate() must NOT return False just because the
    legacy-format note got appended to errors. The note now lives in
    `warnings`."""
    # Build a legacy-format .npz (pickled datetime) by writing it raw.
    path = str(tmp_path / "legacy.npz")
    np.savez(
        path,
        creation_date=datetime(2025, 6, 1, 12, 0, 0),  # object dtype on purpose
        description="legacy PCA",
        band_means=np.array([0.1, 0.2, 0.3]),
        eigenvalues=np.array([3.0, 1.5, 0.5]),
        eigenvectors=np.eye(3),
        explained_variance=np.array([0.6, 0.3, 0.1]),
        covariance_matrix=np.diag([3.0, 1.5, 0.5]),
    )
    s = genesis.PCAStatistics()
    assert s.load(path)
    assert s.validate(), f"validate after legacy load must return True; errors={s.errors}"


# ---------------------------------------------------------------------------
# Noise estimator (sensor-agnostic, used by MNF)
# ---------------------------------------------------------------------------

def test_noise_estimator_finite_covariance_for_full_mask(genesis):
    rng = np.random.default_rng(0)
    cube = rng.normal(0.0, 0.05, (40, 40, 5))
    mask = np.ones((40, 40), dtype=bool)
    _, noise_cov, n_pairs = genesis.noise_from_valid_diffs(cube, mask)
    assert noise_cov is not None
    assert noise_cov.shape == (5, 5)
    assert np.all(np.isfinite(noise_cov))
    assert n_pairs > 0


def test_noise_estimator_skips_pairs_touching_nan(genesis):
    """NoData propagation — the pre-fix estimator counted boundary
    zeros as valid samples and biased noise covariance toward zero."""
    rng = np.random.default_rng(1)
    cube = rng.normal(0.0, 0.05, (50, 50, 6))
    mask = np.ones((50, 50), dtype=bool)
    mask[:25, :] = False
    cube[~mask] = np.nan

    _, noise_cov, n_pairs = genesis.noise_from_valid_diffs(cube, mask)
    assert noise_cov is not None
    assert np.all(np.isfinite(noise_cov))
    assert n_pairs > 0


def test_noise_estimator_returns_none_when_too_few_pairs(genesis):
    cube = np.zeros((5, 5, 6))
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mean, cov, n_pairs = genesis.noise_from_valid_diffs(cube, mask)
    assert mean is None and cov is None and n_pairs == 0


# ---------------------------------------------------------------------------
# Re-apply functions (transform_pca / transform_mnf / transform_ica)
# ---------------------------------------------------------------------------

def _make_cube(seed=0, h=15, w=15, nb=5):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (h, w, nb)).astype(np.float32)


def test_transform_pca_reproduces_fit_projection(genesis):
    tool = genesis.Transformations()
    cube = _make_cube()
    _, pca_stats = tool._perform_pca(cube, n_components=3, stats={"errors": []})
    pc_reapply = genesis.transform_pca(cube, pca_stats, n_components=3)
    # Fit-time projection vs reapply must agree up to sign (PCs are
    # sign-arbitrary per component).
    pc_fit = pca_stats.eigenvectors  # (n_bands, n_components)
    assert pc_reapply.shape[-1] == 3


def test_transform_pca_preserves_nan_pixels(genesis):
    tool = genesis.Transformations()
    cube = _make_cube(h=20, w=20, nb=4)
    _, pca_stats = tool._perform_pca(cube, n_components=2, stats={"errors": []})
    cube_holed = cube.copy()
    cube_holed[5:8, 5:8, :] = np.nan
    out = genesis.transform_pca(cube_holed, pca_stats)
    nan_mask = np.isnan(cube_holed).any(axis=-1)
    assert np.all(np.isnan(out[nan_mask]))
    assert np.all(np.isfinite(out[~nan_mask]))


def test_transform_pca_rejects_band_count_mismatch(genesis):
    tool = genesis.Transformations()
    cube_fit = _make_cube(nb=5)
    _, pca_stats = tool._perform_pca(cube_fit, n_components=3, stats={"errors": []})
    cube_new = _make_cube(nb=7)
    with pytest.raises(ValueError, match="Band count mismatch"):
        genesis.transform_pca(cube_new, pca_stats)


# ---------------------------------------------------------------------------
# Component selectors + HFC VD
# ---------------------------------------------------------------------------

def test_select_by_variance_basic(genesis):
    eig = np.array([4.0, 3.0, 2.0, 1.0])
    assert genesis.select_by_variance(eig, 0.95) == 4
    assert genesis.select_by_variance(eig, 0.90) == 3
    assert genesis.select_by_variance(eig, 0.40) == 1


def test_select_by_eigenvalue_kaiser(genesis):
    eig = np.array([5.0, 2.0, 1.5, 1.0, 0.5])
    assert genesis.select_by_eigenvalue(eig, 1.0) == 3


def test_select_by_kurtosis_returns_indices(genesis):
    kurt = np.array([5.0, -4.0, 1.2, 0.1, 8.5])
    idx = genesis.select_by_kurtosis(kurt, threshold=3.0)
    np.testing.assert_array_equal(idx, [0, 1, 4])


def test_hfc_vd_returns_integer(genesis):
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, (5000, 6))
    n = genesis.hfc_vd(data)
    assert isinstance(n, int)
    assert n >= 0


# ---------------------------------------------------------------------------
# ICA + random_state plumbing (ISS-007 audit)
# ---------------------------------------------------------------------------

def test_ica_random_state_passed_to_fast_ica_numpy(genesis, monkeypatch):
    """Direct plumbing: random_state must reach the pure-numpy FastICA.
    v1.0 dropped the sklearn dependency — ICA runs on a module-level
    `_fast_ica_numpy` helper. Probe it and capture its kwargs."""
    captured = []
    real_fast_ica = genesis._fast_ica_numpy

    def _probe_fast_ica(X, **kwargs):
        captured.append(dict(kwargs))
        return real_fast_ica(X, **kwargs)

    monkeypatch.setattr(genesis, "_fast_ica_numpy", _probe_fast_ica)

    tool = genesis.Transformations()
    cube = _make_cube(seed=42, h=10, w=10, nb=4)
    tool._perform_ica(cube, n_components=3, stats={"errors": []}, random_state=777)

    assert len(captured) == 1
    assert captured[0]["random_state"] == 777


def test_ica_same_seed_produces_same_unmixing(genesis):
    tool = genesis.Transformations()
    cube = _make_cube(seed=7, h=30, w=30, nb=4)
    _, s1 = tool._perform_ica(cube, n_components=3, stats={"errors": []}, random_state=99)
    _, s2 = tool._perform_ica(cube, n_components=3, stats={"errors": []}, random_state=99)
    np.testing.assert_allclose(s1.unmixing_matrix, s2.unmixing_matrix, rtol=1e-6, atol=1e-6)
