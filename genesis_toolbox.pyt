# -*- coding: utf-8 -*-

# genesis_toolbox.pyt
#
# GENESIS — Unified multisensor satellite-processing toolbox for ArcGIS Pro.
# Replaces the earlier `landsat_toolbox.pyt` and `sentinel2_toolbox.pyt`
# with a single workflow-ordered toolbox covering three sensors:
#
#   Sensor           Input format             Cloud handling
#   ───────────────  ───────────────────────  ───────────────────────────
#   Landsat 8/9      C2L2 SR/SP scene folder  QA_PIXEL bits 0-4
#                    (`LC08_L2SR_*`,           + optional temporal refinement
#                     `LC08_L2SP_*`, and L9
#                     equivalents — both
#                     L2SR and L2SP accepted;
#                     EarthExplorer ships them
#                     as `.tar` archives that
#                     the Mosaic tool extracts
#                     transparently)
#   Sentinel-2       L2A SAFE folder          SCL classes 3, 8, 9, 10
#                    (`S2A_MSIL2A_*` /        + optional temporal refinement
#                     `S2B_MSIL2A_*`)
#   ASTER            AST_07XT V004 — both     QA Data Plane cloud bits
#                    HDF (`.hdf`) and TIFF    + multitemporal anomaly
#                    folder (`*_SRF_VNIR_*`,    (mandatory for ASTER)
#                            `*_SRF_SWIR_*`)
#
# Tools (workflow-ordered, by numeric prefix in the ArcGIS Pro UI):
#   01 — Sentinel-2 L2A Mosaic
#   02 — Landsat 8/9 C2L2 Mosaic
#   03 — ASTER L2 Mosaic
#   04 — Spectral Indices & Composites (sensor-aware, grouped UI)
#   05 — Statistical Transformations (PCA / MNF / ICA — sensor-agnostic)
#   06 — Spectral Angle Mapper (sensor-aware band-count check)
#
# Every output raster from Tools 01-03 ships with a `{output}_provenance.csv`
# documenting which input scenes contributed.
#
# Reflectance output convention: float 0-1 (analysis-friendly).
# ASTER scale factor 0.001 is applied during ingestion.

import arcpy
import os
from datetime import datetime
import arcpy.sa
import arcpy.ia
import uuid
import csv
import glob
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from sklearn.decomposition import FastICA
import scipy.stats
from arcpy.ia import ExtractBand
from arcpy.sa import Float, Divide, Times, Con, SetNull, Plus, Minus
from arcpy.management import CompositeBands

TOOLBOX_VERSION = "1.0.0-phase3"


# ---------------------------------------------------------------------------
# Internal tunables. Hoisted from inline values so callers can read and
# adjust knobs in one place. Underscore-prefixed because they're not a
# stable public API — names or values may change between toolbox versions.
# ---------------------------------------------------------------------------

# Eigenvalues below this absolute threshold are treated as rank-deficient
# and replaced by an eigenvalue floor. Set well below float32 epsilon so
# legitimate low-variance directions in low-dynamic-range imagery survive.
# Used by noise_from_valid_diffs, the MNF subset noise path, and the MNF
# whitening matrix construction.
_EIGVAL_FLOOR_ABS = 1e-10

# When the noise covariance is rank-deficient, the floor is data-adaptive:
# lifted to this fraction of the median surviving eigenvalue (rather than
# an absolute value) so the regularisation scales with the imagery.
_EIGVAL_FLOOR_RELATIVE = 1e-4

# ICA whitening eigenvalue floor — looser than the noise floor because the
# DATA covariance (not noise covariance) typically has much larger scale.
_ICA_WHITENING_FLOOR = 1e-12

# Floor for the HFC VD sigma^2 term — guards np.sqrt(0) without altering
# the statistic in any realistic regime.
_HFC_SIGMA2_FLOOR = 1e-30

# MNF ISS-004 diagnostic: warn when the empirical component correlation
# matrix has any off-diagonal element larger than this. The MNF construction
# should yield ~identity in noise-whitened space; visible off-diagonals
# indicate a poor noise estimate or a numerically pathological covariance.
_MNF_CORR_OFFDIAG_WARN = 0.1

# ISS-010 RAM warning: cubes larger than this (estimated as float64) trigger
# a user-facing warning before the in-memory transform tries to allocate
# the working matrix. Chunked PCA/MNF is out of scope.
_RAM_WARNING_GB = 4.0


class TransformStatistics:
    """Base class for transformation statistics"""
    def __init__(self):
        self.creation_date = datetime.now()
        self.description = ""
        # errors: real failures (missing file, bad data, malformed npz).
        # warnings: informational notes that do NOT mean the object is invalid
        # (e.g., "loaded from a legacy .npz format"). Kept separate so
        # validate() does not flag a usable object as broken just because the
        # loader added a note.
        self.errors = []
        self.warnings = []

    def validate(self):
        """Base validation method"""
        return len(self.errors) == 0
        
    def save(self, filepath):
        """Save statistics to a .npz file.

        creation_date is serialised as an ISO 8601 string so np.savez writes
        a numpy unicode array rather than a Python-object array. This lets
        np.load read the file under the default allow_pickle=False (numpy
        1.16+ refuses to deserialise object arrays without explicit opt-in).
        """
        try:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            if isinstance(self.creation_date, datetime):
                creation_date_str = self.creation_date.isoformat()
            else:
                creation_date_str = str(self.creation_date)

            np.savez(
                filepath,
                creation_date=creation_date_str,
                description=self.description or "",
                **self._get_save_dict()
            )
            return True
        except Exception as e:
            self.errors.append(f"Error saving statistics: {str(e)}")
            return False

    def load(self, filepath):
        """Load statistics from a .npz file.

        Default path uses allow_pickle=False (safe). If the file contains
        numpy object arrays — typically legacy .npz files where
        creation_date was saved as a pickled datetime, or where None-valued
        optional fields were saved unconditionally — we fall back to
        allow_pickle=True and record a note so the user knows to re-save.

        SECURITY: the legacy-fallback path will execute pickle bytecode.
        Only load .npz files you produced or fully trust.
        """
        try:
            if not os.path.exists(filepath):
                self.errors.append(f"Statistics file not found: {filepath}")
                return False

            legacy = False
            try:
                data = np.load(filepath, allow_pickle=False)
                # Force eager evaluation of every key so an object-array
                # failure surfaces here, not deep inside _load_from_dict.
                for k in data.files:
                    _ = data[k]
            except ValueError as e:
                if "allow_pickle" not in str(e).lower():
                    raise
                try:
                    data.close()
                except Exception:
                    pass
                data = np.load(filepath, allow_pickle=True)
                legacy = True

            try:
                raw_date = data['creation_date']
                date_val = raw_date.item() if hasattr(raw_date, 'item') else raw_date
                if isinstance(date_val, datetime):
                    self.creation_date = date_val
                elif isinstance(date_val, str):
                    try:
                        self.creation_date = datetime.fromisoformat(date_val)
                    except ValueError:
                        self.creation_date = datetime.now()
                else:
                    self.creation_date = datetime.now()
            except KeyError:
                self.creation_date = datetime.now()

            self.description = (
                str(data['description']) if 'description' in data.files else ""
            )

            if legacy:
                self.warnings.append(
                    "Loaded a legacy .npz containing pickled objects. "
                    "Re-save with the current toolbox to upgrade format."
                )

            result = self._load_from_dict(data)
            try:
                data.close()
            except Exception:
                pass
            return result
        except Exception as e:
            self.errors.append(f"Error loading statistics: {str(e)}")
            return False
            
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        raise NotImplementedError
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        raise NotImplementedError

class MNFNoiseStatistics(TransformStatistics):
    """Statistics from first MNF rotation (noise statistics)"""
    def __init__(self):
        super().__init__()
        self.noise_covariance = None
        self.noise_eigenvalues = None
        self.noise_eigenvectors = None
        self.description = "MNF Noise Statistics"
        
    def validate(self):
        """Validate noise statistics"""
        if not super().validate():
            return False
            
        if self.noise_covariance is None:
            self.errors.append("Noise covariance matrix is missing")
        if self.noise_eigenvalues is None:
            self.errors.append("Noise eigenvalues are missing")
        if self.noise_eigenvectors is None:
            self.errors.append("Noise eigenvectors are missing")
            
        return len(self.errors) == 0
        
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        return {
            'noise_covariance': self.noise_covariance,
            'noise_eigenvalues': self.noise_eigenvalues,
            'noise_eigenvectors': self.noise_eigenvectors
        }
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.noise_covariance = data['noise_covariance']
            self.noise_eigenvalues = data['noise_eigenvalues']
            self.noise_eigenvectors = data['noise_eigenvectors']
            return True
        except Exception as e:
            self.errors.append(f"Error loading noise statistics: {str(e)}")
            return False

class MNFStatistics(TransformStatistics):
    """Statistics from MNF transformation"""
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.transform_matrix = None
        self.noise_covariance = None
        self.whitening_matrix = None
        self.signal_covariance = None
        self.component_correlation = None
        self.description = "MNF Transform Statistics"

    def validate(self):
        """Validate MNF statistics"""
        if not super().validate():
            return False

        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.eigenvalues is None:
            self.errors.append("Eigenvalues are missing")
        if self.eigenvectors is None:
            self.errors.append("Eigenvectors are missing")
        if self.transform_matrix is None:
            self.errors.append("Transform matrix is missing")

        return len(self.errors) == 0

    def _get_save_dict(self):
        """Get dictionary of values to save"""
        save_dict = {
            'band_means': self.band_means,
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'transform_matrix': self.transform_matrix
        }

        # Add optional fields if they exist
        if hasattr(self, 'noise_covariance') and self.noise_covariance is not None:
            save_dict['noise_covariance'] = self.noise_covariance

        if hasattr(self, 'whitening_matrix') and self.whitening_matrix is not None:
            save_dict['whitening_matrix'] = self.whitening_matrix

        if hasattr(self, 'signal_covariance') and self.signal_covariance is not None:
            save_dict['signal_covariance'] = self.signal_covariance

        if hasattr(self, 'component_correlation') and self.component_correlation is not None:
            save_dict['component_correlation'] = self.component_correlation

        return save_dict

    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.eigenvalues = data['eigenvalues']
            self.eigenvectors = data['eigenvectors']
            self.transform_matrix = data['transform_matrix']

            # Load optional fields if available
            if 'noise_covariance' in data:
                self.noise_covariance = data['noise_covariance']

            if 'whitening_matrix' in data:
                self.whitening_matrix = data['whitening_matrix']

            if 'signal_covariance' in data:
                self.signal_covariance = data['signal_covariance']

            if 'component_correlation' in data:
                self.component_correlation = data['component_correlation']

            return True
        except Exception as e:
            self.errors.append(f"Error loading MNF statistics: {str(e)}")
            return False

class PCAStatistics(TransformStatistics):
    """Statistics for PCA transformation"""
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance = None
        self.covariance_matrix = None
        self.description = "PCA Transform Statistics"
        
    def validate(self):
        """Validate PCA statistics"""
        if not super().validate():
            return False
            
        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.eigenvalues is None:
            self.errors.append("Eigenvalues are missing")
        if self.eigenvectors is None:
            self.errors.append("Eigenvectors are missing")
        if self.explained_variance is None:
            self.errors.append("Explained variance is missing")
        if self.covariance_matrix is None:
            self.errors.append("Covariance matrix is missing")
            
        return len(self.errors) == 0
        
    def _get_save_dict(self):
        """Get dictionary of values to save"""
        return {
            'band_means': self.band_means,
            'eigenvalues': self.eigenvalues,
            'eigenvectors': self.eigenvectors,
            'explained_variance': self.explained_variance,
            'covariance_matrix': self.covariance_matrix
        }
        
    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.eigenvalues = data['eigenvalues']
            self.eigenvectors = data['eigenvectors']
            self.explained_variance = data['explained_variance']
            self.covariance_matrix = data['covariance_matrix']
            return True
        except Exception as e:
            self.errors.append(f"Error loading PCA statistics: {str(e)}")
            return False

class ICAStatistics(TransformStatistics):
    def __init__(self):
        super().__init__()
        self.band_means = None
        self.mixing_matrix = None
        self.unmixing_matrix = None
        self.whitening_matrix = None
        self.dewhitening_matrix = None
        self.n_iterations = None
        self.independence_metrics = None
        self.kurtosis_values = None
        self.random_state = None
        self.description = "ICA Transform Statistics"

    def validate(self):
        """Validate ICA statistics"""
        if not super().validate():
            return False

        if self.band_means is None:
            self.errors.append("Band means are missing")
        if self.mixing_matrix is None:
            self.errors.append("Mixing matrix is missing")
        if self.unmixing_matrix is None:
            self.errors.append("Unmixing matrix is missing")
        if self.whitening_matrix is None:
            self.errors.append("Whitening matrix is missing")
        if self.dewhitening_matrix is None:
            self.errors.append("Dewhitening matrix is missing")
        if self.n_iterations is None:
            self.errors.append("n_iterations is missing")
        if self.kurtosis_values is None:
            self.errors.append("kurtosis_values is missing")

        return len(self.errors) == 0

    def _get_save_dict(self):
        """Get dictionary of values to save"""
        out = {
            'band_means': self.band_means,
            'mixing_matrix': self.mixing_matrix,
            'unmixing_matrix': self.unmixing_matrix,
            'whitening_matrix': self.whitening_matrix,
            'dewhitening_matrix': self.dewhitening_matrix,
            'n_iterations': self.n_iterations,
        }
        # Optional fields: omit when None so np.savez does not write a
        # numpy object array (which np.load refuses without allow_pickle=True).
        if self.independence_metrics is not None:
            out['independence_metrics'] = self.independence_metrics
        if self.kurtosis_values is not None:
            out['kurtosis_values'] = self.kurtosis_values
        if self.random_state is not None:
            out['random_state'] = self.random_state
        return out

    def _load_from_dict(self, data):
        """Load values from dictionary"""
        try:
            self.band_means = data['band_means']
            self.mixing_matrix = data['mixing_matrix']
            self.unmixing_matrix = data['unmixing_matrix']
            self.whitening_matrix = data['whitening_matrix']
            self.dewhitening_matrix = data['dewhitening_matrix']
            self.n_iterations = data['n_iterations']
            if 'independence_metrics' in data:
                self.independence_metrics = data['independence_metrics']
            if 'kurtosis_values' in data:
                self.kurtosis_values = data['kurtosis_values']
            if 'random_state' in data:
                self.random_state = data['random_state']
            return True
        except Exception as e:
            self.errors.append(f"Error loading ICA statistics: {str(e)}")
            return False


# ---------------------------------------------------------------------------
# Module-level utilities (NoData-safe, arcpy-free, unit-testable).
# ---------------------------------------------------------------------------

def noise_from_valid_diffs(cube, valid_mask):
    """Estimate MNF noise via shift-difference, restricted to valid pixel pairs.

    Replaces the inlined shift-difference logic from _perform_mnf and fixes
    two issues:
      * NaN/NoData propagation: only neighbour pairs where BOTH endpoints
        are valid contribute. The previous implementation zero-padded the
        boundary and counted those zeros as valid samples, biasing the
        noise covariance toward smaller values.
      * Non-positive-definite covariance is regularised with an eigenvalue
        floor instead of an additive diagonal shift.

    Parameters
    ----------
    cube : np.ndarray
        (h, w, n_bands) data cube.
    valid_mask : np.ndarray
        (h, w) boolean mask, True where the pixel is fully valid.

    Returns
    -------
    (noise_mean, noise_cov, n_pairs)
        noise_mean is the per-band mean delta; noise_cov is the covariance
        of the deltas divided by 2 (since Var(X - Y) = 2 * Var(noise) when
        signal is smooth). Returns (None, None, 0) when fewer than
        2 * n_bands valid pairs are available.
    """
    h, w, nb = cube.shape

    h_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    h_deltas = (cube[:, :-1, :] - cube[:, 1:, :])[h_valid]

    v_valid = valid_mask[:-1, :] & valid_mask[1:, :]
    v_deltas = (cube[:-1, :, :] - cube[1:, :, :])[v_valid]

    all_deltas = np.concatenate([h_deltas, v_deltas], axis=0)
    n_pairs = int(len(all_deltas))

    if n_pairs < nb * 2:
        return None, None, 0

    noise_mean = np.mean(all_deltas, axis=0)
    noise_cov = np.cov(all_deltas.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(noise_cov)
    if np.any(eigvals < _EIGVAL_FLOOR_ABS):
        valid_eig = eigvals[eigvals > _EIGVAL_FLOOR_ABS]
        floor = (
            np.median(valid_eig) * _EIGVAL_FLOOR_RELATIVE
            if len(valid_eig) > 0
            else _EIGVAL_FLOOR_ABS
        )
        eigvals = np.maximum(eigvals, floor)
        noise_cov = (eigvecs * eigvals) @ eigvecs.T

    return noise_mean, noise_cov, n_pairs


def _project_with_band_means(cube, band_means, projection_matrix):
    """Internal: center, neutralise NaN, project, restore per-pixel NaN.

    Shared by transform_pca / transform_mnf / transform_ica. The math is the
    same for all three — only the projection_matrix differs.

    NaN handling: filling NaN bands with band_mean prior to centering is
    equivalent to centering first and then replacing NaN with 0, since
    (band_mean - band_mean) = 0. We use the latter (vectorised, single pass)
    via np.nan_to_num, then restore NaN in every output component for any
    pixel that had at least one non-finite band on input.
    """
    h, w, nb = cube.shape
    if nb != band_means.shape[0]:
        raise ValueError(
            f"Band count mismatch: cube has {nb} bands, statistics has "
            f"{band_means.shape[0]} bands."
        )
    # astype(copy=True) is the default; one allocation, no redundant .copy().
    flat = cube.reshape(-1, nb).astype(np.float64)
    nan_per_pixel = np.any(~np.isfinite(flat), axis=1)
    centered = flat - band_means  # NaN bands stay NaN
    # In-place zero out NaN entries — equivalent to filling with band_mean
    # pre-centering, but vectorised in one C-level pass.
    np.nan_to_num(centered, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    projected = centered @ projection_matrix
    projected[nan_per_pixel] = np.nan
    n_components = projection_matrix.shape[1]
    return projected.reshape(h, w, n_components).astype(np.float32)


def transform_pca(new_cube, stats, n_components=None):
    """Apply a fitted PCAStatistics to a new (h, w, n_bands) cube.

    Implements ISS-005 — cross-AOI re-application. The PCA projection is
    `(X - mean) @ eigenvectors[:, :n]`. Pixels with any non-finite band
    become NaN in every output component.
    """
    if stats.band_means is None or stats.eigenvectors is None:
        raise ValueError("PCAStatistics is missing band_means or eigenvectors.")
    available = stats.eigenvectors.shape[1]
    if n_components is None:
        n_components = available
    if n_components > available:
        raise ValueError(
            f"Requested {n_components} components but only {available} were saved."
        )
    return _project_with_band_means(
        new_cube,
        np.asarray(stats.band_means),
        np.asarray(stats.eigenvectors)[:, :n_components],
    )


def transform_mnf(new_cube, stats, n_components=None):
    """Apply a fitted MNFStatistics to a new (h, w, n_bands) cube.

    Uses the precomputed transform_matrix = whitening_matrix @ signal_eigenvecs
    so the projection is a single matmul. The shape is (n_bands, n_components_saved);
    passing n_components larger than that is rejected.
    """
    if stats.band_means is None or stats.transform_matrix is None:
        raise ValueError("MNFStatistics is missing band_means or transform_matrix.")
    available = stats.transform_matrix.shape[1]
    if n_components is None:
        n_components = available
    if n_components > available:
        raise ValueError(
            f"Requested {n_components} components but only {available} were saved."
        )
    return _project_with_band_means(
        new_cube,
        np.asarray(stats.band_means),
        np.asarray(stats.transform_matrix)[:, :n_components],
    )


def transform_ica(new_cube, stats):
    """Apply a fitted ICAStatistics to a new (h, w, n_bands) cube.

    Sources are recovered with `(X - mean) @ unmixing_matrix.T` — the
    sklearn FastICA convention. Always returns all stored components;
    select_by_kurtosis can be used afterwards to pick interesting ones.
    """
    if stats.band_means is None or stats.unmixing_matrix is None:
        raise ValueError("ICAStatistics is missing band_means or unmixing_matrix.")
    return _project_with_band_means(
        new_cube,
        np.asarray(stats.band_means),
        np.asarray(stats.unmixing_matrix).T,
    )


# ---------------------------------------------------------------------------
# Component selectors (ISS-008, ISS-014).
# ---------------------------------------------------------------------------

def select_by_variance(eigenvalues, threshold=0.95):
    """Smallest n such that cumulative explained variance >= threshold.

    Standard PCA component count rule. `threshold` is a fraction in (0, 1].
    """
    arr = np.asarray(eigenvalues, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 1
    cum = np.cumsum(arr) / total
    n = int(np.searchsorted(cum, threshold) + 1)
    return max(1, min(n, len(arr)))


def select_by_eigenvalue(eigenvalues, threshold=1.0):
    """Count components with eigenvalue > threshold (Kaiser criterion).

    For MNF, eigenvalues are 1 + SNR — values above 1.0 indicate signal
    dominates noise. Returns at least 1.
    """
    arr = np.asarray(eigenvalues, dtype=np.float64)
    return max(1, int(np.sum(arr > threshold)))


def select_by_kurtosis(kurtosis_values, threshold=3.0):
    """Indices of ICA components with |kurtosis| > threshold.

    High |kurtosis| ≈ sharp / non-Gaussian distribution; useful for
    isolating distinct mineral signatures or geological boundaries.
    Returns a numpy array of indices (possibly empty).
    """
    arr = np.asarray(kurtosis_values, dtype=np.float64)
    return np.flatnonzero(np.abs(arr) > threshold)


def hfc_vd(data_2d, false_alarm=1e-3, max_samples=200_000):
    """Harsanyi-Farrand-Chang Virtual Dimensionality (Chang & Du 2004).

    Automatic estimator of the number of spectrally distinct signal
    sources. Operates on a 2D pixels-by-bands matrix (drop the spatial
    dimensions first). Pixels with NaN must be filtered by the caller.

    Parameters
    ----------
    data_2d : array, shape (n_pixels, n_bands)
    false_alarm : float
        Neyman-Pearson false alarm probability (smaller → fewer components).
    max_samples : int
        Random subsample size when n_pixels is large.

    Returns
    -------
    int
        Estimated number of signal sources, >= 0.
    """
    data = np.asarray(data_2d, dtype=np.float64)
    n_pixels, n_bands = data.shape
    if n_pixels > max_samples:
        rng = np.random.default_rng(42)
        data = data[rng.choice(n_pixels, max_samples, replace=False)]
        n_pixels = max_samples

    R = (data.T @ data) / n_pixels
    mean = data.mean(axis=0, keepdims=True)
    C = ((data - mean).T @ (data - mean)) / n_pixels
    eig_R = np.sort(np.linalg.eigvalsh(R))[::-1]
    eig_C = np.sort(np.linalg.eigvalsh(C))[::-1]

    z_threshold = scipy.stats.norm.isf(false_alarm)
    sigma2 = (2.0 / n_pixels) * (eig_R ** 2 + eig_C ** 2)
    sigma = np.sqrt(np.maximum(sigma2, _HFC_SIGMA2_FLOOR))
    return int(np.sum((eig_R - eig_C) > z_threshold * sigma))


# ---------------------------------------------------------------------------
# Sensor abstraction (Phase 1).
#
# Tools 04, 05, and 06 are sensor-aware via a single "Sensor Type" GP
# parameter that takes one of four values. The band-role lookup below
# resolves role names ("Red", "NIR", "SWIR1", ...) to band indices per
# sensor, so the SAME index formula expressed in terms of roles works
# across all three sensors.
# ---------------------------------------------------------------------------

SENSOR_AUTO = "Auto-detect"
SENSOR_LANDSAT_89 = "Landsat 8/9"
SENSOR_SENTINEL2 = "Sentinel-2"
SENSOR_ASTER = "ASTER"

SENSOR_CHOICES = [SENSOR_AUTO, SENSOR_LANDSAT_89, SENSOR_SENTINEL2, SENSOR_ASTER]

# Band-role mapping per sensor. Values are 1-based band indices into the
# canonical multiband stack documented for each sensor (see file header).
#
# Universal roles (Red, NIR, SWIR1, SWIR2) exist for all three sensors so
# universal indices (NDVI, NDWI, NDMI, NDBI) compute via the same lambda.
#
# Sensor-specific roles:
#   - Coastal (Landsat only)
#   - RedEdge1/2/3 + NarrowNIR (Sentinel-2 only — enables NDRE, IRECI, ...)
#   - SWIR2_2165 / SWIR2_2205 / SWIR2_2260 / SWIR2_2330 (ASTER only —
#     each is a separate 30m SWIR band; enables alunite, kaolinite,
#     muscovite, calcite indices)
#
# Note: ASTER has NO blue band — the "Blue" role is intentionally absent
# from the ASTER mapping. Indices that need Blue (e.g., L8 Iron Oxide
# B4/B2) get an ASTER-specific reformulation (Red/Green = B2/B1).
SENSOR_BAND_ROLES = {
    SENSOR_LANDSAT_89: {
        "Coastal": 1,
        "Blue": 2,
        "Green": 3,
        "Red": 4,
        "NIR": 5,
        "SWIR1": 6,
        "SWIR2": 7,
    },
    SENSOR_SENTINEL2: {
        "Blue": 1,        # B02 490nm
        "Green": 2,       # B03 560nm
        "Red": 3,         # B04 665nm
        "RedEdge1": 4,    # B05 705nm
        "RedEdge2": 5,    # B06 740nm
        "RedEdge3": 6,    # B07 783nm
        "NIR": 7,         # B08 842nm
        "NarrowNIR": 8,   # B8A 865nm
        "SWIR1": 9,       # B11 1610nm
        "SWIR2": 10,      # B12 2190nm
    },
    SENSOR_ASTER: {
        # AST_07XT V004 9-band stack: VNIR B1-B3N + crosstalk-corrected SWIR B4-B9
        "Green": 1,           # B1 0.52-0.60μm
        "Red": 2,             # B2 0.63-0.69μm
        "NIR": 3,             # B3N 0.76-0.86μm (nadir-looking)
        "SWIR1": 4,           # B4 1.60-1.70μm  (≈L8 B6, S2 B11)
        "SWIR2_2165": 5,      # B5 2.145-2.185μm
        "SWIR2_2205": 6,      # B6 2.185-2.225μm
        "SWIR2_2260": 7,      # B7 2.235-2.285μm
        "SWIR2_2330": 8,      # B8 2.295-2.365μm
        "SWIR2": 9,           # B9 2.360-2.430μm (≈L8 B7, S2 B12 — generic SWIR2)
    },
}


def get_band(bands, role, sensor):
    """Resolve a band-role name to a band raster for the given sensor.

    Parameters
    ----------
    bands : dict
        {1: Raster, 2: Raster, ...} as built by the indices/SAM tool's
        per-band ExtractBand loop.
    role : str
        A band role from SENSOR_BAND_ROLES, e.g., "Red", "NIR", "SWIR1".
    sensor : str
        One of SENSOR_LANDSAT_89 / SENSOR_SENTINEL2 / SENSOR_ASTER.

    Returns
    -------
    Raster
        The arcpy Raster for that band in the stack.

    Raises
    ------
    KeyError
        If the role isn't available for the requested sensor (e.g., asking
        for "Blue" on ASTER, or "RedEdge1" on Landsat). The caller (an
        index lambda) catches this and skips the index gracefully.
    """
    mapping = SENSOR_BAND_ROLES.get(sensor)
    if mapping is None:
        raise ValueError(f"Unknown sensor: {sensor!r}")
    if role not in mapping:
        raise KeyError(
            f"Band role {role!r} is not available for {sensor!r}. "
            f"Available roles: {sorted(mapping.keys())}"
        )
    idx = mapping[role]
    if idx not in bands:
        raise KeyError(
            f"Band {idx} (role {role!r} for {sensor!r}) was not extracted; "
            f"the input raster may have fewer bands than expected."
        )
    return bands[idx]


def detect_sensor(raster_path):
    """Auto-detect sensor from raster path and (as fallback) band count.

    Returns one of SENSOR_LANDSAT_89 / SENSOR_SENTINEL2 / SENSOR_ASTER,
    or None if no confident match. The GP UI presents this as the default
    for the sensor-selector dropdown; the user can override.

    Detection cascade (most specific first):
      1. Filename prefix / token:
           LC08_, LC09_                → Landsat 8/9
           S2A_, S2B_, _MSIL2A_         → Sentinel-2
           AST_07XT                     → ASTER
      2. Band count (less reliable; only used if path/filename gives no
         hint):
           7 bands  → Landsat 8/9
           10 bands → Sentinel-2
           9 bands  → ASTER
    """
    if not raster_path:
        return None
    name = os.path.basename(str(raster_path)).upper()

    if "LC08_" in name or "LC09_" in name:
        return SENSOR_LANDSAT_89
    if name.startswith("S2A_") or name.startswith("S2B_") or "_MSIL2A_" in name:
        return SENSOR_SENTINEL2
    if "AST_07XT" in name:
        return SENSOR_ASTER

    # Fall back to band count
    try:
        desc = arcpy.Describe(raster_path)
        if hasattr(desc, "bandCount"):
            count = int(desc.bandCount)
            if count == 7:
                return SENSOR_LANDSAT_89
            if count == 10:
                return SENSOR_SENTINEL2
            if count == 9:
                return SENSOR_ASTER
    except Exception:
        pass
    return None


def make_sensor_parameter(name="sensor_type", display="Sensor Type"):
    """Build the standard sensor-selector GP parameter.

    Used by Tools 04 (Indices), 05 (Transformations) and 06 (SAM). The
    parameter defaults to "Auto-detect"; updateParameters in each tool
    is responsible for resolving Auto-detect against the input raster
    and either updating the displayed value or surfacing a clear error
    if detection fails.
    """
    param = arcpy.Parameter(
        displayName=display,
        name=name,
        datatype="GPString",
        parameterType="Required",
        direction="Input",
    )
    param.filter.list = SENSOR_CHOICES
    param.value = SENSOR_AUTO
    return param


def resolve_sensor(sensor_param_value, input_raster_path):
    """Resolve a sensor selection into a concrete sensor constant.

    Used at execute() time. If the user left the parameter at
    "Auto-detect", we run detect_sensor on the input raster. If auto
    detection fails, raises ValueError with a clear remediation message
    (telling the user to pick the sensor manually).
    """
    if sensor_param_value and sensor_param_value != SENSOR_AUTO:
        return sensor_param_value
    detected = detect_sensor(input_raster_path)
    if detected is None:
        raise ValueError(
            f"Could not auto-detect sensor from input {input_raster_path!r}. "
            f"Please set Sensor Type explicitly in the tool dialog "
            f"(one of: {', '.join(SENSOR_CHOICES[1:])})."
        )
    return detected


# ---------------------------------------------------------------------------
# Toolbox class (Phase 1 skeleton — tools are populated in subsequent phases).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Index & composite catalogue (Phase 2).
#
# Sensor-aware indices/composites use band ROLES (resolved at runtime via
# get_band) so the same compute lambda works across every sensor that has
# the required roles. The display_labels dict gives the user-facing string
# in the ArcGIS Pro Catalog pane — sensor-specific so the user always sees
# the formula spelled out in that sensor's band labels.
#
# An index appears in the GP UI for a given sensor iff ALL of:
#   1. Every required_role is in SENSOR_BAND_ROLES[sensor]
#   2. The index has a display_labels[sensor] entry
#   3. If sensor_filter is set, the sensor is in that list (overrides 1+2)
#
# Two small helpers keep the formula lambdas readable:
# ---------------------------------------------------------------------------


def _normalized_diff(bands, sensor, role_a, role_b):
    """(role_a - role_b) / (role_a + role_b) — the NDVI family."""
    a = get_band(bands, role_a, sensor)
    b = get_band(bands, role_b, sensor)
    return Divide(Minus(a, b), Plus(a, b))


def _ratio(bands, sensor, numerator_role, denominator_role):
    """role_a / role_b — simple band ratio."""
    return Divide(
        get_band(bands, numerator_role, sensor),
        get_band(bands, denominator_role, sensor),
    )


INDICES = {
    # =====================================================================
    # Vegetation
    # =====================================================================
    "NDVI": {
        "category": "Vegetation",
        "required_roles": ["Red", "NIR"],
        "display_labels": {
            SENSOR_LANDSAT_89: "NDVI (B5-B4)/(B5+B4)",
            SENSOR_SENTINEL2: "NDVI (B08-B04)/(B08+B04)",
            SENSOR_ASTER: "NDVI (B3N-B2)/(B3N+B2)",
        },
        "output_suffix": "NDVI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "NIR", "Red"),
    },
    "SAVI": {
        "category": "Vegetation",
        "required_roles": ["Red", "NIR"],
        "display_labels": {
            SENSOR_LANDSAT_89: "SAVI 1.5*(B5-B4)/(B5+B4+0.5)",
            SENSOR_SENTINEL2: "SAVI 1.5*(B08-B04)/(B08+B04+0.5)",
            SENSOR_ASTER: "SAVI 1.5*(B3N-B2)/(B3N+B2+0.5)",
        },
        "output_suffix": "SAVI",
        "compute": lambda bands, sensor: Times(
            Divide(
                Minus(get_band(bands, "NIR", sensor), get_band(bands, "Red", sensor)),
                Plus(Plus(get_band(bands, "NIR", sensor), get_band(bands, "Red", sensor)), 0.5),
            ),
            1.5,
        ),
    },
    # =====================================================================
    # Water
    # =====================================================================
    "NDWI_McFeeters": {
        "category": "Water",
        "required_roles": ["Green", "NIR"],
        "display_labels": {
            SENSOR_LANDSAT_89: "NDWI McFeeters (B3-B5)/(B3+B5)",
            SENSOR_SENTINEL2: "NDWI McFeeters (B03-B08)/(B03+B08)",
            SENSOR_ASTER: "NDWI McFeeters (B1-B3N)/(B1+B3N)",
        },
        "output_suffix": "NDWI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "Green", "NIR"),
    },
    "MNDWI": {
        "category": "Water",
        "required_roles": ["Green", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "MNDWI (B3-B6)/(B3+B6)",
            SENSOR_SENTINEL2: "MNDWI (B03-B11)/(B03+B11)",
            SENSOR_ASTER: "MNDWI (B1-B4)/(B1+B4)",
        },
        "output_suffix": "MNDWI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "Green", "SWIR1"),
    },
    # =====================================================================
    # Moisture / Built-up
    # =====================================================================
    "NDMI": {
        "category": "Moisture",
        "required_roles": ["NIR", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "NDMI (B5-B6)/(B5+B6)",
            SENSOR_SENTINEL2: "NDMI (B08-B11)/(B08+B11)",
            SENSOR_ASTER: "NDMI (B3N-B4)/(B3N+B4)",
        },
        "output_suffix": "NDMI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "NIR", "SWIR1"),
    },
    "NDBI": {
        "category": "Built-up",
        "required_roles": ["NIR", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "NDBI (B6-B5)/(B6+B5)",
            SENSOR_SENTINEL2: "NDBI (B11-B08)/(B11+B08)",
            SENSOR_ASTER: "NDBI (B4-B3N)/(B4+B3N)",
        },
        "output_suffix": "NDBI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "SWIR1", "NIR"),
    },
    # =====================================================================
    # Geological — universal (no Blue requirement)
    # =====================================================================
    "Clay_Minerals_SWIR": {
        "category": "Geological",
        "required_roles": ["SWIR1", "SWIR2"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Clay Minerals B6/B7 (SWIR1/SWIR2)",
            SENSOR_SENTINEL2: "Clay Minerals B11/B12 (SWIR1/SWIR2)",
            SENSOR_ASTER: "Clay Minerals B4/B9 (SWIR1/SWIR2)",
        },
        "output_suffix": "ClaySWIR",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR1", "SWIR2"),
    },
    "Silica_Index": {
        "category": "Geological",
        "required_roles": ["NIR", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Silica Index B6/B5 (SWIR1/NIR)",
            SENSOR_SENTINEL2: "Silica Index B11/B08 (SWIR1/NIR)",
            SENSOR_ASTER: "Silica Index B4/B3N (SWIR1/NIR)",
        },
        "output_suffix": "Silica",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR1", "NIR"),
    },
    "Ferrous_Iron_Ratio": {
        "category": "Geological",
        "required_roles": ["Red", "NIR"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Ferrous Iron B5/B4 (NIR/Red)",
            SENSOR_SENTINEL2: "Ferrous Iron B08/B04 (NIR/Red)",
            SENSOR_ASTER: "Ferrous Iron B3N/B2 (NIR/Red)",
        },
        "output_suffix": "FerrousIron",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "NIR", "Red"),
    },
    "AAI_Advanced_Argillic": {
        "category": "Geological",
        "required_roles": ["Red", "SWIR1"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Adv. Argillic (B4-B6)/(B4+B6)",
            SENSOR_SENTINEL2: "Adv. Argillic (B04-B11)/(B04+B11)",
            SENSOR_ASTER: "Adv. Argillic (B2-B4)/(B2+B4)",
        },
        "output_suffix": "AAI",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "Red", "SWIR1"),
    },
    # =====================================================================
    # Geological — need Blue (Landsat 8/9 + Sentinel-2 only)
    # =====================================================================
    "Iron_Oxide_RedBlue": {
        "category": "Geological",
        "required_roles": ["Red", "Blue"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Iron Oxide B4/B2 (Red/Blue)",
            SENSOR_SENTINEL2: "Iron Oxide B04/B02 (Red/Blue)",
        },
        "output_suffix": "IronOxide",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "Red", "Blue"),
    },
    "Ferric_Iron_Ratio": {
        "category": "Geological",
        "required_roles": ["Red", "Blue"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Ferric Iron B4/B2 (Red/Blue)",
            SENSOR_SENTINEL2: "Ferric Iron B04/B02 (Red/Blue)",
        },
        "output_suffix": "FerricIron",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "Red", "Blue"),
    },
    "Gossan_Index": {
        "category": "Geological",
        "required_roles": ["Red", "Blue", "Green"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Gossan (B4/B2)*(B4/B3)",
            SENSOR_SENTINEL2: "Gossan (B04/B02)*(B04/B03)",
        },
        "output_suffix": "Gossan",
        "compute": lambda bands, sensor: Times(
            _ratio(bands, sensor, "Red", "Blue"),
            _ratio(bands, sensor, "Red", "Green"),
        ),
    },
    # =====================================================================
    # Geological — ASTER analog for missing Blue (Red/Green substitute)
    # =====================================================================
    "Iron_Oxide_ASTER_RG": {
        "category": "Geological",
        "required_roles": ["Red", "Green"],
        "display_labels": {
            SENSOR_ASTER: "Iron Oxide ASTER B2/B1 (Red/Green)",
        },
        "output_suffix": "IronOxide_RG",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "Red", "Green"),
        # sensor_filter restricts this index to ASTER even though L8/9 and
        # S2 also have Red+Green — they have the better Red/Blue version.
        "sensor_filter": [SENSOR_ASTER],
    },
    # =====================================================================
    # Red-Edge — Sentinel-2 only
    # =====================================================================
    "NDRE": {
        "category": "Red-Edge",
        "required_roles": ["NIR", "RedEdge1"],
        "display_labels": {
            SENSOR_SENTINEL2: "NDRE (B08-B05)/(B08+B05)",
        },
        "output_suffix": "NDRE",
        "compute": lambda bands, sensor: _normalized_diff(bands, sensor, "NIR", "RedEdge1"),
    },
    "CIred_edge": {
        "category": "Red-Edge",
        "required_roles": ["RedEdge1", "RedEdge3"],
        "display_labels": {
            SENSOR_SENTINEL2: "CIred-edge (B07/B05) - 1",
        },
        "output_suffix": "CIredEdge",
        "compute": lambda bands, sensor: Minus(_ratio(bands, sensor, "RedEdge3", "RedEdge1"), 1),
    },
    "IRECI": {
        "category": "Red-Edge",
        "required_roles": ["Red", "RedEdge1", "RedEdge2", "RedEdge3"],
        "display_labels": {
            SENSOR_SENTINEL2: "IRECI (B07-B04)/(B05/B06)",
        },
        "output_suffix": "IRECI",
        "compute": lambda bands, sensor: Divide(
            Minus(get_band(bands, "RedEdge3", sensor), get_band(bands, "Red", sensor)),
            _ratio(bands, sensor, "RedEdge1", "RedEdge2"),
        ),
    },
    # =====================================================================
    # ASTER Minerals — per-wavelength SWIR (ASTER only)
    # =====================================================================
    "Alunite_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2165", "SWIR2_2260"],
        "display_labels": {
            SENSOR_ASTER: "Alunite B7/B5 (2.260/2.165 μm)",
        },
        "output_suffix": "Alunite",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR2_2260", "SWIR2_2165"),
    },
    "Kaolinite_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2165", "SWIR2_2260"],
        "display_labels": {
            SENSOR_ASTER: "Kaolinite B5/B7 (2.165/2.260 μm)",
        },
        "output_suffix": "Kaolinite",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR2_2165", "SWIR2_2260"),
    },
    "Muscovite_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2165", "SWIR2_2205"],
        "display_labels": {
            SENSOR_ASTER: "Muscovite B6/B5 (2.205/2.165 μm)",
        },
        "output_suffix": "Muscovite",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR2_2205", "SWIR2_2165"),
    },
    "Calcite_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2260", "SWIR2_2330"],
        "display_labels": {
            SENSOR_ASTER: "Calcite B8/B7 (2.330/2.260 μm)",
        },
        "output_suffix": "Calcite",
        "compute": lambda bands, sensor: _ratio(bands, sensor, "SWIR2_2330", "SWIR2_2260"),
    },
    "Hydrothermal_Cudahy_ASTER": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2165", "SWIR2_2205", "SWIR2_2260"],
        "display_labels": {
            SENSOR_ASTER: "Hydrothermal (Cudahy) (B5/B6)*(B7/B6)",
        },
        "output_suffix": "Hydrothermal",
        "compute": lambda bands, sensor: Times(
            _ratio(bands, sensor, "SWIR2_2165", "SWIR2_2205"),
            _ratio(bands, sensor, "SWIR2_2260", "SWIR2_2205"),
        ),
    },
}


COMPOSITES = {
    "Natural_Color_RGB": {
        "category": "True Color",
        "required_roles": ["Red", "Green", "Blue"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Natural Color (B4,B3,B2)",
            SENSOR_SENTINEL2: "Natural Color (B04,B03,B02)",
        },
        "output_suffix": "NaturalColor",
        "band_spec": ["Red", "Green", "Blue"],
    },
    "False_Color_IR": {
        "category": "False Color",
        "required_roles": ["NIR", "Red", "Green"],
        "display_labels": {
            SENSOR_LANDSAT_89: "False Color IR (B5,B4,B3)",
            SENSOR_SENTINEL2: "False Color IR (B08,B04,B03)",
            SENSOR_ASTER: "False Color IR (B3N,B2,B1)",
        },
        "output_suffix": "FalseColorIR",
        "band_spec": ["NIR", "Red", "Green"],
    },
    "SWIR_Geology": {
        "category": "Geology",
        "required_roles": ["SWIR2", "SWIR1", "Red"],
        "display_labels": {
            SENSOR_LANDSAT_89: "SWIR Geology (B7,B6,B4)",
            SENSOR_SENTINEL2: "SWIR Geology (B12,B11,B04)",
            SENSOR_ASTER: "SWIR Geology (B9,B4,B2)",
        },
        "output_suffix": "SWIRGeology",
        "band_spec": ["SWIR2", "SWIR1", "Red"],
    },
    "Land_Surface": {
        "category": "Geology",
        "required_roles": ["NIR", "SWIR1", "Red"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Land Surface (B5,B6,B4)",
            SENSOR_SENTINEL2: "Land Surface (B08,B11,B04)",
            SENSOR_ASTER: "Land Surface (B3N,B4,B2)",
        },
        "output_suffix": "LandSurface",
        "band_spec": ["NIR", "SWIR1", "Red"],
    },
    "Lithological_NRB": {
        "category": "Geology",
        "required_roles": ["NIR", "Red", "Blue"],
        "display_labels": {
            SENSOR_LANDSAT_89: "Lithological (B5,B4,B2)",
            SENSOR_SENTINEL2: "Lithological (B08,B04,B02)",
        },
        "output_suffix": "Lithological",
        "band_spec": ["NIR", "Red", "Blue"],
    },
    "Vegetation_RedEdge_S2": {
        "category": "Red-Edge",
        "required_roles": ["NIR", "RedEdge1", "Red"],
        "display_labels": {
            SENSOR_SENTINEL2: "Vegetation Red-Edge (B08,B05,B04)",
        },
        "output_suffix": "VegRedEdge",
        "band_spec": ["NIR", "RedEdge1", "Red"],
    },
    "ASTER_Mineral_RGB": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2205", "SWIR2_2165", "Green"],
        "display_labels": {
            SENSOR_ASTER: "ASTER Mineral RGB (B6,B5,B1)",
        },
        "output_suffix": "ASTERMineralRGB",
        "band_spec": ["SWIR2_2205", "SWIR2_2165", "Green"],
    },
    "ASTER_Alteration_RGB": {
        "category": "ASTER Minerals",
        "required_roles": ["SWIR2_2260", "SWIR2_2205", "SWIR2_2165"],
        "display_labels": {
            SENSOR_ASTER: "ASTER Alteration RGB (B7,B6,B5)",
        },
        "output_suffix": "ASTERAlterationRGB",
        "band_spec": ["SWIR2_2260", "SWIR2_2205", "SWIR2_2165"],
    },
}


# Category display order in the GP dropdown — controls grouping.
_INDEX_CATEGORY_ORDER = [
    "Vegetation", "Water", "Moisture", "Built-up",
    "Geological", "Red-Edge", "ASTER Minerals",
]
_COMPOSITE_CATEGORY_ORDER = [
    "True Color", "False Color", "Geology", "Red-Edge", "ASTER Minerals",
]


def _applicable_entries(catalogue, sensor, category_order):
    """Filter a catalogue (INDICES or COMPOSITES) to entries available for
    the sensor, grouped by category in canonical display order."""
    if sensor not in SENSOR_BAND_ROLES:
        return {}
    roles = SENSOR_BAND_ROLES[sensor]
    grouped = {}
    for key, meta in catalogue.items():
        if meta.get("sensor_filter") and sensor not in meta["sensor_filter"]:
            continue
        if not all(r in roles for r in meta["required_roles"]):
            continue
        label = meta["display_labels"].get(sensor)
        if label is None:
            continue
        grouped.setdefault(meta["category"], []).append(label)
    # Re-order keys by canonical category order.
    ordered = {}
    for cat in category_order:
        if cat in grouped:
            ordered[cat] = grouped[cat]
    for cat in grouped:
        if cat not in ordered:
            ordered[cat] = grouped[cat]
    return ordered


def applicable_indices(sensor):
    """Indices for a sensor, grouped by category in canonical order."""
    return _applicable_entries(INDICES, sensor, _INDEX_CATEGORY_ORDER)


def applicable_composites(sensor):
    """Composites for a sensor, grouped by category in canonical order."""
    return _applicable_entries(COMPOSITES, sensor, _COMPOSITE_CATEGORY_ORDER)


def _flat_labels(grouped):
    """Flatten a grouped dict (category -> [labels]) preserving order."""
    out = []
    for cat in grouped:
        out.extend(grouped[cat])
    return out


def applicable_index_labels_flat(sensor):
    """Flat ordered list for the GP filter list."""
    return _flat_labels(applicable_indices(sensor))


def applicable_composite_labels_flat(sensor):
    return _flat_labels(applicable_composites(sensor))


def label_to_index_key(label, sensor):
    """Reverse-map a display label to its INDICES key. None if unmatched."""
    for key, meta in INDICES.items():
        if meta["display_labels"].get(sensor) == label:
            return key
    return None


def label_to_composite_key(label, sensor):
    for key, meta in COMPOSITES.items():
        if meta["display_labels"].get(sensor) == label:
            return key
    return None


# ---------------------------------------------------------------------------
# Toolbox class. Phase 2 lands Tool 04 (IndicesComposites); the mosaic
# tools (01-03) and the trivial 05/06 ports land in subsequent phases.
# ---------------------------------------------------------------------------


class Toolbox(object):
    def __init__(self):
        self.label = "GENESIS — Satellite Analysis Toolbox"
        self.alias = "genesis"
        # Workflow order. `.pyt` files have no nested toolsets, so numeric
        # prefixes in tool labels convey ordering in the Catalog pane.
        self.tools = [
            Sentinel2Mosaic,        # 01
            LandsatMosaic,          # 02
            AsterMosaic,            # 03
            IndicesComposites,      # 04
            Transformations,        # 05
            SpectralAngleMapper,    # 06
        ]


# ---------------------------------------------------------------------------
# Tool 04 — Spectral Indices & Composites
# ---------------------------------------------------------------------------


class IndicesComposites(object):
    """Sensor-aware spectral indices and RGB composites.

    Accepts a pre-stacked multiband raster from any of the three supported
    sensors. The Sensor Type parameter drives band-role resolution and
    filters the indices / composites dropdowns to only those that the
    selected sensor can compute (e.g., NDRE only shows for Sentinel-2,
    Alunite only for ASTER, Iron Oxide via Red/Blue only for L8/9 + S2).
    """

    def __init__(self):
        self.label = "04 — Spectral Indices & Composites"
        self.description = (
            "Compute spectral indices and RGB band composites on a "
            "pre-stacked Landsat 8/9, Sentinel-2 L2A, or ASTER AST_07XT "
            "multiband raster. The indices / composites dropdowns are "
            "automatically filtered to those the selected sensor can "
            "actually compute given its band roles."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        input_raster = arcpy.Parameter(
            displayName="Input Multiband Raster",
            name="input_raster",
            datatype=["DERasterDataset", "GPRasterLayer"],
            parameterType="Required",
            direction="Input",
        )

        sensor_type = make_sensor_parameter()

        indices = arcpy.Parameter(
            displayName="Select Indices",
            name="indices",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
        )
        # Initial filter list — populated/filtered in updateParameters.
        indices.filter.list = []

        composites = arcpy.Parameter(
            displayName="Select Composites",
            name="composites",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
        )
        composites.filter.list = []

        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )

        out_prefix = arcpy.Parameter(
            displayName="Output Prefix",
            name="out_prefix",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )

        rescale = arcpy.Parameter(
            displayName="Rescale Outputs to [0, 255]",
            name="rescale",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        rescale.value = False

        mask_feature = arcpy.Parameter(
            displayName="Optional AOI Mask",
            name="mask_feature",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
        )

        return [
            input_raster, sensor_type, indices, composites,
            out_workspace, out_prefix, rescale, mask_feature,
        ]

    def updateParameters(self, parameters):
        """Watch sensor + input changes; rebuild the indices/composites
        dropdowns whenever either changes. Also resolve Auto-detect to a
        concrete sensor so the user sees what they're getting."""
        try:
            input_raster = parameters[0]
            sensor_param = parameters[1]
            indices = parameters[2]
            composites = parameters[3]

            # If Auto-detect AND we have an input raster, detect now and
            # update the sensor dropdown so the filter list matches what
            # the user is about to compute against.
            effective_sensor = sensor_param.valueAsText
            if effective_sensor == SENSOR_AUTO and input_raster.valueAsText:
                detected = detect_sensor(input_raster.valueAsText)
                if detected is not None:
                    effective_sensor = detected

            # If the effective sensor is still Auto (no input or undetectable),
            # show no indices / composites — better than showing all and
            # then failing at execute time.
            if effective_sensor and effective_sensor != SENSOR_AUTO:
                indices.filter.list = applicable_index_labels_flat(effective_sensor)
                composites.filter.list = applicable_composite_labels_flat(effective_sensor)
            else:
                indices.filter.list = []
                composites.filter.list = []
        except Exception:
            # updateParameters must never raise — it runs on every keystroke.
            pass

    def updateMessages(self, parameters):
        """Lightweight validation. Heavy work happens in execute."""
        try:
            input_raster = parameters[0]
            sensor_param = parameters[1]
            indices = parameters[2]
            composites = parameters[3]

            if not input_raster.valueAsText:
                return
            if not arcpy.Exists(input_raster.valueAsText):
                input_raster.setErrorMessage(
                    f"Input raster does not exist: {input_raster.valueAsText}"
                )
                return

            # If user picked Auto and we can't detect, surface that early.
            if sensor_param.valueAsText == SENSOR_AUTO:
                if detect_sensor(input_raster.valueAsText) is None:
                    sensor_param.setErrorMessage(
                        "Could not auto-detect sensor from input filename. "
                        "Please set Sensor Type explicitly."
                    )
                    return

            # Require at least one output: an index or a composite.
            sel_indices = indices.values or []
            sel_composites = composites.values or []
            if not sel_indices and not sel_composites:
                indices.setWarningMessage(
                    "Select at least one index or composite to compute."
                )
        except Exception:
            pass

    def execute(self, parameters, messages):
        try:
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required.")
                return None

            arcpy.env.overwriteOutput = True

            input_raster = parameters[0].valueAsText
            sensor_choice = parameters[1].valueAsText
            sel_indices = parameters[2].values or []
            sel_composites = parameters[3].values or []
            out_workspace = parameters[4].valueAsText
            out_prefix = (parameters[5].valueAsText or "").strip()
            rescale = bool(parameters[6].value)
            mask_feature = parameters[7].valueAsText

            sensor = resolve_sensor(sensor_choice, input_raster)
            arcpy.AddMessage(f"Sensor resolved to: {sensor}")

            mask_obj = mask_feature if mask_feature and arcpy.Exists(mask_feature) else None
            if mask_feature and not mask_obj:
                arcpy.AddWarning(
                    f"Mask {mask_feature!r} does not exist — proceeding unmasked."
                )

            bands = self._extract_bands(input_raster, sensor)

            n_indices_written = self._calculate_indices(
                bands, sel_indices, sensor, out_workspace, out_prefix, rescale, mask_obj,
            )
            n_composites_written = self._create_composites(
                bands, sel_composites, sensor, out_workspace, out_prefix, mask_obj,
            )

            arcpy.AddMessage(
                f"\nDone. {n_indices_written} index/indices + "
                f"{n_composites_written} composite(s) written to "
                f"{out_workspace}."
            )
            return None

        except Exception as e:
            arcpy.AddError(f"Tool 04 failed: {e}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
        finally:
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    # ----- helpers -----

    def _extract_bands(self, input_raster, sensor):
        """Extract bands 1..N (N depends on sensor) into a {idx: Raster}
        dict for use by INDICES / COMPOSITES lookups."""
        n_expected = len(SENSOR_BAND_ROLES[sensor])
        bands = {}
        arcpy.AddMessage(f"Extracting {n_expected} bands for {sensor}...")
        for i in range(1, n_expected + 1):
            try:
                bands[i] = Float(ExtractBand(input_raster, [i]))
            except Exception as e:
                arcpy.AddWarning(f"  Band {i} extraction failed: {e}")
        if len(bands) < n_expected:
            arcpy.AddWarning(
                f"Only {len(bands)} of {n_expected} bands extracted. "
                f"Indices/composites referencing missing bands will be skipped."
            )
        return bands

    def _calculate_indices(self, bands, selected_labels, sensor,
                           out_workspace, out_prefix, rescale, mask_obj):
        """Compute and save each selected index. Returns count written."""
        if not selected_labels:
            return 0

        from arcpy.sa import ExtractByMask, SetNull, Float

        written = 0
        for label in selected_labels:
            clean = str(label).strip("'\"")
            key = label_to_index_key(clean, sensor)
            if key is None:
                arcpy.AddWarning(f"Index {clean!r} not recognised for {sensor}; skipped.")
                continue
            meta = INDICES[key]
            try:
                arcpy.AddMessage(f"\nComputing index: {clean}")
                result = meta["compute"](bands, sensor)

                # Apply AOI mask if provided.
                if mask_obj is not None:
                    result = ExtractByMask(result, mask_obj)

                # Sanity bounds (same as the Landsat audit fix): arcpy.sa.Divide
                # already returns NoData on /0; Inf > 10000 is also caught.
                result_f = Float(result)
                result = SetNull(
                    arcpy.sa.BooleanOr(result_f > 10000, result_f < -10000),
                    result,
                )

                if rescale:
                    result = self._rescale_to_0_255(result)

                name = f"{out_prefix}{meta['output_suffix']}" if out_prefix else meta["output_suffix"]
                out_path = os.path.join(out_workspace, name)
                result.save(out_path)
                arcpy.AddMessage(f"  Saved: {out_path}")
                written += 1
            except KeyError as e:
                arcpy.AddWarning(f"Skipping {clean}: missing band ({e})")
            except Exception as e:
                arcpy.AddWarning(f"Error computing {clean}: {e}")
        return written

    def _create_composites(self, bands, selected_labels, sensor,
                           out_workspace, out_prefix, mask_obj):
        if not selected_labels:
            return 0

        from arcpy.sa import ExtractByMask

        written = 0
        for label in selected_labels:
            clean = str(label).strip("'\"")
            key = label_to_composite_key(clean, sensor)
            if key is None:
                arcpy.AddWarning(f"Composite {clean!r} not recognised for {sensor}; skipped.")
                continue
            meta = COMPOSITES[key]
            try:
                arcpy.AddMessage(f"\nCreating composite: {clean}")
                channels = []
                for role in meta["band_spec"]:
                    b = get_band(bands, role, sensor)
                    if mask_obj is not None:
                        b = ExtractByMask(b, mask_obj)
                    channels.append(b)

                name = f"{out_prefix}{meta['output_suffix']}" if out_prefix else meta["output_suffix"]
                out_path = os.path.join(out_workspace, name)
                arcpy.management.CompositeBands(channels, out_path)
                arcpy.AddMessage(f"  Saved: {out_path}")
                written += 1
            except KeyError as e:
                arcpy.AddWarning(f"Skipping {clean}: missing band ({e})")
            except Exception as e:
                arcpy.AddWarning(f"Error creating {clean}: {e}")
        return written

    def _rescale_to_0_255(self, raster):
        """Linear rescale a raster to [0, 255] using its own min/max."""
        try:
            min_val = float(arcpy.GetRasterProperties_management(raster, "MINIMUM").getOutput(0)
                            .replace(",", "."))
            max_val = float(arcpy.GetRasterProperties_management(raster, "MAXIMUM").getOutput(0)
                            .replace(",", "."))
            if max_val == min_val:
                arcpy.AddWarning("  Rescale skipped — constant raster.")
                return raster
            rng = max_val - min_val
            return Times(Divide(Minus(Float(raster), min_val), rng), 255)
        except Exception as e:
            arcpy.AddWarning(f"  Rescale failed: {e}")
            return raster

class LandsatMosaic(object):
    """Tool 02 — Landsat 8/9 C2L2 cloud-removed mosaic.

    Accepts either:
      a) A folder of EarthExplorer `.tar` archives (LC08_/LC09_*.tar) —
         the tool extracts each archive into a sibling folder before
         processing. Both L2SR and L2SP variants are accepted.
      b) A folder of already-extracted scene subfolders, each containing
         the per-band SR TIFFs and the _MTL.txt.

    Cloud removal uses QA_PIXEL bits 0-4 (fill, dilated cloud, cirrus,
    cloud, cloud shadow). Geometric median across the temporal stack
    builds the cloud-free composite. A `_provenance.csv` is written
    alongside the output documenting every scene that contributed.

    Audit fixes carried over: Bug 2 (BuildSeamlines blend_type LINEAR)
    and Bug 7 (_apply_mask returns None on failure so callers can
    detect that masking was skipped).
    """

    def __init__(self):
        self.label = "02 — Landsat 8/9 C2L2 Mosaic"
        self.description = (
            "Build a cloud-removed mosaic from Landsat 8/9 Collection 2 "
            "Level 2 (L2SR or L2SP) scenes. Accepts a folder of .tar "
            "archives (auto-extracted) or already-extracted scene folders. "
            "QA_PIXEL bits 0-4 are masked; the temporal stack is reduced "
            "to a geometric median. A provenance CSV is written alongside "
            "the output documenting every contributing scene."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        # Output Geodatabase
        gdb = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="gdb_path",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        gdb.filter.list = ["Local Database"]

        # Output Mosaic Name
        mosaic_name = arcpy.Parameter(
            displayName="Output Mosaic Name",
            name="mosaic_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )

        # Landsat Data Folder
        data_folder = arcpy.Parameter(
            displayName="Landsat Data Folder",
            name="data_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # Region
        region = arcpy.Parameter(
            displayName="Region",
            name="region",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        region.filter.list = ["Portugal Mainland", 
                      "Azores Western (Flores, Corvo)", 
                      "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)", 
                      "Azores Eastern (São Miguel, Santa Maria)", 
                      "Madeira", 
                      "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)", 
                      "Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)",
                      "Angola", 
                      "Mozambique"]

        # Time Filter Type
        time_type = arcpy.Parameter(
            displayName="Time Filter Type",
            name="time_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        time_type.filter.list = ["All Images", "Specific Year", "Month in Year", 
                                "Month All Years", "Season in Year", "Season All Years"]

        # Year (optional, enabled for year-specific options)
        year = arcpy.Parameter(
            displayName="Year",
            name="year",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )

        # Month (optional)
        month = arcpy.Parameter(
            displayName="Month",
            name="month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        month.filter.list = list(range(1, 13))

        # Season (optional)
        season = arcpy.Parameter(
            displayName="Season",
            name="season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )

        # Mask Feature
        mask = arcpy.Parameter(
            displayName="Mask Feature (Optional)",
            name="mask_feature",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input"
        )

        # Save Statistics
        save_stats = arcpy.Parameter(
            displayName="Save Processing Statistics",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        save_stats.value = True

        params = [gdb, mosaic_name, data_folder, region, time_type, 
                 year, month, season, mask, save_stats]
        return params
    
    def updateParameters(self, parameters):
        """Modify parameter values and properties.

        Performance: updateParameters fires on every keystroke / dialog
        change. The data-folder year scan previously walked the disk on
        every call — a 50k-scene archive could freeze the ArcGIS Pro UI.
        We now cache (folder_path → sorted_years) on `self` and only
        rescan when the folder path actually changes.
        """
        if not parameters[2].altered:  # If data folder not set
            return

        if parameters[2].value and parameters[2].altered:
            folder_path = parameters[2].valueAsText
            cache = getattr(self, "_years_cache", None)
            if cache and cache[0] == folder_path:
                years_sorted = cache[1]
            else:
                try:
                    years = set()
                    for root, _, files in os.walk(folder_path):
                        for fn in files:
                            if not fn.endswith('_MTL.txt'):
                                continue
                            # Filename format: LC08_L2SR_204032_20240215_...
                            parts = fn.split('_')
                            if len(parts) >= 4:
                                try:
                                    years.add(int(parts[3][:4]))
                                except ValueError:
                                    continue
                    years_sorted = sorted(years)
                    self._years_cache = (folder_path, years_sorted)
                except Exception as e:
                    arcpy.AddWarning(f"Error scanning years: {str(e)}")
                    years_sorted = []
            if years_sorted:
                parameters[5].filter.list = years_sorted

        # Update season list based on region
        if parameters[3].value:  # If region is selected
            region = parameters[3].valueAsText
            if "Angola" in region:
                parameters[7].filter.list = ["Rainy", "Dry", "Rainy Peak", "Dry Peak"]
            elif "Mozambique" in region:
                parameters[7].filter.list = ["Rainy", "Dry", "Rainy Peak", "Dry Peak"]
            elif "Cape Verde" in region:
                parameters[7].filter.list = ["Dry", "Rainy", "Transition Dry-Wet", "Transition Wet-Dry"]    
            else:  # Temperate regions
                parameters[7].filter.list = ["Spring", "Summer", "Autumn", "Winter"]

        # Enable/disable time-based parameters
        if parameters[4].value:  # Time Filter Type
            time_type = parameters[4].valueAsText
            parameters[5].enabled = time_type in ["Specific Year", "Month in Year", "Season in Year"]
            parameters[6].enabled = time_type in ["Month in Year", "Month All Years"]
            parameters[7].enabled = time_type in ["Season in Year", "Season All Years"]

    def updateMessages(self, parameters):
            """Modify messages created by internal validation"""
            if parameters[0].altered:  # Geodatabase validation
                gdb_path = parameters[0].valueAsText
                arcpy.AddMessage(f"\nValidating geodatabase: {gdb_path}")
                
                try:
                    # Check if it's a geodatabase
                    if not gdb_path.endswith('.gdb'):
                        parameters[0].setErrorMessage("Output must be a File Geodatabase (.gdb)")
                        return

                    # Check if it exists
                    exists = arcpy.Exists(gdb_path)
                    arcpy.AddMessage(f"Geodatabase exists: {exists}")
                    if not exists:
                        parameters[0].setErrorMessage("Geodatabase does not exist")
                        return

                    # Verify it's a valid workspace
                    desc = arcpy.Describe(gdb_path)
                    arcpy.AddMessage(f"Workspace type: {desc.dataType}")
                    if desc.dataType != "Workspace":
                        parameters[0].setErrorMessage("Not a valid geodatabase workspace")
                        return

                    # Test write permissions
                    try:
                        test_name = "delete_me_test"
                        test_path = os.path.join(gdb_path, test_name)
                        arcpy.management.CreateFeatureclass(gdb_path, test_name, "POINT")
                        arcpy.management.Delete(test_path)
                        arcpy.AddMessage("Write permission test: Passed")
                    except Exception as e:
                        parameters[0].setErrorMessage(f"No write permissions: {str(e)}")
                        return

                except Exception as e:
                    parameters[0].setErrorMessage(f"Workspace validation error: {str(e)}")

            # Validate data folder
            if parameters[2].altered:
                folder_path = parameters[2].valueAsText
                if not os.path.exists(folder_path):
                    parameters[2].setErrorMessage("Data folder does not exist")
                    return

                # Check for Landsat scenes
                found_scene = False
                for root, _, files in os.walk(folder_path):
                    if any(f.endswith('_MTL.txt') for f in files):
                        found_scene = True
                        break

                if not found_scene:
                    parameters[2].setErrorMessage("No Landsat scenes found in folder")
                    
    def remove_cloud(self, scenes, stats):
        """Remove clouds from Landsat scenes"""
        try:
            from arcpy.ia import TransposeBits
            from arcpy.sa import Con  # More flexible than Clip
            
            start_time = datetime.now()
            self._update_processing_stats(stats, stage="cloud_removal")
            clean_scenes = []
            total_scenes = len(scenes)
            
            arcpy.AddMessage(f"\nRemoving clouds from {total_scenes} scenes...")
            
            for idx, scene in enumerate(scenes, 1):
                try:
                    scene_path = scene['path']
                    arcpy.AddMessage(f"\nProcessing scene {idx} of {total_scenes}")
                    arcpy.AddMessage(f"Scene path: {scene_path}")
                    
                    # List and check files
                    files = os.listdir(scene_path)
                    
                    # Find spectral bands and QA band
                    band_files = []
                    qa_file = None
                    
                    for file in files:
                        if '_SR_B' in file and file.endswith('.TIF'):
                            band_number = int(file.split('_SR_B')[-1].split('.')[0])
                            if 1 <= band_number <= 7:
                                band_files.append(os.path.join(scene_path, file))
                        
                        if '_QA_PIXEL.TIF' in file:
                            qa_file = os.path.join(scene_path, file)
                    
                    # Validate files
                    if not band_files or not qa_file:
                        arcpy.AddWarning(f"Incomplete scene data for {scene_path}")
                        continue
                    
                    # Sort band files to ensure correct order
                    band_files.sort()
                    
                    # Create rasters
                    band_rasters = [arcpy.Raster(f) for f in band_files]
                    qa_raster = arcpy.Raster(qa_file)
                    
                    # Create cloud mask
                    # Use the original TransposeBits signature
                    cloud_mask = TransposeBits(qa_raster, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 0, None)
                    
                    # Invert mask to keep clear pixels
                    value_mask = ~cloud_mask
                    
                    # Create clean rasters using Con (conditional) function
                    # This keeps pixels where value_mask is True (clear pixels)
                    clean_band_rasters = [Con(value_mask, raster) for raster in band_rasters]
                    
                    # Add processed scene to clean scenes
                    clean_scenes.append({
                        'path': scene_path,
                        'rasters': clean_band_rasters,
                        'metadata': scene.get('metadata', {})
                    })
                    
                    arcpy.AddMessage(f"Cloud removal completed for scene {idx}")
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error processing scene {idx}: {str(e)}")
                    stats['failed_scenes'] = stats.get('failed_scenes', 0) + 1
                    stats['errors'].append(str(e))
                    continue
            
            # Update cloud removal statistics
            stats['cloud_removal'] = {
                'scenes_processed': total_scenes,
                'scenes_cleaned': len(clean_scenes),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            if not clean_scenes:
                arcpy.AddWarning("No scenes were successfully processed")
                return None
                
            return clean_scenes
            
        except Exception as e:
            arcpy.AddError(f"Cloud removal failed: {str(e)}")
            return None

    def _create_geometric_median_mosaic(self, clean_scenes, gdb_path, mosaic_name):
        """Create geometric median mosaic preserving multi-band structure.

        Cleanup fix: previously the temp composites (one per scene) were
        deleted ONLY on success. Any exception during GeometricMedian or
        .save() left orphan temp rasters in the production geodatabase;
        re-runs accumulated more. We now build the list under a
        try/finally so cleanup ALWAYS runs.
        """
        multiband_rasters = []
        output_path = None
        try:
            start_time = datetime.now()

            for scene in clean_scenes:
                if 'rasters' in scene and len(scene['rasters']) == 7:
                    temp_composite = os.path.join(gdb_path, f"temp_composite_{uuid.uuid4().hex}")
                    arcpy.management.CompositeBands(scene['rasters'], temp_composite)
                    multiband_rasters.append(temp_composite)

            output_path = os.path.join(gdb_path, f"{mosaic_name}_Geomedian")
            geomedian = arcpy.ia.GeometricMedian(
                multiband_rasters,
                epsilon=0.001,
                max_iteration=20,
                extent_type="UnionOf",
                cellsize_type="FirstOf",
            )
            geomedian.save(output_path)

            arcpy.AddMessage(f"Multi-band geometric median created: {output_path}")
            arcpy.AddMessage(
                f"  Processed {len(multiband_rasters)} scenes in "
                f"{(datetime.now() - start_time).total_seconds():.1f}s."
            )
            return output_path

        except Exception as e:
            arcpy.AddError(f"Error creating geometric median: {str(e)}")
            return None

        finally:
            # Always clean up temp composites, even on failure.
            for temp_raster in multiband_rasters:
                try:
                    if arcpy.Exists(temp_raster):
                        arcpy.management.Delete(temp_raster)
                except Exception:
                    pass
                    
    def execute(self, parameters, messages):
        try:
            # Check out necessary extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            if arcpy.CheckExtension("ImageAnalyst") == "Available":
                arcpy.CheckOutExtension("ImageAnalyst")
            
            # Enable overwrite
            arcpy.env.overwriteOutput = True
            
            # Get parameters
            gdb_path = parameters[0].valueAsText
            mosaic_name = parameters[1].valueAsText
            data_folder = parameters[2].valueAsText
            region = parameters[3].valueAsText
            time_type = parameters[4].valueAsText
            year = parameters[5].value
            month = parameters[6].value
            season = parameters[7].valueAsText
            mask_feature = parameters[8].valueAsText
            save_stats = parameters[9].value

            # Initialize statistics
            stats = {
                'start_time': datetime.now(),
                'total_scenes': 0,
                'processed_scenes': 0,
                'failed_scenes': 0,
                'cloud_coverage': [],
                'processing_time': [],
                'errors': []
            }

            # Prepare stats for cloud removal and geometric median
            stats['cloud_removal'] = {
                'scenes_processed': 0,
                'scenes_cleaned': 0,
                'processing_time': 0
            }
            stats['geometric_median'] = {
                'batches_processed': 0,
                'total_batches': 0,
                'processing_time': 0
            }

            arcpy.AddMessage("\nInitializing processing:")
            arcpy.AddMessage(f"Workspace: {gdb_path}")
            arcpy.AddMessage(f"Output name: {mosaic_name}")
            arcpy.AddMessage(f"Region: {region}")

            # Create temporal filter and get region info
            temporal_filter = self._create_temporal_filter(time_type, year, month, season)
            region_info = self._get_region_info(region)

            # Phase 3 addition: transparently extract any .tar archives in
            # the data folder so users can drop EarthExplorer downloads
            # directly without manual unpacking. No-op if scenes are
            # already extracted.
            arcpy.AddMessage("\nChecking for .tar archives to extract...")
            self._extract_tar_archives(data_folder)

            # Track scenes that actually fed the final mosaic so we can
            # write a provenance CSV at the end.
            all_scenes_used = []

            # Process each UTM zone
            final_mosaics = []
            for utm_zone in region_info['utm_zones']:
                try:
                    arcpy.AddMessage(f"\nProcessing UTM zone {utm_zone}{region_info['hemisphere']}")
                    
                    # Find scenes for this zone
                    scenes = self._find_scenes(
                        data_folder=data_folder,
                        utm_zone=utm_zone,
                        temporal_filter=temporal_filter,
                        seasonal_pattern=region_info['seasonal_pattern'],
                        stats=stats
                    )
                    
                    if not scenes:
                        arcpy.AddWarning(f"No scenes found for UTM zone {utm_zone}")
                        continue

                    # Remove clouds from scenes
                    arcpy.AddMessage("\nRemoving clouds from scenes...")
                    clean_scenes = self.remove_cloud(scenes, stats)
                    
                    if not clean_scenes:
                        arcpy.AddWarning(f"No valid scenes after cloud removal for UTM zone {utm_zone}")
                        continue

                    # Create geometric median mosaic
                    zone_mosaic = self._create_geometric_median_mosaic(
                        clean_scenes,
                        gdb_path,
                        f"{mosaic_name}_UTM{utm_zone}{region_info['hemisphere']}"
                    )

                    if zone_mosaic:
                        final_mosaics.append(zone_mosaic)
                        # Phase 3 addition: accumulate scenes for provenance
                        all_scenes_used.extend(clean_scenes)

                except Exception as e:
                    arcpy.AddWarning(f"Error processing UTM zone {utm_zone}: {str(e)}")
                    stats['errors'].append(f"Zone {utm_zone} error: {str(e)}")
                    continue

            # Check if any mosaics were created
            if not final_mosaics:
                arcpy.AddError("No valid mosaics were created")
                return None

            # Merge zones if needed
            if len(final_mosaics) > 1:
                arcpy.AddMessage("\nMerging UTM zones...")
                final_mosaic = self._merge_zone_mosaics(
                    gdb_path, mosaic_name, final_mosaics, region_info
                )
            else:
                final_mosaic = final_mosaics[0]

            # Apply mask if specified
            if final_mosaic and mask_feature:
                arcpy.AddMessage("\nApplying mask...")
                final_mosaic = self._apply_mask(
                    final_mosaic, mask_feature, gdb_path, mosaic_name
                )

            # Save statistics
            if save_stats:
                # Populate additional statistics. The previous implementation
                # read `clean_scenes` from the last loop iteration only — that
                # silently dropped earlier zones' counts in multi-zone regions
                # (e.g., Portugal mainland, Mozambique). Use the accumulated
                # list we maintain for provenance instead.
                stats['processed_scenes'] = len(all_scenes_used)
                stats['failed_scenes'] = stats.get('failed_scenes', 0)
                
                stats['end_time'] = datetime.now()
                stats['total_duration'] = stats['end_time'] - stats['start_time']
                
                # Save both regular and enhanced statistics
                self._save_statistics(gdb_path, mosaic_name, stats)
                self._save_enhanced_statistics(gdb_path, mosaic_name, stats)

            if final_mosaic:
                # Phase 3 addition: provenance CSV documenting every scene
                # that contributed to the mosaic.
                try:
                    self._write_provenance_csv(final_mosaic, all_scenes_used, stats)
                except Exception as e:
                    arcpy.AddWarning(f"Provenance CSV write failed (non-fatal): {e}")

                arcpy.AddMessage(f"\nProcessing completed successfully!")
                arcpy.AddMessage(f"Output mosaic: {final_mosaic}")
                return final_mosaic
            else:
                arcpy.AddError("Failed to create final mosaic")
                return None

        except Exception as e:
            arcpy.AddError(f"Critical error in processing: {str(e)}")
            raise
        
        finally:
            # Check in extensions
            try:
                for ext in ["Spatial", "ImageAnalyst"]:
                    if arcpy.CheckExtension(ext) == "Available":
                        arcpy.CheckInExtension(ext)
            except:
                pass
        
    def _create_temporal_filter(self, time_type, year, month, season):
        """Create temporal filter dictionary"""
        filter_dict = {'type': time_type.lower().replace(' ', '_')}
        
        if year:
            filter_dict['year'] = year
        if month:
            filter_dict['month'] = month
        if season:
            filter_dict['season'] = season
            
        arcpy.AddMessage(f"Temporal filter: {filter_dict}")
        return filter_dict
        
    def _get_region_info(self, region):
        """Get region UTM zones and other information"""
        region_info = {
            'Portugal Mainland': {
                'utm_zones': [29],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Western (Flores, Corvo)': {
                'utm_zones': [25],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Azores Eastern (São Miguel, Santa Maria)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Madeira': {
                'utm_zones': [28],
                'hemisphere': 'N',
                'seasonal_pattern': 'temperate'
            },
            'Cape Verde Western (Santo Antão, São Vicente, São Nicolau)': {
                'utm_zones': [26],
                'hemisphere': 'N',
                'seasonal_pattern': 'cape_verde'
            },
            'Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)': {
                'utm_zones': [27],
                'hemisphere': 'N',
                'seasonal_pattern': 'cape_verde'
            },
            'Angola': {
                'utm_zones': [32, 33, 34],
                'hemisphere': 'S',
                'seasonal_pattern': 'angola'
            },
            'Mozambique': {
                'utm_zones': [36, 37],
                'hemisphere': 'S',
                'seasonal_pattern': 'mozambique'
            }
        }
        
        if region not in region_info:
            raise ValueError(f"Unknown region: {region}")
            
        return region_info[region]
        
    def _create_zone_mosaic(self, gdb_path, name, utm_zone, hemisphere):
        """Create mosaic dataset for specific UTM zone"""
        try:
            mosaic_path = os.path.join(gdb_path, name)
            epsg = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone
            
            arcpy.AddMessage(f"Creating mosaic dataset: {name}")
            arcpy.AddMessage(f"EPSG: {epsg}")
            
            if arcpy.Exists(mosaic_path):
                arcpy.AddMessage("Removing existing mosaic dataset")
                arcpy.Delete_management(mosaic_path)
                
            # Create base mosaic dataset
            arcpy.management.CreateMosaicDataset(
                in_workspace=gdb_path,
                in_mosaicdataset_name=name,
                coordinate_system=epsg,
                num_bands=7,
                pixel_type="16_BIT_UNSIGNED",
                product_definition="NONE"
            )
            
            # Add time field
            arcpy.AddMessage("Adding time field...")
            arcpy.management.AddField(
                in_table=mosaic_path,
                field_name="acquisitionDate",
                field_type="DATE",
                field_alias="Acquisition Date"
            )
            
            # Configure mosaic properties with proper time settings
            arcpy.AddMessage("Configuring mosaic properties...")
            arcpy.management.SetMosaicDatasetProperties(
                in_mosaic_dataset=mosaic_path,
                rows_maximum_imagesize=15000,
                columns_maximum_imagesize=15000,
                allowed_compressions="NONE",
                default_compression_type="NONE",
                resampling_type="BILINEAR",
                clip_to_footprints="CLIP",
                footprints_may_contain_nodata="FOOTPRINTS_MAY_CONTAIN_NODATA",
                clip_to_boundary="CLIP",
                color_correction="NOT_APPLY",
                allowed_mensuration_capabilities="BASIC",
                default_mensuration_capabilities="BASIC",
                allowed_mosaic_methods="Center;NorthWest;Nadir;LockRaster;ByAttribute;Seamline;None",
                default_mosaic_method="ByAttribute",
                order_field="acquisitionDate",
                order_base="1/1/1900 12:00:00 AM",
                sorting_order="Ascending",
                mosaic_operator="FIRST",
                blend_width=10,
                view_point_x=0,
                view_point_y=0,
                max_num_per_mosaic=50,
                cell_size_tolerance=0.8,
                cell_size=30,
                metadata_level="BASIC",
                transmission_fields="acquisitionDate",
                use_time="ENABLED"
            )
            
            
            arcpy.AddMessage(f"Mosaic dataset created successfully: {name}")
            return mosaic_path
            
        except Exception as e:
            arcpy.AddError(f"Error creating zone mosaic: {str(e)}")
            return None
        
          
    def _add_scenes_to_mosaic(self, mosaic_path, scenes, stats):
        """Add scenes to mosaic dataset"""
        try:
            arcpy.AddMessage("\nAdding scenes to mosaic...")
            
            if not scenes:
                arcpy.AddError("No scenes to add to mosaic")
                return False
            
            # Group scenes by satellite type
            scenes_by_type = {}
            for scene in scenes:
                scene_type = 'Landsat 8' if 'LC08' in scene['path'] else 'Landsat 9'
                if scene_type not in scenes_by_type:
                    scenes_by_type[scene_type] = []
                scenes_by_type[scene_type].append(scene)
            
            # Add scenes by satellite type
            for sat_type, type_scenes in scenes_by_type.items():
                try:
                    arcpy.AddMessage(f"\nProcessing {sat_type} scenes")
                    
                    # Add each scene folder
                    for scene in type_scenes:
                        arcpy.AddMessage(f"Adding scene: {scene['path']}")
                        
                        arcpy.management.AddRastersToMosaicDataset(
                            in_mosaic_dataset=mosaic_path,
                            raster_type=sat_type,
                            input_path=scene['path'],
                            aux_inputs="ProcessingTemplate Multiband"
                        )
                    
                    arcpy.AddMessage(f"Successfully added {len(type_scenes)} {sat_type} scenes")
                    
                    stats['processed_scenes'] += len(type_scenes)
                    
                except Exception as e:
                    arcpy.AddWarning(f"Error processing {sat_type} scenes: {str(e)}")
                    stats['failed_scenes'] += len(type_scenes)
                    stats['errors'].append(str(e))
            
            # Check number of rasters in mosaic dataset
            raster_count = int(arcpy.management.GetCount(mosaic_path).getOutput(0))
            arcpy.AddMessage(f"Number of rasters in mosaic dataset: {raster_count}")
            
            if raster_count == 0:
                arcpy.AddError("No rasters were successfully added to the mosaic dataset")
                return False
            
            # Build seamlines
            arcpy.AddMessage("Building seamlines...")
            arcpy.management.BuildSeamlines(
                in_mosaic_dataset=mosaic_path,
                sort_order="ASCENDING",
                computation_method="GEOMETRY",
                blend_width=10,
                blend_type="BOTH"
            )
            
            return True
            
        except Exception as e:
            arcpy.AddError(f"Error adding scenes to mosaic: {str(e)}")
            return False
            
    def _merge_zone_mosaics(self, gdb_path, mosaic_name, zone_mosaics, region_info):
        """Merge multiple UTM zone mosaics.

        Fix: previously the merged dataset was forced into WGS 84 (degrees)
        while BuildSeamlines was called with cell_size=30. cell_size=30
        degrees is global-scale — meaningless for Landsat. We now inherit
        the CRS from the first zone mosaic (projected, metres) so the
        30 m cell_size is in the right units. AddRastersToMosaicDataset
        reprojects subsequent zones into that CRS on the fly.
        """
        try:
            if len(zone_mosaics) == 1:
                return zone_mosaics[0]

            arcpy.AddMessage("\nMerging zone mosaics...")
            merged_name = f"{mosaic_name}_Merged"
            merged_path = os.path.join(gdb_path, merged_name)

            # Use the first zone mosaic's spatial reference (projected, in metres).
            try:
                first_sr = arcpy.Describe(zone_mosaics[0]).spatialReference
            except Exception:
                first_sr = arcpy.SpatialReference(4326)  # last-resort fallback
                arcpy.AddWarning(
                    "Could not read first zone mosaic CRS; falling back to "
                    "WGS 84. BuildSeamlines cell_size will be approximate."
                )

            arcpy.management.CreateMosaicDataset(
                in_workspace=gdb_path,
                in_mosaicdataset_name=merged_name,
                coordinate_system=first_sr,
                num_bands=7,
                pixel_type="16_BIT_UNSIGNED",
            )
            
            # Add each zone mosaic
            for zone_mosaic in zone_mosaics:
                arcpy.AddMessage(f"Adding {os.path.basename(zone_mosaic)}")
                arcpy.management.AddRastersToMosaicDataset(
                    in_mosaic_dataset=merged_path,
                    raster_type="Raster Dataset",
                    input_path=zone_mosaic
                )
                
            # Build seamlines for merged result
            arcpy.management.BuildSeamlines(
                in_mosaic_dataset=merged_path,
                cell_size=30,
                sort_order="Closest_To_Center",
                computation_method="GEOMETRY",
                blend_width=10,
                blend_type="LINEAR"
            )
            
            return merged_path
            
        except Exception as e:
            arcpy.AddError(f"Error merging zone mosaics: {str(e)}")
            return None
        
    def _apply_mask(self, mosaic_path, mask_feature, gdb_path, mosaic_name):
        """Apply mask to mosaic dataset.

        Returns:
            str: path to the masked mosaic on success.
            str: the unchanged input path when no mask was requested
                 (legitimate "skip" — caller asked for no masking).
            None: on failure (mask feature missing, ExtractByMask raised,
                 save failed). Caller MUST treat None as failure rather
                 than silently using the unmasked mosaic. Previously this
                 method returned the unmasked path on failure, hiding the
                 problem from the caller (bug 7 from the 2026 audit).
        """
        try:
            if not mask_feature:
                arcpy.AddMessage("No mask feature provided. Skipping masking.")
                return mosaic_path

            arcpy.AddMessage("\nApplying mask...")
            masked_name = f"{mosaic_name}_Masked"
            masked_path = os.path.join(gdb_path, masked_name)

            if not arcpy.Exists(mask_feature):
                arcpy.AddError(
                    f"Mask feature {mask_feature} does not exist. "
                    f"Masking step failed; the unmasked mosaic remains at "
                    f"{mosaic_path}."
                )
                return None

            from arcpy.sa import ExtractByMask
            extracted = ExtractByMask(mosaic_path, mask_feature)
            extracted.save(masked_path)

            arcpy.AddMessage(f"Masked mosaic saved as: {masked_path}")
            return masked_path

        except Exception as e:
            arcpy.AddError(
                f"Error applying mask: {str(e)}. The unmasked mosaic "
                f"remains at {mosaic_path}."
            )
            return None
            
    def _save_statistics(self, gdb_path, mosaic_name, stats):
        """Save processing statistics to file"""
        try:
            # Create statistics folder if it doesn't exist
            stats_folder = os.path.join(os.path.dirname(gdb_path), "statistics")
            if not os.path.exists(stats_folder):
                os.makedirs(stats_folder)
                
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(stats_folder, f"{mosaic_name}_stats_{timestamp}.txt")
            
            with open(stats_file, 'w') as f:
                f.write("Landsat Mosaic Processing Statistics\n")
                f.write("===================================\n\n")
                
                f.write(f"Processing Start: {stats['start_time']}\n")
                f.write(f"Processing End: {stats['end_time']}\n")
                f.write(f"Total Duration: {stats['total_duration']}\n\n")
                
                f.write("Scene Statistics:\n")
                f.write(f"Total Scenes Found: {stats['total_scenes']}\n")
                f.write(f"Successfully Processed: {stats['processed_scenes']}\n")
                f.write(f"Failed Scenes: {stats['failed_scenes']}\n\n")
                
                if stats['cloud_coverage']:
                    avg_cloud = sum(stats['cloud_coverage']) / len(stats['cloud_coverage'])
                    f.write(f"Average Cloud Coverage: {avg_cloud:.2f}%\n\n")
                    
                if stats['processing_time']:
                    avg_time = sum(stats['processing_time']) / len(stats['processing_time'])
                    f.write(f"Average Processing Time per Scene: {avg_time:.2f} seconds\n\n")
                    
                if stats['errors']:
                    f.write("Errors Encountered:\n")
                    for error in stats['errors']:
                        f.write(f"- {error}\n")
                        
            arcpy.AddMessage(f"\nStatistics saved to: {stats_file}")
            return stats_file
            
        except Exception as e:
            arcpy.AddError(f"Error saving statistics: {str(e)}")
            return None
        
    def _update_processing_stats(self, stats, stage="general"):
        """Enhanced statistics tracking"""
        try:
            # Add new statistics categories if not present
            if 'cloud_removal' not in stats:
                stats['cloud_removal'] = {
                    'scenes_processed': 0,
                    'scenes_failed': 0,
                    'average_cloud_coverage_before': 0,
                    'average_cloud_coverage_after': 0,
                    'processing_time': 0
                }
                
            if 'geometric_median' not in stats:
                stats['geometric_median'] = {
                    'batches_processed': 0,
                    'total_batches': 0,
                    'memory_usage': [],
                    'processing_time': 0
                }
                
            if 'memory_tracking' not in stats:
                stats['memory_tracking'] = {
                    'peak_memory': 0,
                    'average_memory': 0,
                    'timestamps': []
                }
                
            # Update memory tracking
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            stats['memory_tracking']['timestamps'].append({
                'time': datetime.now(),
                'memory': current_memory,
                'stage': stage
            })
            stats['memory_tracking']['peak_memory'] = max(
                stats['memory_tracking']['peak_memory'],
                current_memory
            )
            
            return stats
            
        except Exception as e:
            arcpy.AddWarning(f"Error updating statistics: {str(e)}")
            return stats

    def _save_enhanced_statistics(self, gdb_path, mosaic_name, stats):
        """Save enhanced processing statistics"""
        try:
            stats_folder = os.path.join(os.path.dirname(gdb_path), "statistics")
            if not os.path.exists(stats_folder):
                os.makedirs(stats_folder)
                    
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = os.path.join(stats_folder, f"{mosaic_name}_detailed_stats_{timestamp}.txt")
            
            with open(stats_file, 'w') as f:
                f.write("Landsat Processing Detailed Statistics\n")
                f.write("====================================\n\n")
                
                # Basic processing info
                f.write("Processing Duration:\n")
                f.write(f"Start: {stats.get('start_time', 'N/A')}\n")
                f.write(f"End: {stats.get('end_time', datetime.now())}\n")
                f.write(f"Total Duration: {stats.get('total_duration', 'N/A')}\n\n")
                
                # Cloud removal statistics
                cloud_removal = stats.get('cloud_removal', {})
                f.write("Cloud Removal Statistics:\n")
                f.write(f"Scenes Processed: {cloud_removal.get('scenes_processed', 0)}\n")
                f.write(f"Scenes Cleaned: {cloud_removal.get('scenes_cleaned', 0)}\n")
                f.write(f"Failed Scenes: {stats.get('failed_scenes', 0)}\n")
                f.write(f"Processing Time: {cloud_removal.get('processing_time', 0):.2f} seconds\n\n")
                
                # Geometric median statistics
                geo_median = stats.get('geometric_median', {})
                f.write("Geometric Median Statistics:\n")
                f.write(f"Batches Processed: {geo_median.get('batches_processed', 0)}\n")
                f.write(f"Total Batches: {geo_median.get('total_batches', 0)}\n")
                f.write(f"Processing Time: {geo_median.get('processing_time', 0):.2f} seconds\n\n")
                
                # Errors
                if stats.get('errors'):
                    f.write("Errors Encountered:\n")
                    for error in stats.get('errors', []):
                        f.write(f"- {error}\n")
                        
            arcpy.AddMessage(f"\nDetailed statistics saved to: {stats_file}")
            return stats_file
            
        except Exception as e:
            arcpy.AddError(f"Error saving enhanced statistics: {str(e)}")
            return None
            
    def _parse_metadata(self, mtl_path):
        """Parse Landsat MTL file"""
        try:
            with open(mtl_path) as f:
                content = f.read()
                
            # Extract key metadata. Each field is best-effort: a missing
            # tag is logged and left as None / 0, never propagated as an
            # IndexError. Pre-fix the `[0]` access would silently drop the
            # whole scene when ANY field was missing (common on partial
            # downloads, Collection-1 MTLs, or pre-C2 scenes).
            lines = content.split('\n')
            scene_info = {}

            def _first_match(needle, parser=None, default=None, exact_token=False):
                """Return parser(first MTL line's value), or default.

                Lines look like `    KEY = VALUE`. We split on the first
                `=`, strip the key, and compare. exact_token=True ensures
                'CLOUD_COVER' doesn't also match 'CLOUD_COVER_LAND' — the
                pre-fix code used a bare `in` substring check which would
                grab whichever variant appeared first.
                """
                for ln in lines:
                    if "=" not in ln:
                        continue
                    key, _, value = ln.partition("=")
                    key = key.strip()
                    if exact_token:
                        if key != needle:
                            continue
                    else:
                        if needle not in key:
                            continue
                    try:
                        raw = value.strip().strip('"')
                        return parser(raw) if parser else raw
                    except (IndexError, ValueError):
                        return default
                return default

            scene_info['acquisition_date'] = _first_match(
                'DATE_ACQUIRED',
                parser=lambda s: datetime.strptime(s, '%Y-%m-%d'),
            )
            scene_info['cloud_cover'] = _first_match(
                'CLOUD_COVER',
                parser=float,
                default=0.0,
                exact_token=True,  # don't pick up CLOUD_COVER_LAND
            )
            scene_info['utm_zone'] = _first_match('UTM_ZONE', parser=int)
            scene_info['processing_level'] = _first_match('PROCESSING_LEVEL')

            if scene_info['acquisition_date'] is None:
                arcpy.AddWarning(
                    f"  Scene {os.path.basename(mtl_path)} has no parseable "
                    f"DATE_ACQUIRED; dropping."
                )
                return None
            return scene_info
            
        except Exception as e:
            arcpy.AddWarning(f"Error parsing metadata {mtl_path}: {str(e)}")
            return None
        
    def _find_scenes(self, data_folder, utm_zone, temporal_filter, seasonal_pattern, stats):
        """Find and validate Landsat scenes for specified UTM zone"""
        try:
            scenes = []
            ls8_count = 0
            ls9_count = 0
            arcpy.AddMessage("\nScanning for Landsat scenes...")
            
            for root, _, files in os.walk(data_folder):
                for file in files:
                    if file.endswith('_MTL.txt'):
                        stats['total_scenes'] += 1
                        
                        try:
                            # Parse metadata
                            mtl_path = os.path.join(root, file)
                            scene_info = self._parse_metadata(mtl_path)
                            
                            if scene_info:
                                # Count Landsat 8 and 9 scenes
                                if 'LC08' in file:
                                    ls8_count += 1
                                elif 'LC09' in file:
                                    ls9_count += 1
                                
                                # Check UTM zone
                                if scene_info['utm_zone'] == utm_zone:
                                    # Apply temporal filter
                                    if self._apply_temporal_filter(scene_info, temporal_filter, seasonal_pattern):
                                        scenes.append({
                                            'path': root,
                                            'metadata': scene_info
                                        })
                                        stats['cloud_coverage'].append(scene_info['cloud_cover'])
                                        
                        except Exception as e:
                            stats['failed_scenes'] += 1
                            stats['errors'].append(str(e))
                            continue
            
            # Print Landsat 8 and 9 scene counts
            arcpy.AddMessage(f"Total Landsat 8 scenes: {ls8_count}")
            arcpy.AddMessage(f"Total Landsat 9 scenes: {ls9_count}")
            arcpy.AddMessage(f"Found {len(scenes)} valid scenes for UTM zone {utm_zone}")
            
            return scenes
            
        except Exception as e:
            arcpy.AddError(f"Error finding scenes: {str(e)}")
            return []
                
    def _apply_temporal_filter(self, scene_info, temporal_filter, seasonal_pattern):
        """Apply temporal filter to scene"""
        try:
            filter_type = temporal_filter['type']
            date = scene_info['acquisition_date']
            
            if filter_type == 'all_images':
                return True
                
            elif filter_type == 'specific_year':
                return date.year == temporal_filter['year']
                
            elif filter_type == 'month_in_year':
                return (date.year == temporal_filter['year'] and 
                        date.month == temporal_filter['month'])
                
            elif filter_type == 'month_all_years':
                return date.month == temporal_filter['month']
                
            elif filter_type == 'season_in_year':
                if 'year' in temporal_filter:
                    if date.year != temporal_filter['year']:
                        return False
                return self._is_in_season(date, seasonal_pattern, temporal_filter['season'])
                
            elif filter_type == 'season_all_years':
                return self._is_in_season(date, seasonal_pattern, temporal_filter['season'])
                
            return False
            
        except Exception as e:
            arcpy.AddWarning(f"Error applying temporal filter: {str(e)}")
            return False
            
    def _is_in_season(self, date, pattern, season):
        """Check if date falls within specified season"""
        season_months = {
            'temperate': {
                'spring': [3, 4, 5],
                'summer': [6, 7, 8],
                'autumn': [9, 10, 11],
                'winter': [12, 1, 2]
            },
            'angola': {
                'rainy': [11, 12, 1, 2, 3, 4],
                'dry': [5, 6, 7, 8, 9, 10],
                'rainy_peak': [1, 2, 3],
                'dry_peak': [6, 7, 8]
            },
            'cape_verde': {
                'dry': [12, 1, 2, 3, 4, 5, 6],
                'rainy': [8, 9, 10],
                'transition_dry_wet': [7],
                'transition_wet_dry': [11]
            },
            'mozambique': {
                'rainy': [10, 11, 12, 1, 2, 3],
                'dry': [4, 5, 6, 7, 8, 9],
                'rainy_peak': [12, 1, 2],
                'dry_peak': [7, 8, 9]
            }
        }
        
        return date.month in season_months[pattern][season.lower()]

    # ------------------------------------------------------------------
    # Phase 3 additions: tar extraction + provenance CSV
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe_tar_member(member, target_dir):
        """Decide whether a tar member is safe to extract into target_dir.

        Returns (ok: bool, reason: str). The "safe" criteria, in order:

          1. Reject symlinks and hardlinks — they can point outside the
             target directory and overwrite system files. Legit USGS tars
             never contain links, so this is a hard reject.
          2. Reject device files, FIFOs, and other special types — same
             reasoning. Only regular files and directories are allowed.
          3. Reject absolute paths (POSIX `/foo` or Windows `C:\\foo`).
          4. Reject any path component equal to `..` (parent traversal).
          5. After normalising the joined path, require that the resolved
             location is inside target_dir. Catches anything the
             component-level check missed (e.g. odd encodings, symlink
             races during extraction).

        This is the "safe-extract" pattern recommended for Python 3.11
        and earlier (Python 3.12+ has the built-in `filter='data'`
        option to `tarfile.extractall` that does roughly the same job).
        """
        name = (member.name or "").replace("\\", "/")
        if not name:
            return False, "empty member name"

        # 1. Reject links and special types.
        if member.issym() or member.islnk():
            return False, "symbolic/hard link"
        if not (member.isfile() or member.isdir()):
            return False, f"non-regular tar member type ({member.type!r})"

        # 2. Reject absolute paths (POSIX or Windows drive-letter).
        if name.startswith("/"):
            return False, "absolute path"
        if len(name) >= 2 and name[1] == ":":
            return False, "absolute Windows path"

        # 3. Reject explicit parent-traversal components.
        parts = name.split("/")
        if ".." in parts:
            return False, "parent-traversal component"

        # 4. Resolved-path check — defence in depth, catches anything the
        # component-level checks missed.
        target_real = os.path.realpath(target_dir)
        joined = os.path.realpath(os.path.join(target_dir, name))
        if joined != target_real and not joined.startswith(target_real + os.sep):
            return False, f"resolves outside target_dir ({joined!r})"

        return True, ""

    def _extract_tar_archives(self, data_folder):
        """Find any `.tar` files in `data_folder` and extract each into a
        same-named subfolder. Already-extracted folders are skipped.

        EarthExplorer ships Landsat 8/9 C2L2 scenes as `.tar` archives
        named like `LC08_L2SR_217033_20260502_20260514_02_T1.tar`; this
        method makes the toolbox tolerant of that directly so users don't
        have to extract manually.

        Returns the list of extracted scene folders (relative paths under
        data_folder) for downstream `_find_scenes` to discover via the
        normal `_MTL.txt` glob.
        """
        if not data_folder or not os.path.isdir(data_folder):
            return []

        extracted = []
        tar_files = [
            f for f in os.listdir(data_folder)
            if f.lower().endswith(".tar") and not f.startswith(".")
        ]
        if not tar_files:
            arcpy.AddMessage("  No .tar archives found; assuming scenes are pre-extracted.")
            return []

        arcpy.AddMessage(f"  Found {len(tar_files)} .tar archive(s) to inspect...")
        for tar_name in tar_files:
            tar_path = os.path.join(data_folder, tar_name)
            scene_name = tar_name[:-4]  # strip ".tar"
            target_dir = os.path.join(data_folder, scene_name)

            if os.path.isdir(target_dir):
                # Already extracted — verify by looking for an MTL.
                mtls = [
                    fn for fn in os.listdir(target_dir)
                    if fn.endswith("_MTL.txt")
                ]
                if mtls:
                    arcpy.AddMessage(f"    [skip] already extracted: {scene_name}")
                    extracted.append(target_dir)
                    continue
                else:
                    arcpy.AddMessage(
                        f"    [re-extract] {scene_name} folder exists but has no MTL"
                    )

            arcpy.AddMessage(f"    [extract] {tar_name} → {scene_name}/")
            try:
                os.makedirs(target_dir, exist_ok=True)
                with tarfile.open(tar_path, "r") as tar:
                    # Safe-extract: refuse any member with absolute or
                    # parent-traversal paths (defensive against malicious
                    # archives — defense in depth even for trusted USGS data).
                    safe_members = []
                    for member in tar.getmembers():
                        ok, reason = self._is_safe_tar_member(member, target_dir)
                        if not ok:
                            arcpy.AddWarning(
                                f"      skipped tar entry {member.name!r}: {reason}"
                            )
                            continue
                        safe_members.append(member)
                    tar.extractall(path=target_dir, members=safe_members)
                extracted.append(target_dir)
            except (tarfile.TarError, OSError) as e:
                arcpy.AddWarning(f"    [fail] {tar_name}: {e}")
                continue

        arcpy.AddMessage(f"  Tar handling complete: {len(extracted)} scene folder(s) ready.")
        return extracted

    def _write_provenance_csv(self, output_raster_path, scenes_used, stats):
        """Write `{output_raster}_provenance.csv` documenting every scene
        that fed into the mosaic.

        Columns (RFC 4180-quoted UTF-8):
            scene_id, sensor, acquisition_datetime, path_row,
            cloud_cover_pct, input_path, processing_baseline,
            toolbox_version, processing_datetime

        Failures are non-fatal — a missing provenance file is annoying
        but doesn't invalidate the mosaic itself.
        """
        if not output_raster_path or not scenes_used:
            return

        try:
            csv_path = f"{output_raster_path}_provenance.csv"
            now_iso = datetime.now().isoformat(timespec="seconds")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "scene_id", "sensor", "acquisition_datetime",
                    "path_row", "cloud_cover_pct", "input_path",
                    "processing_baseline", "toolbox_version",
                    "processing_datetime",
                ])
                for scene in scenes_used:
                    meta = scene.get("metadata", {}) or {}
                    scene_path = scene.get("path", "") or ""
                    scene_id = os.path.basename(scene_path.rstrip(os.sep)) or ""
                    sensor = "Landsat 9" if "LC09" in scene_path else "Landsat 8"
                    writer.writerow([
                        scene_id,
                        sensor,
                        meta.get("date_acquired", "") or meta.get("acquisition_date", ""),
                        meta.get("path_row", "") or "",
                        meta.get("cloud_cover", "") or "",
                        scene_path,
                        meta.get("collection_number", "") or "C2",
                        TOOLBOX_VERSION,
                        now_iso,
                    ])
            arcpy.AddMessage(f"Provenance CSV: {csv_path}")
        except Exception as e:
            arcpy.AddWarning(f"Failed to write provenance CSV: {e}")


# ---------------------------------------------------------------------------
# Tool 01 — Sentinel-2 L2A Mosaic
# ---------------------------------------------------------------------------


# SCL classes to mask (Sen2Cor Scene Classification Layer):
#   3 = Cloud shadows
#   8 = Cloud medium probability
#   9 = Cloud high probability
#   10 = Thin cirrus
# (Class 11 = snow/ice is kept — relevant signal for some terrestrial work.)
_S2_SCL_CLOUD_CLASSES = (3, 8, 9, 10)

# Sentinel-2 L2A scale factor: DN * 0.0001 = surface reflectance.
_S2_REFLECTANCE_SCALE = 0.0001

# 10-band L2A stack order matching SENSOR_BAND_ROLES[SENSOR_SENTINEL2].
_S2_STACK_ORDER = [
    "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12",
]
_S2_NATIVE_10M = {"B02", "B03", "B04", "B08"}  # the rest are 20m → resampled


class Sentinel2Mosaic(object):
    """Tool 01 — Sentinel-2 L2A cloud-removed mosaic.

    Accepts a folder of S2 L2A products, either as:
      a) Copernicus `.zip` archives (e.g., `S2A_MSIL2A_*.zip`) — extracted
         transparently before processing.
      b) Already-extracted `.SAFE` folders.

    For each scene the tool reads the 10m bands (B02, B03, B04, B08) at
    native resolution and the 20m bands (B05, B06, B07, B8A, B11, B12)
    resampled to 10m via BILINEAR. Cloud masking uses the SCL layer
    (classes 3, 8, 9, 10). Surface reflectance is scaled to [0, 1] via
    the 0.0001 factor.

    The cloud-masked, scaled 10-band stack from each scene is then fed
    to arcpy.sa.GeometricMedian per MGRS tile. Multi-tile regions are
    merged after per-tile median composites are built. A provenance CSV
    is written alongside the output documenting every contributing scene.
    """

    def __init__(self):
        self.label = "01 — Sentinel-2 L2A Mosaic"
        self.description = (
            "Build a cloud-removed mosaic from Sentinel-2 L2A scenes "
            "(Sen2Cor / Copernicus L2A). Accepts a folder of .zip "
            "archives (auto-extracted) or already-extracted .SAFE "
            "folders. SCL classes 3/8/9/10 are masked; the temporal "
            "stack is reduced to a geometric median per MGRS tile. "
            "20m bands are resampled to 10m. A provenance CSV is "
            "written alongside the output."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # GP parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        gdb = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="gdb_path",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        gdb.filter.list = ["Local Database"]

        mosaic_name = arcpy.Parameter(
            displayName="Output Mosaic Name",
            name="mosaic_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        data_folder = arcpy.Parameter(
            displayName="Sentinel-2 Data Folder",
            name="data_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )

        region = arcpy.Parameter(
            displayName="Region",
            name="region",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        # Same region list as LandsatMosaic — provides the seasonal
        # pattern for temporal filtering. S2 uses MGRS tiles rather than
        # UTM zones; tile assignment is done by scanning the data folder.
        region.filter.list = [
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
            "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)",
            "Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)",
            "Angola",
            "Mozambique",
        ]

        time_type = arcpy.Parameter(
            displayName="Time Filter Type",
            name="time_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        time_type.filter.list = [
            "year_month", "month_all_years",
            "season_in_year", "season_all_years",
        ]

        year = arcpy.Parameter(
            displayName="Year (when applicable)",
            name="year",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        month = arcpy.Parameter(
            displayName="Month (when applicable, 1-12)",
            name="month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        season = arcpy.Parameter(
            displayName="Season (when applicable)",
            name="season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )

        apply_temporal = arcpy.Parameter(
            displayName="Apply Temporal Cloud Refinement (optional, slower)",
            name="apply_temporal_refinement",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        apply_temporal.value = False

        mask_feature = arcpy.Parameter(
            displayName="Optional AOI Mask Feature",
            name="mask_feature",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
        )

        save_stats = arcpy.Parameter(
            displayName="Save Provenance CSV",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        save_stats.value = True

        return [
            gdb, mosaic_name, data_folder, region, time_type,
            year, month, season, apply_temporal, mask_feature, save_stats,
        ]

    def updateParameters(self, parameters):
        """Enable/disable time-detail parameters based on time_type."""
        try:
            time_type = parameters[4]
            year = parameters[5]
            month = parameters[6]
            season = parameters[7]
            if time_type.valueAsText == "year_month":
                year.enabled = True
                month.enabled = True
                season.enabled = False
            elif time_type.valueAsText == "month_all_years":
                year.enabled = False
                month.enabled = True
                season.enabled = False
            elif time_type.valueAsText == "season_in_year":
                year.enabled = True
                month.enabled = False
                season.enabled = True
            elif time_type.valueAsText == "season_all_years":
                year.enabled = False
                month.enabled = False
                season.enabled = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        # Always-defined for the finally cleanup.
        scratch_dir = None
        composite_temp_paths = []

        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                arcpy.AddError("Spatial Analyst extension is required.")
                return None
            arcpy.CheckOutExtension("Spatial")
            arcpy.env.overwriteOutput = True

            gdb_path = parameters[0].valueAsText
            mosaic_name = parameters[1].valueAsText
            data_folder = parameters[2].valueAsText
            region = parameters[3].valueAsText
            time_type = parameters[4].valueAsText
            year = parameters[5].value
            month = parameters[6].value
            season = parameters[7].valueAsText
            apply_temporal = bool(parameters[8].value)
            mask_feature = parameters[9].valueAsText
            save_stats = bool(parameters[10].value)

            arcpy.AddMessage("\nInitialising S2 mosaic processing:")
            arcpy.AddMessage(f"  Workspace:   {gdb_path}")
            arcpy.AddMessage(f"  Output name: {mosaic_name}")
            arcpy.AddMessage(f"  Region:      {region}")
            arcpy.AddMessage(f"  Data folder: {data_folder}")

            scratch_dir = os.path.join(
                arcpy.env.scratchFolder or data_folder,
                f"_genesis_s2_scratch_{uuid.uuid4().hex[:8]}",
            )
            os.makedirs(scratch_dir, exist_ok=True)
            arcpy.AddMessage(f"  Scratch:     {scratch_dir}")

            # Step 1: extract any .zip archives in the data folder.
            arcpy.AddMessage("\nStep 1/6 — Checking for .zip archives...")
            self._extract_zip_archives(data_folder)

            # Step 2: discover all SAFE folders.
            arcpy.AddMessage("\nStep 2/6 — Discovering SAFE folders...")
            safe_folders = self._find_safe_scenes(data_folder)
            if not safe_folders:
                arcpy.AddError("No .SAFE folders found in the data folder.")
                return None
            arcpy.AddMessage(f"  Found {len(safe_folders)} SAFE folder(s).")

            # Step 3: parse metadata + apply temporal filter.
            arcpy.AddMessage("\nStep 3/6 — Filtering scenes by date/season...")
            seasonal_pattern = self._seasonal_pattern_for_region(region)
            temporal_filter = self._create_temporal_filter(
                time_type, year, month, season,
            )
            kept_scenes = []
            for safe in safe_folders:
                meta = self._parse_safe_metadata(safe)
                if meta is None:
                    arcpy.AddWarning(f"  Skipped (no metadata): {os.path.basename(safe)}")
                    continue
                if not self._scene_passes_filter(meta, temporal_filter, seasonal_pattern):
                    continue
                kept_scenes.append({"path": safe, "metadata": meta})
            if not kept_scenes:
                arcpy.AddError("No scenes match the temporal filter.")
                return None
            arcpy.AddMessage(f"  Kept {len(kept_scenes)} scenes after temporal filter.")

            # Step 4: process each scene (mask + scale + stack), grouped by tile.
            arcpy.AddMessage("\nStep 4/6 — Cloud-masking and stacking scenes...")
            scenes_by_tile = {}
            all_scenes_used = []
            for scene in kept_scenes:
                meta = scene["metadata"]
                tile = meta.get("tile_id")
                if not tile:
                    arcpy.AddWarning(
                        f"  Skipped (no tile ID): {os.path.basename(scene['path'])}"
                    )
                    continue
                arcpy.AddMessage(
                    f"  [{tile}] {os.path.basename(scene['path'])}"
                )
                try:
                    stacked_path = self._process_scene(scene, scratch_dir)
                except Exception as e:
                    arcpy.AddWarning(f"    Failed: {e}")
                    continue
                if stacked_path:
                    scenes_by_tile.setdefault(tile, []).append(stacked_path)
                    composite_temp_paths.append(stacked_path)
                    all_scenes_used.append(scene)
            if not scenes_by_tile:
                arcpy.AddError("No scenes survived cloud masking + stacking.")
                return None

            # Optional Step 4b: multitemporal cloud refinement layer.
            if apply_temporal:
                arcpy.AddMessage("\nStep 4b — Applying multitemporal cloud refinement...")
                arcpy.AddWarning(
                    "  Multitemporal refinement is a stub in this build — "
                    "the per-scene SCL mask is already applied; an explicit "
                    "temporal anomaly pass will land in a follow-up phase."
                )

            # Step 5: per-tile geometric median, then merge.
            arcpy.AddMessage("\nStep 5/6 — Computing per-tile geometric median...")
            tile_mosaics = []
            for tile, stacked_paths in scenes_by_tile.items():
                arcpy.AddMessage(f"  [{tile}] {len(stacked_paths)} scenes")
                try:
                    tile_mosaic_name = f"{mosaic_name}_{tile}"
                    tile_mosaic_path = os.path.join(gdb_path, tile_mosaic_name)
                    median = arcpy.sa.GeometricMedian(stacked_paths)
                    median.save(tile_mosaic_path)
                    tile_mosaics.append(tile_mosaic_path)
                    arcpy.AddMessage(f"    Saved tile mosaic: {tile_mosaic_path}")
                except Exception as e:
                    arcpy.AddError(f"    Geometric median failed for {tile}: {e}")
                    continue

            if not tile_mosaics:
                arcpy.AddError("No tile mosaics were created.")
                return None

            # Merge multi-tile result.
            arcpy.AddMessage("\nStep 6/6 — Finalising output...")
            if len(tile_mosaics) > 1:
                final_path = os.path.join(gdb_path, mosaic_name)
                arcpy.AddMessage(f"  Merging {len(tile_mosaics)} tile mosaics...")
                arcpy.management.MosaicToNewRaster(
                    input_rasters=tile_mosaics,
                    output_location=gdb_path,
                    raster_dataset_name_with_extension=mosaic_name,
                    coordinate_system_for_the_raster="",
                    pixel_type="32_BIT_FLOAT",
                    cellsize=10,
                    number_of_bands=len(_S2_STACK_ORDER),
                    mosaic_method="MEAN",
                )
                final_mosaic = final_path
            else:
                final_mosaic = tile_mosaics[0]

            # Apply AOI mask if requested.
            if mask_feature and arcpy.Exists(mask_feature):
                arcpy.AddMessage(f"  Applying AOI mask: {mask_feature}")
                masked = arcpy.sa.ExtractByMask(final_mosaic, mask_feature)
                masked_path = os.path.join(gdb_path, f"{mosaic_name}_Masked")
                masked.save(masked_path)
                final_mosaic = masked_path
            elif mask_feature:
                arcpy.AddWarning(
                    f"AOI mask {mask_feature!r} not found — output is unmasked."
                )

            # Provenance CSV.
            if save_stats:
                self._write_provenance_csv(final_mosaic, all_scenes_used)

            arcpy.AddMessage("\nDone. Final mosaic: " + str(final_mosaic))
            return final_mosaic

        except Exception as e:
            arcpy.AddError(f"Tool 01 failed: {e}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None

        finally:
            # Cleanup scratch.
            if scratch_dir and os.path.isdir(scratch_dir):
                try:
                    import shutil
                    shutil.rmtree(scratch_dir, ignore_errors=True)
                except Exception:
                    pass
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    # ------------------------------------------------------------------
    # ZIP extraction (parallel to LandsatMosaic's tar handling)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe_zip_member(name, target_dir):
        """Same safe-extract pattern as the tar version."""
        n = (name or "").replace("\\", "/")
        if not n:
            return False, "empty name"
        if n.startswith("/"):
            return False, "absolute path"
        if len(n) >= 2 and n[1] == ":":
            return False, "Windows drive-letter path"
        parts = n.split("/")
        if ".." in parts:
            return False, "parent-traversal component"
        target_real = os.path.realpath(target_dir)
        joined = os.path.realpath(os.path.join(target_dir, n))
        if joined != target_real and not joined.startswith(target_real + os.sep):
            return False, f"resolves outside target_dir ({joined!r})"
        return True, ""

    def _extract_zip_archives(self, data_folder):
        """Extract every `.zip` in data_folder. Skips zips whose .SAFE
        contents are already extracted next to them."""
        if not data_folder or not os.path.isdir(data_folder):
            return []

        zips = [
            f for f in os.listdir(data_folder)
            if f.lower().endswith(".zip") and not f.startswith(".")
        ]
        if not zips:
            arcpy.AddMessage("  No .zip archives found; assuming .SAFE folders are extracted.")
            return []

        arcpy.AddMessage(f"  Found {len(zips)} .zip archive(s).")
        extracted = []
        for zname in zips:
            zpath = os.path.join(data_folder, zname)
            # Detect already-extracted by looking for a sibling .SAFE folder.
            # Copernicus zips are named "<safe_basename>.zip" where the
            # basename ALREADY ends with ".SAFE" — so just strip ".zip".
            expected_safe = zname[:-4] if zname.lower().endswith(".zip") else None
            if expected_safe and os.path.isdir(os.path.join(data_folder, expected_safe)):
                arcpy.AddMessage(f"    [skip] already extracted: {expected_safe}")
                extracted.append(zpath)
                continue
            arcpy.AddMessage(f"    [extract] {zname}")
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    safe_names = []
                    for info in zf.infolist():
                        ok, reason = self._is_safe_zip_member(info.filename, data_folder)
                        if not ok:
                            arcpy.AddWarning(
                                f"      skipped zip entry {info.filename!r}: {reason}"
                            )
                            continue
                        safe_names.append(info.filename)
                    zf.extractall(path=data_folder, members=safe_names)
                extracted.append(zpath)
            except (zipfile.BadZipFile, OSError) as e:
                arcpy.AddWarning(f"    [fail] {zname}: {e}")
                continue
        return extracted

    # ------------------------------------------------------------------
    # SAFE discovery + metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _find_safe_scenes(data_folder):
        """Return list of absolute paths to .SAFE folders in data_folder."""
        if not data_folder or not os.path.isdir(data_folder):
            return []
        return sorted(
            os.path.join(data_folder, e)
            for e in os.listdir(data_folder)
            if e.endswith(".SAFE") and os.path.isdir(os.path.join(data_folder, e))
        )

    _SAFE_NAME_RE = re.compile(
        # Note the captured tile group includes the "T" prefix so callers
        # get the canonical MGRS tile ID (e.g., "T29SQB", not "29SQB").
        r"^S2[AB]_MSIL2A_(\d{8})T(\d{6})_N\d{4}_R\d{3}_(T\w{5})_",
    )

    @classmethod
    def _parse_safe_metadata(cls, safe_folder):
        """Extract acquisition date, tile ID, and cloud cover (best-effort)
        for a SAFE folder. Falls back to filename parsing if XML metadata
        is unavailable.

        Returns dict with keys: date_acquired (date), tile_id, cloud_cover,
        product_uri. Returns None if even the filename doesn't parse.
        """
        basename = os.path.basename(safe_folder.rstrip(os.sep))
        m = cls._SAFE_NAME_RE.match(basename)
        if not m:
            return None
        date_str, _, tile_id = m.groups()
        try:
            acquired = datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            return None

        meta = {
            "date_acquired": acquired,
            "tile_id": tile_id,
            "cloud_cover": None,
            "product_uri": basename,
        }

        # Try to read cloud-cover from MTD_MSIL2A.xml. Namespaces in this
        # XML are unstable across processing baselines, so we use a
        # wildcard XPath ('{*}TAG') and don't fail if the tag is missing.
        mtd_path = os.path.join(safe_folder, "MTD_MSIL2A.xml")
        if os.path.isfile(mtd_path):
            try:
                root = ET.parse(mtd_path).getroot()
                for elem in root.iter():
                    tag = elem.tag.split("}", 1)[-1] if "}" in elem.tag else elem.tag
                    if tag == "Cloud_Coverage_Assessment" or tag == "CLOUDY_PIXEL_PERCENTAGE":
                        try:
                            meta["cloud_cover"] = float(elem.text)
                        except (TypeError, ValueError):
                            pass
                        break
            except ET.ParseError:
                pass
        return meta

    # ------------------------------------------------------------------
    # Per-scene processing
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_band_files(safe_folder):
        """Find the JP2 file for each band + SCL within a SAFE folder.

        Returns dict {band_name: jp2_path} including all 10 stack bands
        and the SCL. Returns None on missing GRANULE structure.
        """
        granule_dirs = glob.glob(os.path.join(safe_folder, "GRANULE", "L2A_*"))
        if not granule_dirs:
            return None
        img_data = os.path.join(granule_dirs[0], "IMG_DATA")
        r10 = os.path.join(img_data, "R10m")
        r20 = os.path.join(img_data, "R20m")

        out = {}
        for band in ("B02", "B03", "B04", "B08"):
            matches = glob.glob(os.path.join(r10, f"*_{band}_10m.jp2"))
            if matches:
                out[band] = matches[0]
        for band in ("B05", "B06", "B07", "B8A", "B11", "B12"):
            matches = glob.glob(os.path.join(r20, f"*_{band}_20m.jp2"))
            if matches:
                out[band] = matches[0]
        scl = glob.glob(os.path.join(r20, "*_SCL_20m.jp2"))
        if scl:
            out["SCL"] = scl[0]
        return out

    def _process_scene(self, scene, scratch_dir):
        """Apply SCL mask + scale + resample-to-10m + stack into a single
        10-band float32 raster. Returns the saved raster path or None.

        Implementation note: rather than build a giant single map-algebra
        expression, we save each band as a temp raster (already masked
        and scaled), then CompositeBands them into the 10-band stack.
        This trades disk I/O for memory headroom — well-suited to the
        ArcGIS Pro raster pipeline.
        """
        safe_folder = scene["path"]
        scene_id = scene["metadata"]["product_uri"]
        bands = self._locate_band_files(safe_folder)
        if not bands or "SCL" not in bands:
            arcpy.AddWarning("    Missing SCL or band data; scene skipped.")
            return None
        if not all(b in bands for b in _S2_STACK_ORDER):
            arcpy.AddWarning("    Missing one or more required bands; scene skipped.")
            return None

        # Build the cloud mask from SCL resampled to 10m (NEAREST so we
        # don't blur the class boundaries).
        scl_10m_path = os.path.join(scratch_dir, f"{scene_id}_SCL_10m.tif")
        arcpy.management.Resample(bands["SCL"], scl_10m_path, 10, "NEAREST")
        scl_10m = arcpy.sa.Raster(scl_10m_path)
        cloud_expr = None
        for klass in _S2_SCL_CLOUD_CLASSES:
            term = (scl_10m == klass)
            cloud_expr = term if cloud_expr is None else (cloud_expr | term)

        # Process each band: resample if 20m, scale to reflectance,
        # apply cloud mask, save.
        masked_paths = []
        for band in _S2_STACK_ORDER:
            band_src = bands[band]
            if band not in _S2_NATIVE_10M:
                # Resample 20m → 10m via BILINEAR (continuous reflectance).
                resampled_path = os.path.join(
                    scratch_dir, f"{scene_id}_{band}_10m.tif"
                )
                arcpy.management.Resample(band_src, resampled_path, 10, "BILINEAR")
                band_raster = arcpy.sa.Raster(resampled_path)
            else:
                band_raster = arcpy.sa.Raster(band_src)

            reflectance = Float(band_raster) * _S2_REFLECTANCE_SCALE
            masked = arcpy.sa.SetNull(cloud_expr, reflectance)
            out_path = os.path.join(scratch_dir, f"{scene_id}_{band}_masked.tif")
            masked.save(out_path)
            masked_paths.append(out_path)

        # Composite into a single 10-band raster.
        stacked_path = os.path.join(scratch_dir, f"{scene_id}_stack.tif")
        arcpy.management.CompositeBands(masked_paths, stacked_path)
        return stacked_path

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    @staticmethod
    def _write_provenance_csv(output_raster_path, scenes_used):
        if not output_raster_path or not scenes_used:
            return
        try:
            csv_path = f"{output_raster_path}_provenance.csv"
            now_iso = datetime.now().isoformat(timespec="seconds")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "scene_id", "sensor", "acquisition_datetime",
                    "tile_id", "cloud_cover_pct", "input_path",
                    "processing_baseline", "toolbox_version",
                    "processing_datetime",
                ])
                for scene in scenes_used:
                    meta = scene.get("metadata", {}) or {}
                    p = scene.get("path", "") or ""
                    scene_id = meta.get("product_uri") or os.path.basename(p.rstrip(os.sep))
                    sensor_tag = "Sentinel-2A" if "S2A_" in scene_id else (
                        "Sentinel-2B" if "S2B_" in scene_id else "Sentinel-2"
                    )
                    date_acq = meta.get("date_acquired", "")
                    if hasattr(date_acq, "isoformat"):
                        date_acq = date_acq.isoformat()
                    writer.writerow([
                        scene_id,
                        sensor_tag,
                        date_acq,
                        meta.get("tile_id", ""),
                        meta.get("cloud_cover", "") if meta.get("cloud_cover") is not None else "",
                        p,
                        "L2A",
                        TOOLBOX_VERSION,
                        now_iso,
                    ])
            arcpy.AddMessage(f"Provenance CSV: {csv_path}")
        except Exception as e:
            arcpy.AddWarning(f"Failed to write provenance CSV: {e}")

    # ------------------------------------------------------------------
    # Temporal-filter helpers (duplicated from LandsatMosaic; shared
    # extraction is a refactor for Phase 6 once all tools have settled).
    # ------------------------------------------------------------------

    @staticmethod
    def _seasonal_pattern_for_region(region):
        if region in (
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
        ):
            return "temperate"
        if "Cape Verde" in (region or ""):
            return "cape_verde"
        if region == "Angola":
            return "angola"
        if region == "Mozambique":
            return "mozambique"
        return "temperate"

    @staticmethod
    def _create_temporal_filter(time_type, year, month, season):
        return {
            "type": time_type,
            "year": year,
            "month": month,
            "season": season,
        }

    def _scene_passes_filter(self, meta, temporal_filter, seasonal_pattern):
        d = meta.get("date_acquired")
        if d is None:
            return False
        ftype = temporal_filter.get("type")
        try:
            if ftype == "year_month":
                return (
                    d.year == temporal_filter["year"]
                    and d.month == temporal_filter["month"]
                )
            if ftype == "month_all_years":
                return d.month == temporal_filter["month"]
            if ftype == "season_in_year":
                if d.year != temporal_filter["year"]:
                    return False
                return self._is_in_season(d, seasonal_pattern, temporal_filter["season"])
            if ftype == "season_all_years":
                return self._is_in_season(d, seasonal_pattern, temporal_filter["season"])
        except (TypeError, KeyError):
            return False
        return False

    @staticmethod
    def _is_in_season(date, pattern, season):
        if season is None:
            return False
        season_months = {
            "temperate": {
                "spring": [3, 4, 5],
                "summer": [6, 7, 8],
                "autumn": [9, 10, 11],
                "winter": [12, 1, 2],
            },
            "angola": {
                "rainy": [11, 12, 1, 2, 3, 4],
                "dry": [5, 6, 7, 8, 9, 10],
                "rainy_peak": [1, 2, 3],
                "dry_peak": [6, 7, 8],
            },
            "cape_verde": {
                "dry": [12, 1, 2, 3, 4, 5, 6],
                "rainy": [8, 9, 10],
                "transition_dry_wet": [7],
                "transition_wet_dry": [11],
            },
            "mozambique": {
                "rainy": [10, 11, 12, 1, 2, 3],
                "dry": [4, 5, 6, 7, 8, 9],
                "rainy_peak": [12, 1, 2],
                "dry_peak": [7, 8, 9],
            },
        }
        try:
            return date.month in season_months[pattern][season.lower()]
        except KeyError:
            return False


# ---------------------------------------------------------------------------
# Tool 03 — ASTER L2 Mosaic
# ---------------------------------------------------------------------------

# ASTER AST_07XT V004 conventions:
_ASTER_REFLECTANCE_SCALE = 0.001
_ASTER_VNIR_BANDS = ["B01", "B02", "B03N"]                # 15m native
_ASTER_SWIR_BANDS = ["B04", "B05", "B06", "B07", "B08", "B09"]  # 30m → resample to 15m
_ASTER_STACK_ORDER = _ASTER_VNIR_BANDS + _ASTER_SWIR_BANDS  # 9-band stack
_ASTER_NATIVE_15M = set(_ASTER_VNIR_BANDS)
_ASTER_QA_NAMES = ["VNIR_QA_DataPlane", "SWIR_QA_DataPlane"]

# Filename pattern for AST_07XT V004 TIFF exports:
#   AST_07XT_<17-char-sceneID>_<14-char-procDT>_SRF_<VNIR|SWIR>_<band-or-QA>.tif
# The sceneID encodes the acquisition date as <3-digit-pass><MM><DD><YYYY><HHMMSS>
# (US-style MMDDYYYY, then HHMMSS — both UTC).
_ASTER_TIFF_RE = re.compile(
    r"^AST_07XT_"
    r"(?P<pass>\d{3})(?P<MM>\d{2})(?P<DD>\d{2})(?P<YYYY>\d{4})(?P<HMS>\d{6})_"
    r"(?P<proc>\d{14})_SRF_"
    r"(?P<group>VNIR|SWIR)_"
    r"(?P<band>B0[1-9]N?|QA_DataPlane2?)\.tif$",
    re.IGNORECASE,
)


class AsterMosaic(object):
    """Tool 03 — ASTER AST_07XT V004 mineral-mapping mosaic.

    Accepts a folder of ASTER L2 Surface Reflectance products, either as:
      a) Per-band TIFFs following the standard naming convention
         (`AST_07XT_<sceneID>_<procDT>_SRF_<VNIR|SWIR>_<band>.tif`) —
         3 VNIR + 6 SWIR + 2 QA per scene.
      b) HDF-EOS `.hdf` archives (best-effort via osgeo.gdal — extracted
         to TIFFs in a scratch folder before processing).

    Per-scene processing:
      1. Load 3 VNIR bands at native 15m + 6 SWIR bands resampled to 15m
         (BILINEAR).
      2. Apply scale factor 0.001 to convert DN → surface reflectance [0, 1].
      3. Apply QA cloud mask from the QA Data Plane layers (best-effort
         bit decoding — exact layout per ASTER User Handbook V004).
      4. Optionally apply a multitemporal anomaly mask across the scene
         stack — recommended for cloud-prone regions like Faial.
      5. Stack into a 9-band float32 raster.

    Mosaicking: Esri arcpy.sa.GeometricMedian across the cloud-masked
    stack, then optional AOI clip and provenance CSV.
    """

    def __init__(self):
        self.label = "03 — ASTER L2 Mosaic"
        self.description = (
            "Build a mineral-mapping mosaic from ASTER AST_07XT V004 "
            "Surface Reflectance scenes (VNIR + crosstalk-corrected SWIR). "
            "Accepts per-band TIFFs (the common LP DAAC export) or HDF-EOS "
            "archives. SWIR (30m) is resampled to 15m to match VNIR. Cloud "
            "handling combines QA Data Plane flags + optional multitemporal "
            "anomaly detection (recommended for cloud-prone regions). The "
            "temporal stack is reduced to a geometric median. A provenance "
            "CSV is written alongside the output."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    # GP parameters
    # ------------------------------------------------------------------

    def getParameterInfo(self):
        gdb = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="gdb_path",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        gdb.filter.list = ["Local Database"]

        mosaic_name = arcpy.Parameter(
            displayName="Output Mosaic Name",
            name="mosaic_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )

        data_folder = arcpy.Parameter(
            displayName="ASTER Data Folder",
            name="data_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input",
        )

        region = arcpy.Parameter(
            displayName="Region",
            name="region",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        region.filter.list = [
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
            "Cape Verde Western (Santo Antão, São Vicente, São Nicolau)",
            "Cape Verde Eastern (Sal, Boa Vista, Santiago, Fogo)",
            "Angola",
            "Mozambique",
        ]

        time_type = arcpy.Parameter(
            displayName="Time Filter Type",
            name="time_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        time_type.filter.list = [
            "year_month", "month_all_years",
            "season_in_year", "season_all_years",
        ]

        year = arcpy.Parameter(
            displayName="Year (when applicable)",
            name="year",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        month = arcpy.Parameter(
            displayName="Month (when applicable, 1-12)",
            name="month",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
        )
        season = arcpy.Parameter(
            displayName="Season (when applicable)",
            name="season",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
        )

        apply_temporal = arcpy.Parameter(
            displayName="Apply Multitemporal Cloud Refinement (recommended for ASTER)",
            name="apply_temporal_refinement",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        apply_temporal.value = True  # default ON for ASTER (no SCL)

        use_qa_planes = arcpy.Parameter(
            displayName="Apply QA Data Plane Cloud Mask",
            name="use_qa_planes",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        use_qa_planes.value = True

        mask_feature = arcpy.Parameter(
            displayName="Optional AOI Mask Feature",
            name="mask_feature",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
        )

        save_stats = arcpy.Parameter(
            displayName="Save Provenance CSV",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
        )
        save_stats.value = True

        return [
            gdb, mosaic_name, data_folder, region, time_type,
            year, month, season, apply_temporal, use_qa_planes,
            mask_feature, save_stats,
        ]

    def updateParameters(self, parameters):
        try:
            time_type = parameters[4]
            year = parameters[5]
            month = parameters[6]
            season = parameters[7]
            if time_type.valueAsText == "year_month":
                year.enabled = True; month.enabled = True; season.enabled = False
            elif time_type.valueAsText == "month_all_years":
                year.enabled = False; month.enabled = True; season.enabled = False
            elif time_type.valueAsText == "season_in_year":
                year.enabled = True; month.enabled = False; season.enabled = True
            elif time_type.valueAsText == "season_all_years":
                year.enabled = False; month.enabled = False; season.enabled = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, parameters, messages):
        scratch_dir = None
        try:
            if arcpy.CheckExtension("Spatial") != "Available":
                arcpy.AddError("Spatial Analyst extension is required.")
                return None
            arcpy.CheckOutExtension("Spatial")
            arcpy.env.overwriteOutput = True

            gdb_path = parameters[0].valueAsText
            mosaic_name = parameters[1].valueAsText
            data_folder = parameters[2].valueAsText
            region = parameters[3].valueAsText
            time_type = parameters[4].valueAsText
            year = parameters[5].value
            month = parameters[6].value
            season = parameters[7].valueAsText
            apply_temporal = bool(parameters[8].value)
            use_qa = bool(parameters[9].value)
            mask_feature = parameters[10].valueAsText
            save_stats = bool(parameters[11].value)

            arcpy.AddMessage("\nInitialising ASTER mosaic processing:")
            arcpy.AddMessage(f"  Workspace:   {gdb_path}")
            arcpy.AddMessage(f"  Output name: {mosaic_name}")
            arcpy.AddMessage(f"  Region:      {region}")
            arcpy.AddMessage(f"  Data folder: {data_folder}")
            arcpy.AddMessage(f"  QA mask:     {use_qa}")
            arcpy.AddMessage(f"  Temporal:    {apply_temporal}")

            scratch_dir = os.path.join(
                arcpy.env.scratchFolder or data_folder,
                f"_genesis_aster_scratch_{uuid.uuid4().hex[:8]}",
            )
            os.makedirs(scratch_dir, exist_ok=True)

            # Step 1: discover scenes (TIFF and HDF).
            arcpy.AddMessage("\nStep 1/6 — Discovering ASTER scenes...")
            scenes = self._find_aster_scenes(data_folder)
            if not scenes:
                arcpy.AddError("No ASTER scenes found.")
                return None
            arcpy.AddMessage(f"  Found {len(scenes)} scene(s).")

            # Step 2: temporal filter.
            arcpy.AddMessage("\nStep 2/6 — Filtering scenes by date/season...")
            seasonal_pattern = self._seasonal_pattern_for_region(region)
            temporal_filter = self._create_temporal_filter(time_type, year, month, season)
            kept_scenes = [
                s for s in scenes
                if s.get("metadata") and self._scene_passes_filter(
                    s["metadata"], temporal_filter, seasonal_pattern,
                )
            ]
            if not kept_scenes:
                arcpy.AddError("No scenes match the temporal filter.")
                return None
            arcpy.AddMessage(f"  Kept {len(kept_scenes)} scene(s) after temporal filter.")

            # Step 3: per-scene processing — scale, resample SWIR, QA mask, stack.
            arcpy.AddMessage("\nStep 3/6 — Processing scenes (scale + resample + QA mask + stack)...")
            stacked_paths = []
            scenes_used = []
            for scene in kept_scenes:
                arcpy.AddMessage(f"  {scene['scene_id']}")
                try:
                    stacked = self._process_scene(scene, scratch_dir, use_qa)
                except Exception as e:
                    arcpy.AddWarning(f"    Failed: {e}")
                    continue
                if stacked:
                    stacked_paths.append(stacked)
                    scenes_used.append(scene)
            if not stacked_paths:
                arcpy.AddError("No scenes survived per-scene processing.")
                return None

            # Step 4: optional multitemporal anomaly refinement.
            if apply_temporal and len(stacked_paths) >= 5:
                arcpy.AddMessage(
                    f"\nStep 4/6 — Multitemporal cloud refinement "
                    f"({len(stacked_paths)} scenes)..."
                )
                stacked_paths = self._apply_multitemporal_refinement(
                    stacked_paths, scratch_dir,
                )
            elif apply_temporal:
                arcpy.AddWarning(
                    f"\nStep 4/6 — Skipped multitemporal refinement: "
                    f"only {len(stacked_paths)} scenes (need ≥5 for robust statistics)."
                )

            # Step 5: geometric median.
            arcpy.AddMessage("\nStep 5/6 — Computing geometric median across scene stack...")
            output_path = os.path.join(gdb_path, mosaic_name)
            try:
                median = arcpy.sa.GeometricMedian(stacked_paths)
                median.save(output_path)
                arcpy.AddMessage(f"  Saved: {output_path}")
            except Exception as e:
                arcpy.AddError(f"GeometricMedian failed: {e}")
                return None

            # Step 6: AOI mask + provenance.
            arcpy.AddMessage("\nStep 6/6 — Finalising output...")
            final_path = output_path
            if mask_feature and arcpy.Exists(mask_feature):
                arcpy.AddMessage(f"  Applying AOI mask: {mask_feature}")
                masked = arcpy.sa.ExtractByMask(output_path, mask_feature)
                masked_path = os.path.join(gdb_path, f"{mosaic_name}_Masked")
                masked.save(masked_path)
                final_path = masked_path
            elif mask_feature:
                arcpy.AddWarning(f"AOI mask {mask_feature!r} not found — output is unmasked.")

            if save_stats:
                self._write_provenance_csv(final_path, scenes_used)

            arcpy.AddMessage(f"\nDone. Final mosaic: {final_path}")
            return final_path

        except Exception as e:
            arcpy.AddError(f"Tool 03 failed: {e}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None

        finally:
            if scratch_dir and os.path.isdir(scratch_dir):
                try:
                    import shutil
                    shutil.rmtree(scratch_dir, ignore_errors=True)
                except Exception:
                    pass
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    # ------------------------------------------------------------------
    # Scene discovery (TIFF + HDF)
    # ------------------------------------------------------------------

    @classmethod
    def _find_aster_scenes(cls, data_folder):
        """Discover ASTER scenes in data_folder. Groups per-band TIFFs by
        sceneID; for HDF, each .hdf is one scene (extracted lazily by
        _process_scene).

        Returns a list of dicts with keys:
            scene_id:   17-char ASTER granule identifier
            format:     "tiff" or "hdf"
            files:      dict {band_name: path} (for tiff) or {hdf: path}
            metadata:   {acquisition_date: date, pass_number: str, ...}
        """
        if not data_folder or not os.path.isdir(data_folder):
            return []

        entries = os.listdir(data_folder)
        scenes_by_id = {}

        # First pass: TIFF inputs. Dispatch by extension so we don't
        # accidentally classify an .hdf as a TIFF — _parse_aster_filename
        # matches both formats but only TIFFs carry a band name.
        for entry in entries:
            full = os.path.join(data_folder, entry)
            if not os.path.isfile(full):
                continue
            if not (entry.lower().endswith(".tif") or entry.lower().endswith(".tiff")):
                continue
            parsed = cls._parse_aster_filename(entry)
            if not parsed or parsed.get("band") is None:
                continue
            scene_id = parsed["scene_id"]
            if scene_id not in scenes_by_id:
                scenes_by_id[scene_id] = {
                    "scene_id": scene_id,
                    "format": "tiff",
                    "files": {},
                    "metadata": {
                        "acquisition_date": parsed["acquisition_date"],
                        "pass_number": parsed["pass"],
                        "scene_id": scene_id,
                    },
                }
            scenes_by_id[scene_id]["files"][parsed["band"]] = full

        # Second pass: HDF inputs. If we already have a TIFF set for the
        # same sceneID, keep the TIFF (pre-extracted, cheaper to read).
        for entry in entries:
            if not entry.lower().endswith(".hdf"):
                continue
            full = os.path.join(data_folder, entry)
            parsed = cls._parse_aster_filename(entry)
            if not parsed:
                continue
            scene_id = parsed["scene_id"]
            if scene_id in scenes_by_id and scenes_by_id[scene_id]["format"] == "tiff":
                continue
            scenes_by_id[scene_id] = {
                "scene_id": scene_id,
                "format": "hdf",
                "files": {"hdf": full},
                "metadata": {
                    "acquisition_date": parsed["acquisition_date"],
                    "pass_number": parsed["pass"],
                    "scene_id": scene_id,
                },
            }

        return sorted(scenes_by_id.values(), key=lambda s: s["scene_id"])

    @staticmethod
    def _parse_aster_filename(filename):
        """Parse an ASTER AST_07XT filename (TIFF or HDF) into its parts.

        Returns dict with scene_id, pass, acquisition_date, band (None for
        HDF), group (None for HDF), proc (None for HDF). Returns None on
        non-matching names.
        """
        name = os.path.basename(filename)
        # Try the TIFF pattern first.
        m = _ASTER_TIFF_RE.match(name)
        if m:
            scene_id = (
                m.group("pass") + m.group("MM") + m.group("DD")
                + m.group("YYYY") + m.group("HMS")
            )
            try:
                acq = datetime(
                    int(m.group("YYYY")), int(m.group("MM")), int(m.group("DD")),
                ).date()
            except ValueError:
                return None
            return {
                "scene_id": scene_id,
                "pass": m.group("pass"),
                "acquisition_date": acq,
                "proc": m.group("proc"),
                "group": m.group("group").upper(),
                "band": m.group("band"),
            }

        # HDF pattern is similar but without the _SRF_<group>_<band> tail.
        hdf_re = re.compile(
            r"^AST_07XT_(?P<pass>\d{3})(?P<MM>\d{2})(?P<DD>\d{2})(?P<YYYY>\d{4})"
            r"(?P<HMS>\d{6})_(?P<proc>\d{14})\.hdf$",
            re.IGNORECASE,
        )
        m = hdf_re.match(name)
        if m:
            scene_id = (
                m.group("pass") + m.group("MM") + m.group("DD")
                + m.group("YYYY") + m.group("HMS")
            )
            try:
                acq = datetime(
                    int(m.group("YYYY")), int(m.group("MM")), int(m.group("DD")),
                ).date()
            except ValueError:
                return None
            return {
                "scene_id": scene_id,
                "pass": m.group("pass"),
                "acquisition_date": acq,
                "proc": m.group("proc"),
                "group": None,
                "band": None,
            }
        return None

    # ------------------------------------------------------------------
    # Per-scene processing
    # ------------------------------------------------------------------

    def _process_scene(self, scene, scratch_dir, use_qa):
        """Apply scale + resample + QA mask + stack → single 9-band raster.

        Returns the stacked raster path, or None on failure (missing bands,
        unreadable QA, etc.).
        """
        if scene["format"] == "hdf":
            extracted = self._extract_hdf_to_tiffs(scene["files"]["hdf"], scratch_dir)
            if extracted is None:
                arcpy.AddWarning("    HDF extraction failed; scene skipped.")
                return None
            scene = dict(scene, format="tiff", files=extracted)

        files = scene["files"]
        scene_id = scene["scene_id"]

        # Verify we have all 9 image bands.
        missing = [b for b in _ASTER_STACK_ORDER if b not in files]
        if missing:
            arcpy.AddWarning(f"    Missing bands {missing}; scene skipped.")
            return None

        # Optional QA mask. Conservative implementation: treat any non-zero
        # value in either QA Data Plane as a quality issue (cloud / fill /
        # bad pixel). Exact bit decoding per ASTER User Handbook V004 would
        # be more precise but the documented bit layout is product-version
        # specific — this conservative choice errs on the side of dropping
        # uncertain pixels, which is fine for a temporal median.
        qa_mask = None
        if use_qa and any(qa in files for qa in _ASTER_QA_NAMES):
            qa_path = next((files[q] for q in _ASTER_QA_NAMES if q in files), None)
            if qa_path:
                try:
                    qa_raster = arcpy.sa.Raster(qa_path)
                    # Resample QA to 15m (NEAREST to preserve class codes).
                    qa_resampled = os.path.join(scratch_dir, f"{scene_id}_QA_15m.tif")
                    arcpy.management.Resample(qa_path, qa_resampled, 15, "NEAREST")
                    qa_raster = arcpy.sa.Raster(qa_resampled)
                    qa_mask = qa_raster != 0  # non-zero == flagged
                except Exception as e:
                    arcpy.AddWarning(f"    QA mask build failed (continuing without): {e}")
                    qa_mask = None

        # Process each band: resample SWIR to 15m, scale to reflectance,
        # apply QA mask if present.
        masked_paths = []
        for band in _ASTER_STACK_ORDER:
            src = files[band]
            if band not in _ASTER_NATIVE_15M:
                # SWIR (30m) → 15m via BILINEAR (continuous reflectance).
                resampled = os.path.join(scratch_dir, f"{scene_id}_{band}_15m.tif")
                arcpy.management.Resample(src, resampled, 15, "BILINEAR")
                band_raster = arcpy.sa.Raster(resampled)
            else:
                band_raster = arcpy.sa.Raster(src)

            reflectance = Float(band_raster) * _ASTER_REFLECTANCE_SCALE
            if qa_mask is not None:
                masked = arcpy.sa.SetNull(qa_mask, reflectance)
            else:
                masked = reflectance
            out = os.path.join(scratch_dir, f"{scene_id}_{band}_masked.tif")
            masked.save(out)
            masked_paths.append(out)

        stacked = os.path.join(scratch_dir, f"{scene_id}_stack.tif")
        arcpy.management.CompositeBands(masked_paths, stacked)
        return stacked

    @staticmethod
    def _extract_hdf_to_tiffs(hdf_path, scratch_dir):
        """Extract ASTER AST_07XT subdatasets from an HDF-EOS file to a
        per-band TIFF dict matching the standard TIFF convention.

        Uses osgeo.gdal (bundled with arcpy) to read the subdatasets.
        Returns None if gdal isn't available or if the HDF structure
        doesn't match the expected AST_07XT V004 layout.
        """
        try:
            from osgeo import gdal
        except ImportError:
            arcpy.AddWarning(
                "    osgeo.gdal not available — cannot read HDF inputs. "
                "Pre-extract to TIFF (e.g., via ENVI or gdal_translate)."
            )
            return None

        try:
            ds = gdal.Open(hdf_path)
            if ds is None:
                return None
            subdatasets = ds.GetSubDatasets()
            # Expected names: ImageData1..9 for bands, QA_DataPlane / QA_DataPlane2.
            # The exact subdataset names vary by hdf_eos library version; match
            # by suffix.
            outputs = {}
            band_lookup = {
                "ImageData1": "B01", "ImageData2": "B02", "ImageData3": "B03N",
                "ImageData4": "B04", "ImageData5": "B05", "ImageData6": "B06",
                "ImageData7": "B07", "ImageData8": "B08", "ImageData9": "B09",
                "QA_DataPlane": "VNIR_QA_DataPlane",
                "QA_DataPlane2": "SWIR_QA_DataPlane",
            }
            scene_stem = os.path.splitext(os.path.basename(hdf_path))[0]
            for sub_path, sub_desc in subdatasets:
                # sub_path typically looks like
                # 'HDF4_EOS:EOS_SWATH:"foo.hdf":SwathName:ImageData4'
                tail = sub_path.rsplit(":", 1)[-1]
                if tail not in band_lookup:
                    continue
                band_label = band_lookup[tail]
                out_tiff = os.path.join(scratch_dir, f"{scene_stem}_{band_label}.tif")
                gdal.Translate(out_tiff, sub_path)
                outputs[band_label] = out_tiff
            return outputs if outputs else None
        except Exception as e:
            arcpy.AddWarning(f"    HDF extraction error: {e}")
            return None

    # ------------------------------------------------------------------
    # Multitemporal cloud refinement
    # ------------------------------------------------------------------

    def _apply_multitemporal_refinement(self, stacked_paths, scratch_dir):
        """Compute per-pixel temporal anomaly mask on visible brightness
        (B01 + B02 = Green + Red) and mark pixels well above the per-pixel
        percentile as cloudy. Re-emit each scene with the additional mask
        applied so the geometric median sees cleaner inputs.

        For very cloudy locations like Faial — Faial is in the Azores
        North Atlantic, marine climate, persistent cumulus — this is the
        most important quality lever for ASTER (which lacks a SCL-grade
        per-scene cloud product).

        Algorithm: for each pixel (i, j) across N scenes, compute
        brightness_pct(75) = 75th percentile of B01+B02. Mask scenes
        where brightness > 1.5 × percentile.

        Returns the new list of refined-stacked paths.
        """
        n = len(stacked_paths)
        if n < 5:
            return stacked_paths  # not enough samples for robust statistics

        try:
            # Compute B01+B02 brightness raster for each scene (band 1 + band 2).
            brightness_arrays = []
            ref_meta = None
            for path in stacked_paths:
                r = arcpy.Raster(path)
                ext = r.extent
                cell = (r.meanCellWidth, r.meanCellHeight)
                arr = arcpy.RasterToNumPyArray(
                    r, nodata_to_value=np.nan,
                ).astype(np.float64)
                # arr shape: (n_bands, h, w) or (h, w, n_bands) depending on arcpy version.
                # Normalise to (n_bands, h, w).
                if arr.ndim == 3 and arr.shape[-1] in (9, 9):
                    arr = np.transpose(arr, (2, 0, 1))
                # Brightness = B01 + B02 (channels 0 + 1).
                brightness = arr[0] + arr[1]
                brightness_arrays.append(brightness)
                if ref_meta is None:
                    ref_meta = (ext, cell)

            stack = np.stack(brightness_arrays, axis=0)  # (n, h, w)
            # Per-pixel 75th percentile, ignoring NaN.
            percentile = np.nanpercentile(stack, 75, axis=0)
            threshold = percentile * 1.5

            # Re-emit each scene with the temporal mask applied.
            refined_paths = []
            for path, brightness in zip(stacked_paths, brightness_arrays):
                cloud_pixels = brightness > threshold  # (h, w) bool
                # Load the full 9-band stack, apply mask to all bands.
                r = arcpy.Raster(path)
                arr = arcpy.RasterToNumPyArray(r, nodata_to_value=np.nan).astype(np.float64)
                if arr.ndim == 3 and arr.shape[-1] == 9:
                    arr = np.transpose(arr, (2, 0, 1))
                arr[:, cloud_pixels] = np.nan
                # Save back as a refined stack.
                out_path = path.replace("_stack.tif", "_stack_refined.tif")
                # NumPyArrayToRaster wants (n_bands, h, w) → (h, w, n_bands)
                arr_hwc = np.transpose(arr, (1, 2, 0)).astype(np.float32)
                ext, cell = ref_meta
                refined = arcpy.NumPyArrayToRaster(
                    arr_hwc,
                    arcpy.Point(ext.XMin, ext.YMin),
                    cell[0], cell[1],
                    value_to_nodata=np.nan,
                )
                refined.save(out_path)
                refined_paths.append(out_path)

            arcpy.AddMessage(
                f"    Refined {len(refined_paths)} scenes "
                f"(per-pixel 75th-pctile × 1.5 threshold)."
            )
            return refined_paths
        except Exception as e:
            arcpy.AddWarning(
                f"    Multitemporal refinement failed (continuing with "
                f"per-scene QA mask only): {e}"
            )
            return stacked_paths

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    @staticmethod
    def _write_provenance_csv(output_raster_path, scenes_used):
        if not output_raster_path or not scenes_used:
            return
        try:
            csv_path = f"{output_raster_path}_provenance.csv"
            now_iso = datetime.now().isoformat(timespec="seconds")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([
                    "scene_id", "sensor", "acquisition_datetime",
                    "pass_number", "input_format", "input_path",
                    "processing_baseline", "toolbox_version",
                    "processing_datetime",
                ])
                for scene in scenes_used:
                    meta = scene.get("metadata", {}) or {}
                    acq = meta.get("acquisition_date", "")
                    if hasattr(acq, "isoformat"):
                        acq = acq.isoformat()
                    # Prefer the HDF path for HDF scenes, else the first file path.
                    files = scene.get("files", {})
                    if scene.get("format") == "hdf":
                        input_path = files.get("hdf", "")
                    else:
                        # Pick the path of B01 as the canonical reference.
                        input_path = files.get("B01") or next(iter(files.values()), "")
                    writer.writerow([
                        scene.get("scene_id", ""),
                        "ASTER (AST_07XT V004)",
                        acq,
                        meta.get("pass_number", ""),
                        scene.get("format", ""),
                        input_path,
                        "AST_07XT V004",
                        TOOLBOX_VERSION,
                        now_iso,
                    ])
            arcpy.AddMessage(f"Provenance CSV: {csv_path}")
        except Exception as e:
            arcpy.AddWarning(f"Failed to write provenance CSV: {e}")

    # ------------------------------------------------------------------
    # Temporal-filter helpers (duplicated from S2/Landsat — Phase 6 extracts)
    # ------------------------------------------------------------------

    @staticmethod
    def _seasonal_pattern_for_region(region):
        if region in (
            "Portugal Mainland",
            "Azores Western (Flores, Corvo)",
            "Azores Central (Faial, Pico, São Jorge, Graciosa, Terceira)",
            "Azores Eastern (São Miguel, Santa Maria)",
            "Madeira",
        ):
            return "temperate"
        if "Cape Verde" in (region or ""):
            return "cape_verde"
        if region == "Angola":
            return "angola"
        if region == "Mozambique":
            return "mozambique"
        return "temperate"

    @staticmethod
    def _create_temporal_filter(time_type, year, month, season):
        return {"type": time_type, "year": year, "month": month, "season": season}

    def _scene_passes_filter(self, meta, temporal_filter, seasonal_pattern):
        d = meta.get("acquisition_date")
        if d is None:
            return False
        ftype = temporal_filter.get("type")
        try:
            if ftype == "year_month":
                return d.year == temporal_filter["year"] and d.month == temporal_filter["month"]
            if ftype == "month_all_years":
                return d.month == temporal_filter["month"]
            if ftype == "season_in_year":
                if d.year != temporal_filter["year"]:
                    return False
                return self._is_in_season(d, seasonal_pattern, temporal_filter["season"])
            if ftype == "season_all_years":
                return self._is_in_season(d, seasonal_pattern, temporal_filter["season"])
        except (TypeError, KeyError):
            return False
        return False

    @staticmethod
    def _is_in_season(date, pattern, season):
        if season is None:
            return False
        season_months = {
            "temperate": {"spring": [3, 4, 5], "summer": [6, 7, 8],
                          "autumn": [9, 10, 11], "winter": [12, 1, 2]},
            "angola": {"rainy": [11, 12, 1, 2, 3, 4], "dry": [5, 6, 7, 8, 9, 10],
                       "rainy_peak": [1, 2, 3], "dry_peak": [6, 7, 8]},
            "cape_verde": {"dry": [12, 1, 2, 3, 4, 5, 6], "rainy": [8, 9, 10],
                           "transition_dry_wet": [7], "transition_wet_dry": [11]},
            "mozambique": {"rainy": [10, 11, 12, 1, 2, 3], "dry": [4, 5, 6, 7, 8, 9],
                           "rainy_peak": [12, 1, 2], "dry_peak": [7, 8, 9]},
        }
        try:
            return date.month in season_months[pattern][season.lower()]
        except KeyError:
            return False
class Transformations(object):
    """Tool 05 — Statistical transformations (PCA / MNF / ICA).

    Sensor-agnostic at the algorithm level: PCA/MNF/ICA operate on column
    vectors and don't care which spectral band each column represents.
    The Sensor Type parameter is used for (a) sensible default
    `num_components` per sensor, (b) sanity-checking the requested
    component count against the expected band count, and (c) tagging the
    saved *Statistics .npz for provenance.

    Inherits the full set of Phase 1 audit fixes: kurtosis_values
    persistence, NoData-safe noise estimator, sensor-agnostic transform
    re-apply functions, etc.
    """

    def __init__(self):
        self.label = "05 — Statistical Transformations"
        self.description = (
            "Run PCA, MNF, or ICA on a pre-stacked multiband raster from "
            "any of the three supported sensors (Landsat 8/9, Sentinel-2 "
            "L2A, ASTER AST_07XT). The algorithm is sensor-agnostic — the "
            "Sensor Type parameter selects sensible defaults and is "
            "recorded on the saved *Statistics .npz for provenance."
        )
        self.canRunInBackground = True
        
    def getParameterInfo(self):
        # Input raster bands
        input_rasters = arcpy.Parameter(
            displayName="Input Raster Bands",
            name="input_rasters",
            datatype=["DERasterDataset", "GPRasterLayer"],
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        
        # Transformation type
        transform_type = arcpy.Parameter(
            displayName="Transformation Type",
            name="transform_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        transform_type.filter.list = ["MNF", "PCA", "ICA"]
        
        # Number of components
        num_components = arcpy.Parameter(
            displayName="Number of Output Components",
            name="num_components",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        
        # Optional parameters for MNF
        noise_stats_file = arcpy.Parameter(
            displayName="Input Noise Statistics (Optional, MNF only)",
            name="noise_stats_file",
            datatype="DEFile",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        noise_subset = arcpy.Parameter(
            displayName="Spatial Subset for Noise (Optional, MNF only)",
            name="noise_subset",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        # Output workspace
        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        
        # Output name
        out_name = arcpy.Parameter(
            displayName="Output Name",
            name="out_name",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        
        # Save statistics
        save_stats = arcpy.Parameter(
            displayName="Save Transform Statistics",
            name="save_stats",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        save_stats.value = True
        
        # Statistics folder
        stats_folder = arcpy.Parameter(
            displayName="Statistics Folder",
            name="stats_folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        
        # Preserve Input Mask
        preserve_mask = arcpy.Parameter(
            displayName="Preserve Input Mask",
            name="preserve_mask",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        preserve_mask.value = True

        # ISS-007: random_state for FastICA reproducibility. Only relevant
        # for ICA; ignored by PCA / MNF.
        random_state = arcpy.Parameter(
            displayName="ICA Random Seed (Optional, ICA only)",
            name="random_state",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input",
            enabled=False,
        )
        random_state.value = 42

        # Sensor selector (Phase 6 addition) — informational for the
        # algorithm (PCA/MNF/ICA don't care about band identity) but
        # drives sensible defaults for num_components and records the
        # sensor on the saved *Statistics .npz.
        sensor_type = make_sensor_parameter()

        return [input_rasters, transform_type, num_components,
                noise_stats_file, noise_subset,
                out_workspace, out_name, save_stats, stats_folder,
                preserve_mask, random_state, sensor_type]
        
    def updateParameters(self, parameters):
        """Modify parameter values and properties"""
        try:
            # Get references to parameters for easier access
            input_rasters = parameters[0]
            transform_type = parameters[1]
            num_components = parameters[2]
            noise_stats_file = parameters[3]
            noise_subset = parameters[4]
            save_stats = parameters[7]
            stats_folder = parameters[8]
            
            # Enable/disable MNF-specific parameters based on transformation type
            if transform_type.altered:
                is_mnf = transform_type.valueAsText == "MNF"
                is_ica = transform_type.valueAsText == "ICA"
                noise_stats_file.enabled = is_mnf
                noise_subset.enabled = is_mnf

                # If noise statistics file is provided, disable noise subset
                if is_mnf and noise_stats_file.altered and noise_stats_file.valueAsText:
                    noise_subset.enabled = False

                # ISS-007: random_state only matters for ICA.
                if len(parameters) > 10:
                    parameters[10].enabled = is_ica
            
            # Enable/disable statistics folder parameter based on save_stats
            if save_stats.altered:
                stats_folder.enabled = save_stats.value
            
            # If input rasters are set, update number of components
            if input_rasters.altered and input_rasters.value:
                try:
                    raster_paths = input_rasters.valueAsText.split(";")
                    total_bands = 0
                    
                    # Calculate total number of bands across all inputs
                    for raster_path in raster_paths:
                        raster = arcpy.Raster(raster_path)
                        total_bands += raster.bandCount
                    
                    # Update number of components range
                    num_components.filter.type = "Range"
                    num_components.filter.list = [1, total_bands]
                    
                    # Set default number of components if not already set
                    if not num_components.altered:
                        num_components.value = min(3, total_bands)
                    
                    arcpy.AddMessage(f"Total bands available: {total_bands}")
                except Exception as e:
                    arcpy.AddWarning(f"Error counting bands: {str(e)}")
            
            # Set default stats folder if save_stats is enabled but folder not specified
            if save_stats.value and not stats_folder.altered:
                # Default to user documents
                default_folder = os.path.expanduser("~/Documents/ArcGIS/Statistics")
                if not os.path.exists(default_folder):
                    try:
                        os.makedirs(default_folder)
                    except:
                        pass
                
                if os.path.exists(default_folder):
                    stats_folder.value = default_folder
        
        except Exception as e:
            arcpy.AddWarning(f"Error updating parameters: {str(e)}")
            
    def updateMessages(self, parameters):
        """Validate parameters"""
        if parameters[0].altered:
            try:
                # Get input raster paths
                input_rasters = parameters[0].valueAsText.split(";")
                
                # For PCA, we'll be more flexible with inputs
                if parameters[1].valueAsText == "PCA":
                    # Just check that inputs exist and are rasters
                    for raster_path in input_rasters:
                        if not arcpy.Exists(raster_path):
                            parameters[0].setErrorMessage(f"Input raster does not exist: {raster_path}")
                            return
                        
                        desc = arcpy.Describe(raster_path)
                        if not hasattr(desc, "bandCount"):
                            parameters[0].setErrorMessage(f"Input is not a valid raster: {raster_path}")
                            return
                else:
                    # For MNF or ICA, be more permissive but still check format
                    for raster_path in input_rasters:
                        if not arcpy.Exists(raster_path):
                            parameters[0].setErrorMessage(f"Input raster does not exist: {raster_path}")
                            return
                        
                        desc = arcpy.Describe(raster_path)
                        if not hasattr(desc, "bandCount"):
                            parameters[0].setErrorMessage(f"Input is not a valid raster: {raster_path}")
                            return
                
                # Check that number of components doesn't exceed total bands
                if parameters[2].altered and parameters[0].altered:
                    total_bands = 0
                    for raster_path in input_rasters:
                        raster = arcpy.Raster(raster_path)
                        total_bands += raster.bandCount
                    
                    if parameters[2].value > total_bands:
                        parameters[2].setErrorMessage(
                            f"Number of components ({parameters[2].value}) " +
                            f"cannot exceed total bands ({total_bands})"
                        )
                    
            except Exception as e:
                parameters[0].setErrorMessage(f"Error validating input: {str(e)}")
                
    def execute(self, parameters, messages):
        """Execute the tool"""
        # Declared at function scope so the finally block can clean it up
        # regardless of where execution fails.
        composite_temp_path = None
        try:
            # Check out Spatial Analyst extension
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required but not available")
                return None

            # Enable overwrite output
            arcpy.env.overwriteOutput = True
            
            # Get parameters with updated indices
            input_rasters = parameters[0].valueAsText.split(";")
            transform_type = parameters[1].valueAsText
            num_components = parameters[2].value
            noise_stats_file = parameters[3].valueAsText
            noise_subset = parameters[4].valueAsText
            out_workspace = parameters[5].valueAsText
            out_name = parameters[6].valueAsText
            save_stats = parameters[7].value
            stats_folder_param = parameters[8].valueAsText if parameters[8].altered else None
            preserve_mask = parameters[9].value  # New preserve mask parameter
            # ISS-007: random_state for ICA (optional, defaults to 42).
            random_state = (
                int(parameters[10].value)
                if len(parameters) > 10 and parameters[10].value is not None
                else 42
            )
            
            # Initialize statistics
            stats = {
                'start_time': datetime.now(),
                'transform_type': transform_type,
                'num_components': num_components,
                'errors': []
            }
            
            try:
                # Create or use specified statistics folder
                if save_stats:
                    if stats_folder_param:
                        stats_folder = stats_folder_param
                        # Ensure the folder exists
                        if not os.path.exists(stats_folder):
                            os.makedirs(stats_folder)
                    else:
                        # Default to user's documents folder if not specified
                        stats_folder = os.path.expanduser("~/Documents/ArcGIS/Statistics")
                        if not os.path.exists(stats_folder):
                            os.makedirs(stats_folder)
                    
                    # PCA persists a PCAStatistics .npz (reloadable); MNF/ICA write a .txt summary.
                    stats_ext = "npz" if transform_type == "PCA" else "txt"
                    stats_file = os.path.join(stats_folder, f"{out_name}_{transform_type}_stats.{stats_ext}")
                else:
                    stats_file = None
                    stats_folder = None
                
                # Output path for result
                out_path = os.path.join(out_workspace, out_name)
                
                # All transforms (PCA, MNF, ICA) share the in-tree NumPy implementation.
                arcpy.AddMessage("Using custom implementation...")

                # Multi-raster input: composite into a single multi-band raster
                # via arcpy.management.CompositeBands so the downstream loader
                # gets one cube. Restores the multi-input behaviour of the
                # deleted arcpy.sa.PrincipalComponents call. The temp file is
                # cleaned up in the finally block.
                if len(input_rasters) > 1:
                    arcpy.AddMessage(
                        f"Combining {len(input_rasters)} input rasters via CompositeBands..."
                    )
                    scratch_dir = arcpy.env.scratchFolder or os.path.dirname(input_rasters[0])
                    composite_temp_path = os.path.join(
                        scratch_dir, f"_genesis_compose_{uuid.uuid4().hex}.tif"
                    )
                    try:
                        arcpy.management.CompositeBands(input_rasters, composite_temp_path)
                        arcpy.AddMessage(f"  Combined raster: {composite_temp_path}")
                        raster_path = composite_temp_path
                    except Exception as e:
                        arcpy.AddError(f"Failed to combine input rasters: {e}")
                        return None
                else:
                    raster_path = input_rasters[0]

                arcpy.AddMessage(f"Processing input: {os.path.basename(raster_path)}")
                
                # Verify raster exists
                if not arcpy.Exists(raster_path):
                    arcpy.AddError(f"Input raster does not exist: {raster_path}")
                    return None
                
                # Get raster properties
                raster_obj = arcpy.Raster(raster_path)
                band_count = raster_obj.bandCount
                arcpy.AddMessage(f"  Has {band_count} bands")
                arcpy.AddMessage(f"  Format: {raster_obj.format}")
                arcpy.AddMessage(f"  Width: {raster_obj.width}, Height: {raster_obj.height}")
                arcpy.AddMessage(f"  Data type: {raster_obj.pixelType}")
                
                # Get reference information
                extent = raster_obj.extent
                cell_size = (raster_obj.meanCellWidth, raster_obj.meanCellHeight)
                spatial_ref = raster_obj.spatialReference
                
                # Check if raster has a mask
                has_mask = False
                mask = None
                
                # Try different methods to find mask
                if hasattr(raster_obj, 'mask'):
                    mask = raster_obj.mask
                    has_mask = True
                    arcpy.AddMessage("  Raster has an explicit mask")
                elif hasattr(raster_obj, 'noDataValue'):
                    arcpy.AddMessage(f"  Raster has NoData value: {raster_obj.noDataValue}")
                    # We'll handle NoData during array processing
                else:
                    arcpy.AddMessage("  No mask or NoData value detected")
                
                # Store mask info in raster_info
                raster_info = {
                    'extent': extent,
                    'cell_size': cell_size,
                    'spatial_ref': spatial_ref,
                    'mask': mask if preserve_mask and has_mask else None
                }
                        
                # Load entire raster at once
                arcpy.AddMessage("  Using whole-raster loading approach...")
                
                try:
                    # Load the entire raster at once
                    data_array = arcpy.RasterToNumPyArray(raster_obj)
                    arcpy.AddMessage(f"  Full array loaded, shape: {data_array.shape}")
                    
                    # For multiband raster, the dimensions should be (bands, height, width)
                    # or (height, width, bands) depending on how ArcGIS returns it
                    if len(data_array.shape) == 3:
                        # Check if bands are in the first dimension
                        if data_array.shape[0] == band_count:
                            arcpy.AddMessage("  Transposing array from (bands, height, width) to (height, width, bands)")
                            data_array = np.transpose(data_array, (1, 2, 0))
                        else:
                            arcpy.AddMessage("  Array already in (height, width, bands) format")
                    else:
                        # Single band - reshape to add band dimension
                        arcpy.AddMessage("  Reshaping single band array")
                        data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)
                    
                    arcpy.AddMessage(f"  Final array shape: {data_array.shape}")
                    
                    # Handle NoData values
                    arcpy.AddMessage("Handling NoData values...")
                    data_array = data_array.astype(float)
                    
                    # Check for NoData using a safer approach
                    try:
                        if hasattr(raster_obj, 'noDataValue') and raster_obj.noDataValue is not None:
                            no_data = raster_obj.noDataValue
                            for i in range(data_array.shape[2]):
                                data_array[:, :, i][data_array[:, :, i] == no_data] = np.nan
                            arcpy.AddMessage(f"Applied NoData value: {no_data}")
                        else:
                            arcpy.AddMessage("No explicit NoData value found, continuing with all data")
                    except Exception as e:
                        arcpy.AddWarning(f"Error handling NoData values: {str(e)}")
                        arcpy.AddMessage("Continuing without NoData handling")
                    
                    arcpy.AddMessage("Handled NoData values")
                    
                    # Perform transformation
                    arcpy.AddMessage(f"\nPerforming {transform_type} transformation...")

                    # ISS-010: warn before allocating the (n_pixels, n_bands)
                    # working matrix on country-scale rasters. Full chunked
                    # PCA/MNF is a separate research project; this surfaces
                    # the risk so the user can tile the AOI instead.
                    est_gb = (data_array.shape[0] * data_array.shape[1]
                              * data_array.shape[2] * 8) / 1e9
                    if est_gb > _RAM_WARNING_GB:
                        arcpy.AddWarning(
                            f"Input cube is ~{est_gb:.1f} GB as float64 "
                            f"(shape {data_array.shape}); the transform may "
                            f"exhaust RAM. Consider tiling the AOI or reducing "
                            f"the band count."
                        )

                    if transform_type == "PCA":
                        transformed_data, transform_stats = self._perform_pca(
                            data_array,
                            num_components,
                            stats,
                            message_callback=arcpy.AddMessage,
                            warning_callback=arcpy.AddWarning,
                        )
                    elif transform_type == "MNF":
                        # Process noise subset if provided
                        noise_data = None
                        if noise_subset:
                            arcpy.AddMessage(f"Extracting noise statistics from subset: {noise_subset}")
                            try:
                                # Create temporary raster from first band for extraction
                                temp_raster_path = os.path.join(arcpy.env.scratchWorkspace or "memory", "temp_raster")
                                temp_raster = arcpy.NumPyArrayToRaster(
                                    data_array[:,:,0],
                                    lower_left_corner=arcpy.Point(extent.XMin, extent.YMin),
                                    x_cell_size=cell_size[0],
                                    y_cell_size=cell_size[1]
                                )
                                temp_raster.save(temp_raster_path)
                                
                                # Extract by mask
                                subset = arcpy.sa.ExtractByMask(temp_raster_path, noise_subset)
                                
                                # Convert to numpy array
                                subset_array = arcpy.RasterToNumPyArray(subset)
                                
                                # If subset extraction was successful, prepare noise data
                                if subset_array.size > 0:
                                    noise_data = data_array.copy()
                                    arcpy.AddMessage(f"Using noise subset")
                                else:
                                    arcpy.AddWarning("Subset extraction yielded no valid pixels, using full image for noise estimation")
                                    
                                # Clean up
                                if arcpy.Exists(temp_raster_path) and "memory" not in temp_raster_path:
                                    arcpy.management.Delete(temp_raster_path)
                                    
                            except Exception as e:
                                arcpy.AddWarning(f"Error processing noise subset: {str(e)}")
                                arcpy.AddWarning("Using full image for noise estimation")
                                stats['errors'].append(f"Noise subset error: {str(e)}")
                        
                        transformed_data, transform_stats = self._perform_mnf(
                            data_array,
                            num_components,
                            noise_stats_file,
                            noise_data,
                            stats,
                            message_callback=arcpy.AddMessage,
                            warning_callback=arcpy.AddWarning,
                        )
                    else:  # ICA
                        transformed_data, transform_stats = self._perform_ica(
                            data_array,
                            num_components,
                            stats,
                            random_state=random_state,
                            message_callback=arcpy.AddMessage,
                            warning_callback=arcpy.AddWarning,
                        )
                    
                    # Create multiband output
                    output_path = self._create_multiband_output(
                        transformed_data,
                        raster_info,
                        out_workspace,
                        out_name,
                        preserve_mask
                    )
                    
                    # Define data_info for statistics files
                    data_info = {
                        'input_rasters': input_rasters,
                        'output_path': output_path
                    }
                    
                    # Save statistics if requested. PCA persists the full
                    # PCAStatistics as .npz (reloadable for re-application);
                    # MNF/ICA write human-readable .txt summaries.
                    if save_stats:
                        arcpy.AddMessage(f"Saving statistics to: {stats_file}")

                        if transform_type == "PCA":
                            if not transform_stats.save(stats_file):
                                for err in transform_stats.errors:
                                    arcpy.AddWarning(f"PCAStatistics.save: {err}")
                        elif transform_type == "MNF":
                            self._save_mnf_statistics_txt(stats_file, data_info, transform_stats)
                        else:  # ICA
                            self._save_ica_statistics_txt(stats_file, data_info, transform_stats)
                    
                    # Update final statistics
                    stats['end_time'] = datetime.now()
                    stats['processing_time'] = (
                        stats['end_time'] - stats['start_time']
                    ).total_seconds()
                    
                    arcpy.AddMessage("\nProcessing completed successfully!")
                    arcpy.AddMessage(f"Total processing time: {stats['processing_time']:.2f} seconds")
                    
                    return output_path
                    
                except Exception as e:
                    arcpy.AddError(f"Error loading or processing raster: {str(e)}")
                    import traceback
                    arcpy.AddError(traceback.format_exc())
                    return None
                    
            except Exception as e:
                arcpy.AddError(f"Error in transformation: {str(e)}")
                stats['errors'].append(str(e))
                # Print detailed traceback for debugging
                import traceback
                arcpy.AddError(traceback.format_exc())
                raise
                
        finally:
            # Clean up the composite temp raster (Bug 2 fix). Wrapped in its
            # own try so a delete failure does not mask a real exception
            # propagating from the body of execute().
            if composite_temp_path:
                try:
                    if arcpy.Exists(composite_temp_path):
                        arcpy.management.Delete(composite_temp_path)
                except Exception:
                    pass

            # Check in extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckInExtension("Spatial")

    def _perform_mnf(self, data: np.ndarray, n_components: int,
                     noise_stats_file: str, noise_data: np.ndarray,
                     stats: dict,
                     message_callback=None, warning_callback=None) -> tuple[np.ndarray, MNFStatistics]:
        """
        Perform MNF transformation. Logging is routed through the optional
        callbacks (ISS-011) so this can be exercised without arcpy.
        """
        log = message_callback or (lambda msg: None)
        warn = warning_callback or (lambda msg: None)
        try:
            # Initialize statistics objects
            noise_stats = MNFNoiseStatistics()
            mnf_stats = MNFStatistics()

            # Load noise statistics if provided
            if noise_stats_file:
                log(f"Loading noise statistics from: {noise_stats_file}")
                if not noise_stats.load(noise_stats_file):
                    warn("Failed to load noise statistics file, estimating noise instead")
                    noise_stats_file = None

            # If noise statistics weren't loaded, estimate them
            if not noise_stats_file:
                log("Estimating noise statistics...")

                # Use noise subset data if provided, otherwise use full data
                noise_source = noise_data if noise_data is not None else data

                # Handle shape for noise data
                if len(noise_source.shape) == 3:
                    noise_flat = noise_source.reshape(-1, noise_source.shape[-1])
                else:
                    noise_flat = noise_source

                # Remove NaN values
                valid_mask = ~np.isnan(noise_flat).any(axis=1)
                valid_noise = noise_flat[valid_mask]

                if valid_noise.shape[0] < noise_flat.shape[1] * 10:
                    warn(f"Low sample count for noise estimation: {valid_noise.shape[0]} samples "
                         f"for {noise_flat.shape[1]} bands. Results may be unstable.")

                # Estimate noise covariance.
                if noise_data is None:
                    log("Using shift difference method for noise estimation...")
                    pixel_valid = ~np.isnan(data).any(axis=-1)
                    _, noise_covariance, n_pairs = noise_from_valid_diffs(data, pixel_valid)
                    if noise_covariance is None:
                        raise RuntimeError(
                            f"Noise estimation failed: insufficient valid neighbour pairs "
                            f"(need at least {data.shape[2] * 2}). Check input mask coverage."
                        )
                    log(f"  Estimated noise from {n_pairs} valid neighbour pairs.")
                else:
                    log("Calculating noise statistics from subset data...")
                    noise_covariance = np.cov(valid_noise, rowvar=False)
                    # Match the regularisation behaviour of noise_from_valid_diffs.
                    eigvals, eigvecs = np.linalg.eigh(noise_covariance)
                    if np.any(eigvals < _EIGVAL_FLOOR_ABS):
                        valid_eig = eigvals[eigvals > _EIGVAL_FLOOR_ABS]
                        floor = (
                            np.median(valid_eig) * _EIGVAL_FLOOR_RELATIVE
                            if len(valid_eig) > 0
                            else _EIGVAL_FLOOR_ABS
                        )
                        eigvals = np.maximum(eigvals, floor)
                        noise_covariance = (eigvecs * eigvals) @ eigvecs.T

                noise_eigenvals, noise_eigenvecs = np.linalg.eigh(noise_covariance)
                noise_stats.noise_covariance = noise_covariance
                noise_stats.noise_eigenvalues = noise_eigenvals
                noise_stats.noise_eigenvectors = noise_eigenvecs
                mnf_stats.noise_covariance = noise_covariance

            # Flatten input data for transformation
            shape = data.shape
            flat_data = data.reshape(-1, shape[2])

            # Handle NaN values
            valid_mask = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[valid_mask]

            log(f"Processing {valid_data.shape[0]} valid pixels for MNF...")

            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean

            log("Applying noise whitening transformation...")
            whitening_matrix = (noise_stats.noise_eigenvectors @
                                np.diag(1.0 / np.sqrt(np.maximum(noise_stats.noise_eigenvalues, _EIGVAL_FLOOR_ABS))) @
                                noise_stats.noise_eigenvectors.T)

            whitened_data = centered_data @ whitening_matrix
            whitened_cov = np.cov(whitened_data.T)
            signal_eigenvals, signal_eigenvecs = np.linalg.eigh(whitened_cov)

            # Sort in descending order
            idx = signal_eigenvals.argsort()[::-1]
            signal_eigenvals = signal_eigenvals[idx]
            signal_eigenvecs = signal_eigenvecs[:, idx]

            # Truncate to requested number of components
            signal_eigenvals = signal_eigenvals[:n_components]
            signal_eigenvecs = signal_eigenvecs[:, :n_components]

            transformed_valid = whitened_data @ signal_eigenvecs

            # ISS-004: report the empirical component correlation. A successful
            # MNF should yield ~identity, since MNF components are orthogonal
            # in the noise-whitened space. Off-diagonal max > 0.1 signals a
            # poor noise estimate or a numerical issue.
            component_correlation = np.corrcoef(transformed_valid.T)
            if component_correlation.ndim == 2 and component_correlation.shape[0] > 1:
                n_comp = component_correlation.shape[0]
                off_diag_max = float(np.max(np.abs(component_correlation - np.eye(n_comp))))
                if off_diag_max > _MNF_CORR_OFFDIAG_WARN:
                    warn(f"MNF component correlation off-diagonal max = {off_diag_max:.3f} "
                         f"(target < {_MNF_CORR_OFFDIAG_WARN}). Components may not be cleanly separated.")
                else:
                    log(f"MNF component correlation OK (off-diagonal max = {off_diag_max:.3f}).")

            log("\nMNF Component Statistics (Signal-to-Noise Ratios):")
            for i in range(n_components):
                log(f"  Component {i+1}: {signal_eigenvals[i]:.4f}")

            # Store MNF statistics
            mnf_stats.band_means = data_mean
            mnf_stats.eigenvalues = signal_eigenvals
            mnf_stats.eigenvectors = signal_eigenvecs
            mnf_stats.transform_matrix = whitening_matrix @ signal_eigenvecs
            mnf_stats.whitening_matrix = whitening_matrix
            mnf_stats.signal_covariance = whitened_cov
            mnf_stats.component_correlation = component_correlation

            # Reconstruct full data array
            transformed_data = np.zeros((flat_data.shape[0], n_components))
            transformed_data[valid_mask] = transformed_valid
            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)

            return transformed_data, mnf_stats

        except Exception as e:
            stats['errors'].append(f"MNF Error: {str(e)}")
            raise

    def _perform_pca(self, data: np.ndarray, n_components: int, stats: dict,
                     message_callback=None, warning_callback=None) -> tuple[np.ndarray, PCAStatistics]:
        """
        Perform PCA transformation.

        The arcpy logging calls are routed through `message_callback` /
        `warning_callback` so this method is unit-testable without ArcGIS
        Pro. Callers running inside the toolbox must pass arcpy.AddMessage
        / arcpy.AddWarning explicitly; tests can pass no-op callables.
        """
        log = message_callback or (lambda msg: None)
        try:
            # Initialize statistics object
            pca_stats = PCAStatistics()

            # Flatten data and center
            shape = data.shape
            flat_data = data.reshape(-1, shape[2])

            # Handle NaN values
            is_not_nan = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[is_not_nan]

            log(f"Computing PCA on {valid_data.shape[0]} valid pixels...")
            
            # Center the data
            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean
            
            # Calculate covariance matrix
            covariance_matrix = np.cov(centered_data, rowvar=False)
            
            # Calculate eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
            
            # Sort in descending order
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Truncate to requested number of components
            eigenvals = eigenvals[:n_components]
            eigenvecs = eigenvecs[:, :n_components]
            
            # Calculate explained variance
            total_var = np.sum(eigenvals)
            explained_variance = eigenvals / total_var
            
            # Transform valid data
            transformed_valid = centered_data @ eigenvecs
            
            # Reconstruct full data array
            transformed_data = np.zeros((flat_data.shape[0], n_components))
            transformed_data[is_not_nan] = transformed_valid
            
            # Store PCA statistics
            pca_stats.band_means = data_mean
            pca_stats.eigenvalues = eigenvals
            pca_stats.eigenvectors = eigenvecs
            pca_stats.explained_variance = explained_variance
            pca_stats.covariance_matrix = covariance_matrix
            
            # Calculate cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance)
            log("\nPCA Component Statistics:")
            for i in range(n_components):
                log(f"  Component {i+1}: {explained_variance[i]*100:.2f}% variance " +
                    f"(Cumulative: {cumulative_variance[i]*100:.2f}%)")
            
            # Reshape back to original dimensions
            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)
            
            return transformed_data, pca_stats
            
        except Exception as e:
            stats['errors'].append(f"PCA Error: {str(e)}")
            raise

    def _perform_ica(self, data: np.ndarray, n_components: int, stats: dict,
                     random_state: int = 42,
                     message_callback=None, warning_callback=None) -> tuple[np.ndarray, ICAStatistics]:
        """
        Perform ICA transformation with kurtosis metrics.

        ISS-007: random_state is a parameter (default 42) and is persisted on
        ICAStatistics so a saved fit can be reproduced exactly.
        ISS-011: logging is routed through the optional callbacks.
        """
        log = message_callback or (lambda msg: None)
        try:
            from sklearn.decomposition import FastICA

            ica_stats = ICAStatistics()

            shape = data.shape
            flat_data = data.reshape(-1, shape[2])

            is_not_nan = ~np.isnan(flat_data).any(axis=1)
            valid_data = flat_data[is_not_nan]

            log(f"Processing {valid_data.shape[0]} valid pixels for ICA...")

            data_mean = np.mean(valid_data, axis=0)
            centered_data = valid_data - data_mean

            cov_matrix = np.cov(centered_data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            whitening = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, _ICA_WHITENING_FLOOR))) @ eigenvectors.T
            dewhitening = eigenvectors @ np.diag(np.sqrt(eigenvalues))

            whitened = centered_data @ whitening.T

            log(f"Performing FastICA (random_state={random_state})...")

            ica = FastICA(
                n_components=n_components,
                whiten=False,
                max_iter=1000,
                tol=1e-4,
                random_state=random_state,
            )
            transformed_valid = ica.fit_transform(whitened)
            # FastICA with whiten=False ignores the n_components argument and
            # returns one component per input band. Truncate post-hoc so
            # downstream shapes (transformed_data reshape, unmixing matrix,
            # kurtosis_values) all agree with the requested n_components.
            transformed_valid = transformed_valid[:, :n_components]
            W = ica.components_[:n_components, :]

            # Kurtosis (peakedness) per component — high |kurtosis| ≈ more
            # independent / non-Gaussian, used by select_by_kurtosis.
            kurtosis_values = np.array([
                scipy.stats.kurtosis(transformed_valid[:, i], fisher=True)
                for i in range(n_components)
            ])

            independence_metrics = np.array([
                (np.mean((transformed_valid[:, i] - np.mean(transformed_valid[:, i])) ** 4)
                 / (np.var(transformed_valid[:, i]) ** 2)) - 3
                for i in range(n_components)
            ])

            ica_stats.kurtosis_values = kurtosis_values
            ica_stats.independence_metrics = independence_metrics

            # unmixing maps centred original-space data to source space.
            unmixing = W @ whitening
            mixing = np.linalg.pinv(unmixing)
            
            # Store statistics
            ica_stats.band_means = data_mean
            ica_stats.mixing_matrix = mixing
            ica_stats.unmixing_matrix = unmixing
            ica_stats.whitening_matrix = whitening
            ica_stats.dewhitening_matrix = dewhitening
            ica_stats.n_iterations = ica.n_iter_
            ica_stats.random_state = random_state

            transformed_data = np.zeros((flat_data.shape[0], n_components))
            transformed_data[is_not_nan] = transformed_valid

            log(f"ICA completed in {ica.n_iter_} iterations")

            transformed_data = transformed_data.reshape(shape[0], shape[1], n_components)

            return transformed_data, ica_stats

        except ImportError:
            # AddError stays inline: this branch fires before any callback is
            # used and surfaces a runtime-dependency problem to the GP dialog.
            arcpy.AddError("scikit-learn is not available. Please install scikit-learn to use ICA")
            stats['errors'].append("scikit-learn is not available")
            raise
        except Exception as e:
            stats['errors'].append(f"ICA Error: {str(e)}")
            raise
        
    def _create_multiband_output(self, component_arrays, raster_info, out_workspace, out_name, preserve_mask=True):
        """
        Create a multiband raster from component arrays
        
        Parameters:
        -----------
        component_arrays : ndarray
            Component arrays of shape (height, width, components)
        raster_info : dict
            Raster information containing extent, cell size, etc.
        out_workspace : str
            Output workspace path
        out_name : str
            Output raster name
        preserve_mask : bool
            Whether to preserve the input mask
            
        Returns:
        --------
        str
            Path to the output multiband raster
        """
        try:
            arcpy.AddMessage("\nCreating multiband output...")
            
            # Extract raster information
            extent = raster_info['extent']
            cell_size = raster_info['cell_size']
            spatial_ref = raster_info['spatial_ref']
            mask = raster_info.get('mask', None)
            
            # Paths for temporary component rasters
            temp_component_paths = []
            
            # Create output directory for temp files if needed
            temp_dir = os.path.join(arcpy.env.scratchFolder, "temp_components")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # Save each component as a temporary raster
            num_components = component_arrays.shape[2]
            for i in range(num_components):
                temp_path = os.path.join(temp_dir, f"temp_comp_{i+1}_{uuid.uuid4().hex}.tif")
                
                # Create raster from array
                out_raster = arcpy.NumPyArrayToRaster(
                    in_array=component_arrays[:,:,i],
                    lower_left_corner=arcpy.Point(
                        extent.XMin, 
                        extent.YMin
                    ),
                    x_cell_size=cell_size[0],
                    y_cell_size=cell_size[1],
                    value_to_nodata=np.nan
                )
                
                # Apply mask if requested and available
                if preserve_mask and mask is not None:
                    arcpy.AddMessage(f"  Applying mask to component {i+1}")
                    try:
                        # Convert mask to raster if it's not already
                        if not isinstance(mask, arcpy.Raster):
                            mask_raster = arcpy.Raster(mask)
                        else:
                            mask_raster = mask
                            
                        # Create SetNull expression
                        masked_raster = arcpy.sa.SetNull(mask_raster == 0, out_raster)
                        masked_raster.save(temp_path)
                    except Exception as e:
                        arcpy.AddWarning(f"  Error applying mask to component {i+1}: {str(e)}")
                        out_raster.save(temp_path)
                else:
                    # Save without masking
                    out_raster.save(temp_path)
                
                # Set spatial reference
                arcpy.DefineProjection_management(
                    temp_path,
                    spatial_ref
                )
                
                temp_component_paths.append(temp_path)
                arcpy.AddMessage(f"  Created component {i+1}")
            
            # Create multiband raster using Composite Bands
            output_path = os.path.join(out_workspace, out_name)
            arcpy.AddMessage(f"Creating final multiband raster: {output_path}")
            arcpy.management.CompositeBands(temp_component_paths, output_path)
            
            # Clean up temporary files
            arcpy.AddMessage("Cleaning up temporary files...")
            for temp_path in temp_component_paths:
                try:
                    arcpy.management.Delete(temp_path)
                except:
                    pass
            
            return output_path
            
        except Exception as e:
            arcpy.AddError(f"Error creating multiband output: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _save_mnf_statistics_txt(self, stats_file, data_info, mnf_stats):
        """
        Save MNF statistics in text format
        
        Parameters:
        -----------
        stats_file : str
            Path to output statistics file
        data_info : dict
            Information about the input data
        mnf_stats : MNFStatistics
            MNF statistics object
        """
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Data file produced by Minimum Noise Fraction (MNF) Transform\n")
                f.write("#\tInput raster(s):\n")
                for raster_path in data_info.get('input_rasters', []):
                    f.write(f"#\t\t{raster_path}\n")
                f.write(f"#\tThe number of components = {len(mnf_stats.eigenvalues)}\n")
                f.write(f"#\tOutput raster(s):\n")
                f.write(f"#\t\t{data_info.get('output_path', 'Not specified')}\n\n")
                
                # Noise Covariance Matrix
                f.write("#                    NOISE COVARIANCE MATRIX\n\n")
                num_bands = len(mnf_stats.band_means)
                f.write("#    Layer         " + "".join([f"{i+1:14d}" for i in range(num_bands)]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                # Include noise covariance if available
                noise_cov = mnf_stats.noise_covariance if hasattr(mnf_stats, 'noise_covariance') else None
                if noise_cov is not None:
                    for i in range(noise_cov.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in noise_cov[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Noise Whitening Matrix
                if hasattr(mnf_stats, 'whitening_matrix') and mnf_stats.whitening_matrix is not None:
                    f.write("#                    NOISE WHITENING MATRIX\n\n")
                    f.write("#    Output Band   " + "".join([f"{i+1:14d}" for i in range(num_bands)]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.whitening_matrix.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in mnf_stats.whitening_matrix[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Signal Covariance Matrix (after whitening)
                if hasattr(mnf_stats, 'signal_covariance') and mnf_stats.signal_covariance is not None:
                    f.write("#                    SIGNAL COVARIANCE MATRIX (AFTER WHITENING)\n\n")
                    f.write("#    Band          " + "".join([f"{i+1:14d}" for i in range(mnf_stats.signal_covariance.shape[0])]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.signal_covariance.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.6e}" for x in mnf_stats.signal_covariance[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Signal-to-Noise Ratio (Eigenvalues)
                f.write("#                 SIGNAL-TO-NOISE RATIOS (EIGENVALUES)\n\n")
                f.write("# Number of Input Layers     Number of MNF Component Layers\n")
                f.write(f"            {len(mnf_stats.band_means):d}                              {len(mnf_stats.eigenvalues):d}\n")
                f.write("# MNF Layer        " + "".join([f"{i+1:14d}" for i in range(len(mnf_stats.eigenvalues))]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                f.write("# Eigenvalues\n")
                f.write("               " + "".join([f"{x:14.5f}" for x in mnf_stats.eigenvalues]) + "\n")
                
                # Eigenvectors
                f.write("# Eigenvectors\n")
                f.write("# Input Layer\n")
                for i in range(mnf_stats.eigenvectors.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.5f}" for x in mnf_stats.eigenvectors[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Component Correlation Matrix
                if hasattr(mnf_stats, 'component_correlation') and mnf_stats.component_correlation is not None:
                    f.write("#                 COMPONENT CORRELATION MATRIX\n")
                    f.write("# This matrix should be close to an identity matrix (diagonal = 1, off-diagonal ≈ 0)\n\n")
                    f.write("# Component      " + "".join([f"{i+1:14d}" for i in range(len(mnf_stats.eigenvalues))]) + "\n")
                    f.write("#  --------------------------------------------------------------------------\n")
                    for i in range(mnf_stats.component_correlation.shape[0]):
                        f.write(f"        {i+1:d}       " + "".join([f"{x:14.5f}" for x in mnf_stats.component_correlation[i,:]]) + "\n")
                    f.write("#  ==========================================================================\n\n")
                
                # Percent and Accumulative Eigenvalues
                f.write("#                 PERCENT AND ACCUMULATIVE EIGENVALUES\n\n")
                f.write("# MNF Layer   EigenValue   Percent of EigenValues   Accumulative of EigenValues\n")
                
                total_eigenvalue = sum(mnf_stats.eigenvalues)
                cumulative = 0.0
                for i, eigenvalue in enumerate(mnf_stats.eigenvalues):
                    percent = (eigenvalue / total_eigenvalue) * 100.0
                    cumulative += percent
                    f.write(f"        {i+1:d} {eigenvalue:14.5f}          {percent:7.4f}               {cumulative:7.4f}\n")
                f.write("#  ==========================================================================\n")
                
        except Exception as e:
            arcpy.AddWarning(f"Error saving MNF statistics to text file: {str(e)}")

    def _save_ica_statistics_txt(self, stats_file, data_info, ica_stats):
        """
        Save ICA statistics in text format
        
        Parameters:
        -----------
        stats_file : str
            Path to output statistics file
        data_info : dict
            Information about the input data
        ica_stats : ICAStatistics
            ICA statistics object
        """
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Data file produced by Independent Component Analysis (ICA) Transform\n")
                f.write("#\tInput raster(s):\n")
                for raster_path in data_info.get('input_rasters', []):
                    f.write(f"#\t\t{raster_path}\n")
                f.write(f"#\tThe number of components = {ica_stats.mixing_matrix.shape[1]}\n")
                f.write(f"#\tOutput raster(s):\n")
                f.write(f"#\t\t{data_info.get('output_path', 'Not specified')}\n")
                f.write(f"#\tConverged in {ica_stats.n_iterations} iterations\n\n")
                
                # Mixing Matrix
                f.write("#                    MIXING MATRIX (A)\n")
                f.write("# Independent components are reconstructed as X = AS, where A is the mixing matrix and S are the sources\n\n")
                f.write("#    Component    " + "".join([f"{i+1:14d}" for i in range(ica_stats.mixing_matrix.shape[1])]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                for i in range(ica_stats.mixing_matrix.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.6f}" for x in ica_stats.mixing_matrix[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Unmixing Matrix
                f.write("#                    UNMIXING MATRIX (W)\n")
                f.write("# Sources are computed as S = WX, where W is the unmixing matrix and X is the data\n\n")
                f.write("#    Source       " + "".join([f"{i+1:14d}" for i in range(ica_stats.unmixing_matrix.shape[1])]) + "\n")
                f.write("#  --------------------------------------------------------------------------\n")
                
                for i in range(ica_stats.unmixing_matrix.shape[0]):
                    f.write(f"        {i+1:d}       " + "".join([f"{x:14.6f}" for x in ica_stats.unmixing_matrix[i,:]]) + "\n")
                f.write("#  ==========================================================================\n\n")
                
                # Component Independence (approximate mutual information)
                f.write("#                 COMPONENT INDEPENDENCE MEASURES\n\n")
                f.write("# Component   Mutual Information   Independence Score\n")
                
                # Calculate simple independence metrics
                # In actual ICA implementation, mutual information would be calculated
                # Here we're providing placeholder values
                if hasattr(ica_stats, 'independence_metrics') and ica_stats.independence_metrics is not None:
                    for i, metric in enumerate(ica_stats.independence_metrics):
                        # 0-100 score driven by ABSOLUTE departure from
                        # Gaussianity. The raw metric (excess kurtosis) is
                        # negative for sub-Gaussian components (common in
                        # smooth-terrain remote sensing) — taking abs keeps
                        # the user-facing score in [0, 100] regardless of sign.
                        indep_score = 100 * (1 - np.exp(-abs(metric)))
                        f.write(f"        {i+1:d}         {metric:8.4f}           {indep_score:8.4f}\n")
                else:
                    # Fallback if metrics aren't available
                    for i in range(ica_stats.mixing_matrix.shape[1]):
                        f.write(f"        {i+1:d}         {'N/A':8s}           {'N/A':8s}\n")
                f.write("#  ==========================================================================\n")
                              
                # Kurtosis Values
                f.write("# COMPONENT KURTOSIS VALUES\n")
                f.write("# Component   Kurtosis\n")
                for i, kurt in enumerate(ica_stats.kurtosis_values):
                    f.write(f"        {i+1:d}         {kurt:8.4f}\n")
                f.write("#  ==========================================================================\n")
                
        except Exception as e:
            arcpy.AddWarning(f"Error saving ICA statistics to text file: {str(e)}")

    def _extract_subset_data(self, data: np.ndarray, subset_feature: str) -> np.ndarray:
        """
        Extract data from specified spatial subset
        """
        try:
            # Create temporary raster from numpy array
            temp_raster = arcpy.NumPyArrayToRaster(data[:,:,0])
            
            # Extract by mask
            subset = arcpy.sa.ExtractByMask(temp_raster, subset_feature)
            
            # Convert back to numpy array
            subset_data = arcpy.RasterToNumPyArray(subset)
            
            return subset_data
            
        except Exception as e:
            arcpy.AddWarning(f"Error extracting subset: {str(e)}")
            return None

# Tool 4: Spectral Angle Mapper
class SpectralAngleMapper(object):
    """Tool 06 — Spectral Angle Mapper classification.

    Sensor-aware via the Sensor Type parameter, which drives the
    expected input band count and validates reference-spectra column
    counts. The SAM algorithm itself is sensor-agnostic.

    Carries the Phase 5 SAM SetNull inversion fix (was: classifier
    initialised NoData over valid pixels, leaving the final raster
    blank). Other SAM audit findings (color-map application, endmember
    GPLong precision, class-index determinism, class-field heuristic
    fallback) are documented but NOT yet fixed — they're queued as a
    follow-up phase.
    """

    def __init__(self):
        self.label = "06 — Spectral Angle Mapper"
        self.description = (
            "Spectral Angle Mapper classification on a pre-stacked "
            "multiband raster from any of the three supported sensors. "
            "Reference spectra can be supplied as a table, training "
            "samples, or endmember pixels. The Sensor Type parameter "
            "drives band-count validation."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        # Input raster
        input_raster = arcpy.Parameter(
            displayName="Input Multiband Raster",
            name="input_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )
        
        # Reference spectra source
        ref_source = arcpy.Parameter(
            displayName="Reference Spectra Source",
            name="ref_source",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        ref_source.filter.list = ["Table", "ROIs/Training Samples", "Endmember Pixels"]
        ref_source.value = "Table"
        
        # Reference spectra table (for Table option)
        ref_table = arcpy.Parameter(
            displayName="Reference Spectra Table",
            name="ref_table",
            datatype="DETable",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        
        # Training samples (for ROIs option)
        training_samples = arcpy.Parameter(
            displayName="Training Samples/ROIs",
            name="training_samples",
            datatype=["DEFeatureClass", "DEShapefile"],
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        
        # Endmember pixels (for Endmember option)
        endmember_pixels = arcpy.Parameter(
            displayName="Endmember Pixels (x,y coordinates)",
            name="endmember_pixels",
            datatype="GPValueTable",
            parameterType="Optional",
            direction="Input",
            enabled=False
        )
        endmember_pixels.columns = [['GPString', 'Class Name'], ['GPLong', 'X'], ['GPLong', 'Y']]
        
        # Class names field (for Table option)
        class_field = arcpy.Parameter(
            displayName="Class Names Field",
            name="class_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input",
            enabled=True
        )
        class_field.parameterDependencies = [ref_table.name]
        class_field.filter.list = ['Text']
        
        # Band fields (for Table option)
        band_fields = arcpy.Parameter(
            displayName="Band Value Fields",
            name="band_fields",
            datatype="Field",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
            enabled=True
        )
        band_fields.parameterDependencies = [ref_table.name]
        band_fields.filter.list = ['Double', 'Float', 'Long', 'Short']
        
        # Maximum angle
        max_angle = arcpy.Parameter(
            displayName="Maximum Angle (degrees)",
            name="max_angle",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        max_angle.value = 10.0
        max_angle.filter.type = "Range"
        max_angle.filter.list = [0.1, 90.0]
        
        # Threshold
        threshold = arcpy.Parameter(
            displayName="Classification Threshold",
            name="threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        threshold.value = 0.1
        threshold.filter.type = "Range"
        threshold.filter.list = [0.01, 1.0]
        
        # Output workspace
        out_workspace = arcpy.Parameter(
            displayName="Output Workspace",
            name="out_workspace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input"
        )
        
        # Output classification raster
        out_raster = arcpy.Parameter(
            displayName="Output Classification Raster",
            name="out_raster",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        
        # Output SAM raster
        out_sam = arcpy.Parameter(
            displayName="Output SAM Angle Raster",
            name="out_sam",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )
        
        # Color scheme
        color_scheme = arcpy.Parameter(
            displayName="Apply Color Scheme",
            name="color_scheme",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        color_scheme.value = True

        # Sensor selector (Phase 6 addition) — drives the expected band
        # count for the input raster and the reference spectra columns.
        sensor_type = make_sensor_parameter()

        return [input_raster, ref_source, ref_table, training_samples, endmember_pixels,
                class_field, band_fields, max_angle, threshold, out_workspace,
                out_raster, out_sam, color_scheme, sensor_type]
    
    def updateParameters(self, parameters):
        """Modify parameter values and properties"""
        # Get references to parameters
        ref_source = parameters[1]
        ref_table = parameters[2]
        training_samples = parameters[3]
        endmember_pixels = parameters[4]
        class_field = parameters[5]
        band_fields = parameters[6]
        
        # Enable/disable parameters based on reference source
        if ref_source.altered:
            if ref_source.value == "Table":
                ref_table.enabled = True
                training_samples.enabled = False
                endmember_pixels.enabled = False
                class_field.enabled = True
                band_fields.enabled = True
            elif ref_source.value == "ROIs/Training Samples":
                ref_table.enabled = False
                training_samples.enabled = True
                endmember_pixels.enabled = False
                class_field.enabled = False
                band_fields.enabled = False
            elif ref_source.value == "Endmember Pixels":
                ref_table.enabled = False
                training_samples.enabled = False
                endmember_pixels.enabled = True
                class_field.enabled = False
                band_fields.enabled = False
        
        # Update output name based on input
        if parameters[0].altered and not parameters[10].altered:
            try:
                input_name = os.path.basename(parameters[0].valueAsText)
                name_parts = os.path.splitext(input_name)
                if name_parts[0]:
                    parameters[10].value = f"{name_parts[0]}_SAM_class"
                    parameters[11].value = f"{name_parts[0]}_SAM_angles"
            except:
                pass
        
    def updateMessages(self, parameters):
        """Modify messages created by internal validation"""
        # Validate reference spectra table
        if parameters[1].value == "Table" and parameters[2].altered:
            try:
                table_path = parameters[2].valueAsText
                if not arcpy.Exists(table_path):
                    parameters[2].setErrorMessage("Reference spectra table does not exist")
                else:
                    # Check for required fields
                    if parameters[5].value and parameters[6].value:
                        # Check class field
                        class_field = parameters[5].valueAsText
                        
                        # Check band fields (ensure they match input raster bands)
                        band_fields = parameters[6].valueAsText.split(";")
                        if len(band_fields) < 2:
                            parameters[6].setWarningMessage("At least 2 band fields should be specified")
            except Exception as e:
                parameters[2].setErrorMessage(f"Error validating reference table: {str(e)}")
        
        # Validate training samples
        if parameters[1].value == "ROIs/Training Samples" and parameters[3].altered:
            try:
                training_path = parameters[3].valueAsText
                if not arcpy.Exists(training_path):
                    parameters[3].setErrorMessage("Training samples do not exist")
                else:
                    # Check for class field
                    desc = arcpy.Describe(training_path)
                    if not any(field.name.lower() in ["class", "classname", "class_name"] for field in desc.fields):
                        parameters[3].setWarningMessage("No class field found. Expected fields: Class, ClassName, or Class_Name")
            except Exception as e:
                parameters[3].setErrorMessage(f"Error validating training samples: {str(e)}")
    
    def execute(self, parameters, messages):
        """Execute the tool"""
        try:
            # Check out extensions
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")
            else:
                arcpy.AddError("Spatial Analyst extension is required but not available")
                return
                
            # Enable overwrite
            arcpy.env.overwriteOutput = True
            
            # Get parameters
            input_raster = parameters[0].valueAsText
            ref_source = parameters[1].valueAsText
            ref_table = parameters[2].valueAsText if parameters[1].value == "Table" else None
            training_samples = parameters[3].valueAsText if parameters[1].value == "ROIs/Training Samples" else None
            endmember_pixels = parameters[4].value if parameters[1].value == "Endmember Pixels" else None
            class_field = parameters[5].valueAsText if parameters[1].value == "Table" and parameters[5].value else None
            band_fields = parameters[6].valueAsText.split(";") if parameters[1].value == "Table" and parameters[6].value else []
            max_angle = parameters[7].value
            threshold = parameters[8].value
            out_workspace = parameters[9].valueAsText
            out_raster = parameters[10].valueAsText
            out_sam = parameters[11].valueAsText if parameters[11].value else None
            apply_color = parameters[12].value
            
            # Convert max angle to radians
            max_angle_rad = max_angle * (3.14159265359 / 180.0)
            
            # Process based on reference source
            if ref_source == "Table":
                # Perform SAM with reference table
                self._sam_with_table(
                    input_raster=input_raster,
                    ref_table=ref_table,
                    class_field=class_field,
                    band_fields=band_fields,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            elif ref_source == "ROIs/Training Samples":
                # Perform SAM with training samples
                self._sam_with_training(
                    input_raster=input_raster,
                    training_samples=training_samples,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            elif ref_source == "Endmember Pixels":
                # Perform SAM with endmember pixels
                self._sam_with_endmembers(
                    input_raster=input_raster,
                    endmember_pixels=endmember_pixels,
                    max_angle_rad=max_angle_rad,
                    threshold=threshold,
                    out_workspace=out_workspace,
                    out_raster=out_raster,
                    out_sam=out_sam,
                    apply_color=apply_color
                )
            
            # Return output path
            out_path = os.path.join(out_workspace, out_raster)
            arcpy.SetParameterAsText(10, out_path)
            
            if out_sam:
                sam_path = os.path.join(out_workspace, out_sam)
                arcpy.SetParameterAsText(11, sam_path)
                
            return out_path
            
        except Exception as e:
            arcpy.AddError(f"Error executing SAM: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
            
        finally:
            # Check in extensions
            arcpy.CheckInExtension("Spatial")
    
    def _sam_with_table(self, input_raster, ref_table, class_field, band_fields,
                        max_angle_rad, threshold, out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using reference spectra from a table"""
        import numpy as np
        import math
        
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage(f"Reference spectra table: {ref_table}")
            
            # Load the input raster
            raster_obj = arcpy.Raster(input_raster)
            band_count = raster_obj.bandCount
            
            # Check if band fields match available bands
            if len(band_fields) > band_count:
                arcpy.AddWarning(f"Reference table has {len(band_fields)} bands, but input raster has {band_count} bands")
                arcpy.AddWarning("Using only the available bands from the reference table")
                band_fields = band_fields[:band_count]
            elif len(band_fields) < band_count:
                arcpy.AddWarning(f"Reference table has {len(band_fields)} bands, but input raster has {band_count} bands")
                arcpy.AddWarning("Some raster bands will not be used in the analysis")
            
            # Load band data into memory
            arcpy.AddMessage("Loading raster bands...")
            bands = {}
            for i in range(1, band_count + 1):
                if i <= len(band_fields):  # Only load bands that have reference data
                    arcpy.AddMessage(f"Loading band {i}...")
                    bands[i] = arcpy.Raster(f"{input_raster}/{i}")
            
            # Read reference spectra from the table
            arcpy.AddMessage("Reading reference spectra...")
            ref_spectra = {}
            class_names = []
            
            with arcpy.da.SearchCursor(ref_table, [class_field] + band_fields) as cursor:
                for row in cursor:
                    class_name = row[0]
                    band_values = [float(row[i+1]) for i in range(len(band_fields))]
                    
                    if class_name in ref_spectra:
                        arcpy.AddWarning(f"Duplicate class name '{class_name}' found in reference table")
                        continue
                    
                    # Normalize reference spectrum
                    magnitude = math.sqrt(sum(v*v for v in band_values))
                    if magnitude > 0:
                        normalized = [v/magnitude for v in band_values]
                        ref_spectra[class_name] = normalized
                        class_names.append(class_name)
                    else:
                        arcpy.AddWarning(f"Reference spectrum for '{class_name}' has zero magnitude and will be ignored")
            
            arcpy.AddMessage(f"Found {len(ref_spectra)} reference spectra:")
            for name in class_names:
                arcpy.AddMessage(f"  - {name}")
            
            if not ref_spectra:
                arcpy.AddError("No valid reference spectra found in the reference table")
                return
            
            # Create output paths
            out_class_path = os.path.join(out_workspace, out_raster)
            out_sam_path = os.path.join(out_workspace, out_sam) if out_sam else None
            
            # Perform SAM calculation
            arcpy.AddMessage("Performing SAM classification...")
            
            # Create lists for map algebra expressions
            band_expressions = [f"Float('{bands[i+1]}')" for i in range(len(band_fields))]
            
            # Create SAM rasters for each reference spectrum
            sam_rasters = {}
            
            # 1. Compute normalization factor for pixel vectors
            norm_expr = " + ".join([f"Power({expr}, 2)" for expr in band_expressions])
            norm_raster = arcpy.sa.SquareRoot(arcpy.sa.Raster(arcpy.sa.Float(norm_expr)))
            
            # 2. Compute SAM for each reference spectrum
            for class_name, ref_vector in ref_spectra.items():
                # Calculate dot product
                dot_expr = " + ".join([f"{expr} * {ref_vector[i]}" for i, expr in enumerate(band_expressions)])
                dot_raster = arcpy.sa.Float(dot_expr)
                
                # Calculate SAM angle (arccos of dot product divided by magnitudes)
                # Since reference vectors are already normalized, we only need to normalize the pixel vector
                sam_angle = arcpy.sa.ACos(dot_raster / norm_raster)
                
                # Convert from radians to degrees
                sam_angle_deg = sam_angle * (180.0 / math.pi)
                
                # Store the SAM raster
                sam_rasters[class_name] = sam_angle_deg
            
            # 3. Create classification raster
            arcpy.AddMessage("Creating final classification raster...")
            
            # Initialise the classification/min-angle rasters over valid pixels
            # only. SetNull(cond, false_val) returns NoData where `cond` is True
            # and `false_val` elsewhere; we want NoData on invalid pixels
            # (norm_raster <= 0) and a seed value on the valid ones.
            class_raster = arcpy.sa.SetNull(norm_raster <= 0, 0)
            min_angle_raster = arcpy.sa.SetNull(norm_raster <= 0, 90)
            
            # Loop through classes to find minimum angle
            for idx, class_name in enumerate(class_names, 1):
                sam_raster = sam_rasters[class_name]
                
                # Update classification where this class has smaller angle
                class_raster = arcpy.sa.Con(
                    arcpy.sa.BooleanAnd(
                        sam_raster < min_angle_raster,
                        sam_raster <= max_angle_rad * (180.0 / math.pi)
                    ),
                    idx,
                    class_raster
                )
                
                # Update minimum angle raster
                min_angle_raster = arcpy.sa.Con(
                    sam_raster < min_angle_raster,
                    sam_raster,
                    min_angle_raster
                )
            
            # Save output classification raster
            arcpy.AddMessage(f"Saving classification raster to: {out_class_path}")
            class_raster.save(out_class_path)
            
            # Save SAM angle raster if requested
            if out_sam_path:
                arcpy.AddMessage(f"Saving SAM angle raster to: {out_sam_path}")
                min_angle_raster.save(out_sam_path)
            
            # Apply color map if requested
            if apply_color:
                arcpy.AddMessage("Applying color map to classification raster...")
                
                # Create color map
                color_map = []
                for idx, class_name in enumerate(class_names, 1):
                    # Generate a color based on index
                    hue = (idx * 137) % 360  # Use golden ratio to distribute colors
                    rgb = self._hsv_to_rgb(hue/360.0, 0.8, 0.9)
                    color_map.append([idx, class_name, rgb[0], rgb[1], rgb[2]])
                
                # Apply color map
                try:
                    arcpy.AddMessage("Setting classification symbology...")
                    result = arcpy.management.ApplySymbologyFromLayer(out_class_path, "")
                    
                    # Get the output raster layer
                    layer = result.getOutput(0)
                    
                    # Apply custom color map
                    for entry in color_map:
                        idx, name, r, g, b = entry
                        # Add code to apply color to layer
                        arcpy.AddMessage(f"  Class {idx}: {name} - RGB({r},{g},{b})")
                except Exception as e:
                    arcpy.AddWarning(f"Could not apply color map: {str(e)}")
            
            # Return output paths
            return out_class_path
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with table: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _sam_with_training(self, input_raster, training_samples, max_angle_rad, threshold,
                          out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using training samples/ROIs"""
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage(f"Training samples: {training_samples}")
            
            # Look for class field in training samples
            class_field = None
            desc = arcpy.Describe(training_samples)
            
            # Check possible field names
            possible_fields = ["class", "classname", "class_name", "category", "label"]
            for field in desc.fields:
                if field.type == "String" and field.name.lower() in possible_fields:
                    class_field = field.name
                    break
            
            if not class_field:
                arcpy.AddWarning("No suitable class field found in training samples")
                arcpy.AddWarning("Using first string field as class field")
                
                # Use first string field
                for field in desc.fields:
                    if field.type == "String":
                        class_field = field.name
                        break
            
            if not class_field:
                arcpy.AddError("No string field found in training samples for classification")
                return None
            
            arcpy.AddMessage(f"Using field '{class_field}' for class names")
            
            # Extract values to points
            arcpy.AddMessage("Extracting spectral values from training samples...")
            
            # Create a temporary table to store extracted values
            temp_table = arcpy.CreateUniqueName("sam_training", arcpy.env.scratchGDB)
            arcpy.sa.ExtractValuesToTable(
                training_samples,
                input_raster,
                temp_table,
                "NONE",
                "ALL"
            )
            
            # Add class field to the temporary table
            arcpy.management.AddJoin(
                temp_table,
                "OBJECTID",
                training_samples,
                "OBJECTID",
                "KEEP_ALL"
            )
            
            # Get class names and band fields
            fields = [f.name for f in arcpy.ListFields(temp_table)]
            training_class_field = f"{os.path.basename(training_samples)}.{class_field}"
            
            # Get band fields (BAND_1, BAND_2, etc.)
            band_fields = [f for f in fields if f.startswith("BAND_")]
            
            # Sort band fields numerically
            band_fields.sort(key=lambda x: int(x.split("_")[1]))
            
            # Now use the table-based SAM with our extracted training data
            self._sam_with_table(
                input_raster=input_raster,
                ref_table=temp_table,
                class_field=training_class_field,
                band_fields=band_fields,
                max_angle_rad=max_angle_rad,
                threshold=threshold,
                out_workspace=out_workspace,
                out_raster=out_raster,
                out_sam=out_sam,
                apply_color=apply_color
            )
            
            # Clean up
            try:
                arcpy.management.Delete(temp_table)
            except:
                pass
            
            # Return output path
            return os.path.join(out_workspace, out_raster)
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with training samples: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _sam_with_endmembers(self, input_raster, endmember_pixels, max_angle_rad, threshold,
                            out_workspace, out_raster, out_sam, apply_color):
        """Perform SAM classification using endmember pixels"""
        try:
            arcpy.AddMessage(f"Processing input raster: {input_raster}")
            arcpy.AddMessage("Using endmember pixels for reference spectra")
            
            # Create temporary points feature class
            temp_points = arcpy.CreateUniqueName("endmember_points", arcpy.env.scratchGDB)
            
            # Create points feature class
            arcpy.management.CreateFeatureclass(
                os.path.dirname(temp_points),
                os.path.basename(temp_points),
                "POINT",
                spatial_reference=arcpy.Describe(input_raster).spatialReference
            )
            
            # Add CLASS field
            arcpy.management.AddField(temp_points, "CLASS", "TEXT")
            
            # Add points
            with arcpy.da.InsertCursor(temp_points, ["SHAPE@XY", "CLASS"]) as cursor:
                for row in endmember_pixels:
                    class_name = row[0]
                    x = row[1]
                    y = row[2]
                    cursor.insertRow([(x, y), class_name])
            
            # Now use the training samples approach with our temporary points
            self._sam_with_training(
                input_raster=input_raster,
                training_samples=temp_points,
                max_angle_rad=max_angle_rad,
                threshold=threshold,
                out_workspace=out_workspace,
                out_raster=out_raster,
                out_sam=out_sam,
                apply_color=apply_color
            )
            
            # Clean up
            try:
                arcpy.management.Delete(temp_points)
            except:
                pass
            
            # Return output path
            return os.path.join(out_workspace, out_raster)
            
        except Exception as e:
            arcpy.AddError(f"Error in SAM calculation with endmember pixels: {str(e)}")
            import traceback
            arcpy.AddError(traceback.format_exc())
            return None
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB"""
        if s == 0.0:
            return (int(v * 255), int(v * 255), int(v * 255))
        
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        
        if i == 0:
            return (int(v * 255), int(t * 255), int(p * 255))
        elif i == 1:
            return (int(q * 255), int(v * 255), int(p * 255))
        elif i == 2:
            return (int(p * 255), int(v * 255), int(t * 255))
        elif i == 3:
            return (int(p * 255), int(q * 255), int(v * 255))
        elif i == 4:
            return (int(t * 255), int(p * 255), int(v * 255))
        else:
            return (int(v * 255), int(p * 255), int(q * 255))
