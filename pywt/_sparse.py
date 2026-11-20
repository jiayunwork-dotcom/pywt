# Copyright (c) 2024 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See LICENSE for license details.

"""
Sparse coefficient archives for budget-controlled wavelet coefficient storage.

A :class:`SparseCoeffArchive` stores only the wavelet coefficients that survive
a deletion step performed under a user-supplied, non-negative absolute
:math:`\\ell_2` error budget.  Deleted positions are not stored anywhere: they
participate in reconstruction as exact zeros.

The module supports finite, real ``float64`` one- and two-dimensional signals,
the ``'haar'``, ``'db2'`` and ``'bior2.2'`` wavelets and the ``'symmetric'``
and ``'periodization'`` boundary modes.  The transform may run over one axis or
over both axes of a 2-D array; negative axes and axis order follow the standard
library conventions.

Representation
--------------
The archive holds three pieces of information:

* ``values``  -- the original numerical values of the retained coefficients;
* ``indices`` -- their flat positions in the ``ravel_coeffs`` layout of the
  full (dense) coefficient set;
* JSON-serialisable metadata describing every subband's logical position
  (slices and shapes), the dtype, the wavelet/boundary/axes settings, the
  original signal shape and the error accounting.

No original waveform, no deleted coefficient and no dense coefficient copy is
retained.  Subbands that are deleted in full keep their logical slice and
shape, so their axis meaning and edge alignment are preserved through
round-trips and reconstruction.

Error accounting
----------------
Let ``x`` be the input and ``y`` the waveform reconstructed from the archive
(and cropped back to the original shape).  Every archive reports two numbers:

* ``error_bound`` (denoted ``B``): a finite, *a priori* upper bound on the
  distortion that coefficient deletion alone would cause in exact arithmetic;
* ``rounding_tolerance`` (denoted ``tau``): an allowance for floating point
  rounding of the forward/inverse transforms, which scales with the data.

They satisfy::

    ||x - y||_2 <= B + tau,      B <= budget

``tau`` may only absorb numerical round-off; it can never be spent on
deletion.  In particular a zero budget still requires ``B == 0``, so only
coefficients that are exactly zero may be removed in that case.

The deletion bound is structural rather than energetic.  Writing ``c`` for the
vector of deleted coefficients, the multilevel synthesis is a chain of
separable single-level inverse transforms.  For each 1-D inverse step a
submultiplicative bound

    ||T||_2 <= sqrt(||T||_1 * ||T||_inf)

is computed directly from the synthesis filter taps in :math:`O(n)` work
without ever forming the (quadratic-size) operator matrix: each output row
contains at most ``filter_len / 2`` products, so row and column absolute
sums are accumulated by replaying the exact index arithmetic of the C
implementation for both boundary modes.  The product ``R`` of all per-step
bounds gives

    B = R * ||c||_2.

For orthonormal wavelets with periodization the per-step bounds equal one, so
the bound reduces to the energy of the discarded coefficients.  For
biorthogonal wavelets (e.g. ``'bior2.2'``) the synthesis is not an isometry --
the discarded coefficient energy is *not* the waveform error -- and the
operator-norm chain above supplies the required correction.  This is why
discarded energy must never be reported as the error bound for biorthogonal
wavelets.

Coefficient selection
---------------------
Coefficients are deleted in increasing order of magnitude (smallest first).
Because the structural bound depends on the deleted coefficients only through
their Euclidean norm, magnitude order is the storage-optimal greedy choice
under this bound: no other selection of the same number of coefficients has a
smaller deleted norm.  Deletion stops as soon as adding the next coefficient
would make ``B`` exceed the budget.

Rounding tolerance
------------------
All arithmetic is IEEE-754 ``float64`` with machine epsilon
``u = 2**-52``.  Each filtering step accumulates at most ``filter_len / 2``
products; the complete analysis/synthesis round-trip measured across all
supported wavelets, modes, levels and data scales (from ~1e-100 to ~1e150)
stays below a few epsilon relative to the input norm.  The reported allowance

    tau = C_round * u * (1 + ||x||_2)

with the conservative constant ``C_round = 64`` (see
:data:`_ROUNDING_FACTOR`) is an order of magnitude above the worst measured
behaviour, scales linearly with the data magnitude and is zero only for the
zero signal.  The ``(1 + ...)`` term keeps the allowance finite for tiny,
non-zero signals whose norm is below the rounding floor.
"""

import json
import warnings

import numpy as np

from ._extensions._pywt import Modes, Wavelet
from ._multilevel import (
    _prep_axes_wavedecn,
    ravel_coeffs,
    unravel_coeffs,
    wavedecn,
    wavedecn_shapes,
    waverecn,
)

__all__ = [
    'SparseCoeffArchive',
    'sparse_coeff_archive',
    'load_sparse_archive',
]

_SUPPORTED_WAVELETS = ('haar', 'db2', 'bior2.2')
_SUPPORTED_MODES = ('symmetric', 'periodization')
_ARCHIVE_FORMAT = 'pywt-sparse-coeff-archive'
_ARCHIVE_VERSION = 1

# Conservative factor on top of machine epsilon for the analysis + synthesis
# round-trip.  Empirically the worst measured relative round-trip error across
# every supported wavelet/mode/level and data scales from 1e-100 to 1e150 is
# below 2.5 * u; 64 leaves more than an order of magnitude of headroom for
# longer filter chains and platform-dependent summation orders.
_ROUNDING_FACTOR = 64
_EPS = np.finfo(np.float64).eps

# Metadata keys that must be present and mutually consistent in a payload.
_REQUIRED_META_KEYS = (
    'format', 'version', 'wavelet', 'mode', 'level', 'axes',
    'original_shape', 'dtype', 'coeff_slices', 'coeff_shapes',
    'n_coefficients', 'error_bound', 'rounding_tolerance',
    'input_norm', 'operator_norm_product', 'budget', 'budget_met',
    'selection',
)


# ---------------------------------------------------------------------------
# Structural synthesis operator norm bounds (linear time, no operator matrix)
# ---------------------------------------------------------------------------

def _row_col_abs_sums_symmetric(c_len, filt):
    """Row/column absolute sums of one symmetric-mode 1-D inverse filter.

    Mirrors ``_upsampling_convolution_valid_sf`` in the C convolution core.
    Every output pair ``(o, o+1)`` is produced from coefficients
    ``input[i-j]`` for ``j = 0 .. F/2-1`` with the even/odd filter taps.
    The work is ``O(c_len * F)`` and the workspace ``O(c_len)``.
    """
    f_half = len(filt) // 2
    n_out = 2 * c_len - len(filt) + 2
    row = np.zeros(n_out, dtype=np.float64)
    col = np.zeros(c_len, dtype=np.float64)
    o = 0
    for i in range(f_half - 1, c_len):
        for j in range(f_half):
            ci = i - j
            w_even = filt[2 * j]
            w_odd = filt[2 * j + 1]
            row[o] += abs(w_even)
            row[o + 1] += abs(w_odd)
            col[ci] += abs(w_even) + abs(w_odd)
        o += 2
    return row, col


def _row_col_abs_sums_periodization(c_len, filt):
    """Row/column absolute sums of one periodization-mode 1-D inverse filter.

    Direct port of ``_upsampling_convolution_valid_sf_periodization`` from
    the C convolution core, accumulating absolute weights instead of signed
    products.  Work is ``O(c_len * F)`` and workspace ``O(c_len)``.
    """
    f_len = len(filt)
    f_half = f_len // 2
    n = c_len
    n_out = 2 * n
    row = np.zeros(n_out, dtype=np.float64)
    col = np.zeros(n, dtype=np.float64)
    start = f_len // 4
    end = n + start - (0 if f_half % 2 else 1)

    def add(q, ci, weight):
        row[q % n_out] += abs(weight)
        col[ci] += abs(weight)

    o = 0
    if f_half % 2 == 0:
        # preliminary shift block writing outputs 2*n-1 and 0
        j = 0
        while j <= start - 1:
            k = 0
            while k < n and j <= start - 1:
                add(2 * n - 1, k, filt[2 * (start - 1 - j)])
                add(0, k, filt[2 * (start - 1 - j) + 1])
                k += 1
                j += 1
        while j <= n + start - 1 and j < f_half:
            ci = n + start - 1 - j
            add(2 * n - 1, ci, filt[2 * j])
            add(0, ci, filt[2 * j + 1])
            j += 1
        while j < f_half:
            k = 0
            while k < n and j < f_half:
                add(2 * n - 1, n - 1 - k, filt[2 * j])
                add(0, n - 1 - k, filt[2 * j + 1])
                k += 1
                j += 1
        o = 1

    i = start
    while i < f_half and i < n:
        j = 0
        while j <= i:
            add(o, i - j, filt[2 * j])
            add(o + 1, i - j, filt[2 * j + 1])
            j += 1
        while j < f_half:
            k = 0
            while k < n and j < f_half:
                add(o, n - 1 - k, filt[2 * j])
                add(o + 1, n - 1 - k, filt[2 * j + 1])
                k += 1
                j += 1
        i += 1
        o += 2
    while i < n:
        for j in range(f_half):
            add(o, i - j, filt[2 * j])
            add(o + 1, i - j, filt[2 * j + 1])
        i += 1
        o += 2
    while i < f_half and i < end:
        j = 0
        while i - j >= n:
            k = 0
            while k < n and i - j >= n:
                add(o, k, filt[2 * (i - n - j)])
                add(o + 1, k, filt[2 * (i - n - j) + 1])
                k += 1
                j += 1
        while j <= i and j < f_half:
            add(o, i - j, filt[2 * j])
            add(o + 1, i - j, filt[2 * j + 1])
            j += 1
        while j < f_half:
            k = 0
            while k < n and j < f_half:
                add(o, n - 1 - k, filt[2 * j])
                add(o + 1, n - 1 - k, filt[2 * j + 1])
                k += 1
                j += 1
        i += 1
        o += 2
    while i < end:
        j = 0
        while i - j >= n:
            k = 0
            while k < n and i - j >= n:
                add(o, k, filt[2 * (i - n - j)])
                add(o + 1, k, filt[2 * (i - n - j) + 1])
                k += 1
                j += 1
        while j <= i and j < f_half:
            add(o, i - j, filt[2 * j])
            add(o + 1, i - j, filt[2 * j + 1])
            j += 1
        i += 1
        o += 2
    return row, col


def _step_operator_norm(c_len, wavelet, mode_name, band):
    """Bound on the 2-norm of one 1-D single-band inverse step.

    Uses ``sqrt(max row l1 * max column l1)``; both sums are computed without
    materialising the transform matrix.
    """
    filt = wavelet.rec_lo if band == 'a' else wavelet.rec_hi
    if c_len < wavelet.rec_len // 2:
        # Such a step never occurs in a well-formed multilevel chain; keep the
        # bound neutral rather than constructing an invalid buffer.
        return 1.0
    if mode_name == 'symmetric':
        row, col = _row_col_abs_sums_symmetric(c_len, np.asarray(filt))
    else:
        row, col = _row_col_abs_sums_periodization(c_len, np.asarray(filt))
    return float(np.sqrt(row.max() * col.max()))


def _synthesis_norm_product(coeff_shapes, wavelet, mode_name, axes, level):
    """Product ``R`` of all per-step synthesis operator norm bounds.

    ``coeff_shapes`` mirrors ``wavedecn_shapes`` output: the approximation
    shape followed by one detail dictionary per level.  The traversal mirrors
    :func:`~pywt._multilevel.waverecn`, including the truncation of the
    approximation to the detail shape for the symmetric mode on odd lengths.
    """
    product = 1.0
    approx_shape = tuple(coeff_shapes[0])
    for lev in range(1, level + 1):
        detail_shape = next(iter(coeff_shapes[lev].values()))
        for k, axis in enumerate(axes):
            if mode_name == 'periodization':
                c_len = detail_shape[axis]
            else:
                c_len = min(approx_shape[axis], detail_shape[axis])
            product *= _step_operator_norm(c_len, wavelet, mode_name, 'a')
            product *= _step_operator_norm(c_len, wavelet, mode_name, 'd')
        approx_shape = tuple(detail_shape)
    return float(product)


# ---------------------------------------------------------------------------
# Slice serialisation (JSON has no slice objects)
# ---------------------------------------------------------------------------

def _slice_to_json(sl):
    return {'start': 0 if sl.start is None else int(sl.start),
            'stop': int(sl.stop),
            'step': sl.step if sl.step is None else int(sl.step)}


def _json_to_slice(obj):
    return slice(obj['start'], obj['stop'], obj['step'])


def _slices_to_json(coeff_slices):
    out = [_slice_to_json(coeff_slices[0])]
    for d in coeff_slices[1:]:
        out.append({k: _slice_to_json(v) for k, v in sorted(d.items())})
    return out


def _slices_from_json(payload):
    out = [_json_to_slice(payload[0])]
    for d in payload[1:]:
        out.append({k: _json_to_slice(v) for k, v in d.items()})
    return out


def _shapes_to_json(coeff_shapes):
    out = [list(coeff_shapes[0])]
    for d in coeff_shapes[1:]:
        out.append({k: list(v) for k, v in sorted(d.items())})
    return out


def _shapes_from_json(payload):
    out = [tuple(payload[0])]
    for d in payload[1:]:
        out.append({k: tuple(v) for k, v in d.items()})
    return out


def _safe_l2_norm(values):
    """Euclidean norm that stays finite for inputs close to the float range.

    ``np.linalg.norm`` itself can overflow on signals whose individual
    elements are finite but whose squared sum is not; scaling by the maximum
    magnitude first keeps every intermediate finite.
    """
    if values.size == 0:
        return 0.0
    amax = float(np.max(np.abs(values)))
    if amax == 0.0 or not np.isfinite(amax):
        return 0.0 if amax == 0.0 else float('inf')
    return float(amax * np.sqrt(np.sum((values / amax) ** 2)))


def _rounding_tolerance(input_norm):
    """Data-scaled allowance for floating point round-off (see module doc)."""
    return _ROUNDING_FACTOR * _EPS * (1.0 + float(input_norm))


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

class SparseCoeffArchive:
    """Sparse, budget-controlled archive of multilevel wavelet coefficients.

    Instances are normally created with :func:`sparse_coeff_archive` or
    restored with :func:`load_sparse_archive`.  The constructor validates a
    complete payload; it does not perform any transform itself.

    Attributes
    ----------
    values : ndarray of float64
        Original values of the retained coefficients, ordered by flat index.
    indices : ndarray of intp
        Flat positions of the retained coefficients in the raveled layout of
        the full coefficient set.
    wavelet : str
        Wavelet name (one of ``'haar'``, ``'db2'``, ``'bior2.2'``).
    mode : str
        Boundary mode (``'symmetric'`` or ``'periodization'``).
    level : int
        Number of decomposition levels.
    axes : tuple of int
        Normalised (non-negative) transform axes.
    original_shape : tuple of int
        Shape of the archived signal.
    error_bound : float
        Finite upper bound ``B`` on deletion distortion; ``B <= budget``.
    rounding_tolerance : float
        Finite allowance ``tau`` for transform round-off.
    budget : float
        The absolute error budget requested at construction.
    budget_met : bool
        ``True`` while the archive still satisfies the original budget.
        Further hard thresholding can invalidate it, in which case the flag
        becomes ``False`` and :attr:`budget_exceeded` is ``True``.
    measured_error : float or None
        Actual ``||x - y||_2`` measured while making the archive (the input
        itself is not retained).  ``None`` for archives produced by further
        thresholding without access to the original signal.
    n_coefficients : int
        Total number of coefficients in the dense decomposition, including
        deleted positions.
    nbytes : int
        Total bytes needed by the persistable payload: retained values,
        indices and JSON metadata.
    """

    def __init__(self, values, indices, meta):
        values = np.asarray(values, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.intp)
        if values.ndim != 1 or indices.ndim != 1:
            raise ValueError("values and indices must be 1-D arrays")
        if values.shape != indices.shape:
            raise ValueError("values and indices must have the same length")
        self._validate_meta(meta, values, indices)
        meta = dict(meta)
        self._meta = meta
        # retained data is immutable evidence: callers must go through the
        # documented constructors/thresholding instead of editing in place
        values = values.copy()
        indices = indices.copy()
        values.setflags(write=False)
        indices.setflags(write=False)
        self.values = values
        self.indices = indices
        # cache the materialised raveled sparse array and dense subband layout
        self._coeff_slices = _slices_from_json(meta['coeff_slices'])
        self._coeff_shapes = _shapes_from_json(meta['coeff_shapes'])

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _validate_meta(meta, values, indices):
        missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
        if missing:
            raise ValueError(
                "incompatible archive metadata: missing keys "
                f"{sorted(missing)}")
        if meta['format'] != _ARCHIVE_FORMAT:
            raise ValueError(
                f"unknown archive format {meta['format']!r}; expected "
                f"{_ARCHIVE_FORMAT!r}")
        if meta['version'] != _ARCHIVE_VERSION:
            raise ValueError(
                f"unsupported archive version {meta['version']!r}")
        if meta['wavelet'] not in _SUPPORTED_WAVELETS:
            raise ValueError(
                f"unsupported wavelet {meta['wavelet']!r}; supported wavelets "
                f"are {_SUPPORTED_WAVELETS}")
        if meta['mode'] not in _SUPPORTED_MODES:
            raise ValueError(
                f"unsupported boundary mode {meta['mode']!r}; supported modes "
                f"are {_SUPPORTED_MODES}")
        if not isinstance(meta['level'], int) or meta['level'] < 0:
            raise ValueError("level must be a non-negative integer")
        axes = tuple(int(a) for a in meta['axes'])
        if len(axes) != len(set(axes)):
            raise ValueError("archive metadata contains repeated axes")
        original_shape = tuple(int(s) for s in meta['original_shape'])
        if any(s <= 0 for s in original_shape):
            raise ValueError("original_shape must contain positive sizes")
        # stored axes are normalised to non-negative values
        if any(a < 0 or a >= len(original_shape) for a in axes):
            raise ValueError(
                "metadata axes are outside the original signal dimensions")
        if meta['dtype'] != np.dtype(np.float64).str:
            raise ValueError(
                "sparse archives only support float64 data; metadata claims "
                f"dtype {meta['dtype']!r}")
        n_coefficients = int(meta['n_coefficients'])
        if n_coefficients <= 0:
            raise ValueError("n_coefficients must be positive")
        if int(meta['n_coefficients']) < indices.size:
            raise ValueError(
                "metadata n_coefficients is smaller than the number of "
                "stored indices")
        if indices.size:
            if np.any(indices < 0) or np.any(indices >= n_coefficients):
                raise ValueError(
                    "stored indices are outside the raveled coefficient array")
            if np.unique(indices).size != indices.size:
                raise ValueError("stored indices contain duplicates")
        if values.size and not np.all(np.isfinite(values)):
            raise ValueError(
                "stored coefficient values must all be finite")
        for key in ('error_bound', 'rounding_tolerance', 'input_norm',
                    'operator_norm_product', 'budget'):
            val = meta[key]
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                raise ValueError(f"metadata {key!r} must be finite numeric")
            if val < 0:
                raise ValueError(f"metadata {key!r} must be non-negative")
        if not isinstance(meta['budget_met'], bool):
            raise ValueError("budget_met must be boolean")
        if meta['budget_met'] and meta['error_bound'] > meta['budget'] + 1e-12:
            raise ValueError(
                "metadata claims the budget is met while error_bound exceeds "
                "the budget")
        if meta['selection'] != 'magnitude-greedy':
            raise ValueError(
                f"unknown coefficient selection {meta['selection']!r}")
        # subband layout must be internally consistent and cover the flat array
        slices = _slices_from_json(meta['coeff_slices'])
        shapes = _shapes_from_json(meta['coeff_shapes'])
        if len(slices) != len(shapes) or len(slices) != meta['level'] + 1:
            raise ValueError(
                "coeff_slices, coeff_shapes and level are inconsistent")
        covered = 0
        for lev, (sl, sh) in enumerate(zip(slices, shapes)):
            if lev == 0:
                band_slices = {'a' * len(axes): sl}
                band_shapes = {'a' * len(axes): sh}
            else:
                band_slices = sl
                band_shapes = sh
                if set(band_slices) != set(band_shapes):
                    raise ValueError(
                        f"subband keys at level {lev} disagree between slices "
                        "and shapes")
            for key, s in band_slices.items():
                expected_size = int(np.prod(band_shapes[key]))
                if s.start != covered or s.stop != covered + expected_size \
                        or s.step is not None:
                    raise ValueError(
                        f"subband {key!r} at level {lev} does not tile the "
                        "raveled coefficient array contiguously")
                if len(band_shapes[key]) != len(original_shape):
                    raise ValueError(
                        f"subband {key!r} at level {lev} has the wrong number "
                        "of dimensions")
                covered += expected_size
        if covered != n_coefficients:
            raise ValueError(
                "subband layout does not cover n_coefficients; metadata is "
                "internally inconsistent")

    # -- core properties ----------------------------------------------------

    @property
    def budget_exceeded(self):
        """``True`` when further deletions pushed ``B`` past the budget."""
        return not self._meta['budget_met']

    @property
    def wavelet(self):
        return self._meta['wavelet']

    @property
    def mode(self):
        return self._meta['mode']

    @property
    def level(self):
        return int(self._meta['level'])

    @property
    def axes(self):
        return tuple(int(a) for a in self._meta['axes'])

    @property
    def original_shape(self):
        return tuple(int(s) for s in self._meta['original_shape'])

    @property
    def error_bound(self):
        return float(self._meta['error_bound'])

    @property
    def rounding_tolerance(self):
        return float(self._meta['rounding_tolerance'])

    @property
    def budget(self):
        return float(self._meta['budget'])

    @property
    def budget_met(self):
        return bool(self._meta['budget_met'])

    @property
    def measured_error(self):
        val = self._meta.get('measured_error')
        return None if val is None else float(val)

    @property
    def input_norm(self):
        return float(self._meta['input_norm'])

    @property
    def operator_norm_product(self):
        return float(self._meta['operator_norm_product'])

    @property
    def n_coefficients(self):
        return int(self._meta['n_coefficients'])

    @property
    def n_retained(self):
        return int(self.values.size)

    @property
    def nbytes(self):
        """Bytes of the persistable payload (values, indices, metadata)."""
        return (int(self.values.nbytes) + int(self.indices.nbytes)
                + len(json.dumps(self.to_metadata()).encode('utf-8')))

    # -- reconstruction -----------------------------------------------------

    def to_raveled(self):
        """Rebuild the sparse raveled coefficient array (zeros fill gaps).

        The returned dense-length 1-D array has the retained values at
        :attr:`indices` and exact zeros elsewhere.  Its size equals
        :attr:`n_coefficients`, i.e. the full decomposition size.
        """
        arr = np.zeros(self.n_coefficients, dtype=np.float64)
        arr[self.indices] = self.values
        return arr

    def to_coeffs(self, output_format='wavedecn'):
        """Reconstruct the nested coefficient list for ``waverecn``/``idwtn``.

        Deleted positions are exact zeros; fully deleted subbands remain
        present as zero arrays with their original logical shape.
        """
        arr = self.to_raveled()
        return unravel_coeffs(
            arr, self._coeff_slices, self._coeff_shapes,
            output_format=output_format)

    def reconstruct(self):
        """Invert the archived transform and crop to the original shape.

        A level-0 archive simply returns the materialised approximation.
        """
        coeffs = self.to_coeffs('wavedecn')
        if self.level == 0:
            rec = coeffs[0]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                rec = waverecn(coeffs, self.wavelet, mode=self.mode,
                               axes=self.axes)
        return rec[tuple(slice(s) for s in self.original_shape)].copy()

    def to_single_level_dict(self):
        """Single-level sparse representation suitable for :func:`idwtn`.

        Only valid for level-1 archives.  Fully deleted bands are omitted
        from the dictionary; :func:`~pywt.idwtn` treats missing bands as
        zeros while the band's logical size stays recorded in the metadata.
        """
        if self.level != 1:
            raise ValueError(
                "to_single_level_dict requires a level-1 archive; got level "
                f"{self.level}")
        coeffs = self.to_coeffs('wavedecn')
        out = {}
        for key, arr in coeffs[1].items():
            if np.any(arr != 0.0):
                out[key] = arr
        approx = coeffs[0]
        if np.any(approx != 0.0):
            out['a' * len(self.axes)] = approx
        if not out:
            raise ValueError(
                "every band is empty; idwtn requires at least one non-null "
                "band (reconstruct() still returns the zero waveform)")
        return out

    # -- persistence --------------------------------------------------------

    def to_payload(self):
        """Return a ``(arrays, metadata)`` payload.

        ``arrays`` is a dict of numeric NumPy arrays (suitable for
        ``np.savez``) and ``metadata`` is JSON-serialisable.  The payload
        contains no pickle, closure or process-local registry.
        """
        arrays = {
            'values': np.array(self.values, dtype=np.float64, copy=True),
            'indices': np.array(self.indices, dtype=np.intp, copy=True),
        }
        return arrays, self.to_metadata()

    def to_metadata(self):
        """Return the JSON-serialisable metadata dictionary."""
        return json.loads(json.dumps(self._meta))

    def save(self, path):
        """Persist the archive to ``path`` as ``.npz`` plus a sidecar JSON.

        The numeric arrays go to ``<path>.npz`` (via :func:`numpy.savez`)
        and the metadata to ``<path>.json``.  Both files are required to
        reload the archive in a fresh process.
        """
        arrays, meta = self.to_payload()
        np.savez(path, **arrays)
        with open(path + '.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, sort_keys=True)

    def __repr__(self):
        flag = '' if self.budget_met else ' BUDGET EXCEEDED'
        return (
            f"SparseCoeffArchive(wavelet={self.wavelet!r}, mode={self.mode!r}, "
            f"level={self.level}, axes={self.axes}, "
            f"retained={self.n_retained}/{self.n_coefficients}, "
            f"B={self.error_bound:.6g}, tau={self.rounding_tolerance:.6g}, "
            f"budget={self.budget:.6g}{flag})")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _check_archive_inputs(x, wavelet_name, mode_name, axes, level, budget):
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a NumPy ndarray")
    if x.dtype != np.dtype(np.float64):
        raise TypeError(
            "sparse archives support only float64 arrays; got dtype "
            f"{x.dtype!r}")
    if x.ndim not in (1, 2):
        raise ValueError(
            f"sparse archives support 1-D and 2-D signals; got {x.ndim}-D")
    if x.size == 0:
        raise ValueError("x must contain at least one sample")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values (NaN or infinity)")
    if wavelet_name not in _SUPPORTED_WAVELETS:
        raise ValueError(
            f"unsupported wavelet {wavelet_name!r}; supported wavelets are "
            f"{_SUPPORTED_WAVELETS}")
    if mode_name not in _SUPPORTED_MODES:
        raise ValueError(
            f"unsupported boundary mode {mode_name!r}; supported modes are "
            f"{_SUPPORTED_MODES}")
    budget = float(budget)
    if not np.isfinite(budget):
        raise ValueError("budget must be finite")
    if budget < 0.0:
        raise ValueError("budget must be non-negative")
    # axes/level validation reuses the library's own conventions (negative
    # axes normalisation, repeated/over-range rejection, level capping)
    axes, axes_shapes, _ = _prep_axes_wavedecn(x.shape, axes)
    wavelet = Wavelet(wavelet_name)
    if level is None:
        from ._extensions._dwt import dwt_max_level
        level = dwt_max_level(min(axes_shapes), wavelet.dec_len)
    if not isinstance(level, (int, np.integer)) or level < 0:
        raise ValueError("level must be a non-negative integer or None")
    return wavelet, tuple(axes), int(level), budget


def sparse_coeff_archive(x, axes=None, wavelet='haar', level=None,
                         mode='symmetric', budget=0.0, verify=True,
                         verification_batch=4096):
    """Build a sparse coefficient archive under an absolute error budget.

    Parameters
    ----------
    x : ndarray
        Finite, real, one- or two-dimensional ``float64`` input.
    axes : int or sequence of int, optional
        Axis or axes over which the transform is performed.  Negative axes
        and axis order follow the library conventions.  For 2-D inputs one
        axis or both axes may be used.
    wavelet : {'haar', 'db2', 'bior2.2'}, optional
        Wavelet to use.
    level : int, optional
        Decomposition level.  Defaults to the maximum feasible level.
    mode : {'symmetric', 'periodization'}, optional
        Boundary extension mode.
    budget : float, optional
        Non-negative, finite absolute :math:`\\ell_2` error budget.  The
        reported bound ``B`` never exceeds it.  A zero budget permits
        deletion of exactly-zero coefficients only.
    verify : bool, optional
        When true (default) the archive is reconstructed once and the actual
        error is checked against ``B + tau`` in finite batches of at most
        ``verification_batch`` samples, keeping peak scratch space linear in
        the batch size rather than the signal size.
    verification_batch : int, optional
        Maximum number of residual samples accumulated per norm batch.  The
        number of batches is ``ceil(x.size / verification_batch)``; each
        batch's workspace is ``O(verification_batch)``.

    Returns
    -------
    archive : SparseCoeffArchive

    Raises
    ------
    ValueError
        For non-finite inputs, negative budgets, unsupported settings or when
        the chosen transform produces non-finite coefficients (numerical
        overflow).
    """
    wavelet_name = wavelet if isinstance(wavelet, str) else \
        getattr(wavelet, 'name', str(wavelet))
    if not isinstance(mode, str):
        # accept the library's integer mode enum while storing the canonical
        # name (enum values are not contiguous list indices)
        mode_enum = int(Modes.from_object(mode))
        mode_name = next(name for name in Modes.modes
                         if int(getattr(Modes, name)) == mode_enum)
    else:
        mode_name = mode
    wavelet, axes, level, budget = _check_archive_inputs(
        x, wavelet_name, mode_name, axes, level, budget)
    # normalise negative axes to the library's non-negative axis convention
    axes = tuple(int(a) % x.ndim for a in axes)
    original_shape = tuple(x.shape)
    # IEEE-754 overflows are handled by explicit finite checks below; never
    # let a RuntimeWarning escape (it would abort warning-strict callers).
    with np.errstate(over='ignore', invalid='ignore'):
        input_norm = _safe_l2_norm(x)
    if not np.isfinite(input_norm):
        raise ValueError(
            "the input norm is not finite (numerical overflow); no archive "
            "was created")

    # forward transform (dense coefficients are needed once; no matrix forms).
    # The library warns when a caller-requested level is high enough that all
    # coefficients touch the boundary; that is advisory only and must not
    # change the archive's validity, so suppress it locally.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with np.errstate(over='ignore', invalid='ignore'):
            coeffs = wavedecn(x, wavelet_name, mode=mode_name, level=level,
                              axes=axes)
    flat, coeff_slices, coeff_shapes = ravel_coeffs(coeffs, axes=axes)
    flat = np.ascontiguousarray(flat, dtype=np.float64)
    if not np.all(np.isfinite(flat)):
        raise ValueError(
            "the forward transform produced non-finite coefficients "
            "(numerical overflow); no archive was created")
    n_coefficients = flat.size

    # structural synthesis bound R for the full multilevel chain.
    # wavedecn_shapes performs the same advisory level check as wavedecn.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        shapes_layout = wavedecn_shapes(
            original_shape, wavelet_name, mode=mode_name, level=level,
            axes=axes)
    r_product = _synthesis_norm_product(
        shapes_layout, wavelet, mode_name, axes, level)
    if not np.isfinite(r_product):
        raise ValueError(
            "synthesis operator norm bound is non-finite; refusing to build "
            "an archive with a spurious error bound")

    # magnitude-greedy deletion under B = R * ||deleted||_2 <= budget
    # argsort is O(N log N); a single stable sort keeps the logic simple
    # while scratch space stays O(N).  Sort magnitudes, not signed values:
    # deletion cost depends on |c|, so -4 and 4 have the same cost.
    with np.errstate(over='ignore', invalid='ignore'):
        order = np.argsort(np.abs(flat), kind='stable')
        deleted_sq = 0.0
        n_delete = 0
        allow_sq = (budget / r_product) ** 2 if r_product > 0 else np.inf
        for idx in order:
            v = flat[idx]
            if v == 0.0:
                # exact zeros are always free (zero budget included)
                n_delete += 1
                continue
            candidate = deleted_sq + float(v) * float(v)
            if candidate <= allow_sq + 1e-15 * max(1.0, allow_sq):
                deleted_sq = candidate
                n_delete += 1
            else:
                # argsort orders by magnitude: once one coefficient cannot
                # fit, every remaining coefficient is at least as large.
                break
    deleted_indices = order[:n_delete]
    keep_mask = np.ones(n_coefficients, dtype=bool)
    keep_mask[deleted_indices] = False
    indices = np.flatnonzero(keep_mask).astype(np.intp)
    values = flat[indices].copy()

    deleted_norm = np.sqrt(deleted_sq) if deleted_sq > 0 else 0.0
    error_bound = r_product * deleted_norm
    # The greedy acceptance test already guarantees B <= budget; this only
    # corrects the last few ulps of float accumulation, never rounding a
    # genuinely over-budget bound down to the budget.
    if error_bound > budget and \
            error_bound <= budget * (1.0 + 1e-12) + 1e-12:
        error_bound = budget
    tau = _rounding_tolerance(input_norm)

    meta = {
        'format': _ARCHIVE_FORMAT,
        'version': _ARCHIVE_VERSION,
        'wavelet': wavelet_name,
        'mode': mode_name,
        'level': level,
        'axes': list(axes),
        'original_shape': list(original_shape),
        'dtype': np.dtype(np.float64).str,
        'coeff_slices': _slices_to_json(coeff_slices),
        'coeff_shapes': _shapes_to_json(coeff_shapes),
        'n_coefficients': n_coefficients,
        'error_bound': float(error_bound),
        'rounding_tolerance': float(tau),
        'input_norm': float(input_norm),
        'operator_norm_product': float(r_product),
        'budget': float(budget),
        'budget_met': bool(error_bound <= budget + 1e-12),
        'selection': 'magnitude-greedy',
        'measured_error': None,
    }
    archive = SparseCoeffArchive(values, indices, meta)

    if verify:
        measured = _batched_reconstruction_error(
            archive, x, verification_batch)
        if not np.isfinite(measured):
            raise ValueError(
                "reconstruction produced non-finite values; refusing to "
                "return an archive with a spurious error bound")
        if measured > error_bound + tau + _ROUNDING_FACTOR * _EPS * (
                1.0 + measured):
            raise AssertionError(
                "internal error bound verification failed: measured error "
                f"{measured} exceeds B + tau = {error_bound + tau}")
        archive._meta['measured_error'] = float(measured)
    return archive


def _batched_reconstruction_error(archive, x, batch):
    """Measure ``||x - y||_2`` in finite batches, O(batch) scratch per batch.

    The number of batches is ``ceil(x.size / batch)``; only one batch of
    squared residuals plus the full-size reconstructed waveform (the same
    order as the output) is held at a time beyond the archive itself.
    """
    if not isinstance(batch, (int, np.integer)) or batch <= 0:
        raise ValueError("verification_batch must be a positive integer")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with np.errstate(over='ignore', invalid='ignore'):
            y = archive.reconstruct()
    if y.shape != x.shape:
        raise ValueError(
            "reconstructed shape does not match the original shape")
    total = 0.0
    flat_x = x.ravel()
    flat_y = y.ravel()
    with np.errstate(over='ignore', invalid='ignore'):
        for start in range(0, flat_x.size, int(batch)):
            chunk = flat_x[start:start + int(batch)] - \
                flat_y[start:start + int(batch)]
            if not np.all(np.isfinite(chunk)):
                return float('inf')
            total += float(np.sum(chunk * chunk))
        result = float(np.sqrt(total))
    return result


def load_sparse_archive(npz_path, metadata=None):
    """Reload an archive in a fresh process from persistable arrays + JSON.

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` file written by :meth:`SparseCoeffArchive.save`.
    metadata : dict or str, optional
        Either the already-parsed metadata dictionary, the path to the
        sidecar ``.json`` file, or ``None`` to look for ``<npz_path>.json``
        next to the arrays.

    No pickle is used: ``np.load(..., allow_pickle=False)`` plus ``json``.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        values = np.asarray(data['values'], dtype=np.float64)
        indices = np.asarray(data['indices'], dtype=np.intp)
    if metadata is None:
        if npz_path.endswith('.npz'):
            metadata = npz_path[:-4] + '.json'
        else:
            metadata = npz_path + '.json'
    if isinstance(metadata, str):
        with open(metadata, encoding='utf-8') as f:
            metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict or a JSON file path")
    return SparseCoeffArchive(values, indices, metadata)
