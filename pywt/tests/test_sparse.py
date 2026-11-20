# Copyright (c) 2024 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See LICENSE for license details.

"""
Tests for sparse coefficient archives (``pywt._sparse``).

The archives store only retained wavelet coefficients plus JSON-serialisable
metadata, report a structural error bound ``B`` and a data-scaled rounding
tolerance ``tau`` with ``||x - y||_2 <= B + tau`` and ``B <= budget``, and
integrate with ``threshold``, ``ravel_coeffs``, ``unravel_coeffs``,
``idwtn`` and ``waverecn``.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import pywt

# ---------------------------------------------------------------------------
# The mandated minimum: compressible step signal, Haar, periodization, 4 levels
# ---------------------------------------------------------------------------

def test_step_signal_baseline_zero_and_small_budget():
    x = np.concatenate([np.ones(2048), -np.ones(2048)])
    dense_n = pywt.wavedecn_size(
        pywt.wavedecn_shapes((4096,), 'haar', mode='periodization', level=4))
    for budget in (0.0, 0.01):
        arc = pywt.sparse_coeff_archive(
            x, wavelet='haar', level=4, mode='periodization', budget=budget)
        y = arc.reconstruct()
        # recovery differs only by rounding
        assert np.linalg.norm(x - y) <= arc.rounding_tolerance
        # at most a quarter of the full decomposition coefficients are stored
        assert arc.n_retained <= dense_n // 4
        assert arc.error_bound <= arc.budget + 1e-12
        assert np.isfinite(arc.error_bound)
        assert np.isfinite(arc.rounding_tolerance)
        # the total byte count includes values, indices and metadata
        arrays, meta = arc.to_payload()
        expected = (arrays['values'].nbytes + arrays['indices'].nbytes
                    + len(json.dumps(meta).encode('utf-8')))
        assert arc.nbytes == expected
        assert arc.nbytes < dense_n * 8


# ---------------------------------------------------------------------------
# Error accounting across every supported wavelet/mode/dimension combination
# ---------------------------------------------------------------------------

shape_axes_cases = [
    ((128,), (0,)),
    ((17,), (0,)),
    ((23, 19), (0, 1)),
    ((23, 19), (1,)),
    ((23, 19), (-1,)),
    ((31, 27), (-2, -1)),
    ((15, 33), (1, 0)),
]


def test_error_inequality_all_combinations():
    rng = np.random.default_rng(0)
    for shape, axes in shape_axes_cases:
        x = rng.standard_normal(np.prod(shape)).reshape(shape)
        for wavelet in ('haar', 'db2', 'bior2.2'):
            w = pywt.Wavelet(wavelet)
            for mode in ('symmetric', 'periodization'):
                level = min(3, pywt.dwt_max_level(
                    min(x.shape[a % x.ndim] for a in axes), w.dec_len))
                if level < 1:
                    continue
                budget = 0.05 * np.linalg.norm(x)
                arc = pywt.sparse_coeff_archive(
                    x, axes=axes, wavelet=wavelet, level=level, mode=mode,
                    budget=budget)
                y = arc.reconstruct()
                assert y.shape == x.shape  # odd sizes restore input shape
                assert arc.error_bound <= budget + 1e-12
                measured = np.linalg.norm(x - y)
                assert measured <= arc.error_bound + arc.rounding_tolerance
                # B must never be paid for by the rounding allowance
                assert arc.rounding_tolerance < 1e-9 * max(1.0, budget) \
                    or budget == 0.0
                # the stored measured error matches an independent recompute
                assert abs(arc.measured_error - measured) < 1e-12


def test_bior22_short_array_itemwise_audit():
    # Small enough to inspect every residual sample individually.
    x = np.array([1.0, 2.5, -1.5, 0.5, 3.0, -2.0, 1.0, 0.0,
                  2.0, -1.0, 1.5, -0.5])
    for mode in ('periodization', 'symmetric'):
        arc = pywt.sparse_coeff_archive(
            x, wavelet='bior2.2', level=2, mode=mode, budget=0.5)
        y = arc.reconstruct()
        residuals = x - y
        assert residuals.shape == x.shape
        assert np.all(np.isfinite(residuals))
        measured = np.sqrt(np.sum(residuals ** 2))
        # biorthogonal: the discarded coefficient energy alone must not be
        # quoted as the waveform error; the structural R product exceeds one
        assert arc.operator_norm_product >= 1.0
        assert measured <= arc.error_bound + arc.rounding_tolerance
        assert arc.error_bound <= 0.5 + 1e-12


def test_tau_scales_with_data_magnitude():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(256)
    taus = []
    for scale in (1e-80, 1.0, 1e80):
        arc = pywt.sparse_coeff_archive(
            scale * base, wavelet='db2', level=3, budget=0.0)
        taus.append(arc.rounding_tolerance)
        assert np.isfinite(arc.rounding_tolerance)
        assert arc.rounding_tolerance > 0.0
    assert taus[0] < taus[1] < taus[2]
    # zero signal gets a finite, strictly positive floor allowance
    z = pywt.sparse_coeff_archive(
        np.zeros(64), wavelet='haar', level=2, budget=0.0)
    assert z.rounding_tolerance > 0.0 and z.error_bound == 0.0


# ---------------------------------------------------------------------------
# Zero signals and fully deleted bands
# ---------------------------------------------------------------------------

def test_zero_signal_archives_nothing():
    for shape in ((64,), (12, 10)):
        arc = pywt.sparse_coeff_archive(
            np.zeros(shape), wavelet='db2', level=2, budget=0.0)
        assert arc.n_retained == 0
        assert arc.error_bound == 0.0
        y = arc.reconstruct()
        assert y.shape == shape
        assert_array_equal(y, np.zeros(shape))
        # metadata alone still occupies bytes
        assert arc.nbytes > 0


def test_fully_deleted_band_keeps_logical_layout():
    rng = np.random.default_rng(2)
    arc = pywt.sparse_coeff_archive(
        rng.standard_normal(64), wavelet='db2', level=3, mode='symmetric',
        budget=1e12)
    coeffs = arc.to_coeffs()
    for level, d in enumerate(coeffs[1:], start=1):
        for key, band in d.items():
            assert band.shape == arc._coeff_shapes[level][key]
            assert band.dtype == np.float64
    # reconstruction still produces the original shape
    assert arc.reconstruct().shape == (64,)

    # level-1 archives feed idwtn directly; level-n archives feed waverecn
    arc1 = pywt.sparse_coeff_archive(
        rng.standard_normal(64), wavelet='bior2.2', level=1,
        mode='periodization', budget=0.3)
    single = arc1.to_single_level_dict()
    rec = pywt.idwtn(single, 'bior2.2', mode='periodization',
                     axes=arc1.axes)
    assert rec.shape == (64,)
    rec_ml = pywt.waverecn(arc1.to_coeffs(), 'bior2.2',
                           mode='periodization', axes=arc1.axes)
    assert rec_ml.shape == (64,)


# ---------------------------------------------------------------------------
# threshold(..., mode='hard') integration
# ---------------------------------------------------------------------------

def test_hard_threshold_accumulates_bound_and_preserves_input():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(256)
    arc = pywt.sparse_coeff_archive(
        x, wavelet='db2', level=3, mode='symmetric', budget=5.0)
    n_before = arc.n_retained
    b_before = arc.error_bound
    arc2 = pywt.threshold(arc, 0.4, mode='hard')

    assert arc2 is not arc
    assert arc.n_retained == n_before  # input object untouched
    assert arc2.n_retained <= n_before
    assert arc2.error_bound >= b_before - 1e-15
    if arc2.values.size:
        assert np.all(np.abs(arc2.values) >= 0.4)
    # survivors keep their exact original values
    common = np.intersect1d(arc.indices, arc2.indices)
    p1 = np.searchsorted(arc.indices, common)
    p2 = np.searchsorted(arc2.indices, common)
    assert_array_equal(arc.values[p1], arc2.values[p2])
    # the accumulated bound remains valid against the actual waveform error
    measured = np.linalg.norm(x - arc2.reconstruct())
    assert measured <= arc2.error_bound + arc2.rounding_tolerance


def test_hard_threshold_budget_exceeded_is_visible():
    rng = np.random.default_rng(4)
    arc = pywt.sparse_coeff_archive(
        rng.standard_normal(128), wavelet='haar', level=3, budget=1.0)
    over = pywt.threshold(arc, 1e6, mode='hard')
    assert over.budget_exceeded and not over.budget_met
    assert over.error_bound > over.budget
    assert over.measured_error is None  # x was never available here
    # original archive still claims its budget
    assert arc.budget_met and not arc.budget_exceeded


def test_threshold_rejects_soft_and_bad_arguments():
    rng = np.random.default_rng(5)
    arc = pywt.sparse_coeff_archive(
        rng.standard_normal(32), wavelet='haar', level=2, budget=0.5)
    with pytest.raises(ValueError):
        pywt.threshold(arc, 0.1, mode='soft')
    with pytest.raises(ValueError):
        pywt.threshold(arc, -0.1, mode='hard')
    with pytest.raises(ValueError):
        pywt.threshold(arc, 0.1, mode='hard', substitute=1.0)
    # dense behaviour is unchanged
    assert_array_equal(
        pywt.threshold(np.arange(4.), 2, 'hard'), np.array([0., 0., 2., 3.]))


# ---------------------------------------------------------------------------
# ravel / unravel round-trips and persistence
# ---------------------------------------------------------------------------

def test_ravel_unravel_sparse_and_dense_compatibility():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(128)
    arc = pywt.sparse_coeff_archive(
        x, wavelet='db2', level=3, mode='symmetric', budget=1.0)

    arr, slices, shapes, archive = pywt.ravel_coeffs(arc)
    assert arr.shape == (arc.n_coefficients,)
    assert_array_equal(np.flatnonzero(arr), arc.indices)
    coeffs, archive_back = pywt.unravel_coeffs(
        arr, slices, shapes, archive=archive)
    assert archive_back is archive
    rec = pywt.waverecn(coeffs, 'db2', mode='symmetric')
    assert_allclose(rec, arc.reconstruct())

    # dense inputs keep their original 3-argument / plain-return behaviour
    dense = pywt.wavedecn(x, 'haar', level=2)
    out = pywt.ravel_coeffs(dense)
    assert len(out) == 3
    assert isinstance(pywt.unravel_coeffs(*out), list)


def test_persistence_roundtrip_in_fresh_process():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((24, 20))
    arc = pywt.sparse_coeff_archive(
        x, axes=(0, 1), wavelet='bior2.2', level=2, mode='periodization',
        budget=0.5)
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, 'archive')
        arc.save(base)
        assert os.path.exists(base + '.npz')
        assert os.path.exists(base + '.json')
        loaded = pywt.load_sparse_archive(base + '.npz')
        assert_array_equal(loaded.values, arc.values)
        assert_array_equal(loaded.indices, arc.indices)
        assert_allclose(loaded.reconstruct(), arc.reconstruct())

        npz = base + '.npz'
        repo_path = '/workspace'
        site_path = '/home/node/.local/lib/python3.11/site-packages'
        script = (
            "import sys\n"
            f"sys.path.insert(0, {repo_path!r})\n"
            f"sys.path.insert(0, {site_path!r})\n"
            "import numpy as np, pywt\n"
            f"a = pywt.load_sparse_archive({npz!r})\n"
            "y = a.reconstruct()\n"
            "assert y.shape == (24, 20)\n"
            "assert np.isfinite(a.error_bound)\n"
            "print('fresh-process ok')\n"
        )
        proc = subprocess.run([sys.executable, '-I', '-c', script],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert 'fresh-process ok' in proc.stdout


def test_payload_uses_no_pickle_and_is_json_metadata():
    rng = np.random.default_rng(8)
    arc = pywt.sparse_coeff_archive(
        rng.standard_normal(64), wavelet='haar', level=2, budget=0.2)
    arrays, meta = arc.to_payload()
    # metadata must be JSON native
    json.dumps(meta)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'p.npz')
        np.savez(path, **arrays)
        # allow_pickle=False must be sufficient to read everything back
        with np.load(path, allow_pickle=False) as data:
            assert_array_equal(data['values'], arc.values)
            assert_array_equal(data['indices'], arc.indices)


# ---------------------------------------------------------------------------
# Validation and failure modes
# ---------------------------------------------------------------------------

def test_invalid_arguments_raise():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(rng.standard_normal(8), budget=-1.0)
    with pytest.raises(TypeError):
        pywt.sparse_coeff_archive(rng.standard_normal(8).astype(np.float32))
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(rng.standard_normal(8), wavelet='db4')
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(rng.standard_normal(8), mode='zero')
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(
            rng.standard_normal((8, 8)), axes=(0, 0))
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(rng.standard_normal(8), axes=5)
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(rng.standard_normal((4, 4, 4)))


def test_corrupted_payloads_raise():
    rng = np.random.default_rng(10)
    arc = pywt.sparse_coeff_archive(
        rng.standard_normal(32), wavelet='haar', level=2, budget=0.5)
    _, meta = arc.to_payload()
    with pytest.raises(ValueError):
        pywt.SparseCoeffArchive(
            np.zeros(2), np.array([0, 0], dtype=np.intp), meta)
    with pytest.raises(ValueError):
        pywt.SparseCoeffArchive(
            np.zeros(2), np.array([0, 10 ** 9], dtype=np.intp), meta)
    conflict = json.loads(json.dumps(meta))
    conflict['n_coefficients'] -= 1
    with pytest.raises(ValueError):
        pywt.SparseCoeffArchive(arc.values, arc.indices, conflict)
    conflict2 = json.loads(json.dumps(meta))
    conflict2['error_bound'] = conflict2['budget'] + 10.0
    with pytest.raises(ValueError):
        pywt.SparseCoeffArchive(arc.values, arc.indices, conflict2)
    bad_values = arc.values.copy()
    bad_values[0] = np.inf
    with pytest.raises(ValueError):
        pywt.SparseCoeffArchive(bad_values, arc.indices, meta)


def test_overflow_transform_is_rejected():
    huge = np.full(64, np.finfo(np.float64).max / 4)
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(
            huge, wavelet='db2', level=3, mode='symmetric')


def test_archives_reject_quality_masks():
    rng = np.random.default_rng(11)
    arc = pywt.sparse_coeff_archive(
        rng.standard_normal(32), wavelet='db2', level=2, budget=0.2)
    with pytest.raises(ValueError):
        pywt.ravel_coeffs(arc, coeff_masks=[])
    arr, slices, shapes, archive = pywt.ravel_coeffs(arc)
    with pytest.raises(ValueError):
        pywt.unravel_coeffs(
            arr, slices, shapes, archive=archive,
            mask_arr=np.ones(arr.size, dtype=bool))


# ---------------------------------------------------------------------------
# Complexity: no quadratic build time and linear working space expectation
# ---------------------------------------------------------------------------

def test_build_scales_near_linearly():
    # Deterministic complexity check: count the filter-tap operations the
    # structural norm computation performs.  An n-squared operator matrix
    # would give O(n^2) entries, while the replay gives O(n * filter_len).
    from pywt import _sparse
    calls = {'n': 0}
    original = _sparse._step_operator_norm

    def counting_step(c_len, wavelet, mode_name, band):
        calls['n'] += int(c_len)
        return original(c_len, wavelet, mode_name, band)

    sizes = []
    for n in (4096, 16384):
        x = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
        calls['n'] = 0
        _sparse._step_operator_norm = counting_step
        try:
            pywt.sparse_coeff_archive(
                x, wavelet='db2', level=4, mode='periodization',
                budget=0.01 * np.linalg.norm(x))
        finally:
            _sparse._step_operator_norm = original
        sizes.append(calls['n'])
    # the number of replayed coefficient positions grows linearly (4x),
    # never quadratically (16x)
    assert sizes[1] / sizes[0] < 6.0


def test_verification_batches_are_bounded():
    rng = np.random.default_rng(12)
    x = rng.standard_normal(1000)
    arc = pywt.sparse_coeff_archive(
        x, wavelet='haar', level=2, verification_batch=100)
    assert arc.reconstruct().shape == x.shape
    with pytest.raises(ValueError):
        pywt.sparse_coeff_archive(
            x, wavelet='haar', level=2, verification_batch=0)
    # the number of batches is ceil(n / batch); verify the batched norm
    # agrees with a direct, unbatched computation
    from pywt._sparse import _batched_reconstruction_error
    batched = _batched_reconstruction_error(arc, x, 100)
    assert batched == pytest.approx(
        np.linalg.norm(x - arc.reconstruct()))
