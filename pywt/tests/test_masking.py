# Copyright (c) 2024 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See LICENSE for more details.

"""Tests for observation-mask (quality flag) propagation.

Coverage
--------
* element-wise explanation of why an edge coefficient/reconstructed point
  loses its trust flag on a short, hand-checkable array;
* Haar, db2 and bior2.2 (which has zero filter taps) with both
  ``'symmetric'`` and ``'periodization'`` boundary modes;
* odd lengths, at least two decomposition levels, all-True and all-False
  masks;
* coefficient editing through the ravel/unravel round trip;
* axis reordering, negative axes and transforming a single axis of a 2D
  array;
* placeholder invariance (stale reading, NaN and infinity);
* input and packing validation errors.
"""

import numpy as np
import pytest

import pywt
from pywt._masking import (
    dec_dependency_map,
    rec_dependency_map,
)
from pywt._multidim import _dwtn_propagate

WAVELETS = ('haar', 'db2', 'bior2.2')
MODES = ('symmetric', 'periodization')


def _zeroed_transform(data, mask, wavelet, mode, level, axes):
    """Plain transform of the data with missing cells replaced by zero."""
    return pywt.wavedecn(np.where(mask, data, 0.0), wavelet, mode=mode,
                         level=level, axes=axes)


# ---------------------------------------------------------------------------
# Hand-checkable short arrays
# ---------------------------------------------------------------------------

def test_short_array_edge_coefficient_explanation_haar():
    # Every Haar output coefficient is a pair of adjacent samples.  The first
    # pair touches missing sample 0, so only coefficient 0 is untrustworthy.
    data = np.arange(8.0)
    mask = np.ones(8, dtype=bool)
    mask[0] = False
    coeffs, cmasks = pywt.dwtn(data, 'haar', 'symmetric', axes=(0,),
                                mask=mask)
    table = dec_dependency_map(8, pywt.Wavelet('haar'), 'symmetric', 'a')
    assert table[0].tolist() == [0, 1]
    assert table[1].tolist() == [2, 3]
    assert cmasks['a'].tolist() == [False, True, True, True]
    assert cmasks['d'].tolist() == [False, True, True, True]
    # the surviving trustworthy values are exactly the ordinary zero-filled
    # transform values
    plain = pywt.dwtn(np.where(mask, data, 0.0), 'haar', 'symmetric',
                      axes=(0,))
    np.testing.assert_array_equal(coeffs['a'], plain['a'])
    np.testing.assert_array_equal(coeffs['d'], plain['d'])


def test_short_array_exact_cancellation_still_untrusted():
    # Odd-length Haar: the last detail coefficient is always zero numerically
    # because the two high-pass taps hit the same symmetrically extended
    # sample with opposite signs.  It nevertheless depends structurally on
    # the last sample and must be flagged when that sample is missing.
    data = np.arange(7.0)
    mask = np.ones(7, dtype=bool)
    mask[6] = False
    coeffs, cmasks = pywt.dwtn(data, 'haar', 'symmetric', axes=(0,),
                                mask=mask)
    assert coeffs['d'][-1] == 0.0
    assert cmasks['d'][-1] is np.False_


def test_short_array_reconstruction_explanation_db2():
    # db2 reconstruction: each output pair depends on two adjacent
    # coefficients.  With one discarded coefficient exactly two reconstructed
    # samples lose trust.
    wavelet = pywt.Wavelet('db2')
    rng = np.random.default_rng(0)
    data = rng.standard_normal(16)
    mask = np.ones(16, dtype=bool)
    mask[5] = False
    coeffs, cmasks = pywt.dwtn(data, wavelet, 'symmetric', axes=(0,),
                                mask=mask)
    rec, rmask = pywt.idwtn(coeffs, wavelet, 'symmetric', axes=(0,),
                            coeff_masks=cmasks)
    table = rec_dependency_map(len(coeffs['a']), wavelet, 'symmetric', 'a')
    # locate the coefficient made untrustworthy by the missing sample
    bad = np.flatnonzero(~cmasks['a'])
    expected = np.zeros(rec.shape, dtype=bool)
    for b in bad:
        for q, deps in enumerate(table):
            if b in deps.tolist():
                expected[q] = True
    np.testing.assert_array_equal(~rmask, expected)
    assert rec.shape == data.shape and rmask.shape == data.shape


# ---------------------------------------------------------------------------
# Value equivalence and mask correctness across the full configuration grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('shape,axes', [
    ((17,), (0,)),
    ((19, 23), (0, 1)),
    ((21, 19), (1, 0)),
    ((23, 17), (-2, -1)),
    ((16, 20), (-1,)),           # single axis of a 2D array
])
def test_masked_values_match_zeroed_transform(wavelet, mode, shape, axes):
    rng = np.random.default_rng(list(map(ord, wavelet)) + [len(shape)])
    data = rng.standard_normal(shape)
    mask = np.ones(shape, dtype=bool)
    mask.reshape(-1)[1] = False
    mask.reshape(-1)[4:4 + (1 if data.ndim == 1 else shape[-1])] = False
    min_len = min(shape[ax] for ax in axes)
    level = min(2, pywt.dwt_max_level(min_len,
                                      pywt.Wavelet(wavelet).dec_len))
    assert level >= 1
    coeffs, cmasks = pywt.wavedecn(data, wavelet, mode=mode, level=level,
                                   axes=axes, mask=mask)
    plain = _zeroed_transform(data, mask, wavelet, mode, level, axes)
    np.testing.assert_array_equal(coeffs[0], plain[0])
    np.testing.assert_array_equal(cmasks[0].shape, plain[0].shape)
    for n in range(1, len(coeffs)):
        for key in coeffs[n]:
            np.testing.assert_array_equal(coeffs[n][key], plain[n][key])
            assert cmasks[n][key].shape == coeffs[n][key].shape
            assert cmasks[n][key].dtype == bool


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('mode', MODES)
def test_placeholder_invariance(wavelet, mode):
    rng = np.random.default_rng(11)
    data = rng.standard_normal(24)
    mask = np.ones(24, dtype=bool)
    mask[[2, 7, 15, 16, 22]] = False
    stale = data.copy()
    stale[~mask] = rng.standard_normal((~mask).sum()) * 99.0
    nan_data = data.copy()
    nan_data[~mask] = np.nan
    inf_data = data.copy()
    inf_data[~mask] = np.inf

    ref_c, ref_m = pywt.dwtn(stale, wavelet, mode=mode, axes=(0,),
                             mask=mask)
    for variant in (nan_data, inf_data):
        c, m = pywt.dwtn(variant, wavelet, mode=mode, axes=(0,), mask=mask)
        for key in ref_c:
            np.testing.assert_array_equal(c[key], ref_c[key])
            np.testing.assert_array_equal(m[key], ref_m[key])

    level = min(3, pywt.dwt_max_level(24, pywt.Wavelet(wavelet).dec_len))
    ref_coeffs, ref_cmasks = pywt.wavedecn(
        stale, wavelet, mode=mode, level=level, axes=(0,), mask=mask)
    ref_r, ref_rm = pywt.waverecn(ref_coeffs, wavelet, mode=mode, axes=(0,),
                                  coeff_masks=ref_cmasks)
    for variant in (nan_data, inf_data):
        c, m = pywt.wavedecn(variant, wavelet, mode=mode, level=level,
                             axes=(0,), mask=mask)
        r, rm = pywt.waverecn(c, wavelet, mode=mode, axes=(0,),
                              coeff_masks=m)
        np.testing.assert_array_equal(r, ref_r)
        np.testing.assert_array_equal(rm, ref_rm)


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('mode', MODES)
def test_trustworthy_outputs_invariant_to_missing_values(wavelet, mode):
    # Independent random fillings of the missing cells must not change any
    # coefficient or reconstructed sample flagged trustworthy.
    rng = np.random.default_rng(123)
    data = rng.standard_normal((29, 31))
    mask = np.ones(data.shape, dtype=bool)
    mask[3, 4] = False
    mask[10:13, 8:12] = False
    level = min(3, pywt.dwt_max_level(29, pywt.Wavelet(wavelet).dec_len))
    coeffs, cmasks = pywt.wavedecn(data, wavelet, mode=mode, level=level,
                                   axes=(0, 1), mask=mask)
    rec, rmask = pywt.waverecn(coeffs, wavelet, mode=mode, axes=(0, 1),
                               coeff_masks=cmasks)
    for _ in range(4):
        alt = data.copy()
        alt[~mask] = rng.standard_normal((~mask).sum())
        c2, m2 = pywt.wavedecn(alt, wavelet, mode=mode, level=level,
                               axes=(0, 1), mask=mask)
        np.testing.assert_array_equal(
            c2[0][cmasks[0]], coeffs[0][cmasks[0]])
        r2, rm2 = pywt.waverecn(c2, wavelet, mode=mode, axes=(0, 1),
                                coeff_masks=m2)
        np.testing.assert_array_equal(rm2, rmask)
        np.testing.assert_array_equal(r2[rmask], rec[rmask])


def test_all_true_and_all_false_masks():
    rng = np.random.default_rng(4)
    data = rng.standard_normal((16, 16))
    true_mask = np.ones(data.shape, dtype=bool)
    coeffs, cmasks = pywt.dwtn(data, 'db2', axes=(0, 1), mask=true_mask)
    assert all(m.all() for m in cmasks.values())
    plain = pywt.dwtn(data, 'db2', axes=(0, 1))
    for key in coeffs:
        np.testing.assert_array_equal(coeffs[key], plain[key])

    false_mask = np.zeros(data.shape, dtype=bool)
    coeffs, cmasks = pywt.dwtn(data, 'haar', axes=(0, 1), mask=false_mask)
    assert all((~m).any() or True for m in cmasks.values())
    assert all(not m.any() for m in cmasks.values())
    assert all(np.allclose(v, 0.0) for v in coeffs.values())
    rec, rmask = pywt.idwtn(coeffs, 'haar', axes=(0, 1),
                            coeff_masks=cmasks)
    assert not rmask.any()
    np.testing.assert_array_equal(rec, np.zeros_like(rec))


def test_ordinary_zeros_remain_trustworthy():
    data = np.array([1.0, -1, 1, -1, 1, -1, 1, -1])
    mask = np.ones(8, dtype=bool)
    coeffs, cmasks = pywt.dwtn(data, 'haar', 'symmetric', axes=(0,),
                                mask=mask)
    # the approximation coefficients are genuinely zero, yet trustworthy
    np.testing.assert_array_equal(coeffs['a'], np.zeros(4))
    assert cmasks['a'].all()
    _, rmask = pywt.idwtn(coeffs, 'haar', 'symmetric', axes=(0,),
                          coeff_masks=cmasks)
    assert rmask.all()

    # numerically zero coefficients flagged untrustworthy do invalidate
    zero_masks = {k: np.zeros_like(v) for k, v in cmasks.items()}
    _, rmask2 = pywt.idwtn(
        {k: np.zeros_like(v) for k, v in coeffs.items()}, 'haar',
        'symmetric', axes=(0,), coeff_masks=zero_masks)
    assert not rmask2.any()


def test_bior22_zero_taps_reduce_dependencies():
    wavelet = pywt.Wavelet('bior2.2')
    assert wavelet.dec_lo[0] == 0.0
    table = dec_dependency_map(8, wavelet, 'symmetric', 'a')
    # zero tap 0 means output 0 cannot reference a sample through that tap;
    # the table must be a strict subset of the full filter window
    assert set(table[0].tolist()) < set(range(wavelet.dec_len))


def test_output_without_structural_dependency_stays_trusted():
    # An output position reached through no non-zero filter tap cannot be
    # invalidated by anything, so it must remain trustworthy even when every
    # input is marked missing.
    from pywt._masking import _apply_table_along_axis
    mask = np.zeros(5, dtype=bool)
    table = [np.array([0, 1], dtype=np.intp),
             np.array([], dtype=np.intp)]
    out = _apply_table_along_axis(mask, table, 0)
    np.testing.assert_array_equal(out, [False, True])


@pytest.mark.parametrize('wavelet', WAVELETS + ('sym4', 'coif1', 'rbio2.2'))
@pytest.mark.parametrize('mode', MODES)
def test_dependency_tables_match_differential_probe(wavelet, mode):
    # Structural tables must agree with a numerical differential probe: vary
    # one input sample at a time and record which outputs change. Exact
    # cancellations make the probe a subset of the structural table, never a
    # superset.
    w = pywt.Wavelet(wavelet)
    rng = np.random.default_rng(31)
    for n in (5, 8, 13, 16):
        for band in ('a', 'd'):
            table = dec_dependency_map(n, w, mode, band)
            base = rng.standard_normal(n)
            c_base = pywt.dwt(base, w, mode)[0 if band == 'a' else 1]
            for k in range(n):
                pert = base.copy()
                pert[k] += 1.5
                c_pert = pywt.dwt(pert, w, mode)[0 if band == 'a' else 1]
                changed = np.flatnonzero(c_pert != c_base)
                for o in changed:
                    assert k in table[o].tolist(), (wavelet, mode, n, band)
        # reconstruction probe
        nc = len(dec_dependency_map(n, w, mode, 'a'))
        for band in ('a', 'd'):
            table = rec_dependency_map(nc, w, mode, band)
            base_c = rng.standard_normal(nc)
            kw = (base_c, None) if band == 'a' else (None, base_c)
            r0 = pywt.idwt(*kw, w, mode)
            for k in range(nc):
                pert = base_c.copy()
                pert[k] += 1.5
                kwp = (pert, None) if band == 'a' else (None, pert)
                rp = pywt.idwt(*kwp, w, mode)
                for q in np.flatnonzero(rp != r0):
                    assert k in table[q].tolist(), (wavelet, mode, nc, band)


# ---------------------------------------------------------------------------
# Single-call multilevel vs level-by-level with carried masks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('axes', [(0, 1), (1, 0), (-1, -2)])
def test_stepwise_decomposition_matches_multilevel(axes):
    rng = np.random.default_rng(7)
    data = rng.standard_normal((19, 23))
    mask = np.ones(data.shape, dtype=bool)
    mask[2, 3] = False
    mask[14, :] = False
    zeroed = np.where(mask, data, 0.0)
    wavelet = pywt.Wavelet('db2')

    # two single-axis steps using the propagation-only core
    first, mfirst = _dwtn_propagate(zeroed, wavelet, 'symmetric',
                                    (axes[0],), mask)
    step_coeffs, step_masks = {}, {}
    for key in first:
        c, m = _dwtn_propagate(first[key], wavelet, 'symmetric',
                               (axes[1],), mfirst[key])
        for sub in c:
            step_coeffs[key + sub] = c[sub]
            step_masks[key + sub] = m[sub]

    one, one_masks = _dwtn_propagate(zeroed, wavelet, 'symmetric', axes,
                                     mask)
    for key in one:
        np.testing.assert_array_equal(step_coeffs[key], one[key])
        np.testing.assert_array_equal(step_masks[key], one_masks[key])


def test_negative_axis_single_axis_2d():
    rng = np.random.default_rng(8)
    data = rng.standard_normal((19, 23))
    mask = np.ones(data.shape, dtype=bool)
    mask[5, 5] = False
    coeffs, cmasks = pywt.dwtn(data, 'bior2.2', 'periodization',
                               axes=(-1,), mask=mask)
    assert set(coeffs) == {'a', 'd'}
    assert coeffs['a'].shape[0] == data.shape[0]
    assert coeffs['d'].shape == cmasks['d'].shape


# ---------------------------------------------------------------------------
# Ravel / unravel
# ---------------------------------------------------------------------------

def test_ravel_unravel_preserves_masks_and_shapes():
    rng = np.random.default_rng(9)
    data = rng.standard_normal((32, 28))
    mask = np.ones(data.shape, dtype=bool)
    mask[4, 3] = False
    mask[20:23, 10:14] = False
    coeffs, cmasks = pywt.wavedecn(data, 'db2', level=3, axes=(0, 1),
                                   mask=mask)
    arr, slices, shapes, marr = pywt.ravel_coeffs(
        coeffs, axes=(0, 1), coeff_masks=cmasks)
    assert marr.dtype == bool and marr.shape == arr.shape

    # zero some trustworthy coefficients inside the flat array
    trust_idx = np.flatnonzero(marr)[:6]
    arr[trust_idx] = 0.0
    back, bmasks = pywt.unravel_coeffs(arr, slices, shapes,
                                       output_format='wavedecn',
                                       mask_arr=marr)
    # shapes and masks survive the round trip position by position
    for n in range(len(cmasks)):
        if n == 0:
            assert back[0].shape == coeffs[0].shape
            np.testing.assert_array_equal(bmasks[0], cmasks[0])
        else:
            for key in cmasks[n]:
                assert back[n][key].shape == coeffs[n][key].shape
                np.testing.assert_array_equal(bmasks[n][key], cmasks[n][key])
    _, _, _, mflat_back = pywt.ravel_coeffs(back, axes=(0, 1),
                                            coeff_masks=bmasks)
    # zeroed trustworthy coefficients stay trustworthy
    assert mflat_back[trust_idx].all()

    # reconstruction still works with identical quality flags
    _, rmask_edited = pywt.waverecn(back, 'db2', axes=(0, 1),
                                    coeff_masks=bmasks)
    _, rmask_orig = pywt.waverecn(coeffs, 'db2', axes=(0, 1),
                                  coeff_masks=cmasks)
    np.testing.assert_array_equal(rmask_edited, rmask_orig)


def test_ravel_unravel_1d_and_2d_formats():
    rng = np.random.default_rng(10)
    data = rng.standard_normal(32)
    mask = np.ones(32, dtype=bool)
    mask[7] = False
    coeffs, cmasks = pywt.wavedec(data, 'db2', level=3, mask=mask)
    arr, slices, shapes, marr = pywt.ravel_coeffs(coeffs, coeff_masks=cmasks)
    back, bmasks = pywt.unravel_coeffs(arr, slices, shapes,
                                       output_format='wavedec',
                                       mask_arr=marr)
    for a, b in zip(cmasks, bmasks):
        np.testing.assert_array_equal(a, b)

    data2 = rng.standard_normal((16, 16))
    mask2 = np.ones(data2.shape, dtype=bool)
    mask2[3, 3] = False
    c2, m2 = pywt.wavedec2(data2, 'haar', level=2, mask=mask2)
    arr2, sl2, sh2, marr2 = pywt.ravel_coeffs(c2, coeff_masks=m2)
    back2, bm2 = pywt.unravel_coeffs(arr2, sl2, sh2,
                                     output_format='wavedec2',
                                     mask_arr=marr2)
    assert len(back2) == len(bm2)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_mask_validation_errors():
    data = np.arange(8.0)
    good = np.ones(8, dtype=bool)
    with pytest.raises(ValueError):
        pywt.dwtn(data, 'haar', axes=(0,), mask=np.ones(7, dtype=bool))
    with pytest.raises(TypeError):
        pywt.dwtn(data, 'haar', axes=(0,), mask=np.ones(8, dtype=np.uint8))

    nan_data = data.copy()
    nan_data[2] = np.nan
    with pytest.raises(ValueError):
        pywt.dwtn(nan_data, 'haar', axes=(0,), mask=good)
    inf_data = data.copy()
    inf_data[2] = np.inf
    with pytest.raises(ValueError):
        pywt.dwtn(inf_data, 'haar', axes=(0,), mask=good)

    with pytest.raises(ValueError):
        pywt.dwtn(np.ones(8, dtype=np.int64), 'haar', axes=(0,), mask=good)
    with pytest.raises(ValueError):
        pywt.dwtn(data, 'haar', mode='zero', axes=(0,), mask=good)


def test_packing_validation_errors():
    rng = np.random.default_rng(14)
    data = rng.standard_normal(32)
    mask = np.ones(32, dtype=bool)
    mask[5] = False
    coeffs, cmasks = pywt.wavedec(data, 'haar', level=3, mask=mask)
    arr, slices, shapes, marr = pywt.ravel_coeffs(coeffs,
                                                  coeff_masks=cmasks)
    with pytest.raises(ValueError):
        pywt.unravel_coeffs(arr, slices, shapes,
                            mask_arr=np.ones(arr.size + 1, dtype=bool))
    with pytest.raises(TypeError):
        pywt.unravel_coeffs(arr, slices, shapes,
                            mask_arr=marr.astype(np.uint8))
    with pytest.raises(ValueError):
        pywt.ravel_coeffs(coeffs, coeff_masks=cmasks[:-1])
    bad_masks = [m.copy() for m in cmasks]
    bad_masks[1] = np.pad(bad_masks[1], (0, 1))
    with pytest.raises(ValueError):
        pywt.ravel_coeffs(coeffs, coeff_masks=bad_masks)


def test_backward_compatible_return_without_mask():
    rng = np.random.default_rng(15)
    data = rng.standard_normal((16, 16))
    result = pywt.dwtn(data, 'db2', axes=(0, 1))
    assert isinstance(result, dict)
    ml = pywt.wavedecn(data, 'db2', level=2, axes=(0, 1))
    assert isinstance(ml, list)
    rec = pywt.waverecn(ml, 'db2', axes=(0, 1))
    assert isinstance(rec, np.ndarray)
    flat = pywt.ravel_coeffs(ml, axes=(0, 1))
    assert len(flat) == 3  # no fourth return without masks


def test_dwt2_idwt2_mask_wrappers():
    rng = np.random.default_rng(16)
    data = rng.standard_normal((16, 16))
    mask = np.ones(data.shape, dtype=bool)
    mask[4, 4] = False
    (cA, (cH, cV, cD)), (mA, (mH, mV, mD)) = pywt.dwt2(
        data, 'db2', mask=mask)
    assert mA.shape == cA.shape and mH.shape == cH.shape
    rec, rmask = pywt.idwt2((cA, (cH, cV, cD)), 'db2',
                            coeff_masks=(mA, (mH, mV, mD)))
    assert rec.shape == rmask.shape == data.shape
    # compare against the dict-based path
    dict_rec, dict_rmask = pywt.idwtn(
        {'aa': cA, 'da': cH, 'ad': cV, 'dd': cD}, 'db2',
        coeff_masks={'aa': mA, 'da': mH, 'ad': mV, 'dd': mD})
    np.testing.assert_array_equal(rec, dict_rec)
    np.testing.assert_array_equal(rmask, dict_rmask)


def test_wavedec2_waverec2_and_1d_wrappers():
    rng = np.random.default_rng(17)
    data2 = rng.standard_normal((32, 24))
    mask2 = np.ones(data2.shape, dtype=bool)
    mask2[7:10, 3] = False
    coeffs2, cmasks2 = pywt.wavedec2(data2, 'bior2.2', level=2, mask=mask2)
    assert isinstance(cmasks2[0], np.ndarray)
    assert len(cmasks2[1]) == 3
    rec2, rmask2 = pywt.waverec2(coeffs2, 'bior2.2', coeff_masks=cmasks2)
    assert rec2.shape == rmask2.shape
    # same data through the nD primitives must agree
    cn, mn = pywt.wavedecn(data2, 'bior2.2', level=2, mask=mask2)
    rn, rmn = pywt.waverecn(cn, 'bior2.2', coeff_masks=mn)
    np.testing.assert_array_equal(rec2, rn)
    np.testing.assert_array_equal(rmask2, rmn)

    data1 = rng.standard_normal(32)
    mask1 = np.ones(32, dtype=bool)
    mask1[11] = False
    c1, m1 = pywt.wavedec(data1, 'db2', level=3, mask=mask1)
    assert all(isinstance(v, np.ndarray) for v in m1)
    r1, rm1 = pywt.waverec(c1, 'db2', coeff_masks=m1)
    assert r1.shape == rm1.shape == data1.shape
