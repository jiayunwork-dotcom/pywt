# Copyright (c) 2024 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See LICENSE for more details.

"""
Quality-mask propagation for discrete wavelet transforms.

A boolean *observation mask* marks which samples of an input signal are real
observations (``True``) and which are missing (``False``).  The functions in
this module propagate that information through the same separable filtering,
boundary extension, upsampling and cropping operations that the numerical
transform performs, so every resulting coefficient (and every reconstructed
sample) carries a trustworthy flag derived purely from the *structure* of the
transform.

Design
------
For a fixed wavelet, axis and boundary mode the dependency pattern is the
same for every 1D line.  It is precomputed once as a list of index arrays,
one entry per output position, listing the input positions reached through
non-zero filter taps *after* the actual boundary extension used by the C
implementation.  Consequently:

* repeated references to the same edge sample are counted (the entry simply
  contains that index);
* paths whose numerical contributions happen to cancel are still counted,
  because the dependency is structural rather than numerical;
* zero filter taps (e.g. the leading tap of ``bior2.2``) never create
  dependencies.

The maps are applied with vectorised boolean reductions, giving time and
memory linear in the number of samples and coefficients.

Only real floating point 1D/2D (and in general nD) arrays, the built-in
discrete wavelets and the ``'symmetric'`` and ``'periodization'`` boundary
modes are supported.
"""

import numpy as np

from ._extensions._pywt import Modes, Wavelet

__all__ = [
    'check_mask',
    'dec_dependency_map',
    'rec_dependency_map',
    'propagate_mask_dwt_axis',
    'propagate_mask_idwt_axis',
]

_SUPPORTED_MODES = ('symmetric', 'periodization')
_MODE_ENUM_TO_NAME = {
    Modes.from_object(name): name for name in _SUPPORTED_MODES
}

# Dependency tables depend only on (length, wavelet, mode, band); every 1D
# line of an nD transform shares the same table, so cache them by a hashable
# key.  The cache is bounded to avoid unbounded growth across many lengths.
_TABLE_CACHE = {}
_TABLE_CACHE_MAX = 256


def _cached_table(cache_key, builder):
    """Return the cached master table (read-only use inside this module)."""
    table = _TABLE_CACHE.get(cache_key)
    if table is None:
        table = builder()
        if len(_TABLE_CACHE) >= _TABLE_CACHE_MAX:
            # drop the oldest entry (FIFO is adequate for transform workloads
            # where lengths typically repeat within one call chain)
            _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
        _TABLE_CACHE[cache_key] = table
    return table


def _mode_name(mode):
    if isinstance(mode, str):
        name = mode
    elif mode in _MODE_ENUM_TO_NAME:
        name = _MODE_ENUM_TO_NAME[mode]
    else:
        try:
            name = Modes.from_object(mode)
        except Exception:
            raise ValueError(
                f"mask propagation supports only modes {_SUPPORTED_MODES}, "
                f"got {mode!r}")
    if name not in _SUPPORTED_MODES:
        raise ValueError(
            f"mask propagation supports only modes {_SUPPORTED_MODES}, "
            f"got {name!r}")
    return name


def check_mask(mask, shape):
    """Validate a user supplied observation mask.

    Parameters
    ----------
    mask : array_like or None
        Boolean array, ``True`` where a real observation exists.
    shape : tuple of int
        Required shape (the shape of the matching data array).

    Returns
    -------
    mask : ndarray of bool
        A boolean ndarray with the requested shape.
    """
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.dtype != bool:
        raise TypeError(
            "mask must be a boolean array (dtype bool); got dtype "
            f"{mask.dtype!r}")
    if mask.shape != tuple(shape):
        raise ValueError(
            f"mask shape {mask.shape} does not match data shape "
            f"{tuple(shape)}")
    return mask


def check_finite(data, mask):
    """Raise if a trustworthy (observed) cell holds a non-finite value.

    Missing cells may contain arbitrary placeholders (stale readings, NaN or
    infinity); the numerical transform always sees the missing positions as
    zero regardless of the placeholder.
    """
    if mask is None:
        return
    finite = np.isfinite(data)
    bad = np.count_nonzero(~finite & mask)
    if bad:
        raise ValueError(
            f"{bad} trustworthy mask position(s) contain non-finite values; "
            "only missing (mask == False) cells may hold NaN or infinity")


def _nonzero_taps(filter_coeffs):
    return np.flatnonzero(np.abs(np.asarray(filter_coeffs)) > 0)


def _symmetric_extended_index(position, n):
    """Original-sample index seen at ``position`` under symmetric extension.

    Half-sample symmetric mirroring with edge repetition, matching the
    ``MODE_SYMMETRIC`` behaviour of the C convolution routines.
    """
    period = 2 * n
    q = int(position) % period
    return q if q < n else period - 1 - q


def dec_dependency_map(input_len, wavelet, mode, band):
    """Input-index dependency table for one 1D decomposition step.

    Parameters
    ----------
    input_len : int
        Number of input samples.
    wavelet : Wavelet
    mode : {'symmetric', 'periodization'}
    band : {'a', 'd'}
        Approximation or detail filter bank.

    Returns
    -------
    table : list of ndarray
        ``table[o]`` are the input positions that coefficient ``o`` depends
        on through non-zero taps of the actual boundary-extended filter.
    """
    if not isinstance(wavelet, Wavelet):
        raise TypeError("wavelet must be a pywt.Wavelet instance")
    mode = _mode_name(mode)
    key = ('dec', input_len, wavelet.name, mode, band)
    table = _cached_table(key, lambda: _build_dec_map(
        input_len, wavelet, mode, band))
    # defensive copies: callers must not be able to corrupt the cache
    return [np.copy(entry) for entry in table]


def _build_dec_map(input_len, wavelet, mode, band):
    filt = wavelet.dec_lo if band == 'a' else wavelet.dec_hi
    f_len = len(filt)
    taps = _nonzero_taps(filt)

    if mode == 'periodization':
        padded_len = input_len + (input_len % 2)
        base = np.arange(padded_len)
        base = np.where(base >= input_len, input_len - 1, base)
        out_len = padded_len // 2
        table = []
        for o in range(out_len):
            i = 2 * o + f_len // 2
            table.append(np.unique(base[(i - taps) % padded_len]).astype(np.intp))
        return table

    out_len = (input_len + f_len - 1) // 2
    table = []
    for o in range(out_len):
        i = 2 * o + 1
        idx = [_symmetric_extended_index(i - j, input_len) for j in taps]
        table.append(np.unique(np.asarray(idx, dtype=np.intp)))
    return table


def rec_dependency_map(coeff_len, wavelet, mode, band):
    """Coefficient-index dependency table for one 1D reconstruction step.

    Parameters
    ----------
    coeff_len : int
        Number of coefficients fed to the reconstruction filter.
    wavelet : Wavelet
    mode : {'symmetric', 'periodization'}
    band : {'a', 'd'}
        Approximation or detail reconstruction filter.

    Returns
    -------
    table : list of ndarray
        ``table[q]`` are the coefficient positions that reconstructed sample
        ``q`` depends on through non-zero reconstruction taps.
    """
    if not isinstance(wavelet, Wavelet):
        raise TypeError("wavelet must be a pywt.Wavelet instance")
    mode = _mode_name(mode)
    key = ('rec', coeff_len, wavelet.name, mode, band)
    table = _cached_table(key, lambda: _build_rec_map(
        coeff_len, wavelet, mode, band))
    return [np.copy(entry) for entry in table]


def _dec_map_internal(input_len, wavelet, mode, band):
    """Cached master table for read-only use on the propagation hot path."""
    mode = _mode_name(mode)
    key = ('dec', input_len, wavelet.name, mode, band)
    return _cached_table(key, lambda: _build_dec_map(
        input_len, wavelet, mode, band))


def _rec_map_internal(coeff_len, wavelet, mode, band):
    """Cached master table for read-only use on the propagation hot path."""
    mode = _mode_name(mode)
    key = ('rec', coeff_len, wavelet.name, mode, band)
    return _cached_table(key, lambda: _build_rec_map(
        coeff_len, wavelet, mode, band))


def _build_rec_map(coeff_len, wavelet, mode, band):
    filt = wavelet.rec_lo if band == 'a' else wavelet.rec_hi
    f_len = len(filt)
    taps = _nonzero_taps(filt)

    if mode == 'periodization':
        return _rec_periodization_map(coeff_len, f_len, taps)

    rec_len = 2 * coeff_len - f_len + 2
    if rec_len < 1:
        # Such inputs are rejected by idwt itself; no well-formed multilevel
        # transform reaches this branch.
        raise ValueError(
            "reconstruction buffer length would be non-positive for the "
            "given coefficient count and wavelet")
    table = [[] for _ in range(rec_len)]
    out_cursor = 0
    for i in range(f_len // 2 - 1, coeff_len):
        for j in taps:
            k = j // 2
            parity = j % 2
            q = out_cursor + parity
            ci = i - k
            if 0 <= ci < coeff_len and 0 <= q < rec_len:
                table[q].append(ci)
        out_cursor += 2
    return [np.unique(np.asarray(idx, dtype=np.intp)) for idx in table]


def _rec_periodization_map(coeff_len, f_len, taps):
    """Structural translation of the periodization IDWT convolution."""
    f_half = f_len // 2
    start = f_len // 4
    end = coeff_len + start - (0 if f_half % 2 else 1)
    out_len = 2 * coeff_len
    table = [set() for _ in range(out_len)]
    tap_set = set(int(t) for t in taps)

    def emit(q, ci):
        if 0 <= ci < coeff_len:
            table[q % out_len].add(ci)

    if f_half % 2 == 0:
        # preliminary shift block writing outputs 2*N-1 and 0
        j = 0
        while j <= start - 1:
            for k in range(coeff_len):
                if j > start - 1:
                    break
                if 2 * (start - 1 - j) in tap_set:
                    table[out_len - 1].add(k)
                if 2 * (start - 1 - j) + 1 in tap_set:
                    table[0].add(k)
                j += 1
        while j <= coeff_len + start - 1 and j < f_half:
            ci = coeff_len + start - 1 - j
            if 2 * j in tap_set:
                table[out_len - 1].add(ci)
            if 2 * j + 1 in tap_set:
                table[0].add(ci)
            j += 1
        while j < f_half:
            for k in range(coeff_len):
                if j >= f_half:
                    break
                if 2 * j in tap_set:
                    table[out_len - 1].add(coeff_len - 1 - k)
                if 2 * j + 1 in tap_set:
                    table[0].add(coeff_len - 1 - k)
                j += 1

    i = start
    o = 1 if f_half % 2 == 0 else 0

    def emit_pair(ci, first_tap):
        if first_tap in tap_set:
            emit(o, ci)
        if first_tap + 1 in tap_set:
            emit(o + 1, ci)

    while i < f_half and i < coeff_len:
        for j in range(i + 1):
            emit_pair(i - j, 2 * j)
        j = i + 1
        while j < f_half:
            for k in range(coeff_len):
                if j >= f_half:
                    break
                emit_pair(coeff_len - 1 - k, 2 * j)
                j += 1
        i += 1
        o += 2
    while i < coeff_len:
        for j in range(f_half):
            emit_pair(i - j, 2 * j)
        i += 1
        o += 2
    while i < f_half and i < end:
        j = 0
        while i - j >= coeff_len:
            for k in range(coeff_len):
                if i - j < coeff_len:
                    break
                emit_pair(k, 2 * (i - coeff_len - j))
                j += 1
        while j <= i and j < f_half:
            emit_pair(i - j, 2 * j)
            j += 1
        while j < f_half:
            for k in range(coeff_len):
                if j >= f_half:
                    break
                emit_pair(coeff_len - 1 - k, 2 * j)
                j += 1
        i += 1
        o += 2
    while i < end:
        j = 0
        while i - j >= coeff_len:
            for k in range(coeff_len):
                if i - j < coeff_len:
                    break
                emit_pair(k, 2 * (i - coeff_len - j))
                j += 1
        while j <= i and j < f_half:
            emit_pair(i - j, 2 * j)
            j += 1
        i += 1
        o += 2

    return [np.unique(np.asarray(sorted(s), dtype=np.intp)) for s in table]


def _apply_table_along_axis(mask, table, axis):
    """Apply a 1D dependency table along ``axis`` of an nD boolean array."""
    moved = np.moveaxis(mask, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = np.empty((len(table), flat.shape[1]), dtype=bool)
    for o, idx in enumerate(table):
        if idx.size:
            np.all(flat[idx, :], axis=0, out=out[o])
        else:
            # no structural dependency: nothing can invalidate this output
            out[o].fill(True)
    out = out.reshape((len(table),) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def propagate_mask_dwt_axis(mask, wavelet, mode, axis, band):
    """Propagate a mask through one decomposition filter along one axis."""
    table = _dec_map_internal(mask.shape[axis], wavelet, mode, band)
    return _apply_table_along_axis(mask, table, axis)


def propagate_mask_idwt_axis(mask_a, mask_d, wavelet, mode, axis,
                             output_shape):
    """Propagate masks through one reconstruction step along one axis.

    Either of ``mask_a`` / ``mask_d`` may be ``None`` when the corresponding
    coefficient band was omitted (treated as zero by ``idwt``).
    ``output_shape`` is the shape actually produced by ``idwt_axis`` for the
    given inputs.
    """
    if mask_a is None and mask_d is None:
        return np.ones(output_shape, dtype=bool)
    table_a = (_rec_map_internal(mask_a.shape[axis], wavelet, mode, 'a')
               if mask_a is not None else None)
    table_d = (_rec_map_internal(mask_d.shape[axis], wavelet, mode, 'd')
               if mask_d is not None else None)
    rec_len = len(table_a) if table_a is not None else len(table_d)
    trust = np.ones(output_shape, dtype=bool)
    moved_trust = np.moveaxis(trust, axis, 0)
    tflat = moved_trust.reshape(rec_len, -1)

    for m, table in ((mask_a, table_a), (mask_d, table_d)):
        if m is None:
            continue
        moved = np.moveaxis(m, axis, 0)
        # bands at an odd-length level differ by one along the axis; the
        # extra approximation entry is simply not referenced by this filter
        flat = moved.reshape(moved.shape[0], -1)
        for q, idx in enumerate(table):
            if idx.size:
                tflat[q] &= np.all(flat[idx, :], axis=0)
    return trust
