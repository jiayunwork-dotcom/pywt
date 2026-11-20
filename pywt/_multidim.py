# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# Copyright (c) 2012-2016 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See COPYING for license details.

"""
2D and nD Discrete Wavelet Transforms and Inverse Discrete Wavelet Transforms.
"""


from itertools import product

import numpy as np

from ._c99_config import _have_c99_complex
from ._extensions._dwt import dwt_axis, idwt_axis
from ._masking import (
    check_finite,
    check_mask,
    propagate_mask_dwt_axis,
    propagate_mask_idwt_axis,
)
from ._utils import AxisError, _modes_per_axis, _wavelets_per_axis

__all__ = ['dwt2', 'idwt2', 'dwtn', 'idwtn']


def dwt2(data, wavelet, mode='symmetric', axes=(-2, -1), mask=None):
    """
    2D Discrete Wavelet Transform.

    Parameters
    ----------
    data : array_like
        2D array with input data
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or 2-tuple of strings, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple of modes specifying the mode to use on each axis in
        ``axes``.
    axes : 2-tuple of ints, optional
        Axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes.
    mask : array_like of bool, optional
        Observation mask, see :func:`dwtn`. When given, trust masks
        ``(mA, (mH, mV, mD))`` are returned as an additional second result.

    Returns
    -------
    (cA, (cH, cV, cD)) : tuple
        Approximation, horizontal detail, vertical detail and diagonal
        detail coefficients respectively.  Horizontal refers to array axis 0
        (or ``axes[0]`` for user-specified ``axes``).

    Examples
    --------
    >>> import numpy as np
    >>> import pywt
    >>> data = np.ones((4,4), dtype=np.float64)
    >>> coeffs = pywt.dwt2(data, 'haar')
    >>> cA, (cH, cV, cD) = coeffs
    >>> cA
    array([[ 2.,  2.],
           [ 2.,  2.]])
    >>> cV
    array([[ 0.,  0.],
           [ 0.,  0.]])

    """
    axes = tuple(axes)
    data = np.asarray(data)
    if len(axes) != 2:
        raise ValueError("Expected 2 axes")
    if data.ndim < len(np.unique(axes)):
        raise ValueError("Input array has fewer dimensions than the specified "
                         "axes")

    if mask is None:
        coefs = dwtn(data, wavelet, mode, axes)
        return coefs['aa'], (coefs['da'], coefs['ad'], coefs['dd'])
    coefs, cmasks = dwtn(data, wavelet, mode, axes, mask=mask)
    return ((coefs['aa'], (coefs['da'], coefs['ad'], coefs['dd'])),
            (cmasks['aa'],
             (cmasks['da'], cmasks['ad'], cmasks['dd'])))


def idwt2(coeffs, wavelet, mode='symmetric', axes=(-2, -1),
          coeff_masks=None):
    """
    2-D Inverse Discrete Wavelet Transform.

    Reconstructs data from coefficient arrays.

    Parameters
    ----------
    coeffs : tuple
        (cA, (cH, cV, cD)) A tuple with approximation coefficients and three
        details coefficients 2D arrays like from ``dwt2``.  If any of these
        components are set to ``None``, it will be treated as zeros.
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or 2-tuple of strings, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple of modes specifying the mode to use on each axis in
        ``axes``.
    axes : 2-tuple of ints, optional
        Axes over which to compute the IDWT. Repeated elements mean the IDWT
        will be performed multiple times along these axes.
    coeff_masks : tuple, optional
        ``(mA, (mH, mV, mD))`` trust masks matching ``coeffs``, see
        :func:`idwtn`. When given, the reconstruction and its boolean mask
        are returned as a 2-tuple.

    Examples
    --------
    >>> import numpy as np
    >>> import pywt
    >>> data = np.array([[1,2], [3,4]], dtype=np.float64)
    >>> coeffs = pywt.dwt2(data, 'haar')
    >>> pywt.idwt2(coeffs, 'haar')
    array([[ 1.,  2.],
           [ 3.,  4.]])

    """
    # L -low-pass data, H - high-pass data
    LL, (HL, LH, HH) = coeffs
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("Expected 2 axes")

    coeff_dict = {'aa': LL, 'da': HL, 'ad': LH, 'dd': HH}
    if coeff_masks is None:
        return idwtn(coeff_dict, wavelet, mode, axes)
    mL, (mH, mV, mD) = coeff_masks
    mask_dict = {'aa': mL, 'da': mH, 'ad': mV, 'dd': mD}
    return idwtn(coeff_dict, wavelet, mode, axes, coeff_masks=mask_dict)


def dwtn(data, wavelet, mode='symmetric', axes=None, mask=None):
    """
    Single-level n-dimensional Discrete Wavelet Transform.

    Parameters
    ----------
    data : array_like
        n-dimensional array with input data.
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or tuple of string, optional
        Signal extension mode used in the decomposition,
        see :ref:`Modes <ref-modes>`. This can also be a tuple of modes
        specifying the mode to use on each axis in ``axes``. Mask
        propagation supports ``'symmetric'`` and ``'periodization'``.
    axes : sequence of ints, optional
        Axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes. A value of ``None`` (the
        default) selects all axes.

        Axes may be repeated, but information about the original size may be
        lost if it is not divisible by ``2 ** nrepeats``. The reconstruction
        will be larger, with additional values derived according to the
        ``mode`` parameter. ``pywt.wavedecn`` should be used for multilevel
        decomposition.
    mask : array_like of bool, optional
        Observation mask with the same shape as ``data``. ``True`` marks a
        real observation, ``False`` a missing sample. When given, the values
        at missing positions are treated as zero for the numerical transform
        (regardless of whether the placeholder is a stale reading, NaN or
        infinity) and a second dictionary of boolean coefficient masks is
        returned alongside the coefficients. A coefficient is marked
        ``False`` whenever any non-zero filter tap of the actual, boundary
        extended decomposition reaches a missing sample.

    Returns
    -------
    coeffs : dict
        Results are arranged in a dictionary, where key specifies
        the transform type on each dimension and value is a n-dimensional
        coefficients array.

        For example, for a 2D case the result will look something like this::

            {'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
             'ad': <coeffs>  # V(LH) - approx. on 1st dim, det. on 2nd dim
             'da': <coeffs>  # H(HL) - det. on 1st dim, approx. on 2nd dim
             'dd': <coeffs>  # D(HH) - det. on 1st dim, det. on 2nd dim
            }

        For user-specified ``axes``, the order of the characters in the
        dictionary keys map to the specified ``axes``.
    coeff_masks : dict, optional
        Only returned when ``mask`` is not None. Dictionary of boolean arrays
        with the same keys and shapes as ``coeffs``; ``True`` marks a
        coefficient that depends only on observed samples.

    """
    data = np.asarray(data)
    if not _have_c99_complex and np.iscomplexobj(data):
        if mask is not None:
            raise ValueError(
                "mask propagation is only supported for real floating point "
                "arrays")
        real = dwtn(data.real, wavelet, mode, axes)
        imag = dwtn(data.imag, wavelet, mode, axes)
        return {k: real[k] + 1j * imag[k] for k in real}

    if data.dtype == np.dtype('object'):
        raise TypeError("Input must be a numeric array-like")
    if data.ndim < 1:
        raise ValueError("Input data must be at least 1D")

    if mask is not None:
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError(
                "mask propagation is only supported for real floating point "
                "arrays")
        mask = check_mask(mask, data.shape)
        check_finite(data, mask)
        # Missing cells are replaced by zero only at the raw-data entry
        # point.  Internal multilevel calls use ``_dwtn_propagate`` so that
        # derived partial sums are never re-zeroed.
        data = np.where(mask, data, 0.0)
        coeffs, masks = _dwtn_propagate(data, wavelet, mode, axes, mask)
        return coeffs, masks

    return _dwtn_propagate(data, wavelet, mode, axes, None)[0]


def _dwtn_propagate(data, wavelet, mode, axes, mask):
    """Core separable transform with optional structural mask propagation.

    The caller is responsible for replacing missing raw samples with zero.
    This helper deliberately never modifies numerical values according to the
    mask, which is essential to keep multilevel decomposition equivalent to a
    plain transform of the once-zeroed input.
    """
    if axes is None:
        axes = range(data.ndim)
    axes = [a + data.ndim if a < 0 else a for a in axes]

    modes = _modes_per_axis(mode, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)

    coeffs = [('', data)]
    masks = [('', mask)] if mask is not None else None
    for axis, wav, mode in zip(axes, wavelets, modes):
        new_coeffs = []
        new_masks = []
        for subband, x in coeffs:
            cA, cD = dwt_axis(x, wav, mode, axis)
            new_coeffs.extend([(subband + 'a', cA),
                               (subband + 'd', cD)])
        coeffs = new_coeffs
        if masks is not None:
            for subband, m in masks:
                mA = propagate_mask_dwt_axis(m, wav, mode, axis, 'a')
                mD = propagate_mask_dwt_axis(m, wav, mode, axis, 'd')
                new_masks.extend([(subband + 'a', mA),
                                  (subband + 'd', mD)])
            masks = new_masks
    if mask is None:
        return dict(coeffs), None
    return dict(coeffs), dict(masks)


def _fix_coeffs(coeffs):
    missing_keys = [k for k, v in coeffs.items() if v is None]
    if missing_keys:
        raise ValueError(
            "The following detail coefficients were set to None:\n"
            f"{missing_keys}\n"
            "For multilevel transforms, rather than setting\n"
            "\tcoeffs[key] = None\n"
            "use\n"
            "\tcoeffs[key] = np.zeros_like(coeffs[key])\n")

    invalid_keys = [k for k, v in coeffs.items() if
                    not set(k) <= set('ad')]
    if invalid_keys:
        raise ValueError(
            "The following invalid keys were found in the detail "
            f"coefficient dictionary: {invalid_keys}.")

    key_lengths = [len(k) for k in coeffs]
    if len(np.unique(key_lengths)) > 1:
        raise ValueError(
            "All detail coefficient names must have equal length.")

    return {k: np.asarray(v) for k, v in coeffs.items()}


def idwtn(coeffs, wavelet, mode='symmetric', axes=None, coeff_masks=None):
    """
    Single-level n-dimensional Inverse Discrete Wavelet Transform.

    Parameters
    ----------
    coeffs: dict
        Dictionary as in output of ``dwtn``. Missing or ``None`` items
        will be treated as zeros.
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or list of string, optional
        Signal extension mode used in the decomposition,
        see :ref:`Modes <ref-modes>`. This can also be a tuple of modes
        specifying the mode to use on each axis in ``axes``. Mask
        propagation supports ``'symmetric'`` and ``'periodization'``.
    axes : sequence of ints, optional
        Axes over which to compute the IDWT. Repeated elements mean the IDWT
        will be performed multiple times along these axes. A value of ``None``
        (the default) selects all axes.

        For the most accurate reconstruction, the axes should be provided in
        the same order as they were provided to ``dwtn``.
    coeff_masks : dict of bool arrays, optional
        Trust masks matching ``coeffs`` (same keys and shapes). ``False``
        marks an untrustworthy coefficient; such coefficients participate in
        the numerical reconstruction as zeros. Missing mask entries are
        treated as fully untrustworthy. When provided, a boolean mask of the
        reconstructed samples is returned together with the reconstruction.

    Returns
    -------
    data : ndarray
        Original signal reconstructed from input data.
    rec_mask : ndarray of bool, optional
        Only returned when ``coeff_masks`` is not None. ``True`` marks a
        reconstructed sample reachable only from trustworthy coefficients
        through non-zero synthesis taps.

    """

    # drop the keys corresponding to value = None
    coeffs = {k: v for k, v in coeffs.items() if v is not None}

    # drop the keys corresponding to value = None
    coeffs = {k: v for k, v in coeffs.items() if v is not None}

    # Raise error for invalid key combinations
    coeffs = _fix_coeffs(coeffs)

    if coeff_masks is not None:
        coeff_masks = {k: v for k, v in coeff_masks.items()
                       if v is not None}
        for k, v in coeff_masks.items():
            v = np.asarray(v)
            if v.dtype != bool:
                raise TypeError(
                    f"coeff_masks[{k!r}] must be a boolean array")
            if k not in coeffs:
                raise ValueError(
                    f"coeff_masks contains key {k!r} absent from coeffs")
            if v.shape != coeffs[k].shape:
                raise ValueError(
                    f"coeff_masks[{k!r}] shape {v.shape} does not match "
                    f"coefficient shape {coeffs[k].shape}")
        if not np.issubdtype(
                next(iter(coeffs.values())).dtype, np.floating):
            raise ValueError(
                "mask propagation is only supported for real floating point "
                "arrays")
        # untrustworthy coefficients participate as zeros numerically
        coeffs = {k: np.where(coeff_masks.get(k, np.zeros_like(v, dtype=bool)),
                              v, 0.0)
                  for k, v in coeffs.items()}

    if (not _have_c99_complex and
            any(np.iscomplexobj(v) for v in coeffs.values())):
        if coeff_masks is not None:
            raise ValueError(
                "mask propagation is only supported for real floating point "
                "arrays")
        real_coeffs = {k: v.real for k, v in coeffs.items()}
        imag_coeffs = {k: v.imag for k, v in coeffs.items()}
        return (idwtn(real_coeffs, wavelet, mode, axes) +
                1j * idwtn(imag_coeffs, wavelet, mode, axes))

    # key length matches the number of axes transformed
    ndim_transform = max(len(key) for key in coeffs)

    try:
        coeff_shapes = (v.shape for k, v in coeffs.items()
                        if v is not None and len(k) == ndim_transform)
        coeff_shape = next(coeff_shapes)
    except StopIteration:
        raise ValueError("`coeffs` must contain at least one non-null wavelet "
                         "band")
    if any(s != coeff_shape for s in coeff_shapes):
        raise ValueError("`coeffs` must all be of equal size (or None)")

    if axes is None:
        axes = range(ndim_transform)
        ndim = ndim_transform
    else:
        ndim = len(coeff_shape)
    axes = [a + ndim if a < 0 else a for a in axes]

    modes = _modes_per_axis(mode, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)

    masks = (dict(coeff_masks) if coeff_masks is not None else None)
    for key_length, (axis, wav, mode) in reversed(
            list(enumerate(zip(axes, wavelets, modes)))):
        if axis < 0 or axis >= ndim:
            raise AxisError("Axis greater than data dimensions")

        new_coeffs = {}
        new_masks = {}
        new_keys = [''.join(coef) for coef in product('ad', repeat=key_length)]

        for key in new_keys:
            L = coeffs.get(key + 'a', None)
            H = coeffs.get(key + 'd', None)
            if L is not None and H is not None:
                if L.dtype != H.dtype:
                    # upcast to a common dtype (float64 or complex128)
                    if L.dtype.kind == 'c' or H.dtype.kind == 'c':
                        dtype = np.complex128
                    else:
                        dtype = np.float64
                    L = np.asarray(L, dtype=dtype)
                    H = np.asarray(H, dtype=dtype)
            rec = idwt_axis(L, H, wav, mode, axis)
            new_coeffs[key] = rec
            if masks is not None:
                mL = masks.get(key + 'a', None)
                mH = masks.get(key + 'd', None)
                # a band absent from the mask dict is fully untrusted, but
                # an absent numerical band contributes nothing (zero)
                if L is None:
                    mL = None
                elif mL is None:
                    mL = np.zeros(L.shape, dtype=bool)
                if H is None:
                    mH = None
                elif mH is None:
                    mH = np.zeros(H.shape, dtype=bool)
                new_masks[key] = propagate_mask_idwt_axis(
                    mL, mH, wav, mode, axis, rec.shape)
        coeffs = new_coeffs
        if masks is not None:
            masks = new_masks

    if coeff_masks is None:
        return coeffs['']
    return coeffs[''], masks['']
