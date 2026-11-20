.. _ref-sparse-archive:

==========================================
Sparse Coefficient Archives (budget-based)
==========================================

.. currentmodule:: pywt

A sparse coefficient archive stores only the wavelet coefficients that survive
a deletion step performed under a user-supplied absolute :math:`\ell_2` error
budget.  It is built from finite, real ``float64`` one- or two-dimensional
signals using the ``'haar'``, ``'db2'`` or ``'bior2.2'`` wavelets and the
``'symmetric'`` or ``'periodization'`` boundary modes.  The transform may use
one axis or both axes of a 2-D array; negative axes and axis order follow the
usual library conventions and odd edge lengths are restored to the exact input
shape on reconstruction.

.. autofunction:: sparse_coeff_archive
.. autoclass:: SparseCoeffArchive
   :members:
.. autofunction:: load_sparse_archive

Representation
-------------

An archive contains three pieces of information:

* the original numerical **values** of the retained coefficients;
* their flat **indices** in the ``ravel_coeffs`` layout of the full (dense)
  coefficient set;
* JSON-serialisable **metadata** holding every subband's logical slice and
  shape, the dtype, the wavelet/mode/axes/level settings, the original signal
  shape and the error accounting.

Deleted positions are never stored: they participate in reconstruction as
exact zeros.  The archive contains neither the original waveform, nor the
deleted coefficients, nor a dense coefficient copy.  A subband whose
coefficients are all deleted keeps its logical slice and shape, so axis
meaning and edge alignment are preserved through every round-trip.

The persistable payload is purely numeric arrays (``np.savez`` with
``allow_pickle=False``) plus a JSON metadata sidecar; it reloads in a fresh
process without pickle, closures or process-local registries.
``SparseCoeffArchive.nbytes`` reports the total payload size including the
indices and the metadata bytes.

Error basis
-----------

Let ``x`` be the input and ``y`` the waveform reconstructed from the archive
and cropped back to the original shape.  Every archive reports two numbers:

* ``error_bound`` (denoted ``B``) -- an a-priori, finite upper bound on the
  distortion caused by coefficient deletion alone, computed in exact
  arithmetic;
* ``rounding_tolerance`` (denoted ``tau``) -- an allowance for floating point
  round-off of the analysis/synthesis arithmetic, scaling with the data.

They satisfy::

    ||x - y||_2 <= B + tau,      B <= budget.

The deletion bound is structural, not energetic.  Writing ``c`` for the vector
of deleted coefficients, the multilevel synthesis is a chain of separable
single-level inverse transforms.  For each 1-D inverse step the submultiplicative
bound

    ||T||_2 <= sqrt(||T||_1 * ||T||_inf)

is evaluated directly from the synthesis filter taps in linear time without
ever forming the (quadratic-size) operator matrix: each output row contains at
most ``filter_len / 2`` products, so the row and column absolute sums are
accumulated by replaying the exact index arithmetic of the C convolution
implementation for both boundary modes.  The product ``R`` of all per-step
bounds gives

    B = R * ||c||_2.

For orthonormal wavelets with periodization the per-step bounds equal one, so
the bound reduces to the energy of the discarded coefficients.  For
biorthogonal wavelets such as ``'bior2.2'`` the synthesis is not an isometry,
and the discarded coefficient energy is therefore *not* the waveform error;
the operator-norm chain supplies the required correction.  This is why
discarded energy must never be reported as the error bound for a biorthogonal
wavelet.

Coefficient selection and budget allocation
-------------------------------------------

Coefficients are deleted in increasing order of magnitude (smallest first).
Because the structural bound depends on the deleted set only through its
Euclidean norm, magnitude order is the storage-optimal greedy choice under
this bound: no other selection of the same number of coefficients has a
smaller deleted norm.  All levels share a single global norm budget rather
than per-level allocations, which is what makes that greedy argument valid.
Deletion stops as soon as adding the next coefficient would make
``B`` exceed the budget.  A zero budget permits deletion of exactly-zero
coefficients only; ``B`` is then exactly zero and recovery differs from the
input only by rounding.

Rounding tolerance
------------------

All arithmetic is IEEE-754 ``float64`` with machine epsilon
``u = 2**-52``.  Measured across every supported wavelet, mode, level and data
scale from approximately ``1e-100`` to ``1e150``, the full analysis/synthesis
round-trip stays below a few epsilon relative to the input norm.  The reported
allowance is

    tau = 64 * u * (1 + ||x||_2).

The factor 64 leaves more than an order of magnitude of headroom over the
worst observed behaviour, the allowance scales linearly with the data
magnitude (and remains a finite positive floor for tiny signals), and it can
never be spent on deletion: zero-budget archives always have ``B == 0``.

Resource use
------------

For fixed wavelet, dimensionality and level the peak working space is linear
in the number of samples and coefficients.  No transform matrix is built and
the data is never rebuilt per deletion; the only super-linear step is one
``argsort`` over the coefficients.  The optional end-to-end verification
reconstructs once and accumulates the residual norm in finite batches of
``verification_batch`` samples, so batch scratch space is
``O(verification_batch)`` and the number of batches is
``ceil(x.size / verification_batch)``.  Reconstruction uses working space of
the same order as its output and the on-disk payload remains sparse.

Thresholding and round-trips
----------------------------

``threshold`` accepts an archive in ``mode='hard'``: retained values below the
threshold join the deleted set (reconstructing as zero), surviving values keep
their exact numbers, the added distortion is accumulated onto the bound
relative to the original signal without access to ``x`` or to previously
discarded values, and a new archive is returned without modifying the input.
If the accumulated bound exceeds the original budget the archive reports
``budget_met == False`` (and ``budget_exceeded == True``); the old compliance
claim can then no longer be used.

``ravel_coeffs`` and ``unravel_coeffs`` round-trip archive data while
preserving the per-band logical positions, dtype, transform settings,
original shape and error basis.  Level-1 archives also expose a single-level
dictionary for ``idwtn``; multilevel archives reconstruct through
``waverecn``.  Dense inputs, the ordinary packed return values and the
missing-quality (mask) calls keep their previous behaviour; archives
themselves do not accept quality masks.

Example: the compressible step signal
-------------------------------------

>>> import numpy as np
>>> import pywt
>>> x = np.concatenate([np.ones(2048), -np.ones(2048)])
>>> arc = pywt.sparse_coeff_archive(
...     x, wavelet='haar', level=4, mode='periodization', budget=0.01)
>>> arc.n_retained <= pywt.wavedecn_size(
...     pywt.wavedecn_shapes((4096,), 'haar', mode='periodization',
...                          level=4)) // 4
True
>>> y = arc.reconstruct()
>>> np.linalg.norm(x - y) <= arc.rounding_tolerance
True
>>> arc.error_bound <= arc.budget
True

Persistence and reloading in another process::

    arc.save('archive')                 # writes archive.npz + archive.json
    reloaded = pywt.load_sparse_archive('archive.npz')
    reloaded.reconstruct().shape        # (4096,)

See :doc:`dwt-coefficient-handling` for the dense coefficient packing helpers
and :doc:`thresholding-functions` for the ordinary thresholding functions.
