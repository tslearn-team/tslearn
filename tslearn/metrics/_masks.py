""" Masks """
from numba import njit, prange

import numpy

from tslearn.backend import instantiate_backend
from tslearn.backend.pytorch_backend import HAS_TORCH


GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}


def __make_sakoe_chiba_mask(backend):
    if backend is numpy:
        range_ = prange
    else:
        range_ = range

    def _sakoe_chiba_mask_generic(sz1, sz2, radius=1):
        mask = backend.full((sz1, sz2), False)
        if sz1 > sz2:
            width = sz1 - sz2 + radius
            for i in range_(sz2):
                lower = max(0, i - radius)
                upper = min(sz1, i + width) + 1
                mask[lower:upper, i] = True
        else:
            width = sz2 - sz1 + radius
            for i in range_(sz1):
                lower = max(0, i - radius)
                upper = min(sz2, i + width) + 1
                mask[i, lower:upper] = True
        return mask
    if backend is numpy:
        return njit(nogil=True, parallel=True)(_sakoe_chiba_mask_generic)
    else:
        return _sakoe_chiba_mask_generic

_njit_sakoe_chiba_mask = __make_sakoe_chiba_mask(numpy)
if HAS_TORCH:
    _sakoe_chiba_mask = __make_sakoe_chiba_mask(instantiate_backend("torch"))
else:
    _sakoe_chiba_mask = _njit_sakoe_chiba_mask


def sakoe_chiba_mask(sz1, sz2, radius=1, be=None):
    """Compute the Sakoe-Chiba mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series
    sz2 : int
        The size of the second time series.
    radius : int
        The radius of the band.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    mask : array-like, shape=(sz1, sz2)
        Sakoe-Chiba mask.

    Examples
    --------
    >>> sakoe_chiba_mask(4, 4, radius=2)
    array([[ True,  True,  True, False],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [False,  True,  True,  True]])
    """
    be = instantiate_backend(be)
    if be.is_numpy:
        return _njit_sakoe_chiba_mask(
            sz1,
            sz2,
            radius,
        )
    else:
        return _sakoe_chiba_mask(
            sz1,
            sz2,
            radius,
        )


def __make_itakura_mask(backend):
    if backend is numpy:
        range_ = prange
    else:
        range_ = range

    def _itakura_mask_generic(sz1, sz2, max_slope=2.0):
        min_slope = 1 / float(max_slope)
        max_slope *= float(sz1) / float(sz2)
        min_slope *= float(sz1) / float(sz2)

        lower_bound = backend.empty((2, sz2))
        lower_bound[0] = min_slope * backend.arange(sz2)
        lower_bound[1] = (sz1 - 1) - max_slope * (sz2 - 1) + max_slope * backend.arange(sz2)
        lower_bound_ = backend.empty(sz2)
        for i in range_(sz2):
            lower_bound_[i] = max(
                backend.round(lower_bound[0, i], decimals=2),
                backend.round(lower_bound[1, i], decimals=2)
            )
        lower_bound_ = backend.ceil(lower_bound_)

        upper_bound = backend.empty((2, sz2))
        upper_bound[0] = max_slope * backend.arange(sz2)
        upper_bound[1] = (sz1 - 1) - min_slope * (sz2 - 1) + min_slope * backend.arange(sz2)
        upper_bound_ = backend.empty(sz2)
        for i in range_(sz2):
            upper_bound_[i] = min(
                backend.round(upper_bound[0, i], decimals=2),
                backend.round(upper_bound[1, i], decimals=2)
            )
        upper_bound_ = backend.floor(upper_bound_ + 1)

        mask = backend.full((sz1, sz2), False)
        for i in range_(sz2):
            mask[int(lower_bound_[i]): int(upper_bound_[i]), i] = True

        # Post-check
        raise_warning = False
        for i in range(sz1):
            if not backend.any(mask[i]):
                raise_warning = True
                break
        if not raise_warning:
            for j in range(sz2):
                if not backend.any(mask[:, j]):
                    raise_warning = True
                    break
        if raise_warning:
            raise RuntimeWarning(
                "'itakura_max_slope' constraint is unfeasible "
                "(ie. leads to no admissible path) for the "
                "provided time series sizes",
            )

        return mask

    if backend is numpy:
        return njit(nogil=True, parallel=True)(_itakura_mask_generic)
    else:
        return _itakura_mask_generic

_njit_itakura_mask = __make_itakura_mask(numpy)
if HAS_TORCH:
    _itakura_mask = __make_itakura_mask(instantiate_backend("torch"))
else:
    _itakura_mask = _njit_itakura_mask


def itakura_mask(sz1, sz2, max_slope=2.0, be=None):
    """Compute the Itakura mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series
    sz2 : int
        The size of the second time series.
    max_slope : float (default = 2)
        The maximum slope of the parallelogram.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    mask : array-like, shape=(sz1, sz2)
        Itakura mask.

    Examples
    --------
    >>> itakura_mask(6, 6, max_slope=3)
    array([[ True, False, False, False, False, False],
           [False,  True,  True,  True, False, False],
           [False,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True, False],
           [False, False,  True,  True,  True, False],
           [False, False, False, False, False,  True]])

    """
    be = instantiate_backend(be)

    if be.is_numpy:
        mask = _njit_itakura_mask(sz1, sz2, max_slope=max_slope)
    else:
        mask = _itakura_mask(sz1, sz2, max_slope=max_slope)

    return mask


def compute_mask(
    s1,
    s2,
    global_constraint=0,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute the mask (region constraint).

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,) or int
        A time series or integer.
        If shape is (sz1,), the time series is assumed to be univariate.
        If int, size sz1 used to dimension the mask.
    s2 : array-like, shape=(sz2, d) or (sz2,) or int
        Another time series or integer.
        If shape is (sz2,), the time series is assumed to be univariate.
        If int, size sz2 used to dimension the mask.
    global_constraint : {0, 1, 2} (default: 0)
        Global constraint to restrict admissible paths for DTW:
        - "itakura" if 1
        - "sakoe_chiba" if 2
        - no constraint otherwise
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to 2 (sakoe-chiba), a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to 1 (itakura), a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    mask : array-like, shape=(sz1, sz2)
        Constraint region.

    Examples
    --------
    >>> compute_mask(4, 4)
    array([[ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True],
           [ True,  True,  True,  True]])
    >>> compute_mask(4, 4, sakoe_chiba_radius=1)
    array([[ True,  True, False, False],
           [ True,  True,  True, False],
           [False,  True,  True,  True],
           [False, False,  True,  True]])
    >>> compute_mask(4, 4, itakura_max_slope=2)
    array([[ True, False, False, False],
           [False,  True,  True, False],
           [False,  True,  True, False],
           [False, False, False,  True]])
    """
    be = instantiate_backend(be, s1, s2)

    # The output mask will be of shape (sz1, sz2)
    if isinstance(s1, int) and isinstance(s2, int):
        sz1, sz2 = s1, s2
    else:
        sz1 = be.array(s1).shape[0]
        sz2 = be.array(s2).shape[0]

    if be.is_numpy:
        x = _njit_compute_mask(
            sz1,
            sz2,
            global_constraint,
            sakoe_chiba_radius,
            itakura_max_slope,
        )
        return x
    else:
        return _compute_mask(
            sz1,
            sz2,
            global_constraint,
            sakoe_chiba_radius,
            itakura_max_slope,
        )


def __make_compute_mask(backend):
    if backend is numpy:
        sakoe_chiba_mask_ = _njit_sakoe_chiba_mask
        itakura_mask_ = _njit_itakura_mask
    else:
        sakoe_chiba_mask_ = _sakoe_chiba_mask
        itakura_mask_ = _itakura_mask

    def _compute_mask_generic(
            sz1,
            sz2,
            global_constraint=0,
            sakoe_chiba_radius=None,
            itakura_max_slope=None,
    ):
        if (
                global_constraint == 0
                and sakoe_chiba_radius is not None
                and itakura_max_slope is not None
        ):
            raise RuntimeWarning(
                "global_constraint is not set for DTW, but both "
                "sakoe_chiba_radius and itakura_max_slope are "
                "set, hence global_constraint cannot be inferred "
                "and no global constraint will be used."
            )
        if global_constraint == 2 or (
                global_constraint == 0 and sakoe_chiba_radius is not None
        ):
            if sakoe_chiba_radius is None:
                sakoe_chiba_radius = 1
            mask = sakoe_chiba_mask_(sz1, sz2, radius=sakoe_chiba_radius)

        elif global_constraint == 1 or (
                global_constraint == 0 and itakura_max_slope is not None
        ):
            if itakura_max_slope is None:
                itakura_max_slope = 2.0
            mask = itakura_mask_(sz1, sz2, max_slope=itakura_max_slope)
        else:
            mask = backend.full((sz1, sz2), True)
        return mask
    if backend is numpy:
        return njit(nogil=True)(_compute_mask_generic)
    else:
        return _compute_mask_generic

_njit_compute_mask = __make_compute_mask(numpy)
if HAS_TORCH:
    _compute_mask = __make_compute_mask(instantiate_backend("torch"))
else:
    _compute_mask = _njit_compute_mask
