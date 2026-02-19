"""
The :mod:`tslearn.metrics` module delivers time-series specific metrics to be 
used at the core of machine learning algorithms.

**User guide:** See the :ref:`Dynamic Time Warping (DTW) <dtw>` section for 
further details.
"""
from ._masks import (
    GLOBAL_CONSTRAINT_CODE,
    compute_mask,
    itakura_mask,
    sakoe_chiba_mask
)
from .ctw import (
    ctw,
    ctw_path,
    cdist_ctw,
    _cdist_ctw
)
from ._dtw import(
    dtw,
    dtw_path,
    cdist_dtw,
    _cdist_dtw,
    accumulated_matrix as dtw_accumulated_matrix
)
from ._gak import (
    sigma_gak,
    _sigma_gak,
    gak,
    unnormalized_gak,
    cdist_gak,
    _cdist_gak,
)
from .dtw_variants import (
   dtw_limited_warping_length,
   dtw_path_limited_warping_length,
   subsequence_path,
   subsequence_cost_matrix,
   dtw_path_from_metric,
   dtw_subsequence_path,
   lb_envelope,
   lb_keogh,
   lcss,
   lcss_path,
   lcss_path_from_metric
)
from .sax import cdist_sax, _cdist_sax
from .softdtw_variants import (
    cdist_soft_dtw,
    _cdist_soft_dtw,
    cdist_soft_dtw_normalized,
    _cdist_soft_dtw_normalized,
    soft_dtw,
    soft_dtw_alignment,
    gamma_soft_dtw,
    SquaredEuclidean,
    SoftDTW
)
from .soft_dtw_loss_pytorch import SoftDTWLossPyTorch
from .cycc import cdist_normalized_cc, y_shifted_sbd_vec
from ._frechet import (
    frechet,
    frechet_path,
    frechet_path_from_metric,
    cdist_frechet,
    _cdist_frechet,
    accumulated_matrix as frechet_accumulated_matrix
)

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

TSLEARN_VALID_METRICS = ["ctw", "dtw", "gak", "sax", "softdtw",
                         "softdtw_normalized", "lcss", "frechet"]
VARIABLE_LENGTH_METRICS = ["ctw", "dtw", "gak", "sax", "softdtw",
                           "softdtw_normalized", "lcss", "frechet"]


METRIC_TO_FUNCTION = {
    "ctw": _cdist_ctw,
    "dtw": _cdist_dtw,
    "gak": _cdist_gak,
    "sax": _cdist_sax,
    "softdtw": _cdist_soft_dtw,
    "softdtw_normalized": _cdist_soft_dtw_normalized,
    "frechet": _cdist_frechet
}


__all__ = [
    "TSLEARN_VALID_METRICS",
    "VARIABLE_LENGTH_METRICS",
    "GLOBAL_CONSTRAINT_CODE",
    "compute_mask",
    "itakura_mask",
    "sakoe_chiba_mask",
    "dtw",
    "dtw_path",
    "dtw_accumulated_matrix",
    "dtw_path_from_metric",
    "cdist_dtw",
    "dtw_limited_warping_length",
    "dtw_path_limited_warping_length",
    "subsequence_path",
    "subsequence_cost_matrix",
    "dtw_limited_warping_length",
    "dtw_path_limited_warping_length",
    "dtw_subsequence_path",
    "lb_envelope",
    "lb_keogh",
    "lcss",
    "lcss_path",
    "lcss_path_from_metric",
    "ctw_path",
    "ctw",
    "cdist_ctw",
    "cdist_sax",
    "cdist_soft_dtw",
    "cdist_soft_dtw_normalized",
    "sigma_gak",
    "gak",
    "unnormalized_gak",
    "cdist_gak",
    "soft_dtw",
    "soft_dtw_alignment",
    "gamma_soft_dtw",
    "SquaredEuclidean",
    "SoftDTW",
    "SoftDTWLossPyTorch",
    "cdist_normalized_cc",
    "y_shifted_sbd_vec",
    "frechet",
    "frechet_path",
    "frechet_accumulated_matrix",
    "frechet_path_from_metric",
    "cdist_frechet"
]
