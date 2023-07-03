"""
The :mod:`tslearn.metrics` module delivers time-series specific metrics to be 
used at the core of machine learning algorithms.

**User guide:** See the :ref:`Dynamic Time Warping (DTW) <dtw>` section for 
further details.
"""
from .dtw_variants import (dtw, dtw_limited_warping_length,
                           dtw_path_limited_warping_length, subsequence_path,
                           subsequence_cost_matrix,
                           dtw_path, dtw_path_from_metric,
                           dtw_subsequence_path, cdist_dtw,
                           GLOBAL_CONSTRAINT_CODE,
                           lb_envelope, lb_keogh,
                           sakoe_chiba_mask, itakura_mask,
                           lcss, lcss_path, lcss_path_from_metric)
from .ctw import ctw_path, ctw, cdist_ctw
from .sax import cdist_sax
from .softdtw_variants import (cdist_soft_dtw, cdist_gak,
                               cdist_soft_dtw_normalized, gak, soft_dtw,
                               soft_dtw_alignment,
                               sigma_gak, gamma_soft_dtw, SquaredEuclidean,
                               SoftDTW)
from .soft_dtw_loss_pytorch import SoftDTWLossPyTorch
from .cycc import cdist_normalized_cc, y_shifted_sbd_vec

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

TSLEARN_VALID_METRICS = ["ctw", "dtw", "gak", "sax", "softdtw", "lcss"]
VARIABLE_LENGTH_METRICS = ["ctw", "dtw", "gak", "sax", "softdtw", "lcss"]

__all__ = [
    "TSLEARN_VALID_METRICS", "VARIABLE_LENGTH_METRICS",

    "dtw", "dtw_limited_warping_length",
    "dtw_path_limited_warping_length", "subsequence_path",
    "subsequence_cost_matrix",
    "dtw_path", "dtw_path_from_metric",
    "dtw_subsequence_path", "cdist_dtw",
    "GLOBAL_CONSTRAINT_CODE",
    "lb_envelope", "lb_keogh",
    "sakoe_chiba_mask", "itakura_mask",
    "lcss", "lcss_path", "lcss_path_from_metric",

    "ctw_path", "ctw", "cdist_ctw",

    "cdist_sax",

    "cdist_soft_dtw", "cdist_gak",
    "cdist_soft_dtw_normalized", "gak", "soft_dtw", "soft_dtw_alignment",
    "sigma_gak", "gamma_soft_dtw", "SquaredEuclidean", "SoftDTW",

    "SoftDTWLossPyTorch",

    "cdist_normalized_cc", "y_shifted_sbd_vec"
]
