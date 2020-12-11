import numpy as np
from tslearn.preprocessing import (TimeSeriesScalerMeanVariance,
                                   TimeSeriesScalerMinMax)
from tslearn.utils import to_time_series_dataset


def test_single_value_ts_no_nan():
    X = to_time_series_dataset([[1, 1, 1, 1]])

    standard_scaler = TimeSeriesScalerMeanVariance()
    assert np.sum(np.isnan(standard_scaler.fit_transform(X))) == 0

    minmax_scaler = TimeSeriesScalerMinMax()
    assert np.sum(np.isnan(minmax_scaler.fit_transform(X))) == 0
