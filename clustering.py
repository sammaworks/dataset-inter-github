from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np

@staticmethod
def stl_deseasonal_single_series_no_leak(series, period, split_point, sts_return_model=False):
    """
    Applies STL decomposition using ONLY data up to split_point (train)
    to estimate seasonality, then repeats that seasonal pattern over
    the full index (train + test) to avoid data leakage.

    Args
    ----
    series : pd.Series
        Full time series (train + test).
    period : int
        Seasonal period (e.g. 12 for monthly with yearly seasonality).
    split_point : str or Timestamp
        Date that separates train from test.
    sts_return_model : bool
        If True, also return the fitted STL object (for diagnostics).

    Returns
    -------
    deseasonalized : pd.Series
        Deseasonalized version of `series` (full index), with seasonality
        estimated only from train data.
    """
    series = series.sort_index()
    split_point = pd.to_datetime(split_point)

    # 1) Restrict STL fit to TRAIN only (no future info)
    series_train = series.loc[:split_point]

    stl = STL(series_train, period=period, robust=True)
    res = stl.fit()
    seasonal_train = res.seasonal

    # 2) Build a seasonal pattern (one full period) from the *end* of train
    if len(seasonal_train) >= period:
        base_pattern = seasonal_train.iloc[-period:].values  # last 12 months
    else:
        # If train is shorter than a full cycle, just use what we have
        base_pattern = seasonal_train.values

    # 3) Repeat this pattern across the full index (train + test)
    full_idx = series.index
    full_seasonal_values = np.tile(base_pattern, int(np.ceil(len(full_idx) / len(base_pattern))))[:len(full_idx)]
    seasonal_full = pd.Series(full_seasonal_values, index=full_idx)

    # 4) Deseasonalize using this *train-derived* seasonal term
    deseasonalized = series - seasonal_full

    if sts_return_model:
        return deseasonalized, stl
    return deseasonalized
