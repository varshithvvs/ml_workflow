# -*- coding: utf-8 -*-
"""
Drop missing features.

@author: Varshith
"""
from functions.imputer import imputer
import dask.dataframe as dd
import pandas as pd
import numpy as np


def drop_missing(features, missing_drop_threshold, start, split, end):
    """Drop missing features by threshold and impute the remaining.

    Parameters
    ----------
    features : TYPE
        DESCRIPTION.
    missing_drop_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #  Percent missing data
    total_missing_data = features.isnull().sum().compute().sort_values(ascending=False)[features.isnull().sum().compute().sort_values(ascending=False) != 0]
    percent_missing_data = (features.isnull().sum().compute().sort_values(ascending=False)/len(features.compute())*100)[features.isnull().sum().compute().sort_values(ascending=False) != 0]
    missing_Data_drop_col = list(percent_missing_data[percent_missing_data > (missing_drop_threshold*100)].index.values)

    # Drop features greater than threshold
    features = features.drop(missing_Data_drop_col, axis=1)
    features = imputer(features, start=start, split=split, end=end)
    return features
