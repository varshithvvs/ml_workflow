# -*- coding: utf-8 -*-
"""
Impute Numerical and Categorical Data.

@author: Varshith
"""
from autoimpute.imputations import MultipleImputer
import dask.dataframe as dd


def imputer(features, num_col):
    """Imputer function to impute categorical and numerical features.

    Parameters
    ----------
    features : features data
        features for the data model.
    num_col :  list
        columns for imputation

    Returns
    -------
    features (Imputed Features).

    """
    x_train_num = features[num_col].compute()

    imputer_num = MultipleImputer(strategy='stochastic', return_list=True, n=5,
                                  seed=101)
    x_train_num_avg = imputer_num.fit_transform(x_train_num)

    x_train_num_concat = x_train_num_avg[0][1]

    for i in range(len(x_train_num_avg)-1):
        x_train_num_concat = dd.concat([x_train_num_concat,
                                        x_train_num_avg[i+1][1]], axis=1)
    x_train_num_avg = x_train_num_concat.compute().groupby(
        by=x_train_num_concat.columns, axis=1).apply(lambda g: g.mean(axis=1))

    features_final = x_train_num_avg
    features = features_final.head(len(features))
    return features_final
