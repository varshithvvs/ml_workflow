# -*- coding: utf-8 -*-
"""
Impute Numerical and Categorical Data.

@author: Varshith
"""
from autoimpute.imputations import MultipleImputer
import dask.dataframe as dd
import numpy as np
from scipy import stats
import pandas as pd


def imputer(features, start, split, end):
    """Imputer function to impute categorical and numerical features.

    Parameters
    ----------
    features : features data
        features for the data model.
    start : int
        start for splitting columns between num and cat.
    split : int
        split column number for splitting features.
    end :  int
        end column for splitting columns between num and cat.

    Returns
    -------
    features (Impute Features).

    """
    x_train_num = features.iloc[:, list(range(start, split))].compute()
    x_train_txt = features.iloc[:, list(range(split, end))].compute()

    imputer_num = MultipleImputer(strategy='stochastic', return_list=True, n=5, seed=101)
    x_train_num_avg = imputer_num.fit_transform(x_train_num)

    x_train_num_concat = x_train_num_avg[0][1]

    for i in range(len(x_train_num_avg)-1):
        x_train_num_concat = dd.concat([x_train_num_concat, x_train_num_avg[i+1][1]], axis=1)
    x_train_num_avg = x_train_num_concat.compute().groupby(by=x_train_num_concat.columns, axis=1).apply(lambda g: g.mean(axis=1))
    x_train_num_avg_col = np.unique(x_train_num_concat.compute().columns.tolist())

    # Categorical Imputer
    imputer_txt = MultipleImputer(strategy='categorical', return_list=True, n=10, seed=101)
    x_train_txt_avg = x_train_txt
    imputer_txt = imputer_txt.fit(x_train_txt)
    #x_train_txt.reset_index(drop=True, inplace=True)
    x_train_txt_avg = imputer_txt.transform(x_train_txt)

    x_train_txt_col = list(x_train_txt.columns)
    x_train_txt_col.sort()
    x_train_txt_concat = x_train_txt_avg[0][1]

    for i in range(len(x_train_txt_avg)-1):
        x_train_txt_concat = dd.concat([x_train_txt_concat, x_train_txt_avg[i+1][1]], axis=1)
    x_train_txt_avg = x_train_txt_concat.compute().groupby(by=x_train_txt_concat.columns, axis=1).apply(lambda g: stats.mode(g, axis=1)[0])
    x_train_txt_avg = x_train_txt_avg.sort_index(axis=0)
    x_train_txt_avg_temp = pd.DataFrame(x_train_txt_avg[0])
    for i in range(len(x_train_txt_avg)-1):
        x_train_txt_avg_temp = dd.concat([x_train_txt_avg_temp, pd.DataFrame(x_train_txt_avg[i+1])], axis=1)
    x_train_txt_avg_temp.columns = x_train_txt_col
    x_train_txt_avg = x_train_txt_avg_temp
    x_train_txt = x_train_txt.sort_index(axis=1)

    features_final = dd.concat([x_train_num_avg, x_train_txt_avg], axis=1)
    features = features_final.head(len(features))
    return features_final
