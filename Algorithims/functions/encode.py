# -*- coding: utf-8 -*-
"""
Encode categorical variables.

@author: Varshith
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import dask.array as da
import dask.dataframe as dd
import pandas as pd

def encode(x_train_txt_label_encode=None, x_train_txt_hot_encode=None):
    """Label encode categorical variables.

    Parameters
    ----------
    x_train_txt_label_encode : TYPE
        DESCRIPTION.
    x_train_txt_hot_encode : TYPE
        DESCRIPTION.

    Returns
    -------
    Encoded dataset of input.

    """

    # Label Encode Categorical Variables
    if x_train_txt_label_encode is not None:
        x_train_txt_label_encode_col = x_train_txt_label_encode.columns.tolist()
        x_train_txt_label_encode = x_train_txt_label_encode.values.compute()
        labelencoder_X = LabelEncoder()
        for i in range(np.shape(x_train_txt_label_encode)[1]):
            x_train_txt_label_encode[:, i] = labelencoder_X.fit_transform(x_train_txt_label_encode[:, i])
        x_train_txt_label_encode = pd.DataFrame(x_train_txt_label_encode, columns=x_train_txt_label_encode_col)

    # Hot Encode Categorical Variables
    if x_train_txt_hot_encode is not None:        
        x_train_txt_hot_encode_col = x_train_txt_hot_encode.columns.tolist()
        x_train_txt_hot_encode = x_train_txt_hot_encode.values.compute()
        onehotencoder = OneHotEncoder()
        x_train_txt_hot_encode = onehotencoder.fit_transform(x_train_txt_hot_encode)
        x_train_txt_hot_encode_col = onehotencoder.categories_
        x_train_txt_hot_encode_col = np.concatenate(onehotencoder.categories_, axis=0).tolist()
        x_train_txt_hot_encode = pd.DataFrame.sparse.from_spmatrix(x_train_txt_hot_encode, columns=x_train_txt_hot_encode_col)
    
    # Concat Encoded Features
    categorical_Encoded = pd.concat([x_train_txt_label_encode, x_train_txt_hot_encode], axis=1)
    
    return categorical_Encoded
