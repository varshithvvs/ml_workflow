# -*- coding: utf-8 -*-
"""
Encode categorical variables.

@author: Varshith
"""
from dask_ml.preprocessing import LabelEncoder


def encode(x_train_txt_label_encode=None, x_train_txt_hot_encode=None):
    """Label encode categorical variables.

    Parameters
    ----------
    x_train_txt_label_encode : TYPE
        DESCRIPTION.

    Returns
    -------
    Encoded dataset of input.
    """
    labelencoder_X = LabelEncoder(use_categorical=True)
    # for i in range(np.shape(x_train_txt_label_encode)[1]):
    x_train_txt_label_encode['product_category'] = labelencoder_X.fit_transform(x_train_txt_label_encode['product_category'])
    x_train_txt_label_encode['product_subcategory'] = labelencoder_X.fit_transform(x_train_txt_label_encode['product_subcategory'])

    return x_train_txt_label_encode
