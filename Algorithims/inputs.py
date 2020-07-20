# -*- coding: utf-8 -*-.
"""
Loads required inputs and data files for the model.

1. Load the required inputs for data modelling
2. Perform the necessary ETL, EDA and feature engineering to feed data
   into the statistical model

@author: Varshith
"""


# Import required libraries

import dask.array as da
import sweetviz as swz
import numpy as np
from dask_ml.preprocessing import StandardScaler
from etl_inputs import etl_inputs


def inputs(output_variable):
    """Input Processing function uses EDA,Feature Engieering to feed data into the model.

    Parameters
    ----------
    output_variable : TYPE
        DESCRIPTION.

    Returns
    -------
    x_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    test_features : TYPE
        DESCRIPTION.

    """
    # Load input files
    train_data, test_data = etl_inputs()
    train_data = train_data.drop_duplicates()

    # EDA for Raw Data
    '''featureupdates = swz.FeatureConfig(force_num=["day", "month", "year", "week_day"], force_cat=["product", "product_category", "product_subcategory"])
    EDA_Raw = swz.analyze(train_data.compute(), target_feat=output_variable, feat_cfg=featureupdates)
    EDA_Raw = swz.compare([train_data.compute(), "Train"], [test_data.compute(), "Test"], output_variable)
    EDA_Raw.show_html(".\Data Files\Raw_Processed EDA.html")
    EDA_test_Raw = swz.analyze(test_data.compute(), feat_cfg=featureupdates)
    EDA_test_Raw.show_html(".\Data Files\Test_Raw_Processed EDA.html")
    '''
    # Create train and test features
    y_train = train_data[output_variable]
    train_features = train_data.drop([output_variable], axis=1)
    test_features = test_data

    none_cols = ['discount_flag']
    num_cols = ['footfall']
    test_cat_cols = ['product_category', 'product_subcategory']
    test_num_cols = ['var_1', 'var_2', 'var_3', 'var_5', 'var_6', 'var_8', 'var_9', 'var_10']

    # Drop and Impute missing data (Tune the split and threshold for more control on numerical and text features)
    train_columns = train_features[num_cols].replace(np.nan, train_features[num_cols].mean())
    train_features[train_columns.columns.tolist()] = train_columns
    train_columns = train_features[none_cols].replace(np.nan, None)
    train_features[train_columns.columns.tolist()] = train_columns
    test_columns = test_features[test_cat_cols].replace(np.nan, test_features[test_cat_cols].mean().round(0))
    test_features[test_columns.columns.tolist()] = test_columns
    test_columns = test_features[test_num_cols].replace(np.nan, test_features[test_num_cols].mean())
    test_features[test_columns.columns.tolist()] = test_columns

    # Rescale x_train
    sc = StandardScaler()
    x_train = sc.fit_transform(train_features)
    # test_features = sc.fit_transform(test_features)

    # Check for skewness in y_train and apply log for data modelling
    y_train = da.log1p(da.log1p(y_train))

    return x_train, y_train, test_features
