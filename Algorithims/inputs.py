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
from dask.array.stats import skew
import sweetviz as swz
import numpy as np
import featuretools as ft
from dask_ml.preprocessing import StandardScaler
from datetime import datetime
from etl_inputs import etl_inputs

# def inputs(train, test, output_variable):
"""Input Processing function uses EDA,Feature Engieering to feed data into the model.

Parameters
----------
train_file : TYPE
    DESCRIPTION.
test_file : TYPE
    DESCRIPTION.
output_variable : TYPE
    DESCRIPTION.

Returns
-------
x_train : TYPE
    DESCRIPTION.
y_train : TYPE
    DESCRIPTION.

"""
start = datetime.now()

# Load input files
train_data, test_data = etl_inputs()
output_variable = 'sales'


# EDA for Raw Data
'''featureupdates = swz.FeatureConfig(force_num=["day", "month", "year", "week_day"], force_cat=["product", "product_category", "product_subcategory"])
EDA_Raw = swz.analyze(train_data.compute(), target_feat=output_variable, feat_cfg=featureupdates)
# EDA_Raw = swz.compare([train_data.compute(), "Train"], [test_data.compute(), "Test"], output_variable)
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
test_columns = test_features[test_cat_cols].replace(np.nan, test_features[test_cat_cols].mean().round(0))
test_features[test_columns.columns.tolist()] = test_columns
test_columns = test_features[test_num_cols].replace(np.nan, test_features[test_num_cols].mean())
test_features[test_columns.columns.tolist()] = test_columns

# Feature Engineering
'''feature_type = {'city': ft.variable_types.Categorical,
                'product': ft.variable_types.Categorical,
                'footfall': ft.variable_types.Numeric,
                'discount_flag': ft.variable_types.Categorical,
                'product_category': ft.variable_types.Categorical,
                'product_subcategory': ft.variable_types.Categorical,
                'var_1': ft.variable_types.Numeric,
                'var_2': ft.variable_types.Numeric,
                'var_3': ft.variable_types.Numeric,
                'var_4': ft.variable_types.Numeric,
                'var_5': ft.variable_types.Numeric,
                'var_6': ft.variable_types.Numeric,
                'var_7': ft.variable_types.Numeric,
                'var_8': ft.variable_types.Numeric,
                'var_9': ft.variable_types.Numeric,
                'var_10': ft.variable_types.Numeric,
                'day': ft.variable_types.Datetime,
                'month': ft.variable_types.Datetime,
                'year': ft.variable_types.Datetime,
                'week_day': ft.variable_types.Datetime}
ignore = {'train_feature_engineering': ['city', 'day', 'month', 'year', 'week_day']}
es = ft.EntitySet(id='train_feature_engineering')
es = es.entity_from_dataframe(entity_id='train_feature_engineering',
                              dataframe=train_features, make_index=True, index='index',
                              variable_types=feature_type)
# es = es.normalize_entity(base_entity_id='train_feature_engineering', new_entity_id='sample', index='city')
features, feature_names = ft.dfs(entityset=es, target_entity='train_feature_engineering', ignore_variables=ignore, max_depth=3)
print(feature_names)

# Drop null Columns
col = list(train_features.columns[train_features.isna().any()])
x_train = train_features.drop(col, axis=1)

corelation = x_train.compute().corrwith(y_train.compute())
corelation = corelation.abs()
corelation = corelation.sort_values(ascending=False)
'''

# Rescale x_train
sc = StandardScaler()
x_train = sc.fit_transform(train_features)

# Check for skewness in y_train and apply log for data modelling
y_train = da.log1p(y_train)
y_skew = skew(y_train)

elapsed = datetime.now() - start
# return x_train, y_train, test_features
