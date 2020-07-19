# -*- coding: utf-8 -*-.
"""
Loads required inputs and data files for the model.

1. Load the required inputs for data modelling
2. Perform the necessary ETL, EDA and feature engineering to feed data
   into the statistical model

@author: Varshith
"""


# Import required libraries

import dask.dataframe as dd
from functions.detect_outliner import detect_outliner
import sweetviz as swz
import numpy as np
import featuretools as ft
from sklearn.preprocessing import StandardScaler
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
test_cols = ['product_category', 'product_subcategory', 'var_1', 'var_2',
             'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'var_9',
             'var_10']

# Drop and Impute missing data (Tune the split and threshold for more control on numerical and text features)
train_columns = train_features[num_cols].replace(np.nan, train_features[num_cols].mean())
train_features[train_columns.columns.tolist()] = train_columns
test_columns = test_features[test_cols].replace(np.nan, test_features[test_cols].mean())
test_features[test_columns.columns.tolist()] = test_columns

# Append train and test data for Analysis
features = dd.concat([train_features, test_features])
features[none_cols] = features[none_cols].replace(np.nan, 'None')

# Feature Engineering
feature_type = {'city': ft.variable_types.Text,
                'product': ft.variable_types.Text,
                'footfall': ft.variable_types.Numeric,
                'discount_flag': ft.variable_types.Numeric,
                'product_category': ft.variable_types.Numeric,
                'product_subcategory': ft.variable_types.Numeric,
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
                'week_day': ft.variable_types.Datetime,
                'id': ft.variable_types.Numeric}
ignore = {'train_feature_engineering': ['id']}
es = ft.EntitySet(id='train_feature_engineering')
es = es.entity_from_dataframe(entity_id='train_feature_engineering',
                              dataframe=features, make_index=True, index='index',
                              variable_types=feature_type)
# es = es.normalize_entity(base_entity_id='train_feature_engineering', new_entity_id='Pclass', index='Pclass')
features, feature_names = ft.dfs(entityset=es, target_entity='train_feature_engineering', ignore_variables=ignore)
# features = dd.from_pandas(features_pd, npartitions=1)

elapsed = datetime.now() - start
'''
# Identify Outliners
# features = dd.from_pandas(detect_outliner(features.compute()), npartitions=1)
outliner_count = np.count_nonzero(features['Outliner'] == 1)

# Drop null Columns
for col in features.columns:
    if features[col].compute().isnull().sum() > 0:
        features.drop(col, axis=1, inplace=True)

# Handle Outliner (log transofrm, drop, binning based on the output)
y_train = pd.concat([y_train,features['Outliner']], axis=1)
features = features[features['Outliner']==0]
y_train = y_train[y_train['Outliner']==0]
features = features.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Drop Corelated Data
features = features.drop(columns='Outliner')
y_train = y_train.drop(columns='Outliner')
features_corelation_matrix = features.corr().abs()
upper_matrix = features_corelation_matrix.where(np.triu(np.ones(features_corelation_matrix.shape), k=1).astype(np.bool))
collinear_features = [column for column in upper_matrix.columns if any(upper_matrix[column] > 0.7)]
x_train = features.drop(columns=collinear_features)

processed_data = pd.concat([x_train, y_train], axis=1)
corelation = processed_data.corrwith(processed_data[output_variable])
corelation = corelation.abs()
corelation = corelation.sort_values(ascending=False)

# Rescale your x_train if necessary
sc = StandardScaler()
x_train_col = x_train.columns
x_train = sc.fit_transform(x_train)
x_train = pd.DataFrame(x_train, columns=x_train_col)

# Check for skewness in y_train and alter accordingly


# return x_train, y_train
'''