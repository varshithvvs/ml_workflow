# -*- coding: utf-8 -*-.
"""
Loads required inputs and data files for the model.

1. Load the required inputs for data modelling
2. Perform the necessary ETL, EDA and feature engineering to feed data
   into the statistical model

@author: Varshith
"""


# Import required libraries
import warnings
warnings.filterwarnings("ignore")

import dask.dataframe as dd
from functions.drop_missing import drop_missing
from functions.encode import encode
from functions.detect_outliner import detect_outliner
import sweetviz as swz
import numpy as np
import pandas as pd
import featuretools as ft
from mlbox.preprocessing.reader import Reader
from mlbox.preprocessing.drift_thresholder import Drift_thresholder
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
train, test = etl_inputs()
output_variable = 'sales'
# Load input files
train_data = train
test_data = test

# EDA for Raw Data
#EDA_Raw = 
EDA_Raw = swz.compare([train_data.compute(), "Train"], [test_data.compute(), "Test"], output_variable)
elapsed = datetime.now() - start
print("Total runtime" + elapsed)
# EDA_Raw.show_html(".\Data Files\Raw EDA.html")
'''
# Create train and test features
y_train = pd.DataFrame(train_data[output_variable].compute(), columns=output_variable.split())
train_features = train_data.drop([output_variable], axis=1)
test_features = test_data

none_cols = []  # Columns where Nan means none
zero_cols = []  # Columns where Nan means 0
merge_key = 'PassengerId'
# Describe data
print("\n\nTrain Data\n----------")
print(train_features.describe().compute())
print("\nTest Data\n---------")
print(test_features.describe().compute())

# Drop and Impute missing data (Tune the split and threshold for more control on numerical and text features)
train_features = drop_missing(features=train_features, missing_drop_threshold=0.2, start=2, split=8, end=10)
test_features = drop_missing(features=test_features, missing_drop_threshold=0.21, start=2, split=8, end=10)

# Append train and test data for Analysis
train_features = train_features.merge(train_data[none_cols+zero_cols+merge_key.split()].compute(), how='left', on=merge_key)
test_features = test_features.merge(test_data[none_cols+zero_cols+merge_key.split()].compute(), how='left', on=merge_key)
features = dd.concat([train_features, test_features])
features = features.assign(idx=1)
features = features.set_index(features.idx.cumsum()-1)
features = features.drop('idx', axis=1)

# Impute none_cols
for col in none_cols:
    features[col].replace(np.nan, 'None', inplace=True)

# Impute zero_cols
for col in zero_cols:
    features[col].replace(np.nan, 0, inplace=True)

# Encode categorical variables
label_encode = features.iloc[:, list(range(6, 8))]
hot_encode = None
Encoded_data = encode(x_train_txt_label_encode=label_encode, x_train_txt_hot_encode=hot_encode)
for col in Encoded_data.columns.tolist():
    features[col] = Encoded_data[col]

# Feature Engineering/ Feature Creation

# Create Qualitative feautres if any to optimize model performance


# Create Quantitative features
train_feature_engineering = features.compute()
es = ft.EntitySet(id='train_feature_engineering')
es = es.entity_from_dataframe(entity_id='train_feature_engineering', dataframe=train_feature_engineering, index='PassengerId', variable_types={"Pclass": ft.variable_types.Categorical})
es = es.normalize_entity(base_entity_id='train_feature_engineering', new_entity_id='Pclass', index='Pclass')
features_pd, feature_names = ft.dfs(entityset=es, target_entity='train_feature_engineering', max_depth=3)
features = dd.from_pandas(features_pd, npartitions=1)

# Identify Outliners
features = dd.from_pandas(detect_outliner(features.compute()), npartitions=1)
outliner_count = np.count_nonzero(features['Outliner'] == 1)

# Drop null Columns
for col in features.columns:
    if features[col].compute().isnull().sum() > 0:
        features.drop(col, axis=1, inplace=True)

# Drift Transform
dft = Drift_thresholder()
features_drift_y = features.head(len(train_features))
features_drift = {}
features_drift['train'] = features_drift_y
features_drift['test'] = features.tail(len(test_features))
features_drift['target'] = y_train
features_drift = dft.fit_transform(features_drift)

# Extract Train Data
features = features_drift['train']
features = features.reset_index(drop=True)

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