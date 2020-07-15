# -*- coding: utf-8 -*-
"""Base File to feed Input and Predict Data.

@author: Varshith
"""
from inputs import inputs
import timeit
from dask_ml.model_selection import train_test_split
from tpot import TPOTClassifier
import pandas as pd

start = timeit.default_timer()

train_file = r"Data Files\train.csv"
test_file = r"Data Files\test.csv"
output_variable = 'Survived'

# Create training data
train_x, train_y = inputs(train_file, test_file, output_variable)
train_x = pd.DataFrame(train_x.to_numpy())

# EDA using sweetviz for associations on  features vs target
# Create and Train Models
if train_x.all().isna().sum() == 0:
    print("No Null Values in model feed")
    x_train, x_test, y_train, y_test = train_test_split(train_x.values, train_y.values, test_size=0.25)
    tpot = TPOTClassifier(n_jobs=-1, verbosity=2, use_dask=True)
    features = x_train.astype(float)
    target = y_train.astype(float).ravel()
    #tpot.fit(features, target)
    # tpot.score(x_test, y_test)
    # tpot.export(output_file_name='pipeline.py')

elapsed = timeit.default_timer() - start
