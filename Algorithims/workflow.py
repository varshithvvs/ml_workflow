# -*- coding: utf-8 -*-
"""Base File to feed Input and Predict Data.

@author: Varshith
"""
from inputs import inputs
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from tpot import TPOTClassifier
from etl_inputs import etl_inputs

start = datetime.now()

train_file = r"Data Files\train.csv"
test_file = r"Data Files\test.csv"
output_variable = 'Survived'

# Create training data
etl_inputs(train_file,test_file)

train_x, train_y = inputs(train_file, test_file, output_variable)
train_x = train_x[train_x.columns.drop(list(train_x.filter(regex='feature')))]

# EDA using sweetviz for associations on  features vs target
# Create and Train Models
if train_x.all().isna().sum() == 0:
    print("\n")
    print("No Null Values in model input")
    print("-----------------------------\n\n")
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.25)
    tpot = TPOTClassifier(generations=20, n_jobs=-1, verbosity=2)
    features = x_train.astype(float)
    target = pd.Series(y_train['Survived'], index=y_train.index).astype(float)
    tpot.fit(features, target)
elapsed = datetime.now() - start

'''
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.25)
    classifier = RandomForestClassifier(n_estimators=20, criterion='entropy', n_jobs=-1)
    classifier.fit(x_train, y_train.values.ravel())
    y_pred = classifier.predict(x_test)
    score = classifier.score(x_test, y_test)
    cm = confusion_matrix(y_test, y_pred)

    features = {}
    features['train'] = x_train
    features['target'] = pd.Series(y_train['Survived'], index=y_train.index).astype(int)
    features['test'] = x_test
    opt = Optimiser(scoring='accuracy', n_folds=5)
    space = {
            'est__strategy': {'space': 'RandomForest'}
             }
    score = opt.optimise(None, features)
    prd = Predictor()
    prd.fit_predict(score, features)
    y_pred = pd.read_csv(r'save\Survived_predictions.csv')
    y_pred = y_pred['Survived_predicted'].round(0)
    cm = confusion_matrix(y_test, y_pred)

'''