# -*- coding: utf-8 -*-
"""Base File to feed Input and Predict Data.

@author: Varshith
"""
from inputs import inputs
from datetime import datetime
from dask_ml.model_selection import train_test_split
from tpot import TPOTRegressor
from etl_inputs import etl_inputs

start = datetime.now()

output_variable = 'sales'

# Create training and testing data
train, test = etl_inputs()
train_x, train_y, test_data = inputs(output_variable)


# Create and Train Models
if train_x.all().isna().sum() == 0:
    print("\n")
    print("No Null Values in model input")
    print("-----------------------------\n\n")
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.25)
    tpot = TPOTRegressor(generations=2, n_jobs=-1, verbosity=2,
                         scoring='neg_root_mean_squared_error', use_dask=True)
    features = x_train
    target = y_train[output_variable]
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
    
    
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.25)
    tpot = TPOTRegressor(generations=20, n_jobs=-1, verbosity=2, use_dask=True)
    features = x_train.astype(float)
    target = pd.Series(y_train['Survived'], index=y_train.index).astype(float)
    tpot.fit(features, target)
'''