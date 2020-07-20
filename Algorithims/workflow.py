# -*- coding: utf-8 -*-
"""Base File to feed Input and Predict Data.

@author: Varshith
"""
from inputs import inputs
from datetime import datetime
from dask_ml.model_selection import train_test_split
from tpot import TPOTRegressor
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

output_variable = 'sales'


def workflow():
    """Workflow Function.

    Returns
    -------
    None.

    """
    # Create training and testing data
    train_x, train_y, test_data = inputs(output_variable)
    panel_train = train_x['discount_flag'] + train_x['city'] + train_x['product_category'] + train_x['product_subcategory']
    train_x['panel'] = train_x['discount_flag'] + train_x['city'] + train_x['product_category'] + train_x['product_subcategory']
    panel_train = panel_train.drop_duplicates().compute().tolist()
    panel_test = test_data['discount_flag'] + test_data['city'] + test_data['product_category'] + test_data['product_subcategory']
    test_data['panel'] = test_data['discount_flag'] + test_data['city'] + test_data['product_category'] + test_data['product_subcategory']
    panel_test = panel_test.drop_duplicates().compute().tolist()
    panel = [x for x in panel_train if x in panel_test]
    train_data = dd.concat([train_x, train_y], axis=1)
    train_data = train_data.drop(['discount_flag', 'city', 'product_category', 'product_subcategory', 'product'])
    test_data = test_data.drop(['discount_flag', 'city', 'product_category', 'product_subcategory', 'product'])
    train_data = train_data.loc[train_x['panel'] == panel]
    train_data = train_data.groupby(train_data.panel).tolist()
    test_data = test_data.groupby(test_data.panel).tolist()


    #print(elapsed_init)
    # Create and Train Models
    if train_x.isna().any() == False:
        print("\nNo Null Values in model input")
        print("-----------------------------\n\n")
        tpot = TPOTRegressor(generations=5, n_jobs=-1, verbosity=2,
                             scoring='neg_root_mean_squared_error', use_dask=True)
        score = None
        for i in range(panel):
            train_data[i] = train_data[i].drop(['panel'], axis=1)
            test_data[i] = test_data[i].drop(['panel'], axis=1)
            train_y = train_data[i]['sales']
            train_x = train_data[i].drop(['sales'], axis=1)
            x_train, x_test, y_train, y_test = train_test_split(train_x[i], train_y[i], test_size=0.25)
            tpot[i].fit(x_train, y_train)
            score[i] = tpot[i].score(x_test, y_test)
            test_data[i]['sales'] = tpot[i].predict(test_data[i].drop(['id'], axis=1))
            test_data[i].to_csv(r"Data Files\Submission.csv", mode='a')
            # .values.reshape(len(y_train.index), 1)
        test_pred = dd.read_csv(r"Data Files\Submission.csv")
        test_pred = test_pred.drop_duplicates().compute()
        test_pred.to_csv(r"Data Files\Submission.csv")
        return score
    else:
        return "Null values present"



if __name__ == '__main__':
    start = datetime.now()
    print(start)
    cluster = LocalCluster()
    client = Client(cluster)

    future = client.submit(workflow)
    print(future.result())
    elapsed = datetime.now() - start
    print("Total elapsed{}".format(elapsed))

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