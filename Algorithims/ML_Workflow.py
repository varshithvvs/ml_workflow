# Data Preprocessing
"""ML_Workflow template to build a pipeline for predictions.

@author:Varshtih
"""
import pandas as pd
import numpy as np
from autoimpute.imputations import MultipleImputer
from sklearn.preprocessing import StandardScaler # Use minmax scaler to avoid relative distance distortion
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import sweetviz
import seaborn as sns
from pyod.models.iforest import IForest
import featuretools as ft
import dask

# Load Input Files
train_data = pd.read_csv(r"C:\Users\svvar\PycharmProjects\ml_workflow\algorithims\data files\train.csv")
test_data = pd.read_csv(r"C:\Users\svvar\PycharmProjects\ml_workflow\algorithims\data files\test.csv")

train_data.info()
test_data.info()

# Fill in required Inputs
x_train = train_data.iloc[:, list(range(3, 11))]
y_train = train_data.iloc[:, list(range(11, 12))]
x_train_num = train_data.iloc[:, list(range(3, 9))]
x_train_txt = train_data.iloc[:, list(range(9, 11))]
x_train_txt_encode_split = 2  # Split at Column Number

x_test = test_data.iloc[:, list(range(3, 11))]
x_test_num = test_data.iloc[:, list(range(3, 9))]
x_test_txt = test_data.iloc[:, list(range(9, 11))]
x_test_txt_encode_split = 2  # Split at Column Number

# Impute Missing values

# Numerical Imputer
imputer_num = MultipleImputer(strategy='stochastic', return_list=True, n=5, seed=101)
x_train_num_avg = imputer_num.fit_transform(x_train_num)

x_train_num_concat = x_train_num_avg[0][1]

for i in range(len(x_train_num_avg)-1):
    x_train_num_concat = pd.concat([x_train_num_concat, x_train_num_avg[i+1][1]], axis=1)
x_train_num_avg = x_train_num_concat.groupby(by=x_train_num_concat.columns, axis=1).apply(lambda g: g.mean(axis=1))
x_train_num_avg_col = np.unique(list(x_train_num_concat.columns))

x_test_num_avg = imputer_num.fit_transform(x_test_num)

x_test_num_concat = x_test_num_avg[0][1]

for i in range(len(x_test_num_avg)-1):
    x_test_num_concat = pd.concat([x_test_num_concat, x_test_num_avg[i+1][1]], axis=1)
x_test_num_avg = x_test_num_concat.groupby(by=x_test_num_concat.columns, axis=1).apply(lambda g: g.mean(axis=1))
x_test_num_avg_col = np.unique(list(x_test_num_concat.columns))

# Categorical Imputer
imputer_txt = MultipleImputer(strategy='categorical', return_list=True, n=10, seed=101)
x_train_txt_avg = imputer_txt.fit_transform(x_train_txt)

x_train_txt_col = list(x_train_txt.columns)
x_train_txt_col.sort()
x_train_txt_concat = x_train_txt_avg[0][1]

for i in range(len(x_train_txt_avg)-1):
    x_train_txt_concat = pd.concat([x_train_txt_concat, x_train_txt_avg[i+1][1]], axis=1)
x_train_txt_avg = x_train_txt_concat.groupby(by=x_train_txt_concat.columns, axis=1).apply(lambda g: stats.mode(g, axis=1)[0])
x_train_txt_avg = x_train_txt_avg.sort_index(axis=0)
x_train_txt_avg_temp = pd.DataFrame(x_train_txt_avg[0])
for i in range(len(x_train_txt_avg)-1):
    x_train_txt_avg_temp = pd.concat([x_train_txt_avg_temp, pd.DataFrame(x_train_txt_avg[i+1])], axis=1)
x_train_txt_avg_temp.columns = x_train_txt_col
x_train_txt_avg = x_train_txt_avg_temp
x_train_txt = x_train_txt.sort_index(axis=1)


x_test_txt_avg = imputer_txt.fit_transform(x_test_txt)

x_test_txt_col = list(x_test_txt.columns)
x_test_txt_col.sort()
x_test_txt_concat = x_test_txt_avg[0][1]

for i in range(len(x_test_txt_avg)-1):
    x_test_txt_concat = pd.concat([x_test_txt_concat, x_test_txt_avg[i+1][1]], axis=1)
x_test_txt_avg = x_test_txt_concat.groupby(by=x_test_txt_concat.columns, axis=1).apply(lambda g: stats.mode(g, axis=1)[0])
x_test_txt_avg = x_test_txt_avg.sort_index(axis=0)
x_test_txt_avg_temp = pd.DataFrame(x_test_txt_avg[0])
for i in range(len(x_test_txt_avg)-1):
    x_test_txt_avg_temp = pd.concat([x_test_txt_avg_temp, pd.DataFrame(x_test_txt_avg[i+1])], axis=1)
x_test_txt_avg_temp.columns = x_test_txt_col
x_test_txt_avg = x_test_txt_avg_temp
x_test_txt = x_test_txt.sort_index(axis=1)

# Merge Imputed Training Data and Convert to Values
x_train_Imp = pd.concat([x_train_num, x_train_txt], axis=1)
x_train_Imp = x_train_Imp.iloc[:, :].values
x_train_num_avg = x_train_num_avg.iloc[:, :].values
x_train_txt_label_encode = x_train_txt_avg.iloc[:, list(range(0, x_train_txt_encode_split))].values
x_train_txt_hot_encode = x_train_txt_avg.iloc[:, list(range(x_train_txt_encode_split, len(x_train_txt_avg.columns)))].values
x_train_txt_label_encode_col = list(x_train_txt_avg.iloc[:, list(range(0, x_train_txt_encode_split))].columns)
x_train_txt_hot_encode_col = list(x_train_txt_avg.iloc[:, list(range(x_train_txt_encode_split, len(x_train_txt_avg.columns)))].columns)


x_test_Imp = pd.concat([x_test_num, x_test_txt], axis=1)
x_test_Imp = x_test_Imp.iloc[:, :].values
x_test_num_avg = x_test_num_avg.iloc[:, :].values
x_test_txt_label_encode = x_test_txt_avg.iloc[:, list(range(0, x_test_txt_encode_split))].values
x_test_txt_hot_encode = x_test_txt_avg.iloc[:, list(range(x_test_txt_encode_split, len(x_test_txt_avg.columns)))].values
x_test_txt_label_encode_col = list(x_test_txt_avg.iloc[:, list(range(0, x_test_txt_encode_split))].columns)
x_test_txt_hot_encode_col = list(x_test_txt_avg.iloc[:, list(range(x_test_txt_encode_split, len(x_test_txt_avg.columns)))].columns)


# Label Encode Categorical Variables
# Update onelabel eligible features only
labelencoder_X = LabelEncoder()
for i in range(np.shape(x_train_txt_label_encode)[1]):
     x_train_txt_label_encode[:, i-1] = labelencoder_X.fit_transform(x_train_txt_label_encode[:, i-1])

for i in range(np.shape(x_test_txt_label_encode)[1]):
    x_test_txt_label_encode[:, i-1] = labelencoder_X.fit_transform(x_test_txt_label_encode[:, i-1])

# Hot Encode Categorical Variables
# x_train_txt_hot_encode = pd.get_dummies(data=x_train_txt_avg.iloc[:, list(range(x_train_txt_encode_split, len(x_train_txt_avg.columns)))], columns=x_train_txt_hot_encode_col)
# x_train_txt_hot_encoded_col = list(x_train_txt_hot_encode.columns)
# x_train_txt_hot_encode = x_train_txt_hot_encode.values
# x_test_txt_hot_encode = pd.get_dummies(data=x_test_txt_avg.iloc[:, list(range(x_test_txt_encode_split, len(x_test_txt_avg.columns)))], columns=x_test_txt_hot_encode_col)
# x_test_txt_hot_encoded_col = list(x_test_txt_hot_encode.columns)
# x_test_txt_hot_encode = x_test_txt_hot_encode.values

x_train_Imp_En = pd.concat([pd.DataFrame(x_train_num_avg), pd.DataFrame(x_train_txt_label_encode)], axis=1)  # Update with Hot Encode Data if available
x_test_Imp_En = pd.concat([pd.DataFrame(x_test_num_avg), pd.DataFrame(x_test_txt_label_encode)], axis=1)  # Update with Hot Encode Data if available
x_train_Imp_En_Col = x_train_num_avg_col.tolist() + x_train_txt_label_encode_col
x_train_Imp_En.columns = x_train_Imp_En_Col
x_test_Imp_En_Col = x_test_num_avg_col.tolist() + x_test_txt_label_encode_col
x_test_Imp_En.columns = x_test_Imp_En_Col


# Feature Scaling
sc_x = StandardScaler()
x_train_Imp_En_Fs = sc_x.fit_transform(x_train_Imp_En)
x_test_Imp_En_Fs = sc_x.transform(x_test_Imp_En)
sc_y = StandardScaler()
y_train_Imp_En = y_train
y_train_Imp_En_Fs = sc_y.fit_transform(y_train_Imp_En)

# Detect and Handle Outliners (use the required variables accordingly with the data)
outliners = IForest(contamination=0.05)
x_train_outliner_data = x_train_Imp_En.copy()
outliners.fit(x_train_outliner_data)
x_train_outliner_anamoly_score = outliners.decision_function(x_train_outliner_data)*-1
x_train_outliner_detection = outliners.predict(x_train_outliner_data)
x_train_outliner_data['Outliner'] = x_train_outliner_detection.tolist()
x_train_outliner_count = np.count_nonzero(x_train_outliner_detection == 1)

# EDA Analysis with required data (update with necessary processed data based on the need)
# EDA = sweetviz.compare([train_data, "Train"], [test_data, "Test"], "Survived")
# EDA.show_html("Titanic Data EDA.html")
# EDA_describe = train_data.describe()

# Feature Engineering
train_feature_engineering = pd.concat([x_train_Imp_En], axis=1)
es = ft.EntitySet(id='train_feature_engineering')
es = es.entity_from_dataframe(entity_id='train_feature_engineering', dataframe=train_feature_engineering, index='PassengerId', variable_types={"Pclass": ft.variable_types.Categorical})
es = es.normalize_entity(base_entity_id='train_feature_engineering', new_entity_id='Pclass', index='Pclass')
features, feature_names = ft.dfs(entityset=es, target_entity='train_feature_engineering', max_depth=3)
features.info()
for col in features.columns:
    if features[col].isnull().sum() > 0:
        features.drop(col, axis=1, inplace=True)
features_corelation_matrix = features.corr().abs()
upper_matrix = features_corelation_matrix.where(np.triu(np.ones(features_corelation_matrix.shape), k=1).astype(np.bool))
collinear_features = [column for column in upper_matrix.columns if any(upper_matrix[column] > 0.7)]
x_train_final = features.drop(columns=collinear_features)
x_train_final.info()
