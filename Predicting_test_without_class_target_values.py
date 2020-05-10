import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import Main
import ML_majority_voting_script_main



X_train= Main.X_train
X_test= Main.X_test
y_train= Main.y_train
y_test= Main.y_test


data_to_predict = pd.read_csv("Adults_test_without_class.tab", sep='\t')
data_to_predict = data_to_predict.drop(labels=["education","native-country","relationship","y"],axis=1)


def DatasetPreprocessing2(data, columns_to_map):
    # mapowanie
    data_clean = data
    mapper = Main.mapper
    for column_name in columns_to_map:
        data_clean[column_name] = data[column_name].map(mapper)
    return data_clean

data_clean_to_predict = DatasetPreprocessing2(data_to_predict,["workclass","marital-status","occupation","race","sex"])

# print(data_clean_to_predict)

# y_pred_ensable_test = ML_majority_voting_script_main.ensableClassifier(
#     [RandomForestClassifier(),KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier()], Main.dataset_X, data_clean_to_predict, Main.y)
#
# print(y_pred_ensable_test)
# with open("wynik.txt","w") as a:
#     a.write(str(y_pred_ensable_test))

