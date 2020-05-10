import pandas as pd
import Main



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

print(data_clean_to_predict)