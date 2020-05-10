import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 20)

#otwarcie pliku i pobranie danych
data = pd.read_csv("Adult_train.tab", sep='\t')

#zamiana ? na wartosci nan i wyrzucenie wierszy ktore je zawieraja
data = data.replace('?',np.nan)
# print(data.isnull().sum())
# print(data.corr())
data = data.dropna(how='any')
# print(data.isnull().sum())


#oczyszczanie danych
# print(data)
data = data.drop(labels=["education","native-country","relationship"],axis=1)
# print(data.describe)
# print(data.shape)
# print(data.dtypes)
# print(data.describe(include=[np.object]))

#mapowanie danych obiektowych na liczbowe

mapper = {}
def DatasetPreprocessing(data, columns_to_map):
    # mapowanie
    data_clean = data
    for column_name in columns_to_map:
        global mapper
        for index, category in enumerate(data[column_name].unique()):
            mapper[category] = index

        data_clean[column_name] = data[column_name].map(mapper)
    # print(mapper)
    return data_clean

#wywołanie funkcji mapującej, ktora zmienia wartosci typu Object na wartosci liczbowe.
#y<=50k to będzie 0, a powyżej 50k to będzie 1
data_clean = (DatasetPreprocessing(data,["workclass","marital-status","occupation","race","sex","y"]))
# print(data_clean)

#utworzenie tabeli bazy danych
#później

#podzial zbbioru na czesc treningowa i testowa
def splitDatasetIntoTrainAndTest(X, y, train_split_percent=0.6):

    # print(X.info())
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split_percent)
    return X_train, X_test, y_train, y_test


dataset_X = data_clean.filter(["age","workclass","fnlwgt","education-num","marital-status","occupation",
                           "race","sex","capital-gain","capital-loss","hours-per-week"])
y=data_clean["y"]

# splitDatasetIntoTrainAndTest(X=,y=)
X_train, X_test, y_train, y_test = splitDatasetIntoTrainAndTest(X=dataset_X,y=y)

