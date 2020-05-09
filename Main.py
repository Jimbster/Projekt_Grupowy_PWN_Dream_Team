#importy

import pandas as pd
import sklearn as sklearn
import numpy as np

pd.set_option('display.max_columns', 20)

#otwarcie pliku i pobranie danych

data = pd.read_csv("Adult_train.tab", sep='\t')

#zamiana ? na wartosci nan i wyrzucenie wierszy ktore je zawieraja
data = data.replace('?',np.nan)
print(data.isnull().sum())
print(data.corr())
data = data.dropna(how='any')
print(data.isnull().sum())


#oczyszczanie danych

print(data)
data = data.drop(labels=["education","native-country","relationship"],axis=1)
print(data.describe)
print(data.shape)
print(data.dtypes)
print(data.describe(include=[np.object]))

#mapowanie danych obiektowych na liczbowe

def DatasetPreprocessing(data, columns_to_map):
    # mapowanie
    data_clean = data
    for column_name in columns_to_map:
        mapper = {}
        for index, category in enumerate(data[column_name].unique()):
            mapper[category] = index

        data_clean[column_name] = data[column_name].map(mapper)

    return data_clean

print(DatasetPreprocessing(data,"workclass"))
#utworzenie tabeli bazy danych


#podzial zbbioru na czesc treningowa i testowa