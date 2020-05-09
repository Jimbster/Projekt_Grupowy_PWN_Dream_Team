#importy

import pandas as pd
import sklearn as sklearn
import numpy as np

pd.set_option('display.max_columns', 20)
#pd.set_option('display.max_rows', None)


#otwarcie pliku i pobranie danych
data = pd.read_csv("Adult_train.tab", sep='\t', header=2)
# print(data)
# print(data.isnull().sum())
# print(data.shape)
# print(data.head())
print(data.describe(include=[pd.np.object]))
print(data.dtypes)

#oczyszczanie danych


#utworzenie tabeli bazy danych


#podzial zbbioru na czesc treningowa i testowa