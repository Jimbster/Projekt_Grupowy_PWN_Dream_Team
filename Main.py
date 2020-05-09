#importy

import pandas as pd
import sklearn as sklearn
import numpy as np

pd.set_option('display.max_columns', 20)
#pd.set_option('display.max_rows', None)


#otwarcie pliku i pobranie danych
#data = pd.read_csv("Adult_train.tab", sep='\t', header=2)
data = pd.read_csv("Adult_train.tab", sep='\t')
# print(data)
# print(data.isnull().sum())
# print(data.shape)
# print(data.head())
#print(data.describe(include=[pd.np.object]))
#print(data.dtypes)

data = data.replace('?',np.nan)
print(data.isnull().sum())
print(data.corr())
data = data.dropna(how='any')
print(data.isnull().sum())
#oczyszczanie danych

# do wywalenia - education, native country, relationship
#utworzenie tabeli bazy danych


#podzial zbbioru na czesc treningowa i testowa