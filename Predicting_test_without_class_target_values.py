from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import Main
import ML_algo_testing_main
import ML_majority_voting_script_main
from ML_majority_voting_script_main import ensableClassifier


X_train= Main.X_train
X_test= Main.X_test
y_train= Main.y_train
y_test= Main.y_test


# y_prediction = ensableClassifier([RandomForestClassifier(),KNeighborsClassifier(n_neighbors=5),
#                    DecisionTreeClassifier()], X_train, X_test, y_train)

data_to_predict = pd.read_csv("Adults_test_without_class.tab")
print(data_to_predict)