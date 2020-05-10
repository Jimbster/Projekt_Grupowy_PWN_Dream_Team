from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

import Main
import ML_algo_testing_main


X_train= Main.X_train
X_test= Main.X_test
y_train= Main.y_train
y_test= Main.y_test

def ensableClassifier(clfs, X_train, X_test, y_train):
    y_preds = []
    # trenowanie i testowanie wszystkich klasyfikatorów z listy clfs
    for clf in clfs:
        clf.fit(X_train, y_train)
        y_preds.append(clf.predict(X_test))
    # głosowanie większościowe
    y_result = y_preds[0]
    clf_index = 1
    while (clf_index < len(y_preds)):
        index = 0
        while (index < len(y_result)):
            y_result[index] = y_result[index] + y_preds[clf_index][index]
            index += 1
        clf_index += 1
    # uśrednianie i zaokrąglanie
    for index, y in enumerate(y_result):
        y_result[index] = round(y_result[index] / len(clfs))
    return y_result

y_pred_ensable_train = ensableClassifier(
    [RandomForestClassifier(),KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier()], X_train, X_train, y_train)
y_pred_ensable_test = ensableClassifier(
    [RandomForestClassifier(),KNeighborsClassifier(n_neighbors=5), DecisionTreeClassifier()], X_train, X_test, y_train)

# ML_algo_testing_main.getClassificationScore("Ensemble",y_test,y_pred_ensable_test)

#male_fnlwgt = X_train['education-num'].value_counts()
#print(male_fnlwgt)


def plotRestults():


    pyplot.title("age and education")
    pyplot.xlabel("age")
    pyplot.ylabel("education")
    pyplot.scatter(X_train["age"], X_train["education-num"], c=y_pred_ensable_train)

    pyplot.show()

plotRestults()