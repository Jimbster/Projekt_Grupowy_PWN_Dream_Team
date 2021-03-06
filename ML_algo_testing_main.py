from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import Main



X_train= Main.X_train
X_test= Main.X_test
y_train= Main.y_train
y_test= Main.y_test

#trenowanie modelu
def trainAndTestClassifier(clf, X_train, X_test, y_train):
    # print(clf)
    # trenowanie
    clf.fit(X_train, y_train)
    # testowanie
    y_pred = clf.predict(X_test)
    return y_pred

def getClassificationScore(clf_name ,y_test, y_pred):
    print("Nazwa klasyfikatora: " + clf_name)
    print("Dokładność klasyfikacji: " + str(accuracy_score(y_test, y_pred)))
    print("Macierz konfuzji: " + str(confusion_matrix(y_test, y_pred)))
    print("F1: " + str(f1_score(y_test, y_pred)))
    print("Czułość: "+str(precision_score(y_test, y_pred)))
    print("Pole pod krzywą AUC: " + str(roc_auc_score(y_test, y_pred)))
    print("Specyficzność: "+str(recall_score(y_test, y_pred)))


# y_pred_Knn = trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5),X_train,X_test, y_train)
# getClassificationScore("Knn", y_test, y_pred_Knn)
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
#
# y_pred_tree = trainAndTestClassifier(DecisionTreeClassifier(), X_train,X_test,y_train)
# getClassificationScore("Decision Tree", y_test, y_pred_tree)
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
#
# y_pred_forrest = trainAndTestClassifier(RandomForestClassifier(), X_train,X_test,y_train)
# getClassificationScore("Random Forrest", y_test, y_pred_forrest)
#
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# #do tego ewentualnie wrócić w przyszłości! Brak stabilnych wyników dla kążdego przebiegu!
# y_pred_svm_rbf_train = trainAndTestClassifier(SVC(kernel='linear', gamma='auto', max_iter=500), X_train,X_test,y_train)
# getClassificationScore("SVC", y_test, y_pred_svm_rbf_train)
#
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def crossValidation(clf, clf_name, X, y, folds=5):
    print("cross-validation: " + clf_name)
    scores = cross_val_score(clf, X, y, cv=folds)
    print(scores)
    # wynik mean i stddev testów kroswalidacji
    print("mean: " + str(scores.mean()))
    print("stddev: " + str(scores.std()))


# print("\n\n\n"+"@"*10+" CROSS WALIDACJA "+"@"*10)
#
# clf = RandomForestClassifier()
# crossValidation(clf, 'Random Forrest', Main.dataset_X, Main.y, folds=5)
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# clf = KNeighborsClassifier(n_neighbors=5)
# crossValidation(clf, 'Knn-5', Main.dataset_X, Main.y, folds=5)
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# clf = DecisionTreeClassifier()
# crossValidation(clf, 'Decision Tree', Main.dataset_X, Main.y, folds=5)
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# clf=GaussianNB()
# crossValidation(clf, 'Naive Bayes', Main.dataset_X, Main.y, folds=5)
#
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# clf=MLPClassifier()
# crossValidation(clf, 'MLP Classifier', Main.dataset_X, Main.y, folds=5)


#uznany za nieefektywny
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# clf = SVC(kernel='linear', gamma='auto', max_iter=30000)
# crossValidation(clf, 'SVC', Main.dataset_X, Main.y, folds=5)



