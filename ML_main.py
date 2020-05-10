from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

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
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

y_pred_SVM = trainAndTestClassifier(SVC(kernel="linear"),X_train,X_test, y_train)

# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# getClassificationScore("SVC", y_test, y_pred_SVM)












# y_pred_knn5_train = c.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_train,y_train)
# y_pred_knn5_test = c.trainAndTestClassifier(KNeighborsClassifier(n_neighbors=5), X_train,X_test,y_train)
# y_pred_tree_train = c.trainAndTestClassifier(DecisionTreeClassifier(), X_train,X_train,y_train)
# y_pred_tree_test = c.trainAndTestClassifier(DecisionTreeClassifier(), X_train,X_test,y_train)
# y_pred_svm_lin_train = c.trainAndTestClassifier(SVC(kernel='linear'), X_train,X_train,y_train)
# y_pred_svm_lin_test = c.trainAndTestClassifier(SVC(kernel='linear'), X_train,X_test,y_train)
# y_pred_svm_rbf_train = c.trainAndTestClassifier(SVC(kernel='rbf', gamma='auto'), X_train,X_train,y_train)
# y_pred_svm_rbf_test = c.trainAndTestClassifier(SVC(kernel='rbf', gamma='auto'), X_train,X_test,y_train)