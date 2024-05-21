from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def classifiers(x_train, x_test, train_label, test_label):
    # SVM
    svm_clf = svm.SVC()
    svm_clf.fit(x_train, train_label)
    svm_pred_label = svm_clf.predict(x_test)
    svm_acc = accuracy_score(test_label, svm_pred_label)
    svm_f1 = f1_score(test_label, svm_pred_label, average='micro')

    # RF
    rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
    rf_clf.fit(x_train, train_label)
    rf_pred_label = rf_clf.predict(x_test)
    rf_acc = accuracy_score(test_label, rf_pred_label)
    rf_f1 = f1_score(test_label, rf_pred_label, average='micro')

    #KNN
    knn_cls = KNeighborsClassifier(n_neighbors=3)
    knn_cls.fit(x_train, train_label)
    knn_pred_label = knn_cls.predict(x_test)
    knn_acc = accuracy_score(test_label, knn_pred_label)
    knn_f1 = f1_score(test_label, knn_pred_label, average='micro')

    return svm_acc, svm_f1, rf_acc, rf_f1, knn_acc, knn_f1