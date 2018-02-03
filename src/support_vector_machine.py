from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def get_classifier(X_train, X_test, y_train, y_test, kernel, **kargs):
    # Fitting SVM to the Training set
    classifier = SVC(kernel=kernel, **kargs)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    print('svm with {} kernel\n'.format(kernel), confusion_matrix(y_test, y_pred))

def svm(X_train, X_test, y_train, y_test, **kargs):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    get_classifier(X_train, X_test, y_train, y_test, 'linear', **kargs)
    get_classifier(X_train, X_test, y_train, y_test, 'rbf', **kargs)
    get_classifier(X_train, X_test, y_train, y_test, 'poly', **kargs)
    get_classifier(X_train, X_test, y_train, y_test, 'sigmoid', **kargs)
