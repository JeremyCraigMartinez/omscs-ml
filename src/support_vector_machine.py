from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from helpers.scoring import metrics

def svm(X_train, X_test, y_train, y_test, **kargs):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting SVM to the Training set
    classifier = SVC(**kargs)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    return metrics(y_test, y_pred)
