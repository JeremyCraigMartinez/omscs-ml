# classifier did significantly worse when attempting to prune with
# max_depth and tweak parameters like leaf weight and random state

from sklearn.ensemble import AdaBoostClassifier

from helpers.scoring import accuracy

def boosting(X_train, X_test, y_train, y_test):
    classifier = AdaBoostClassifier()

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    return accuracy(y_test, y_pred, classifier)
