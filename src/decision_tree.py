from nl_preprocessing import get_train_test_set

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = get_train_test_set()

def decision_tree(X_train, X_test, y_train, y_test):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Decision Tree Classification to the Training set
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    return cm

if __name__ == '__main__':
    split_data_set = get_train_test_set()
    cm = naive_bayes(*split_data_set)
    print(cm)
