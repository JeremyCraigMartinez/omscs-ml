# From scikit-learn docs:
#     Decision-tree learners can create over-complex trees that do not
#     generalise the data well. This is called overfitting. Mechanisms
#     such as pruning (not currently supported), setting the minimum
#     number of samples required at a leaf node or setting the maximum
#     depth of the tree are necessary to avoid this problem.
#
#     Pruning done to avoid this: precondition approached such as setting
#     max depth and weights to leaf nodes (where applicable)

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from helpers.scoring import accuracy

def decision_tree(X_train,
                  X_test,
                  y_train,
                  y_test,
                  max_depth=None,
                  random_state=None,
                  min_weight_fraction_leaf=0.0):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Decision Tree Classification to the Training set
    classifier = DecisionTreeClassifier(criterion='entropy',
                                        random_state=random_state,
                                        max_depth=max_depth,
                                        min_weight_fraction_leaf=min_weight_fraction_leaf)

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    accuracy(X_train, y_train, y_test, y_pred, classifier)
