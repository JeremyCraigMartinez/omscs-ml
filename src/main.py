#!/usr/bin/env python

import warnings
import numpy as np

from helpers.reviews_preprocessing import get_train_test_set as rp
from helpers.health_data_preprocessing import get_train_test_set as hp

from boosting import boosting
from k_nearest_neighbor import knn
from decision_tree import decision_tree
from neural_network import ann
from support_vector_machine import svm

warnings.filterwarnings('ignore')

rp_alg_arguments = {
    'decision_tree': {
        'random_state': 100,
    },
    'neural_network': {
        'input_dim': 1500,
        'units': 400,
    },
    'support_vector_machine': {},
    'boosting': {},
}

hp_alg_arguments = {
    'decision_tree': {
        'max_depth': 5,
        'min_weight_fraction_leaf': 0.05,
        'random_state': 100,
    },
    'neural_network': {
        'input_dim': 46,
        'units': 600,
    },
    'support_vector_machine': {
        'coef0': 1.25
    },
    'boosting': {},
}

def run_each(X, y, algorithm, alg_name, **kargs):
    print('Confusion matrix for {}'.format(alg_name))

    kfold = StratifiedKFold(n_splits=7, shuffle=True)
    scores = []
    cms = []

    for train, test in kfold.split(X, y):
        cm, mean = algorithm(X[train],
                             X[test],
                             y[train],
                             y[test],
                             **kargs)
        scores.append(mean)
        cms.append(cm)

    print("{}".format(np.array(cms).sum(axis=0) / 10))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))

from sklearn.model_selection import StratifiedKFold
def run_all(data, **kargs):
    X, y = split_data_set = data()
    #run_each(X, y, knn, 'knn', **kargs['knn'])
    run_each(X, y, ann, 'neural_network', **kargs['neural_network'])
    run_each(X, y, boosting, 'boosting', **kargs['boosting'])
    run_each(X, y, decision_tree, 'decision_tree', **kargs['decision_tree'])

    # all svm kernels
    run_each(X, y, svm, 'support_vector_machine', **{**kargs['support_vector_machine'], 'kernel': 'linear'})
    run_each(X, y, svm, 'support_vector_machine', **{**kargs['support_vector_machine'], 'kernel': 'rbf'})
    run_each(X, y, svm, 'support_vector_machine', **{**kargs['support_vector_machine'], 'kernel': 'poly'})
    run_each(X, y, svm, 'support_vector_machine', **{**kargs['support_vector_machine'], 'kernel': 'sigmoid'})

if __name__ == '__main__':
    # setting max depth for this dataset does not improve predicition accuracy
    run_all(hp, **hp_alg_arguments)
    run_all(rp, **rp_alg_arguments)
