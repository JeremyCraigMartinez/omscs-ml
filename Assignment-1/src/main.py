#!/usr/bin/env python

import sys
sys.path.append('/Users/jeremy.martinez/georgia-tech-code/omscs-ml/src')

from time import time
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold

from helpers.reviews_preprocessing import get_train_test_set as rp
from helpers.health_data_preprocessing import get_train_test_set as hp
from helpers.graph import plot

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
    'knn': {
        'n_neighbors': 3
    },
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
    'knn': {
        'n_neighbors': 3,
    },
}

def run_each(X, y, algorithm, alg_name, **kargs):
    print('\x1b[6;30;42m' + 'Confusion matrix for {}'.format(alg_name) + '\x1b[0m')
    start_time = time()

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy = []
    precision = []
    specificity = []
    true_pos_rate = []
    true_neg_rate = []
    cms = []

    for train, test in kfold.split(X, y):
        metrics = algorithm(X[train],
                            X[test],
                            y[train],
                            y[test],
                            **kargs)
        classifier, cm, acc, prec, spec, tpr, fpr = metrics
        accuracy.append(acc)
        precision.append(prec)
        specificity.append(spec)
        true_pos_rate.append(tpr)
        true_neg_rate.append(fpr)
        cms.append(cm)

    mean = lambda n: np.mean(n) * 100
    std = lambda n: np.std(n) * 100
    print("{}".format(np.array(cms).sum(axis=0) / 10))
    print("accuracy: %.2f%% (+/- %.2f%%)" % (mean(accuracy), std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (mean(precision), std(precision)))
    print("specificity: %.2f%% (+/- %.2f%%)" % (mean(specificity), std(specificity)))
    print("true positive rate: %.2f%% (+/- %.2f%%)" % (mean(true_pos_rate), std(true_pos_rate)))
    print("false positive rate: %.2f%% (+/- %.2f%%)" % (mean(true_neg_rate), std(true_neg_rate)))

    elapsed_time = time() - start_time
    print('elapsed time {}'.format(elapsed_time))

    return classifier

def run_all(data, **kargs):
    X, y = data()
    classifiers = []
    classifiers.append((run_each(X, y, ann, 'neural_network', **kargs['neural_network']), 'neural_network'))

    classifiers.append((run_each(X, y, boosting, 'boosting', **kargs['boosting']), 'boosting'))

    classifiers.append((run_each(X, y, decision_tree, 'decision_tree', **kargs['decision_tree']), 'decision_tree'))

    # best k for kNN is 3
    classifiers.append((run_each(X, y, knn, 'knn', **kargs['knn'])))
    # other values of k for your consideration
    #classifiers.append((run_each(X, y, knn, 'knn', **{ 'n_neighbors': 4 }), 'knn'))
    #classifiers.append((run_each(X, y, knn, 'knn', **{ 'n_neighbors': 5 }), 'knn'))
    #classifiers.append((run_each(X, y, knn, 'knn', **{ 'n_neighbors': 6 }), 'knn'))

    # all svm kernels
    classifiers.append((run_each(X, y, svm, 'support_vector_machine', **{**kargs['support_vector_machine'], 'kernel': 'linear'}), 'support_vector_machine'))
    classifiers.append((run_each(X, y, svm, 'support_vector_machine', **{**kargs['support_vector_machine'], 'kernel': 'rbf'}), 'support_vector_machine'))
    classifiers.append((run_each(X, y, svm, 'support_vector_machine', **{**kargs['support_vector_machine'], 'kernel': 'poly'}), 'support_vector_machine'))
    classifiers.append((run_each(X, y, svm, 'support_vector_machine', **{**kargs['support_vector_machine'], 'kernel': 'sigmoid'}), 'support_vector_machine'))

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    plot(classifiers, X, y, cv=kfold)

if __name__ == '__main__':
    print('\x1b[3;33;44m' + '~~~~~~~~~~~~~~~~~~~ Cervical Cancer ~~~~~~~~~~~~~~~~~~~~' + '\x1b[0m')
    run_all(hp, **hp_alg_arguments)
    print('\x1b[3;33;44m' + '~~~~~~~~~~~~~~~~~~ Restaurant Reviews ~~~~~~~~~~~~~~~~~~' + '\x1b[0m')
    run_all(rp, **rp_alg_arguments)
