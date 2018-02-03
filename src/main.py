#!/usr/bin/env python

import warnings

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
    }
}
def run_all(data, **kargs):
    split_data_set = data()

    print('Confusion matrix for boosting')
    boosting(*split_data_set)
    print('Confusion matrix for k_nearest_neighbor')
    knn(*split_data_set)
    print('Confusion matrix for decision_tree')
    decision_tree(*split_data_set, **kargs['decision_tree'])
    print('Confusion matrix for neural_network')
    ann(*split_data_set, **kargs['neural_network'])
    print('Confusion matrix for support_vector_machine')
    svm(*split_data_set, **kargs['support_vector_machine'])

if __name__ == '__main__':
    # setting max depth for this dataset does not improve predicition accuracy
    run_all(rp, **rp_alg_arguments)
    run_all(hp, **hp_alg_arguments)

# for rp, we want even split
# for hp, we want minimal bottom left
