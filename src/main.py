#!/usr/bin/env python

import warnings

from reviews_preprocessing import get_train_test_set as rp
from health_data_preprocessing import get_train_test_set as hp

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
        'input_dim': 1500
    },
}

hp_alg_arguments = {
    'decision_tree': {
        'max_depth': 5,
        'min_weight_fraction_leaf': 0.05,
        'random_state': 100,
    },
    'neural_network': {
        'input_dim': 46
    },
}
def run_all(data, **kargs):
    split_data_set = data()
    results = []

    results.append(('boosting', boosting(*split_data_set)))
    results.append(('k_nearest_neighbor', knn(*split_data_set)))
    results.append(('decision_tree', decision_tree(*split_data_set, **kargs['decision_tree'])))
    results.append(('neural_network', ann(*split_data_set, input_dim=kargs['input_dim'])))
    results.append(('support_vector_machine', svm(*split_data_set)))

    for result in results:
        alg = result[0]
        cm = result[1]

        print('Confusion matrix for {}: {}'.format(alg, cm))

if __name__ == '__main__':
    # setting max depth for this dataset does not improve predicition accuracy
    run_all(rp, **rp_alg_arguments)
    run_all(hp, **hp_alg_arguments)
