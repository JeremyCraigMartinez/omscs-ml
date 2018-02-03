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

def run_all(data, **kargs):
    split_data_set = data()
    results = []

    results.append(('boosting', boosting(*split_data_set)))
    results.append(('k_nearest_neighbor', knn(*split_data_set)))
    results.append(('decision_tree', decision_tree(*split_data_set)))
    results.append(('neural_network', ann(*split_data_set, **kargs)))
    results.append(('support_vector_machine', svm(*split_data_set)))

    for result in results:
        alg = result[0]
        cm = result[1]

        print('Confusion matrix for {}: {}'.format(alg, cm))

if __name__ == '__main__':
    run_all(rp, input_dim=1500)
    run_all(hp, input_dim=46)
