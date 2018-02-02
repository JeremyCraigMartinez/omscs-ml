from reviews_preprocessing import get_train_test_set

from boosting import boosting
from k_nearest_neighbor import knn
from decision_tree import decision_tree
from neural_network import ann
from support_vector_machine import svm

if __name__ == '__main__':
    split_data_set = get_train_test_set()
    results = []

    results.append(('boosting', boosting(*split_data_set)))
    results.append(('k_nearest_neighbor', knn(*split_data_set)))
    results.append(('decision_tree', decision_tree(*split_data_set)))
    results.append(('neural_network', ann(*split_data_set)))
    results.append(('support_vector_machine', svm(*split_data_set)))

    for result in results:
        alg = result[0]
        cm = result[1]

        print('Confusion matrix for {}: {}'.format(alg, cm))
