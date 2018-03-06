#!/usr/bin/env python
# python-2.7

import sys
import os
import csv
import pdb # pylint: disable=W0611
import time

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CWD)
sys.path.append('{}/../ABAGAIL/bin'.format(CWD))

from func.nn.backprop import BackPropagationNetworkFactory, RPROPUpdateRule, BatchBackPropagationTrainer
from func.nn.activation import RELU
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

INPUT_LAYER = 100
HIDDEN_LAYER_1 = 40
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 4001

class RandomSearch(object):
    """docstring for RandomSearch"""
    def __init__(self, **kargs):
        super(RandomSearch, self).__init__()
        self.outfile = '{}/../csv/OPTIMIZATION_FUNCTIONS/{}.tsv'.format(CWD, kargs['outfile'])
        self.search_alg = kargs['search_alg'] if 'search_alg' in kargs else None
        self.squared_error = SumOfSquaresError()
        self.network = None
        self.train_ds = []

    def __str__(self):
        return self.outfile

    def read_data(self, dataset, f):
        _dataset = []
        with open(f, 'r') as stream:
            _file = csv.reader(stream)
            for row in _file:
                attrs = row[:-1]
                label = row[-1]
                i = Instance([int(val) for val in attrs])
                i.setLabel(Instance(int(label)))
                _dataset.append(i)
        setattr(self, dataset, _dataset)

    def error(self, data):
        correct = incorrect = error = 0
        for row in getattr(self, data):
            self.network.setInputValues(row.getData())
            self.network.run()
            actual = row.getLabel().getContinuous()
            pred = self.network.getOutputValues().get(0)
            pred = max(min(pred, 1), 0)

            if abs(pred - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1

            output = row.getLabel()
            output_vals = self.network.getOutputValues()

            example = Instance(output_vals, Instance(output_vals.get(0)))
            error += self.squared_error.value(output, example)
        squared_error = error/float(len(getattr(self, data)))
        accuracy = correct/float(correct+incorrect)
        return (squared_error, accuracy)

    def write_header(self):
        with open(self.outfile, 'w') as f:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('iteration', 'error_train', 'error_test', 'accuracy_train', 'accuracy_test', 'elapsed'))

    def write_row(self, row):
        with open(self.outfile, 'a+') as f:
            f.write(row)
            print self.outfile, row

    def train(self):
        self.write_header()
        print "\nError results for %s\n---------------------------" % (self.outfile,)
        start = time.clock()
        for iteration in xrange(TRAINING_ITERATIONS):
            self.search_alg.train()
            elapsed = time.clock()-start
            if iteration % 10 == 0:
                error_train, acc_train = self.error('train_ds')
                error_test, acc_test = self.error('test_ds')
                row = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(iteration, error_train, error_test, acc_train, acc_test, elapsed)
                self.write_row(row)

    def run(self):
        self.read_data('train_ds', '{}/../csv/train.csv'.format(CWD))
        self.read_data('test_ds', '{}/../csv/test.csv'.format(CWD))
        getattr(self, 'train_ds')

        factory = BackPropagationNetworkFactory()
        train_ds = DataSet(self.train_ds)
        rule = RPROPUpdateRule()

        node_counts = [INPUT_LAYER, HIDDEN_LAYER_1, OUTPUT_LAYER]
        setattr(self, 'network', factory.createClassificationNetwork(node_counts, RELU()))

        optimizer = NeuralNetworkOptimizationProblem(train_ds, self.network, self.squared_error)
        # update binded function with optimizer, now ready to use
        if self.search_alg is None:
            self.search_alg = BatchBackPropagationTrainer(train_ds, self.network, self.squared_error, rule)
        else:
            setattr(self, 'search_alg', self.search_alg(optimizer))

        self.train()
