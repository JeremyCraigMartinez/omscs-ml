#!/usr/bin/env python

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from helpers.reviews_preprocessing import get_train_test_set as rp

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU

class RandomSearch(object):
    """docstring for RandomSearch"""
    def __init__(self, **kargs):
        super(RandomSearch, self).__init__()
        self.outfile = kargs['outfile']
        self.search_alg = kargs['search_alg']

    def __str__(self):
        return self.outfile

    def read_data(self):
        X_train, X_test, y_train, y_test = rp()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_error(self, data):
        return 'error'

if __name__ == '__main__':
    rhc = RandomSearch(**{'outfile': 'RHC'})
    print(rhc)
