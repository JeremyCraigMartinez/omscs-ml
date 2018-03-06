#!/usr/bin/env python
# python-3.6

import pandas as pd
import numpy as np

def read_optimization_function_data(path_name):
    data = pd.read_csv(path_name, sep="\t")
    size = len(data.values)
    iteration = data.values[0:size, 0]
    error_train = data.values[0:size, 1]
    error_test = data.values[0:size, 2]
    accuracy_train = data.values[0:size, 3]
    accuracy_test = data.values[0:size, 4]
    elapsed = data.values[0:size, 5]
    return (
        iteration,
        error_train,
        error_test,
        accuracy_train,
        accuracy_test,
        elapsed,
    )

def read_fitness_function_data(path_name):
    data = pd.read_csv(path_name, sep=",")
    size = len(data.values)
    iterations = data.values[0:size, 0]
    fitness_score = data.values[0:size, 1]
    time = data.values[0:size, 2]
    return iterations, fitness_score, time

def average(func, path_name):
    vectors = []

    vectors.append(func(path_name.replace('NNN', str(1))))
    vectors.append(func(path_name.replace('NNN', str(2))))
    vectors.append(func(path_name.replace('NNN', str(3))))
    vectors.append(func(path_name.replace('NNN', str(4))))
    vectors.append(func(path_name.replace('NNN', str(5))))

    return np.mean(np.array(vectors), axis=0)
