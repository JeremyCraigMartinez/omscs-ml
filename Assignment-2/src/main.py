#!/usr/bin/env python
# python-2.7

import os
import sys
from functools import partial

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CWD)
sys.path.append('{}/../ABAGAIL/bin'.format(CWD))

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

from RandomSearch import RandomSearch

if __name__ == '__main__':
    '''
    check cli arg for parallelized script, if not there, run all synchronously
    '''
    argv = sys.argv[1].split(',')

    if argv[0] == '0' or argv is None:
        # Back Propogation
        bp = RandomSearch(**{'outfile': 'BP/BP'})
        bp.run()

    if argv[0] == '1' or argv is None:
        # Genetic Algorithm
        pop = int(argv[1]) # [40, 50]
        mate = int(argv[2]) # [20, 10]
        mutate = int(argv[3]) # [20, 10]
        alg = partial(StandardGeneticAlgorithm, pop, mate, mutate)
        ga = RandomSearch(**{'outfile': 'GA/GA-{}-{}-{}'.format(pop, mate, mutate), 'search_alg': alg})
        ga.run()

    if argv[0] == '2' or argv is None:
        # Randomized Hill Climbing
        rhc = RandomSearch(**{'outfile': 'RHC/RHC', 'search_alg': RandomizedHillClimbing})
        rhc.run()

    if argv[0] == '3' or argv is None:
        T = float(argv[1]) # [0.1,0.3,0.5,0.7,0.9]
        alg = partial(SimulatedAnnealing, 1E10, T)
        sa = RandomSearch(**{'outfile': 'SA/SA-{}'.format(T), 'search_alg': alg})
        sa.run()
