import os
import sys
from functools import partial
from threading import Thread

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

    if sys.argv[1] == '0' or sys.argv[1] is None:
        # Back Propogation
        bp = RandomSearch(**{'outfile': 'BP'})
        bp.run()

    if sys.argv[1] == '1' or sys.argv[1] is None:
        # Genetic Algorithm
        alg = partial(StandardGeneticAlgorithm, 50, 20, 20)
        ga = RandomSearch(**{'outfile': 'GA', 'search_alg': alg})
        ga.run()

    if sys.argv[1] == '2' or sys.argv[1] is None:
        # Randomized Hill Climbing
        rhc = RandomSearch(**{'outfile': 'RHC', 'search_alg': RandomizedHillClimbing})
        rhc.run()

    if sys.argv[1] == '3' or sys.argv[1] is None:
        alg = partial(SimulatedAnnealing, 1E10, 0.15)
        sa = RandomSearch(**{'outfile': 'SA', 'search_alg': alg})
        sa.run()
