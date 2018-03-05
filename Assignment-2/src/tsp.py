#!/usr/bin/env python
# python-2.7

import sys
import os
from itertools import product
from array import array
from threading import Thread

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CWD)
sys.path.append('{}/../ABAGAIL/bin'.format(CWD))

import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import shared.FixedIterationTrainer as FixedIterationTrainer

from helpers.fit import fit

# set N value.  This is the number of points
N = 100
random = Random()
numTrials = 5

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i, _ in enumerate(points):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
outfile_dir = '{}/../csv/TSP'.format(CWD)

ef = TravelingSalesmanRouteEvaluationFunction(points)

#MIMIC
def mimic():
    fill = [N] * N
    ranges = array('i', fill)
    _odd = DiscreteUniformDistribution(ranges)
    for t in range(numTrials):
        for samples, keep, m in product([100], [50], [0.1, 0.3, 0.5, 0.7, 0.9]):
            fname = '{}/MIMIC-{}-{}-{}-{}'.format(outfile_dir, samples, keep, m, t + 1)
            df = DiscreteDependencyTree(m, ranges)
            pop = GenericProbabilisticOptimizationProblem(ef, _odd, df)
            _mimic = MIMIC(samples, keep, pop)
            trainer = FixedIterationTrainer(_mimic, 10)
            fit(trainer, ef, _mimic, fname)

odd = DiscretePermutationDistribution(N)

#GA
def ga():
    for t in range(numTrials):
        for pop, mate, mutate in product([100], [50, 30, 10], [50, 30, 10]):
            fname = '{}/GA-{}-{}-{}-{}'.format(outfile_dir, pop, mate, mutate, t + 1)
            cf = TravelingSalesmanCrossOver(ef)
            mf = SwapMutation()
            gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
            _ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
            trainer = FixedIterationTrainer(_ga, 10)
            fit(trainer, ef, _ga, fname)

nf = SwapNeighbor()
hcp = GenericHillClimbingProblem(ef, odd, nf)

# RHC
def rhc():
    for t in range(numTrials):
        fname = '{}/RHC-{}'.format(outfile_dir, t + 1)
        _rhc = RandomizedHillClimbing(hcp)
        trainer = FixedIterationTrainer(_rhc, 10)
        fit(trainer, ef, _rhc, fname)

# SA
def sa():
    for t in range(numTrials):
        for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
            fname = '{}/SA-{}-{}'.format(outfile_dir, CE, t + 1)
            _sa = SimulatedAnnealing(1E10, CE, hcp)
            trainer = FixedIterationTrainer(_sa, 10)
            fit(trainer, ef, _sa, fname)

Thread(target=mimic).start()
Thread(target=ga).start()
Thread(target=rhc).start()
Thread(target=sa).start()
