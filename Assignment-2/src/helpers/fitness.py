#!/usr/bin/env python
# python-2.7

import sys
import os
from itertools import product
from array import array
from threading import Thread
from functools import partial

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CWD)
sys.path.append('{}/../../ABAGAIL/bin'.format(CWD))

import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer

from helpers.fit import fit

# set N value.  This is the number of points
N = 100
random = Random()
numTrials = 5
fill = [N] * N
ranges = array('i', fill)

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i, _ in enumerate(points):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

odd = DiscreteUniformDistribution(ranges)

def mimic(ef, outfile_dir):
    for t in range(numTrials):
        # population of 55 did better so we'll only run that one here
        for samples, keep, m in product([55], [20], [0.5, 0.6, 0.7, 0.8, 0.9]):
            fname = '{}/MIMIC-{}-{}-{}-{}'.format(outfile_dir, samples, keep, m, t + 1)
            df = DiscreteDependencyTree(m, ranges)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            _mimic = MIMIC(samples, keep, pop)
            trainer = FixedIterationTrainer(_mimic, 10)
            partialfit = partial(fit, trainer, ef, _mimic, fname)
            Thread(target=partialfit).start()

def ga(ef, outfile_dir, _cf, _mf):
    cf = _cf()
    mf = _mf()
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    for t in range(numTrials):
        # population of 55 did better so we'll only run that one here
        for pop, mate, mutate in product([55], [10, 20], [10, 20]):
            fname = '{}/GA-{}-{}-{}-{}'.format(outfile_dir, pop, mate, mutate, t + 1)
            _ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
            trainer = FixedIterationTrainer(_ga, 10)
            partialfit = partial(fit, trainer, ef, _ga, fname)
            Thread(target=partialfit).start()

def rhc(ef, outfile_dir):
    nf = _nf()
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    for t in range(numTrials):
        fname = '{}/RHC-{}'.format(outfile_dir, t + 1)
        _rhc = RandomizedHillClimbing(hcp)
        trainer = FixedIterationTrainer(_rhc, 10)
        partialfit = partial(fit, trainer, ef, _rhc, fname)
        Thread(target=partialfit).start()

def sa(ef, outfile_dir, _nf):
    nf = _nf()
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    for t in range(numTrials):
        for CE in [0.5, 0.6, 0.7, 0.8, 0.9]:
            fname = '{}/SA-{}-{}'.format(outfile_dir, CE, t + 1)
            _sa = SimulatedAnnealing(1E10, CE, hcp)
            trainer = FixedIterationTrainer(_sa, 10)
            partialfit = partial(fit, trainer, ef, _sa, fname)
            Thread(target=partialfit).start()
