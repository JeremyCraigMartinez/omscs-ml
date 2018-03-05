#!/usr/bin/env python
# python-2.7

import os
import sys
from array import array
from itertools import product
from threading import Thread

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CWD)
sys.path.append('{}/../ABAGAIL/bin'.format(CWD))

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
import shared.FixedIterationTrainer as FixedIterationTrainer

from helpers.fit import fit

N = 100
T = 49
numTrials = 5
fill = [2] * N
ranges = array('i', fill)
outfile_dir = '{}/../csv/PEAKS'.format(CWD)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)

def mimic():
    for t in range(numTrials):
        for samples, keep, m in product([100], [50], [0.1, 0.3, 0.5, 0.7, 0.9]):
            fname = '{}/MIMIC-{}-{}-{}-{}.tsv'.format(outfile_dir, samples, keep, m, t + 1)
            df = DiscreteDependencyTree(m, ranges)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            _mimic = MIMIC(samples, keep, pop)
            trainer = FixedIterationTrainer(_mimic, 10)
            fit(trainer, ef, _mimic, fname)

def ga():
    mf = DiscreteChangeOneMutation(ranges)
    cf = SingleCrossOver()
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    for t in range(numTrials):
        for pop, mate, mutate in product([100], [50, 30, 10], [50, 30, 10]):
            fname = '{}/GA-{}-{}-{}.tsv'.format(outfile_dir, pop, mate, mutate, t + 1)
            _ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
            trainer = FixedIterationTrainer(_ga, 10)
            fit(trainer, ef, _ga, fname)

nf = DiscreteChangeOneNeighbor(ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)

def rhc():
    for t in range(numTrials):
        fname = '{}/RHC-{}.tsv'.format(outfile_dir, t + 1)
        _rhc = RandomizedHillClimbing(hcp)
        trainer = FixedIterationTrainer(_rhc, 10)
        fit(trainer, ef, _rhc, fname)

def sa():
    for t in range(numTrials):
        for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
            fname = '{}/SA-{}-{}.tsv'.format(outfile_dir, CE, t + 1)
            _sa = SimulatedAnnealing(1E10, CE, hcp)
            trainer = FixedIterationTrainer(_sa, 10)
            fit(trainer, ef, _sa, fname)

Thread(target=mimic).start()
Thread(target=ga).start()
Thread(target=rhc).start()
Thread(target=sa).start()
