#!/usr/bin/env python
# python-2.7

import os
import sys
from array import array
from itertools import product

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

# RHC
for t in range(numTrials):
    fname = '{}/../csv/RHC-{}'.format(CWD, t + 1)
    ef = ContinuousPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)
    trainer = FixedIterationTrainer(rhc, 10)
    fit(trainer, ef, rhc, fname)

# SA
for t in range(numTrials):
    for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
        fname = '{}/../csv/SA-{}-{}'.format(CWD, CE, t + 1)
        ef = ContinuousPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(1E10, CE, hcp)
        trainer = FixedIterationTrainer(sa, 10)
        fit(trainer, ef, sa, fname)

#GA
for t in range(numTrials):
    for pop, mate, mutate in product([100], [50, 30, 10], [50, 30, 10]):
        fname = '{}/../csv/GA-{}-{}-{}'.format(CWD, pop, mate, mutate, t + 1)
        ef = ContinuousPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = FixedIterationTrainer(ga, 10)
        fit(trainer, ef, ga, fname)

#MIMIC
for t in range(numTrials):
    for samples, keep, m in product([100], [50], [0.1, 0.3, 0.5, 0.7, 0.9]):
        fname = '{}/../csv/MIMIC-{}-{}-{}-{}'.format(CWD, samples, keep, m, t + 1)
        ef = ContinuousPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        df = DiscreteDependencyTree(m, ranges)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = FixedIterationTrainer(mimic, 10)
        fit(trainer, ef, mimic, fname)
