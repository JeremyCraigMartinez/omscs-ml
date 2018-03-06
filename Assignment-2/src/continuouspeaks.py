#!/usr/bin/env python
# python-2.7

import os
import sys
from array import array

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CWD)
sys.path.append('{}/../ABAGAIL/bin'.format(CWD))

import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution

from helpers.fitness import mimic, ga, rhc, sa

N = 100
T = 49
fill = [2] * N
ranges = array('i', fill)

outfile_dir = '{}/../csv/PEAKS'.format(CWD)

ef = ContinuousPeaksEvaluationFunction(T)
cf = SingleCrossOver()
mf = DiscreteChangeOneMutation(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
odd = DiscreteUniformDistribution(ranges)

mimic(ef, odd, outfile_dir, ranges)
ga(ef, odd, outfile_dir, cf, mf)
rhc(ef, odd, outfile_dir, nf)
sa(ef, odd, outfile_dir, nf)
