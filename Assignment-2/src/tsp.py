#!/usr/bin/env python
# python-2.7

import sys
import os
from array import array

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CWD)
sys.path.append('{}/../ABAGAIL/bin'.format(CWD))

import java.util.Random as Random

import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution

from helpers.fitness import mimic, ga, rhc, sa

# set N value.  This is the number of points
N = 100
random = Random()
fill = [2] * N
ranges = array('i', fill)

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i, _ in enumerate(points):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
outfile_dir = '{}/../csv/TSP'.format(CWD)

ef = TravelingSalesmanRouteEvaluationFunction(points)
cf = TravelingSalesmanCrossOver(ef)
mf = SwapMutation()
nf = SwapNeighbor()
odd = DiscreteUniformDistribution(ranges)

mimic(ef, odd, outfile_dir, ranges)
ga(ef, odd, outfile_dir, cf, mf)
rhc(ef, odd, outfile_dir, nf)
sa(ef, odd, outfile_dir, nf)
