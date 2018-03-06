#!/usr/bin/env python
# python-2.7

import sys
import os

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CWD)
sys.path.append('{}/../ABAGAIL/bin'.format(CWD))

import java.util.Random as Random

import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation

from helpers.fit import fit
from helpers.fitness import mimic, ga, rhc, sa

# set N value.  This is the number of points
N = 100
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i, _ in enumerate(points):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
outfile_dir = '{}/../csv/TSP'.format(CWD)

ef = TravelingSalesmanRouteEvaluationFunction(points)
cf = TravelingSalesmanCrossOver(ef)
mf = SwapMutation
nf = SwapNeighbor

mimic(ef, outfile_dir)
ga(ef, outfile_dir, cf, mf)
rhc(ef, outfile_dir, nf)
sa(ef, outfile_dir, nf)
