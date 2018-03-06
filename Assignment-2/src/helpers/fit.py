#!/usr/bin/env python
# python-2.7

import time
from time import clock

maxIters = 4001

def fit(trainer, fitness, alg, fname):
    with open(fname, 'w') as f:
        f.write('iterations,fitness,time\n')

    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        trainer.train()
        elapsed = time.clock()-start
        times.append(times[-1]+elapsed)
        print fitness
        score = fitness.value(alg.getOptimal())
        st = '{},{},{}\n'.format(i, score, times[-1])
        print st
        with open(fname, 'a') as f:
            f.write(st)
