import sys
import numpy as np
from numpy import array
from math import sqrt
import matplotlib.pyplot as plt
from chainer import cuda, Function, gradient_check, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList, training, utils, report
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import os
from itertools import product
import hashlib, pickle
import itertools
from mini_eye import eye, l1norm, l2normsq, run_with_reg

## other regularizations
def penalty(r, alpha=1, l1_ratio=0.5):
    def f_(theta):
        return 2 * alpha * (l1_ratio * l1norm((1-r)*theta) +
                            0.5 * (1-l1_ratio) * l2normsq(r*theta))
    return f_

def weightedLasso(w, alpha=1): # w is the weight vector
    def f_(theta):
        return alpha * l1norm(w * theta)
    return f_

def weightedRidge(w, alpha=1): # w is the weight vector
    def f_(theta):
        return 0.5 * alpha * l2normsq(w * theta)
    return f_

def OWL(w, alpha=1):
    w = np.sort(np.abs(w))
    def f_(theta):
        order = np.argsort(abs(theta).data).astype(np.int32)
        return alpha * F.sum(F.permutate(abs(theta), order) * w)
    return f_

def lasso(alpha=1):
    def f_(theta):
        return alpha * l1norm(theta)
    return f_

def ridge(alpha=1):
    def f_(theta):
        return 0.5 * alpha * l2normsq(theta)
    return f_

def enet(alpha=1, l1_ratio=0.5):
    def f_(theta):
        return alpha * (lasso(l1_ratio)(theta) +
                        0.5 * ridge(1-l1_ratio)(theta))
    return f_

## parameter search
def paramAtomsNd(risk):
    w1 = 2-risk
    owl1 = np.zeros(risk.size)  # inf norm
    owl1[0] = 1
    alpha = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
    beta = tuple(i/10 for i in range(0,11,3))

    params_cand = [
        ('lasso',   (alpha,)),
        ('enet',    (alpha, beta)),
        ('penalty', ((risk,), alpha, beta)),
        ('wlasso',  ((w1,), alpha)),
        ('wridge',  ((w1,), alpha)),
        ('owl',     ((owl1, ), alpha)),
        ('eye',     ((risk,), alpha))
    ]

    return [(m, arg) for m, args in params_cand for arg in product(*args)]

# map regularization name to function
REG2FUNC = {'lasso':   lasso, 
            'ridge':   ridge, 
            'enet':    enet, 
            'penalty': penalty,        
            'wlasso':  weightedLasso, 
            'wridge':  weightedRidge, 
            'owl':     OWL, 
            'eye':     eye}

def experiment(paramAtoms, datagen, num_runs=100,
               basedir_prefix="", niterations=1000,
               printreport=False, resume=True,
               namebases=None, validate=False):

    if not namebases or len(namebases) < len(paramAtoms):
        namebases = [None] * len(paramAtoms)
    
    def run_with_reg_wrapper(num_runs):
        def _f(*args,**kwargs):
            return comb_loss.run_with_reg(*args, **kwargs,
                                          num_runs=num_runs,
                                          printreport=printreport,
                                          resume=resume,
                                          datagen=datagen,
                                          niterations=niterations)
        return _f
    run = run_with_reg_wrapper(num_runs)

    # actual run
    for i, (method, args) in enumerate(paramAtoms):
        if not validate:
            outdir = os.path.join(basedir_prefix, "result_" + method)
            run(REG2FUNC[method](*args), outdir, namebase=namebases[i])            
        else: # validation just doesn't save theta to avoid race condition
            outdir = basedir_prefix
            run(REG2FUNC[method](*args), outdir, namebase=namebases[i])

# 130 parameters to search through
# todo: to be filled in with the real risk factors
# todo: put the run in eye_penalty_for_sparse_data in a if __name__ == '__main__'
# todo one job runs 1 data point
from eye_penalty_for_sparse_data import risk, datareturn

if __name__ == '__main__':
    kwargs = sys.argv[1:]
    for i in kwargs:
        print(i)
        exec(i)
        # theindex[0-14]

    start = 0 # dummy here
    end = 10 # dummy here
    atoms = paramAtomsNd(risk)[start:end]

    experiment(atoms, datagen=datareturn(), num_runs=1,
               basedir_prefix="cdiff",
               namebases=list(range(start, end)),
               printreport=False,
               resume=True,
               niterations=30,
               validate=True)



