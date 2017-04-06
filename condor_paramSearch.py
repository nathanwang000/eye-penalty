import numpy as np
import comb_loss, sys, math, os
from scipy.linalg import block_diag
from experiment import experiment
from itertools import product

# choose parameters using bootstrap from test set
def paramAtoms2d():
    risk  = comb_loss.generate_risk(0, 0, 0, '2d')
    w0    = 1-risk
    w1    = 2-risk
    alpha = (0.1, 0.01, 0.001, 0.0001)
    beta  = tuple(i/10 for i in range(11))

    params_cand = [
        ('lasso',   (alpha,)),
        ('ridge',   (alpha,)),
        ('enet',    (alpha, beta)),
        ('penalty', ((risk,), alpha, beta)),
        ('wlasso',  ((w1,), alpha)),
        ('wridge',  ((w1,), alpha)),
        ('owl',     (([2,1], [1,1], [1,0]), alpha)),
        ('eye',     ((risk,), alpha))
    ]

    return [(m, arg) for m, args in params_cand for arg in product(*args)]

def paramAtomsNd(risk):
    w1 = 2-risk
    owl1 = np.zeros(risk.size)  # inf norm
    owl1[0] = 1
    alpha = (5e-2, 1e-2, 5e-3, 1e-3, 5e-4)
    beta = tuple(i/10 for i in range(0,11,3))

    params_cand = [
        ('lasso',   (alpha,)),
        # ('ridge',   (alpha,)),
        ('enet',    (alpha, beta)),
        ('penalty', ((risk,), alpha, beta)),
        ('wlasso',  ((w1,), alpha)),
        ('wridge',  ((w1,), alpha)),
        ('owl',     ((owl1, ), alpha)),
        ('eye',     ((risk,), alpha))
    ]

    return [(m, arg) for m, args in params_cand for arg in product(*args)]

# user functions
def noise2d(index, numprocess, niterations=5000,
            signoise=0, name="default"):
    name = os.path.join("noise2dCV", name)
    os.makedirs(name, exist_ok=True)    
    datagen = lambda: comb_loss.gendata(signoise=signoise, n=100)

    paramAtoms = paramAtoms2d()
    mytask = math.ceil(len(paramAtoms) / numprocess)
    experiment(paramAtoms[index*mytask: (index+1)*mytask],
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               namebases=list(range(index*mytask, (index+1)*mytask)),
               printreport=False, 
               resume=True,
               niterations=niterations,
               validate=True)

def sweepBinaryR(index, numprocess, niterations=3000, name="default"):
    name = os.path.join("binaryRiskCV", name)
    os.makedirs(name, exist_ok=True)    

    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # setup
    gridSearch = comb_loss.paramsSearchMd
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "binary_r")
    basedir = os.path.join(name, 'val')
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*(nrgroups+nirgroups)))
    theta = np.zeros((nrgroups + nirgroups) * pergroup)
    theta[:nrgroups*pergroup] = 1
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=5)    

    # pipeline for training
    paramAtoms = paramAtomsNd(risk)    
    mytask = math.ceil(len(paramAtoms) / numprocess)
    experiment(paramAtoms[index*mytask: (index+1)*mytask],
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               namebases=list(range(index*mytask, (index+1)*mytask)),
               printreport=False, 
               resume=True,
               niterations=niterations,
               validate=True)

if __name__ == '__main__':
    pid = int(sys.argv[1])
    numprocess = int(sys.argv[2])

    # noise2d(pid, numprocess)
    sweepBinaryR(pid, numprocess)

###################################################################### todo
def diffTheta(index, numprocess,
              niterations=1500, name="default"):
    name = os.path.join("diffThetaCV", name)
    os.makedirs(name, exist_ok=True)    

    # todo fix a datagen
    datagen, (theta, risk, nd) = comb_loss.genDiffTheta(n=5000)
    
    # need to save theta, risk, nd
    np.save(os.path.join(name, "theta.npy"), theta)
    np.save(os.path.join(name, "risk.npy"), risk)
    np.save(os.path.join(name, "nd.npy"), nd)        


    mytask = math.ceil(len(paramAtoms) / numprocess)
    paramAtoms = paramAtomsNd(risk)
    experiment(paramAtoms[index*mytask: (index+1)*mytask],
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               namebases=list(range(index*mytask, (index+1)*mytask)),
               printreport=False, 
               resume=True,
               niterations=niterations,
               validate=True)

