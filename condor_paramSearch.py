import numpy as np
import comb_loss, sys, math, os, pickle
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
    name = os.path.join("noise2d", name, "val")
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

def ndfactory(index, numprocess, name, risk, datagen, niterations):
    os.makedirs(name, exist_ok=True)
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
    
    
def sweepBinaryR(index, numprocess, niterations=3000, name="default"):
    name = os.path.join("binaryRisk", 'val', name)

    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # setup
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "binary_r")
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*(nrgroups+nirgroups)))
    theta = np.zeros((nrgroups + nirgroups) * pergroup)
    theta[:nrgroups*pergroup] = 1
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)    

    ndfactory(index, numprocess, name, risk, datagen, niterations)    

def sweepCov(index, numprocess, niterations=1000, name="more"):
    name = os.path.join("corr", 'val', name)

    nrgroups = 10
    nirgroups = 0
    pergroup = 10 # was 4 for default
    n = 2000
    # setup
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "corr")
    # gen data
    correlations = [i/nrgroups for i in range(nrgroups)]
    blocks = []
    for c in correlations:
        base = np.diag(np.ones(pergroup))        
        base[base==0] = c
        blocks.append(base)
    C = block_diag(*blocks)
    theta = np.ones(nrgroups*pergroup)
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=5)

    ndfactory(index, numprocess, name, risk, datagen, niterations)

def sweepFracR(index, numprocess, niterations=3000, name="default"):
    name = os.path.join("fracR", 'val', name)

    nrgroups = 12
    nirgroups = 0
    pergroup = 10
    n = 2000
    # setup
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "frac_r")
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*nrgroups))
    theta = np.ones(nrgroups*pergroup)
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)

    ndfactory(index, numprocess, name, risk, datagen, niterations)    
    
def sweepFracRN(index, numprocess, niterations=3000, name="default"):
    name = os.path.join("fracRN", 'val', name)

    nrgroups = 12
    nirgroups = 0
    pergroup = 10
    n = 2000
    # setup
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "frac_r")
    # normalize risk to 1
    for i in range(0, risk.size, pergroup):
        risk[i:i+pergroup] /= risk[i:i+pergroup].sum()
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*nrgroups))
    theta = np.ones(nrgroups*pergroup)
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)    
    
    ndfactory(index, numprocess, name, risk, datagen, niterations)

def diffTheta(index, numprocess, niterations=2000, name="default"):
    name = os.path.join("diffTheta", 'val', name)
    os.makedirs(name, exist_ok=True)

    n=5000
    def _datagen(CovM, theta, n=n):
        X = comb_loss.genCovX(C=CovM, n=n)
        y = comb_loss.sigmoid(X.dot(theta))
        for i in range(n):
            y[i] = np.random.binomial(1,y[i]) # bernoulli
        return X.astype(np.float32), y.astype(np.float32).reshape(y.size,1) 
    
    if os.path.exists(os.path.join(name, 'CovM.npy')):
        CovM = np.load(os.path.join(name, "CovM.npy"))        
        theta = np.load(os.path.join(name, "theta.npy"))
        risk = np.load(os.path.join(name, "risk.npy"))
        nd = np.load(os.path.join(name, "nd.npy"))
        datagen = lambda: _datagen(CovM, theta)

        ndfactory(index, numprocess, name, risk, datagen, niterations)
    else: # fix a datagen
        if index == 0:
            datagen, (theta, risk, nd, CovM) = comb_loss.genDiffTheta(n=n)
            # need to save theta, risk, nd
            np.save(os.path.join(name, "CovM.npy"), CovM)            
            np.save(os.path.join(name, "theta.npy"), theta)
            np.save(os.path.join(name, "risk.npy"), risk)
            np.save(os.path.join(name, "nd.npy"), nd)
        print("generated difftheta data")
    
def logExp(index, numprocess, niterations=2000, name="logExp"):
    name = os.path.join("fracR", 'val', name)

    nd = 100
    n = 2000
    
    x = np.linspace(0,1,nd)
    xs = [0, 0.4, 0.6, 1]
    ys = [0, 0.5, 0.5, 1]
    a,b,c,d = np.polyfit(xs, ys, 3)

    risk = a*x**3 + b*x**2 + c*x +d
    C = np.diag(np.ones(nd))
    C[C==0] = 0.99
    theta = np.ones(nd)
    
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)

    ndfactory(index, numprocess, name, risk, datagen, niterations)

def logFracR(index, numprocess, niterations=2000, name="logFracR"):
    name = os.path.join("fracR", 'val', name)

    nd = 100
    n = 2000

    nd = 100
    n = 2000
    
    x = np.linspace(0,1,nd)
    risk = np.log(30*x+1)
    C = np.diag(np.ones(nd))
    C[C==0] = 0.99
    theta = np.ones(nd)
    
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)

    ndfactory(index, numprocess, name, risk, datagen, niterations)
    
if __name__ == '__main__':
    pid = int(sys.argv[1])
    numprocess = int(sys.argv[2])

    # noise2d(pid, numprocess)
    # sweepBinaryR(pid, numprocess)
    # sweepCov(pid, numprocess)
    # sweepFracR(pid, numprocess)
    # sweepFracRN(pid, numprocess)
    # diffTheta(pid, numprocess)
    # logExp(pid, numprocess)
    logFracR(pid, numprocess)
