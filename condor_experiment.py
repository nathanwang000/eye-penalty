from experiment import experiment, REG2FUNC
import numpy as np
import comb_loss, sys, os, condor_paramSearch
from scipy.linalg import block_diag
from itertools import product
import math

############# helper functions
from chainer import serializers, functions
from scipy import stats
from collections import defaultdict    

def loadModel(index, basedir, atoms):
    reg, args = atoms[index]
    model, optimizer = comb_loss.getModelOptimizer(REG2FUNC[reg](*args))
    modelpath = os.path.join(basedir, "%d.model" % index)
    if os.path.isfile(modelpath):
        serializers.load_npz(modelpath, model)
        theta = functions.flatten(model.predictor.l1.W)
        sparsity = comb_loss.abs_gini(theta.data)
    return (reg, model, sparsity) # if not exist, use the null model

def evalModel(model, X, y):
    yhat = model.predictor(X)
    # eval on auroc
    auroc = comb_loss.calcAuroc(yhat, y)
    return auroc

# no stats difference in auroc with pvalue=0.05
def siftModels(models, datagen, alpha=0.05, nruns=30):
    def perturb():
        return 1e-6 * np.random.randn(nruns)

    def sift(models, performance, criteria="t", alpha=alpha):

        keep = np.ones(len(models))
        if criteria == 't':
            # the t test is actually transitive, so just linearly find the max
            j = 0 # best index
            for m in models:
                i = m[0]
                t, p = stats.ttest_rel(performance[:,i] + perturb(),
                                       performance[:,j] + perturb())
                if t > 0: j = i
            for local_i, m in enumerate(models):
                i = m[0]
                t, p = stats.ttest_rel(performance[:,i] + perturb(),
                                       performance[:,j] + perturb())
                if p < alpha: keep[local_i] = 0
        else:
            # linearly interpolate
            alpha = 0.9
            m = np.min([m[-1] for m in models])
            M = np.max([m[-1] for m in models])
            threshold = m + (M-m)*alpha
            for i in range(len(models)):
                keep[i] = 0 if models[i][-1] < threshold else 1
        return [models[i] for i in range(len(models)) if keep[i]]     
        
    performance = []
    sparsity = []
    for i in range(nruns):
        X, y = datagen()
        ps = []
        for reg, m, s in models:
            p = evalModel(m, X, y)
            ps.append(p)
        performance.append(ps)
    performance = np.array(performance)
    models = [(i, *m, performance[:,i].mean()) for i,m in enumerate(models)]

    for m in models: print(m)
    
    # choose the following models based on sparsity
    modelDict = defaultdict(list)
    for m in models:
        modelDict[m[1]].append(m)

    cand = []
    for m in modelDict:
        cand.extend(sift(modelDict[m], performance, criteria='t'))

    for c in cand: print(c)
    
    d = defaultdict(lambda: (0,-1, 0, 0))
    for i, r, m, s, p in cand:
        if d[r][1] < s: d[r] = (m, s, i, p)
    return [(r, p, s, i) for r,(m,s,i,p) in d.items()]

#############
def noise2d(index, regs=None, niterations=5000, signoise=0, alpha=0.01, name="default"):
    taskname = "noise2d"
    valdir = os.path.join(taskname, 'val', name)
    name = os.path.join(taskname, name, str(index))
    
    os.makedirs(name, exist_ok=True)    
    datagen = lambda: comb_loss.gendata(signoise=signoise, n=100)
    risk = comb_loss.generate_risk(0, 0, 0, '2d')

    atoms = condor_paramSearch.paramAtoms2d()
    models = siftModels([loadModel(i, valdir, atoms) for i in range(len(atoms))],
                        datagen=datagen)
    paramAtoms = [atoms[m[-1]] for m in models]

    if regs: paramAtoms = list(filter(lambda a: a[0] in regs, paramAtoms))
    experiment(paramAtoms,
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               printreport=False, 
               resume=True,
               niterations=niterations)

def ndfactory(index, taskname, name, datagen, risk, niterations):
    valdir = os.path.join(taskname, 'val', name)
    name = os.path.join(taskname, name, str(index))
    os.makedirs(name, exist_ok=True)

    atoms = condor_paramSearch.paramAtomsNd(risk)
    models = siftModels([loadModel(i, valdir, atoms) for i in range(len(atoms))],
                        datagen=datagen)
    paramAtoms = [atoms[m[-1]] for m in models]

    print (paramAtoms)
    experiment(paramAtoms,
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               printreport=False, 
               resume=True,
               niterations=niterations)

def sweepBinaryR(index, niterations=3000, name="default"):
    taskname = "binaryRisk"

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
                                           n=n, signoise=10)    

    ndfactory(index, taskname, name, datagen, risk, niterations)

def sweepCov(index, niterations=1000, name="huge"):
    taskname = "corr"

    nrgroups = 10
    nirgroups = 0
    pergroup = 30 # was 4 for default, 10 for more, 30 for huge
    n = 5000
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

    ndfactory(index, taskname, name, datagen, risk, niterations)

def sweepFracR(index, niterations=3000, name="default"):
    taskname = "fracR"

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

    ndfactory(index, taskname, name, datagen, risk, niterations)

##### special experiments ###########################################
def diffTheta(index, niterations=1000, name="default", binary=True):
    taskname = "diffTheta"
    
    name = os.path.join(taskname, name)
    valdir = os.path.join(name, 'val')
    name = os.path.join(name, str(index))    
    
    os.makedirs(name, exist_ok=True)
    datagen, (theta, risk, nd, CovM) = comb_loss.genDiffTheta(n=5000, binary=binary)
    
    # need to save theta, risk, nd
    np.save(os.path.join(name, "theta.npy"), theta)
    np.save(os.path.join(name, "risk.npy"), risk)
    np.save(os.path.join(name, "nd.npy"), nd)        

    w1 = 2-risk
    owl1 = np.zeros(risk.size)
    owl1[0] = 1
    paramAtoms = [
        ('eye', (risk, 0.01)),
        ('wlasso', (w1, 0.01)),
        ('wridge', (w1, 0.01)),
        # ('penalty', (risk, 0.01, 0.4)),
        ('owl', (owl1, 0.01)),
        ('lasso', (0.01,)),
        ('enet', (0.01, 0.2))
    ]
    experiment(paramAtoms,
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               printreport=False,
               resume=True,
               niterations=niterations)

def InferiorPenalty(index, numprocess, niterations=10000, name="default"):
    taskname = "inferiorPenalty"

    name = os.path.join(taskname, name)
    name = os.path.join(name, str(index))    

    index = index-1 # because condor_experiment is 1 indexed
    os.makedirs(name, exist_ok=True)
    datagen = lambda: comb_loss.gendata(signoise=0, n=100)
    
    risk  = comb_loss.generate_risk(0, 0, 0, '2d')
    alpha = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    #alpha = [5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9, 1e-9, 5e-10, 1e-10, 5e-11, 1e-11, 5e-12, 1e-12]
    beta  = np.linspace(0,1,11)

    params_cand = [
        ('penalty', ((risk,), alpha, beta)),
        ('eye',     ((risk,), alpha))
    ]

    paramAtoms = [(m, arg) for m, args in params_cand for arg in product(*args)]
    mytask = math.ceil(len(paramAtoms) / numprocess)
    experiment(paramAtoms[index*mytask: (index+1)*mytask],
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               namebases=list(range(index*mytask, (index+1)*mytask)),               
               printreport=False,
               resume=True,
               niterations=niterations)

def unknownRelevant(index, numprocess, niterations=10000, name="unknownRelevant"):
    taskname = "inferiorPenalty"

    name = os.path.join(taskname, name)
    name = os.path.join(name, str(index))    

    index = index-1 # because condor_experiment is 1 indexed
    os.makedirs(name, exist_ok=True)

    risk  = np.array([1,0])
    theta = np.array([1,5])
    datagen = lambda: comb_loss.genCovData(C=np.eye(2), theta=theta, signoise=1, n=300)
    
    alpha = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    beta  = np.linspace(0,1,11)

    params_cand = [
        ('penalty', ((risk,), alpha, beta)),
        ('eye',     ((risk,), alpha))
    ]

    paramAtoms = [(m, arg) for m, args in params_cand for arg in product(*args)]
    mytask = math.ceil(len(paramAtoms) / numprocess)
    experiment(paramAtoms[index*mytask: (index+1)*mytask],
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               namebases=list(range(index*mytask, (index+1)*mytask)),               
               printreport=False,
               resume=True,
               niterations=niterations)

######################### derived experiments ##############################
def sweepFracRN(index, niterations=3000, name="default"):
    taskname = "fracRN"

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

    ndfactory(index, taskname, name, datagen, risk, niterations)    


def logExp(index, niterations=2000, name="logExp"): # fracR variant
    taskname = "fracR"

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

    ndfactory(index, taskname, name, datagen, risk, niterations)

def logFracR(index, niterations=2000, name="logFracR"): # fracR variant
    taskname = "fracR"

    nd = 100
    n = 2000
    
    x = np.linspace(0,1,nd)
    risk = np.log(30*x+1)
    risk /= risk.max()
    C = np.diag(np.ones(nd))
    C[C==0] = 0.99
    theta = np.ones(nd)
    
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)

    ndfactory(index, taskname, name, datagen, risk, niterations)
    

if __name__ == '__main__':
    pid = int(sys.argv[1])
    numprocess = int(sys.argv[2])
    
    # sweep noise level
    # for s in np.linspace(0,2,10):
    #     noise2d(sys.argv[1], signoise=s, name="noise%.2f" % s)

    # noise2d(sys.argv[1], alpha=1e-3, niterations=15000, name="alpha0.001")
    # noise2d(sys.argv[1], alpha=1e-4, niterations=30000, name="alpha0.0001")    

    # noise2d(pid)
    # sweepBinaryR(pid)
    # sweepCov(pid)
    # sweepFracR(pid)
    # sweepFracRN(pid)
    # diffTheta(pid)
    # logExp(pid)
    # logFracR(pid)
    # InferiorPenalty(pid, numprocess) 
    unknownRelevant(pid, numprocess) 
    
