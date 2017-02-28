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
from chainer.datasets import TupleDataset
import os
from itertools import product
from convergeTrigger import ConvergeTrigger
from sklearn.metrics import roc_auc_score, roc_curve
import hashlib, pickle
import itertools
from pyemd import emd_with_flow
from scipy.linalg import block_diag

# todo
# 1. look for all unknown but different weight (balanced data)
# 2. run 2 more experiments

############ utility functions ##################
def l1norm(v):
    return F.sum(abs(v))

def l2normsq(v):
    return F.sum(v**2)

def normalizer(mu=2, std=2):
    currsum = 0
    n = 0
    currx2 = 0

    def transform(X):
        mean = currsum / n
        var  = currx2 / n - mean**2
        return (X-mean) / np.sqrt(var) * std + mu

    def normalize(X, train=True, refit=True):
        nonlocal n, currsum, currx2        
        # X is design matrix
        if not train:
            assert n!=0, "not fitted yet"
            return transform(X)
        if refit:
            n = 0
            currsum = 0
            currx2 = 0
        currsum += np.sum(X, 0)
        currx2  += np.sum(X**2, 0)
        n += X.shape[0]
        return transform(X)
    return normalize

############ penalties ##########################
def eye(r, alpha=1):
    # l1_ratio change will just stretch the unit norm ball
    # so it can be fixed
    l1_ratio = 0.5 
    if l1_ratio == 0 or l1_ratio == 1:
        return penalty(r, alpha, l1_ratio)
    
    def solveQuadratic(a, b, c):
        return (-b + F.sqrt(b**2-4*a*c)) / (2*a)

    # want to force f_(theta/t) = c for which c
    # 45 degree, which is slope of -1, where f_ is penalty
    # solve for t, just a quadratic equation!
    # and in fact t always real because c > 0
    def f_(theta):
        # a (1/t)**2 + b (1/t) = c
        a = (1-l1_ratio) * l2normsq(r*theta)        
        b = 2 * l1_ratio * l1norm((1-r)*theta)
        c = l1_ratio**2 / (1-l1_ratio)
        # degenerative case: b (1/t) = c
        if a.data == 0: return alpha * b / c
        # general case
        return alpha * 1.0 / solveQuadratic(a, b, -c)
    return f_

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

#############synthetic data##################
# fake 2d data perfectly correlated
niterations = 1000
n = 100
d = 2
group_theta = np.ones(1000)[:,None]

def genCovData(C=None, theta=None, n=n, dim=d):
    # C is the covariance matrice (assume to be psd)
    # y = Bernoulli(\theta x), z~Norm(0,I), x = Az
    if C is None:
        C = np.diag(np.ones(dim))
    A = np.linalg.cholesky(C)
    d, nc = A.shape
    if theta is None:
        theta = np.ones(d)            
    assert d==theta.size, "size mismatch"
    Z = np.random.randn(n, d)
    X = Z.dot(A.T) # linearly separable
    y = X.dot(theta) > 0
    return X.astype(np.float32), y.astype(np.float32)

def setupCovCorr():
    # sweep correlation from 0 to 0.9, 1 case is shown in 2d
    # fix r and theta, all relevant b/c irrelvant case
    # is explored in the previous round
    # evaluate by correlation weighted kl/emd metric
    nrgroups = 10
    pergroup = 4
    rbase = np.zeros(pergroup)
    rbase[:pergroup//2] = 1
    r = np.concatenate([rbase for _ in range(nrgroups)])
    correlations = [i/nrgroups for i in range(nrgroups)]

    blocks = []
    for c in correlations:
        base = np.diag(np.ones(pergroup))        
        base[base==0] = c
        blocks.append(base)

    C = block_diag(*blocks)
    theta = np.ones(nrgroups*pergroup)
    X, y = genCovData(C=C, theta=theta, n=2000)

def setupCovR():
    # sweep fractional r: fix correlation to 0.95, theta to 1
    # todo
    c = 0.99
    # consider sigmoid, log, exp
    def sigmoid_(a, x):
        return 1/(1+e**(-a*x))
    def exp_(a, x):
        return np.exp(a*x) / np.max(np.exp(a*x))
    def log_(a, x):
        return np.log(a*x+1) / np.max(a*x+1)
    pergroup = 10
    alphas = [0.1,0.2,1,2,10]
    C = block_diag()
    theta = np.ones(nrgroups*pergroup)
    X, y = genCovData(C=C, theta=theta, n=2000)
    
    
def setupCovTheta():
    # sweep theta in relevant all unknown to see sparseness
    # todo
    pass
    
def genPartitionData(n=n, nrgroups=11, nirgroups=11, pergroup=10,\
                     signoise=0.2, scaleh=2, munoise=0,\
                     left=-3, right=1,
                     plot=False, name=None):
    mu = (left + right) / 2
    # nrgroups: relevant groups
    # nirgroups: irrelevant groups
    ngroups = nrgroups + nirgroups
    d = ngroups * pergroup
    H = np.random.random((n,ngroups)) * abs(right-left) + left
    # y = sum(\theta_i * h_i) for h_i being relevant group
    g_theta = group_theta[:nrgroups]
    y = H[:,:nrgroups].dot(g_theta)/sum(abs(g_theta)) >= mu
    X = np.repeat(H, pergroup, axis=1)
    
    noise = np.random.randn(n,d)  * signoise + munoise
    alpha = np.diag(np.random.randint(1,scaleh,size=d))
    X = X.dot(alpha) + noise

    if plot and d == 2:
        plt.scatter(X[:,0],X[:,1],c=y.astype(np.int64),alpha=0.5)
        plt.title("dataset: $X_i$=$\\alpha_i$h+N(0,$\sigma$)")
        name = name or 'data'
        plt.savefig('figures/'+name+'Partition.png', bbox_inches='tight')
        plt.show()        
    return X.astype(np.float32), y.astype(np.float32)

def gendata(plot=False, name=None, d=d,
            signoise=0.2, scaleh=2, munoise=0,
            left=-2.5, right=1.5):
    mu = (left + right) / 2    
    h = np.linspace(left,right,n).reshape(n,1)
    y = h > mu
    X = np.repeat(h, d, 1)
    noise = np.random.randn(n,d)  * signoise + munoise
    alpha = np.diag(np.random.randint(1,scaleh,size=d))
    X = X.dot(alpha) + noise

    if plot and d == 2:
        plt.scatter(X[:,0],X[:,1],c=y.astype(np.int64),alpha=0.5)
        plt.axvline(x=mu*alpha[0,0], color='k', linestyle='--')
        plt.axhline(y=mu*alpha[1,1], color='k', linestyle='--')        
        plt.title("dataset: $X_i$=$\\alpha_i$h+N(0,$\sigma$)")
        name = name or 'data'
        plt.savefig('figures/'+name+'.png', bbox_inches='tight')
        plt.show()        
    return X.astype(np.float32), y.astype(np.float32)



#############the model#########################
class Predictor(Chain):
    def __init__(self, n_out):
        super(Predictor, self).__init__(
            l1 = L.Linear(None, n_out) # n_in -> n_out
        )
    def __call__(self, x):
        y = self.l1(x)
        F.sigmoid(y)
        return F.sigmoid(y)

    
def lossfun(y, t):
    def linscale(x, xmin, xmax):
        # linear scaling of x \in [0,1] to xmin, xmax
        y = x * (xmax-xmin) + xmin
        return y
    
    # max likelihood
    smooth = 1e-6
    # yhat = F.clip(y, smooth, 1-smooth)
    yhat = linscale(y, smooth, 1-smooth)
    logloss = F.sum(F.where(t.data > 0,
                            -F.log(yhat),
                            -F.log(1-yhat))) / len(t)
    return logloss

class Regresser(Chain):
    def __init__(self, predictor, lossfun, regularizer=None):
        super(Regresser, self).__init__(
            predictor=predictor
        )
        self.lossfun = lossfun
        self.regularizer = regularizer
        
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = self.lossfun(y, t)
        regloss = 0
        sparsity = 0 # not really working as most are not exact 0
        if self.regularizer:
            W = self.predictor.l1.W
            b = self.predictor.l1.b
            # theta = F.concat((F.flatten(W), b), 0)
            theta = F.flatten(W) # don't regularize b
            regloss = self.regularizer(theta)
            sparsity = np.isclose(0,theta.data).sum() /\
                       theta.data.size
            
        acc = (n-np.sum(abs((y.data > 0.5) - t.data))) / n
        try: # y may be nan if predictor.W is nan
            y_true = t.data if isinstance(t, Variable) else t 
            auroc = roc_auc_score(y_true, y.data)
            fpr, tpr, threshold = roc_curve(y_true, y.data)
        except:
            auroc = -1
        report({'loss': loss,
                'penalty': regloss,
                'accuracy': acc,
                'auroc': auroc,
                'sparsity': sparsity}, self)
        return loss + regloss

############ run #############################
def run_with_reg(reg, outdir="tmp", num_runs=1, datagen=gendata,
                 printreport=False, resume=True):
    print(outdir)
    thetas = []

    for i in range(num_runs):
        X, y = datagen()
        Xval, yval = datagen()
        if printreport:
            print("percentage of ones:", y.mean())
        # preprocess
        normalize = normalizer()
        X = normalize(X)
        Xval = normalize(Xval, train=False)
        # define model        
        model = Regresser(Predictor(1),
                          lossfun=lossfun,
                          regularizer=reg)
        optimizer = optimizers.AdaDelta()
        optimizer.setup(model)
        # train model
        train_iter = iterators.SerialIterator(TupleDataset(X,y),
                                                  batch_size=100,
                                                  shuffle=False)
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (niterations, 'epoch'),
                                   out=outdir)
        trainer.extend(extensions.LogReport(log_name="log_"+str(i)))
        if printreport:
            # validate
            test_iter = iterators.SerialIterator\
                        (TupleDataset(Xval,yval),
                         batch_size=100,
                         repeat=False,
                         shuffle=False)
            trainer.extend(extensions.Evaluator(test_iter, model))
            trainer.extend(extensions.PrintReport(['main/accuracy',
                                                   'main/penalty',
                                                   'validation/main/loss',
                                                   'main/loss']))
            
        try:
            trainer.run()
            should_break=False
        except:
            should_break=True
        # save model
        if trainer.observation.get('main/loss') is not None and\
           trainer.observation.get('main/penalty') is not None and\
           (trainer.observation['main/loss'].data == np.nan or\
            trainer.observation['main/loss'].data == np.inf or\
            trainer.observation['main/penalty'].data == np.nan or\
            trainer.observation['main/penalty'].data == np.inf): continue
        W = model.predictor.l1.W
        b = model.predictor.l1.b
        theta = F.concat((F.flatten(W), b), 0)
        thetas.append(theta.data)

        if should_break: break
        
    if resume and os.path.isfile("theta.npy"):
        thetas = np.vstack((np.load("theta.npy"), np.array(thetas)))
    np.save(os.path.join(outdir, "theta"), np.array(thetas))
    

################# set up for known ###########
# penalyze b 
# r = np.zeros(d+1) # for modified penalty
# if d == 2: r[0] = 1
# w0 = 1 - r # don't penalize known
# w1 = w0 + 1 # penalize unknown more

# don't penalyze b
r = np.zeros(d) # for modified penalty
if d == 2: r[0] = 1
w0 = 1 - r # don't penalize known
w1 = w0 + 1 # penalize unknown more

################# parameter search ###########
def paramsSearch():
    Xtrain, ytrain = gendata(True, 'train')
    Xval, yval = gendata(True, 'val')
    # preprocess
    normalize = normalizer()
    Xtrain = normalize(Xtrain)
    Xval = normalize(Xval, train=False)    
    # train model and choose parameter based on performance on
    # validation data
    params_cand = {
        'lasso': (lasso, (0.1, 0.01, 0.001, 0.0001)),
        'ridge': (ridge, (0.1, 0.01, 0.001, 0.0001)),
        'enet': (enet, (0.1, 0.01, 0.001, 0.0001),
                 tuple(i/10 for i in range(11))),
        'penalty': (penalty, (r,), (0.1, 0.01, 0.001, 0.0001),
                    tuple(i/10 for i in range(11))),
        'wlasso': (weightedLasso, (w1,),
                   (0.1, 0.01, 0.001, 0.0001)),
        'wridge': (weightedRidge, (w1,),
                   (0.1, 0.01, 0.001, 0.0001)),
        'owl': (OWL, ([2,1], [1,1], [1,0]),
                (0.1, 0.01, 0.001, 0.0001)),
        'eye': (eye, (r,), (0.1, 0.01, 0.001, 0.0001))
    }

    for method in params_cand:
        m = params_cand[method]
        f, args = m[0], m[1:]
        for arg in product(*args):
            outdir = "val/" + method + str(arg)
            print(outdir)
            reg = f(*arg)
            # train
            model = Regresser(Predictor(1),
                              lossfun=lossfun,
                              regularizer=reg)
            optimizer = optimizers.AdaDelta()
            optimizer.setup(model)
            train_iter = iterators.SerialIterator\
                         (TupleDataset(Xtrain,ytrain),
                          batch_size=100,
                          shuffle=False)
            updater = training.StandardUpdater(train_iter,
                                               optimizer)
            trainer = training.Trainer(updater, (niterations,
                                                 'epoch'),
                                       out=outdir)

            # validate
            test_iter = iterators.SerialIterator\
                        (TupleDataset(Xval,yval),
                         batch_size=100,
                         repeat=False,
                         shuffle=False)
            trainer.extend(extensions.Evaluator(test_iter, model))
            trainer.extend(extensions.LogReport(log_name="log"))
            trainer.run()

def paramsSearchNd():
    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # construct known and unknown variables
    # in this case there's 220 variables with 110 variables being noise    
    nrvars = nrgroups * pergroup
    r = np.zeros(nrvars)
    for i in range(0, nrvars//pergroup):
        r[(i*pergroup):(i*pergroup+i)] = 1
    r = np.concatenate((r,r))
    
    # gen data
    Xtrain, ytrain = genPartitionData(nrgroups=nrgroups,
                                      nirgroups=nirgroups,
                                      pergroup=pergroup,n=n)
    Xval, yval = genPartitionData(nrgroups=nrgroups,
                                  nirgroups=nirgroups,
                                  pergroup=pergroup,n=n)

    # preprocess 5000 examples
    normalize = normalizer()
    Xtrain = normalize(Xtrain)
    Xval = normalize(Xval, train=False)    
    # train model and choose parameter based on performance on
    # validation data
    w1 = 2-r
    owl1 = np.arange(r.size) # polytope
    owl2 = np.zeros(r.size)  # inf norm
    owl2[0] = 1
    params_cand = {
        'lasso': (lasso, (1e-2, 5e-3, 1e-3, 5e-4)),
        'ridge': (ridge, (1e-2, 5e-3, 1e-3, 5e-4)),
        'enet': (enet, (1e-2, 5e-3, 1e-3, 5e-4),
                 tuple(i/10 for i in range(1,10))),
        'eye': (eye, (r,), (1e-2, 5e-3, 1e-3, 5e-4)),
        'wlasso': (weightedLasso, (w1,),
                   (1e-2, 5e-3, 1e-3, 5e-4)),
        'wridge': (weightedRidge, (w1,),
                   (1e-2, 5e-3, 1e-3, 5e-4)),
        'owl': (OWL, (owl1, owl2),
                (1e-2, 5e-3, 1e-3, 5e-4))
    }

    hash_map = {}
    for method in params_cand:
        m = params_cand[method]
        f, args = m[0], m[1:]
        for arg in product(*args):
            arghash = hashlib.md5(str(arg).encode()).hexdigest()
            hash_map[arghash] = arg
            pickle.dump(hash_map, open("val/hashmap","wb"))
            outdir = "val/" + method + "(" + arghash
            print(outdir)
            reg = f(*arg)
            # train
            model = Regresser(Predictor(1),
                              lossfun=lossfun,
                              regularizer=reg)
            optimizer = optimizers.AdaDelta()
            optimizer.setup(model)
            train_iter = iterators.SerialIterator\
                         (TupleDataset(Xtrain,ytrain),
                          batch_size=100,
                          shuffle=False)
            updater = training.StandardUpdater(train_iter,
                                               optimizer)
            trainer = training.Trainer(updater, (niterations,
                                                 'epoch'),
                                       out=outdir)
            # validate
            test_iter = iterators.SerialIterator\
                        (TupleDataset(Xval,yval),
                         batch_size=100,
                         repeat=False,
                         shuffle=False)
            trainer.extend(extensions.Evaluator(test_iter, model))
            trainer.extend(extensions.LogReport(log_name="log"))
            trainer.run()
            
############ set up regularizers #############

def experiment2d(num_runs=100):
    def run_with_reg_wrapper(num_runs):
        def _f(*args,**kwargs):
            return run_with_reg(*args, **kwargs, num_runs=num_runs)
        return _f
    run = run_with_reg_wrapper(num_runs)
    # actual run
    run(enet(0.01, 0.2), "result_enet") 
    run(eye(array([ 1.,  0.]), 0.01), "result_eye")
    run(lasso(0.0001), "result_lasso") 
    run(OWL([2, 1], 0.01), "result_owl")
    run(penalty(array([ 1.,  0.]), 0.1, 1.0),
                 "result_penalty")
    run(ridge(0.001), "result_ridge")
    run(weightedLasso(array([ 1.,  2.]), 0.01),
                 "result_wlasso")
    run(weightedRidge(array([ 1.,  2.]), 0.01), 
                 "result_wridge")


def experimentNd(num_runs=10):
    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # construct known and unknown variables
    nrvars = nrgroups * pergroup
    r = np.zeros(nrvars)
    for i in range(0, nrvars//pergroup):
        r[(i*pergroup):(i*pergroup+i)] = 1
    r = np.concatenate((r,r))
    w1 = 2-r
    owl1 = np.arange(r.size) # polytope
    owl2 = np.zeros(r.size)  # inf norm
    owl2[0] = 1
    # gen data
    datagen = lambda: genPartitionData(nrgroups=nrgroups,
                                       nirgroups=nirgroups,
                                       pergroup=pergroup,n=n)
    def run_with_reg_wrapper(num_runs, datagen):
        def _f(*args,**kwargs):
            return run_with_reg(*args, **kwargs,
                                num_runs=num_runs,
                                datagen=datagen)
        return _f
    run = run_with_reg_wrapper(num_runs, datagen)    
    # run
    run(eye(r, 0.01), outdir="result_eye")
    # run(enet(0.001, 0.4), outdir="result_enet")
    # run(lasso(0.001), outdir="result_lasso")
    # run(ridge(0.001), outdir="result_ridge")
    # run(weightedLasso(w1, 0.01), outdir="result_wlasso")
    # run(weightedRidge(w1, 0.01), outdir="result_wridge")
    # run(OWL(owl2, 0.001), outdir="result_owl")     

############# helpers
def distribution_normalizer(p, q):
    # normalize p and q to abs, p,q are np array
    if np.abs(p).sum() == 0:
        pnorm = np.ones_like(p) / p.size
    if np.abs(q).sum() == 0:
        qnorm = np.ones_like(p) / q.size
    pnorm = np.abs(p) / np.abs(p).sum()
    qnorm = np.abs(q) / np.abs(q).sum()
    # smooth
    smooth = 1e-6
    if (pnorm == 0).sum() > 0:
        pnorm = (np.abs(pnorm)+smooth) / (np.abs(pnorm)+smooth).sum()
    if (qnorm == 0).sum() > 0:
        qnorm = (np.abs(qnorm)+smooth) / (np.abs(qnorm)+smooth).sum()
    return pnorm, qnorm

def kl(p, q):
    # calculate KL(p||q) = E(log p/q)
    pnorm, qnorm = distribution_normalizer(p, q)
    return (pnorm * np.log(pnorm) - pnorm * np.log(qnorm)).sum()

def emd(p,q):
    pnorm, qnorm = distribution_normalizer(p, q)
    m, n = pnorm.size, qnorm.size
    dist_matrix = np.zeros((m, n))
    for i in range(m):
        dist_matrix[i,:] = np.abs(range(-i, -i+n))
    work, flow = emd_with_flow(pnorm.astype(np.float64),
                               qnorm.astype(np.float64), dist_matrix)
    return work
    
def knownRiskFactorReader(r, t, nrgroups, nirgroups, pergroup):
    # assume r is formulated such that nrgroups are first
    # and pergroup are together in r
    # t is theta
    assert r.size == (nrgroups + nirgroups) * pergroup, "size not checked"
    assert r.size == t.size, "size mismatch"
    ptr_s = 0
    while ptr_s < r.size:
        ptr_e = ptr_s + pergroup
        rout = r[ptr_s:ptr_e]
        tout = t[ptr_s:ptr_e]        
        if ptr_s < nrgroups * pergroup:
            yield rout, tout, 'relevant'
        else:
            yield rout, tout, 'irrelevant'
        ptr_s += pergroup

def naiveKLmetric(risk, theta, nrgroups, nirgroups, pergroup, method="kl"):
    # not intended for client: for report use    
    # assume r is formulated such that nrgroups are first
    # and pergroup are together in r
    loss = 0
    g = knownRiskFactorReader(risk, theta,
                              nrgroups, nirgroups, pergroup)
    metric = {"kl": kl, "emd": emd}[method]
    for r, t, tag in g:
        l =  0
        w = np.abs(t).sum() # weight for loss
        if tag == "relevant":
            if np.all(r==0): # all unknown (want spike)
                target = np.zeros_like(r)
                target[0] = 1
                l = min(metric(np.roll(target, i), t)
                        for i in range(r.size))
            else: # has known
                l = metric(r, t)
        elif tag == "irrelevant":
            target = np.ones_like(r)
            l = metric(target, t)
        loss += l
        yield l,w,r,t,tag
    
    # the following 2 may not be important b/c the fact of same performance
    # already says the following is true
    # deal with irrelvant: should all be 0
    # each should be weighted against the true theta_i for each group
    return loss    

def KLmetricUser(t, method="kl"):
    # not intended for client: for internal report use
    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # construct known and unknown variables
    nrvars = nrgroups * pergroup
    r = np.zeros(nrvars)
    for i in range(0, nrvars//pergroup):
        r[(i*pergroup):(i*pergroup+i)] = 1
    r = np.concatenate((r,r))
    n = naiveKLmetric(r, t, nrgroups, nirgroups, pergroup, method)    
    return n

def reportKL(fn, f=lambda x: x.mean(), method="kl"):
    # report kl mean for each function
    t = np.load(os.path.join(fn, "theta.npy"))
    t = t[:,:-1] # exclude b
    n, d = t.shape
    iterables = tuple(KLmetricUser(t[i], method=method) for i in range(n))
    tagged_stats = []
    for items in itertools.zip_longest(*iterables):
        tag = items[0][4]
        l = np.array(list(map(lambda x: x[0], items)))
        tagged_stats.append((tag, f(l)))
    return tagged_stats

def reportTheta(fn, f=lambda x: x.mean(), method="kl"):
    # report kl mean for each function
    t = np.load(os.path.join(fn, "theta.npy"))
    t = t[:,:-1] # exclude b
    n, d = t.shape
    iterables = tuple(KLmetricUser(t[i], method=method) for i in range(n))
    tagged_stats = []
    for items in itertools.zip_longest(*iterables):
        tag = items[0][4]
        theta = np.array(list(map(lambda x: x[3], items)))
        tagged_stats.append((tag, f(theta)))
    return tagged_stats

#------------temporary function###################
def tmp():
    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # construct known and unknown variables
    nrvars = nrgroups * pergroup
    r = np.zeros(nrvars)
    for i in range(0, nrvars//pergroup):
        r[(i*pergroup):(i*pergroup+i)] = 1
    r = np.concatenate((r,r))
    w1 = 2-r
    owl1 = np.arange(r.size) # polytope
    owl2 = np.zeros(r.size)  # inf norm
    owl2[0] = 1
    # gen data
    datagen = lambda: genPartitionData(nrgroups=nrgroups,
                                       nirgroups=nirgroups,
                                       pergroup=pergroup,n=n)
    def run_with_reg_wrapper(datagen):
        def _f(*args,**kwargs):
            return run_with_reg(*args, **kwargs,
                                datagen=datagen,
                                printreport=True,
                                resume=False)
        return _f
    run = run_with_reg_wrapper(datagen)    
    return run, r
