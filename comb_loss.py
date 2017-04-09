import numpy as np
import traceback, json
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
from sklearn.metrics import roc_auc_score, roc_curve
import hashlib, pickle
import itertools, shutil
from pyemd import emd_with_flow
from scipy.linalg import block_diag

############ utility functions ##################
SMOOTH_CONST = 1e-7

def l1norm(v):
    return F.sum(abs(v))

def l2normsq(v):
    return F.sum(v**2)

def normalizer(mu=0, std=1):
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

def abs_gini(c):
    # calculate the gini coefficient for array np.abs(c)
    c = np.sort(np.abs(c)) / np.sum(np.abs(c))
    n = c.size
    w = np.array([(n-k+0.5)/n for k in range(1, n+1)])    
    return 1-2*(c*w).sum()

############ penalties ##########################
def eye(r, alpha=1):
    def f_(theta):
        return alpha * (l1norm((1-r)*theta) +
                        F.sqrt(l1norm((1-r)*theta)**2 +
                               l2normsq(r*theta)))
    return f_

def old_eye(r, alpha=1):
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
niterations = 1000
n = 100
d = 2

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def genCovX(C, n): # helper for difftheta
    # C is the covariance matrice (assume to be psd)
    # n is number of examples
    A = np.linalg.cholesky(C)
    d, _ = C.shape
    Z = np.random.randn(n, d)
    X = Z.dot(A.T) 
    return X.astype(np.float32)

def genCovData(C=None, theta=None, n=n, dim=d, signoise=5): # signoise on y
    # C is the covariance matrice (assume to be psd)
    # y = Bernoulli(sigmoid(\theta x)), z~Norm(0,I), x = Az
    noise = np.random.randn(n) * signoise
    if C is None:
        C = np.diag(np.ones(dim))
    if theta is None:
        theta = np.ones((d,1))         
    assert C.shape[0]==theta.size, "size mismatch"
    X = genCovX(C, n)

    # y = sigmoid(X.dot(theta) + noise)
    # for i in range(n):
    #     y[i] = np.random.binomial(1,y[i]) # bernoulli
    y = (X.dot(theta) + noise > 0).reshape(n,1)
    return X.astype(np.float32), y.astype(np.float32).reshape(y.size,1) 

def genDiffTheta(n=1000): # bernoulli so noise also on y
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import block_diag
    ng = np.random.poisson(10)
    w = np.random.normal(0,1,ng)
    nd = np.random.poisson(20,ng)
    d = nd.sum() # feature dimension
    risk = np.random.uniform(0,1,d)
    # linearly distribute w_i to theta_i according to risk_i
    theta = risk.copy()
    istart = 0
    for ndi in nd:
        iend = istart+ndi
        theta[istart:iend] /= risk[istart:iend].sum()
        istart = iend
    theta *= np.repeat(w, nd)
    InGrpCov = 0.95
    CovErr = np.random.normal(0.002, 0.01, (d,d))
    CovErr = (CovErr + CovErr.T) / 2 # not psd so no noise added here
    blocks = []
    for ndi in nd:
        block = np.diag(np.ones(ndi))
        block[block==0] = InGrpCov
        blocks.append(block)
    CovM = np.clip(block_diag(*blocks) + 0*CovErr, 0, 1)
    def _datagen():
        X = genCovX(C=CovM, n=n)
        y = sigmoid(X.dot(theta))
        for i in range(n):
            y[i] = np.random.binomial(1,y[i]) # bernoulli
        return X.astype(np.float32), y.astype(np.float32).reshape(y.size,1) 
        
    return _datagen, (theta, risk, nd, CovM)

# noise added on X, so really is correlation structure that's changed
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
    g_theta = np.ones(nrgroups)[:,None]
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

# noise added on X, so really is correlation structure that's changed
def gendata(plot=False, name=None, d=2,
            signoise=0.2, scaleh=2, munoise=0,
            left=-2.5, right=1.5, n=100):
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
    
    # max likelihood: has trouble when yhat=0 or 1 so need to scale it
    yhat = linscale(y, SMOOTH_CONST, 1-SMOOTH_CONST)
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

        if self.regularizer:
            W = self.predictor.l1.W
            b = self.predictor.l1.b
            # theta = F.concat((F.flatten(W), b), 0)
            theta = F.flatten(W) # don't regularize b
            regloss = self.regularizer(theta)
            sparsity = abs_gini(theta.data)

        acc = calcAcc(y, t)
        auroc = calcAuroc(y, t)

        report({'loss': loss,
                'total_loss': loss+regloss,
                'penalty': loss/regloss, 
                'sparsity': sparsity,
                'accuracy': acc,
                'auroc': auroc}, self)
        return loss + regloss

def calcAcc(y, t): # y is a variable
    return (n-np.sum(abs((y.data > 0.5) - t.data))) / y.data.size

def calcAuroc(y, t):
    try: # y may be nan if self.predictor.l1.W is nan, or y could only have 1 label
        y_true = t.data if isinstance(t, Variable) else t 
        auroc = roc_auc_score(y_true, y.data)        
        # fpr, tpr, threshold = roc_curve(y_true, y.data)
    except:
        auroc = -1
    return auroc
    
def getModelOptimizer(reg):
    # define model and optimizer
    model = Regresser(Predictor(1),
                      lossfun=lossfun,
                      regularizer=reg)
    optimizer = optimizers.AdaDelta()
    optimizer.setup(model)
    return model, optimizer
    
############ run #############################
def run_with_reg(reg, outdir="tmp", num_runs=1, datagen=gendata,
                 printreport=False, resume=True, niterations=1000,
                 namebase=None, validate=False):

    print(outdir)
    thetas = []

    # open dir to see from log? to start
    if namebase is None:
        namebase = 0
        if resume and os.path.exists(outdir):
            for fn in os.listdir(outdir):
                if fn.startswith("log_"):
                    number = int(fn[4:])
                    if number >= namebase:
                        namebase = number+1
    
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
        model, optimizer = getModelOptimizer(reg)
        modelpath = os.path.join(outdir, "%d.model" % (namebase+i))
        optpath = os.path.join(outdir, "%d.state" % (namebase+i))
        logname = "log_%d" % (namebase+i)
        logpath = os.path.join(outdir, logname)
        # if resume and the model we are using has a log file
        using_old_model = False        
        if resume and\
           os.path.isfile(modelpath) and\
           os.path.isfile(optpath) and\
           os.path.isfile(logpath):
            using_old_model = True
            serializers.load_npz(modelpath, model)            
            serializers.load_npz(optpath, optimizer)
            shutil.move(logpath, logpath+".old")
        
        # train model
        train_iter = iterators.SerialIterator(TupleDataset(X,y),
                                                  batch_size=100,
                                                  shuffle=False)
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (niterations, 'epoch'),
                                   out=outdir)

        trainer.extend(extensions.LogReport(log_name=logname))

        # validate
        test_iter = iterators.SerialIterator\
                    (TupleDataset(Xval,yval),
                     batch_size=100,
                     repeat=False,
                     shuffle=False)
        trainer.extend(extensions.Evaluator(test_iter, model))
        if printreport:
            trainer.extend(extensions.PrintReport(['validation/main/auroc',
                                                   'main/auroc',
                                                   'validation/main/accuracy',
                                                   # 'main/accuracy',
                                                   'validation/main/loss',
                                                   'main/penalty'
                                                   # 'main/loss',
                                                   ]))

        try:
            trainer.run()            
            should_break=False
        except KeyboardInterrupt:
            traceback.print_exc()
            should_break=True
        # save model
        if trainer.observation.get('main/loss') is not None and\
           trainer.observation.get('main/penalty') is not None and\
           (trainer.observation['main/loss'].data == np.nan or\
            trainer.observation['main/loss'].data == np.inf): continue
        W = model.predictor.l1.W
        b = model.predictor.l1.b
        theta = F.concat((F.flatten(W), b), 0)
        thetas.append(theta.data)

        serializers.save_npz(modelpath, model)
        serializers.save_npz(optpath, optimizer)

        if using_old_model:
            # combine old and new logs            
            newlog = json.load(open(logpath))
            oldlog = json.load(open(logpath+".old"))
            oldlog.extend(newlog)
            json.dump(oldlog, open(logpath, 'w'))
            
        if should_break: break

    if not validate:
        thetafn = os.path.join(outdir, "theta.npy")        
        if resume and os.path.isfile(thetafn):
            thetas = np.vstack((np.load(thetafn), np.array(thetas)))
        np.save(thetafn, np.array(thetas))
    

################# set up for known ###########
# r = np.zeros(d) # for modified penalty
# if d == 2: r[0] = 1
# w0 = 1 - r # don't penalize known
# w1 = w0 + 1 # penalize unknown more

################# parameter search ###########
def paramsSearch2d(datagen=lambda: gendata(n=100,signoise=0.2),
                   risk = np.array([1,0]),
                   basedir="val",
                   methods=['lasso', 'enet', 'penalty',
                            'wlasso', 'wridge', 'owl', 'eye'],
                   niterations = 1000):
    # don't penalyze b
    w1 = 2-risk # penalize unknown more
    
    Xtrain, ytrain = datagen()
    Xval, yval = datagen()
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
        'penalty': (penalty, (risk,), (0.1, 0.01, 0.001, 0.0001),
                    tuple(i/10 for i in range(11))),
        'wlasso': (weightedLasso, (w1,),
                   (0.1, 0.01, 0.001, 0.0001)),
        'wridge': (weightedRidge, (w1,),
                   (0.1, 0.01, 0.001, 0.0001)),
        'owl': (OWL, ([2,1], [1,1], [1,0]),
                (0.1, 0.01, 0.001, 0.0001)),
        'eye': (eye, (risk,), (0.1, 0.01, 0.001, 0.0001))
    }

    for method in methods:
        m = params_cand[method]
        f, args = m[0], m[1:]
        for arg in product(*args):
            outdir = os.path.join(basedir, method + str(arg))
            print(outdir)
            reg = f(*arg)
            # train
            model, optimizer = getModelOptimizer(reg)
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

def paramsSearchMd(datagen,
                   risk,
                   basedir="val",
                   methods=['lasso', 'eye', 'owl',
                            'wlasso', 'wridge', 'enet', 'penalty'],
                   niterations=1000):
    Xtrain, ytrain = datagen()
    Xval, yval = datagen()
    
    # preprocessing
    normalize = normalizer()
    Xtrain = normalize(Xtrain)
    Xval = normalize(Xval, train=False)    
    # train model and choose parameter based on performance on
    # validation data
    w1 = 2-risk
    # owl1 = np.arange(risk.size) # polytope
    owl2 = np.zeros(risk.size)  # inf norm
    owl2[0] = 1
    params_cand = {
        'eye': (eye, (risk,), (5e-2, 1e-2, 5e-3)),
        'wlasso': (weightedLasso, (w1,),
                   (1e-1, 1e-2, 5e-3)),
        'wridge': (weightedRidge, (w1,),
                   (1e-1, 1e-2, 5e-3)),
        'lasso': (lasso, (1e-1, 1e-2, 5e-3, 1e-3, 5e-4)),
        'ridge': (ridge, (1e-1, 1e-2, 5e-3, 1e-3, 5e-4)),
        'owl': (OWL, (owl2,),
                (1e-1, 1e-2, 5e-3, 1e-3, 5e-4)),
        # 30 + 55 + 55
        'enet': (enet, (1e-1, 1e-2, 5e-3, 1e-3, 5e-4),
                 tuple(i/10 for i in range(11))),
        'penalty': (penalty, (risk,), (1e-1, 1e-2, 5e-3, 1e-3, 5e-4),
                    tuple(i/10 for i in range(11)))

    }

    os.makedirs(basedir, exist_ok=True)           
    hash_map = {}
    for method in methods:
        m = params_cand[method]
        f, args = m[0], m[1:]
        for arg in product(*args):
            arghash = hashlib.md5(str(arg).encode()).hexdigest()
            hash_map[arghash] = arg
            pickle.dump(hash_map, open(os.path.join(basedir, "hashmap"),"wb"))
            outdir = os.path.join(basedir, method + "(" + arghash)
            print(outdir)
            reg = f(*arg)
            # train
            model, optimizer = getModelOptimizer(reg)
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
def generate_risk(nrgroups, nirgroups, pergroup, experiment):
    if experiment == '2d':
        r = np.array([1,0])
    elif experiment == "binary_r":
        nrvars = nrgroups * pergroup
        r = np.zeros(nrvars)
        for i in range(0, nrvars//pergroup):
            r[(i*pergroup):(i*pergroup+i)] = 1
        r = np.concatenate((r,r))
    elif experiment == "corr":
        rbase = np.zeros(pergroup)
        rbase[:pergroup//2] = 1
        r = np.concatenate([rbase for _ in range(nrgroups)])
    elif experiment == "frac_r":
        # consider sigmoid, log, exp
        def clip_(f):
            def f_(a, n):
                res = f(a, n)
                m, M = np.min(res), np.max(res)
                return (res-m) / (M-m)
            return f_
        @clip_
        def sigmoid_(a, n):
            x = np.linspace(-1,1,n)
            return 1/(1+np.exp(-a*x))
        @clip_
        def exp_(a, n):
            x = np.linspace(-1,1,n)
            return np.exp(a*x)
        @clip_
        def log_(a, n):
            a = 30 * a
            x = np.linspace(0,1,n)
            return np.log(a*x+1)
        funcs = {
            sigmoid_,
            exp_,
            log_
        }
        alphas = [1,3,6,10]         
        r = np.ones(nrgroups * pergroup)
        rindex = 0
        for f in funcs:
            for a in alphas:
                r[rindex:rindex+pergroup] = f(a, pergroup)
                rindex += pergroup
    return r

    
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
    if (pnorm == 0).sum() > 0:
        pnorm = (np.abs(pnorm)+SMOOTH_CONST) / (np.abs(pnorm)+SMOOTH_CONST).sum()
    if (qnorm == 0).sum() > 0:
        qnorm = (np.abs(qnorm)+SMOOTH_CONST) / (np.abs(qnorm)+SMOOTH_CONST).sum()
    return pnorm, qnorm

def kl(p, q): # balanced version
    # calculate (KL(p||q) + K(q||p)) / 2 where KL(p||q) = E(log p/q)
    pnorm, qnorm = distribution_normalizer(p, q)
    def kl_(p, q):
        return (p * np.log(p) - p * np.log(q)).sum()
    return (kl_(pnorm, qnorm) + kl_(qnorm, pnorm)) / 2

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
    assert r.size == t.size, "size mismatch r:%d, t:%d" % (r.size, t.size)
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

def KLmetricUser(t, nrgroups=11, nirgroups=11, pergroup=10, method="kl",
                 experiment="binary_r"):
    # not intended for client: for internal report use
    # construct known and unknown variables
    r = generate_risk(nrgroups, nirgroups, pergroup, experiment)
    n = naiveKLmetric(r, t, nrgroups, nirgroups, pergroup, method)    
    return n

def reportKL(fn, f=lambda x: x.mean(), method="kl",
             nrgroups=11, nirgroups=11, pergroup=10,
             experiment="binary_r"):
    # report kl mean for each function
    t = np.load(os.path.join(fn, "theta.npy"))
    t = t[:,:-1] # exclude b
    n, d = t.shape
    iterables = tuple(KLmetricUser(t[i], method=method,
                                   nrgroups=nrgroups,
                                   nirgroups=nirgroups,
                                   pergroup=pergroup,
                                   experiment=experiment) for i in range(n))
    tagged_stats = []
    for items in itertools.zip_longest(*iterables):
        tag = items[0][4]
        l = np.array(list(map(lambda x: x[0], items)))
        tagged_stats.append((tag, f(l)))
    return tagged_stats

def reportTheta(fn, f=lambda x: x.mean(), method="kl",
                nrgroups=11, nirgroups=11, pergroup=10,
                experiment="binary_r"):
    # report kl mean for each function
    t = np.load(os.path.join(fn, "theta.npy"))
    t = t[:,:-1] # exclude b
    n, d = t.shape
    iterables = tuple(KLmetricUser(t[i], method=method,
                                   nrgroups=nrgroups,
                                   nirgroups=nirgroups,
                                   pergroup=pergroup,
                                   experiment=experiment) for i in range(n))

    tagged_stats = []
    for items in itertools.zip_longest(*iterables):
        tag = items[0][4]
        theta = np.array(list(map(lambda x: x[3], items)))
        tagged_stats.append((tag, f(theta)))
    return tagged_stats

#------------temporary function###################
def _test(signoise=0.2):
    #################
    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # setup
    gridSearch = paramsSearchMd
    risk = generate_risk(nrgroups, nirgroups, pergroup, "binary_r")
    basedir = 'val_binaryR'
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*(nrgroups+nirgroups)))
    theta = np.zeros((nrgroups + nirgroups) * pergroup)
    theta[:nrgroups*pergroup] = 1
    datagen = lambda: genCovData(C=C, theta=theta,
                                 n=n, signoise=signoise)

    ###################
    def run_with_reg_wrapper(datagen):
        def _f(*args,**kwargs):
            return run_with_reg(*args, **kwargs,
                                datagen=datagen,
                                printreport=True,
                                resume=False)
        return _f
    run = run_with_reg_wrapper(datagen)    
    return run, risk

def tmp(signoise=5, d=2, datagen=None):
    ################
    gridSearch = paramsSearch2d
    risk = np.ones(d) #generate_risk(0, 0, 0, '2d')
    if not datagen:
        datagen = lambda: gen2ddata(signoise=signoise, n=100, d=d, plot=True)

    ################
    def run_with_reg_wrapper(datagen):
        def _f(*args,**kwargs):
            return run_with_reg(*args, **kwargs,
                                datagen=datagen,
                                printreport=True,
                                resume=False)
        return _f
    run = run_with_reg_wrapper(datagen)    
    return run, risk
    
def gen2ddata(plot=False, name=None, d=2,
              signoise=0.2, scaleh=2, munoise=0,
              left=-2.5, right=1.5, n=100):
    mu = (left + right) / 2    
    h = np.linspace(left,right,n).reshape(n,1)
    X = np.repeat(h, d, 1)
    noise = np.random.randn(n,1)  * signoise + munoise
    alpha = np.diag(np.random.randint(1,scaleh,size=d))
    X = X.dot(alpha)

    # y = X.dot(0.5*np.ones(d)[:,None]) + noise > mu

    y = sigmoid(X.dot(np.ones(d)[:,None]) + noise)
    for i in range(n):
        y[i] = np.random.binomial(1,y[i]) # bernoulli

    if plot and d==2:
        plt.scatter(X[:,0],X[:,1],c=y.astype(np.int64),alpha=0.5)
        plt.axvline(x=mu*alpha[0,0], color='k', linestyle='--')
        plt.axhline(y=mu*alpha[1,1], color='k', linestyle='--')        
        plt.title("dataset: $X_i$=$\\alpha_i$h+N(0,$\sigma$)")
        name = name or 'data'
        plt.savefig('figures/'+name+'.png', bbox_inches='tight')
        plt.show()        
    return X.astype(np.float32), y.astype(np.float32).reshape(y.size,1)
