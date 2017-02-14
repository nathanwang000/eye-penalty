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

# todo
# 5. consider higher dimension (can be easily generated)

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
def eye(r, alpha=1, l1_ratio=0.5):
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
left = -2.5
right = 1.5
mu = (left + right) / 2
scaleh = 4
signoise = 0.2
munoise = 0

def gendata(plot=False, name=None, d=d):
    h = np.linspace(left,right,n).reshape(n,1)
    y = h > mu
    X = np.repeat(h, d, 1)
    noise = np.random.randn(n,d)  * signoise + munoise
    alpha = np.diag(np.random.randint(1,scaleh,size=d))
    X = X.dot(alpha) + noise

    if plot and d == 2:
        plt.scatter(X[:,0],X[:,1],c=y.astype(np.int64),alpha=0.5)
        plt.axvline(x=mu, color='k', linestyle='--')
        plt.title("dataset: $X_i$=$\\alpha_i$h+N(0,$\sigma$)")
        plt.show()
        name = name or 'data'
        plt.savefig('figures/'+name+'.png', bbox_inches='tight')
    return X.astype(np.float32), y.astype(np.float32)

#############the model#########################
class Predictor(Chain):
    def __init__(self, n_out):
        super(Predictor, self).__init__(
            l1 = L.Linear(None, n_out) # n_in -> n_out
        )
    def __call__(self, x):
        y = self.l1(x)
        return F.sigmoid(y)

def lossfun(y, t):
    # max likelihood
    logloss = F.sum(F.where(t.data > 0,
                            -F.log(y), -F.log(1-y))) / len(t)
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
        acc = (n-np.sum(abs((y.data > 0.5) - t.data))) / n
        y_true = t.data if isinstance(t, Variable) else t
        try: # y may be nan if predictor.W is nan
            auroc = roc_auc_score(y_true, y.data)
            fpr, tpr, threshold = roc_curve(y_true, y.data)
        except:
            auroc = -1
        report({'loss': loss,
                'penalty': regloss,
                'accuracy': acc,
                'auroc': auroc}, self)
        return loss + regloss

############ run #############################
def run_with_reg(reg, outdir="tmp", num_runs=1):
    thetas = []
    # define model
    for i in range(num_runs):
        X, y = gendata()
        # preprocess
        normalize = normalizer()
        X = normalize(X)
        
        model = Regresser(Predictor(1),
                          lossfun=lossfun,
                          regularizer=reg)
        optimizer = optimizers.AdaDelta()
        optimizer.setup(model)
        # train model
        train_iter = iterators.SerialIterator(TupleDataset(X,y),
                                          batch_size=n,
                                          shuffle=False)
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (niterations, 'epoch'),
                                   out=outdir)
        trainer.extend(extensions.LogReport(log_name="log_"+str(i)))
        # trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'main/penalty', 'main/loss']))       
        trainer.run()
        # save model
        W = model.predictor.l1.W
        b = model.predictor.l1.b
        theta = F.concat((F.flatten(W), b), 0)
        thetas.append(theta.data)
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
        'wlasso': (weightedLasso, (w0, w1),
                   (0.1, 0.01, 0.001, 0.0001)),
        'wridge': (weightedRidge, (w0, w1),
                   (0.1, 0.01, 0.001, 0.0001)),
        'owl': (OWL, ([2,1], [1,1], [1,0]),
                (0.1, 0.01, 0.001, 0.0001)),
        'eye': (eye, (r,), (0.1, 0.01, 0.001, 0.0001),
                tuple(i/10 for i in range(1,10)))
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
                          batch_size=n,
                          shuffle=False)
            updater = training.StandardUpdater(train_iter,
                                               optimizer)
            trainer = training.Trainer(updater, (niterations,
                                                 'epoch'),
                                       out=outdir)

            # validate
            test_iter = iterators.SerialIterator\
                        (TupleDataset(Xval,yval),
                         batch_size=n,
                         repeat=False,
                         shuffle=False)
            trainer.extend(extensions.Evaluator(test_iter, model))
            trainer.extend(extensions.LogReport(log_name="log"))
            trainer.run()

############ set up regularizers #############
def experiment(num_runs=100):
    def helper(num_runs):
        def _f(*args,**kwargs):
            return run_with_reg(*args, **kwargs, num_runs=num_runs)
        return _f
    run = helper(num_runs)
    # actual run
    run(enet(0.01, 0.6), "result_enet")
    run(eye(array([ 1.,  0.]), 0.01, 0.3), "result_eye")
    run(lasso(0.001), "result_lasso")
    run(OWL([1, 0], 0.01), "result_owl")
    run(ridge(0.001), "result_ridge")
    run(weightedLasso(array([ 1.,  2.]), 0.01),
                 "result_wlasso")
    run(weightedRidge(array([ 1.,  2.]), 0.01),
                 "result_wridge")
    run(penalty(array([ 1.,  0.]), 0.0001, 0.8),
                 "result_penalty")

