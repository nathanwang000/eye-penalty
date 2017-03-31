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
from sklearn.metrics import roc_auc_score, roc_curve
import hashlib, pickle
import itertools
from scipy.linalg import block_diag
from scipy import sparse
from sparseDataset import SparseDataset

# example sparse data
def genSparseData(n, d):
    nonzero = 1000
    dataUpper = 1000
    row = np.random.randint(0,n,nonzero)
    col = np.random.randint(0,d, nonzero)
    data = np.random.randint(0,dataUpper,nonzero)
    X = sparse.csr_matrix((data, (row, col)), shape=(n,d), dtype=np.float32)
    y = np.random.randint(0,2,(n,1)).astype(np.float32)
    return X, y

def gendata(n=800000, d=50000):
    return lambda: genSparseData(n, d)

# utility function
def l1norm(v):
    return F.sum(abs(v))

def l2normsq(v):
    return F.sum(v**2)

def normalizer(mu=0, std=1):
    currsum = 0
    n = 0
    currx2 = 0

    def mult_(a, b):
        if sparse.issparse(a) or sparse.issparse(b):
            return a.multiply(b)
        else:
            return np.multiply(a, b)
        
    def transform(X):
        mean = currsum / n
        var  = currx2 / n - mult_(mean, mean)
        res = (X-mean) / np.sqrt(var) * std + mu
        return np.asarray(res)

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
        currx2  += np.sum(mult_(X,X), 0)
        n += X.shape[0]
        return transform(X)
    return normalize

# the penalty
def eye(r, alpha=1):
    def f_(theta):
        return alpha * (l1norm((1-r)*theta) +
                        F.sqrt(l1norm((1-r)*theta)**2 + l2normsq(r*theta)))
    return f_

# the model: logistic regression
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
        if self.regularizer:
            W = self.predictor.l1.W
            b = self.predictor.l1.b
            theta = F.flatten(W) # don't regularize b
            regloss = self.regularizer(theta)

        n = y.data.size
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
                'auroc': auroc}, self)
        return loss + regloss

# training: change datagen to actual data
def run_with_reg(reg, outdir="tmp", num_runs=1, datagen=gendata(10,2),
                 printreport=False, resume=True, niterations=10):
    print(outdir, "printreport="+str(printreport), "num_runs="+str(num_runs))
    thetas = []

    # open dir to see from log? to start
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
            
        # preprocess: assume already normalized
        # normalize = normalizer()
        # X = normalize(X)
        # Xval = normalize(Xval, train=False)
        
        # define model        
        model = Regresser(Predictor(1),
                          lossfun=lossfun,
                          regularizer=reg)
        optimizer = optimizers.AdaDelta()
        optimizer.setup(model)
        # train model
        train_iter = iterators.SerialIterator(SparseDataset(X,y),
                                              batch_size=100,
                                              shuffle=False)
        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, (niterations, 'epoch'),
                                   out=outdir)
        logname = "log_"+str(namebase+i)
        trainer.extend(extensions.LogReport(log_name=logname))
        if printreport:
            # validate
            test_iter = iterators.SerialIterator\
                        (SparseDataset(Xval,yval),
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
        # save model: don't save if weights are invalid
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

    thetafn = os.path.join(outdir, "theta.npy")        
    if resume and os.path.isfile(thetafn):
        thetas = np.vstack((np.load(thetafn), np.array(thetas)))
    np.save(thetafn, np.array(thetas))

# example run
def example_run():
    risk = np.array([1,0])
    alpha = 0.001
    run_with_reg(eye(risk, alpha),
                 datagen=gendata(1000,2), # gendata should return X, y
                 printreport=True, # print progress
                 resume=False, # continue from last time
                 niterations=30) # how many epoch to train

if __name__ == '__main__':
    example_run()
