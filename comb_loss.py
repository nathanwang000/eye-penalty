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

## todo:
## 1. change trigger to convergent (not practical) (done)
## 2. create validation set and use that to set hyper params (done)
## 3. add in ordered weighted lasso (done)
## 4. how to compare relative size??

############ utility functions ##################
def l1norm(v):
    return F.sum(abs(v))

def l2normsq(v):
    return F.sum(v**2)

############ penalties ##########################
def eye(r, alpha=1, l1_ratio=0.5):
    def solveQuadratic(a, b, c):
        return (-b + sqrt(b**2-4*a*c)) / (2*a)

    # want to force f_(theta/t) = c for which c
    # 45 degree, which is slope of -1, where f_ is penalty
    # solve for t, just a quadratic equation!
    # and in fact t always real because c > 0
    def f_(theta):
        if l1_ratio == 0 or l1_ratio == 1:
            return penalty(r, alpha, l1_ratio)(theta)
        b = l1_ratio * l1norm((1-r)*theta)
        a = 0.5 * (1-l1_ratio) * l2normsq(r*theta)
        c = alpha * l1_ratio**2 / (1-l1_ratio)
        # a (1/t)**2 + b (1/t) = c
        return 1 / solveQuadratic(a, b, -c)
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

# using http://scikit-learn.org/stable/modules/linear_model.html
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
n = 100
d = 2
mu = -0.5
sig = 0.5

def gendata(plot=False, name=None):
    column = np.sort(mu + np.random.randn(n,1) * sig, 0)
    X = np.zeros((n, d), dtype=np.float32)
    X[:,0] = column[:,0]
    X[:,1] = column[:,0] * 2
    # X = np.repeat(column, d, 1)
    y = (X[:,0] > mu).reshape(n,1).astype(np.float32)
    if plot:
        plt.plot(X[:,0], X[:,1], 'o')
        plt.axvline(x=mu, color='k', linestyle='--')
        plt.title("generated dataset, $x_1$ = 2*$x_0$")
        name = name or 'data'
        plt.savefig('figures/'+name+'.png', bbox_inches='tight')
        plt.clf()
    return X, y

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
        report({'loss': loss,
                'penalty': regloss,
                'accuracy': acc}, self)
        return loss + regloss

############ run #############################
def run_with_reg(reg, outdir, num_runs=100):
    thetas = []
    # define model
    for i in range(num_runs):
        X, y = gendata()        
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
        trainer = training.Trainer(updater, (1000, 'epoch'),
                                   out=outdir)
        # trainer = training.Trainer(updater, ConvergeTrigger(),
        #                            out=outdir) 
        trainer.extend(extensions.LogReport(log_name="log_"+str(i)))
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
    # train model and choose parameter based on performance on
    # validation data
    params_cand = {
        # penalyze 'b'
        # 'owl': (OWL, ([2,1,1], [1,1,1], [1,0,1]),
        #         (1, 0.1, 0.01, 0.001, 0.0001)),
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
        # don't penalyze 'b'
        'owl': (OWL, ([2,1], [1,1], [1,0]),
                (0.1, 0.01, 0.001, 0.0001))
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
            trainer = training.Trainer(updater, (1000, 'epoch'),
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
def experiment():
    run_with_reg(OWL([1, 0], 0.01), "result_owl")
    run_with_reg(ridge(0.01), "result_ridge")
    run_with_reg(lasso(0.0001), "result_lasso")
    run_with_reg(enet(0.0001, 1.0), "result_enet")
    run_with_reg(penalty(array([ 1.,  0.]), 0.001, 0.6), "result_penalty")
    run_with_reg(weightedLasso(array([ 1.,  2.]), 0.001), "result_wlasso")
    run_with_reg(weightedRidge(array([ 1.,  2.]), 0.001), "result_wridge")        

