from experiment import experiment
import numpy as np
import comb_loss, sys, os

# todo:
# parameter choice criteria
# for each method, choose the one with the lowest sparsity score
# essentially eyeball the result because has to keep performance constant

# in fact choose the sparsest sln in the top 20% of the criteria
# should avoid race condition
def noise2d(index):
    name = "noise2d/" + str(index)
    os.makedirs(name, exist_ok=True)    
    datagen = lambda: comb_loss.gendata(signoise=0.2, n=100)
    risk = comb_loss.generate_risk(0, 0, 0, '2d')

    w1 = 2-risk # penalize unknown more
    
    paramsDict = {
        'eye':     [(risk, 0.01)],
        'wlasso':  [(w1, 0.01)],
        'wridge':  [(w1, 0.01)],
        'penalty': [(risk, 0.01, 0.4)],
        'owl':     [([2,1], 0.01)],
        'lasso':   [(0.01,)],
        'enet':    [(0.01, 0.2)]
    }
    experiment(paramsDict,
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               niterations=1000)

def diffTheta(index):
    name = "diff_theta/" + str(index)
    os.makedirs(name, exist_ok=True)
    datagen, (theta, risk, nd) = comb_loss.genDiffTheta(n=5000)
    
    # need to save theta, risk, nd
    np.save(os.path.join(name, "theta.npy"), theta)
    np.save(os.path.join(name, "risk.npy"), risk)
    np.save(os.path.join(name, "nd.npy"), nd)        
    
    w1 = 2-risk
    owl1 = np.zeros(risk.size)
    owl1[0] = 1
    paramsDict = {
        'eye':     [(risk, 0.01)],
        'wlasso':  [(w1, 0.01)],
        'wridge':  [(w1, 0.01)],
        'penalty': [(risk, 0.01, 0.4)],
        'owl':     [(owl1, 0.01)],
        'lasso':   [(0.01,)],
        'enet':    [(0.01, 0.2)]
    }
    experiment(paramsDict,
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               niterations=1000
    )


# diffTheta(sys.argv[1])
noise2d(sys.argv[1])

