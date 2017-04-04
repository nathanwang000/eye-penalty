import comb_loss
from experiment import experiment
# in fact choose the sparsest sln in the top 20% of the criteria

# TODO to think about how to run this!!!
risk  = comb_loss.generate_risk(0, 0, 0, '2d')
w0    = 1-risk
w1    = 2-risk
alpha = (0.1, 0.01, 0.001, 0.0001)
beta  = tuple(i/10 for i in range(11))

params_cand = {
    'lasso':   (alpha,),
    'ridge':   (alpha,),
    'enet':    (alpha, beta),
    'penalty': ((risk,), alpha, beta),
    'wlasso':  ((w1,), alpha),
    'wridge':  ((w1,), alpha),
    'owl':     (([2,1], [1,1], [1,0]), alpha),
    'eye':     ((risk,), alpha)
}



def noise2d(index, niterations=5000, signoise=0, name="default"):
    name = os.path.join("noise2dCV", name, str(index))
    os.makedirs(name, exist_ok=True)    
    datagen = lambda: comb_loss.gendata(signoise=signoise, n=100)

    experiment(paramsDict,
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               printreport=False, 
               resume=True,
               niterations=niterations)

