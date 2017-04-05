import comb_loss, sys, math, os
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
    alpha = (1e-1, 1e-2, 5e-3, 1e-3, 5e-4)
    beta = tuple(i/10 for i in range(11))

    params_cand = [
        ('lasso',   (alpha,)),
        ('ridge',   (alpha,)),
        ('enet',    (alpha, beta)),
        ('penalty', ((risk,), alpha, beta)),
        ('wlasso',  ((w1,), alpha)),
        ('wridge',  ((w1,), alpha)),
        ('owl',     ((owl1, ), alpha)),
        ('eye',     ((risk,), alpha))
    ]

    return [(m, arg) for m, args in params_cand for arg in product(*args)]

# user functions
def noise2d(index, numprocess, paramAtoms,
            niterations=5000, signoise=0, name="default"):
    name = os.path.join("noise2dCV", name)
    os.makedirs(name, exist_ok=True)    
    datagen = lambda: comb_loss.gendata(signoise=signoise, n=100)

    mytask = math.ceil(len(paramAtoms) / numprocess)

    experiment(paramAtoms[index*mytask: (index+1)*mytask],
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               namebases=list(range(index*mytask, (index+1)*mytask)),
               printreport=False, 
               resume=True,
               niterations=niterations)

if __name__ == '__main__':
    pid = int(sys.argv[1])
    numprocess = int(sys.argv[2])

    atoms2d = paramAtoms2d()
    noise2d(pid, numprocess, atoms2d)
