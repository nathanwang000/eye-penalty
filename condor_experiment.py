from experiment import experiment
import numpy as np
import comb_loss, sys, os

# should avoid race condition
def noise2d(index, regs=None, niterations=5000, signoise=0, alpha=0.01, name="default"):
    name = os.path.join("noise2d", name, str(index))
    os.makedirs(name, exist_ok=True)    
    datagen = lambda: comb_loss.gendata(signoise=signoise, n=100)
    risk = comb_loss.generate_risk(0, 0, 0, '2d')

    w0 = 1-risk
    w1 = 2-risk # penalize unknown more

    paramAtoms = [
        ('eye', (risk, alpha)),
        ('wlasso', (w1, alpha)),
        ('wridge', (w1, alpha)),
        ('penalty', (risk, alpha, 0.4)),
        ('owl', ([2,1], alpha)),
        ('lasso', (alpha,)),
        ('enet', (alpha, 0.2))
    ]

    if regs: paramAtoms = list(filter(lambda a: a[0] in regs, paramAtoms))
    experiment(paramAtoms,
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               printreport=False, 
               resume=True,
               niterations=niterations)

def diffTheta(index, niterations=1000, name="default"):
    name = os.path.join("diff_theta/", name, str(index))
    os.makedirs(name, exist_ok=True)
    datagen, (theta, risk, nd) = comb_loss.genDiffTheta(n=5000)
    
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
        ('penalty', (risk, 0.01, 0.4)),
        ('owl', (owl1, 0.01)),
        ('lasso', (0.01,)),
        ('enet', (0.01, 0.2))
    ]
    experiment(paramAtoms,
               datagen=datagen,
               num_runs=1,
               basedir_prefix=name,
               niterations=niterations)

if __name__ == '__main__':
    # sweep noise level
    # for s in np.linspace(0,2,10):
    #     noise2d(sys.argv[1], signoise=s, name="noise%.2f" % s)

    # noise2d(sys.argv[1], alpha=1e-3, niterations=15000, name="alpha0.001")
    # noise2d(sys.argv[1], alpha=1e-4, niterations=30000, name="alpha0.0001")    
    diffTheta(sys.argv[1])
    
