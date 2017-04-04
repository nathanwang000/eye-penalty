import numpy as np
import os
import comb_loss
import plotResult
from scipy.linalg import block_diag

def experiment(paramsDict, datagen, num_runs=100,
               basedir_prefix="", niterations=1000,
               printreport=False, resume=False):
    def run_with_reg_wrapper(num_runs):
        def _f(*args,**kwargs):
            return comb_loss.run_with_reg(*args, **kwargs,
                                          num_runs=num_runs,
                                          printreport=printreport,
                                          resume=resume,
                                          datagen=datagen,
                                          niterations=niterations)
        return _f
    run = run_with_reg_wrapper(num_runs)
    # map regularization name to function
    r2f = {'lasso':   comb_loss.lasso, 
           'ridge':   comb_loss.ridge, 
           'enet':    comb_loss.enet, 
           'penalty': comb_loss.penalty,        
           'wlasso':  comb_loss.weightedLasso, 
           'wridge':  comb_loss.weightedRidge, 
           'owl':     comb_loss.OWL, 
           'eye':     comb_loss.eye}
    # actual run
    for method in paramsDict:
        args = paramsDict[method]
        ####function(args)#####,directory to save
        outdir = os.path.join(basedir_prefix, "result_" + method)
        run(r2f[method](*args), outdir)


############################################### user funtions (all are outdated)
'''
def example():
    name = "example"
    gridSearch = comb_loss.paramsSearch2d
    datagen = lambda: comb_loss.gendata(signoise=0.2)
    basedir = os.path.join(name, 'val')
    risk = comb_loss.generate_risk(0, 0, 0, '2d')
    
    # pipeline for training
    gridSearch(datagen=datagen, risk=risk, basedir=basedir)
    experiment(plotResult.returnBest(dirname=basedir),
               datagen=datagen,
               num_runs=30,
               basedir_prefix=name)


def noise2d(): # torun
    name = "noise2d"
    gridSearch = comb_loss.paramsSearch2d
    risk = comb_loss.generate_risk(0, 0, 0, '2d')

    for s in np.linspace(0,2,10):
        basedir = os.path.join(name, 'val%.2f' % s)
        datagen = lambda: comb_loss.gendata(signoise=s, n=100)
    
        # pipeline for training
        gridSearch(datagen=datagen, risk=risk, basedir=basedir)
        experiment(plotResult.returnBest(dirname=basedir),
                   num_runs=30, datagen=datagen,
                   basedir_prefix="noise2d/%.2f" % s)

def diffTheta(): # running
    name = "diff_theta"
    gridSearch = comb_loss.paramsSearchMd
    datagen, (theta, risk, nd) = comb_loss.genDiffTheta(n=5000)
    basedir = os.path.join(name, 'val')
    
    # pipeline for training
    gridSearch(datagen=datagen, risk=risk, basedir=basedir)
    experiment(plotResult.returnBest(criteria='validation/main/auroc',
                                     minimize=False,
                                     dirname=basedir),
               datagen=datagen,
               num_runs=30,
               basedir_prefix=name)

def sweepBinaryR(): # torun
    name="binaryR"
    nrgroups = 11
    nirgroups = nrgroups
    pergroup = 10
    n = 5000
    # setup
    gridSearch = comb_loss.paramsSearchMd
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "binary_r")
    basedir = os.path.join(name, 'val')
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*(nrgroups+nirgroups)))
    theta = np.zeros((nrgroups + nirgroups) * pergroup)
    theta[:nrgroups*pergroup] = 1
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)    

    # pipeline for training
    gridSearch(datagen=datagen, risk=risk, basedir=basedir)
    experiment(plotResult.returnBest(dirname=basedir),
               datagen=datagen,
               num_runs=30,
               basedir_prefix=name)

def sweepCov(): # torun
    name="corr"
    nrgroups = 10
    nirgroups = 0
    pergroup = 4
    n = 2000
    # setup
    gridSearch = comb_loss.paramsSearchMd
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "corr")
    basedir = os.path.join(name, 'val')
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
                                           n=n, signoise=10)

    # pipeline for training
    gridSearch(datagen=datagen, risk=risk, basedir=basedir)
    experiment(plotResult.returnBest(dirname=basedir),
               datagen=datagen,
               num_runs=30,
               basedir_prefix=name)

def sweepFracR(): # torun
    name = "fracR"
    nrgroups = 12
    nirgroups = 0
    pergroup = 10
    n = 2000
    # setup
    gridSearch = comb_loss.paramsSearchMd
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "frac_r")
    basedir = os.path.join(name, 'val')
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*nrgroups))
    theta = np.ones(nrgroups*pergroup)
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)    
    # pipeline for training
    gridSearch(datagen=datagen, risk=risk, basedir=basedir)
    experiment(plotResult.returnBest(dirname=basedir),
               datagen=datagen,
               num_runs=30,
               basedir_prefix=name)

def sweepFracRnormalized(): # running
    name = "fracRN"
    nrgroups = 12
    nirgroups = 0
    pergroup = 10
    n = 2000
    # setup
    gridSearch = comb_loss.paramsSearchMd
    risk = comb_loss.generate_risk(nrgroups, nirgroups, pergroup, "frac_r")
    # normalize risk
    for i in range(0, risk.size, pergroup):
        risk[i:i+pergroup] /= risk[i:i+pergroup].sum()
    # normalization done
    basedir = os.path.join(name, 'val')
    # gen data
    base = np.diag(np.ones(pergroup))       
    base[base==0] = 0.99
    C = block_diag(*([base]*nrgroups))
    theta = np.ones(nrgroups*pergroup)
    datagen = lambda: comb_loss.genCovData(C=C, theta=theta,
                                           n=n, signoise=15)    
    # pipeline for training
    gridSearch(datagen=datagen, risk=risk, basedir=basedir)
    experiment(plotResult.returnBest(dirname=basedir),
               datagen=datagen,
               num_runs=30,
               basedir_prefix=name)
    

if __name__ == '__main__':
    #diffTheta()
    sweepFracRnormalized()        
    # sweepBinaryR()
    # sweepFracR()    
    # sweepCov()
'''
