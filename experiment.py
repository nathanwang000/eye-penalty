import numpy as np
import comb_loss
import plotResult

def experiment(paramsDict, datagen, num_runs=100, basedir_prefix="result"):
    def run_with_reg_wrapper(num_runs):
        def _f(*args,**kwargs):
            return comb_loss.run_with_reg(*args, **kwargs,
                                          num_runs=num_runs,
                                          printreport=False,
                                          resume=True,
                                          datagen=datagen)
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
        args = paramsDict[method][0]
        ####function(args)#####,directory to save
        run(r2f[method](*args), basedir_prefix + "_" + method)

### user funtions
def example():
    gridSearch = comb_loss.paramsSearch2d
    datagen = lambda: comb_loss.gendata(signoise=0.2)
    basedir = 'val'
    
    # pipeline for training
    gridSearch(datagen=datagen, basedir=basedir)
    experiment(plotResult.returnBest(), num_runs=30)
    
def noise2d():
    gridSearch = comb_loss.paramsSearch2d
    
    for s in np.linspace(0,2,10):
        basedir = 'val' + str(s)        
        datagen = lambda: comb_loss.gendata(signoise=s, n=100)
    
        # pipeline for training
        gridSearch(datagen=datagen, basedir=basedir)
        experiment(plotResult.returnBest(dirname=basedir),
                   num_runs=30, datagen=datagen)

if __name__ == '__main__':
    noise2d()
