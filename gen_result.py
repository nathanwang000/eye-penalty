import os
import numpy as np
import matplotlib.pyplot as plt

regs = ['eye','wlasso', 'wridge',
        'lasso', 'ridge',
        'owl', 'enet'] # penalty

NUM_COLORS = len(regs)
cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

def loadData(method):
    data = np.load('result_'+method+'/theta.npy')
    return data[~np.isnan(data).any(axis=1)]

nbins = 50

def plotFeature(f_num, nbins=50, savefig=False):
    for i, reg in enumerate(regs):
        data = loadData(reg)
        plt.hist(data[:,f_num], nbins, alpha=0.5,
                 label=reg, color=colors[i])
        tstr = "$x_%d$ distribution" % f_num
        plt.title(tstr)
        plt.legend()
    if savefig: plt.savefig('figures/'+tstr.replace(" ","_")+".png")
    else: plt.show()

# plotFeature(0) # x0
# plotFeature(1) # x1
# plotFeature(2) # b

# nfeatures for x0 x1 b is 3
def plotBar(n_features, savefig=False): 
    rects = []
    means = []
    stds  = []
    # fill in means and stds
    for reg in regs:
        data = loadData(reg)
        means.append([np.mean(data[:, i]) for i \
                      in range(n_features)])
        stds.append([np.std(data[:, i]) for i \
                     in range(n_features)])
    # general setting
    fig, ax = plt.subplots()
    index = np.arange(n_features)
    bar_width = 1.0 / (len(regs)+1)
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    # fill in rects
    for i, reg in enumerate(regs):
        bar = plt.bar(index + i * bar_width,
                      means[i], bar_width,
                      color=colors[i],
                      alpha=opacity,
                      yerr=stds[i],
                      error_kw=error_config,
                      label=reg)
        rects.append(bar)
    plt.xlabel('features')
    plt.ylabel('weights')
    plt.title('weights by regularization method')
    plt.xticks(index + (len(regs)/2)*bar_width,
               tuple('$x_%d$' % i for i in range(n_features)))
    plt.legend()
    plt.tight_layout()
    if savefig: plt.savefig("figures/avg_reg.png")
    else: plt.show()
    plt.clf()
    
# plotBar(3, True)

################### for nd case #################
from comb_loss import reportKL, reportTheta, generate_risk

EXPERIMENT_SETTINGS = {
    "binary_r": (11,11,10),
    "corr": (10,0,4),
    "frac_r": (12,0,10)
}

def gen_nd_loss_csv(regs=regs, fn=None, stats="mean", method='kl',
                    experiment="binary_r", basedir=None):
    basedir = basedir or "./"
    if not fn: fn="tmp_"+stats+".csv"
    from itertools import zip_longest
    nrgroups, nirgroups, pergroup = EXPERIMENT_SETTINGS[experiment]
    
    f = {"mean": lambda x: x.mean(),
         "var": lambda x: x.var()
    }[stats]
    
    lines = [["method"]+regs]
    iterables = tuple(reportKL(os.path.join(basedir, "result_"+reg),
                               f=f, method=method,
                               nrgroups=nrgroups,
                               nirgroups=nirgroups,
                               pergroup=pergroup,
                               experiment=experiment)
                      for reg in regs)
    prevtag = "relevant"
    count = 0
    for items in zip_longest(*iterables):
        tag = items[0][0]
        if prevtag != tag:
            count = 0
            prevtag = tag
        tag = tag + str(count)
        lines.append([tag]+[str(l) for _,l in items])
        count += 1
    # print all these lines
    with open(fn,'w') as f:
        f.write("\n".join([",".join(l) for l in lines]))

def gen_nd_weight_csv(regs=regs, fn=None, experiment="binary_r",
                      basedir=None):
    basedir = basedir or "./"    
    if not fn: fn="tmp_weights.csv"
    from itertools import zip_longest
    nrgroups, nirgroups, pergroup = EXPERIMENT_SETTINGS[experiment]
    
    f = lambda x: x.mean()

    lines = [["method"]+regs]  
    iterables = tuple(reportTheta(os.path.join(basedir, "result_"+reg),
                                  f=f,
                                  nrgroups=nrgroups,
                                  nirgroups=nirgroups,
                                  pergroup=pergroup,
                                  experiment=experiment)
                      for reg in regs)
    prevtag = "relevant"
    count = 0
    for items in zip_longest(*iterables):
        tag = items[0][0]
        if prevtag != tag:
            count = 0
            prevtag = tag
        tag = tag + str(count)
        lines.append([tag]+[str(l) for _,l in items])
        count += 1
    # print all these lines
    with open(fn,'w') as f:
        f.write("\n".join([",".join(l) for l in lines]))

def gen_nd_weight_var_csv(regs=regs, fn=None,  basedir=None,
                          f=lambda x: x.mean(1).var(),
                          experiment="binary_r"):

    basedir = basedir or "./"    
    if not fn: fn="tmp_weights_var.csv"
    from itertools import zip_longest
    nrgroups, nirgroups, pergroup = EXPERIMENT_SETTINGS[experiment]

    lines = [["method"]+regs]
    iterables = tuple(reportTheta(os.path.join(basedir, "result_"+reg),
                                  f=f,
                                  nrgroups=nrgroups,
                                  nirgroups=nirgroups,
                                  pergroup=pergroup,
                                  experiment=experiment) 
                      for reg in regs)
    prevtag = "relevant"
    count = 0
    for items in zip_longest(*iterables):
        tag = items[0][0]
        if prevtag != tag:
            count = 0
            prevtag = tag
        tag = tag + str(count)
        lines.append([tag]+[str(l) for _,l in items])
        count += 1
    # print all these lines
    with open(fn,'w') as f:
        f.write("\n".join([",".join(l) for l in lines]))
        
# sweep binary r = {nrgroups=11, nirgroups=11, pergroup=10}
# sweep corr = {nrgroups=10, nirgroups=0, pergroup=4}
# sweep frac r = {nrgroups=12, nirgroups=0, pergroup=10}
def reportNdLoss(regs=regs,tag="relevant", method="kl", basedir=None,
                 experiment="binary_r"):
    nrgroups, nirgroups, pergroup = EXPERIMENT_SETTINGS[experiment]
    basedir = basedir or "./"
    # method can be kl or emd
    if not tag: tagf=lambda x: True
    else: tagf=lambda x: tag==x
    for reg in regs:
        fn_path = os.path.join(basedir, "result_"+reg)
        g = reportKL(fn_path, f=lambda x:x.mean(), method=method,
                     nrgroups=nrgroups, nirgroups=nirgroups,
                     pergroup=pergroup, experiment=experiment)
        print(reg+":", sum(list(map(lambda x: x[1],
                                    filter(lambda x: tagf(x[0]), g)))))

