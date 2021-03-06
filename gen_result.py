import numpy as np
import matplotlib.pyplot as plt
import os, json
from plotResult import extract

regs = ['eye','wlasso', 'wridge',
        'lasso', #'ridge',
        'owl', 'enet'#, 'penalty'
]

NUM_COLORS = len(regs)
cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

def loadData(method, basedir="./"):
    fname = os.path.join(basedir, 'result_'+method+'/theta.npy')
    data = np.load(fname)
    return data[~np.isnan(data).any(axis=1)]

nbins = 50

def plotFeature(f_num, nbins=nbins, savefig=False, basedir="./", regs=regs):
    for i, reg in enumerate(regs):
        data = loadData(reg, basedir=basedir)
        plt.hist(data[:,f_num], nbins, alpha=0.5,
                 label=reg, color=colors[i])
        tstr = r"$\theta_%d$ distribution" % f_num
        plt.title(tstr)
        plt.legend()
    if savefig: plt.savefig('figures/'+tstr.replace(" ","_")+".png")
    else: plt.show()

# plotFeature(0) # x0
# plotFeature(1) # x1
# plotFeature(2) # b

def plotRatio(nbins=nbins, savefig=False, basedir="./", regs=regs):
    for i, reg in enumerate(regs):
        data = loadData(reg, basedir=basedir)
        plt.hist(np.abs(data[:,0] / data[:, 1])-1, nbins, alpha=0.5,
                 label=reg, color=colors[i])
        tstr = r"$\frac{\theta_0}{\theta_1}-1$ distribution"
        plt.title(tstr)
        plt.legend()
    if savefig: plt.savefig('figures/'+tstr.replace(" ","_")+".png")
    else: plt.show()

# nfeatures for x0 x1 b is 3
def plotBar(n_features, savefig=False, basedir="./", regs=regs): 
    rects = []
    means = []
    stds  = []
    # fill in means and stds
    for reg in regs:
        data = loadData(reg, basedir=basedir)
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

def plotPerformance(basedir="./", regs=regs, title=""):
    rects = []
    means = []
    stds  = []
    criteria = [
        "validation/main/loss",
        "validation/main/accuracy",
        "validation/main/auroc"
    ]

    # fill in means and stds
    for reg in regs:
        data = []
        dirname = os.path.join(basedir, "result_"+reg)
        for fn in os.listdir(dirname):
            if fn.startswith("log_"):
                record = json.load(open(os.path.join(dirname, fn)))
                extractRecord = extract(record)
                data.append([extractRecord(c)[-1] for c in criteria])
        old_data = np.array(data)
        data = old_data[np.isfinite(old_data).all(axis=1)]
        delta = old_data.shape[0] - data.shape[0]
        if delta > 0: print(reg, "contains %d/%d" % (delta, old_data.shape[0]), "nan")
        means.append([np.mean(data[:, i]) for i \
                      in range(len(criteria))])
        stds.append([np.std(data[:, i]) for i \
                     in range(len(criteria))])
    # general setting
    fig, ax = plt.subplots()
    index = np.arange(len(criteria))
    bar_width = 1.0 / (len(regs)+1)
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    # output real value
    format_title = "method\t" + ("%s\t" * len(criteria))
    print(format_title % tuple(criteria))
    for i, reg in enumerate(regs):
        format_str = "%s\t" + ("%.4f+-%.4f\t" * len(criteria))
        lol = [(means[i][j], stds[i][j]) for j in range(len(criteria))]
        content = [reg] + [item for sublist in lol for item in sublist]
        print(format_str % tuple(content))
    
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
    plt.xlabel('criteria')
    plt.ylabel('value')
    plt.title(title)
    plt.xticks(index + (len(regs)/2)*bar_width, criteria)
    plt.legend(bbox_to_anchor=(-0.2, 1))
    plt.tight_layout()
    plt.show()
    plt.clf()


################### for nd case #################
from comb_loss import reportKL, reportTheta

EXPERIMENTS = {
    #name        #nrgroup   #nirgroups    #pergroup
    '2d':        (0,        0,            0),
    'binary_r':  (11,       11,           10),
    'corr':      (10,       0,            4),
    'frac_r':    (12,       0,            10),
    'corr_more': (10,       0,            10),
    'corr_huge': (10,       0,            30)
}

def nd_kl_helper(regs=regs, fn='./', stats="mean", method='kl',
                 experiment="binary_r"):
    from itertools import zip_longest
    f = {"mean": lambda x: x.mean(),
         "var": lambda x: x.var()
    }[stats]

    lines = []
    labels = regs
    tags = []
    
    iterables = tuple(reportKL(os.path.join(fn, "result_"+reg),
                               f=f,
                               method=method,
                               nrgroups=EXPERIMENTS[experiment][0],
                               nirgroups=EXPERIMENTS[experiment][1],
                               pergroup=EXPERIMENTS[experiment][2],
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
        tags.append(tag)
        
        lines.append([l for _,l in items])
        count += 1
    return np.array(lines), labels, tags

def plot_nd_kl(experiment, regs=regs, fn="./", method='kl',
               name=None, xtickLabels=None, xlabel=None, upto=None):
    data, labels, tags = nd_kl_helper(fn=fn, stats="mean",
                                      experiment=experiment, regs=regs)
    var_data, _, _ = nd_kl_helper(fn=fn, stats="var",
                                  experiment=experiment, regs=regs)

    if upto:
        data = data[:upto]
        var_data = var_data[:upto]
        
    m, n = data.shape
    x = np.arange(m)

       
    linestyles = ["-o", ":v", "-*", "-s",  "--^", "-.D", "-1", "-+"]
    fig, ax = plt.subplots()
    import seaborn as sns
    
    for i in range(n):
        plt.errorbar(x, data[:,i], #yerr=np.sqrt(var_data[:,i]),
                     fmt=linestyles[i], markersize=12, lw=4)#, color=colors[i])
        if xtickLabels:
            plt.xticks(x, xtickLabels, rotation=-45, fontsize=20)
        plt.legend(labels, fontsize=15)
    plt.ylabel("symmetric kl", fontsize=25)
    if xlabel: plt.xlabel(xlabel, fontsize=25)

    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.despine(ax=ax, offset=0) 
    
    if name: plt.savefig("figures/%s.eps" % name, bbox_inches="tight")
    plt.show()
    
from comb_loss import generate_risk
def fracRfit(experiment, basedir, normalize=False):
    nrgroups, nirgroups, pergroup = EXPERIMENTS[experiment]
    t = np.load(os.path.join(basedir, "theta.npy"))[:,:-1] # exclude b
    r = generate_risk(nrgroups, nirgroups, pergroup, experiment)
    print(t.shape,t.mean(0).shape, r.shape, nrgroups, nirgroups, pergroup)
    
    for start in range(0,r.size,pergroup):
        plt.subplot(5, 5, start//pergroup + 1)
        plt.plot(r[start:start+pergroup], '-o', label="risk")
        plt.plot(t.mean(0)[start:start+pergroup], '-o',label="\theta")
        plt.axis("off")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()
        
    if normalize:
        normed_t = t.mean(0)
        for start in range(0,r.size,pergroup):
            normed_t[start:start+pergroup] -= np.min(normed_t[start:start+pergroup])
            normed_t[start:start+pergroup] /= np.max(normed_t[start:start+pergroup])
            
            # frac r
            for start in range(0,r.size,pergroup):
                plt.subplot(5, 5, start//pergroup + 1)
                l1 = plt.plot(r[start:start+pergroup], "-o", label="risk")
                l2 = plt.plot(normed_t[start:start+pergroup], "-o", label="\theta")
                plt.axis("off")
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.show()

##############old start################

def gen_nd_loss_csv(regs=regs, fn=None, stats="mean", method='kl'):
    if not fn: fn="tmp_"+stats+".csv"
    from itertools import zip_longest
    f = {"mean": lambda x: x.mean(),
         "var": lambda x: x.var()
    }[stats]
    
    lines = [["method"]+regs]
    iterables = tuple(reportKL("result_"+reg, f=f, method=method)
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

def gen_nd_weight_csv(regs=regs, fn=None):
    if not fn: fn="tmp_weights.csv"
    from itertools import zip_longest
    f = lambda x: x.mean()

    lines = [["method"]+regs]
    iterables = tuple(reportTheta("result_"+reg, f=f)
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

def gen_nd_weight_var_csv(regs=regs, fn=None,
                          f=lambda x: x.mean(1).var()):
    if not fn: fn="tmp_weights_var.csv"
    from itertools import zip_longest

    lines = [["method"]+regs]
    iterables = tuple(reportTheta("result_"+reg, f=f)
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
        

def reportNdLoss(regs=regs,tag=None, method="kl"):
    # method can be kl or emd
    if not tag: tagf=lambda x: True
    else: tagf=lambda x: tag==x
    for reg in regs:
        g = reportKL("result_"+reg, f=lambda x:x.mean(), method=method)
        print(reg+":", sum(list(map(lambda x: x[1],
                                    filter(lambda x: tagf(x[0]), g)))))
##############old end################
