import numpy as np
import matplotlib.pyplot as plt

regs = ['wlasso', 'eye', 'wridge', 'penalty', 'lasso', 'ridge',
        'owl', 'enet']

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
    plt.clf()

# plotFeature(0, savefig=True) # x0
# plotFeature(1, savefig=True) # x1
# plotFeature(2, savefig=True) # b

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
