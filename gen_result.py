import numpy as np
import matplotlib.pyplot as plt

def loadData(method):
    data = np.load('result_'+method+'/theta.npy')
    return data[~np.isnan(data).any(axis=1)]

regs = ['penalty', 'lasso', 'ridge', 'wlasso',
        'wridge', 'owl', 'enet']

nbins = 50

def plotFeature(f_num, nbins=50, savefig=False):
    for reg in regs:
        data = loadData(reg)
        plt.hist(data[:,f_num], nbins, alpha=0.5, label=reg)
        tstr = "$x_%d$ distribution" % f_num
        plt.title(tstr)
        plt.legend()
    if savefig: plt.savefig('figures/'+tstr+".png")
    else: plt.show()

plotFeature(0) # x0
plotFeature(1) # x1
plotFeature(2) # b

# lass = loadData('lasso')
# pena = loadData('penalty')
# ridg = loadData('ridge')
# enet = loadData('enet')

# nfeatures for x0 x1 b is 3
def plotBar(n_features, savefig=False): 
    rects = []
    means = []
    stds  = []
    colors = ['r','g','b','k', 'purple','pink','orange']
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

plotBar(3)
# # avg of all distribution
# n_groups = 3
# lassMean = [np.mean(lass[:,0]),
#             np.mean(lass[:,1]),
#             np.mean(lass[:,2])]
# ridgMean = [np.mean(ridg[:,0]),
#             np.mean(ridg[:,1]),
#             np.mean(ridg[:,2])]
# enetMean = [np.mean(enet[:,0]),
#             np.mean(enet[:,1]),
#             np.mean(enet[:,2])]
# penaMean = [np.mean(pena[:,0]),
#             np.mean(pena[:,1]),
#             np.mean(pena[:,2])]
# lassStd = [np.std(lass[:,0]),
#            np.std(lass[:,1]),
#            np.std(lass[:,2])]
# ridgStd = [np.std(ridg[:,0]),
#            np.std(ridg[:,1]),
#            np.std(ridg[:,2])]
# enetStd = [np.std(enet[:,0]),
#            np.std(enet[:,1]),
#            np.std(enet[:,2])]
# penaStd = [np.std(pena[:,0]),
#            np.std(pena[:,1]),
#            np.std(pena[:,2])]

# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.2

# opacity = 0.4
# error_config = {'ecolor': '0.3'}

# lassRects = plt.bar(index, lassMean, bar_width,
#                     alpha=opacity,
#                     color='b',
#                     yerr=lassStd,
#                     error_kw=error_config,
#                     label='lass')
# ridgRects = plt.bar(index + bar_width, ridgMean, bar_width,
#                     alpha=opacity,
#                     color='r',
#                     yerr=ridgStd,
#                     error_kw=error_config,
#                     label='ridg')
# enetRects = plt.bar(index + 2*bar_width, enetMean, bar_width,
#                     alpha=opacity,
#                     color='g',
#                     yerr=enetStd,
#                     error_kw=error_config,
#                     label='enet')
# penaRects = plt.bar(index + 3*bar_width, penaMean, bar_width,
#                     alpha=opacity,
#                     color='purple',
#                     yerr=penaStd,
#                     error_kw=error_config,
#                     label='pena')


# plt.xlabel('features')
# plt.ylabel('weights')
# plt.title('weights by regularization method')
# plt.xticks(index + 2*bar_width, ('$x_0$', '$x_1$', 'b'))
# plt.legend()

# plt.tight_layout()
# plt.savefig("figures/avg_reg.png")

# tmp
# n_groups = 3
# lassBudget = lass[:,0] + 2 * lass[:,1]
# ridgBudget = ridg[:,0] + 2 * ridg[:,1]
# enetBudget = enet[:,0] + 2 * enet[:,1]
# penaBudget = pena[:,0] + 2 * pena[:,1]

# lassMean = [np.mean(lass[:,0] / lassBudget),
#             np.mean(lass[:,1] / lassBudget),
#             np.mean(lass[:,2])]
# ridgMean = [np.mean(ridg[:,0] / ridgBudget),
#             np.mean(ridg[:,1] / ridgBudget),
#             np.mean(ridg[:,2])]
# enetMean = [np.mean(enet[:,0] / enetBudget),
#             np.mean(enet[:,1] / enetBudget),
#             np.mean(enet[:,2])]
# penaMean = [np.mean(pena[:,0] / penaBudget),
#             np.mean(pena[:,1] / penaBudget),
#             np.mean(pena[:,2])]
# lassStd = [np.std(lass[:,0]),
#            np.std(lass[:,1]),
#            np.std(lass[:,2])]
# ridgStd = [np.std(ridg[:,0]),
#            np.std(ridg[:,1]),
#            np.std(ridg[:,2])]
# enetStd = [np.std(enet[:,0]),
#            np.std(enet[:,1]),
#            np.std(enet[:,2])]
# penaStd = [np.std(pena[:,0]),
#            np.std(pena[:,1]),
#            np.std(pena[:,2])]

# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.2

# opacity = 0.4
# error_config = {'ecolor': '0.3'}

# lassRects = plt.bar(index, lassMean, bar_width,
#                     alpha=opacity,
#                     color='b',
#                     yerr=lassStd,
#                     error_kw=error_config,
#                     label='lass')
# ridgRects = plt.bar(index + bar_width, ridgMean, bar_width,
#                     alpha=opacity,
#                     color='r',
#                     yerr=ridgStd,
#                     error_kw=error_config,
#                     label='ridg')
# enetRects = plt.bar(index + 2*bar_width, enetMean, bar_width,
#                     alpha=opacity,
#                     color='g',
#                     yerr=enetStd,
#                     error_kw=error_config,
#                     label='enet')
# penaRects = plt.bar(index + 3*bar_width, penaMean, bar_width,
#                     alpha=opacity,
#                     color='purple',
#                     yerr=penaStd,
#                     error_kw=error_config,
#                     label='pena')


# plt.xlabel('features')
# plt.ylabel('weights')
# plt.title('weights by regularization method')
# plt.xticks(index + 2*bar_width, ('$x_0$', '$x_1$', 'b'))
# plt.legend()

# plt.tight_layout()
# plt.show()
