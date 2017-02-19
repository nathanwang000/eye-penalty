import numpy as np
from numpy import array
import json
import matplotlib.pyplot as plt
from os import listdir, path
import collections
import math

dirname = 'val'
regs = ['lasso', 'ridge', 'enet', # 'penalty',
        'wlasso', 'wridge', 'owl', 'eye']

def extract(record):
    def extractRecord(item):
        return list(map(lambda e: e[item],record))
    return extractRecord

def plotResult(fn, title=None):
    record = json.load(open(fn))
    
    extractRecord = extract(record)
    valloss = extractRecord('validation/main/loss')
    trloss = extractRecord('main/loss')
    valacc = extractRecord('validation/main/accuracy')
    tracc = extractRecord('main/accuracy')
    
    # loss
    plt.plot(valloss, label='val loss')
    plt.plot(trloss, label='tr loss')
    plt.legend()
    if title: plt.title(title + " loss")
    else: plt.title("loss")
    plt.savefig(path.join(path.dirname(fn),"loss_plot.png"))
    plt.clf()

    # acc
    plt.plot(valacc, label='val acc')
    plt.plot(tracc, label='tr acc')
    plt.legend()
    if title: plt.title(title + " accuracy")
    else: plt.title("acurracy")
    plt.savefig(path.join(path.dirname(fn),"acc_plot.png"))
    plt.clf()

# main
def genplots():
    for fn in listdir(dirname):
        if not path.isdir(path.join(dirname, fn)) or\
           fn.startswith("montage"): continue
        param_index = fn.find('(')
        method = fn[:param_index]
        args = eval(fn[param_index:])
        print(method, args)
        plotResult(path.join(dirname, fn, 'log'), fn)


def returnBest(methods=regs, criteria='validation/main/loss',
               minimize = True,
               report=['validation/main/loss',
                       'validation/main/accuracy',
                       'validation/main/auroc']):
    if type(methods) is str: methods = [methods]
    if criteria in report:
        index = report.index(criteria)
        report = report[:index] + report[index+1:]
    D = {}
    for fn in listdir(dirname):
        if not path.isdir(path.join(dirname, fn)) or\
           fn.startswith("montage"): continue
        param_index = fn.find('(')
        method = fn[:param_index]
        if method not in methods: continue
        args = eval(fn[param_index:])
        record = json.load(open(path.join(dirname, fn, 'log')))
        extractRecord = extract(record)
        key = extractRecord(criteria)
        try:
            print(args, key[-1], *[extractRecord(r)[-1]\
                                   for r in report])
        except:
            continue # key not exist
        if (math.isnan(key[-1]) or math.isinf(key[-1])): continue
        if D.get(method) is None:
            D[method] = (args, key[-1], *[extractRecord(r)[-1]
                                          for r in report])
        else:
            if (minimize and D[method][1] > key[-1]) or\
               (not minimize and D[method][1] < key[-1]):
                D[method] = (args, key[-1],
                             *[extractRecord(r)[-1]
                               for r in report])
    return D


