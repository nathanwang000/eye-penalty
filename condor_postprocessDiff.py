import numpy as np
import sys, os, shutil, sys

# same as condor_postprocess just don't collect theta
def collect(expName, resume=True):

    for fn in os.listdir(expName):
        fpath = os.path.join(expName, fn)
        if os.path.isdir(fpath) and fn != '0':
            for result in os.listdir(fpath):
                if result.startswith("result_"):

                    tdir = os.path.join(expName, '0', result)
                    os.makedirs(tdir, exist_ok=True)

                    index = -1
                    if resume:
                        for logn in os.listdir(tdir):
                            if logn.startswith("log_"):
                                index = max(index, int(logn[4:]))

                    
                    logpath = os.path.join(fpath, result)

                    # collect log
                    for logn in os.listdir(logpath):
                            
                        # collect log
                        if logn.startswith("log_"):
                            targetpath = os.path.join(expName, '0', result,
                                                      "log_"+str(index+1))
                            srcpath = os.path.join(logpath, logn)
                            shutil.move(srcpath, targetpath)
                            index += 1

collect(sys.argv[1])
