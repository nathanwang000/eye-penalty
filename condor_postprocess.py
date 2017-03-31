import numpy as np
import sys, os, shutil, sys

# maybe too many io in this function, optimize it if needed
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

                    theta = None
                    thetafn = os.path.join(expName, '0', result, "theta.npy")
                    if resume and os.path.isfile(thetafn):
                        theta = np.load(thetafn)
                    
                    logpath = os.path.join(fpath, result)

                    # collect log and theta
                    tmp_thetafn = os.path.join(logpath, "theta.npy")
                    if not os.path.isfile(tmp_thetafn): continue
                    # collect theta
                    if theta is not None:
                        theta = np.vstack((theta, np.load(tmp_thetafn)))
                    else:
                        theta = np.load(tmp_thetafn)
                    os.unlink(tmp_thetafn)
                    
                    for logn in os.listdir(logpath):
                            
                        # collect log
                        if logn.startswith("log_"):
                            targetpath = os.path.join(expName, '0', result,
                                                      "log_"+str(index+1))
                            srcpath = os.path.join(logpath, logn)
                            shutil.move(srcpath, targetpath)
                            index += 1

                    # save theta
                    if theta is not None: np.save(thetafn, theta)
                    print(theta.shape)

collect(sys.argv[1])
