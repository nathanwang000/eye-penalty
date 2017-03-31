import numpy as np
import matplotlib.pyplot as plt

a = -4
b = 4
xlist = np.linspace(a, b, 100)
ylist = np.linspace(a, b, 100)
X, Y = np.meshgrid(xlist, ylist)

def drawContour(Z, name, levels=None,
                save=False, c=None, useLabel=True):
    if levels:
        if c:
            cp = plt.contour(X, Y, Z, levels, colors=c)
        else:
            cp = plt.contour(X, Y, Z, levels)            
    else:
        if c:
            cp = plt.contour(X, Y, Z, colors=c)
        else:
            cp = plt.contour(X, Y, Z)
    if useLabel: plt.clabel(cp, inline=True, fontsize=10)
    plt.title(name + ' Contour Plot')
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    if save:
        plt.savefig('contour/'+name+'.png')
    else: plt.show()

alpha = 1
l1_ratio = 0.5

Zridge = 0.5 * alpha * (X**2 + Y**2)
Zlasso = alpha * (abs(X) + abs(Y))
Zenet = alpha * (l1_ratio * (abs(X) + abs(Y)) +
                 0.5 * (1-l1_ratio) * (X**2 + Y**2))

w0 = 1; w1 = 2 # assume x known y unkown
Zwlasso = alpha * (w0 * abs(X) + w1 * abs(Y))
Zwridge = 0.5 * alpha * (w0 * X**2 + (w1 * Y)**2)

def Zpenalty(l1_ratio=l1_ratio):
    return 2 * alpha * (l1_ratio * abs(Y) +
                        0.5 * (1-l1_ratio) * X**2)

def ZOWL(w1, w2):
    if w1 < w2: w1, w2 = w2, w1
    return w1 * np.where(abs(X)>abs(Y), abs(X), abs(Y)) +\
        w2 * np.where(abs(X)<=abs(Y), abs(X), abs(Y))

def Zeye(r1=1, r2=0, l1_ratio=l1_ratio):
    # assert 0 <= r1 <= 1 and 0 <= r2 <=1, "invalid r1 r2"
    def solveQuadratic(a, b, c):
        return (-b + np.sqrt(b**2-4*c*a)) / (2*a)
    if l1_ratio == 0 or l1_ratio == 1:
        return Zpenalty(l1_ratio)
    b = 2 * l1_ratio * (abs((1-r1) * X) + abs((1-r2) * Y))
    a = (1-l1_ratio) * ((r1 * X)**2 + (r2 * Y)**2) 
    c = l1_ratio**2 / (1-l1_ratio)
    if r1 == 0 and r2 == 0: return b/c
    return alpha * 1 / solveQuadratic(a, b, -c)

def Zorthog(r=0.5, l=1, n=1, C=20, D=20):
    Z = np.sqrt((C+(1-r)*abs(Y))**2+D+r*Y**2)
    return X / (1+n*l*r**2/Z) * np.maximum(0, 1-n*l*(1-r)*(1+\
           (C+(1-r)*abs(Y))/Z) / abs(X)) - Y

NUM_COLORS = 10
cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

levels = [3]#[1,2,3,4,5,6,7,8,9,10]

# drawContour(Zorthog(0.5, C=20, D=20), "orthog", levels)
# drawContour(Zorthog(0.9), "orthog", levels)
# drawContour(Zorthog(0.9), "orthog", levels)
    
# i=0
# for l in np.arange(0.1,1,0.1):
#     l1_ratio = l
#     drawContour(Zeye(l1_ratio=l1_ratio),
#                 "eye", levels, c=[colors[i%len(colors)]])
#     i+=1
# plt.show()    

# drawContour(Zeye(0, 0), "eye_lasso", levels) # same as lasso
# drawContour(Zeye(1, 1), "eye_ridge", levels)  # same as wridge
# drawContour(Zeye(0.5,0.5), "eye_enet", levels) # same as enet
# drawContour(Zlasso, "lasso", levels)
# drawContour(Zridge, "ridge", levels)
# drawContour(Zenet, 'enet', levels)
# drawContour(Zwlasso, 'wlasso', levels)
# drawContour(Zwridge, 'wridge', levels)
# drawContour(Zpenalty(), 'penalty', levels)
# drawContour(ZOWL(2,1), 'OWL_w1=2>w2=1', levels)
# drawContour(ZOWL(2,0), 'OWL_w1=2>w2=0', levels)
# drawContour(ZOWL(1,1), 'OWL_w1=1=w2=1', levels)

# a case in point x1 = 2 x0, so x1 + 2 x0 = c
# Zadd = X + 2 * Y # this is the objective
# drawContour(Zadd, "add", [4.69])
# drawContour(Zlasso, "lasso_add", levels, save=True)
# drawContour(Zridge, "ridge_add", levels, save=True)
# drawContour(Zenet, 'enet_add', levels, save=True)
# drawContour(Zwlasso, 'wlasso_add', levels, save=True)
# drawContour(Zwridge, 'wridge_add', levels, save=True)
# drawContour(Zpenalty(), 'penalty_add', levels, save=True)

# for r1 in np.arange(0,1.01,0.1):
#     i = 0
#     for r2 in np.arange(0,r1+0.01,0.1):
#         drawContour(Zeye(r1,r2), "eye_fraction", levels,
#                     c=[colors[i%len(colors)]])
#         print(r1, r2)
#         i += 1
#     plt.show()

# plt.show()

# r1 = 1
# i = 0
# for r2 in np.arange(0,r1+0.01,0.1):
#     drawContour(Zeye(r1,r2), "eye_fraction", levels,
#                 c=[colors[i%len(colors)]])
#     print(r1, r2)
#     i += 1

# plt.show()
