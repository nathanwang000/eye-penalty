import numpy as np
import matplotlib.pyplot as plt

a = -10
b = 10
xlist = np.linspace(a, b, 100)
ylist = np.linspace(a, b, 100)
X, Y = np.meshgrid(xlist, ylist)

def drawContour(Z, name, levels=None, save=False):
    if levels:
        cp = plt.contour(X, Y, Z, levels)
    else:
        cp = plt.contour(X, Y, Z)
    plt.clabel(cp, inline=True,
               fontsize=10)
    plt.title(name + ' Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig('contour/'+name+'.png')

alpha = 1
l1_ratio = 0.5

Zridge = 0.5 * alpha * (X**2 + Y**2)
Zlasso = alpha * (abs(X) + abs(Y))
Zenet = alpha * (l1_ratio * (abs(X) + abs(Y)) +
                 0.5 * (1-l1_ratio) * (X**2 + Y**2))

w0 = 1; w1 = 2 # assume x known y unkown
Zwlasso = alpha * (w0 * abs(X) + w1 * abs(Y))
Zwridge = 0.5 * alpha * (w0 * X**2 + (w1 * Y)**2)

Zpenalty = 2 *alpha * (l1_ratio * abs(Y) +
                    0.5 * (1-l1_ratio) * X**2)
def ZOWL(w1, w2):
    if w1 < w2: w1, w2 = w2, w1
    return w1 * np.where(abs(X)>abs(Y), abs(X), abs(Y)) +\
        w2 * np.where(abs(X)<=abs(Y), abs(X), abs(Y))

def Zeye(X, Y):
    def solveQuadratic(a, b, c):
        return (-b + np.sqrt(b**2-4*c*a)) / (2*a)
    if l1_ratio == 0 or l1_ratio == 1:
        return Zpenalty
    b = l1_ratio * abs(Y)
    a = 0.5 * (1-l1_ratio) * X**2
    c = alpha * l1_ratio**2 / (1-l1_ratio)
    return 1 / solveQuadratic(a, b, -c)

levels = [1,2,3,4,5,6,7,8,9,10]
drawContour(Zeye(X, Y), "eye", levels)
# drawContour(Zlasso, "lasso", levels)
# drawContour(Zridge, "ridge", levels)
# drawContour(Zenet, 'enet', levels)
# drawContour(Zwlasso, 'wlasso', levels)
# drawContour(Zwridge, 'wridge', levels)
# drawContour(Zpenalty, 'penalty', levels)
# drawContour(ZOWL(2,1), 'OWL w1=2 > w2=1', levels)
# drawContour(ZOWL(2,0), 'OWL w1=2 > w2=0', levels)
# drawContour(ZOWL(1,1), 'OWL w1=1 = w2=1', levels)

# a case in point x1 = 2 x0, so x1 + 2 x0 = c
# Zadd = X + 2 * Y # this is the objective
# drawContour(Zadd, "add", [4.69])
# drawContour(Zlasso, "lasso_add", levels, save=True)
# drawContour(Zridge, "ridge_add", levels, save=True)
# drawContour(Zenet, 'enet_add', levels, save=True)
# drawContour(Zwlasso, 'wlasso_add', levels, save=True)
# drawContour(Zwridge, 'wridge_add', levels, save=True)
# drawContour(Zpenalty, 'penalty_add', levels, save=True)

# plt.show()