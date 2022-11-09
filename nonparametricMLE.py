"""
The nonparametric maximum likelihood estimator.

Miriam Ayer et al. An empirical distribution function for sampling with incomplete information. The Annals of Mathematical Statistics 26(4)641-647, 1955.

Piet Groeneboom and Geurt Jongbloed. Nonparametric Estimation under Shape Constraints. Cambridge University Press. New York, 2014.

Original code by Piet Groeneboom and Geurt Jongbloed. Translated from R to Python by Dennis Christensen.

"""

### Import
import numpy as np


# Methods
def convexmin(m, cumw, cs, y):
    y[0] = cs[0]/cumw[0]
   
    for i in range(1, m):
        y[i] = (cs[i] - cs[i-1]) / (cumw[i] - cumw[i-1])
        if y[i-1] > y[i]:
            j = i
            while y[j-1] > y[i] and j > 0:
                j = j-1
                if j > 1:
                    y[i] = (cs[i] - cs[j-1]) / (cumw[i] - cumw[j-1])
                else:
                    y[i] = cs[i] / cumw[i]
                for n in range(j,i):
                    y[n] = y[i]
    return y


def get_nonparametric_mle(t, d):
    s = np.argsort(t)
    ts = t[s]
    ds = d[s]
    n = len(t)
   
    cumw = np.arange(1, n+1)
    cs = np.cumsum(ds)
    y = np.zeros(n)
    y = convexmin(n, cumw, cs, y)
    cumy = np.cumsum(y)
   
    # MLE
   
    j = 0
    if y[0] > 0:
        j = 1
    for i in range(1, n):
        if y[i] > y[i-1]:
            j += 1
    m = j
   
    u = np.zeros(m)
    v = np.zeros(m)
   
    j = 0
    if y[0] > 0:
        j = 1
        u[j-1] = ts[0]
        v[j-1] = y[0]
       
    for i in range(1, n):
        if y[i] > y[i-1]:
            j += 1
            u[j-1] = ts[i]
            v[j-1] = y[i]
    return ts, y
