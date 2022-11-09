## RLAPTRANS

# Import
import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb

# Methods
@nb.njit(nb.float64(nb.int64, nb.int64))
def numba_nCr(m, k):
    minK = min(k, m - k)
    maxK = max(k, m-k)
    result = 0
    for i in range(maxK+1, m+1):
        result += np.log(i)
    for i in range(1, minK + 1):
        result -= np.log(i)
    return np.exp(result)


@nb.njit(nb.float64[:](nb.int64,
                       nb.float64,
                       nb.int64,
                       nb.float64,
                       nb.float64,
                       nb.float64,
                       nb.float64,
                       nb.int64,
                       nb.int64,
                       nb.int64,
                       nb.int64))
def rlaptrans(n, sigma, H, U, tol=1e-7, x0=1.0, xinc=2.0, m=11, L=1, A=19, nburn=38):
    maxiter = 500
    
    # Derived quantiles
    nterms = nburn + m*L
    seqbtL = np.arange(nburn, nterms + L, L)
    y = np.pi * (1j) * np.arange(1, nterms + 1) / L
    expy = np.exp(y)
    A2L = .5 * A / L
    expxt = np.exp(A2L) / L
    coef = np.zeros(m+1)
    for j in range(m+1):
        coef[j] = numba_nCr(m, j) / 2**m
        
    # Generate random sample
    u = np.sort(np.random.random(n))
    xrand = u
    
    # Find upper bound
    t = x0/xinc
    cdf = 0
    kount0 = 0
    set1st = False
    while kount0 < maxiter and cdf < u[n-1]:
        t = xinc * t
        kount0 += 1
        x = A2L / t
        z = x + y/t
        
        # J ~ TS(1/H, sigma, U),
        # so E[exp(-sJ)] = exp[-1/H{(s + U)^sigma - U^sigma}]
        ltx = np.exp(-1/H*((x + U)**sigma - U**sigma))
        ltzexpy = np.exp(-1/H*((z + U)**sigma - U**sigma)) * expy
        
        parsum = .5*ltx.real + np.cumsum(ltzexpy.real)
        parsum2 = .5*(ltx/x).real + np.cumsum((ltzexpy/z).real)
        pdf = expxt * np.sum(coef * parsum[seqbtL - 1]) / t # CHECK INDEX
        cdf = expxt * np.sum(coef * parsum2[seqbtL - 1]) / t # CHECK INDEX
        if (not set1st) and cdf > u[0]:
            cdf1 = cdf
            pdf1 = pdf
            t1 = t
            set1st = True
    if kount0 >= maxiter:
        print('Cannot locate upper quantile')
        return np.zeros(n)
    
    upplim = t
    
    # Modified Newton-Raphson
    lower = 0
    t = t1
    cdf = cdf1
    pdf = pdf1
    kount = np.zeros(n)
    
    maxiter = 1_000
    
    
    for j in range(n): # So j ranges from 0 to n-1, unlike the R script
        # initial bracketing of solution
        upper = upplim
        
        kount[j] = 0
        while (kount[j] < maxiter) and (np.abs(u[j] - cdf) > tol):
            kount[j] += 1
            
            # Update t. Try Newton-Raphson approach. If this goes outside the bounds, use midpoint instead.
            t = t - (cdf - u[j])/pdf
            if (t < lower) or (t > upper):
                t = .5*(lower + upper)
            
            # Calculate cdf and pdf at the updated value of t
            x = A2L / t
            z = x + y/t
            ltx = np.exp(-1/H*((x + U)**sigma - U**sigma))
            ltzexpy = np.exp(-1/H*((z + U)**sigma - U**sigma)) * expy
            parsum = .5*ltx.real + np.cumsum(ltzexpy.real)
            parsum2 = .5*(ltx/x).real + np.cumsum((ltzexpy/z).real)
            pdf = expxt * np.sum(coef * parsum[seqbtL-1]) / t # CHECK INDEX
            cdf = expxt * np.sum(coef * parsum2[seqbtL-1]) / t # CHECK INDEX
            
            # Update bounds
            if cdf <= u[j]:
                lower = t
            else:
                upper = t
        
        if kount[j] >= maxiter:
            print('Warning: desired accuracy not achieced for F(x)=u')
        xrand[j] = t
        lower = t
    
    # Assuming n > 1
    rsample = np.random.permutation(xrand)
    
    return rsample
