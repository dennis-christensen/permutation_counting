### Dirichlet process model with current status data

# Import
import numpy as np
import numba as nb
import math
from methods import *
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Parameters
reqESS = 2_000
mu0, sigma0 = 0, 1 # prior hyperparameters
a = 1 # prior concentration
K = 1_000 # stick-breaking accuracy
mplot = 20

# Data
n = 100
inputs = np.linspace(-3, 3, n//10)
numSuccesses = np.array([0, 0, 2, 1, 4, 6, 9, 10, 10, 10])
obsleft = []
obsright = []
for i in range(n//10):
    obsleft += [inputs[i]]*numSuccesses[i]
    obsright += [inputs[i]]*(10 - numSuccesses[i])
obsleft = np.array(obsleft)
obsright = np.array(obsright)
nleft = len(obsleft)
nright = len(obsright)

# Simulations
@nb.njit(nb.types.Tuple((nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.int64))(nb.int64,
                                                                                     nb.float64[:],
                                                                                     nb.float64[:],
                                                                                     nb.float64,
                                                                                     nb.float64,
                                                                                     nb.float64,
                                                                                     nb.int64,                                                                             nb.int64,
                                                                                     nb.int64))
def simulations(n, obsleft, obsright, mu0, sigma0, a, T, K, reqESS):
    nleft = len(obsleft)
    nright = len(obsright)
    boundaries = np.sort(np.array([-np.inf, np.inf] + list(obsleft) + list(obsright)))
    
    Ws = np.zeros((T, K))
    xis = np.zeros((T, K))
    logpermanents = -np.inf*np.ones(T)
    
    logreqESS = np.log(reqESS)
    logESS = -np.inf
    
    t = 0
    num_zero_permanents = 0
    
    while t < T and logESS < logreqESS:
        # Stick-breaking
        B = np.random.beta(1, a, size=K)
        augB = np.append(B[::-1], 0)[::-1][:K]
        W = np.cumprod(1 - augB)*B
        xi = np.random.normal(mu0, sigma0, size=K)
    
        inds = np.argsort(xi)
        W = W[inds]
        xi = xi[inds]
        
        x = numba_choice_sampler(xi, W/np.sum(W), n)
        x = np.sort(x)
        
        # Continue if permanent is zero
        nonzero_permanent = True
        for i in range(nleft):
            if x[i] > obsleft[i]:
                nonzero_permanent = False
                break
        for i in range(nright):
            if x[nleft + i] <= obsright[i]:
                nonzero_permanent = False
                break
        if not nonzero_permanent:
            num_zero_permanents += 1
            continue
        
        # Add W and xi
        Ws[t] = W
        xis[t] = xi
        
        # Calculate permanent
        alpha, beta, gamma = get_abc(x, obsleft, obsright, boundaries)
        logP = logpermanent_calculator(alpha, beta, gamma, n)
        logpermanents[t] = logP
        
        # Update ESS
        logESS = 2*logsumexp_array(logpermanents) - logsumexp_array(2*logpermanents)       
        t += 1
        
    return Ws, xis, logpermanents, num_zero_permanents


# Results
T = 30_000
Ws, xis, logpermanents, num_zero_permanents = simulations(n, obsleft, obsright, mu0, sigma0, a, T, K, reqESS)

Ws = Ws[logpermanents > -np.inf]
xis = xis[logpermanents > -np.inf]
logpermanents = logpermanents[logpermanents > -np.inf]
T = len(logpermanents)

# Efficiency
print(f'Number of nonzero permanents: {T}')
print(f'Number of zero permanents: {num_zero_permanents}')

# Log marginal likelihood
logfactorial = 0
for i in range(1, n+1):
    logfactorial += np.log(i)

    logconst = 0
for i in numSuccesses:
    logconst += logncr_scalar(10, i)

logML = logsumexp_array(logpermanents) - np.log(T + num_zero_permanents) - logfactorial+ logconst
print(f'Log marginal likelihood: {logML}')


# Posterior mean
numericalxis = np.copy(xis)
numericalWs = np.copy(Ws)

numericalxis = np.column_stack(((np.min(xis) - 1e-10)*np.ones((T, 1)), numericalxis, (np.max(xis) + 1e-10)*np.ones((T, 1))))
numericalWs = np.column_stack((np.zeros((T, 1)), numericalWs, np.ones((T, 1))))

R = 1_000
posteriorspace = np.linspace(np.min(xis), np.max(xis), R)
posteriormean = np.zeros(R)
for t in range(T):
    y = interp1d(numericalxis[t], np.cumsum(numericalWs[t]), kind='previous')(posteriorspace)
    posteriormean += y*np.exp(logpermanents[t]) 
posteriormean /= posteriormean[-1]

# Resampling
posterior_indices = np.random.choice(T, size=T, p=np.exp(logpermanents)/np.sum(np.exp(logpermanents)), replace=True)
posteriorWs = Ws[posterior_indices]
posteriorxis = xis[posterior_indices]

# Successive substitution sampling (SSS)
MSSS = 500
KSSS = 50

xisSSS = np.zeros((MSSS, K))
WsSSS = np.zeros((MSSS, K))

for j in range(MSSS):
    
    non_convergence = False
    
    # Initial stick-breaking
    B = np.random.beta(1, a, size=K)
    augB = np.append(B[::-1], 0)[::-1][:K]
    W = np.cumprod(1 - augB)*B
    xi = np.random.normal(mu0, sigma0, size=K)
    
    inds = np.argsort(xi)
    W = W[inds]
    xi = xi[inds]
    
    for k in range(KSSS):
        # Successive sampling
        xsample = np.zeros(n)
        for i in range(nleft):
            probs = W[xi < obsleft[i]]
            try:
                xsample[i] = np.random.choice(xi[xi < obsleft[i]], p=probs/np.sum(probs))
            except:
                non_convergence = True
        for i in range(nright):
            probs = W[xi >= obsright[i]]
            try:
                xsample[nleft + i] = np.random.choice(xi[xi >= obsright[i]], p=probs/np.sum(probs))
            except:
                non_convergence = True
        if non_convergence:
            break
            
        # Successive stick-breaking
        B = np.random.beta(1, a + n, size=K)
        augB = np.append(B[::-1], 0)[::-1][:K]
        W = np.cumprod(1 - augB)*B
        u = np.random.random(K)
        xi = (u < a/(a + n))*np.random.normal(mu0, sigma0, size=K) \
                            + (u >= a/(a + n))*np.random.choice(xsample, size=K, replace=True)
        
        inds = np.argsort(xi)
        W = W[inds]
        xi = xi[inds]
    xisSSS[j] = xi
    WsSSS[j] = W

for r in range(MSSS):
    inds = np.argsort(xisSSS[r])
    WsSSS[r] = WsSSS[r][inds]
    xisSSS[r] = xisSSS[r][inds]

numericalxisSSS = np.copy(xisSSS)
numericalWsSSS = np.copy(WsSSS)

numericalxisSSS = np.column_stack(((np.min(xisSSS) - 1e-10)*np.ones((MSSS, 1)), 
                                    numericalxisSSS, (np.max(xisSSS) + 1e-10)*np.ones((MSSS, 1))))
numericalWsSSS = np.column_stack((np.zeros((MSSS, 1)), numericalWsSSS, np.ones((MSSS, 1))))

RSSS = 1_000
posteriorspaceSSS = np.linspace(np.min(xisSSS), np.max(xisSSS), RSSS)
posteriormeanSSS = np.zeros(RSSS)
for t in range(len(xisSSS)):
    ySSS = interp1d(numericalxisSSS[t], np.cumsum(numericalWsSSS[t]), kind='previous')(posteriorspaceSSS)
    posteriormeanSSS += ySSS
posteriormeanSSS /= posteriormeanSSS[-1]

# Plot
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

for j in range(mplot):
    B = np.random.beta(1, a, size=K)
    augB = np.append(B[::-1], 0)[::-1][:K]
    W = np.cumprod(1 - augB)*B
    xi = np.random.normal(mu0, sigma0, size=K)
    
    inds = np.argsort(xi)
    W = W[inds]
    xi = xi[inds]
    axs[0].step(xi, np.cumsum(W), alpha=.5, linestyle='dotted')

space = np.linspace(-3, 3, 200)
axs[0].plot(space, norm().cdf(space), color='black', label='Prior mean')
axs[0].set_xlabel('t')
axs[0].set_ylabel('F(t)')

for j in range(mplot):
    random_index = np.random.choice(T)
    W = posteriorWs[random_index]
    xi = posteriorxis[random_index]
    axs[1].step(xi, np.cumsum(W), alpha=.5, linestyle='dotted')
    
axs[1].plot(posteriorspace, posteriormean, color='black')
axs[1].plot(posteriorspaceSSS, posteriormeanSSS, color='black', linestyle='dashed')
axs[1].set_xlabel('t')
axs[0].set_title('Prior')
axs[1].set_title('Posterior')
plt.show()

# Quantiles
qspace = np.linspace(0, 1, 11)[1:-1]
for q in qspace:
    quantile = posteriorspace[np.argmin(np.abs(posteriormean - q))]
    quantileSSS = posteriorspaceSSS[np.argmin(np.abs(posteriormeanSSS - q))]
    print(f'q = {q}')
    print(f' - New: {quantile}')
    print(f' - SSS: {quantileSSS}')
    print('')
