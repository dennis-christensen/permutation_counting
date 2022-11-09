#####
# Methods needed for "Inference for Bayesian nonparametric models with binary response data via permutation counting"
# Dennis Christensen
#####

# Import
import numpy as np
import numba as nb
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Methods
@nb.njit(nb.float64[:](nb.float64[:]))
def numbasort(array):
    if len(array) == 0: 
        return array
    else:
        pivot = array[0]
        lesser = numbasort(np.array([x for x in array[1:] if x < pivot]))
        greater = numbasort(np.array([x for x in array[1:] if x >= pivot]))
        return np.concatenate((lesser, np.array([pivot]), greater))
    
    
@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.int64))
def numba_choice_sampler(array, probabilities, size):
    cumsump = np.cumsum(probabilities)
    result = np.zeros(size)
    for i in range(size):
        u = np.random.random()
        index = np.argmax(u <= cumsump)
        result[i] = array[index]
    return result
    
    
@nb.njit(nb.float64[:](nb.float64[:]))
def minus1d(a):
    for i in range(len(a)):
        if a[i] > -np.inf:
            a[i] = -a[i]
    return a


@nb.njit(nb.float64[:,:](nb.float64[:,:]))
def minus2d(a):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] > -np.inf:
                a[i][j] = -a[i][j]
    return a


@nb.njit(nb.float64(nb.int64))
def logfactorial_scalar(value):
    if value < 0:
        return -np.inf
    result = 0
    for i in range(1, value + 1):
        result += np.log(i)
    return result


@nb.njit(nb.float64[:,:](nb.int64, nb.int64, nb.int64))
def logfactorial_arangearange(base_value, value1, value2):
    result = -np.inf*np.ones((value1 + 1, value2 + 1))
    flatlength = value1 + value2 + 1 - max(0, -base_value)
    if flatlength <= 0:
        return result
    flatrow = np.zeros(flatlength)
    flatrow[0] = logfactorial_scalar(max(0, base_value))
    for i in range(1, flatlength):
        flatrow[i] = flatrow[i-1] + np.log(max(0, base_value) + i)
    for i in range(value1 + 1):
        for j in range(max(0, -base_value - i), value2 + 1):
            result[i][j] = flatrow[j - max(0, -base_value - i) + max(0, i + min(0, base_value))]
    return result


@nb.njit(nb.float64(nb.int64, nb.int64))
def lognpr_scalar(value1, value2):
    if value1 < 0:
        return -np.inf
    result = 0
    for i in range(max(value1 - value2 + 1, 0), value1 + 1):
        result += np.log(i)
    return result


@nb.njit(nb.float64[:](nb.int64, nb.int64[:]))
def lognpr_scalararange(value, array):
    if value < 0:
        return -np.inf*np.ones(len(array))
    result = np.append(1, (value*np.ones(len(array)) - array + 1)[1:])
    result *= (result >= 0)
    return np.cumsum(np.log(result))


@nb.njit(nb.float64[:](nb.int64[:], nb.int64))
def lognpr_arangescalar(array, value):
    result = -np.inf*np.ones(len(array))
    if value > len(array) or value < 0:
        return result
    result[value] = lognpr_scalar(value, value)
    for i in range(value + 1, len(array)):
        result[i] = result[i-1] + np.log(i) - np.log(i - value)
    return result


@nb.njit(nb.float64(nb.int64, nb.int64))
def logncr_scalar(value1, value2):
    if value1 < 0:
        return -np.inf
    result = lognpr_scalar(value1, value2)
    for i in range(1, value2 + 1):
        result -= np.log(i)
    return result


@nb.njit(nb.float64[:](nb.int64, nb.int64[:]))
def logncr_scalararange(value, array):
    result = lognpr_scalararange(value, array) - np.append(0, np.cumsum(np.log(np.arange(1, len(array)))))
    result[np.abs(result) < 1e-10] = 0
    return result


@nb.njit(nb.float64[:](nb.int64[:], nb.int64))
def logncr_arangescalar(array, value):
    if value < 0:
        return -np.inf*np.ones(len(array))
    return lognpr_arangescalar(array, value) - logfactorial_scalar(value)


@nb.njit(nb.float64(nb.float64[:,:]))
def logsumexp2d(array):
    maxarray = np.max(array)
    exptrick = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            exptrick += np.exp(array[i][j] - maxarray)
    return maxarray + np.log(exptrick)


@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:]))
def logdotexp(A, B):
    # Assuming none of the matrices only consist of -np.inf
    maxA = np.max(A)
    maxB = np.max(B)
    
    trickA = A - maxA
    trickB = B - maxB
    
    return np.log(np.dot(np.exp(trickA), np.exp(trickB))) + maxA + maxB


@nb.njit(nb.float64(nb.float64[:]))
def logsumexp_array(array):
    if np.all(array == -np.inf):
        return -np.inf
    maxarray = max(array)
    result = maxarray
    return maxarray + np.log(np.sum(np.exp(array - maxarray)))


@nb.njit(nb.float64[:](nb.int64, nb.int64))
def btQ(amount, value):
    result = np.zeros(value + 1)
    result[-1] = logfactorial_scalar(amount)
    for i in range(value):
        result[value - i - 1] = result[value - i] + np.log(amount + i+1) - np.log(i+1)
    return result
        

@nb.njit(nb.float64[:,:](nb.int64[:], nb.int64[:], nb.int64[:], nb.int64, nb.int64, nb.float64[:,:]))
def reverse_bt(a, b, c, m, amount, perms):
    shiftedperms = np.column_stack((-np.inf*np.ones((a[0]+1, amount)), perms[:, :-amount]))
    Q = btQ(amount, a[-1])
    perms = shiftedperms + Q
    return perms


@nb.njit(nb.float64[:,:](nb.int64[:], nb.int64[:], nb.int64[:], nb.int64, nb.int64, nb.float64[:,:]))
def reverse_tt(a, b, c, m, amount, perms):
    shiftedperms = np.row_stack((-np.inf*np.ones((amount, a[-1]+1)), perms[:-amount]))
    Q = np.expand_dims(btQ(amount, a[0]), 1)
    perms = shiftedperms + Q
    return perms


@nb.njit(nb.float64[:,:](nb.int64))
def bsS1(value):
    result = -np.inf*np.ones((value + 1, value + 1))
    for l in range(value + 1):
        result[l][l:] = lognpr_scalararange(value - l, np.arange(value + 1 - l))
    return result


@nb.njit(nb.float64[:,:](nb.int64, nb.int64))
def bsS2(amount, value):
    result = -np.inf*np.ones((value + 1, value + 1))
    row = logncr_scalararange(amount, np.arange(value + 1))
    for i in range(value + 1):
        result[i][i:] = row[:value + 1 - i]
    return result


@nb.njit(nb.float64[:,:](nb.int64[:], nb.int64[:], nb.int64[:], nb.int64, nb.int64, nb.float64[:,:]))
def reverse_bs(a, b, c, m, amount, perms):
    aSum = np.sum(a[1:-1])
    Q = minus2d(logfactorial_arangearange(aSum - (m + amount), a[0], a[-1]))
    R = perms + logfactorial_arangearange(aSum - m, a[0], a[-1])
    S = bsS1(a[-1]) + bsS2(amount, a[-1])
    return Q + logdotexp(R, S)


@nb.njit(nb.float64[:,:](nb.int64[:], nb.int64[:], nb.int64[:], nb.int64, nb.int64, nb.float64[:,:]))
def reverse_ts(a, b, c, m, amount, perms):
    aSum = np.sum(a[1:-1])
    Q = minus2d(logfactorial_arangearange(aSum - (m + amount), a[0], a[-1]))
    R = perms + logfactorial_arangearange(aSum - m, a[0], a[-1])
    S = bsS1(a[0]).T + bsS2(amount, a[0]).T
    return Q + logdotexp(S, R)


@nb.njit(nb.float64[:,:](nb.int64, nb.int64))
def lmQ1(amount0, amount1):
    sumamount = amount0 + amount1
    result = np.zeros((amount0 + 1, sumamount + 1))
    for l in range(1, sumamount + 1):
        result[0][l] = result[0][l-1] + np.log(max(amount1 - l + 1, 0))
    for r in range(1, amount0 + 1):
        for l in range(r+1, sumamount + 1):
            result[r][l] = result[r-1][l-1]
    return result


@nb.njit(nb.float64[:,:](nb.int64, nb.int64))
def lmQ4(amount0, amount1):
    result = -np.inf*np.ones((amount0 + 1, amount0 + amount1 + 1))
    for r in range(amount0 + 1):
        result[r] = logncr_arangescalar(np.arange(amount0 + amount1 + 1), r)
    return result


@nb.njit(nb.float64[:,:](nb.int64[:], nb.int64[:], nb.int64[:], nb.int64, nb.int64, nb.int64, nb.float64[:,:]))
def reverse_lm(a, b, c, m, amount0, amount1, perms):
    Q1 = lmQ1(amount0, amount1)
    Q2 = lognpr_scalararange(amount0, np.arange(amount0 + 1))
    Q2 = np.expand_dims(Q2, 1)
    Q3 = lognpr_scalararange(amount0 + amount1, np.arange(amount0 + amount1 + 1))
    Q4 = lmQ4(amount0, amount1)
    return logdotexp(Q1 + Q2 + minus1d(Q3) + Q4, perms)


@nb.njit(nb.float64[:,:](nb.int64[:], nb.int64[:], nb.int64[:], nb.int64, nb.int64, nb.int64, nb.float64[:,:]))
def reverse_rm(a, b, c, m, amount0, amount1, perms):
    Q1 = lmQ1(amount1, amount0).T
    Q2 = lognpr_scalararange(amount1, np.arange(amount1 + 1))
    Q3 = lognpr_scalararange(amount0 + amount1, np.arange(amount0 + amount1 + 1))
    Q3 = np.expand_dims(Q3, 1)
    Q4 = lmQ4(amount1, amount0).T
    return logdotexp(perms, Q1 + Q2 + minus2d(Q3) + Q4)


@nb.njit(nb.int64[:](nb.float64[:], nb.float64[:]))
def get_downstairs(simulated, walls):
    return np.array([np.sum((simulated <= walls[i+1])*(simulated >= walls[i])) for i in range(len(walls) - 1)])


@nb.njit(nb.int64[:](nb.int64[:]))
def get_a(downstairs):
    return downstairs[downstairs!=0]


@nb.njit(nb.types.UniTuple(nb.int64[:], 2)(nb.int64[:], 
                                       nb.float64[:], 
                                       nb.int64[:],
                                       nb.float64[:], 
                                       nb.float64[:], 
                                       nb.float64[:]))
def get_b_c(a, simulated, downstairs, walls, xright, xleft):
    reps = np.cumsum(np.append(0, a)[:-1])
    
    b = np.array([sum(xleft > simulated[i]) for i in reps])
    b = b[:-1] - b[1:]
    
    c = np.array([sum(xright < simulated[i]) for i in reps])
    c = c[1:] - c[:-1]
    return b, c


@nb.njit(nb.types.UniTuple(nb.int64[:], 3)(nb.float64[:],
                                           nb.float64[:],
                                           nb.float64[:],
                                           nb.float64[:]))
def get_abc(x, xleft, xright, boundaries):
    a = np.array([np.sum((x <= boundaries[i+1])*(x >= boundaries[i])) \
                      for i in range(len(boundaries) - 1)])
    a = a[a!=0]
    
    reps = np.cumsum(np.append(0, a)[:-1])
    
    b = np.array([sum(xleft > x[i]) for i in reps])
    b = b[:-1] - b[1:]
    
    c = np.array([sum(xright < x[i]) for i in reps])
    c = c[1:] - c[:-1]
    
    return a, b, c


@nb.njit(nb.float64[:,:](nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def get_reduced_subpermanents(a, b, c, m):
    perms = -np.inf * np.ones((a[0]+1, a[1]+1))
    
    k = len(a)
    if k == 2:
        if b[0] == 0 and c[0] == 0 and sum(a) >= m:
            for r in range(a[0] + 1):
                s = m - r
                if s >= 0 and s <= a[1]:
                    perms[r][s] = logncr_scalar(m, r) + lognpr_scalar(a[0], r) + lognpr_scalar(a[1], s)
        
        if b[0] == 0 and c[0] == m and m <= a[1]:
            perms[0][m] = lognpr_scalar(a[1], m)
            
        if b[0] == m and c[0] == 0 and m <= a[0]:
            perms[m][0] = lognpr_scalar(a[0], m)
        
    if k == 3:
        # Know b = [0, m] and c = [m, 0]
        if m <= a[1]:
            perms[0][0] = lognpr_scalar(a[1], m)
    return perms


@nb.njit(nb.boolean(nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def no_moves_possible(a, b, c, m):
    case1 = (len(a) == 2) and (b[0] == 0) and (c[0] == 0)
    case2 = (len(a) == 2) and (b[0] == 0) and (c[0] == m)
    case3 = (len(a) == 2) and (b[0] == m) and (c[0] == 0)
    case4 = (len(a) == 3) and (list(b) == [0, m]) and (list(c) == [m, 0])
    return case1 or case2 or case3 or case4


@nb.njit(nb.boolean(nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def bt_possible(a, b, c, m):
    return m > c[-1] and c[-1] > 0


@nb.njit(nb.boolean(nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def tt_possible(a, b, c, m):
    return m > b[0] and b[0] > 0


@nb.njit(nb.boolean(nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def bs_possible(a, b, c, m):
    return m > c[0] and c[0] > 0


@nb.njit(nb.boolean(nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def ts_possible(a, b, c, m):
    return m > b[-1] and b[-1] > 0


@nb.njit(nb.boolean(nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def lm_possible(a, b, c, m):
    return b[0] == 0 and c[0] == 0


@nb.njit(nb.boolean(nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def rm_possible(a, b, c, m):
    return b[-1] == 0 and c[-1] == 0


@nb.njit(nb.types.Tuple((nb.int64[:],
                        nb.int64[:],
                        nb.int64[:],
                        nb.int64,
                        nb.int64[:],
                        nb.int64[:,:]))(nb.int64[:],
                                       nb.int64[:],
                                       nb.int64[:],
                                       nb.int64))
def reduction(a, b, c, m):
    history = np.zeros(0).astype('int64')
    quantityhistory = np.zeros((0, 2)).astype('int64')
    while True:
        if no_moves_possible(a, b, c, m):
            break
        if tt_possible(a, b, c, m):
            history = np.append(history, 0)
            quantityhistory = np.vstack((quantityhistory, np.array([[0, b[0]]])))
            m -= b[0]
            b[0] = 0
        if no_moves_possible(a, b, c, m):
            break
        if bs_possible(a, b, c, m):
            history = np.append(history, 1)
            quantityhistory = np.vstack((quantityhistory, np.array([[0, c[0]]])))
            m -= c[0]
            c[0] = 0
        if no_moves_possible(a, b, c, m):
            break
        if lm_possible(a, b, c, m):
            history = np.append(history, 2)
            quantityhistory = np.vstack((quantityhistory, np.array([[a[0], a[1]]])))
            a[1] += a[0]
            a = a[1:]
            b = b[1:]
            c = c[1:]
        if no_moves_possible(a, b, c, m):
            break
        if bt_possible(a, b, c, m):
            history = np.append(history, 3)
            quantityhistory = np.vstack((quantityhistory, np.array([[0, c[-1]]])))
            m -= c[-1]
            c[-1] = 0
        if no_moves_possible(a, b, c, m):
            break
        if ts_possible(a, b, c, m):
            history = np.append(history, 4)
            quantityhistory = np.vstack((quantityhistory, np.array([[0, b[-1]]])))
            m -= b[-1]
            b[-1] = 0
        if no_moves_possible(a, b, c, m):
            break
        if rm_possible(a, b, c, m):
            history = np.append(history, 5)
            quantityhistory = np.vstack((quantityhistory, np.array([[a[-2], a[-1]]])))
            a[-2] += a[-1]
            a = a[:-1]
            b = b[:-1]
            c = c[:-1]
        if no_moves_possible(a, b, c, m):
            break
    return a, b, c, m, history, quantityhistory


@nb.njit(nb.float64[:,:](nb.int64[:],
                    nb.int64[:],
                    nb.int64[:],
                    nb.int64,
                    nb.int64[:],
                    nb.int64[:,:],
                    nb.float64[:,:]))
def reverse_reduction(a, b, c, m, history, quantityhistory, perms):
    for i in range(len(history) - 1, -1, -1):
        amount = quantityhistory[i]
        
        if history[i] == 2:
            perms = reverse_lm(a, b, c, m, amount[0], amount[1], perms)
            a[0] -= amount[0]
            a = np.append(amount[0], a)
            b = np.append(0, b)
            c = np.append(0, c)
        if history[i] == 5:
            perms = reverse_rm(a, b, c, m, amount[0], amount[1], perms)
            a[-1] -= amount[1]
            a = np.append(a, amount[1])
            b = np.append(b, 0)
            c = np.append(c, 0)
        if history[i] == 0:
            perms = reverse_tt(a, b, c, m, amount[1], perms)
            b[0] = amount[1]
            m += amount[1]
        if history[i] == 3:
            perms = reverse_bt(a, b, c, m, amount[1], perms)
            c[-1] = amount[1]
            m += amount[1]
        if history[i] == 4:
            perms = reverse_ts(a, b, c, m, amount[1], perms)
            b[-1] = amount[1]
            m += amount[1]
        if history[i] == 1:
            perms = reverse_bs(a, b, c, m, amount[1], perms)
            c[0] = amount[1]
            m += amount[1]
    return perms


@nb.njit(nb.float64(nb.int64[:], nb.int64[:], nb.int64[:], nb.int64))
def logpermanent_calculator(a, b, c, m):
    n = sum(a)
    if sum(b) + sum(c) == 0: # matrix of ones
        return logfactorial_scalar(n)
    
    # Reduce
    a, b, c, m, history, quantityhistory = reduction(a, b, c, m)

    # Get initial subpermanents
    perms = get_reduced_subpermanents(a, b, c, m)

    # Reverse
    perms = reverse_reduction(a, b, c, m, history, quantityhistory, perms)
    
    return logsumexp2d(perms)
