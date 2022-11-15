# permutation_counting
In this GitHub repository, we go over the implementation of the algorithms from the article "Inference for Bayesian nonparametric models with binary response data via permutation counting". In this short tutorial, we cover how to calculate the permanent of a block-rectangular matrix.

## Import
```python
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from methods import *
import math
from scipy.stats import norm
from scipy.interpolate import interp1d
```

## First example: Calculate a permanent
To get started, let us use the code from the methods.py file to calculate the block rectangular matrix $A$ appearing in Figure 3a in the main article. That is, the matrix

```math
A = \begin{pmatrix} 1 & 1 & 1 & 1 & 0 & 0 & 0 \\ 1 & 1 & 1 & 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 1 & 1 & 1 & 0 \\ 0 & 1 & 1 & 1 & 1 & 1 & 1 \\ 0 & 0 & 0 & 0 & 1 & 1 & 1 \\ 0 & 0 & 0 & 0 & 0 & 1 & 1 \end{pmatrix}
```
In order to do so, we must find the parametrisation of the matrix $A$ in terms of $\alpha, \beta, \gamma$ and $m$, as described in the main article. For our matrix $A$, we have
```python
alpha = np.array([1,3,1,1,1])
beta = np.array([0,1,2,1])
gamma = np.array([1,1,1,0])
m = 7
```
Using the logpermanent_calculator, we can calculate the $\log\mathrm{perm} A$ directly:
```python
logP = logpermanent_calculator(alpha, beta, gamma, m)
print(logP)
# 5.402677381872278
```
For a more substantial example, see the file dirichlet.py for the recreation of the Dirichlet process model example from the main article.
