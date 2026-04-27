# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:40:03 2026

@author: Dan Hagborg
"""

import numpy as np
import pandas as pd

from scipy.special import expit
from coverte import anova_JCR
from coverte import ireg_JCR
from coverte import mediation_JCR

rng = np.random.default_rng(6990)

x = rng.multivariate_normal(
    mean = np.array([0,0,0,0]),
    cov = np.diag(np.ones(4)),
    size = 1000
)

w = rng.binomial(
    n = 1,
    p = expit(x @ np.array([0.2,0.2,0.2,0.2])),
    size = 1000
)

coef = np.array([0,1,0.5,-0.5])
icoef = np.array([0.5,0,0.3,-0.2])

correct = pd.DataFrame({
    'conf': np.tile(-1, 500),
    'overall': np.tile(-1, 500)
})

# NDE

for i in range(500):

    m = rng.standard_cauchy(size = 1000) + w + x @ coef + np.diag(w) @ x @ icoef
    
    y = rng.standard_cauchy(size = 1000)*2 + w*2 + m*0.5 + w*m*0.2 + x @ coef + np.diag(w) @ x @ icoef
    
    y_te0 = rng.standard_cauchy()*2 # w = 0, m = 0
    
    y_te1 = rng.standard_cauchy()*2 + 2 # w = 1, m = 0
    
    jcr = mediation_JCR(
        formula_response = "y~w*m+w*x1+w*x2+w*x3+w*x4",
        formula_mediator = "m~w*x1+w*x2+w*x3+w*x4",
        data = pd.DataFrame(
            {'w': w, 'x1':x[:,0], 'x2':x[:,1], 'x3':x[:,2], 'x4':x[:,3], 'm':m, 'y':y}),
        treat_col = 'w',
        param = 'NDE',
        method = 'conform',
        bootstrap_size = 1000,
        bootstrap_coefs = 500,
        bootstrap_scores = 500,
        omega = 2**0.5*1
    )
    
    low, high = jcr.vertex[0], jcr.vertex[0] + jcr.vectors[0][0]
    pr = y_te1 - y_te0 - 2 + low
    plow, phigh = jcr.vertex[1], jcr.vertex[1] + jcr.vectors[1][1]
    
    if low <= 2 and 2 <= high:
        correct.loc[i, 'conf'] = 1
        if (plow <= pr and pr <= phigh):
            correct.loc[i, 'overall'] = 1
        else:
            correct.loc[i, 'overall'] = 0
    else:
        correct.loc[i, 'conf'] = 0
        correct.loc[i, 'overall'] = 0

# CDE

for i in range(500):

    m = rng.standard_normal(size = 1000) + w + x @ coef + np.diag(w) @ x @ icoef
    
    y = rng.standard_normal(size = 1000)*2 + w*2 + m*0.5 + w*m*0.2 + x @ coef + np.diag(w) @ x @ icoef
    
    y_te0 = rng.standard_normal()*2 + 0.25 # w = 0, m = 0.5
    
    y_te1 = rng.standard_normal()*2 + 2.35 # w = 1, m = 0.5
    
    jcr = mediation_JCR(
        formula_response = "y~w*m+w*x1+w*x2+w*x3+w*x4",
        formula_mediator = "m~w*x1+w*x2+w*x3+w*x4",
        data = pd.DataFrame(
            {'w': w, 'x1':x[:,0], 'x2':x[:,1], 'x3':x[:,2], 'x4':x[:,3], 'm':m, 'y':y}),
        treat_col = 'w',
        param = 'CDE',
        param_data = 0.5,
        method = 'conform',
        bootstrap_size = 1000,
        bootstrap_coefs = 500,
        bootstrap_scores = 500,
        omega = 2**0.5*1
    )
    
    low, high = jcr.vertex[0], jcr.vertex[0] + jcr.vectors[0][0]
    pr = y_te1 - y_te0 - 2.1 + low
    plow, phigh = jcr.vertex[1], jcr.vertex[1] + jcr.vectors[1][1]
    
    if low <= 2.1 and 2.1 <= high:
        correct.loc[i, 'conf'] = 1
        if (plow <= pr and pr <= phigh):
            correct.loc[i, 'overall'] = 1
        else:
            correct.loc[i, 'overall'] = 0
    else:
        correct.loc[i, 'conf'] = 0
        correct.loc[i, 'overall'] = 0

# IREG ATE

for i in range(500):
    
    y = rng.standard_normal(size = 1000)*2 + w*2 + x @ coef + np.diag(w) @ x @ icoef
    
    y_te0 = rng.standard_normal()*2 # w = 0
    
    y_te1 = rng.standard_normal()*2 + 2 # w = 1
    
    jcr = ireg_JCR(
        formula = "y~w*x1+w*x2+w*x3+w*x4",
        data = pd.DataFrame(
            {'w': w, 'x1':x[:,0], 'x2':x[:,1], 'x3':x[:,2], 'x4':x[:,3], 'y':y}),
        treat_col = 'w',
        param = 'ATE',
        method = 'pivot',
        bootstrap_size = 1000,
        bootstrap_coefs = 500,
        bootstrap_scores = 500,
        omega = 2**0.5*1
    )
    
    low, high = jcr.vertex[0], jcr.vertex[0] + jcr.vectors[0][0]
    pr = y_te1 - y_te0 - 2 + low
    plow, phigh = jcr.vertex[1], jcr.vertex[1] + jcr.vectors[1][1]
    
    if low <= 2 and 2 <= high:
        correct.loc[i, 'conf'] = 1
        if (plow <= pr and pr <= phigh):
            correct.loc[i, 'overall'] = 1
        else:
            correct.loc[i, 'overall'] = 0
    else:
        correct.loc[i, 'conf'] = 0
        correct.loc[i, 'overall'] = 0

# IREG CATE

for i in range(500):
    
    y = rng.standard_normal(size = 1000)*2 + w*2 + x @ coef + np.diag(w) @ x @ icoef
    
    y_te0 = rng.standard_normal()*2 + 1 # w = 0, x = (1,1,1,1)
    
    y_te1 = rng.standard_normal()*2 + 3.6 # w = 1, x = (1,1,1,1)
    
    jcr = ireg_JCR(
        formula = "y~w*x1+w*x2+w*x3+w*x4",
        data = pd.DataFrame(
            {'w': w, 'x1':x[:,0], 'x2':x[:,1], 'x3':x[:,2], 'x4':x[:,3], 'y':y}),
        treat_col = 'w',
        param = 'CATE',
        param_data = pd.DataFrame(
            {'x1':1, 'x2':1, 'x3':1, 'x4':1}),
        method = 'pivot',
        bootstrap_size = 1000,
        bootstrap_coefs = 500,
        bootstrap_scores = 500,
        omega = 2**0.5*1
    )
    
    low, high = jcr.vertex[0], jcr.vertex[0] + jcr.vectors[0][0]
    pr = y_te1 - y_te0 - 2.6 + low
    plow, phigh = jcr.vertex[1], jcr.vertex[1] + jcr.vectors[1][1]
    
    if low <= 2.6 and 2.6 <= high:
        correct.loc[i, 'conf'] = 1
        if (plow <= pr and pr <= phigh):
            correct.loc[i, 'overall'] = 1
        else:
            correct.loc[i, 'overall'] = 0
    else:
        correct.loc[i, 'conf'] = 0
        correct.loc[i, 'overall'] = 0

# ANOVA CATE

x = np.concat([np.tile('x1', 250), np.tile('x2', 250),
               np.tile('x1', 250), np.tile('x2', 250)])

w = np.concat([np.tile('w1', 500), np.tile('w2', 500)])

xnum = np.concat([np.tile(0, 250), np.tile(1, 250),
                  np.tile(0, 250), np.tile(1, 250)])

wnum = np.concat([np.tile(0, 500), np.tile(1, 500)])

correct = pd.DataFrame({
    'conf': np.tile(-1, 500),
    'overall': np.tile(-1, 500)
})

for i in range(500):
    
    y = rng.standard_cauchy(size = 1000)*2 + wnum + xnum + 0.5*wnum*xnum
    
    y_te0 = rng.standard_cauchy()*2 + 1 # w = 0, x = 1
    
    y_te1 = rng.standard_cauchy()*2 + 2.5 # w = 1, x = 1
    
    jcr = anova_JCR(
        formula = "y~w*x",
        data = pd.DataFrame(
            {'w': w, 'x':x, 'y':y}),
        param_data = pd.DataFrame({
            'w':['w1','w2'], 'x':['x2','x2']}),
        method = 'pivot',
        bootstrap_size = 1000,
        bootstrap_coefs = 500,
        bootstrap_scores = 500,
        omega = 2**0.5*1
    )
    
    low, high = jcr.vertex[0], jcr.vertex[0] + jcr.vectors[0][0]
    pr = y_te1 - y_te0 - 1.5 + low
    plow, phigh = jcr.vertex[1], jcr.vertex[1] + jcr.vectors[1][1]
    
    if low <= 1.5 and 1.5 <= high:
        correct.loc[i, 'conf'] = 1
        if (plow <= pr and pr <= phigh):
            correct.loc[i, 'overall'] = 1
        else:
            correct.loc[i, 'overall'] = 0
    else:
        correct.loc[i, 'conf'] = 0
        correct.loc[i, 'overall'] = 0