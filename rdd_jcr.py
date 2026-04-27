# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:19:39 2026

@author: Dan Hagborg
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from coverte import rdd_JCR
from rdrobust import rdrobust

senate = pd.read_csv('senate.csv', index_col = 0).dropna()
senate.loc[senate['margin'] > 0, 'w'] = 1
senate.loc[senate['margin'] < 0, 'w'] = 0

# Bandwidth Choice

est = np.zeros(50)

for bw in range(1, 51):
    jcr = rdd_JCR(data = senate,
                  treat_col = 'w',
                  resp_col = 'vote',
                  running_col = 'margin',
                  degree = 1,
                  interact_degree = 1,
                  bw = bw,
                  method = 'pivot')
    est[bw-1] = (jcr.projection('param')[0] + jcr.projection('param')[1])/2
    

fig, ax = plt.subplots()
sns.scatterplot(x = range(1, 51), y = est)
ax.set_xlabel("Bandwidth (Percentage Points)")
ax.set_ylabel("Estimate (Percentage Points)")
ax.set_title("Bandwidth Sensitivity")

# Plot

fig, ax = plt.subplots()
sns.scatterplot(senate.loc[senate['margin'].abs() < 15], 
                x = 'margin', y = 'vote', size = 1, legend = False)
ax.set_xlabel("Previous Margin (Percentage)")
ax.set_ylabel("Current Vote (Percentage)")
ax.set_title("Current Vote vs. Previous Margin (BW = 15)")

# Bias-adjusted CI

rdrobust(senate['vote'], x = senate['margin'], 
         p = 1, q = 2, h = 10, level = 97.5) # 7.985, 5.83, 18.013
rdrobust(senate['vote'], x = senate['margin'], 
         p = 1, q = 2, h = 15, level = 97.5) # 7.487, 4.063, 14.108
rdrobust(senate['vote'], x = senate['margin'], 
         p = 1, q = 2, h = 20, level = 97.5) # 7.27, 3.779, 12.55

rdd_JCR(data = senate,
        treat_col = 'w',
        resp_col = 'vote',
        running_col = 'margin',
        degree = 1,
        interact_degree = 1,
        bw = 15,
        low = 4.063, high = 14.108,
        bootstrap_scores = 1000,
        method = 'conform').plot(
            xlab = 'LATE (Percentage Points)',
            ylab = 'Predicted LTE (Percentage Points)',
            main = 'JCR for Effect at Discontinuity')

rdd_JCR(data = senate,
        treat_col = 'w',
        resp_col = 'vote',
        running_col = 'margin',
        degree = 1,
        interact_degree = 1,
        bw = 15,
        low = 4.063, high = 14.108,
        bootstrap_scores = 1000,
        method = 'conform').projection('predict')

# Placebo Tests
                                 
rdrobust(senate['vote'], x = senate['margin'], c = -10, h = 10, 
         level = 97.5)

rdrobust(senate['vote'], x = senate['margin'], c = -10, h = 15, 
         level = 97.5)

rdrobust(senate['vote'], x = senate['margin'], c = -10, h = 20, 
         level = 97.5)

rdrobust(senate['vote'], x = senate['margin'], c = 10, h = 10, 
         level = 97.5)

rdrobust(senate['vote'], x = senate['margin'], c = 10, h = 15, 
         level = 97.5)

rdrobust(senate['vote'], x = senate['margin'], c = 10, h = 20, 
         level = 97.5)
