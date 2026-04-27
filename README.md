# coverte
This is a python package implementing joint coverage regions (JCRs) for expected and predicted treatment effects in regressions in causal inference. For more information on JCRs, see (Dobriban and Lin 2025).

## File descriptions:

coverte.py  
The Python package for JCR construction.

JCR coverage.csv  
Raw simulated coverage data.

JCR coverage.R  
R script for creating the coverage plots using ggplot2 (using data from 'JCR coverage.csv').

JCR_scripts.py  
Scripts for simulating data.

rdd_jcr.py  
Scripts for creating the JCR for the Senate election RDD analysis.

senate.csv  
Senate election data (from Calonico et. al 2015).

## Imported Packages

coverte.py and JCR_scripts.py: numpy, pandas, scipy, patsy, statsmodels, sklearn, matplotlib

rdd_jcr.py: numpy, pandas, matplotlib, seaborn, rdrobust

JCR coverage.R: dplyr, ggplot2, forcats

## Sources

Calonico, S., Cattaneo, M.D., Titiunik, R.: rdrobust: An r package for robust non-
parametric inference in regression-discontinuity designs. The R Journal 7, 38–51
(2015)

Dobriban, E., Lin, Z.: Joint coverage regions: Simultaneous confidence and prediction sets (2025), https://arxiv.org/abs/2303.00203
