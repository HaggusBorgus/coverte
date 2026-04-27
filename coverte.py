# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:08:55 2026

@author: Dan Hagborg
"""

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

from patsy import dmatrix
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D


class JCR:
    """
    General class for JCRs. The constructor for this is meant to be used 
    internally; for use with data and models, use coverte.anova_JCR(), 
    coverte.ireg_JCR(), etc.

    Methods:
        cross_section(name, value):
            Returns a JCR object that is a cross-section at a particular value on a
            particular axis.
        plot(xlab = None, ylab = None, main = "JCR"):
            Plots the bounds of the JCR (must be at most 2 dimensions).
        projection(name):
            Returns a tuple giving the bounds of the JCR when projected on the
            specified axis, effectively giving a marginal confidence/prediction 
            interval.

    Attributes:
        dim:
            (int) Number of dimensions of the JCR.
        names:
            (dict<int>) Dictionary where the keys are dimension names (generally
            "param" and "predict") and values are corresponding indices in
            representations of the JCR in coordinates.
        vectors:
            (numpy.ndarray) Two-dimensional array giving vectors denoting the edges
            of the JCR. The first index denotes which vector; the second denotes
            which entry in the vector.
        vertex:
            (numpy.ndarray) Vector giving the coordinates of one of the vertices
            of the JCR. The other vertices are given by this plus some subset of 
            the edge vectors in "vectors".
    """
    dim = 0
    vertex = None
    vectors = None
    names = {}
    
    def __init__(self, vertex, vectors, names):
        
        # establish dimensionality
        self.dim = len(vertex)
        
        # assert correct dimensionality
        assert len(names) == self.dim
        assert len(vectors) == self.dim
        for v in vectors:
            assert len(v) == self.dim
        
        # initialize
        self.vertex = np.array(vertex)
        self.vectors = np.array(vectors)
        i = 0
        for s in names:
            self.names[s] = i
            i += 1
    
    
    def cross_section(self, name, value):
        """
        Returns a JCR object that is a cross-section at a particular value on a
        particular axis.
        
        Arguments:
            name:
                (String) The axis on which the cross-section is taken (generally
                "param" or "predict").
            value:
                (float) The value at which the cross-section is taken.
        
        Returns:
            (coverte.JCR) New JCR with one less dimension.
        """
        
        ind = self.names[name]
        new_vec = self.vectors.tolist()
        vec = np.array(new_vec.pop(ind))
        
        for i in range(self.dim-1):
            new_vec[i].pop(ind)
        
        new_vert = (self.vertex + vec*(value/vec[ind])).tolist()
        new_vert.pop(ind)
        
        new_names = list(self.names.keys())
        new_names.pop(ind)
        
        return JCR(new_vert, new_vec, new_names)
    
    
    def projection(self, name):
        """
        Returns a tuple giving the bounds of the JCR when projected on the
        specified axis, effectively giving a marginal confidence/prediction 
        interval.
        
        Arguments:
            name:
                (String) The axis on which the projection is taken (generally
                "param" or "predict").
        
        Returns:
            (tuple<float>) Bounds of the projection.
        """
        
        ind = self.names[name]
        low = self.vertex[ind]
        
        return (low, low + np.sum(self.vectors, axis = 0)[ind])
    
    
    def plot(self, xlab = None, ylab = None, main = "JCR"):
        """
        Plots the bounds of the JCR (must be at most 2 dimensions).
        
        Arguments:
            xlab:
                (String) X-axis label (default is the first key in self.names)
            ylab:
                (String) Y-axis label (default is the second key in self.names)
            main:
                (String) Plot title.
        """
        
        if xlab == None:
            xlab = list(self.names.keys())[0]
        if ylab == None:
            ylab = list(self.names.keys())[1]
        
        # Coordinates
        
        xbounds = self.projection(list(self.names.keys())[0])
        
        xedge = xbounds[0] - (xbounds[1] - xbounds[0])*0.1
        
        x_corner1 = self.vertex[0]
        x_corner2 = x_corner1 + self.vectors[0][0]
        x_corner3 = x_corner1 + self.vectors[1][0]
        x_corner4 = x_corner2 + self.vectors[1][0]
        
        y_corner1 = self.vertex[1]
        y_corner2 = y_corner1 + self.vectors[0][1]
        y_corner3 = y_corner1 + self.vectors[1][1]
        y_corner4 = y_corner2 + self.vectors[1][1]
        
        # Dotted Lines
        
        h1 = Line2D([xedge, x_corner1], [y_corner1, y_corner1],
                    c = 'c', ls = '--')
        h2 = Line2D([xedge, x_corner2], [y_corner2, y_corner2],
                    c = 'c', ls = '--')
        h3 = Line2D([xedge, x_corner3], [y_corner3, y_corner3],
                    c = 'c', ls = '--')
        h4 = Line2D([xedge, x_corner4], [y_corner4, y_corner4],
                    c = 'c', ls = '--')
        
        
        fig, ax = plt.subplots(layout="constrained")
        
        # Paralellogram
        
        x1 = np.linspace(x_corner1, x_corner2)
        x2 = np.linspace(x_corner1, x_corner3)
        
        y1 = np.linspace(y_corner1, y_corner2)
        y2 = np.linspace(y_corner1, y_corner3)
        
        x3 = x1 + self.vectors[1][0]
        x4 = x2 + self.vectors[0][0]
        
        y3 = y1 + self.vectors[1][1]
        y4 = y2 + self.vectors[0][1]
        
        ax.plot(x1, y1, c = 'b')
        ax.plot(x2, y2, c = 'b')
        ax.plot(x3, y3, c = 'b')
        ax.plot(x4, y4, c = 'b')
        
        ax.add_line(h1)
        ax.add_line(h2)
        ax.add_line(h3)
        ax.add_line(h4)
        
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(main)
        
        plt.show()


def anova_JCR(formula, data, param_data = None,
              method = "pivot", alpha = 0.05, symmetric = True, cross = True,
              bootstrap_size = 100, bootstrap_coefs = 1000, 
              bootstrap_scores = 1000, omega = 1, low = None,
              high = None):
    """
    Arguments:
        formula:
            (String) R-like formula (see patsy documentation). Predictor variables 
            should all be categorical. Do not use C() to denote categorical 
            variables; just use the variable names. Variable names cannot be 
            substrings of each other.
        data:
            (pandas.DataFrame) The data to be used.
        param_data:
            (pandas.DataFrame) DataFrame with two rows. Estimated parameter will be
            the conditional mean of param_data.loc[1] - param_data.loc[0], and
            predicted counterfactual will use the same contrast. Columns should
            contain ALL predictor variables specified in formula.
        method:
            (String) The method to use, either "pivot" or "conform".
        alpha:
            (float) The level of significance.
        symmetric:
            (bool) Whether or not to assume symmetrical noise. Applicable only to
            method = "conform".
        cross:
            (bool) Whether or not to assume exchangeability between treatment 
            groups. Applicable only to method = "conform".
        bootstrap_size:
            (int) Number of bootstrap draws per sample, for generating the 
            confidence interval. Applicable only to method = "conform", when at
            least on of low or high is not specified.
        bootstrap_coefs:
            (int) Number of bootstrap samples, for generating the confidence 
            interval. Applicable only to method = "conform", when at least on of 
            low or high is not specified.
        bootstrap_scores:
            (int) Number of bootstrap nonconformity scores, for generating JCR 
            band. Applicable only to method = "conform".
        omega:
            (float) The desired slope of the JCR. Applicable only to method = 
            "pivot".
        low:
            (float) The lower bound of the confidence interval. By default, this
            is calculated using the data.
        high:
            (float) The upper bound of the confidence interval. By default, this
            is calculated using the data.

    Returns:
        (coverte.JCR) The constructed JCR.
    """
    
    # Determine response column from formula
    
    resp_col = formula.split('~')[0].strip()
    
    # Initialize inferred predict data
    
    predict_data = param_data
    
    # Preprocessing: re-order factor levels
    
    new_formula = formula.split('~')[1].strip()
    
    for col in param_data.columns:
        
        ref = param_data.loc[0,col]
        
        new_formula = new_formula.replace(col, "C({},Treatment('{}'))".format(
            col, ref))
    
    des = dmatrix(formula_like = new_formula, 
                  data = pd.concat([data.drop(resp_col, axis = 1), 
                                    param_data, predict_data]),
                  return_type = "dataframe")
    
    n = data.shape[0]
    
    if method == 'pivot':
        return _pivot_JCR(X_tr = des.iloc[0:n], 
                          y_tr = data.loc[:, resp_col],
                          X_te = 2**-0.5*(des.iloc[n+3] - des.iloc[n+2]),
                          c = des.iloc[n+1] - des.iloc[n],
                          names = ['param','predict'],
                          alpha = alpha,
                          omega = omega,
                          low = low,
                          high = high)
    if method == 'conform':
        return _conform_JCR(X_tr = des.iloc[0:n], 
                            y_tr = data.loc[:, resp_col], 
                            X_te = des.iloc[(n+2):(n+4)],
                            treat_col = des.columns[1], 
                            names = ['param','predict'],
                            symmetric = symmetric,
                            cross = cross,
                            alpha = alpha,
                            n = bootstrap_size,
                            conf_k = bootstrap_coefs,
                            band_k = bootstrap_scores,
                            low = low,
                            high = high)
    
    return JCR()


def ireg_JCR(formula, data, treat_col, param_data = None,
             param = "ATE", method = "pivot", alpha = 0.05, 
             symmetric = True, cross = True,
             bootstrap_size = 100, bootstrap_coefs = 1000, 
             bootstrap_scores = 1000, omega = 1, low = None,
             high = None):
    """
    Arguments:
        formula:
            (String) R-like formula (see patsy documentation). Categorical 
            varibales must pre-encoded.
        data:
            (pandas.DataFrame) The data to be used.
        treat_col:
            (String) The column name for the treatment indicator.
        param_data:
            (pandas.DataFrame) DataFrame with one row, specifying values of the
            covariates being conditioned on, if using param = "CATE"; None, 
            otherwise. Columns should NOT contain the treatment.
        param:
            (String) Choice of parameter, either "ATE", "ATT", or "CATE".
        method:
            (String) The method to use, either "pivot" or "conform".
        alpha:
            (float) The level of significance.
        symmetric:
            (bool) Whether or not to assume symmetrical noise. Applicable only to
            method = "conform".
        cross:
            (bool) Whether or not to assume exchangeability between treatment 
            groups. Applicable only to method = "conform".
        bootstrap_size:
            (int) Number of bootstrap draws per sample, for generating the 
            confidence interval. Applicable only to method = "conform", when at
            least on of low or high is not specified.
        bootstrap_coefs:
            (int) Number of bootstrap samples, for generating the confidence 
            interval. Applicable only to method = "conform", when at least on of 
            low or high is not specified.
        bootstrap_scores:
            (int) Number of bootstrap nonconformity scores, for generating JCR 
            band. Applicable only to method = "conform".
        omega:
            (float) The desired slope of the JCR. Applicable only to method = 
            "pivot".
        low:
            (float) The lower bound of the confidence interval. By default, this
            is calculated using the data.
        high:
            (float) The upper bound of the confidence interval. By default, this
            is calculated using the data.

    Returns:
        (coverte.JCR) The constructed JCR.
    """
    
    # Determine response column from formula
    
    resp_col = formula.split('~')[0].strip()
    exog_form = formula.split('~')[1].strip()
    
    # Covariates (non-response, non-treatment)
    
    X = data.drop(columns = [resp_col, treat_col])
    
    # Initialize param_data for choice of param
    
    if param == "ATE":
        param_data = X.agg('mean')
    elif param == "ATT":
        param_data = X[data[treat_col] == 1].agg('mean')
    
    predict_data = param_data
    
    # Preprocessing: Center covariates around param_data, and combine
    
    newdata = pd.concat([
        data.loc[:, treat_col],
        X - param_data.iloc[0]
        ], axis = 1)
    
    n = data.shape[0]
    
    zeroes = param_data - param_data
    temp_predict = predict_data - param_data
    
    newdata = pd.concat([newdata, 
                         zeroes.to_frame().T, 
                         temp_predict.to_frame().T],
                         ignore_index = True)
    newdata.loc[n, treat_col] = 1
    newdata.loc[n+1, treat_col] = 2**(-0.5)
    
    des = dmatrix(formula_like = exog_form, 
                  data = newdata,
                  return_type = "dataframe")
    
    des.loc[n, 'Intercept'] = 0
    des.loc[n+1, 'Intercept'] = 0
    
    if method == 'pivot':
        return _pivot_JCR(X_tr = des.iloc[0:n], 
                          y_tr = data.loc[:, resp_col],
                          X_te = des.iloc[n+1],
                          c = des.iloc[n],
                          names = ['param','predict'],
                          alpha = alpha,
                          omega = omega,
                          low = low,
                          high = high)
    if method == 'conform':
        X_te = pd.concat([des.iloc[n+1].to_frame().T, 
                          des.iloc[n+1].to_frame().T], 
                          ignore_index = True)
        X_te.loc[0, 'Intercept'] = 1
        X_te.loc[1, 'Intercept'] = 1
        X_te.loc[0, treat_col] = 0
        X_te.loc[1, treat_col] = 1
        return _conform_JCR(X_tr = des.iloc[0:n], 
                            y_tr = data.loc[:, resp_col],
                            X_te = X_te,
                            treat_col = treat_col, 
                            names = ['param','predict'],
                            symmetric = symmetric,
                            cross = cross,
                            alpha = alpha,
                            n = bootstrap_size,
                            conf_k = bootstrap_coefs,
                            band_k = bootstrap_scores,
                            low = low,
                            high = high)
    
    return JCR()


def matched_JCR(formula, data, treat_col, match_col = None,
                param_data = None, 
                param = "ATT", method = "pivot", alpha = 0.05, 
                symmetric = True, cross = True,
                bootstrap_size = 100, bootstrap_coefs = 1000, 
                bootstrap_scores = 1000, omega = 1, low = None,
                high = None):
    """
    Arguments:
        formula:
            (String) R-like formula (see patsy documentation). Categorical 
            varibales must pre-encoded.
        data:
            (pandas.DataFrame) The data to be used.
        treat_col:
            (String) The column name for the treatment indicator.
        match_col:
            (String) The column name denoting matched set IDs. If matched set 
            intercepts are desired, include C(<match_col>) in the formula. If not,
            this can remain unspecified.
        param_data:
            (pandas.DataFrame) DataFrame with one row, specifying values of the
            covariates being conditioned on, if using param = "CATE"; None, 
            otherwise. Columns should NOT contain the treatment or matched set ID.
        param:
            (String) Choice of parameter, either "ATT" or "CATE".
        method:
            (String) The method to use, either "pivot" or "conform".
        alpha:
            (float) The level of significance.
        symmetric:
            (bool) Whether or not to assume symmetrical noise. Applicable only to
            method = "conform".
        cross:
            (bool) Whether or not to assume exchangeability between treatment 
            groups. Applicable only to method = "conform".
        bootstrap_size:
            (int) Number of bootstrap draws per sample, for generating the 
            confidence interval. Applicable only to method = "conform", when at
            least on of low or high is not specified.
        bootstrap_coefs:
            (int) Number of bootstrap samples, for generating the confidence 
            interval. Applicable only to method = "conform", when at least on of 
            low or high is not specified.
        bootstrap_scores:
            (int) Number of bootstrap nonconformity scores, for generating JCR 
            band. Applicable only to method = "conform".
        omega:
            (float) The desired slope of the JCR. Applicable only to method = 
            "pivot".
        low:
            (float) The lower bound of the confidence interval. By default, this
            is calculated using the data.
        high:
            (float) The upper bound of the confidence interval. By default, this
            is calculated using the data.

    Returns:
        (coverte.JCR) The constructed JCR.
    """
    
    # Determine response column from formula
    
    resp_col = formula.split('~')[0].strip()
    exog_form = formula.split('~')[1].strip()
    
    # Non-match covariates (non-response, non-treatment, non-match)
    
    X = None
    if match_col == None:
        X = data.drop(columns = [resp_col, treat_col])
    else:
        X = data.drop(columns = [resp_col, treat_col, match_col])
    
    # Initialize param_data for choice of param
    
    if param == "ATT":
        param_data = X[data[treat_col] == 1].agg('mean')
    
    predict_data = param_data
    
    # Preprocessing: Center covariates around param_data, and combine
    
    newdata = None
    if X.shape[1] == 0:
        if match_col == None:
            newdata = data.loc[:, treat_col]
        else:
            newdata = pd.concat([
                data.loc[:, treat_col],
                data.loc[:, match_col]
                ], axis = 1)
    else:
        if match_col == None:
            newdata = pd.concat([
                data.loc[:, treat_col],
                X - param_data.iloc[0]
                ], axis = 1)
        else:
            newdata = pd.concat([
                data.loc[:, treat_col],
                data.loc[:, match_col],
                X - param_data.iloc[0]
                ], axis = 1)
    
    n = data.shape[0]
    
    zeroes = param_data - param_data
    temp_predict = predict_data - param_data
    
    newdata = pd.concat([newdata, 
                         zeroes.to_frame().T, 
                         temp_predict.to_frame().T],
                        ignore_index = True)
    newdata.loc[n, treat_col] = 1
    newdata.loc[n+1, treat_col] = 2**(-0.5)
    
    if match_col != None:
        match_temp = data.loc[0, match_col]
        newdata.loc[n, match_col] = match_temp
        newdata.loc[n+1, match_col] = match_temp
    
    des = dmatrix(formula_like = exog_form, 
                  data = newdata,
                  return_type = "dataframe")
    
    des.loc[n, 'Intercept'] = 0
    des.loc[n+1, 'Intercept'] = 0
    
    if method == 'pivot':
        return _pivot_JCR(X_tr = des.iloc[0:n], 
                          y_tr = data.loc[:, resp_col],
                          X_te = des.iloc[n+1],
                          c = des.iloc[n],
                          names = ['param','predict'],
                          alpha = alpha,
                          omega = omega,
                          low = low,
                          high = high)
    if method == 'conform':
        X_te = pd.concat([des.iloc[n+1].to_frame().T, 
                          des.iloc[n+1].to_frame().T], 
                          ignore_index = True)
        X_te.loc[0, 'Intercept'] = 1
        X_te.loc[1, 'Intercept'] = 1
        X_te.loc[0, treat_col] = 0
        X_te.loc[1, treat_col] = 1
        return _conform_JCR(X_tr = des.iloc[0:n], 
                            y_tr = data.loc[:, resp_col], 
                            X_te = X_te,
                            treat_col = treat_col, 
                            names = ['param','predict'],
                            symmetric = symmetric,
                            cross = cross,
                            alpha = alpha,
                            n = bootstrap_size,
                            conf_k = bootstrap_coefs,
                            band_k = bootstrap_scores,
                            low = low,
                            high = high)

    return JCR()


def mediation_JCR(formula_response, formula_mediator, data, treat_col,
                  param_data = None,
                  param = "NDE", method = "pivot", alpha = 0.05, 
                  symmetric = True, cross = True,
                  bootstrap_size = 100, bootstrap_coefs = 1000, 
                  bootstrap_scores = 1000, omega = 1, 
                  low = None, high = None): 
    """
    Arguments:
        formula_response:
            (String) R-like formula for response model (see patsy documentation). 
            Should include the mediator as a predictor. Categorical varibales must 
            pre-encoded.
        formula_mediator:
            (String) R-like formula for mediator model (see patsy documentation). 
            Categorical varibales must pre-encoded.
        data:
            (pandas.DataFrame) The data to be used.
        treat_col:
            (String) The column name for the treatment indicator.
        param_data:
            (float) Value of the mediator used for the CDE, if param = "CDE";
            None, otherwise.
        param:
            (String) Choice of parameter, either "CDE", "NDE", "NIE", or "NE". For 
            param = "NIE" and param = "NE", method = "conform" is not supported.
        method:
            (String) The method to use, either "pivot" or "conform". For 
            param = "NIE" and param = "NE", method = "conform" is not supported.
        alpha:
            (float) The level of significance.
        symmetric:
            (bool) Whether or not to assume symmetrical noise. Applicable only to
            method = "conform".
        cross:
            (bool) Whether or not to assume exchangeability between treatment 
            groups. Applicable only to method = "conform".
        bootstrap_size:
            (int) Number of bootstrap draws per sample, for generating the 
            confidence interval. Applicable only to method = "conform", when at
            least on of low or high is not specified.
        bootstrap_coefs:
            (int) Number of bootstrap samples, for generating the confidence 
            interval. Applicable only to method = "conform", when at least on of 
            low or high is not specified.
        bootstrap_scores:
            (int) Number of bootstrap nonconformity scores, for generating JCR 
            band. Applicable only to method = "conform".
        omega:
            (float) The desired slope of the JCR. Applicable only to method = 
            "pivot".
        low:
            (float) The lower bound of the confidence interval. By default, this
            is calculated using the data.
        high:
            (float) The upper bound of the confidence interval. By default, this
            is calculated using the data.

    Returns:
        (coverte.JCR) The constructed JCR.
    """
    
    # Determine response and mediator columns from formulas
    
    resp_col = formula_response.split('~')[0].strip()
    med_col = formula_mediator.split('~')[0].strip()
    
    # treatment:mediator interaction
    
    inter_col = None
    if "{}:{}".format(treat_col, med_col) in formula_response:
        inter_col = "{}:{}".format(treat_col, med_col)
    if "{}*{}".format(treat_col, med_col) in formula_response:
        inter_col = "{}:{}".format(treat_col, med_col)
    if "{}:{}".format(med_col, treat_col) in formula_response:
        inter_col = "{}:{}".format(med_col, treat_col)
    if "{}*{}".format(med_col, treat_col) in formula_response:
        inter_col = "{}:{}".format(med_col, treat_col)
    exog_form = formula_response.split('~')[1].strip()
    
    # Covariates (non-response, non-treatment, non-mediator)
    # Preprocessing: Center covariates
    
    X = data.drop(columns = [resp_col, treat_col, med_col]).transform(
        func = lambda x: x - x.mean()
        )
    
    # Preprocessing: Center and scale mediator for choice of param
    
    temp_med = pd.Series()
    
    if param == "CDE":
        temp_med = data.loc[:, med_col] - param_data
    elif param == "NE" or param == "NDE" or param == "NIE":
        temp_data = pd.concat([
            data.loc[:, treat_col],
            data.loc[:, med_col],
            X
            ], axis = 1)
        med_result = OLS.from_formula(formula_mediator, temp_data).fit()
        med_shift = med_result.params['Intercept']
        med_scale = med_result.params[treat_col]
        temp_med = (data.loc[:, med_col] - med_shift) / med_scale
    
    # Combine
    
    newdata = pd.concat([
        data.loc[:, treat_col],
        temp_med,
        X
        ], axis = 1)
    
    n = data.shape[0]
    
    zeroes = data.iloc[0] - data.iloc[0]
    
    newdata = pd.concat([newdata, 
                         zeroes.to_frame().T,
                         zeroes.to_frame().T],
                        ignore_index = True)
    
    if param == "CDE" or param == "NE" or param == "NDE":
        newdata.loc[n, treat_col] = 1
        newdata.loc[n+1, treat_col] = 2**(-0.5)
    if param == "NE" or param == "NIE":
        newdata.loc[n, med_col] = 1
        newdata.loc[n+1, med_col] = 1
        if inter_col != None:
            newdata.loc[n, inter_col] = 1
            newdata.loc[n+1, inter_col] = 1
    
    des = dmatrix(formula_like = exog_form, 
                  data = newdata,
                  return_type = "dataframe")
    
    des.loc[n, 'Intercept'] = 0
    des.loc[n+1, 'Intercept'] = 0
    
    if method == 'pivot':
        return _pivot_JCR(X_tr = des.iloc[0:n], 
                          y_tr = data.loc[:, resp_col],
                          X_te = des.iloc[n+1],
                          c = des.iloc[n],
                          names = ['param','predict'],
                          alpha = alpha,
                          omega = omega,
                          low = low,
                          high = high)
    if method == 'conform' and (param == 'CDE' or param == 'NDE'):
        X_te = pd.concat([des.iloc[n+1].to_frame().T, 
                          des.iloc[n+1].to_frame().T], 
                          ignore_index = True)
        X_te.loc[0, 'Intercept'] = 1
        X_te.loc[1, 'Intercept'] = 1
        X_te.loc[0, treat_col] = 0
        X_te.loc[1, treat_col] = 1
        return _conform_JCR(X_tr = des.iloc[0:n], 
                            y_tr = data.loc[:, resp_col],
                            X_te = X_te,
                            treat_col = treat_col, 
                            names = ['param','predict'],
                            symmetric = symmetric,
                            cross = cross,
                            alpha = alpha,
                            n = bootstrap_size,
                            conf_k = bootstrap_coefs,
                            band_k = bootstrap_scores,
                            low = low,
                            high = high)
    
    return JCR()


def rdd_JCR(data, treat_col, resp_col, running_col, degree = 1, 
            interact_degree = 1, cutoff = 0, bw = None, method = "pivot", 
            alpha = 0.05, symmetric = True, cross = True,
            bootstrap_size = 100, bootstrap_coefs = 1000, 
            bootstrap_scores = 1000, omega = 1, low = None,
            high = None):
    """
    Arguments:
        data:
            (pandas.DataFrame) The data to be used.
        treat_col:
            (String) The column name for the treatment indicator.
        resp_col:
            (String) The column name for the resposne.
        running_col:
            (String) The column name for the running variable.
        degree:
            (int) The degree for the local polynomial.
        interact_degree:
            (int) The highest degree for the local polynomial, whose coefficient
            can differ between each side of the cutoff.
        cutoff:
            (float) The value of the running variable where the distcontinuity
            orrcurs.
        bw:
            (float) The bandwidth. Observations where the running variable is
            between cutoff - bw and cutoff + bw are included.
        method:
            (String) The method to use, either "pivot" or "conform".
        alpha:
            (float) The level of significance.
        symmetric:
            (bool) Whether or not to assume symmetrical noise. Applicable only to
            method = "conform".
        cross:
            (bool) Whether or not to assume exchangeability between treatment 
            groups. Applicable only to method = "conform".
        bootstrap_size:
            (int) Number of bootstrap draws per sample, for generating the 
            confidence interval. Applicable only to method = "conform", when at
            least on of low or high is not specified.
        bootstrap_coefs:
            (int) Number of bootstrap samples, for generating the confidence 
            interval. Applicable only to method = "conform", when at least on of 
            low or high is not specified.
        bootstrap_scores:
            (int) Number of bootstrap nonconformity scores, for generating JCR 
            band. Applicable only to method = "conform".
        omega:
            (float) The desired slope of the JCR. Applicable only to method = 
            "pivot".
        low:
            (float) The lower bound of the confidence interval. By default, this
            is calculated using the data.
        high:
            (float) The upper bound of the confidence interval. By default, this
            is calculated using the data.

    Returns:
        (coverte.JCR) The constructed JCR.
    """
    
    # Bandwidth
    
    if bw == None:
        bw = max((data[running_col] - cutoff).abs())
    
    data1 = data.drop(
        data.index[(data[running_col] - cutoff).abs() > bw]).reset_index()
    
    # Preprocessing: Center running variable around cutoff, and combine
    
    newdata = pd.DataFrame({
        treat_col: data1.loc[:, treat_col],
        running_col: data1.loc[:, running_col] - cutoff
        })
    
    n = data1.shape[0]
    
    newdata = add_constant(newdata)
    newdata.loc[n, 'const'] = 0
    newdata.loc[n, treat_col] = 1
    newdata.loc[n, running_col] = 0
    
    # Add polynomial terms
    
    for p in range(2, degree+1):
        label = '{}**{}'.format(running_col, p)
        newdata.loc[:, label] = (newdata.loc[:, running_col] - cutoff)**p
    
    for p in range(1, interact_degree+1):
        label = '{}:{}**{}'.format(treat_col, running_col, p)
        newdata.loc[:, label] = newdata.loc[:, treat_col]*(
            newdata.loc[:, running_col])**p
    
    if method == 'pivot':
        return _pivot_JCR(X_tr = newdata.iloc[0:n], 
                          y_tr = data1.loc[:, resp_col],
                          X_te = newdata.iloc[n]*2**(-0.5),
                          c = newdata.iloc[n],
                          names = ['param','predict'],
                          alpha = alpha,
                          omega = omega,
                          low = low,
                          high = high)
    if method == 'conform':
        X_te = pd.concat([newdata.iloc[n].to_frame().T, 
                          newdata.iloc[n].to_frame().T], 
                          ignore_index = True)
        X_te.loc[0, 'const'] = 1
        X_te.loc[1, 'const'] = 1
        X_te.loc[0, treat_col] = 0
        return _conform_JCR(X_tr = newdata.iloc[0:n], 
                            y_tr = data1.loc[:, resp_col], 
                            X_te = X_te,
                            treat_col = treat_col, 
                            names = ['param','predict'],
                            symmetric = symmetric,
                            cross = cross,
                            alpha = alpha,
                            n = bootstrap_size,
                            conf_k = bootstrap_coefs,
                            band_k = bootstrap_scores,
                            low = low,
                            high = high)
    
    return JCR()

def _pivot_JCR(X_tr, y_tr, X_te, c, names, alpha = 0.5, omega = 1,
               low = None, high = None):
    
    n = y_tr.shape[0]
    p = X_tr.shape[1]
    
    X_pl = pd.concat([X_tr, X_te.to_frame().T])
    
    # choose u with minimum 2-norm, subject to:
    #
    # u^T @ X_pl == c^T
    # u_te == 2**0.5/omega
    
    R,S,V = np.linalg.svd(X_pl)
    vk = V @ c
    eig = np.diag(S**(-1))
    rte_1 = R[n][0:p]
    rte_2 = R[n][p:n+1]
    
    Evk = rte_2 * (2**0.5/omega - rte_1.T @ eig @ vk) / (rte_2.T @ rte_2)
    
    u = R @ np.concatenate([eig @ vk, Evk])
    
    # fit OLS, CI, band
    
    result = OLS(endog = y_tr, exog = X_tr).fit()
    
    low1, high1 = _pivot_conf(c, result, alpha/2)
    
    if low == None:
        low = low1
    if high == None:
        high = high1
    
    d = _pivot_band(X_tr, y_tr, u[0:n], u[n], result, alpha/2)
    
    return _intersect_conf(d, low, high, names)

def _pivot_conf(c, result, alpha = 0.05):
    
    # estimate, SE, df
    
    l = c.T @ result.params
    se = np.sqrt(c.T @ result.cov_params() @ c)
    df = result.df_resid + 1
    
    # t quantile
    
    q = scipy.stats.t.ppf(q = 1 - alpha/2, df = df)
    
    return (l - q*se, l + q*se)

def _conform_JCR(X_tr, y_tr, X_te, treat_col, names, symmetric = True, 
                 cross = True, alpha = 0.05, n = 100, conf_k = 1000, 
                 band_k = 1000, low = None, high = None):
    
    # CI, band
    
    if low == None or high == None:
        low1, high1 = _boot_conf(X_tr, y_tr, treat_col, alpha/2, n, conf_k)
    
    if low == None:
        low = low1
    if high == None:
        high = high1
    
    d = None
    
    d = _new_conform_band(X_tr, y_tr, X_te, treat_col, symmetric, cross, 
                          alpha/2, band_k)
    
    return _intersect_conf(d, low, high, names)

def _boot_conf(X_tr, y_tr, treat_col, alpha = 0.05, n = 100, k = 1000):
    
    rng = np.random.default_rng()
    
    b = pd.Series(index = range(k))
    
    def gen_coef(x):
        ind_boot = rng.choice(y_tr.index,
                              size = n,
                              replace = True)
        
        return OLS(endog = y_tr.loc[ind_boot],
                   exog = X_tr.loc[ind_boot]).fit().params[treat_col]
    
    b = b.apply(gen_coef)
    
    return (np.quantile(b, alpha/2), np.quantile(b, 1-alpha/2))

def _pivot_band(X_tr, y_tr, u_tr, u_te, result = None, alpha = 0.05):
    
    # Root MSE, df
    
    if result == None:
        result = OLS(endog = y_tr, exog = X_tr).fit()
    rmse = np.sqrt(result.mse_resid)
    df = result.df_resid + 1
    
    # t quantile
    
    q1 = scipy.stats.t.ppf(q = alpha/2, df = df)
    
    # 2-norm of combined u vector
    
    u_temp = [u_te]
    u_temp.extend(u_tr)
    u_len = np.sqrt(np.sum(np.array(u_temp)**2))
    
    # u_tr^T y_tr
    
    u_y = np.sum(np.array(u_tr)*np.array(y_tr))
    
    # prediction = intercept + slope*parameter
    
    return {'low':(q1*rmse*u_len/u_te - u_y/u_te)*(2**0.5),   # low intercept
            'high':(-q1*rmse*u_len/u_te - u_y/u_te)*(2**0.5), # high intercept
            'slope':2**0.5/u_te                               # slope
            }

def _new_conform_band(X_tr, y_tr, X_te, treat_col, symmetric = True, 
                      cross = True, alpha = 0.05, k = 1000):
    
    rng = np.random.default_rng()
    
    n = y_tr.shape[0]
    
    # MSE ratio
    
    mse_r = 1
    
    if not cross:
    
        ind_control = X_tr[X_tr.loc[:, treat_col] == 0]
        ind_treat = X_tr[X_tr.loc[:, treat_col] == 1]
        
        n_0 = ind_control.shape[0]
        n_1 = ind_treat.shape[0]
        
        base_mod = OLS(endog = y_tr, exog = X_tr)
        result = base_mod.fit()
        resid_0 = result.resid[ind_control]
        resid_1 = result.resid[ind_treat]
        
        mse_r = (resid_1.T @ resid_1) / (resid_0.T @ resid_0) * (n_0 / n_1)
    
    # (1 - Leverages)**2
    
    X_pl = pd.concat([X_tr, X_te], ignore_index = True)
    
    lev = (1 - np.diag(X_pl 
                       @ np.linalg.inv(X_pl.T @ X_pl) 
                       @ (X_pl.T.to_numpy())))**2
    
    # Choosing indices of bootstrap samples
    
    ind1 = None
    ind2 = None
    
    if cross:
        ind1 = rng.choice(y_tr.index,
                          size = k,
                          replace = True)
        ind2 = rng.choice(y_tr.index,
                          size = k,
                          replace = True)
    else:
        ind1 = rng.choice(y_tr[X_tr[treat_col]==0].index,
                          size = k,
                          replace = True)
        ind2 = rng.choice(y_tr[X_tr[treat_col]==1].index,
                          size = k,
                          replace = True)
    
    def score(i):
        
        temp_mod = LinearRegression()
        temp_mod.fit(X_tr.drop(i), y_tr.drop(i))
        
        yi = y_tr.loc[i]
        yih = temp_mod.predict(X_tr.loc[i].to_frame().T)
        
        return yi - yih
    
    # raw jacknife residuals
    
    jack = pd.Series(range(n)).apply(score)
    jack1 = np.array(jack[ind1])
    jack2 = np.array(jack[ind2])
    
    nc = (jack1 - jack2) / np.sqrt(mse_r*lev[ind1] + lev[ind2])
    
    if symmetric:
        nc = np.abs(nc)
        q2 = np.quantile(nc, 1 - alpha)[0]
        q1 = -q2
    else:
        q1 = np.quantile(nc, alpha/2)[0]
        q2 = np.quantile(nc, 1-alpha/2)[0]
    
    lev_factor = np.sqrt(mse_r*lev[n+1] + lev[n])
    
    return {'low':q1*lev_factor,  # low intercept
            'high':q2*lev_factor, # high intercept
            'slope':1             # slope
            }



def _intersect_conf(d, low, high, names):
    
    ch = high - low
    
    vertex = [low, d['low'] + d['slope']*low]
    
    vectors = [[ch, d['slope']*ch],
               [0, d['high'] - d['low']]]
    
    return JCR(vertex, vectors, names = names)










