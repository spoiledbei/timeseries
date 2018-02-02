#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:20:56 2018

@author: congshanzhang
Dynamic Factor model
"""
import os

path = '/Users/congshanzhang/Documents/Office/metrics_time series/state-space models'
os.chdir(path)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
from extendedDFM import ExtendedDFM

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# Dataset
start = '1979-01-01'
end = '2014-12-01'
indprod = DataReader('IPMAN', 'fred', start=start, end=end)
income = DataReader('W875RX1', 'fred', start=start, end=end)
sales = DataReader('CMRMTSPL', 'fred', start=start, end=end)
emp = DataReader('PAYEMS', 'fred', start=start, end=end)

dta = pd.concat((indprod, income, sales, emp), axis=1)
dta.columns = ['indprod', 'income', 'sales', 'emp']

# plot time series
dta.ix[:, 'indprod':'emp'].plot(subplots=True, layout=(2, 2), figsize=(15, 6));

# Create log-differenced series
dta['dln_indprod'] = (np.log(dta.indprod)).diff() * 100
dta['dln_income'] = (np.log(dta.income)).diff() * 100
dta['dln_sales'] = (np.log(dta.sales)).diff() * 100
dta['dln_emp'] = (np.log(dta.emp)).diff() * 100

# De-mean and standardize
dta['std_indprod'] = (dta['dln_indprod'] - dta['dln_indprod'].mean()) / dta['dln_indprod'].std()
dta['std_income'] = (dta['dln_income'] - dta['dln_income'].mean()) / dta['dln_income'].std()
dta['std_sales'] = (dta['dln_sales'] - dta['dln_sales'].mean()) / dta['dln_sales'].std()
dta['std_emp'] = (dta['dln_emp'] - dta['dln_emp'].mean()) / dta['dln_emp'].std()

"""
Dynamic factor model:
            y_t = Λf_t + Bx_t + u_t 
            f_t = A1f_{t−1}+⋯+Apf_{t−p} + η_t   η_t∼N(0,I)
            u_t = C1u_{t−1}+⋯+C1f_{t−q} + εt    εt∼N(0,Σ)  Σ is diagonal
"""
# Get the endogenous data
endog = dta.ix['1979-02-01':, 'std_indprod':'std_emp']

# Create the model
mod = sm.tsa.DynamicFactor(endog, k_factors=2, factor_order=2, error_order=2,error_cov_type = 'diagonal')
#Note:  k_factors = 1 - (there is 1 unobserved factor)
#       factor_order = 2 - (it follows an AR(2) process)
#       error_var = False - (the errors evolve as independent AR processes rather than jointly as a VAR - note that this is the default option, so it is not specified below)
#       error_order = 2 - (the errors are autocorrelated of order 2: i.e. AR(2) processes)
#       error_cov_type = 'diagonal' - (the innovations are uncorrelated; this is again the default)


# Multivariate models can have a relatively large number of parameters, 
# and it may be difficult to escape from local minima to find the maximized 
# likelihood. In an attempt to mitigate this problem, 
# I perform an initial maximization step (from the model-defined starting paramters) 
# using the modified Powell method available in Scipy (see the minimize documentation 
# for more information). The resulting parameters are then used as starting 
# parameters in the standard LBFGS optimization method.
# two-step estimation procedure
initial_res = mod.fit(method='powell', disp=False)
res = mod.fit(initial_res.params)
print(res.summary(separate_params=False))


# Estimated factors through Kalman Filter
# 1. The sign-related identification issue described above.
# 2. Since the data was differenced, the estimated factor explains the variation in the differenced data, not the original data.
fig, ax = plt.subplots(figsize=(13,3))

# Plot the factor
dates = endog.index._mpl_repr()
ax.plot(dates, res.factors.filtered[0], label='Factor')
ax.plot(dates, res.factors.filtered[1], label='Factor')
ax.legend()

# Retrieve and also plot the NBER recession indicators
rec = DataReader('USREC', 'fred', start=start, end=end) # dummy whether there is a recession
ylim = ax.get_ylim()
ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4,0], facecolor='k', alpha=0.1);


# Explanatory power of each factor on each variables
res.plot_coefficients_of_determination(figsize=(8,2));

"""
step 4) Coincident Index: 
        Create an interpretable series from factors
"""
# the coincident index published by the Federal Reserve Bank of Philadelphia (USPHCI on FRED)
usphci = DataReader('USPHCI', 'fred', start='1979-01-01', end='2014-12-01')['USPHCI']
usphci.plot(figsize=(13,3));

# transform the factor into the original scale
dusphci = usphci.diff()[1:].values
def compute_coincident_index(mod, res, fac = 0):
    # Estimate W(1)
    spec = res.specification
    design = mod.ssm['design']
    transition = mod.ssm['transition']
    ss_kalman_gain = res.filter_results.kalman_gain[:,:,-1]
    k_states = ss_kalman_gain.shape[0]

    W1 = np.linalg.inv(np.eye(k_states) - np.dot(
        np.eye(k_states) - np.dot(ss_kalman_gain, design),
        transition
    )).dot(ss_kalman_gain)[0]

    # Compute the factor mean vector
    factor_mean = np.dot(W1, dta.ix['1972-02-01':, 'dln_indprod':'dln_emp'].mean())
    
    # Normalize the factors
    factor = res.factors.filtered[fac]
    factor *= np.std(usphci.diff()[1:]) / np.std(factor)

    # Compute the coincident index
    coincident_index = np.zeros(mod.nobs+1)
    # The initial value is arbitrary; here it is set to
    # facilitate comparison
    coincident_index[0] = usphci.iloc[0] * factor_mean / dusphci.mean()
    for t in range(0, mod.nobs):
        coincident_index[t+1] = coincident_index[t] + factor[t] + factor_mean
    
    # Attach dates
    coincident_index = pd.Series(coincident_index, index=dta.index).iloc[1:]
    
    # Normalize to use the same base year as USPHCI
    coincident_index *= (usphci.ix['1992-07-01'] / coincident_index.ix['1992-07-01'])
    
    return coincident_index


# plot the calculated coincident index along with the US recessions 
# and the comparison coincident index USPHCI.
fig, ax = plt.subplots(figsize=(13,3))

# Compute the index
coincident_index0 = compute_coincident_index(mod, res, fac=0)
coincident_index1 = compute_coincident_index(mod, res, fac=1)

# Plot the factor
dates = endog.index._mpl_repr()
ax.plot(dates, coincident_index0, label='Coincident index 0')
ax.plot(dates, coincident_index1, label='Coincident index 1')
ax.plot(usphci.index._mpl_repr(), usphci, label='USPHCI')
ax.legend(loc='lower right')

# Retrieve and also plot the NBER recession indicators
ylim = ax.get_ylim()
ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4,0], facecolor='k', alpha=0.1);


"""
step 5) extend DFM for model comparison (code included in the same folder)
"""
# Create the model
extended_mod = ExtendedDFM(endog)
initial_extended_res = extended_mod.fit(maxiter=1000, disp=False)
extended_res = extended_mod.fit(initial_extended_res.params, method='nm', maxiter=1000)
print(extended_res.summary(separate_params=False))

extended_res.plot_coefficients_of_determination(figsize=(8,2));


fig, ax = plt.subplots(figsize=(13,3))

# Compute the index
extended_coincident_index = compute_coincident_index(extended_mod, extended_res)

# Plot the factor
dates = endog.index._mpl_repr()
ax.plot(dates, coincident_index0, '-', linewidth=1, label='Basic model')
ax.plot(dates, extended_coincident_index, '--', linewidth=3, label='Extended model')
ax.plot(usphci.index._mpl_repr(), usphci, label='USPHCI')
ax.legend(loc='lower right')
ax.set(title='Coincident indices, comparison')

# Retrieve and also plot the NBER recession indicators
ylim = ax.get_ylim()
ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4,0], facecolor='k', alpha=0.1);


"""
step 6) Forecasting
"""
# create forecasting model
mod = sm.tsa.DynamicFactor(endog, k_factors=2, factor_order=2, error_order=2,error_cov_type = 'diagonal')
# use subsample to estimate model
mod_sub = sm.tsa.DynamicFactor(endog.ix[:'2011-12-01'], k_factors=2, factor_order=2, error_order=2,error_cov_type = 'diagonal')
initial_res = mod_sub.fit(method='powell', disp=False)
fit_res = mod.fit(initial_res.params)

res = mod.filter(fit_res.params)                            

# In-sample one-step-ahead predictions
predict = res.get_prediction()                                

# Dynamic predictions
predict_dy = res.get_prediction(dynamic='2012-05-01')

# ===========================================
# Graph predictions
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='Personal consumption', xlabel='Date', ylabel='Billions of dollars')

# Plot data points
dta.ix['2011-10-01':, 'std_indprod'].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.ix['2011-12-01':,0].plot(ax=ax, style='r--', label='One-step-ahead forecast')

predict_dy.predicted_mean.ix['2011-12-01':,0].plot(ax=ax, style='g', label='Dynamic forecast (2012)')

legend = ax.legend(loc='lower right')
# ===========================================


