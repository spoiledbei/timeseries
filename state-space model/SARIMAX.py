#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:05:44 2018

@author: congshanzhang
State Space model
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO

# Dataset
wpi1 = requests.get('http://www.stata-press.com/data/r12/wpi1.dta').content
data = pd.read_stata(BytesIO(wpi1))
data.index = data.t

"""
Model 1: ARIMA(1,1,1); first difference the data
"""
# BIC = log(\hat{sigma}^2) + j*log(T)/T 
# It tends to choose a smaller model.
mod = sm.tsa.statespace.SARIMAX(data['wpi'],trend = 'c',order=(1,1,1))
res = mod.fit(disp=False)
print(res.summary())

"""
Model 2: ARIMA(1,1,1); log() difference the data; add seasonal term to MA part
"""
# growth rate is stationary
data['ln_wpi'] = np.log(data['wpi'])
data['D.ln_wpi'] = data['ln_wpi'].diff()

# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

# Levels
axes[0].plot(data.index._mpl_repr(), data['wpi'], '-')
axes[0].set(title='US Wholesale Price Index')

# Log difference
axes[1].plot(data.index._mpl_repr(), data['D.ln_wpi'], '-')
axes[1].hlines(0, data.index[0], data.index[-1], 'r')
axes[1].set(title='US Wholesale Price Index - difference of logs');

# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

fig = sm.graphics.tsa.plot_acf(data.ix[1:, 'D.ln_wpi'], lags=40, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(data.ix[1:, 'D.ln_wpi'], lags=40, ax=axes[1])

# build model
ar = 1          # this is the maximum degree specification
ma = (1,0,0,1)  # \epsilon_{t-1} + \epsilon_{t-4}
mod = sm.tsa.statespace.SARIMAX(data['wpi'], trend='c', order=(ar,1,ma))
res = mod.fit(disp=False)
print(res.summary())

"""
Model 3 (prefered): 1. log() transformed data; 
                    2. multiplicative seasonality; 
                    3. seasonal differencing.
                    4. include both non-seasonale and seasonal lag polynomials for 
                        both AR part and MA part.
        
        model:
        ϕ_p(L) * ϕ̃_P(L) * (1-L)^d *(1-L^s)^D * y_t = A(t) + θ_q(L) θ̃_Q(L) * ϵ_t
"""
# Data and log() transform
air2 = requests.get('http://www.stata-press.com/data/r12/air2.dta').content
data = pd.read_stata(BytesIO(air2))
data.index = pd.date_range(start=datetime(data.time[0], 1, 1), periods=len(data), freq='MS')
data['lnair'] = np.log(data['air'])

# Fit the model
# (p,d,q) = (2,1,0)
# (P,D,Q)_s = (1,1,0,12). Seasonal difference once, include 1 seasonal lag into AR part, which is (1- a*L^12)
mod = sm.tsa.statespace.SARIMAX(data['lnair'],trend='c',order=(2,1,0), seasonal_order=(1,1,0,12), simple_differencing=True)
res = mod.fit(disp=False)
print(res.summary())


"""
Model 4 (one-dim state space model): 
                 1. add exogenous regressors
                 2. use Model 3 to model residual
"""
# Dataset
friedman2 = requests.get('http://www.stata-press.com/data/r12/friedman2.dta').content
data = pd.read_stata(BytesIO(friedman2))
data.index = data.time

# Variables
data = data.ix[:'1981']
endog = data.ix['1959':'1981', 'consump']
exog = sm.add_constant(data.ix['1959':'1981', 'm2'])

# Fit the model
mod = sm.tsa.statespace.SARIMAX(endog.ix[:'1978-01-01'], exog.ix[:'1978-01-01'], order=(1,0,1))
fit_res = mod.fit(disp=False)
print(res.summary())

"""
Forecasting
"""
# create model
mod = sm.tsa.statespace.SARIMAX(endog, exog, order=(1,0,1))   # use all the data
res = mod.filter(fit_res.params)                              # use data upto 1978 (subsample) to estimate model

# In-sample one-step-ahead predictions
predict = res.get_prediction()                                
predict_ci = predict.conf_int()

# Dynamic predictions
predict_dy = res.get_prediction(dynamic='1978-01-01')
predict_dy_ci = predict_dy.conf_int()


# ===========================================
# Graph predictions
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='Personal consumption', xlabel='Date', ylabel='Billions of dollars')

# Plot data points
data.ix['1977-07-01':, 'consump'].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.ix['1977-07-01':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.ix['1977-07-01':]
ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='r', alpha=0.1)
predict_dy.predicted_mean.ix['1977-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
ci = predict_dy_ci.ix['1977-07-01':]
ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='g', alpha=0.1)

legend = ax.legend(loc='lower right')
# ===========================================

# ===========================================
# Graph prediction errors
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='Forecast error', xlabel='Date', ylabel='Forecast - Actual')

# In-sample one-step-ahead predictions and 95% confidence intervals
predict_error = predict.predicted_mean - endog
predict_error.ix['1977-10-01':].plot(ax=ax, label='One-step-ahead forecast')
ci = predict_ci.ix['1977-10-01':].copy()
ci.iloc[:,0] -= endog.loc['1977-10-01':]
ci.iloc[:,1] -= endog.loc['1977-10-01':]
ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], alpha=0.1)

# Dynamic predictions and 95% confidence intervals
predict_dy_error = predict_dy.predicted_mean - endog
predict_dy_error.ix['1977-10-01':].plot(ax=ax, style='r', label='Dynamic forecast (1978)')
ci = predict_dy_ci.ix['1977-10-01':].copy()
ci.iloc[:,0] -= endog.loc['1977-10-01':]
ci.iloc[:,1] -= endog.loc['1977-10-01':]
ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='r', alpha=0.1)

legend = ax.legend(loc='lower left');
legend.get_frame().set_facecolor('w')
# ============================================


"""
Model 5: VARMAX model
        y_t = ν + A1y_{t−1} +⋯+ Apy_{t−p} + Bx_t + ϵt + M1ϵ_{t−1} +…+ Mqϵ_{t−q}
        y_t is (k*1) state variable
"""
dta = sm.datasets.webuse('lutkepohl2', 'http://www.stata-press.com/data/r12/')
dta.index = dta.qtr
endog = dta.ix['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]

exog = endog['dln_consump']
# VARX(2) model in two endogenous variables and an exogenous series, but no constant term
mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(2,0), trend='nc', exog=exog,measurement_error=True)
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())

# impulse response function
ax = res.impulse_responses(10, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');





