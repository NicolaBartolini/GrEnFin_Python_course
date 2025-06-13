# -*- coding: utf-8 -*-
"""
Created on Fri May  2 10:26:11 2025

@author: Nicola
"""

from scipy.stats import norm
import numpy as np 
import pandas as pd
import scipy.stats
from copy import copy
from math import pi
from scipy.optimize import minimize, Bounds
import scipy

####### Simulation functions


def AR_simulate(r0, mu, beta, n_steps, n=1, DoF=4, density='normal'):
    # simulate an autoregressive model (AR)
    # r0 : numpy array of initial values (len(r0)==p) shape r0=(p, )
    p = len(beta)
    rt = np.empty((n_steps, n)) # memory pre-allocation
    rt[0:p,:] = r0
    for i in range(p, n_steps):
        if density=='normal':
            rt[i,:] = mu + beta @ rt[i-p:i,:] + np.random.normal(0,1,(1,n))
        elif density=='t' or density=='student-t':
            rt[i,:] = mu + beta @ rt[i-p:i,:] + np.random.standard_t(DoF, (1,n))
        else:
            raise ValueError('Wrong density')
    return rt


def MA_simulate(r0, mu, phi, n_steps, n=1):
    # simulate a moving average (MA) models
    # r0 : numpy array of initial values (len(r0)==p) shape r0=(p, )
    q = len(phi) # MA order
    
    rt = np.empty((n_steps, n)) # memory pre-allocation
    rt[0:q,:] = r0

    eps = np.empty((n_steps, n)) # memory pre-allocation
    eps[0:q,:] = r0
    
    for i in range(q, n_steps):
        eps[i,:] = np.random.normal(0,1,(1,n))
        rt[i,:] = mu + phi @ eps[i-q:i,:] + eps[i,:]
    return rt, eps


def mAR(r0, mu, beta, Sigma, n_steps, n=1):
    # multivariate AR model with Gaussian distribution
    # r0 : numpy array of initial values (len(r0)==p) shape r0=(p, )
    # m = number of variables
    p, m = np.shape(r0)
    
    rt = np.empty((n, n_steps, m)) # memory pre-allocation
    # rt[:,0:p,:] = r0

    chol_sigma = np.linalg.cholesky(Sigma)
    
    for i in range(0,n):
        trj = copy(rt[i])
        trj[0:p,:] = r0
        # print(trj)
        for t in range(p, n_steps):
            x = trj[t-p:t]
            X = np.flip(x.T, axis=1).flatten()
            X = np.reshape(X, (1, len(X)))
            trj[t,:] = mu + X@beta + (chol_sigma@np.random.normal(0,1,(m,1))).T
        rt[i] = trj
    return rt

############# GARCH model 

def garch_simulate(p, q, r0, sigma_square0, mu, omega, alphas, phis, n_steps, DoF=4, density='normal', N=1):

    m = max([p, q])
        
    rt = np.empty((n_steps, N))
    rt[0:m] = r0
    
    sigma_square = np.empty((n_steps, N))
    sigma_square[0:m] = sigma_square0
    
    eps = np.empty((n_steps, N))
    if density=='normal' or density=='gauss':
        eps[0:m] = np.random.normal(0, 1, (m,N))
    elif density=='t' or density=='student-t':
        eps[0:m] = np.random.standard_t(DoF, (m,N))

    for i in range(m, n_steps):
        eps[i] = np.random.normal(0, 1, (1,N))
        z = rt[i-p:i] - mu
        sigma_square[i] = omega + np.sum(alphas*z**2, axis=0) + np.sum(phis*sigma_square[i-q:i], axis=0)
        rt[i,:] = mu + np.sqrt(sigma_square[i]) * eps[i]

    return rt, sigma_square

def vol_filter(p, q, rt, mu, omega, alphas, phis):
    
    m = max([p, q])
    
    sigma_square = np.empty(len(rt))
    sigma_square[0:m] = np.var(rt)
    
    eps = np.empty(len(rt))
    eps[0:m] = (rt[0:m]-np.mean(rt))/np.std(rt)
    
    for i in range(m, len(rt)):
        
        z = rt[i-p:i] - mu
        sigma_square[i] = omega + np.sum(alphas*z**2, axis=0) + np.sum(phis*sigma_square[i-q:i], axis=0)
        
        eps[i] = (rt[i] - mu) / np.sqrt(sigma_square[i])
        
    return sigma_square, eps


def garch_predict_one_step(p, q, sigma_square0, eps0, mu, omega, alphas, phis):
    
    rt = mu
    vol = np.empty(1)
   
    vol = omega + np.sum(alphas[0:p]*eps0) + np.sum(phis*sigma_square0)
    
    return rt, max(vol, 1e-6)
    

def garch_loglike(rt, p, q, mu=0, omega=0, alphas=0, phis=0, sigma_square0=0.0001, DoF=4, density='normal'):
        
    # mu = intercept of the time series rt
    # omega = intercept of the volatility process
    # gammas = coefficients for the exogenous variables
    # alphas = coefficients for the square noise in the vol process
    # phis = coefficients for the lagged volatilities in the vol process
    # rt = observed time series
    
    result = 0;
            
    m = max([p, q])
    
    eps = np.empty(len(rt));
    eps[0:m] = 0
    sigma_square = np.empty(len(rt));
    sigma_square[0:m] = sigma_square0
    
    for i in range(m,len(rt)):
                        
        z = rt[i-p:i] - mu
        sigma_square[i] = omega + np.sum(alphas*z**2, axis=0) + np.sum(phis*sigma_square[i-q:i], axis=0)
        min_sigma = 1e-8
        sigma_square[i] = max(sigma_square[i], min_sigma)
        
        if sigma_square[i]<10**(-6): # To manage small values
            sigma_square[i] = np.min(sigma_square[:i])
        
        eps[i] = (rt[i] - mu) / np.sqrt(sigma_square[i])
        
        if density=='normal' or density=='gauss':
            
            result += -.5*np.log(2*pi) - 0.5*np.log(sigma_square[i]) -.5*eps[i]**2
            
        elif density=='t' or density=='student-t':
            
            A = scipy.special.gamma(.5*(DoF+1))/(np.sqrt(pi*DoF*sigma_square[i])*scipy.special.gamma(DoF*.5))
            A = np.log(A);
            
            B = -0.5*(DoF+1)*np.log(1 + eps[i]**2/DoF)
            
            # result += scipy.stats.t.logpdf(eps[i], DoF)
            result += A + B;
        else:
            raise ValueError('Wrong density')
    
    return result;

def garch_objfun(params, rt, p, q, density):
        
    # mu = params[0]
    # omega = params[1]
    # alphas = params[2 : 2 + p]
    # phis = params[2 + p : 2 + p + q]
    
    # DoF=4 
    # if density=='t' or density=='student-t':
    #     DoF = params[-2]
    
    # sigma_square0 = params[-1]
    
    mu = params[0]
    # omega = np.exp(params[1])
    # alphas = np.exp(params[2 : 2 + p])
    # phis = np.exp(params[2 + p : 2 + p + q])
    
    MAX_EXP = 10  # adjust if needed
    
    omega = np.exp(np.clip(params[1], -MAX_EXP, MAX_EXP))
    alphas = np.exp(np.clip(params[2 : 2 + p], -MAX_EXP, 0))
    phis = np.exp(np.clip(params[2 + p : 2 + p + q], -MAX_EXP, 0))
    
    DoF=None 
    if density=='t' or density=='student-t':
        DoF = np.exp(params[-2]) 
        # min_dof = 2.1
        # max_dof = 3.5
        # DoF = np.exp(np.clip(params[-2], min_dof, max_dof))
            
    sigma_square0 = np.exp(params[-1])
    
    penalty = np.sum(np.square(params)) * 1e-4  # L2 penalty
    
    return -garch_loglike(rt, p, q, mu, omega, alphas, phis, sigma_square0, DoF, density) + penalty
    

def fit(params, rt, p, q, density, method='BFGS'):
    
    
    if method=='BFGS':
        
        res = minimize(garch_objfun, params, args=(rt, p, q, density), 
                        method='BFGS', jac='2-points',
                        options={'disp':False})
        
    elif method=='L-BFGS-B':
        
        lb = np.ones(len(params)) * -10
        ub = np.zeros(len(params)) 
        ub[0] = 10 
        
        bounds = Bounds(lb, ub)
        
        res = minimize(garch_objfun, params, args=(rt, p, q, density), 
                        method='L-BFGS-B', jac='2-points', bounds=bounds,
                        options={'disp':False})
        
        if density=='t' or density=='student-t':
            
            lb[-2] = 1
            ub[-2] = 3.5
            
            res = minimize(garch_objfun, params, args=(rt, p, q, density), 
                            method=method, jac='2-points',
                            options={'disp':False})
                

    mu = res.x[0]
    omega = np.exp(res.x[1])
    alphas = np.exp(res.x[2 : 2 + p])
    phis = np.exp(res.x[2 + p : 2 + p + q])
    sigma_square0 = np.exp(res.x[-1])
    
    # DoF=4 
    if density=='t' or density=='student-t':
        DoF = np.exp(res.x[-2])
    
    index = ['mu','omega']
    for i in range(0, p):
        index.append('alpha_'+str(i+1))
        
    for i in range(0, q):
        index.append('phi_'+str(i+1))  
    
    if density=='t' or density=='student-t':
        index.append('DoF')
    
    index.append('sigma_square0')

    params_result = np.array([mu, omega])
    
    if density=='t' or density=='student-t':
        params_result = np.hstack((params_result, alphas, phis, np.array([DoF, sigma_square0])))
    else:
        params_result = np.hstack((params_result, alphas, phis, np.array([sigma_square0])))
    
    df_params_result = pd.DataFrame(data=params_result, index=index)

    sigma_square, eps = vol_filter(p, q, rt, mu, omega, alphas, phis)
    
    return df_params_result, res.fun, res.jac, sigma_square, eps


############## OLS fitting functions 


def OLS(X, y):
    beta_hat = np.linalg.inv(X.T@X)@X.T@y
    # computing the residuals
    eps = y - X@beta_hat 
    ols_var = np.var(eps) * np.linalg.inv(X.T@X)
    return beta_hat, eps, ols_var


