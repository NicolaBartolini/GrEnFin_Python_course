# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:18:01 2025

@author: Nicola Bartolini
"""

import numpy as np
import scipy.stats
from copy import copy 
from scipy.stats import norm
from scipy.optimize import minimize, Bounds


################ Black-Schole-Merton option pricing formulas


def Eurocall_Black_Scholes(S0, K, r, q, sigma, T):
    d1 = (np.log(S0/K) + (r - q +.5*sigma**2) * T)/(sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + (r - q -.5*sigma**2) * T)/(sigma*np.sqrt(T))
    return S0*np.exp(-q*T) * scipy.stats.norm.cdf(d1) - K * np.exp(-r*T) * scipy.stats.norm.cdf(d2)

def Europut_Black_Scholes(S0, K, r, q, sigma, T):
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S0 * np.exp(-q * T) * scipy.stats.norm.cdf(-d1)


############### Black-Scholes-Merton Greeks

def Delta(S0, K, T, r, q, sigma, option_type="call"):
    
    d1 = (np.log(S0/K) + (r -q +.5*sigma**2) * T)/(sigma*np.sqrt(T))
    
    if option_type == "call":
        return np.exp(-q*T) * scipy.stats.norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q*T) * (scipy.stats.norm.cdf(d1) - 1)

def Gamma(S0, K, T, r, q, sigma):
    
    d1 = (np.log(S0/K) + (r - q +.5*sigma**2) * T)/(sigma*np.sqrt(T))
    
    return np.exp(-q*T)* scipy.stats.norm.pdf(d1) / (S0 * sigma * np.sqrt(T))

def Vega(S0, K, T, r, q, sigma):
    
    d1 = (np.log(S0/K) + (r - q +.5*sigma**2) * T)/(sigma*np.sqrt(T))
    
    return S0 * scipy.stats.norm.pdf(d1) * np.sqrt(T) * np.exp(-q*T)

def Theta(S0, K, T, r, q, sigma, option_type="call"):
    
    d1 = (np.log(S0/K) + (r - q +.5*sigma**2) * T)/(sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + (r - q -.5*sigma**2) * T)/(sigma*np.sqrt(T))
    
    first_term = - (np.exp(-q*T) * S0 * scipy.stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if option_type == "call":
        return first_term + q * S0 * scipy.stats.norm(d1) * np.exp(-q*T) - r * K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)
    elif option_type == "put":
        return first_term - q * S0 * scipy.stats.norm(-d1) * np.exp(-q*T) + r * K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2)

def Rho(S0, K, T, r, q, sigma, option_type="call"):
    
    d2 = (np.log(S0/K) + (r - q  -.5*sigma**2) * T)/(sigma*np.sqrt(T))
    
    if option_type == "call":
        return K * T * np.exp(-r * T) * scipy.stats.norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * scipy.stats.norm.cdf(-d2)


############ GBM simulation 

def simulate_gbm(S0, r, q, sigma, T, n_steps=1, N=0):
    # This function simulate the univariate GBM
    n = 2**N
    dt = T/n_steps
    St = np.empty((n_steps+1, n)) 
    St[0] = S0 
    for t in range(1, n_steps+1):
        St[t] = St[t-1] * np.exp((r-q-.5*sigma**2)*dt + sigma*np.sqrt(dt) * np.random.normal(0,1, n))
    return St


def simulate_mgbm(S0, r, Sigma, T, n_steps, N=0):
    # This function simulate the multivariate GBM
    n_var = len(Sigma[0])
    n = int(2**N)
    St = np.empty((n, n_steps+1, n_var))
    St[:,0,:] = S0
    chol_sigma = np.linalg.cholesky(Sigma)
    dt = T/n_steps
    for i in range(0, n):
        for t in range(1, n_steps+1):
            eps = chol_sigma@np.random.normal(0, 1, (n_var, 1))
            eps = np.reshape(eps, (1,n_var))
            for j in range(0, n_var):
                St[i, t, j] = St[i, t-1, j] * np.exp((r -.5*Sigma[j, j])*dt + np.sqrt(dt) * eps[0,j])
    return St
        
#################### Cross-Currency options 

def option_struck_foreign_currency(S0, X0, K_d, rd, sigma_x, sigma_f, rho_xf, T, option_type='call'):

    s = sigma_x**2 + sigma_f**2 + 2*rho_xf*sigma_x*sigma_f 

    d1 = (np.log(X0*S0/K_d) + (rd +.5*s) * T)/(np.sqrt(s*T))
    d2 = d1 - np.sqrt(s*T)

    if option_type=='call':
        result = X0*S0 * scipy.stats.norm.cdf(d1) - K_d * np.exp(-rd*T) * scipy.stats.norm.cdf(d2)
    elif option_type=='put':
        result = K_d * np.exp(-rd * T) * scipy.stats.norm.cdf(-d2) - X0*S0 * scipy.stats.norm.cdf(-d1)
    else:
        raise ValueError('option_type only put or call')

    return result 


def MC_option_struck_foreign_currency(S0, X0, K_d, rd, sigma_x, sigma_f, rho_xf, T, n_steps, N, option_type='call'):
    
    Sigma = np.array([[sigma_x**2, sigma_x*sigma_f*rho_xf],
                 [sigma_x*sigma_f*rho_xf, sigma_f**2]])
    
    trj = simulate_mgbm(np.array([X0, S0]), rd, Sigma, T, n_steps, N)

    if option_type=='call':
        # print(np.shape(trj[:,-1,:]))
        payoff = trj[:,-1,0] * trj[:,-1, 1] - K_d 
        payoff[payoff<0] = 0
    elif option_type=='put':
        payoff = K_d - trj[:,-1,0] * trj[:,-1, 1]
        payoff[payoff<0] = 0
    else:
        raise ValueError('option_type only put or call')
    
    result = np.mean(payoff) * np.exp(-rd*T)
    
    return result   
    
    
    

################## Monte-Carlo pricing 


def EuropeanOption(St, K, r, T, payout=1):
    if payout!=1 and payout!=-1:
        raise ValueError('Value error. payout=1 for call option or payout=-1 for put option')
    payoff = payout*(St - K) 
    payoff[payoff<0] = 0
    return np.exp(-r*T) * np.mean(payoff)


def asian_call(S0, K, r, q, sigma, T, n_steps=1, N=10, average_style='aritmetic'):

    St = simulate_gbm(S0, r, q, sigma, T, n_steps, N)

    if average_style=='aritmetic':
        
        payoff = np.mean(St, axis=0) - K
        payoff[payoff<0] = 0
        return np.exp(-r*T) * np.mean(payoff)
        
    elif average_style=='geometric':

        payoff = (np.prod(St, axis=0))**(1/(n_steps+1)) - K
        payoff[payoff<0] = 0
        return np.exp(-r*T) * np.mean(payoff)

    else:
        raise ValueError("Invalid input, choose 'arithmetic' or 'geometric'")
    

def asian_put(St, K, r, q, sigma, T, average_style='aritmetic'):

    if average_style=='aritmetic':
        
        payoff = K - np.mean(St, axis=0)
        payoff[payoff<0] = 0
        return np.exp(-r*T) * np.mean(payoff)
        
    elif average_style=='geometric':
        
        payoff = K - np.exp(np.mean(np.log(St), axis=0))
        payoff[payoff<0] = 0
        return np.exp(-r*T) * np.mean(payoff)

    else:
        raise ValueError("Invalid input, choose 'arithmetic' or 'geometric'")

def european_basket_option(S0, K, w, r, Sigma, T, n_steps, N=0, option_type='call'):
    # w = array with weights of the basket
    St = simulate_mgbm(S0, r, Sigma, T, n_steps, N)

    if option_type=='call':
        payoff = np.sum(St[:,-1,:] * w, axis=1) - K
        payoff[payoff<0] = 0
    elif option_type=='put':
        payoff = K - np.sum(St[:,-1,:] * w, axis=1)
        payoff[payoff<0] = 0
    else:
        raise ValueError('option_type only put or call')
    
    result = np.mean(payoff) * np.exp(-r*T)
    
    return result   


########### Barrier options (analytic formulas Black-Scholes model)

# The European Down-and-out call

def DO_call(S0, K, L, r, sigma, T):
    
    if S0<L:
        return 0
        
    C0 = Eurocall_Black_Scholes(S0, K, r, 0, sigma, T)
    C1 = Eurocall_Black_Scholes(L**2/S0, K, r, 0, sigma, T)
    
    if L<K:
        
        C = C0 - (L/S0)**(2*(r-.5*sigma**2)/sigma**2) * C1
    
    elif L>K:
        
        d = (np.log(L/S0) + (r-.5*sigma**2)*T)/(sigma*np.sqrt(T))
        H = np.exp(-r*T)*scipy.stats.norm.cdf(d)
        
        C = C0 + (L-K) * H - (L/S0)**(2*(r-.5*sigma**2)/sigma**2) * (C1 + (L-K)*H)
    else:
        print('L=K classic Black-Scholes model')
        return C0
    return C 

# Up and out put

def UO_put(S0, K, L, r, sigma, T):

    if S0>=L:
        return 0

    P0 = Europut_Black_Scholes(S0, K, r, 0, sigma, T)
    P1 = Europut_Black_Scholes(L**2/S0, K, r, 0, sigma, T)

    if L<K:
        
        P = P0 - (L/S0)**(2*(r-.5*sigma**2)/sigma**2) * P1
    
    elif L>K:
        
        d = (np.log(L/S0) + (r-.5*sigma**2)*T)/(sigma*np.sqrt(T))
        H = np.exp(-r*T)*scipy.stats.norm.cdf(d)
        
        P = P0 + (L-K) * H - (L/S0)**(2*(r-.5*sigma**2)/sigma**2) * (P1 - (L-K)*H) + (1-(L/S0)**(2*(r-.5*sigma**2)/sigma**2))*(K-L)*np.exp(-r*T)
    else:
        print('L=K classic Black-Scholes model')
        return P0
    
    return P 


def DI_call(S0, K, L, r, sigma, T):
    # Down in call option
    C0 = Eurocall_Black_Scholes(S0, K, r, 0, sigma, T)
    C1 = Eurocall_Black_Scholes(L**2/S0, K, r, 0, sigma, T)
    if S0<=L:
        return C0
    if L<K:
        C = (L/S0)**(2*(r-.5*sigma**2)/sigma**2) * C1
    else:
        d = (np.log(L/S0) + (r-.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = (np.log(S0/L) + (r-.5*sigma**2)*T)/(sigma*np.sqrt(T))
        H = np.exp(-r*T)*scipy.stats.norm.cdf(d)
        H2 = np.exp(-r*T)*scipy.stats.norm.cdf(d2)
        C = (L/S0)**(2*(r-.5*sigma**2)/sigma**2) * (C1 + (L-K)*H) - (L-K)*H2
    return C 


###############################################################################
############################### BOND PART #####################################


################### CIR model 

def non_central_Chisquare(k, theta, sigma, t, T, x, size = 1):

    """k = speed of mean reversion th = long run memory \n

    s = volatility size = sample size (tuple or integer)"""
 
    # dr_t = k(th - r_t)dt + sigma*sqrt(r_t)dW_t
 
    d = np.round((4 * theta * k) / sigma**2, 7)
 
    c = sigma**2 * (1 - np.exp(-k * (T - t))) / (4 * k)
 
    Lambda = x * (np.exp(-k * (T - t)) / c)
    
    if d > 1:
 
        Z = np.random.normal(0, 1, size)

        X = np.random.chisquare(d-1, size)
 
        return c * ((Z + np.sqrt(Lambda))**2 + X)
 
    elif (d <= 1) and (d > 0):
 
        n = np.random.poisson(Lambda/2, size)
 
        X = np.random.chisquare(d + 2*n, size)
 
        return X * c
 
    else:

        print("Error: d <= 0")
 
        return None
 
 
def cir_conditional_mean(x0, k, theta, sigma, dt):

    return x0 * np.exp(-k * dt) + theta * (1 - np.exp(-k*dt))
 
def cir_conditional_variance(x0, k, theta, sigma, dt):

    return x0 * (sigma**2)/k * (np.exp(-k*dt) - np.exp(-2*k*dt)) + (theta*sigma**2)/(2*k) * (1 - np.exp(-k*dt))**2;

def CIR_exact(k, theta, sigma, x0, n_steps, T, N):
    n = 2**N;
    dt = T/n_steps;
    trj = np.empty((n_steps+1, n));
    trj[0] = x0;
    for i in np.arange(1, n_steps+1):
        trj[i] = non_central_Chisquare(k, theta, sigma, 0, dt, trj[i-1], n); # generating the CIR with the non central chi-square
    return trj;
 
def CIR_euler(k, theta, sigma, x0, n_steps, T, N):
    n = 2**N;
    dt = T/n_steps;
    trj = np.empty((n_steps, n));
    trj[0] = x0;
    # BM = np.sqrt(dt) * np.random.normal(0, 1, (n_steps, n));
    for i in np.arange(1, n_steps):
        nu = trj[i-1] + k * (theta - trj[i-1]) * dt + sigma * np.sqrt(trj[i-1]) * np.sqrt(dt)+np.random.normal(0,1,n);
        nu[nu<0] = 0;
        trj[i] = nu;
    return trj;

def CIR_andersen( k, theta, sigma, x0, n_steps, T, N):
    # andersen schema
    n = 2**N;
    dt = T/n_steps;
    trj = np.empty((n_steps+1, n));
    trj[0] = x0;
    uniform_sampling = np.random.uniform(0,1,(n_steps,n)); # generating random numbers with the uniform distribution 
    for i in np.arange(1,n_steps+1):
        m = cir_conditional_mean(trj[i-1], k, theta, sigma, dt);
        s_square = cir_conditional_variance(trj[i-1], k, theta, sigma, dt);
        psi = s_square/(m**2);
        
        psi_normal = copy(psi[psi<=1.5]);
        m_normal = m[psi<=1.5];
        
        b_square = 2/psi_normal - 1 + np.sqrt(2/psi_normal) * np.sqrt(2/psi_normal - 1); 
        a = m_normal/(1+b_square);
        gaussian_noise = norm.ppf(uniform_sampling[i-1]);
        noise = gaussian_noise[psi<=1.5];
        V_normal = a * (np.sqrt(b_square) + noise)**2;
        
        trj[i][psi<=1.5] = V_normal;
        psi_u = copy(psi[psi>1.5]);
        m_u = m[psi>1.5];
        
        p = (psi_u-1)/(psi_u+1);
        beta = (1-p)/m_u;
        u = uniform_sampling[i-1];
        unif_noise = u[psi>1.5];
        V_unif = 1/beta * np.log((1-p)/(1-unif_noise));
        V_unif[V_unif<0] = 0;
        trj[i][psi_u>1.5] = V_unif;
    return trj;

def CIR_bond(r, theta, k, sigma, T):
    '''r = short rata at t  theta = long run mean\n k = speed of mean reversion
    sigma = volatility of r  delta_t = maturity of the bond (year fraction)
    mkt_prices = bond prices on the market'''
 
    th = theta
 
    s = sigma
 
    h = np.sqrt( k**2 + 2 * s**2)
 
    A = ((2 * h * np.exp((k + h) * T * .5)) / (2 * h + (k + h) * (np.exp(T*h) - 1))) ** ((2 * k * th)/ s**2)
 
 
    B = (2 * (np.exp(T * h) - 1)) / (2 * h + (k + h) * (np.exp(T*h) - 1))
 
    return A * np.exp(-B * r) 

def CIR_loss_function_optim(parms, mkt_prices, T):
    '''r = short rata at t  parms[0] = th = (th = theta) long run mean\n
    parms[1] = k = speed of mean reversion\n
    parms[2] = s = volatility (sigma) of r \n
    T = maturity of the bond (year fraction)'''
    th = parms[0]
    k = parms[1]
    s = parms[2]
    r = parms[3]
 
    r_mkt = -np.log(mkt_prices)/T
 
    h = np.sqrt( k**2 + 2 * s**2)
 
    A = (2 * k * th) / s**2 * (np.log(2 * h) + (k + h) * .5 * T - np.log(2 * h + (k + h) * (np.exp(T * h) - 1)))
 
 
    B = (2 * (np.exp(T * h) - 1)) / (2 * h + (k + h) * (np.exp(T*h) - 1))
 
    e = A - B * r

    return np.sum((r_mkt + e/T)**2)


def calibrate_cir(parms, mkt_prices, T, method='L-BFGS-B'):
    
    if method=='nelder-mead':
        # The Nelder-mead method
        bounds = Bounds([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf])
        
        res = minimize(CIR_loss_function_optim,
                   parms, args = (mkt_prices, T),
                   method=method, bounds=bounds,
                   options={'xatol': 1e-8, 'disp': True})
        
    elif method=='L-BFGS-B':
        
        bounds = Bounds([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf])
       
        res = minimize(CIR_loss_function_optim,
                   parms, args = (mkt_prices, T),
                   method=method, bounds=bounds,)
    
    else:
        raise ValueError('Wrong method')
        
    return res



################### VASICEK model

def vasicek_euler(k, theta, sigma, x0, n_steps, T, N):
    
    n = 2**N;
    dt = T/n_steps;
    xt = np.empty((n_steps+1, n));
    xt[0] = x0;
    
    for i in range(1, n_steps+1):
        
        eps = np.random.normal(0,1,(1,n//2))
        eps = np.hstack((eps,-eps)) 
        
        xt[i] = xt[i-1] + k*(theta - xt[i-1])*dt + sigma*np.sqrt(dt)*eps 
    
    return xt 


def Vasicek_bond(r, theta, k, sigma, delta_t):
    '''r = short rata at t  theta = long run mean\n k = speed of mean reversion
    sigma = volatility of r  delta_t = maturity of the bond (year fraction)
    mkt_prices = bond prices on the market'''
 
    B = 1/k * (1 - np.exp( -k * delta_t ) ) # B(t,T) in Brigo-Mercurio
    A = ( (theta - sigma**2/(2 * k**2) ) * (B - delta_t)
            - sigma**2/(4 * k) * B**2 )
 
    P = np.exp(A - B * r)
 
    return P
 
 
def Vasicek_loss_function_optim(parms, mkt_prices, T, r):
    '''r = short rata at t  parm[0] = th = (th = theta) long run mean\n
    parm[1] = k = speed of mean reversion\n
    parm[2] = s = volatility (sigma) of r \n
    delta_t = maturity of the bond (year fraction)'''
    th = parms[0] # long run average
    k = parms[1] # speed of mean reversion
    s = parms[2] # sigma = vol
 
    r_mkt = -np.log(mkt_prices)/T
 
    B = 1/k * (1 - np.exp(-k * T))
 
    A = (th - s**2/(2*k**2)) * (B - T) - s**2/(4*k) * B**2
 
    e = A - B * r # The exponent in the pricing formula in Vasicek
 
    return np.sum((r_mkt + e/T)**2)


def calibrate_vasicek(parms, mkt_prices, T, r0, method='L-BFGS-B'):
    
    if method=='nelder-mead':
        # The Nelder-mead method
        bounds = Bounds([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
        res = minimize(Vasicek_loss_function_optim,
                       parms, args = (mkt_prices, T, r0),
                       method=method, bounds=bounds,
                       options={'xatol': 1e-8, 'disp': True})
        
    elif method=='L-BFGS-B':
        bounds = Bounds([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
        res = minimize(Vasicek_loss_function_optim,
                       parms, args = (mkt_prices, T, r0),
                       method=method, bounds=bounds)
    
    else:
        raise ValueError('Wrong method')
        
    return res






















