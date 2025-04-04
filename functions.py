# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:18:01 2025

@author: Nicola Bartolini
"""

import numpy as np
import scipy.stats




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



