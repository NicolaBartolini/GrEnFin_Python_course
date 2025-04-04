# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 12:24:13 2025

@author: Nicola Bartolini
"""


import numpy as np 

# Formulas for the root finding methods 

def bisection_method(fun, a, b, tol=1e-6, maxiter=1000):

    # see if we found a root with one of our bounds
    if abs(fun(a)) < tol:
        return a
    if abs(fun(b)) < tol:
        return b
    
    if (fun(a) * fun(b)) >= 0:
        raise ValueError("Function values at the interval endpoints must have opposite signs.")

    if b<a:
        raise ValueError('b must be grater than a')

    if fun(a) * fun(b) >= 0:
        raise ValueError("Function values at the interval endpoints must have opposite signs.")
    count = 0 
    while (b - a) / 2 > tol and count<maxiter:
        c = (a + b) / 2
        if fun(c) == 0:
            return c
        elif fun(a) * fun(c) < 0:
            b = c
        else:
            a = c
        count += 1

    if count==maxiter:
        print('Max number of iterations')
    
    return (a + b) / 2


def newton_method(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    count = 0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tol:
            return x
        if dfx == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")
        x -= fx / dfx
        count += 1
    if count>=max_iter:
        print('Maximum number of iteration reached')
    return x

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1) < tol:
            return x1
        if fx1 - fx0 == 0:
            raise ValueError("Zero denominator encountered. Secant method fails.")
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x2
    return x1


############### Numerical integration


def trapezoidal_rule(fun, a, b, n):
    x = np.linspace(a, b, n + 1)  
    result = fun(x[0]) + fun(x[-1])  # First and last terms
    dx = (b - a) / n  # Step size
    result += 2 * np.sum(fun(x[1:-1]))  # Sum of the interior points
    return (dx / 2) * result  # Multiply by dx/2



def midpoint_rule(fun, a, b, n):
    h = (b-a)/n
    x = np.linspace(a,b,n)
    mid_x = .5 * (x[0:-1] + x[1:])
    return h * np.sum(fun(mid_x)) 


def rectangle_rule(fun, a, b, n):
    dx = (b-a)/n # Compute the delta of each sub-intervals
    x = np.linspace(a, b, n, endpoint=False)
    return np.sum(fun(x)) * dx


def midpoint_double(f, a, b, c, d, nx, ny):
    hx = (b - a) / nx
    hy = (d - c) / ny
    xi = a + hx/2 + np.arange(nx) * hx
    yj = c + hy/2 + np.arange(ny) * hy
    X, Y = np.meshgrid(xi, yj, indexing="ij")
    I = np.sum(f(X, Y)) * hx * hy
    return I

def midpoint_triple(fun, a, b, c, d, e, f, nx, ny, nz):
    hx = (b - a) / nx
    hy = (d - c) / ny
    hz = (f - e) / nz
    xi = a + hx/2 + np.arange(nx) * hx
    yj = c + hy/2 + np.arange(ny) * hy
    zk = c + hz/2 + np.arange(nz) * hz
    X, Y, Z = np.meshgrid(xi, yj, zk, indexing="ij")
    I = np.sum(fun(X, Y, Z)) * hx * hy *hz
    return I 





