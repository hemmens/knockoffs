#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:22:44 2023

@author: Christopher Hemmens
"""

import numpy as np
from scipy.stats import kstest, norm, multivariate_normal#, skew, kurtosis
from scipy.optimize import minimize, NonlinearConstraint

def coskew(m,how='left') :
    X = m.copy()
    p, n = X.shape
    
    xm = X.mean(axis=1)
    xs = X.std(axis=1)
    
    xm = np.tile(xm,n).reshape(n,p).T
    xs = np.tile(xs,n).reshape(n,p).T
    X -= xm
    X /= xs
    Y = X.copy()
    
    if how == 'right' :
        output = np.dot(X,(Y**2).T)
    else :
        output = np.dot(X**2,Y.T)
    
    return output/n
    
def cokurt(m,how='center') :
    X = m.copy()
    p, n = X.shape
    
    xm = X.mean(axis=1)
    xs = X.std(axis=1)
    
    xm = np.tile(xm,n).reshape(n,p).T
    xs = np.tile(xs,n).reshape(n,p).T
    X -= xm
    X /= xs
    Y = X.copy()
    
    if how == 'right' :
        output = np.dot(X,(Y**3).T)
    elif how == 'left' :
        output = np.dot(X**3,Y.T)
    else :
        output = np.dot(X**2,(Y**2).T)
    
    return output/n

def get_con_value(m, feature, to_zero='triangle') :
    x = m.copy()
    x -= feature
    if to_zero[:4] == 'diag' :
        np.fill_diagonal(x, 0)
    else :
        x = np.tril(x, k=-1)
        
    x = x**2
    
    return np.sum(x)

def is_mean(x,p,n) :
    xc = x.copy().reshape(p,n)
    xc = xc.mean(axis=1)
    xc -= 1/2
    xc = xc**2
        
    return max(xc)

def is_var(x,p,n) :
    xc = x.copy().reshape(p,n)
    xc = xc.var(axis=1)
    xc -= 1/12
    xc = xc**2
        
    return max(xc)

def is_uniform(x,p,n) :
    xc = x.copy().reshape(p,n)
    values = np.apply_along_axis(kstest,1,xc,cdf='uniform')[:,1]
        
    return min(values)

def cov_match(x,p,n,feature) :
    xc = x.copy().reshape(p,n)
    cov = np.cov(xc,feature)
    
    real = cov[p:,p:].copy()
    knockoff = cov[:p,:p].copy()
    cov = cov[:p,p:].copy()
    
    cov = get_con_value(cov, real, 'diag')
    knockoff = get_con_value(knockoff, real, 'triangle')
    
    return (cov + knockoff)

def cosk_match(x,p,n,feature) :
    xc = x.copy().reshape(p,n)
    cosk = coskew(np.vstack((xc,feature)))
    
    real = cosk[p:,p:].copy()
    knockoff = cosk[:p,:p].copy()
    lcosk = cosk[:p,p:].copy()
    rcosk = cosk[p:,:p].copy()
    
    lcosk = get_con_value(lcosk, real, 'diag')
    rcosk = get_con_value(rcosk, real, 'diag')
    knockoff = get_con_value(knockoff, real, 'diag')
                
    return (lcosk + rcosk + knockoff)

def coku_match(x,p,n,feature) :
    xc = x.copy().reshape(p,n)
    # Center Kurtosis
    coku_c = cokurt(np.vstack((xc,feature)), 'center')
    
    real_c = coku_c[p:,p:].copy()
    knockoff_c = coku_c[:p,:p].copy()
    ccoku = coku_c[:p,p:].copy()
    
    ccoku = get_con_value(ccoku, real_c, 'diag')
    knockoff_c = get_con_value(knockoff_c, real_c, 'triangle')
    
    # Left and Right Kurtosis
    coku = cokurt(np.vstack((xc,feature)), 'left')
    
    real = coku[p:,p:].copy()
    knockoff = coku[:p,:p].copy()
    lcoku = coku[:p,p:].copy()
    rcoku = coku[p:,:p].copy()
    
    lcoku = get_con_value(lcoku, real, 'diag')
    rcoku = get_con_value(rcoku, real, 'diag')
    knockoff = get_con_value(knockoff, real, 'diag')
                
    return (ccoku + lcoku + rcoku + knockoff_c + knockoff)

def get_cov_min_eigval(x, covs_inv) :
    x_diag = np.diag(x)
    covs_t = np.dot(x_diag, np.dot(covs_inv, x_diag))
    return min(np.linalg.eigvals(2*x_diag - covs_t))

def get_diag_weights(x, vs) :
    return sum([y**2 for y in vs-x])

def get_candes(feature) :
    x = feature.copy()
    x = norm.ppf(x)
    
    covs = np.cov(x)
    vs = np.diag(covs)
    covs_inv = np.linalg.inv(covs)

    rng = np.random.default_rng(24)

    while True :
        thresh = rng.uniform(0,1e-4)
        min_eigval = NonlinearConstraint(lambda x: get_cov_min_eigval(x, covs_inv),
                                         thresh, np.inf)
        
        res = minimize(lambda x: get_diag_weights(x, vs),
                       np.zeros(x.shape[0]),
                       constraints=min_eigval)
        
        if get_cov_min_eigval(res.x, covs_inv) >= 0 :
            break
        
    x_diag = np.diag(res.x)

    samples = []
    for kk in range(x.shape[1]) :
        samples += [multivariate_normal.rvs(x[:,kk] - np.dot(x[:,kk], np.dot(covs_inv, x_diag)),
                                            2*x_diag - np.dot(x_diag, np.dot(covs_inv, x_diag)))]
            
    samples = np.array(samples)
    samples = norm.cdf(samples)
    
    return samples.T
    
    
    
    