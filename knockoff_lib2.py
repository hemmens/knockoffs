#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:22:44 2023

@author: Christopher Hemmens
"""

import numpy as np
from scipy.stats import kstest, beta, gamma, norm, truncnorm, t, \
                        uniform, loglaplace, multivariate_normal#, skew, kurtosis
from scipy.optimize import minimize, NonlinearConstraint

# All the possible univariate distributions for the individual features

candidates = {'BetaUnivariate': {'str': 'beta', 'fn': beta},
              'GammaUnivariate': {'str': 'gamma', 'fn': gamma},
              'GaussianUnivariate': {'str': 'norm', 'fn': norm},
              'TruncatedGaussian': {'str': 'truncnorm', 'fn': truncnorm},
              'StudentTUnivariate': {'str': 't', 'fn': t},
              'UniformUnivariate': {'str': 'uniform', 'fn': uniform},
              'LogLaplace': {'str': 'loglaplace', 'fn': loglaplace}}

# Calculate the coskewness matrix between all n random variables of length p in an nxp matrix, m.
# Coskewness can be right- or left-handed.

def coskew(m, how='left') :
    X = m.copy()
    n, p = X.shape
    
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    Y = X.copy()
    
    if how == 'right' :
        output = np.dot(X.T,Y**2)
    else :
        output = np.dot((X**2).T,Y)
    
    return output/n

# Calculate the cokurtosis matrix between all n random variables of length p in an nxp matrix, m.
# Cokurtosis can be right-, center-, or left-handed.
    
def cokurt(m, how='center') :
    X = m.copy()
    n, p = X.shape
    
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    Y = X.copy()
    
    if how == 'right' :
        output = np.dot(X.T,Y**3)
    elif how == 'left' :
        output = np.dot((X**3).T,Y)
    else :
        output = np.dot((X**2).T,Y**2)
    
    return output/n

# Return the score of the co-moment constraint value for the pseudo-perfect knockoff optimisation algorithm
# where 0 indicates a perfect adherence to the co-moment constraint

def get_con_value(m, feature, to_zero='triangle') :
    x = m.copy()
    x -= feature
    if to_zero[:4] == 'diag' :
        np.fill_diagonal(x, 0)
    else :
        x = np.tril(x, k=-1)
        
    x = x**2
    
    return np.sum(x)

# Return the mean value constraints for the individual features

def is_mean(x, feature) :
    p = feature.shape[0]
    n = x.shape[0] // p
    xc = x.copy()
    xc = xc.reshape(n,p)
    xc = xc.mean(axis=0)
    xc -= feature
    xc = xc**2
        
    return sum(xc)
  
# Return the variance value constraints for the individual features

def is_var(x, feature) :
    p = feature.shape[0]
    n = x.shape[0] // p
    xc = x.copy()
    xc = xc.reshape(n,p)
    xc = xc.var(axis=0)
    xc -= feature
    xc = xc**2
        
    return sum(xc)
  
# Return the Kolmogorov-Smirnov value constraints for the individual features

def is_distributed(x, dists) :
    p = len(dists)
    n = x.shape[0] // p
    xc = x.copy()
    xc = xc.reshape(n,p)
    
    xc = np.array([1 - kstest(xc[:,kk],
                    candidates[dists[kk]['type'].split('.')[-1]]['str'],
                    (dists[kk]['a'], dists[kk]['b'],
                         dists[kk]['loc'], dists[kk]['scale'])).pvalue
                   if dists[kk]['type'].split('.')[-1] == 'BetaUnivariate'
                   else 
                   1 - kstest(xc[:,kk],
                            candidates[dists[kk]['type'].split('.')[-1]]['str'],
                            tuple(dists[kk].values())[:-1]).pvalue
                   for kk in range(p)])
    xc = xc**2
        
    return sum(xc)/p

# Return the total correlation match between the features and knockoffs, and between all the knockoffs

def corr_match(x, feature) :
    n, p = feature.shape
    xc = x.copy()
    xc = xc.reshape(n, p)
    corr = np.corrcoef(feature.T, xc.T)
    
    real = corr[:p,:p].copy()
    knockoff = corr[p:,p:].copy()
    corr = corr[:p,p:].copy()
    
    corr = get_con_value(corr, real, 'diag')
    knockoff = get_con_value(knockoff, real, 'triangle')
    
    return (corr + knockoff)
  
# Return the total coskewness match between the features and knockoffs, and between all the knockoffs

def cosk_match(x,feature) :
    n, p = feature.shape
    xc = x.copy()
    xc = xc.reshape(n,p)
    cosk = coskew(np.hstack((feature, xc)))
    
    real = cosk[:p,:p].copy()
    knockoff = cosk[p:,p:].copy()
    lcosk = cosk[:p,p:].copy()
    rcosk = cosk[p:,:p].copy()
    
    lcosk = get_con_value(lcosk, real, 'diag')
    rcosk = get_con_value(rcosk, real, 'diag')
    knockoff = get_con_value(knockoff, real, 'diag')
                
    return (lcosk + rcosk + knockoff)
  
# Return the total cokurtosis match between the features and knockoffs, and between all the knockoffs

def coku_match(x,feature) :
    n, p = feature.shape
    xc = x.copy()
    xc = xc.reshape(n,p)
    # Center Kurtosis
    coku_c = cokurt(np.hstack((feature, xc)), 'center')
    
    real_c = coku_c[:p,:p].copy()
    knockoff_c = coku_c[p:,p:].copy()
    ccoku = coku_c[:p,p:].copy()
    
    ccoku = get_con_value(ccoku, real_c, 'diag')
    knockoff_c = get_con_value(knockoff_c, real_c, 'triangle')
    
    # Left and Right Kurtosis
    coku = cokurt(np.hstack((feature, xc)), 'left')
    
    real = coku[:p,:p].copy()
    knockoff = coku[p:,p:].copy()
    lcoku = coku[:p,p:].copy()
    rcoku = coku[p:,:p].copy()
    
    lcoku = get_con_value(lcoku, real, 'diag')
    rcoku = get_con_value(rcoku, real, 'diag')
    knockoff = get_con_value(knockoff, real, 'diag')
                
    return (ccoku + lcoku + rcoku + knockoff_c + knockoff)

# Minimise the value of the eigenvalues when constructing the Candès knockoffs in the optimisation algorithm

def get_cov_min_eigval(x, covs_inv) :
    x_diag = np.diag(x)
    covs_t = np.dot(x_diag, np.dot(covs_inv, x_diag))
    return min(np.linalg.eigvals(2*x_diag - covs_t))

# Return the summed squared correlations between the features and their knockoffs

def get_diag_weights(x, vs) :
    return sum([y**2 for y in vs-x])

# Get the estimated Candès knockoffs assuming a Normal multivariate distribution

def get_candes(feature, dists) :
    x = feature.copy()
    n, p = x.shape
    for kk in range(p) :
        if dists[kk]['type'].split('.')[-1] == 'BetaUnivariate' :
            x[:,kk] = beta.cdf(feature[:,kk], dists[kk]['a'], dists[kk]['b'],
                               dists[kk]['loc'], dists[kk]['scale'])
        else :
            x[:,kk] = candidates[dists[kk]['type'].split('.')[-1]]['fn'].cdf(feature[:,kk],
                                                            *tuple(dists[kk].values())[:-1])
        
    x = norm.ppf(x)
    x = x.T
    
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
    for kk in range(n) :
        samples += [multivariate_normal.rvs(x[:,kk] - np.dot(x[:,kk], np.dot(covs_inv, x_diag)),
                                            2*x_diag - np.dot(x_diag, np.dot(covs_inv, x_diag)))]
            
    samples = np.array(samples)
    samples = norm.cdf(samples)
    for kk in range(p) :
        if dists[kk]['type'].split('.')[-1] == 'BetaUnivariate' :
            samples[:,kk] = beta.ppf(samples[:,kk], dists[kk]['a'], dists[kk]['b'],
                                     dists[kk]['loc'], dists[kk]['scale'])
        else :
            samples[:,kk] = candidates[dists[kk]['type'].split('.')[-1]]['fn'].ppf(samples[:,kk],
                                                                *tuple(dists[kk].values())[:-1])
    
    return samples
    
    
    
    
