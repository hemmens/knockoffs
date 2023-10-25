#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:22:44 2023

@author: Christopher Hemmens
"""

import numpy as np
from scipy.stats import kstest

def coskew(x,y,how='left') :
    xc = x.copy()
    yc = y.copy()
    
    x_std = np.std(xc)
    y_std = np.std(yc)
    
    xc -= np.mean(xc)
    yc -= np.mean(yc)
    
    if how == 'right' :
        yc = yc**2
        output = np.dot(xc,yc)/y_std
    else :
        xc = xc**2
        output = np.dot(xc,yc)/x_std
        
    output /= x_std
    output /= y_std
    output /= x.shape[0]
    
    return output
    
def cokurt(x,y,how='center') :
    xc = x.copy()
    yc = y.copy()
    
    x_std = np.std(xc)
    y_std = np.std(yc)
    
    xc -= np.mean(xc)
    yc -= np.mean(yc)
    
    if how == 'right' :
        yc = yc**3
        output = np.dot(xc,yc)/(y_std**2)
    elif how == 'left' :
        xc = xc**3
        output = np.dot(xc,yc)/(x_std**2)
    else :
        xc = xc**2
        yc = yc**2
        output = np.dot(xc,yc)/(x_std*y_std)
        
    output /= x_std
    output /= y_std
    output /= x.shape[0]
    
    return output

def is_mean(x,p,n) :
    max_value = 0
    for k in range(p) :
        value = x[k*n:(k+1)*n].mean() - 1/2
        max_value = max(max_value, value**2)
        
    return max_value

def is_var(x,p,n) :
    max_value = 0
    for k in range(p) :
        value = x[k*n:(k+1)*n].var() - 1/12
        max_value = max(max_value, value**2)
        
    return max_value

def is_uniform(x,p,n) :
    min_value = 1
    for k in range(p) :
        ks = kstest(x[k*n:(k+1)*n], 'uniform')
        min_value = min(min_value, ks.pvalue)
        
    return min_value

def cov_match(x,p,n,feature,moment) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 != k1 :
                value = (np.cov(x[k0*n:(k0+1)*n], feature[k1,:])[0,1] \
                                                     - moment[(k0,k1)])**2
                output += value
                
    return output

def cov_ko_match(x,p,n,moment) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 < k1 :
                value = (np.cov(x[k0*n:(k0+1)*n], x[k1*n:(k1+1)*n])[0,1] \
                                                     - moment[(k0,k1)])**2
                output += value
                
    return output

def cosk_match(x,p,n,feature,moment,how) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 != k1 :
                value = (coskew(x[k0*n:(k0+1)*n], feature[k1,:], how) \
                                                 - moment[(k0,k1,how[0])])**2
                output += value
                
    return output

def cosk_ko_match(x,p,n,moment,how) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 < k1 :
                value = (coskew(x[k0*n:(k0+1)*n], x[k1*n:(k1+1)*n], how) \
                                                 - moment[(k0,k1,how[0])])**2
                output += value
                
    return output

def coku_match(x,p,n,feature,moment,how) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 != k1 :
                value = (cokurt(x[k0*n:(k0+1)*n], feature[k1,:], how) \
                                                 - moment[(k0,k1,how[0])])**2
                output += value
                
    return output

def coku_ko_match(x,p,n,moment,how) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 < k1 :
                value = (cokurt(x[k0*n:(k0+1)*n], x[k1*n:(k1+1)*n], how) \
                                                 - moment[(k0,k1,how[0])])**2
                output += value
                
    return output