# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:29:59 2023

@author: Finch
"""

import numpy as np
from scipy.stats import kstest
from scipy.optimize import minimize, NonlinearConstraint#, LinearConstraint

rng = np.random.default_rng(24)

test = rng.uniform(0,1,(4,1000))

pt = test[0,:]/sum(test[0,:])
test[0,:] = rng.choice(test[0,:], size=test.shape[1], replace=False, p=pt)

pt = [1 - x for x in test[1,:]]/sum([1 - x for x in test[1,:]])
test[1,:] = rng.choice(test[1,:], size=test.shape[1], replace=False, p=pt)

pt = [abs(x-0.5) for x in test[2,:]]/sum(abs(x-0.5) for x in test[2,:])
test[2,:] = rng.choice(test[2,:], size=test.shape[1], replace=False, p=pt)

p, n = test.shape

def coskew(x,y,how='left') :
    x0 = x.copy()
    y0 = y.copy()
    
    x_std = np.std(x0)
    y_std = np.std(y0)
    
    x0 -= np.mean(x0)
    y0 -= np.mean(y0)
    
    if how == 'right' :
        y0 = y0**2
        output = np.dot(x0,y0)/y_std
    else :
        x0 = x0**2
        output = np.dot(x0,y0)/x_std
        
    output /= x_std
    output /= y_std
    output /= x.shape[0]
    
    return output
    
def cokurt(x,y,how='center') :
    x0 = x.copy()
    y0 = y.copy()
    
    x_std = np.std(x0)
    y_std = np.std(y0)
    
    x0 -= np.mean(x0)
    y0 -= np.mean(y0)
    
    if how == 'right' :
        y0 = y0**3
        output = np.dot(x0,y0)/(y_std**2)
    elif how == 'left' :
        x0 = x0**3
        output = np.dot(x0,y0)/(x_std**2)
    else :
        x0 = x0**2
        y0 = y0**2
        output = np.dot(x0,y0)/(x_std*y_std)
        
    output /= x_std
    output /= y_std
    output /= x.shape[0]
    output -= 3
    
    return output

moments = {}
moments[2] = {}
moments[3] = {}
moments[4] = {}
for k0 in range(p) :
    for k1 in range(p) :
        if k0 < k1 :
            moments[2][(k0,k1)] = np.cov(test[k0,:], test[k1,:])[0,1]
            moments[2][(k1,k0)] = moments[2][(k0,k1)]
            
            moments[3][(k0,k1,'l')] = coskew(test[k0,:], test[k1,:], 'left')
            moments[3][(k0,k1,'r')] = coskew(test[k0,:], test[k1,:], 'right')
            moments[3][(k1,k0,'l')] = moments[3][(k0,k1,'r')]
            moments[3][(k1,k0,'r')] = moments[3][(k0,k1,'l')]
            
            moments[4][(k0,k1,'l')] = cokurt(test[k0,:], test[k1,:], 'left')
            moments[4][(k0,k1,'r')] = cokurt(test[k0,:], test[k1,:], 'right')
            moments[4][(k0,k1,'c')] = cokurt(test[k0,:], test[k1,:], 'center')
            
            moments[4][(k1,k0,'l')] = moments[4][(k0,k1,'r')]
            moments[4][(k1,k0,'r')] = moments[4][(k0,k1,'l')]
            moments[4][(k1,k0,'c')] = moments[4][(k0,k1,'c')]

def is_uniform(x) :
    min_value = 1
    for k in range(p) :
        ks = kstest(x[k*n:(k+1)*n], 'uniform')
        min_value = min(min_value, ks.pvalue)
        
    return min_value

def cov_match(x) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 != k1 :
                value = (np.cov(x[k0*n:(k0+1)*n], test[k1,:])[0,1] \
                                                 - moments[2][(k0,k1)])**2
                output += value
                
    return output

def cov_ko_match(x) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 < k1 :
                value = (np.cov(x[k0*n:(k0+1)*n], x[k1*n:(k1+1)*n])[0,1] \
                                                 - moments[2][(k0,k1)])**2
                output += value
                
    return output

def cosk_match(x, how) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 != k1 :
                value = (coskew(x[k0*n:(k0+1)*n], test[k1,:], how) \
                                                 - moments[3][(k0,k1,how[0])])**2
                output += value
                
    return output

def cosk_ko_match(x, how) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 < k1 :
                value = (coskew(x[k0*n:(k0+1)*n], x[k1*n:(k1+1)*n], how) \
                                                 - moments[3][(k0,k1,how[0])])**2
                output += value
                
    return output

def coku_match(x, how) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 != k1 :
                value = (cokurt(x[k0*n:(k0+1)*n], test[k1,:], how) \
                                                 - moments[4][(k0,k1,how[0])])**2
                output += value
                
    return output

def coku_ko_match(x, how) :
    output = 0
    for k0 in range(p) :
        for k1 in range(p) :
            if k0 < k1 :
                value = (cokurt(x[k0*n:(k0+1)*n], x[k1*n:(k1+1)*n], how) \
                                                 - moments[4][(k0,k1,how[0])])**2
                output += value
                
    return output

xmean = NonlinearConstraint(is_mean, 0, 0.0004)
xvar = NonlinearConstraint(is_var, 0, 0.000011)
uniform = NonlinearConstraint(is_uniform, 0.1, 1)
cov0 = NonlinearConstraint(cov_match, 0, 0)
cov0_ko = NonlinearConstraint(cov_ko_match, 0, 0)

cosk0 = NonlinearConstraint(lambda x: cosk_match(x, 'left'), 0, 1e-5)
cosk0_ko = NonlinearConstraint(lambda x: cosk_ko_match(x, 'left'), 0, 1e-5)
cosk1 = NonlinearConstraint(lambda x: cosk_match(x, 'right'), 0, 1e-5)
cosk1_ko = NonlinearConstraint(lambda x: cosk_ko_match(x, 'right'), 0, 1e-5)

coku0 = NonlinearConstraint(lambda x: coku_match(x, 'left'), 0, 1e-5)
coku0_ko = NonlinearConstraint(lambda x: coku_ko_match(x, 'left'), 0, 1e-5)
coku1 = NonlinearConstraint(lambda x: coku_match(x, 'right'), 0, 1e-5)
coku1_ko = NonlinearConstraint(lambda x: coku_ko_match(x, 'right'), 0, 1e-5)
coku2 = NonlinearConstraint(lambda x: coku_match(x, 'center'), 0, 1e-5)
coku2_ko = NonlinearConstraint(lambda x: coku_ko_match(x, 'center'), 0, 1e-5)
                
constraints = (xmean, xvar, uniform, cov0)#, cosk0, cosk1)

def squared_cov(x) :
    result = 0
    for k in range(p) :
        value = np.cov(x[k*n:(k+1)*n], test[k,:])[0,1]
        value = value ** 2
        result += value
        
    return result

import datetime as dt
    
print('Start:')
print(dt.datetime.now())
res = minimize(squared_cov, rng.uniform(0,1,n*p),
               bounds=[(0,1) for _ in range(n*p)],
               constraints=constraints)

print('End:')
print(dt.datetime.now())




"""
Feature only cov: 2H46
Feature only cov + coskew: 



for k in range(p) :
    print(k)
    print(f'Mean: {round(res.x[k*n:(k+1)*n].mean(),4)}')
    print(f'Var: {round(res.x[k*n:(k+1)*n].var(),4)}')
    print(f'KS pvalue: {round(kstest(res.x[k*n:(k+1)*n],"uniform").pvalue,4)}')
    
    print('Correlation: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        if k1 != k :
            base = round(np.corrcoef(test[k,:],test[k1,:])[0,1],3)
            feat = round(np.corrcoef(res.x[k*n:(k+1)*n],test[k1,:])[0,1],3)
            knock = round(np.corrcoef(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n])[0,1],3)
            print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
            
    print('Left Coskewness: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        if k1 != k :
            base = round(coskew(test[k,:],test[k1,:],"left"),3)
            feat = round(coskew(res.x[k*n:(k+1)*n],test[k1,:],"left"),3)
            knock = round(coskew(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n],"left"),3)
            print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
            
    print('Right Coskewness: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        if k1 != k :
            base = round(coskew(test[k,:],test[k1,:],"right"),3)
            feat = round(coskew(res.x[k*n:(k+1)*n],test[k1,:],"right"),3)
            knock = round(coskew(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n],"right"),3)
            print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
            
    print(' ')

"""





