# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:29:59 2023

@author: Finch
"""

import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize, NonlinearConstraint

import knockoff as ko

test = pd.read_csv('smi_uniform.csv', index_col=0)
test = test.values
p, n = test.shape

moments = {}
moments[2] = {}
moments[3] = {}
moments[4] = {}
for k0 in range(p) :
    for k1 in range(p) :
        if k0 < k1 :
            moments[2][(k0,k1)] = np.cov(test[k0,:], test[k1,:])[0,1]
            moments[2][(k1,k0)] = moments[2][(k0,k1)]
            
            moments[3][(k0,k1,'l')] = ko.coskew(test[k0,:], test[k1,:], 'left')
            moments[3][(k0,k1,'r')] = ko.coskew(test[k0,:], test[k1,:], 'right')
            moments[3][(k1,k0,'l')] = moments[3][(k0,k1,'r')]
            moments[3][(k1,k0,'r')] = moments[3][(k0,k1,'l')]
            
            moments[4][(k0,k1,'l')] = ko.cokurt(test[k0,:], test[k1,:], 'left')
            moments[4][(k0,k1,'r')] = ko.cokurt(test[k0,:], test[k1,:], 'right')
            moments[4][(k0,k1,'c')] = ko.cokurt(test[k0,:], test[k1,:], 'center')
            
            moments[4][(k1,k0,'l')] = moments[4][(k0,k1,'r')]
            moments[4][(k1,k0,'r')] = moments[4][(k0,k1,'l')]
            moments[4][(k1,k0,'c')] = moments[4][(k0,k1,'c')]

xmean = NonlinearConstraint(lambda x: ko.is_mean(x,p,n), 0, 0.0004)
xvar = NonlinearConstraint(lambda x: ko.is_var(x,p,n), 0, 0.000011)
uniform = NonlinearConstraint(lambda x: ko.is_uniform(x,p,n), 0.1, 1)
cov0 = NonlinearConstraint(lambda x: ko.cov_match(x,p,n,test,moments[2]), 0, 0)
cov0_ko = NonlinearConstraint(lambda x: ko.cov_ko_match(x,p,n,moments[2]), 0, 0)

cosk0 = NonlinearConstraint(lambda x: ko.cosk_match(x,p,n,test,moments[3],'left'), 0, 0)
cosk0_ko = NonlinearConstraint(lambda x: ko.cosk_ko_match(x,p,n,moments[3],'left'), 0, 0)
cosk1 = NonlinearConstraint(lambda x: ko.cosk_match(x,p,n,test,moments[3],'right'), 0, 0)
cosk1_ko = NonlinearConstraint(lambda x: ko.cosk_ko_match(x,p,n,moments[3],'right'), 0, 0)

coku0 = NonlinearConstraint(lambda x: ko.coku_match(x,p,n,test,moments[4],'left'), 0, 0)
coku0_ko = NonlinearConstraint(lambda x: ko.coku_ko_match(x,p,n,moments[4],'left'), 0, 0)
coku1 = NonlinearConstraint(lambda x: ko.coku_match(x,p,n,test,moments[4],'right'), 0, 0)
coku1_ko = NonlinearConstraint(lambda x: ko.coku_ko_match(x,p,n,moments[4],'right'), 0, 0)
coku2 = NonlinearConstraint(lambda x: ko.coku_match(x,p,n,test,moments[4],'center'), 0, 0)
coku2_ko = NonlinearConstraint(lambda x: ko.coku_ko_match(x,p,n,moments[4],'center'), 0, 0)
                
constraints = (xmean, xvar, uniform, # All Uniform (0,1) variables
               cov0, # Covariances between knockoffs and features match
               cov0_ko, # Covariances between knockoffs match
               cosk0, cosk1, # Coskewness between knockoffs and features match
               cosk0_ko, cosk1_ko, # Coskewness between knockoffs match
               coku0, coku1, coku2, # Cokutosis between knockoffs and features match
               coku0_ko, coku1_ko, coku2_ko) # Cokurtosis between knockoffs match

def squared_cov(x) :
    result = 0
    for k in range(p) :
        value = np.cov(x[k*n:(k+1)*n], test[k,:])[0,1]
        value = value ** 2
        result += value
        
    return result

rng = np.random.default_rng(24)
x0 = rng.uniform(0,1,(p,n))
x0 = x0.reshape(-1,)
    
start = dt.datetime.now()
print('Start:')
print(start)
res = minimize(squared_cov, x0,
               bounds=[(0,1) for _ in range(n*p)],
               constraints=constraints,
               tol=1e-3)

end = dt.datetime.now()
print('End:')
print(end)


