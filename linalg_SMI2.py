# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:29:59 2023

@author: Christopher Hemmens
"""

import numpy as np
import pandas as pd
import datetime as dt
import pickle as pkl
from scipy.optimize import minimize, NonlinearConstraint

import knockoff_lib as ko

x0_type = 'candes_knockoff'
# Use "features" to use the features as the initial guess.
# Use "random" to use a random set of uniform variables.
# Use "constant" to use an array of all 0.5.
# Use any other string use the Candes-derived knockoff.

test = pd.read_csv('smi_uniform.csv', index_col=0)
test = test.values
p, n = test.shape
ind = '00_cov'

mean_limit = 3*np.sqrt(((test.mean(axis=1)-1/2)**2).mean())
var_limit = 3*np.sqrt(((test.var(axis=1)-1/12)**2).mean())

xmean = NonlinearConstraint(lambda x: ko.is_mean(x,p,n), 0, mean_limit**2)
xvar = NonlinearConstraint(lambda x: ko.is_var(x,p,n), 0, var_limit**2)
uniform = NonlinearConstraint(lambda x: ko.is_uniform(x,p,n), 0.1, 1)

cov_con = NonlinearConstraint(lambda x: ko.cov_match(x,p,n,test), 0, 0)
cosk_con = NonlinearConstraint(lambda x: ko.cosk_match(x,p,n,test), 0, 0)
coku_con = NonlinearConstraint(lambda x: ko.coku_match(x,p,n,test), 0, 0)
                
constraints = (xmean, xvar, uniform,
               cov_con, cosk_con, coku_con)

# Minimize sum of squared covariance between features and knockoffs
def squared_corr(x) :
    xc = x.copy().reshape(p,n)
    corr = np.corrcoef(xc,test)
    corr = corr[:p,p:].copy()
    corr = np.diag(corr)
    corr = corr**2
        
    return sum(corr)

# Generate Initial Guess
if x0_type == 'features' :
    x0 = test.reshape(-1,)
elif x0_type == 'random' :
    rng = np.random.default_rng(35)
    x0 = rng.uniform(0,1,(p,n))
    x0 = x0.reshape(-1,)
elif x0_type == 'constant' :
    x0 = np.array([0.5 for _ in range(n*p)])
else :
    x0 = ko.get_candes(test)
    x0 = x0.reshape(-1,)
    
    
# Optimization
start = dt.datetime.now()
print('Start:')
print(start)
res = minimize(squared_corr, x0,
               bounds=[(0,1) for _ in range(n*p)],
               constraints=constraints,
               tol=1e-3, options={'maxiter': 5})

end = dt.datetime.now()
print('End:')
print(end)

with open(f'smi_knockoffs_{ind}.pkl', 'wb') as fp :
    pkl.dump(res, fp)


