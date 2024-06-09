# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:49:31 2024

@author: Christopher Hemmens
"""

import math
import numpy as np
import time
import datetime as dt
import pickle as pkl
from scipy.stats import t

import t_core
import knockoff_lib2 as ko

# Running this file generates a pickle file in the format 'KOMC.pkl'

#simulation parameters
df_t = 5 # degree of freedom of t-distribution
p = 30 # dimension of the random vector X
numsamples = 200 # number of samples to generate knockoffs 
rhos = [0.3] * (p-1) # the correlations

#algorithm parameters
halfnumtry = 1 # m/half number of candidates
stepsize = 1.5 # step size in the unit of 1/\sqrt((\Sigma)^{-1}_{jj})

#generate the proposal grid
quantile_x = np.zeros([p, 2*halfnumtry + 1])
sds = [0]*p
sds[0] = math.sqrt(1 - rhos[0]**2)
for i in range(1,p - 1):
    sds[i] = math.sqrt((1 - rhos[i - 1]**2)*(1 - rhos[i]**2) /
                       (1 - rhos[i - 1]**2*rhos[i]**2))
sds[p - 1] = math.sqrt(1 - rhos[p - 2]**2)
for i in range(p):
    quantile_x[i] = [x*sds[i]*stepsize for x in list(
        range(-halfnumtry, halfnumtry + 1))]
    
bigmatrix = np.zeros([numsamples, 2*p]) # store simulation data

#generate each observation and knockoff
start = time.time()
for i in range(numsamples):
    #sample one instance from the Markov Chain
    bigmatrix[i, 0] = t.rvs(df=df_t)*math.sqrt((df_t - 2)/df_t)
    for j in range(1, p):
        bigmatrix[i, j] = math.sqrt(1 - rhos[j - 1]**2)*t.rvs(df=df_t)* \
        math.sqrt((df_t - 2)/df_t) + rhos[j - 1]*bigmatrix[i,j - 1]
    
    #sample the knockoff for the observation
    bigmatrix[i, p:(2*p)] = t_core.SCEP_MH_MC(bigmatrix[i, 0:p], 0.999, 
                                       quantile_x, rhos, df_t)
end = time.time()
print(round(end - start,1))

test = bigmatrix[:, :p].copy()
ko_mc = bigmatrix[:, p:].copy()

print('Fitting copula...')
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import ParametricType, Univariate
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint

import warnings
warnings.filterwarnings("ignore")

univariate = Univariate(parametric=ParametricType.PARAMETRIC)
copula = GaussianMultivariate(distribution=univariate)
copula.fit(test)
params = copula.to_dict()

test_mean = test.mean(axis=0)
test_var = test.var(axis=0)
#test_pv = np.array([kstest(test[:,kk],'t',tuple(params['univariates'][kk].values())[:-1]).pvalue for kk in range(test.shape[1])])

n, p = test.shape

A = np.zeros((p, p*n))
for kk in range(p) :
    A[kk, n*kk:n*(kk+1)] = 1/n
    
xmean = LinearConstraint(A, test_mean, test_mean)
xvar = NonlinearConstraint(lambda x: ko.is_var(x, test_var), 0, 0)
xdist = NonlinearConstraint(lambda x: \
                    ko.is_distributed(x, params['univariates']), 0, 0.25)
    
corr_con = NonlinearConstraint(lambda x: ko.corr_match(x, test), 0, 0)
cosk_con = NonlinearConstraint(lambda x: ko.cosk_match(x, test), 0, 0)
coku_con = NonlinearConstraint(lambda x: ko.coku_match(x, test), 0, 0)
                
constraints = (xmean,
               xvar,
               xdist,
               corr_con,
               cosk_con,
               coku_con
               )

# Minimize sum of squared correlation between features and knockoffs
def squared_corr(x) :
    xc = x.copy()
    xc = xc.reshape(n,p)
    corr = np.corrcoef(xc.T, test.T)
    corr = corr[:p,p:].copy()
    corr = np.diag(corr)
    corr = corr**2
        
    return sum(corr)

print('Getting Candes knockoff...')
# Generate Initial Guess
x0 = ko.get_candes(test, params['univariates'])
x0 = x0.reshape(-1,)

output = {}
output['x0'] = x0.copy()
output['params'] = params

# Optimization
max_iter = [1,2,3,5,10,20]

output = {}

output['features'] = test.copy()
output['ko_mc'] = ko_mc.copy()
output['time'] = end - start

for mi in max_iter :
    output[mi] = {}
    output[mi]['x0'] = x0.copy()
    output[mi]['params'] = params
    
    print(f'Running optimisation... {mi}')
    # Optimization
    start = dt.datetime.now()
    print('Start:')
    print(start)
    res = minimize(squared_corr, x0,
                   constraints=constraints,
                   tol=1e-3,
                   options={'maxiter': mi,
                            'disp': True})
    
    end = dt.datetime.now()
    print('End:')
    print(end)
    
    output[mi]['start'] = start
    output[mi]['end'] = end
    output[mi]['time'] = (end-start).seconds
    
    ko_cnd = x0.reshape(n,p)
    
    x1 = res.x
    ko_hms = x1.reshape(n,p)
    
    output[mi]['x1'] = x1.copy()
    output[mi]['Candes'] = ko_cnd.copy()
    output[mi]['Hemmens'] = ko_hms.copy()
    
    with open('KOMC.pkl', 'wb') as fp :
        pkl.dump(output, fp)
