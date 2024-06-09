# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:29:59 2023

@author: Christopher Hemmens
"""

import numpy as np
import pandas as pd
import datetime as dt
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint

import knockoff_lib2 as ko

max_iter = [5]#, 200]
ind = '24_00'
x0_type = 'candes_knockoff'
# Use "features" to use the features as the initial guess.
# Use "random" to use a random set of uniform variables.
# Use "constant" to use an array of all 0.5.
# Use any other string use the Candes-derived knockoff.

print('Reading data...')
smi = pd.read_csv('SMI.csv', index_col=0, parse_dates=True)
    
cs = pd.read_csv('CS.csv', skiprows=14, index_col=0, parse_dates=True).close
cs = cs.loc[dt.date(2017,5,26):].copy()
cs.name = 'CSGN.SW'

smi = pd.concat([smi, cs], axis=1)

smi.ffill(inplace=True)
smi_cols = smi.columns.tolist()
for col in smi_cols :
    smi[col+'_shift'] = smi[col].shift(1)
    smi[col] = smi[col]/smi[col+'_shift'] - 1
    smi.drop(col+'_shift', axis=1, inplace=True)
    
smi.drop(smi.index[0], inplace=True)
smi.fillna(0, inplace=True)

tickers = pd.read_excel('SMI.xlsx', index_col='Tickers',
                      parse_dates=['Start', 'End'])
tickers.loc['CSGN.SW', 'End'] = np.datetime64('2023-06-12')

cons = tickers[tickers.Start.isna()].index.tolist()
end = tickers.End.min()

smi = smi.loc[:end, cons].copy()

smi_cols = [x[:-3] for x in smi.columns.tolist()]
smi.columns = smi_cols
smi_cols.sort()
smi = smi[smi_cols].copy()

#smi = smi.iloc[:200].copy()

print('Scaling data...')
scaler = StandardScaler()
test = scaler.fit_transform(smi)
smi_sc = pd.DataFrame(test, index=smi.index, columns=smi.columns)

print('Fitting copula...')
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import ParametricType, Univariate

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
                
constraints = (xdist, xmean, xvar, 
               corr_con, cosk_con, coku_con)

# Minimize sum of squared covariance between features and knockoffs
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
if x0_type == 'features' :
    x0 = test.reshape(-1,)
elif x0_type == 'random' :
    rng = np.random.default_rng(35)
    x0 = rng.uniform(0,1,(p,n))
    x0 = x0.reshape(-1,)
elif x0_type == 'constant' :
    x0 = np.array([0.5 for _ in range(n*p)])
else :
    x0 = ko.get_candes(test, params['univariates'])
    x0 = x0.reshape(-1,)
    
for mi in max_iter :
    output = {}
    output['x0'] = x0.copy()
    output['params'] = params
    
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
    
    output['start'] = start
    output['end'] = end
    output['time'] = (end-start).seconds
    
    ko_cnd = x0.reshape(n,p)
    ko_cnd = pd.DataFrame(ko_cnd, index=smi.index, columns=smi.columns)
    
    x1 = res.x
    ko_hms = x1.reshape(n,p)
    ko_hms = pd.DataFrame(ko_hms, index=smi.index, columns=smi.columns)
    
    output['x1'] = x1.copy()
    output['Candes'] = ko_cnd.copy()
    output['Hemmens'] = ko_hms.copy()
    
    with open('SMI_KO_{:03d}_{}.pkl'.format(mi, ind), 'wb') as fp :
        pkl.dump(output, fp)







