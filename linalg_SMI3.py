# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:29:59 2023

@author: Christopher Hemmens
"""

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, NonlinearConstraint

import knockoff_lib as ko

max_iter = 200
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

smi.fillna(method='ffill', inplace=True)
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

smi = smi.iloc[:200].copy()

print('Scaling data...')
scaler = StandardScaler()
smi_sc = scaler.fit_transform(smi)
smi_sc = pd.DataFrame(smi_sc, index=smi.index, columns=smi.columns)

print('Fitting copula...')
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import ParametricType, Univariate

import warnings
warnings.filterwarnings("ignore")

univariate = Univariate(parametric=ParametricType.PARAMETRIC)
copula = GaussianMultivariate(distribution=univariate)
copula.fit(smi_sc)
params = copula.to_dict()

from scipy.stats import t

smi_uf = smi_sc.copy()
for kk in range(20) :
    col = smi_sc.columns[kk]
    if params['univariates'][kk]['type'].split('.')[-1] == 'StudentTUnivariate' :
        values = smi_sc[col].copy()
        smi_uf[col] = t.cdf(values,
                            df=params['univariates'][kk]['df'],
                            loc=params['univariates'][kk]['loc'],
                            scale = params['univariates'][kk]['scale'])
    else :
        raise Exception(f'Column {col} is not Student T')
    
test = smi_uf.copy().transpose()
test = test.values
p, n = test.shape

mean_limit = 3*np.sqrt(((smi_uf.mean()-1/2)**2).mean())
var_limit = 3*np.sqrt(((smi_uf.var()-1/12)**2).mean())

xmean = NonlinearConstraint(lambda x: ko.is_mean(x,p,n), 0, mean_limit**2)
xvar = NonlinearConstraint(lambda x: ko.is_var(x,p,n), 0, var_limit**2)
uniform = NonlinearConstraint(lambda x: ko.is_uniform(x,p,n), 0.1, 1)

cov_con = NonlinearConstraint(lambda x: ko.cov_match(x,test), 0, 0)
cosk_con = NonlinearConstraint(lambda x: ko.cosk_match(x,test), 0, 0)
coku_con = NonlinearConstraint(lambda x: ko.coku_match(x,test), 0, 0)
                
constraints = (xmean, xvar, uniform,
               cov_con, cosk_con, coku_con)

# Minimize sum of squared covariance between features and knockoffs
def squared_corr(x) :
    xc = x.copy().reshape(p,n)
    #for kk in range(p) :
    #    xc[kk,:] = t.ppf(xc[kk,:], df=params['univariates'][kk]['df'], loc=params['univariates'][kk]['loc'], scale = params['univariates'][kk]['scale'])
        
    corr = np.corrcoef(xc,test)
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
    x0 = ko.get_candes(test)
    x0 = x0.reshape(-1,)
    
lb = x0.min() / 2
ub = 1 - (1 - x0.max()) / 2
    
print('Running optimisation...')
# Optimization
start = dt.datetime.now()
print('Start:')
print(start)
res = minimize(squared_corr, x0,
               bounds=[(lb,ub) for _ in range(n*p)],
               constraints=constraints,
               tol=1e-3,
               options={'maxiter': max_iter,
                        'disp': True})

end = dt.datetime.now()
print('End:')
print(end)

ko_cnd = x0.reshape(p,n).transpose()
ko_cnd = pd.DataFrame(ko_cnd, index=smi_uf.index, columns=smi_uf.columns)

x1 = res.x
ko_hms = x1.reshape(p,n).transpose()
ko_hms = pd.DataFrame(ko_hms, index=smi_uf.index, columns=smi_uf.columns)

smi_cnd = smi_sc.copy()
smi_hms = smi_sc.copy()

for kk in range(20) :
    col = smi_sc.columns[kk]
    values = ko_cnd[col].copy()
    smi_cnd[col] = t.ppf(values,
                         df=params['univariates'][kk]['df'],
                         loc=params['univariates'][kk]['loc'],
                         scale = params['univariates'][kk]['scale'])
    
    values = ko_hms[col].copy()
    smi_hms[col] = t.ppf(values,
                         df=params['univariates'][kk]['df'],
                         loc=params['univariates'][kk]['loc'],
                         scale = params['univariates'][kk]['scale'])
    
ko_cols = [x+'_KO' for x in smi_cols]
smi_cnd.columns = ko_cols
smi_hms.columns = ko_cols

test = pd.concat([smi_sc, smi_cnd], axis=1)
test.corr().to_excel('KO_Candes_Correlation_{:03d}_uf.xlsx'.format(max_iter))

test_cols = test.columns
test = test.transpose()
output = pd.DataFrame(ko.coskew(test), index=test_cols, columns=test_cols)
output.to_excel('KO_Candes_Coskewness_{:03d}_uf.xlsx'.format(max_iter))

output = pd.DataFrame(ko.cokurt(test), index=test_cols, columns=test_cols)
output.to_excel('KO_Candes_Cokurtosis_{:03d}_uf.xlsx'.format(max_iter))

test = pd.concat([smi_sc, smi_hms], axis=1)
test.corr().to_excel('KO_Hemmens_Correlation_{:03d}_uf.xlsx'.format(max_iter))

test_cols = test.columns
test = test.transpose()
output = pd.DataFrame(ko.coskew(test), index=test_cols, columns=test_cols)
output.to_excel('KO_Hemmens_Coskewness_{:03d}_uf.xlsx'.format(max_iter))

output = pd.DataFrame(ko.cokurt(test), index=test_cols, columns=test_cols)
output.to_excel('KO_Hemmens_Cokurtosis_{:03d}_uf.xlsx'.format(max_iter))

sc = smi_sc.values.transpose()
cnd = smi_cnd.values.transpose()
hms = smi_hms.values.transpose()

moments = {}
moments['moment'] = ['Correlation', 'Coskewness', 'Cokurtosis']
moments['Candes'] = [ko.cov_match(cnd,sc),
                   ko.cosk_match(cnd,sc),
                   ko.coku_match(cnd,sc)]
moments['Hemmens'] = [ko.cov_match(hms,sc),
                    ko.cosk_match(hms,sc),
                    ko.coku_match(hms,sc)]

moments = pd.DataFrame.from_dict(moments)
moments.set_index('moment', inplace=True, drop=True)
moments.to_excel('KO_Moment_Match_{:03d}_uf.xlsx'.format(max_iter))








