#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:44:36 2023

@author: finch
"""

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf

pd.set_option('display.max_columns', 500)

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import matplotlib.pyplot as plt

from copulas.multivariate import GaussianMultivariate
from copulas.univariate import ParametricType, Univariate

import warnings
warnings.filterwarnings("ignore")

refresh = False
read_df = True
restart_train = True

tickers = pd.read_excel('SMI.xlsx', index_col='Tickers',
                      parse_dates=['Start', 'End'])

if refresh :
    smi = yf.download(['^SSMI']+tickers.index.tolist(),
                      start='2017-05-26')['Adj Close']
    smi.to_csv('SMI.csv')
else :
    smi = pd.read_csv('SMI.csv', index_col=0, parse_dates=True)
    
cs = pd.read_csv('CS.csv', skiprows=14, index_col=0, parse_dates=True).close
cs = cs.loc[dt.date(2017,5,26):].copy()
cs.name = 'CSGN.SW'

smi = pd.concat([smi, cs], axis=1)
smi_ind = smi['^SSMI'].copy()
smi_ind.name = 'SMI'
smi.drop('^SSMI', axis=1, inplace=True)

smi_cols = smi.columns.tolist()
smi_cols.sort()
smi = smi[smi_cols]

smi_cols = [x[:-3] for x in smi.columns]
smi.columns = smi_cols
smi_bkup = smi.copy()

smi.fillna(method='ffill', inplace=True)
smi_cols = smi.columns.tolist()
for col in smi_cols :
    smi[col+'_shift'] = smi[col].shift(1)
    smi[col] = smi[col]/smi[col+'_shift'] - 1
    smi.drop(col+'_shift', axis=1, inplace=True)
    
smi.drop(smi.index[0], inplace=True)
smi.fillna(0, inplace=True)

tickers.loc['CSGN.SW', 'End'] = np.datetime64('2023-06-12')
tickers.index = [x[:-3] for x in tickers.index]

cons = {}
cons_temp = tickers[tickers.Start.isna()].index.tolist()

kk = 0
cons[kk] = {}
cons[kk]['timestamp'] = smi.index[0]
cons[kk]['cons'] = cons_temp.copy()

change_dates = tickers.Start.dropna().sort_values()
for ind in range(change_dates.shape[0]) :
    kk += 1
    to_remove = tickers[tickers.End == change_dates.iloc[ind]].index[0]
    cons_temp.remove(to_remove)
    cons_temp += [change_dates.index[ind]]
    
    cons[kk] = {}
    cons[kk]['timestamp'] = change_dates.iloc[ind]
    cons[kk]['removed'] = to_remove
    cons[kk]['added'] = change_dates.index[ind]
    cons[kk]['cons'] = cons_temp.copy()

test = smi.loc[:cons[1]['timestamp'], cons[0]['cons']].copy()

test_scaled = StandardScaler().fit_transform(test)
test_scaled = pd.DataFrame(test_scaled, index=test.index, columns=test.columns)

if read_df :
    with open('SMI_AE_df_daily_scaled.pkl', 'rb') as fp :
        df = pkl.load(fp)
else :
    hidden_layers = [(20,), (3,), (2,),
                     (10,3,10), (10,2,10), (5,2,5)]
    activations = ['identity', 'tanh']
    solvers = ['lbfgs', 'adam']
    alphas = [0.0001, 0.001, 0.01]
    learning_rates = ['constant', 'invscaling', 'adaptive']
    learning_rate_inits = [0.001, 0.01, 0.1]
    max_iters = [200, 400, 800, 1500, 2500]
    
    if restart_train :
        test_results = {'hidden_layers': [],
                        'activation': [],
                        'solver': [],
                        'alpha': [],
                        'learning_rate': [],
                        'initial_rate': [],
                        'max_iter': [],
                        'score': []}
    else :
        with open('SMI_AE_results.pkl', 'rb') as fp :
            test_results = pkl.load(fp)
            
        already_done = set(test_results['hidden_layers'])
        hidden_layers = [x for x in hidden_layers if x not in already_done]
        
    for hl in hidden_layers :
        with open('SMI_AE_results.pkl', 'wb') as fp :
            pkl.dump(test_results, fp)
            
        df = pd.DataFrame.from_dict(test_results)
        with open('SMI_AE_df_daily_scaled.pkl', 'wb') as fp :
            pkl.dump(df, fp)
            
        for act in activations :
            for solve in solvers :
                print(f'{hl}\t{act}\t{solve}')
                for alpha in alphas :
                    for lr in learning_rates :
                        for lri in learning_rate_inits :
                            for mi  in max_iters :
                                test_results['hidden_layers'] += [hl]
                                test_results['activation'] += [act]
                                test_results['solver'] += [solve]
                                test_results['alpha'] += [alpha]
                                test_results['learning_rate'] += [lr]
                                test_results['initial_rate'] += [lri]
                                test_results['max_iter'] += [mi]
                                
                                best_score = 0
                                for ii in range(5) :
                                    try :
                                        reg = MLPRegressor(hidden_layer_sizes=hl,
                                                           activation=act,
                                                           solver=solve,
                                                           alpha=alpha,
                                                           learning_rate=lr,
                                                           learning_rate_init=lri,
                                                           max_iter=mi,
                                                           tol=1e-8).fit(test, test)
                                        
                                        best_score = max(best_score, reg.score(test, test))
                                    except :
                                        continue
                                    
                                test_results['score'] += [best_score]
                                
    df = pd.DataFrame.from_dict(test_results)
    with open('SMI_AE_df_daily_scaled.pkl', 'wb') as fp :
        pkl.dump(df, fp)
     
hidden = 2
        
if hidden == 3 :
    while True :
        reg = MLPRegressor(hidden_layer_sizes=(3,),
                           activation='identity',
                           solver='lbfgs',
                           alpha=0.0001,
                           learning_rate='constant',
                           learning_rate_init=0.001,
                           max_iter=800, tol=1e-8).fit(test, test)
        
        if reg.score(test, test) > 0.60 :
            break
elif hidden == 2 :
    while True :
        reg = MLPRegressor(hidden_layer_sizes=(2,),
                           activation='identity',
                           solver='lbfgs',
                           alpha=0.0001,
                           learning_rate='constant',
                           learning_rate_init=0.001,
                           max_iter=800, tol=1e-8).fit(test, test)
        
        if reg.score(test, test) > 0.54 :
            break

output = pd.DataFrame(reg.predict(test), index=test.index, columns=test.columns)

scores = {'stock': [], 'score': []}
for stock in test.columns :
    scores['stock'] += [stock]
    scores['score'] += [np.linalg.norm(test[stock] - output[stock])]

scores = pd.DataFrame.from_dict(scores).set_index('stock').squeeze()

smii = smi_ind.copy()
smii.fillna(method='ffill', inplace=True)
smii /= smii.shift(1)
smii.dropna(inplace=True)
smii -= 1
smii = smii.loc[test.index].copy()

from scipy.optimize import minimize, NonlinearConstraint

scores = scores.sort_values()

base = 1
n = 1
l = 0.02
test2 = test[scores.index.tolist()[:base] + scores.index.tolist()[-n:]].copy()

def find_weights(x) :
    return np.linalg.norm(smii - (test2*x).sum(axis=1))**2 + l*np.linalg.norm(x)**2

w0 = [1/(n+base) for x in range(n+base-1)]
w0 += [1-sum(w0)]
res = minimize(find_weights, w0,
               bounds=[(0, 1) for x in range(n+base)],
               constraints=[{'type': 'eq', 'fun': lambda x:  sum(x)-1}])

tracker = (pd.concat([smii, (test2*res.x).sum(axis=1)], axis=1)+1).cumprod()

univariate = Univariate(parametric=ParametricType.PARAMETRIC)
copula = GaussianMultivariate(distribution=univariate)
copula.fit(test)
params = copula.to_dict()

from scipy.stats import norm, truncnorm, beta, gaussian_kde, loglaplace, t, \
                        kstest, multivariate_normal

"""
def unif_transform(y) :
    min_p = []
    for stock in ['NOVN', 'GIVN', 'SGSN', 'UHR', 'CSGN'] :
        test3 = test[[stock]].copy()
        pdf = gaussian_kde(test3[stock].tolist())
        test3['pdf'] = pdf.pdf(test3[stock])
        test3.sort_values(stock, inplace=True)
        test3['cdf'] = test3.pdf.cumsum()
        m = (y[1] - y[0]) / (test3.cdf.iloc[-1] - test3.cdf.iloc[0])
        c = y[1] - m*test3.cdf.iloc[-1]
        test3.cdf *= m
        test3.cdf += c
        test3.sort_index(inplace=True)
        p = 1 - kstest(test3.cdf, 'uniform').pvalue
        min_p += [p]
    return np.prod(min_p)

optim_y = minimize(unif_transform, [0.001, 0.98])
"""

cdf = {}
for kk in range(20) :
    stock = test.columns[kk]
    if params['univariates'][kk]['type'] == 'copulas.univariate.gaussian.GaussianUnivariate' :
        values = norm.cdf(test[stock],
                          loc=params['univariates'][kk]['loc'],
                          scale = params['univariates'][kk]['scale']).tolist()
    elif params['univariates'][kk]['type'] == 'copulas.univariate.student_t.StudentTUnivariate' :
        values = t.cdf(test[stock],
                       df=params['univariates'][kk]['df'],
                       loc=params['univariates'][kk]['loc'],
                       scale = params['univariates'][kk]['scale']).tolist()
        """
    elif params['univariates'][kk]['type'] == 'copulas.univariate.gaussian_kde.GaussianKDE' :
        pdf = gaussian_kde(params['univariates'][kk]['dataset'])
        test3 = test[[stock]].copy()
        test3['pdf'] = pdf.pdf(test3[stock])
        test3.sort_values(stock, inplace=True)
        test3['cdf'] = test3.pdf.cumsum()
        
        y1 = optim_y.x[1]
        y0 = optim_y.x[0]
        
        m = (y1 - y0) / (test3.cdf.iloc[-1] - test3.cdf.iloc[0])
        c = y1 - m*test3.cdf.iloc[-1]
        
        test3.cdf *= m
        test3.cdf += c
        test3.sort_index(inplace=True)
        values = test3.cdf.tolist()
        """
    elif params['univariates'][kk]['type'] == 'copulas.univariate.log_laplace.LogLaplace' :
        values = loglaplace.cdf(test[stock],
                                c=params['univariates'][kk]['c'],
                                loc=params['univariates'][kk]['loc'],
                                scale = params['univariates'][kk]['scale']).tolist()
    elif params['univariates'][kk]['type'] == 'copulas.univariate.beta.BetaUnivariate' :
        values = beta.cdf(test[stock],
                          a=params['univariates'][kk]['a'],
                          b=params['univariates'][kk]['b'],
                          loc=params['univariates'][kk]['loc'],
                          scale = params['univariates'][kk]['scale']).tolist()
        """
    elif params['univariates'][kk]['type'] == 'copulas.univariate.truncated_gaussian.TruncatedGaussian' :
        values = truncnorm.cdf(test[stock],
                               a=params['univariates'][kk]['a'],
                               b=params['univariates'][kk]['b'],
                               loc=params['univariates'][kk]['loc'],
                               scale = params['univariates'][kk]['scale']).tolist()
        """
    
        
    cdf[stock] = values

norm_dist = []

for stock in cdf.keys() :
    test3 = pd.Series(norm.ppf(cdf[stock]), index=test.index, name=stock)
    norm_dist += [test3]
    
norm_dist = pd.concat(norm_dist, axis=1)
    
means = norm_dist.mean()
vs = norm_dist.var()
norm_dist_demean = norm_dist - means

covs = norm_dist.cov()
covs_inv = np.linalg.inv(covs)

def get_cov_min_eigval(x) :
    x_diag = np.diag(x)
    covs_t = np.dot(x_diag, np.dot(covs_inv, x_diag))
    return min(np.linalg.eigvals(2*x_diag - covs_t))

def get_diag_weights(x) :
    return sum([y**2 for y in vs-x])

rng = np.random.default_rng(24)

while True :
    thresh = rng.uniform(1e-9,1e-2)
    min_eigval = NonlinearConstraint(get_cov_min_eigval, thresh, np.inf)
    
    res = minimize(get_diag_weights, np.zeros(20),
                   constraints=min_eigval)
    
    if get_cov_min_eigval(res.x) >= 0 :
        break
    
x_diag = np.diag(res.x)

samples = []
for kk in range(test.shape[0]) :
    samples += [multivariate_normal.rvs(norm_dist.iloc[kk] - np.dot(norm_dist.iloc[kk], np.dot(covs_inv, x_diag)),
                                        2*x_diag - np.dot(x_diag, np.dot(covs_inv, x_diag)))]
        
    
knockoffs = pd.DataFrame(samples,
                         index=test.index,
                         columns=[x+'_KO' for x in test.columns])

test_ko = pd.concat([test, knockoffs], axis=1)
ko_corr = test_ko.corr()

for stock in test.columns :
    print(f'{stock}\t\t{round(ko_corr.loc[stock,stock+"_KO"],2)}')
    
print(ko_corr[[stock, stock+'_KO']])


"""
#multivariate_normal.rvs(means + np.dot(np.dot(covs_ko, covs_inv), norm_dist_demean.iloc[kk]), ko_covs)

def get_cov_min_eigval_old(x) :
    covs_ko = covs - np.diag(x)
    covs_ko_t = np.dot(np.dot(covs_ko, covs_inv), covs_ko)
    return min(np.linalg.eigvals(covs - covs_ko_t))
"""   
 
"""
vol = yf.download(tickers.index.tolist(), start='2017-05-26')['Volume']
volsum = vol.loc[test.index[0]:test.index[-1]].copy().sum()

csvol = pd.read_csv('CS.csv', skiprows=14, index_col=0, parse_dates=True).volume
csvol = csvol.loc[test.index[0]:test.index[-1]].sum()

scores = pd.DataFrame(scores)
for ind in scores.index :
    if ind != 'CSGN' :
        scores.loc[ind, 'volume'] = volsum.loc[ind+'.SW'].copy()
        
scores.loc['CSGN', 'volume'] = csvol

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(scores)
scored = scaler.transform(scores)
linreg = LinearRegression().fit(scored[:,0].reshape(-1,1), scored[:,1])
linreg.score(scored[:,0].reshape(-1,1), scored[:,1])
"""
    





