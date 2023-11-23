# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:29:59 2023

@author: Christopher Hemmens
"""

import numpy as np
import pandas as pd
import datetime as dt
import math
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import kstest

import knockoff_lib as ko

ps = [2,4,8,16,32,64,128]
ns = [100,1000,10000,1000000,1000000]
ind = '04'
max_iter = [3, 10, 20]

x0_type = 'candes_knockoff'
# Use "features" to use the features as the initial guess.
# Use "random" to use a random set of uniform variables.
# Use "constant" to use an array of all 0.5.
# Use any other string use the Candes-derived knockoff.

for p in ps[1:2] :
    for n in ns[:2] :
        print(f"{p}\t{n}")
        rng = np.random.default_rng(24)
        test = rng.uniform(0,1,(p,n))
        
        # Artificially generated covariance, coskewness, and cokurtosis
        for k in range(p) :
            if k % 4 == 0 : 
                pt = test[k,:]/sum(test[k,:])
                test[k,:] = rng.choice(test[k,:], size=n, replace=False, p=pt)
            elif k % 4 == 1 :
                pt = [1 - x for x in test[k,:]]
                pt /= sum(pt)
                test[k,:] = rng.choice(test[k,:], size=n, replace=False, p=pt)
            elif k % 4 == 2 :
                pt = [abs(x-0.5) for x in test[k,:]]
                pt /= sum(pt)
                test[k,:] = rng.choice(test[k,:], size=n, replace=False, p=pt)
                
        # Constraints
            # Determine bounds for knockoff mean and variance using data
        mean_limit = 3*np.sqrt(((test.mean(axis=1)-1/2)**2).mean())
        var_limit = 3*np.sqrt(((test.var(axis=1)-1/12)**2).mean())
        
            # Uniform distribution constraints
        xmean = NonlinearConstraint(lambda x: ko.is_mean(x,p,n), 0, mean_limit**2)
        xvar = NonlinearConstraint(lambda x: ko.is_var(x,p,n), 0, var_limit**2)
        uniform = NonlinearConstraint(lambda x: ko.is_uniform(x,p,n), 0.1, 1)
        
            # Moment match constraints
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
            xc = test.copy()
        elif x0_type == 'random' :
            rng = np.random.default_rng(35)
            xc = rng.uniform(0,1,(p,n))
        elif x0_type == 'constant' :
            xc = np.array([0.5 for _ in range(n*p)])
        else :
            xc = ko.get_candes(test)
            

        for mi in max_iter :
            x0 = xc.copy().reshape(-1,)
            
            # Optimization
            start = dt.datetime.now()
            print("Start:")
            print(start)
            res = minimize(squared_corr, x0,
                           bounds=[(0,1) for _ in range(n*p)],
                           constraints=constraints,
                           tol=1e-3, options={'maxiter': mi})
            
            end = dt.datetime.now()
            print("End:")
            print(end)
            
            # Output Results to Excel
            x1 = res.x.reshape(p,n)
            res_mean = x1.mean(axis=1)
            res_var = x1.var(axis=1)
            res_unif = np.apply_along_axis(kstest,1,x1,cdf='uniform')[:,1]
            
            res_corr = np.corrcoef(x1,test)
            res_cosk = ko.coskew(np.vstack((x1,test)))
            res_coku = ko.cokurt(np.vstack((x1,test)), 'left')
            res_coku_c = ko.cokurt(np.vstack((x1,test)))
            
            x0 = x0.reshape(p,n)
            can_corr = np.corrcoef(x0,test)
            can_cosk = ko.coskew(np.vstack((x0,test)))
            can_coku = ko.cokurt(np.vstack((x0,test)), 'left')
            can_coku_c = ko.cokurt(np.vstack((x0,test)))
            
            results = {}
            results['Knockoff Variable'] = [None]
            results['Comparison Variable'] = [None]
            results['Measure'] = ['Time Taken']
            results['Real Value'] = [(end-start).seconds]
            results['Value with Feature'] = [None]
            results['Candes with Feature'] = [None]
            results['Value with Knockoff'] = [None]
            results['Candes with Knockoff'] = [None]
            for k0 in range(p) :
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [None]
                results['Measure'] += ['Mean'] 
                results['Real Value'] += [round(res_mean[k0],4)]
                results['Value with Feature'] += [None]
                results['Candes with Feature'] += [None]
                results['Value with Knockoff'] += [None]
                results['Candes with Knockoff'] += [None]
                
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [None]
                results['Measure'] += ['Variance'] 
                results['Real Value'] += [round(res_var[k0],4)]
                results['Value with Feature'] += [None]
                results['Candes with Feature'] += [None]
                results['Value with Knockoff'] += [None]
                results['Candes with Knockoff'] += [None]
                
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [None]
                results['Measure'] += ['Uniformity_pvalue'] 
                results['Real Value'] += [round(res_unif[k0],4)]
                results['Value with Feature'] += [None]
                results['Candes with Feature'] += [None]
                results['Value with Knockoff'] += [None]
                results['Candes with Knockoff'] += [None]
                for k1 in range(p) :                
                    results['Knockoff Variable'] += [k0]
                    results['Comparison Variable'] += [k1]
                    results['Measure'] += ['Correlation']
                    results['Real Value'] += [round(res_corr[k0+p,k1+p],3)]
                    results['Value with Feature'] += [round(res_corr[k0,k1+p],3)]
                    results['Candes with Feature'] += [round(can_corr[k0,k1+p],3)]
                    results['Value with Knockoff'] += [round(res_corr[k0,k1],3)]
                    results['Candes with Knockoff'] += [round(can_corr[k0,k1],3)]
                    
                    results['Knockoff Variable'] += [k0]
                    results['Comparison Variable'] += [k1]
                    results['Measure'] += ['Left Coskewness']
                    results['Real Value'] += [round(res_cosk[k0+p,k1+p],3)]
                    results['Value with Feature'] += [round(res_cosk[k0,k1+p],3)]
                    results['Candes with Feature'] += [round(can_cosk[k0,k1+p],3)]
                    results['Value with Knockoff'] += [round(res_cosk[k0,k1],3)]
                    results['Candes with Knockoff'] += [round(can_cosk[k0,k1],3)]
                    
                    results['Knockoff Variable'] += [k0]
                    results['Comparison Variable'] += [k1]
                    results['Measure'] += ['Right Coskewness']
                    results['Real Value'] += [round(res_cosk[k1+p,k0+p],3)]
                    results['Value with Feature'] += [round(res_cosk[k1+p,k0],3)]
                    results['Candes with Feature'] += [round(can_cosk[k1+p,k0],3)]
                    results['Value with Knockoff'] += [round(res_cosk[k1,k0],3)]
                    results['Candes with Knockoff'] += [round(can_cosk[k1,k0],3)]
            
                    results['Knockoff Variable'] += [k0]
                    results['Comparison Variable'] += [k1]
                    results['Measure'] += ['Left Cokurtosis']
                    results['Real Value'] += [round(res_coku[k0+p,k1+p],3)]
                    results['Value with Feature'] += [round(res_coku[k0,k1+p],3)]
                    results['Candes with Feature'] += [round(can_coku[k0,k1+p],3)]
                    results['Value with Knockoff'] += [round(res_coku[k0,k1],3)]
                    results['Candes with Knockoff'] += [round(can_coku[k0,k1],3)]
                    
                    results['Knockoff Variable'] += [k0]
                    results['Comparison Variable'] += [k1]
                    results['Measure'] += ['Central Cokurtosis']
                    results['Real Value'] += [round(res_coku_c[k0+p,k1+p],3)]
                    results['Value with Feature'] += [round(res_coku_c[k0,k1+p],3)]
                    results['Candes with Feature'] += [round(can_coku_c[k0,k1+p],3)]
                    results['Value with Knockoff'] += [round(res_coku_c[k0,k1],3)]
                    results['Candes with Knockoff'] += [round(can_coku_c[k0,k1],3)]
                    
                    results['Knockoff Variable'] += [k0]
                    results['Comparison Variable'] += [k1]
                    results['Measure'] += ['Right Cokurtosis']
                    results['Real Value'] += [round(res_coku[k1+p,k0+p],3)]
                    results['Value with Feature'] += [round(res_coku[k1+p,k0],3)]
                    results['Candes with Feature'] += [round(can_coku[k1+p,k0],3)]
                    results['Value with Knockoff'] += [round(res_coku[k1,k0],3)]
                    results['Candes with Knockoff'] += [round(can_coku[k1,k0],3)]
    
            results = pd.DataFrame.from_dict(results)
            results.fillna('', inplace=True)
            results.to_csv('knockoffs{}_p{}_n{}_{:03d}.csv'.format(ind,round(math.log(p,2)),round(math.log(n,10)),mi))



