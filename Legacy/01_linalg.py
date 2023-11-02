# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:29:59 2023

@author: Finch
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
ind = '01'

for p in ps[:3] :
    for n in ns[:3] :
        print(f'{p}\t{n}')
        rng = np.random.default_rng(24)
        test = rng.uniform(0,1,(p,n))
        
        for k in range(p) :
            if k%4 == 0 : 
                pt = test[k,:]/sum(test[k,:])
                test[k,:] = rng.choice(test[k,:], size=n, replace=False, p=pt)
            elif k%4 == 1 :
                pt = [1 - x for x in test[k,:]]
                pt /= sum(pt)
                test[k,:] = rng.choice(test[k,:], size=n, replace=False, p=pt)
            elif k%4 == 2 :
                pt = [abs(x-0.5) for x in test[k,:]]
                pt /= sum(pt)
                test[k,:] = rng.choice(test[k,:], size=n, replace=False, p=pt)
        
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
        
        results = {}
        results['Knockoff Variable'] = [None]
        results['Comparison Variable'] = [None]
        results['Measure'] = ['Time Taken']
        results['Real Value'] = [(end-start).seconds]
        results['Value with Feature'] = [None]
        results['Value with Knockoff'] = [None]
        for k0 in range(p) :
            results['Knockoff Variable'] += [k0]
            results['Comparison Variable'] += [None]
            results['Measure'] += ['Mean'] 
            results['Real Value'] += [round(res.x[k0*n:(k0+1)*n].mean(),4)]
            results['Value with Feature'] += [None]
            results['Value with Knockoff'] += [None]
            
            results['Knockoff Variable'] += [k0]
            results['Comparison Variable'] += [None]
            results['Measure'] += ['Variance'] 
            results['Real Value'] += [round(res.x[k0*n:(k0+1)*n].var(),4)]
            results['Value with Feature'] += [None]
            results['Value with Knockoff'] += [None]
            
            results['Knockoff Variable'] += [k0]
            results['Comparison Variable'] += [None]
            results['Measure'] += ['Uniformity_pvalue'] 
            results['Real Value'] += [round(kstest(res.x[k0*n:(k0+1)*n],"uniform").pvalue,4)]
            results['Value with Feature'] += [None]
            results['Value with Knockoff'] += [None]
            for k1 in range(p) :                
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [k1]
                results['Measure'] += ['Correlation']
                results['Real Value'] += [round(np.corrcoef(test[k0,:],test[k1,:])[0,1],3)]
                results['Value with Feature'] += [round(np.corrcoef(res.x[k0*n:(k0+1)*n],test[k1,:])[0,1],3)]
                results['Value with Knockoff'] += [round(np.corrcoef(res.x[k0*n:(k0+1)*n],res.x[k1*n:(k1+1)*n])[0,1],3)]
                
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [k1]
                results['Measure'] += ['Left Coskewness']
                results['Real Value'] += [round(ko.coskew(test[k0,:],test[k1,:],"left"),3)]
                results['Value with Feature'] += [round(ko.coskew(res.x[k0*n:(k0+1)*n],test[k1,:],"left"),3)]
                results['Value with Knockoff'] += [round(ko.coskew(res.x[k0*n:(k0+1)*n],res.x[k1*n:(k1+1)*n],"left"),3)]
                
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [k1]
                results['Measure'] += ['Right Coskewness']
                results['Real Value'] += [round(ko.coskew(test[k0,:],test[k1,:],"right"),3)]
                results['Value with Feature'] += [round(ko.coskew(res.x[k0*n:(k0+1)*n],test[k1,:],"right"),3)]
                results['Value with Knockoff'] += [round(ko.coskew(res.x[k0*n:(k0+1)*n],res.x[k1*n:(k1+1)*n],"right"),3)]
        
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [k1]
                results['Measure'] += ['Left Cokurtosis']
                results['Real Value'] += [round(ko.cokurt(test[k0,:],test[k1,:],"left"),3)]
                results['Value with Feature'] += [round(ko.cokurt(res.x[k0*n:(k0+1)*n],test[k1,:],"left"),3)]
                results['Value with Knockoff'] += [round(ko.cokurt(res.x[k0*n:(k0+1)*n],res.x[k1*n:(k1+1)*n],"left"),3)]
                
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [k1]
                results['Measure'] += ['Central Cokurtosis']
                results['Real Value'] += [round(ko.cokurt(test[k0,:],test[k1,:],"center"),3)]
                results['Value with Feature'] += [round(ko.cokurt(res.x[k0*n:(k0+1)*n],test[k1,:],"center"),3)]
                results['Value with Knockoff'] += [round(ko.cokurt(res.x[k0*n:(k0+1)*n],res.x[k1*n:(k1+1)*n],"center"),3)]
                
                results['Knockoff Variable'] += [k0]
                results['Comparison Variable'] += [k1]
                results['Measure'] += ['Right Cokurtosis']
                results['Real Value'] += [round(ko.cokurt(test[k0,:],test[k1,:],"right"),3)]
                results['Value with Feature'] += [round(ko.cokurt(res.x[k0*n:(k0+1)*n],test[k1,:],"right"),3)]
                results['Value with Knockoff'] += [round(ko.cokurt(res.x[k0*n:(k0+1)*n],res.x[k1*n:(k1+1)*n],"right"),3)]

        results = pd.DataFrame.from_dict(results)
        results.to_excel(f'knockoffs{ind}_p{round(math.log(p,2))}_n{round(math.log(n,10))}.xlsx')

"""
Feature only cov: 2H46
Feature only cov + coskew: 2H55
Feature only cov + coskew + cokurt: 3H40

Feat + knockoff cov: 0H22
Feat + knockoff cov + coskew: 0H22



for k in range(p) :
    print(k)
    print(f'Mean: {round(res.x[k*n:(k+1)*n].mean(),4)}')
    print(f'Var: {round(res.x[k*n:(k+1)*n].var(),4)}')
    print(f'KS pvalue: {round(kstest(res.x[k*n:(k+1)*n],"uniform").pvalue,4)}')
    
    print('Correlation: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        base = round(np.corrcoef(test[k,:],test[k1,:])[0,1],3)
        feat = round(np.corrcoef(res.x[k*n:(k+1)*n],test[k1,:])[0,1],3)
        knock = round(np.corrcoef(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n])[0,1],3)
        print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
            
    print('Left Coskewness: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        base = round(coskew(test[k,:],test[k1,:],"left"),3)
        feat = round(coskew(res.x[k*n:(k+1)*n],test[k1,:],"left"),3)
        knock = round(coskew(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n],"left"),3)
        print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
            
    print('Right Coskewness: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        base = round(coskew(test[k,:],test[k1,:],"right"),3)
        feat = round(coskew(res.x[k*n:(k+1)*n],test[k1,:],"right"),3)
        knock = round(coskew(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n],"right"),3)
        print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
        
    print('Left Cokurtosis: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        base = round(cokurt(test[k,:],test[k1,:],"left"),3)
        feat = round(cokurt(res.x[k*n:(k+1)*n],test[k1,:],"left"),3)
        knock = round(cokurt(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n],"left"),3)
        print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
            
    print('Center Cokurtosis: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        base = round(cokurt(test[k,:],test[k1,:],"center"),3)
        feat = round(cokurt(res.x[k*n:(k+1)*n],test[k1,:],"center"),3)
        knock = round(cokurt(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n],"center"),3)
        print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
            
    print('Right Cokurtosis: Real – With Feature – With Knockoff')
    for k1 in range(p) :
        base = round(cokurt(test[k,:],test[k1,:],"right"),3)
        feat = round(cokurt(res.x[k*n:(k+1)*n],test[k1,:],"right"),3)
        knock = round(cokurt(res.x[k*n:(k+1)*n],res.x[k1*n:(k1+1)*n],"right"),3)
        print(f'{k}\t{k1}\t{base}\t{feat}\t{knock}')
            
    print(' ')

"""









