# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:29:59 2023

@author: Christopher Hemmens
"""

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', 10)

p = 3
n = 3
max_val = 2**p - 1
ind = '02'
x = [0,3,10,20]
colors = ['k', 'r', 'm', 'g', 'c', 'b']

tests = {}

for mi in x[1:] :
    tests[mi] = pd.read_csv('knockoffs{}_p{}_n{}_{:03d}.csv'.format(ind,p,n,mi),
                            index_col=0)
    tests[mi].dropna(inplace=True)
    tests[mi].set_index(['Measure', 'Knockoff Variable', 'Comparison Variable'], inplace=True)

measures = ['Correlation',
            'Left Coskewness',
            'Right Coskewness',
            'Left Cokurtosis',
            'Central Cokurtosis',
            'Right Cokurtosis']
ii = 1
jj = 0
for measure in measures :
    m1 = measure.split(' ')
    if m1[0] == 'Left' :
        m2 = 'Right ' + m1[1]
        m1 = ' '.join(m1)
    elif m1[0] == 'Right' :
        m2 = 'Left ' + m1[1]
        m1 = ' '.join(m1)
    else :
        m1 = ' '.join(m1)
        m2 = m1
        
    test2 = tests[x[-1]].loc[measure].copy()
    test2 = test2.loc[[(xi,yi) for xi, yi in test2.index if yi > xi]].copy()
    
    selection = [test2.shape[0]*x//5 for x in range(6)]
    selection[-1] -= 1
    
    test2.sort_values('Real Value', inplace=True)
    test2 = test2.iloc[selection].copy()
    test2.sort_index(inplace=True)
    
    if ii > jj :
        ii = 1
        jj += 1
        fig, ax = plt.subplots(1,jj,figsize=(4*jj,4))
    
    kk = 0
    for xi, yi in test2.index :
        yr = [tests[x[-1]].loc[(m1,xi,yi), 'Real Value'] for _ in range(len(x))]
        y1 = [tests[x[-1]].loc[(m1,xi,yi), 'Candes with Feature']]
        y2 = [tests[x[-1]].loc[(m2,yi,xi), 'Candes with Feature']]
        yk = [tests[x[-1]].loc[(m1,xi,yi), 'Candes with Knockoff']]
        
        for mi in x[1:] :
            y1 += [tests[mi].loc[(m1,xi,yi), 'Value with Feature']]
            y2 += [tests[mi].loc[(m2,yi,xi), 'Value with Feature']]
            yk += [tests[mi].loc[(m1,xi,yi), 'Value with Knockoff']]
        
        col = colors[kk]
        kk += 1
        
        if jj == 1 :
            ax.plot(x, yr, col+':')
            ax.plot(x, y1, col+'-')
            ax.plot(x, y2, col+'-')
            ax.plot(x, yk, col+'--')
        else :
            ax[ii-1].plot(x, yr, col+':')
            ax[ii-1].plot(x, y1, col+'-')
            ax[ii-1].plot(x, y2, col+'-')
            ax[ii-1].plot(x, yk, col+'--')
            
    ii += 1
    if ii > jj :
        plt.savefig(f'Graph_{p}_{n}_{jj}.png')
            
            
            
            
            