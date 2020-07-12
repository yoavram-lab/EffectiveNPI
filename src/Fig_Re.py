#!/usr/bin/env python
# coding: utf-8
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('paper', font_scale=1.3)
red, blue, green = sns.color_palette('Set1', 3)
import pycountry # https://pypi.org/project/pycountry/

import os
from glob import glob
from datetime import datetime, timedelta

from rakott.mpl import savefig_bbox, fig_panel_labels

colors = sns.color_palette('Paired')


if __name__ == '__main__':
    job_id_free = '2020-05-26-Apr11'
    job_id_fixed = '2020-06-25-Apr11-fixedtau'

    dfs = []
    countries = ['Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Italy', 'Norway', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']    
    for country in countries:   
        fname = r'../output/{}/figures/Re_{}.csv'.format(job_id_free, country.replace(' ', '_')) 
        df = pd.read_csv(fname)        
        df['country'] = country
        df['model'] = 'free'
        dfs.append(df)
        fname = r'../output/{}/figures/Re_{}.csv'.format(job_id_fixed, country.replace(' ', '_')) 
        df = pd.read_csv(fname)
        df['country'] = country
        df['model'] = 'fixed'
        dfs.append(df)

    df = pd.concat(dfs)
    grp = df.groupby(['country', 'model'])
    agg = grp.agg(
        low=pd.NamedAgg(column="rel_reduc_Re", aggfunc=lambda x: np.percentile(x, 25)),
        median=pd.NamedAgg(column="rel_reduc_Re", aggfunc=np.median),
        high=pd.NamedAgg(column="rel_reduc_Re", aggfunc=lambda x: np.percentile(x, 75))
    ).reset_index()
    df_free = agg[agg['model'] == 'free']
    df_fixed = agg[agg['model'] == 'fixed']
    countries = agg['country'].unique()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8*2/3), sharex=True, sharey=True)
    ax.errorbar(
        df_free['median'], 
        df_fixed['median'], 
        xerr=[df_free['median']-df_free['low'], df_free['high']-df_free['median']],
        yerr=[df_fixed['median']-df_fixed['low'], df_fixed['high']-df_fixed['median']],
        ls='', marker='o',  ecolor='gray'
    )
    for i, country in enumerate(countries):
        ax.text(
            df_free.loc[df_free['country']==country, 'median']-0.035,
            df_fixed.loc[df_fixed['country']==country, 'median'],
            pycountry.countries.search_fuzzy(country)[0].alpha_2,
            horizontalalignment='left', verticalalignment='bottom', color='k', fontsize=10
        )
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), color='k', ls='--')
    ax.set(xlim=(0, 0.9), ylim=(0, 0.6), xticks=np.arange(0, 1, 0.2), yticks=np.arange(0, 1, 0.2),
           xlabel=r'Relative reduction in $R$ with Free $\tau$', ylabel=r'Relative reduction in $R$ with Fixed $\tau$',)# xticks=np.arange(1, 8), yticks=np.arange(1,3))
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    sns.despine()
    fig_fname = '../figures/Fig_Re.pdf'
    fig.savefig(fig_fname, dpi=100)
    print(fig_fname)
