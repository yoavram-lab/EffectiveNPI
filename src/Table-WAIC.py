#!/usr/bin/env python
# coding: utf-8
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.special import logsumexp
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import os
from datetime import datetime, timedelta

from rakott.mpl import fig_panel_labels, fig_xlabel, fig_ylabel, savefig_bbox

from inference import find_start_day
from Fig_ppc import load_data

sns.set_context('paper', font_scale=1.3)
red, blue, green = sns.color_palette('Set1', 3)


def load_chain(job_id, country, burn_fraction=0.6):
    fname = os.path.join(output_folder, job_id, 'inference', '{}.npz'.format(country))
    inference_data = np.load(fname)
    nsteps, ndim, N, Td1, Td2, model_type = inference_data['params']
    logliks = inference_data['logliks']
    nchains = logliks.size // nsteps
    logliks = logliks.reshape(nchains, nsteps)
    nburn = int(nsteps*burn_fraction)
    logliks = logliks[:, nburn:]
    return logliks


def inliers(logliks, PLOT=False):
    chain_mean_loglik = logliks.mean(axis=1)
    std_mean_loglikg = chain_mean_loglik.std(ddof=1)
    mean_mean_loglikg = chain_mean_loglik.mean()
    idx = abs(chain_mean_loglik - mean_mean_loglikg) < 3*std_mean_loglikg
    if PLOT:
        if idx.any():
            plt.plot(logliks[idx, ::1000].T, '.k', label='inliers')
        if (~idx).any():
            plt.plot(logliks[~idx, ::1000].T, '.r', label='outliers')
        plt.ylabel('Log-likelihood')
        plt.legend()
    return idx


def WAIC(logliks):
    logliks = logliks[inliers(logliks)]
    S = logliks.size
    llpd = -np.log(S) + logsumexp(logliks)
    p1 = 2*(-np.log(S) + logsumexp(logliks) - logliks.mean())
    p2 = np.var(logliks, ddof=1)
    return -2*(llpd + -p1), -2*(llpd + -p2)

if __name__ == '__main__':    
    output_folder = r'/Users/yoavram/Library/Mobile Documents/com~apple~CloudDocs/EffectiveNPI-Data/output'
    
    job_ids = ['2020-05-14-n1-normal-1M', '2020-05-14-n1-notau-1M', '2020-05-15-n1-fixed-tau-1M']
    print('Job IDs:')
    print(job_ids)    
    countries = 'Austria Belgium Denmark France Germany Italy Norway Spain Sweden Switzerland United_Kingdom Wuhan'.split(' ')
    print('Countries:')

    results = []
    for country in tqdm(countries):
        for job_id in job_ids:
            chain_fname = os.path.join(output_folder, job_id, 'inference', '{}.npz'.format(country))
            logliks = load_chain(job_id, country)
            waic1, waic2 = WAIC(logliks)
            results.append(dict(
                country=country,
                job_id=job_id,
                WAIC1=waic1,
                WAIC2=waic2
            ))

    df = pd.DataFrame(results)
    df.loc[df['job_id'] == '2020-05-14-n1-normal-1M', 'job_id'] = 'Free'
    df.loc[df['job_id'] == '2020-05-14-n1-notau-1M', 'job_id'] = 'No'
    df.loc[df['job_id'] == '2020-05-15-n1-fixed-tau-1M', 'job_id'] = 'Fixed'
    df = df.rename(columns={'country':'Country', 'job_id':'Model'})
    df['Country'] = [x.replace('_', ' ') for x in df['Country']]
    df.loc[df['Country']=='Wuhan', 'Country'] = 'Wuhan China'
    
    df = pd.pivot(df, index='Country', columns='Model')
    
    df = df.drop(columns='WAIC1')
    df = df.droplevel(0, axis=1)

    idx = df['Free']==df.min(axis=1)
    df.loc[idx, 'Free'] = ['\\textbf{'+'{:.2f}'.format(x)+'}' for x in df.loc[idx, 'Free']] 
    df.loc[~idx, 'Free'] = ['{:.2f}'.format(x) for x in df.loc[~idx, 'Free']] 
    
    df.to_csv('../figures/Table-WAIC.csv', index='Country', float_format="%.2f")
    print("Saved to ../figures/Table-WAIC.csv")
    
