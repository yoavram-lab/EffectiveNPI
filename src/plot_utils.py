import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import urllib.request
import os
import seaborn as sns
import scipy.stats
from corner import corner 
red, blue, green, purple, orange = sns.color_palette('Set1', 5)
from datetime import timedelta, datetime
from scipy.integrate import odeint
from scipy.stats import poisson
from scipy import stats
sns.set_context('talk')
from rakott.mpl import fig_panel_labels
import warnings
warnings.filterwarnings('ignore')

from inference import ode, simulate, simulate_one

official_τ_dates = {
    'Austria' : datetime(2020, 3, 16),
    'Belgium' : datetime(2020, 3, 18),
    'Denmark' : datetime(2020, 3, 18),
    'France' : datetime(2020, 3, 17),
    'Germany' : datetime(2020, 3, 22),
    'Italy' : datetime(2020, 3, 10),
    'Norway' : datetime(2020, 3, 24),
    'Spain': datetime(2020, 3, 14),
    'Sweden': datetime(2020, 3, 18),
    'Switzerland': datetime(2020, 3, 20),
    'United_Kingdom': datetime(2020, 3, 24),
    'Wuhan' : datetime(2020, 1, 24)
}

def load_data(file_name, burn_fraction=0.6, lim_steps=None):
    # it's the only global point. we initialize all the params here once and don't update it later (only when load_data again for different file_name)
    global official_τ_date,official_τ, incidences, start_date,var_names,nsteps,ndim,N,Td1,Td2,ndays,sample
    data = np.load(file_name)
    incidences = data['incidences']
    start_date = data['start_date']
    var_names = list(data['var_names'])
    nsteps,ndim,N,Td1,Td2 = data['params']
    chain = data['chain']
    ndays = len(incidences)
    nburn = int(nsteps*burn_fraction)
    sample = chain[:, nburn:, :].reshape(-1, ndim)
    if lim_steps:
        sample = chain[:, int(lim_steps*burn_fraction):lim_steps, :].reshape(-1, ndim)
    official_τ_date = official_τ_dates[country_name]
    official_τ = (official_τ_date-pd.to_datetime(start_date)).days

print_list = []
def printt(s):
    print_list.append(s)
def print_all():
    for s in print_list:
        print(s)
    
def tau_to_string(tau):
    return (pd.to_datetime(start_date) + timedelta(days=tau)).strftime('%b %d')

def log_likelihood(θ, X):
    Z, D, μ, β, α1, λ, α2, E0, Iu0, τ = θ
    τ = int(τ)
    
    S, E, Ir, Iu, R, Y = simulate(*θ, ndays, N)
    p1 = 1/Td1
    p2 = 1/Td2
    Xsum = X.cumsum() 
    n = Y[1:] - Xsum[:-1] 
    n = np.maximum(0, n)
    p = ([p1] * τ + [p2] * (ndays - τ))[1:]
    loglik = scipy.stats.poisson.logpmf(X[1:], n * p)
    return loglik.mean()

def generate(Z, D, μ, β1, α1, λ, α2, E0, Iu0,tau,ndays, N):
    tau=int(tau)
    S, E, Ir, Iu, R, Y = simulate(Z, D, μ, β1, α1, λ, α2, E0, Iu0,tau,ndays,N)
    p1 = 1/Td1
    p2 = 1/Td2 
    C = np.zeros_like(Y)
    for t in range(1, len(C)):
        p = p1 if t<tau else p2
        n = Y[t] - C[:t].sum()
        n = max(0,n)
        C[t] = np.random.poisson(n * p)     

    return C

def plot_β(ax=None):
    if ax is None: fig, ax = plt.subplots()
        
    ind = var_names.index('β')
    β_posterior = sample[:,ind]
    counts, bins = np.histogram(β_posterior, 100)
    idx = (β_posterior > bins[counts.argmax()]) & (β_posterior < bins[counts.argmax()+1]) 
    idx[np.random.random(idx.size) > 0.6] = False
    β_posterior= β_posterior[~idx]
    idx = β_posterior > 0.8
    β_posterior= β_posterior[idx]

    β_mean = β_posterior.mean()
    β_median = np.median(β_posterior)
    printt('β mean = {:.2}'.format(β_mean))
    printt('β median = {:.2}'.format(β_median))

    sns.distplot(β_posterior, 100, hist=False, norm_hist=True, kde_kws=dict(bw=0.06), color=red, label=r'Before NPI', ax=ax)
#     ax.axvline(β_mean, color=red, ls='--', alpha=0.7)
#     ax.set_ylim(0, 2)

    ind = var_names.index('λ')
    λ_posterior = sample[:,ind]
    ind = var_names.index('β')
    β_posterior = sample[:,ind]
    βλ_posterior = β_posterior * λ_posterior

    βλ_mean = βλ_posterior.mean()
    βλ_median = np.median(βλ_posterior)
    printt('βλ mean = {:.2}'.format(βλ_mean))
    printt('βλ median = {:.2}'.format(βλ_median))

    sns.distplot(βλ_posterior, 100, hist=False, norm_hist=True, kde_kws=dict(bw=0.06), color=blue, label=r'After NPI', ax=ax)
#     ax.axvline(βλ_mean, color=blue, ls='--', alpha=0.7)
    ax.set_ylabel(r'Posterior density')
    ax.set_xlabel(r'Transmission rate, $\beta$')
#     ax.set_xlim(0.2, 1.5)

    ax.legend()
    sns.despine()
    return ax

def plot_α(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    ind = var_names.index('α1')
    α1_posterior = sample[:,ind]

    α1_mean = α1_posterior.mean()
    α1_median = np.median(α1_posterior)
    printt('α1 mean = {:.2}'.format(α1_mean))
    printt('α1 median = {:.2}'.format(α1_median))

    sns.distplot(α1_posterior, 100, hist=False, norm_hist=True, kde_kws=dict(bw=0.06), color=red, label=r'Before NPI', ax=ax)
#     ax.axvline(α1_mean, color=red, ls='--', alpha=0.7)

    ind = var_names.index('α2')
    α2_posterior = sample[:,ind]

    α2_mean = α2_posterior.mean()
    α2_median = np.median(α2_posterior)
    printt('α2 mean = {:.2}'.format(α2_mean))
    printt('α2 median = {:.2}'.format(α2_median))

    sns.distplot(α2_posterior, 100, hist=False, norm_hist=True, kde_kws=dict(bw=0.06), color=blue, label=r'After NPI', ax=ax)
#     ax.axvline(α2_mean, color=blue, ls='--', alpha=0.7)
    ax.set_ylabel(r'Posterior density')
    ax.set_xlabel(r'Reporting rate, $\alpha$')
    ax.legend()
    sns.despine()
    return ax

def plot_α2_minus_α1(ax=None):
    if ax is None: fig, ax = plt.subplots()

    ind = var_names.index('α1')
    α1_posterior = sample[:,ind]
    ind = var_names.index('α2')
    α2_posterior = sample[:,ind]

    Δα_posterior = α2_posterior - α1_posterior
    sns.distplot(Δα_posterior, 100, kde_kws=dict(bw=0.04), norm_hist=True)
    plt.axvline(0, color='k')
    printt('P(α2 > α1) = {:.2%}'.format((Δα_posterior > 0).mean()))
    plt.xlabel('α2-α1')
    plt.ylabel('Posterior density');

def plot_τ(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    ind = var_names.index('τ')
    τ_posterior = sample[:,ind].astype(int)

    τ_mean = τ_posterior.mean()
    τ_median = np.median(τ_posterior)
    printt('τ mean = {}'.format(tau_to_string(τ_mean)))
    printt('τ median = {}'.format(tau_to_string(τ_median)))
    confidence = 'P(τ > {}) = {:.2%}'.format(tau_to_string(official_τ), (τ_posterior > official_τ).mean())
    printt(confidence)

    ax.hist(τ_posterior, bins=np.arange(ndays), density=True, color='k', align='left', width=1)
    ax.axvline(official_τ, color='k', ls='--', alpha=0.75)
    # plt.axvline(τ_median, color=red)

    days = list(range(0, ndays, round(ndays/10)))
    xticklabels = [tau_to_string(d) for d in days]
    ax.set_xticks(days)
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_xlim(τ_posterior.min(), τ_posterior.max()+2)
    ax.set_ylim(0, 1)
#     ax.text(14.2, 0.9, confidence, fontsize=16)
    ax.set_ylabel(r'Posterior probability')
    ax.set_xlabel('Effective start of NPI, $\\tau$')
    # plt.tight_layout()
    return ax

def plot_all():
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(country_name, fontsize=32)
    gs = gridspec.GridSpec(4, 2)

    ax = plot_τ(fig.add_subplot(gs[0, :]))
    ax = plot_β(fig.add_subplot(gs[1, 0]))
    ax.legend().set_visible(False)
    ax = plot_α(fig.add_subplot(gs[1, 1]))
    plot_incidences(fig.add_subplot(gs[2, 0]))
    ax = plot_α2_minus_α1(fig.add_subplot(gs[2, 1]))
    
    plot_text(fig.add_subplot(gs[3, 0]))
    plot_incidences_and_dates(fig.add_subplot(gs[3, 1]))
#     fig_panel_labels(np.array(fig.axes), xcoord=0.01)
    fig.tight_layout()
    
    return fig
def plot_text(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(0.01,0.01))

    txt = '\n'.join(print_list)
    plt.text(0,0,txt,fontsize=10)
    
    plt.setp(ax, frame_on=False, xticks=(), yticks=());
    print_list.clear()
def plot_incidences_and_dates(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(0.01,0.01))

    lst = ['Cases',
          incidences.__str__(),
           'start date and official tau',
           '{} - {}'.format(tau_to_string(0),tau_to_string(official_τ))
          ]
    txt = '\n'.join(lst)
    plt.text(0,0,txt,fontsize=10)
    plt.setp(ax, frame_on=False, xticks=(), yticks=());

def plot_corner():
    θ = np.mean(sample, axis=0)
    cor = corner(sample, 
        smooth=True,
        labels=var_names,
        show_titles=True,
    )
    axes = np.array(cor.axes).reshape((ndim, ndim))
    for i, var in enumerate(θ):
        axes[i, i].axvline(var, color=green)
def print_tau_dist():
    ind = var_names.index('τ')
    τ_posterior = sample[:,ind].astype(int)
    counts, bins = np.histogram(τ_posterior, np.arange(ndays), density=True)
    for b,c in zip(bins, counts):
        if c > 0:
            print(tau_to_string(int(b)), c)
def plot_incidences(ax=None):
    if ax is None: fig, ax = plt.subplots()

    np.random.seed(10)
    num_simulations = 300
    daily_cases = []
    for _ in range(num_simulations):
        idx = np.random.choice(sample.shape[0])
        y = generate(*sample[idx, :],ndays,N)
        daily_cases.append(y)
    daily_cases = np.array(daily_cases)

    t = np.arange(ndays)
    plt.plot(t, np.median(daily_cases,axis=0), color=blue)
    plt.plot(t, incidences, '.', color=red)
        
    ind = var_names.index('τ')
    τ_posterior = sample[:,ind].astype(int)
    τ_mean = τ_posterior.mean()
    plt.axvline(τ_mean,color=purple, linewidth=1, linestyle='--')

    days = list(range(0, ndays, round(ndays/10)))
    labels = [tau_to_string(d) for d in days]
    plt.xticks(days,labels,rotation=90);
    
    plt.ylabel('Daily cases')
    plt.xlabel('Day')
    sns.despine()
