import csv
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

from normal_prior_model import NormalPriorModel
from uniform_prior_model import UniformPriorModel
from inference import get_first_NPI_date, get_last_NPI_date, params_bounds, params_bounds, get_model_class

def load_data(file_name, country_name, burn_fraction=0.6, lim_steps=None):
    # it's the only global point. we initialize all the params here once and don't update it later (only when load_data again for different file_name)
    # TODO PLEASE DONT USE global anywhere in your code
    global official_τ_date, official_τ, incidences, start_date, var_names, nsteps, ndim, N, Td1, Td2, ndays, sample, lnprobability, logliks, model_type, model
    data = np.load(file_name)
    incidences = data['incidences']
    start_date = data['start_date']
    var_names = list(data['var_names'])
    nsteps, ndim, N, Td1, Td2, model_type = data['params']
    chain = data['chain']
    nwalkers = chain.shape[0]
    ndays = len(incidences)
    nburn = int(nsteps * burn_fraction)
    sample = chain[:, nburn:, :].reshape(-1, ndim)
    lnprobability = data['lnprobability'][:, nburn:]
    logliks = data['logliks'].reshape(nwalkers,nsteps)[:,nburn:].reshape(-1)
    if lim_steps:
        sample = chain[:, int(lim_steps * burn_fraction):lim_steps, :].reshape(-1, ndim)
        lnprobability = data['lnprobability'][:, int(lim_steps * burn_fraction):lim_steps]
        logliks = data['logliks'].reshape(nwalkers,nsteps)[:,int(lim_steps * burn_fraction):lim_steps].reshape(-1)
    official_τ_date = get_last_NPI_date(country_name)
    official_τ = (official_τ_date - pd.to_datetime(start_date)).days

    ModelClass = get_model_class(model_type)
    model = ModelClass(country_name, incidences, pd.to_datetime(start_date), N, get_last_NPI_date(country_name), get_first_NPI_date(country_name), params_bounds, Td1, Td2)

def write_csv_header(file_name):
    mean_headers = [v + ' mean' for v in var_names]
    median_headers = [v + ' median' for v in var_names]
    MAP_headers = [v + ' MAP' for v in var_names]
    params_headers = [e for l in zip(mean_headers, median_headers, MAP_headers) for e in l]

    with open(file_name, mode='w') as file: # use pd to write csv files?
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['country','DIC','loglik(MAP)','loglik(mean)','loglik(median)','N','p_steps','p_model_type','p_Td1','p_Td2','official_τ','τ mean','τ median','τ MAP','official τ from 1 Jan','τ mean from 1 Jan','τ median from 1 Jan','τ MAP from 1 Jan','τ CI median (75%)','τ CI median (95%)','τ CI mean (75%)','τ CI mean (95%)', *params_headers])

def write_csv_data(file_name):
    #params means and medians
    means = [round(sample[:,i].mean(),2) for i in range(len(var_names))]
    medians = [round(np.median(sample[:,i]),2) for i in range(len(var_names))]
    MAPs = [round(m,2) for m in get_MAP()]
    params_values = [e for l in zip(means,medians,MAPs) for e in l]

    #tau
    τ_posterior = sample[:,var_names.index('τ')].astype(int)
    τ_mean = format(τ_to_string(τ_posterior.mean()))
    τ_median =  format(τ_to_string(np.median(τ_posterior)))
    τ_MAP =  format(τ_to_string(MAPs[var_names.index('τ')]))

    τ_official_from1Jar = (pd.to_datetime(start_date) - pd.Timestamp('2020-01-01')).days + official_τ
    τ_mean_from1Jar = (pd.to_datetime(start_date) - pd.Timestamp('2020-01-01')).days + τ_posterior.mean()
    τ_median_from1Jar =  (pd.to_datetime(start_date) - pd.Timestamp('2020-01-01')).days + np.median(τ_posterior)
    τ_MAP_from1Jar =  (pd.to_datetime(start_date) - pd.Timestamp('2020-01-01')).days + MAPs[var_names.index('τ')]

    τ_mean_from1Jar = round(τ_mean_from1Jar,2)
    τ_median_from1Jar = round(τ_median_from1Jar,2)
    τ_MAP_from1Jar = round(τ_MAP_from1Jar,2)
    with open(file_name, mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([country_name, '{:.2f}'.format(calc_DIC()), '{:.2f}'.format(calc_LoglikMAP()),'{:.2f}'.format(calc_Loglik_mean()),'{:.2f}'.format(calc_Loglik_median()), N, nsteps, model_type, Td1, Td2,
                         τ_to_string(official_τ),
                         τ_mean, τ_median, τ_MAP, τ_official_from1Jar,  τ_mean_from1Jar, τ_median_from1Jar, τ_MAP_from1Jar, '{:.2f}'.format(calc_τ_CI_median()),'{:.2f}'.format(calc_τ_CI_median(0.95)), '{:.2f}'.format(calc_τ_CI_mean()),'{:.2f}'.format(calc_τ_CI_mean(0.95)), *params_values])

def calc_DIC():
    loglik_E = calc_Loglik_median()
    E_loglik = logliks.mean()
    DIC = 2*loglik_E - 4*E_loglik
    return DIC

def get_MAP():
    return sample[lnprobability.argmax()]

def calc_LoglikMAP():
    res = model.log_likelihood(get_MAP())
    return res

def calc_Loglik_median():
    medians = [np.median(sample[:,i]) for i in range(len(var_names))]
    res = model.log_likelihood(medians)
    return res

def calc_Loglik_mean():
    means = [sample[:,i].mean() for i in range(len(var_names))]
    res = model.log_likelihood(means)
    return res

def calc_τ_CI_median(p=0.75):
    tau_samples = sample.T[var_names.index('τ')]
    tau_samples_hat = np.median(tau_samples)

    res = np.quantile(abs(tau_samples - tau_samples_hat), p)
    return res

def calc_τ_CI_mean(p=0.75):
    tau_samples = sample.T[var_names.index('τ')]
    tau_samples_hat = tau_samples.mean()

    res = np.quantile(abs(tau_samples - tau_samples_hat), p)
    return res

print_list = []
def printt(s):
    print_list.append(s)
    
def print_all():
    for s in print_list:
        print(s)
    
def τ_to_string(τ):
    return (pd.to_datetime(start_date) + timedelta(days=τ)).strftime('%b %d')

# TODO move function to the Model
def generate(Z, D, μ, β1, α1, λ, α2, E0, Iu0,delta_t0,tau,ndays, N):
    tau=int(tau)
    S, E, Ir, Iu, R, Y = model.simulate(Z, D, μ, β1, α1, λ, α2, E0, Iu0,delta_t0,tau,ndays)
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

def bar_plot_τ(ax=None):
    if ax is None: fig, ax = plt.subplots()

    d = dist(sample,'τ')
    d = [int(dd) for dd in d]
    d = pd.value_counts(d).sort_index()
    for i in range(0,len(incidences)):
        if not d.get(i):
            d[i] = 0
    d = d.sort_index()
    # d = d.rename(lambda x: (pd.to_datetime(start_date)+timedelta(days=x)).strftime('%B %d'))
    d = d.apply(lambda x:x/d.sum())
    d.plot(kind='bar',color=green)
    days = list(range(0,ndays,2))
    labels = [τ_to_string(d) for d in days]
    plt.xticks(days,labels);
    plt.title('tau distribution')
    
    taumean = int(dist(sample,'τ').mean())
    taumedian = int(np.median(dist(sample,'τ')))

def plot_τ(ax=None):
    if ax is None: fig, ax = plt.subplots()
    
    ind = var_names.index('τ')
    τ_posterior = sample[:,ind].astype(int)

    τ_mean = τ_posterior.mean()
    τ_median = np.median(τ_posterior)
    printt('τ mean = {}'.format(τ_to_string(τ_mean)))
    printt('τ median = {}'.format(τ_to_string(τ_median)))
    confidence = 'P(τ > {}) = {:.2%}'.format(τ_to_string(official_τ), (τ_posterior > official_τ).mean())
    printt(confidence)

    ax.hist(τ_posterior, bins=np.arange(ndays), density=True, color='k', align='left', width=1)
    ax.axvline(official_τ, color='k', ls='--', alpha=0.75)
    # plt.axvline(τ_median, color=red)

    days = list(range(0, ndays, round(ndays/10)))
    xticklabels = [τ_to_string(d) for d in days]
    ax.set_xticks(days)
    ax.set_xticklabels(xticklabels, rotation=45)
    ax.set_xlim(τ_posterior.min(), τ_posterior.max()+2)
    ax.set_ylim(0, 1)
#     ax.text(14.2, 0.9, confidence, fontsize=16)
    ax.set_ylabel(r'Posterior probability')
    ax.set_xlabel('Effective start of NPI, $\\tau$')
    # plt.tight_layout()
    return ax

def dist(sample,param_name):
    ind = var_names.index(param_name)
    return sample[:,ind]
def plot_hists():
    def hist_param(sample, a):
        d1 = dist(sample,a)
        pd.Series(d1).hist(bins=20, histtype='step',linewidth=1,color=green,label=a,density=True)
        plt.xlim(d1.min(),d1.max())
        plt.grid(b=None)
        plt.yticks([])

        mean = d1.mean()
        #median + quantiles
        q_16, q_50, q_84 = np.quantile(d1,[0.16, 0.5, 0.84])
        q_m, q_p = q_50-q_16, q_84-q_50
        median = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        median = median.format(round(q_50,2), round(q_m,2), round(q_p,2))
        
        plt.axvline(mean,color='blue', linewidth=1, linestyle='--')
        plt.axvline(q_50,color='red', linewidth=1, linestyle='--')
        plt.title('{} mean: {:.2f} \n median: {}'.format(a, mean, median), y=1.04,fontsize=14)
        plt.rc('xtick', labelsize=11)
  
    fig = plt.figure(figsize=(5*3,5*5))
    spec = gridspec.GridSpec(nrows=5, ncols=3, hspace=0.5,wspace=0.3, figure=fig)
    i = -1;
    for r in range(5):
        for c in range(3):
            i+=1
            if i>=len(var_names):
                break
            fig.add_subplot(spec[r, c])
            hist_param(sample,var_names[i])

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

def plot_text(ax=None): #OLD TODO remove
    if ax is None: fig, ax = plt.subplots(figsize=(0.01,0.01))

    txt = '\n'.join(print_list)
    plt.text(0,0,txt,fontsize=10)
    
    plt.setp(ax, frame_on=False, xticks=(), yticks=());
    print_list.clear()

def plot_text(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(0.01,0.01))
    # means = [sample[:,i].mean() for i in range(len(var_names))]
    # medians = [np.median(sample[:,i]) for i in range(len(var_names))]
    # txt = ['{}    mean: {:.2f}\n    median: {:.2f}'.format(t[0],t[1],t[2]) for t in zip(var_names,means,medians)]
    txt = [str(a) for a in print_τ_dist()]
    txt = '\n'.join(txt)
    plt.text(0,0,txt,fontsize=10)
    plt.setp(ax, frame_on=False, xticks=(), yticks=());

def plot_incidences_and_dates(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(0.01,0.01))

    lst = ['Cases',
          incidences.__str__(),
          '',
          'start_date {}'.format(τ_to_string(0)),
          'official tau {}'.format(τ_to_string(official_τ)),
           ''
          ]

    τ_posterior = sample[:,var_names.index('τ')].astype(int)
    τ_mean = τ_posterior.mean()
    τ_median = np.median(τ_posterior)
    τ_MAP = get_MAP()[var_names.index('τ')]
    lst.append('τ mean = {}'.format(τ_to_string(τ_mean)))
    lst.append('τ median = {}'.format(τ_to_string(τ_median)))
    lst.append('τ MAP = {}'.format(τ_to_string(τ_MAP)))
    lst.append('τ CI median 75%: {:.2f}'.format(calc_τ_CI_median(0.75)))
    lst.append('τ CI median 95%: {:.2f}'.format(calc_τ_CI_median(0.95)))
    confidence = 'P(τ > {}) = {:.2%}'.format(τ_to_string(official_τ), (τ_posterior > official_τ).mean())
    lst.append(confidence)

    α1_posterior = sample[:,var_names.index('α1')]
    α2_posterior = sample[:,var_names.index('α2')]
    Δα_posterior = α2_posterior - α1_posterior
    lst.append('P(α2 > α1) = {:.2%}'.format((Δα_posterior > 0).mean()))

    lst.append('')
    lst.append('DIC: {:.2f}'.format(calc_DIC()))
    lst.append('loglik of MAP: {:.2f}'.format(calc_LoglikMAP()))
    lst.append('loglik of mean: {:.2f}'.format(calc_Loglik_mean()))
    lst.append('loglik of median: {:.2f}'.format(calc_Loglik_median()))

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

def print_τ_dist():
    ind = var_names.index('τ')
    τ_posterior = sample[:,ind].astype(int)
    counts, bins = np.histogram(τ_posterior, np.arange(ndays), density=True)
    arr = []
    for b,c in zip(bins, counts):
        if c > 0.001:
            arr.append((τ_to_string(int(b)), round(c,2)))
    #top 10
    s = sorted(arr, key=lambda t: t[1], reverse=True)[:10]
    return sorted(s,key=lambda t: pd.Timestamp('2020 '+t[0]))

def plot_incidences(ax=None, color=blue):
    global daily_cases, theta
    if ax is None: fig, ax = plt.subplots()

    np.random.seed(10)
    num_simulations = 128
    daily_cases = []
    for i in range(num_simulations):
        idx = np.random.choice(sample.shape[0])
        
        Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ = sample[idx, :]
        Δt0 = int(Δt0)
        τ = int(τ)
        total_zeros = params_bounds['Δt0'][1]
        unrellevant_zeros = total_zeros - Δt0
        τ = τ - unrellevant_zeros
        y = generate(Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ, ndays-unrellevant_zeros ,N)       
        
        daily_cases.append(np.array([0]*unrellevant_zeros + list(y)))
    daily_cases = np.array(daily_cases)

    t = np.arange(ndays)
    plt.plot(t, np.median(daily_cases,axis=0), color=color)
    plt.plot(t, incidences, '.', color=red)
        
    ind = var_names.index('τ')
    τ_posterior = sample[:,ind].astype(int)
    τ_mean = τ_posterior.mean()
    plt.axvline(τ_mean,color=purple, linewidth=1, linestyle='--')

    days = list(range(0, ndays, round(ndays/10)))
    labels = [τ_to_string(d) for d in days]
    plt.xticks(days,labels,rotation=90);
    
    plt.ylabel('Daily cases')
    plt.xlabel('Day')
    sns.despine()
    return ax
