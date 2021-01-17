import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import urllib.request
import os
import seaborn as sns
import scipy.stats
from scipy.special import logsumexp
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

from model.normal_prior_model import NormalPriorModel
from model.uniform_prior_model import UniformPriorModel
from inference import get_first_NPI_date, get_last_NPI_date, params_bounds, params_bounds, get_model_class

country_name = None
def load_data(file_name, _country_name, nburn=2_000_000, lim_steps=None, delete_chain_less_than=None): 
    # it's the only global point. we initialize all the params here once and don't update it later (only when load_data again for different file_name)
    # TODO PLEASE DONT USE global anywhere in your code
    global first_NPI, last_NPI, incidences, start_date, var_names, nsteps, ndim, N, Td1, Td2, ndays, sample, lnprobability, logliks, model_type, model, country_name
    country_name = _country_name
    data = np.load(file_name)
    incidences = data['incidences']
    start_date = data['start_date']
    var_names = list(data['var_names'])
    nsteps, ndim, N, Td1, Td2, model_type = data['params']
    chain = data['chain']
    nwalkers = chain.shape[0] #nwalkers before deleting bad chain
    ndays = len(incidences)        

    if delete_chain_less_than:
        if len((chain[:,1_000_000, var_names.index('τ')]<delete_chain_less_than).nonzero())>1:
            raise AssertionError('too many bad chains')
        bad_chain_ind = (chain[:,1_000_000, var_names.index('τ')]<delete_chain_less_than).nonzero()[0][0]
        chain = np.delete(chain, bad_chain_ind, axis=0)

    sample = chain[:, nburn:lim_steps, :].reshape(-1, ndim)
    # try:
    #     sample[:,var_names.index('τ')] = sample[:,var_names.index('τ')].astype(int) #in inference we allways convert it to int
    # except ValueError: #if the model doesn't have such parameter
    #     None
    # try:
    #     sample[:,var_names.index('Δt0')] = sample[:,var_names.index('Δt0')].astype(int) #in inference we allways convert it to int
    # except ValueError:#if the model doesn't have such parameter
    #     None

    lnprobability = data['lnprobability'][:, nburn:lim_steps]
    logliks = data['logliks'].reshape(nwalkers,nsteps)[:,nburn:lim_steps]
    if delete_chain_less_than:
            lnprobability = np.delete(lnprobability, bad_chain_ind, axis=0)
            logliks = np.delete(logliks, bad_chain_ind, axis=0)

    last_NPI = (get_last_NPI_date(country_name) - pd.to_datetime(start_date)).days
    first_NPI = (get_first_NPI_date(country_name) - pd.to_datetime(start_date)).days

    ModelClass = get_model_class(model_type)
    model = ModelClass(country_name, incidences, pd.to_datetime(start_date), N, get_last_NPI_date(country_name), get_first_NPI_date(country_name), params_bounds, Td1, Td2)

def write_csv_header(file_name):
    mean_headers = [v + ' mean' for v in var_names]
    median_headers = [v + ' median' for v in var_names]
    MAP_headers = [v + ' MAP' for v in var_names]
    params_headers = [e for l in zip(mean_headers, median_headers, MAP_headers) for e in l]

    with open(file_name, mode='w') as file: # use pd to write csv files?
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['country','WAIC', 'DIC using median', 'DIC using mean', 'DIC using MAP','loglik(MAP)','loglik(mean)','loglik(median)','N','p_steps','p_model_type','p_Td1','p_Td2','official_τ','τ mean','τ median','τ MAP','τ hpd 75% from','τ hpd 75% to','τ hpd 95% from',
        'τ hpd 95% to',
        'official τ from 1 Jan','τ mean from 1 Jan','τ median from 1 Jan','τ MAP from 1 Jan','τ CrI median (75%)','τ CrI median (95%)','τ CrI mean (75%)','τ CrI mean (95%)', *params_headers,'τ hpd 75% from','τ hpd 75% to n','τ hpd 95% from n',
        'τ hpd 95% to n', 'τ hpd CrI 75%', 'τ hpd CrI 95%' ])

def write_csv_data(file_name):
    #params means and medians
    means = [round(sample[:,i].mean(),4) for i in range(len(var_names))]
    medians = [round(np.median(sample[:,i]),4) for i in range(len(var_names))]
    MAPs = [round(m,4) for m in get_MAP()]
    params_values = [e for l in zip(means,medians,MAPs) for e in l]

    #tau
    try:
        τ_posterior = sample[:,var_names.index('τ')].astype(int)
        τ_mean = format(τ_to_string(τ_posterior.mean()))
        τ_median = format(τ_to_string(np.median(τ_posterior)))
        τ_hpd_75_from, τ_hpd_75_to =  calc_τ_hpd(mass_frac=0.75)
        τ_hpd_95_from, τ_hpd_95_to =  calc_τ_hpd(mass_frac=0.95)
        τ_hpd_75_from_date = format(τ_to_string(τ_hpd_75_from))
        τ_hpd_75_to_date = format(τ_to_string(τ_hpd_75_to))
        τ_hpd_95_from_date = format(τ_to_string(τ_hpd_95_from))
        τ_hpd_95_to_date = format(τ_to_string(τ_hpd_95_to))
        τ_MAP =  format(τ_to_string(MAPs[var_names.index('τ')]))
        τ_MAP_i = MAPs[var_names.index('τ')]
    except ValueError: # model with fixed τ
        try:
            τ_posterior = np.array([model.τ])
            τ_mean = model.τ
            τ_median =  model.τ
            τ_MAP =  model.τ
            τ_MAP_i = model.τ
        except AttributeError: #model without τ
            τ_posterior =np.array([0])
            τ_mean = 0
            τ_median =  0
            τ_MAP =  0
            τ_MAP_i = 0

    τ_official_from1Jar = (pd.to_datetime(start_date) - pd.Timestamp('2020-01-01')).days + last_NPI
    τ_mean_from1Jar = (pd.to_datetime(start_date) - pd.Timestamp('2020-01-01')).days + τ_posterior.mean()
    τ_median_from1Jar =  (pd.to_datetime(start_date) - pd.Timestamp('2020-01-01')).days + np.median(τ_posterior)
    τ_MAP_from1Jar =  (pd.to_datetime(start_date) - pd.Timestamp('2020-01-01')).days + τ_MAP_i

    τ_mean_from1Jar = round(τ_mean_from1Jar,4)
    τ_median_from1Jar = round(τ_median_from1Jar,4)
    τ_MAP_from1Jar = round(τ_MAP_from1Jar,4)

    with open(file_name, mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow([country_name,'{:.4f}'.format(calc_WAIC()), '{:.4f}'.format(calc_DIC(calc_loglik_median)), '{:.4f}'.format(calc_DIC(calc_loglik_mean)), '{:.4f}'.format(calc_DIC(calc_loglikMAP)), '{:.4f}'.format(calc_loglikMAP()),'{:.4f}'.format(calc_loglik_mean()),'{:.4f}'.format(calc_loglik_median()), N, nsteps, model_type, Td1, Td2,
                         τ_to_string(last_NPI),
                         τ_mean, τ_median,τ_MAP, τ_hpd_75_from_date,τ_hpd_75_to_date, τ_hpd_95_from_date,τ_hpd_95_to_date,
                         τ_official_from1Jar,  τ_mean_from1Jar, τ_median_from1Jar, τ_MAP_from1Jar, '{:.4f}'.format(calc_τ_CI_median()),'{:.4f}'.format(calc_τ_CI_median(0.95)),
                          '{:.4f}'.format(calc_τ_CI_mean()),'{:.4f}'.format(calc_τ_CI_mean(0.95)), *params_values, '{:.4f}'.format(τ_hpd_75_from), '{:.4f}'.format(τ_hpd_75_to), '{:.4f}'.format(τ_hpd_95_from), '{:.4f}'.format(τ_hpd_95_to)
                          , '{:.4f}'.format(τ_hpd_75_to-τ_hpd_75_from), '{:.4f}'.format(τ_hpd_95_to-τ_hpd_95_from)])

def calc_DIC(loglik_E_func):
    loglik_E = loglik_E_func()
    E_loglik = logliks.mean()
    DIC = 2*loglik_E - 4*E_loglik
    return DIC

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

def calc_WAIC():
    l_logliks = logliks[inliers(logliks)]
    S = l_logliks.size
    llpd = -np.log(S) + logsumexp(l_logliks)
    p1 = 2*(-np.log(S) + logsumexp(l_logliks) - l_logliks.mean())
    p2 = np.var(l_logliks, ddof=1)
    return -2*(llpd + -p2)

def get_MAP():
    return sample[lnprobability.argmax()]

def calc_loglikMAP():
    res = model.log_likelihood(get_MAP())
    return res

def calc_loglik_median():
    medians = [np.median(sample[:,i]) for i in range(len(var_names))]
    res = model.log_likelihood(medians)
    return res

def calc_loglik_mean():
    means = [sample[:,i].mean() for i in range(len(var_names))]
    res = model.log_likelihood(means)
    return res

def calc_hpd(trace, mass_frac) :
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])

def calc_τ_hpd(mass_frac=0.75):
    # hpd(sample.T[var_names.index('τ')].astype(int), p)
    return calc_hpd(sample.T[var_names.index('τ')], mass_frac)

def calc_τ_CI_median(p=0.75):
    try:
        tau_samples = sample.T[var_names.index('τ')]
    except ValueError: #model without τ
        return 0
    tau_samples_hat = np.median(tau_samples)

    res = np.quantile(abs(tau_samples - tau_samples_hat), p)
    return res

def calc_τ_CI_mean(p=0.75):
    try:
        tau_samples = sample.T[var_names.index('τ')]
    except ValueError: #model without τ
        return 0
    tau_samples_hat = tau_samples.mean()

    res = np.quantile(abs(tau_samples - tau_samples_hat), p)
    return res
    
def τ_to_string(τ):
    return (pd.to_datetime(start_date) + timedelta(days=τ)).strftime('%b %d')

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

    sns.distplot(β_posterior, 100, hist=False, norm_hist=True, kde_kws=dict(bw=0.06), color=red, label=r'Before NPI', ax=ax)

    ind = var_names.index('λ')
    λ_posterior = sample[:,ind]
    ind = var_names.index('β')
    β_posterior = sample[:,ind]
    βλ_posterior = β_posterior * λ_posterior

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

    sns.distplot(α1_posterior, 100, hist=False, norm_hist=True, kde_kws=dict(bw=0.06), color=red, label=r'Before NPI', ax=ax)
#     ax.axvline(α1_mean, color=red, ls='--', alpha=0.7)

    ind = var_names.index('α2')
    α2_posterior = sample[:,ind]

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
    
    try:
        τ_posterior = model.τ # model with fixed τ
    except AttributeError:
        ind = var_names.index('τ')
        τ_posterior = sample[:,ind].astype(int)

    ax.hist(τ_posterior, bins=np.arange(ndays), density=True, color='k', align='left', width=1)
    ax.axvline(first_NPI, color='k', ls='--', alpha=0.75)
    ax.axvline(last_NPI, color='k', ls='--', alpha=0.75)
    # plt.axvline(τ_median, color=red)

    days = list(range(0, ndays, round(ndays/10)))
    xticklabels = [τ_to_string(d) for d in days]
    ax.set_xticks(days)
    ax.set_xticklabels(xticklabels, rotation=45)

    # TODO do we need it?
    # try:
    #     ax.set_xlim(τ_posterior.min(), τ_posterior.max()+2)
    # except:
    #     None #it's okey for model with fixed τ

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

def plot_joint_dists(randindxs=None, title=None): 
    if not randindxs:
        np.random.seed(10)
        randindxs = np.random.choice(len(sample),20000,replace=False)
        
    def hex_wrapper(*args, **kwargs):
        kwargs['color'] = None
        return plt.hexbin(*args,**kwargs)
    
    post_sample = sample[randindxs]
    #dirty trick to ensure the axis is in [min,max]
    for i,var in enumerate(var_names[:-1]):
        post_sample[0,i] = params_bounds[var][0]
        post_sample[1,i] = params_bounds[var][1]
    # post_sample[0,var_names.index('τ')] = 5
    # post_sample[1,var_names.index('τ')] = 40
    df = pd.DataFrame(post_sample)
    df.columns=var_names    

    g = sns.PairGrid(df, y_vars=["τ"], x_vars=var_names[:-1], height=6)
    cmap = sns.cubehelix_palette(8,start=10,light=1, as_cmap=True)
    g = g.map(hex_wrapper, gridsize=47,cmap=cmap) #'Greys'
    g.fig.suptitle(title)
    return g.fig

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
    
    plot_τ_dist(fig.add_subplot(gs[3, 0]))
    plot_info(fig.add_subplot(gs[3, 1]))
#     fig_panel_labels(np.array(fig.axes), xcoord=0.01)
    fig.tight_layout()
    
    return fig

def plot_τ_dist(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(0.01,0.01))
    # means = [sample[:,i].mean() for i in range(len(var_names))]
    # medians = [np.median(sample[:,i]) for i in range(len(var_names))]
    # txt = ['{}    mean: {:.2f}\n    median: {:.2f}'.format(t[0],t[1],t[2]) for t in zip(var_names,means,medians)]
    
    try:
        txt = ['fixed tau: '+τ_to_string(model.τ)]
    except AttributeError:
        txt = [str(a) for a in print_τ_dist()]
    txt = '\n'.join(txt)
    plt.text(0,0,txt,fontsize=10)
    plt.setp(ax, frame_on=False, xticks=(), yticks=());

def plot_info(ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(0.01,0.01))

    lst = ['Cases',
          incidences.__str__(),
          '',
          'start_date {}'.format(τ_to_string(0)),
          'first NPI {}'.format(τ_to_string(first_NPI)),
          'official tau (last NPI) {}'.format(τ_to_string(last_NPI)),
           ''
          ]

    try:
        τ_posterior = sample[:,var_names.index('τ')].astype(int)
        τ_mean = τ_posterior.mean()
        τ_median = np.median(τ_posterior)
        τ_MAP = get_MAP()[var_names.index('τ')]
        lst.append('τ mean = {}'.format(τ_to_string(τ_mean)))
        lst.append('τ median = {}'.format(τ_to_string(τ_median)))
        lst.append('τ MAP = {}'.format(τ_to_string(τ_MAP)))
        lst.append('τ CI median 75%: {:.2f}'.format(calc_τ_CI_median(0.75)))
        lst.append('τ CI median 95%: {:.2f}'.format(calc_τ_CI_median(0.95)))
        confidence = 'P(τ > {}) = {:.2%}'.format(τ_to_string(last_NPI), (τ_posterior > last_NPI).mean())
        lst.append(confidence)
    except:
        None # there are models without τ of with fixed τ

    α1_posterior = sample[:,var_names.index('α1')]
    α2_posterior = sample[:,var_names.index('α2')]
    Δα_posterior = α2_posterior - α1_posterior
    lst.append('P(α2 > α1) = {:.2%}'.format((Δα_posterior > 0).mean()))

    lst.append('')
    lst.append('WAIC: {:.2f}'.format(calc_WAIC()))
    lst.append('DIC (using median): {:.2f}'.format(calc_DIC(calc_loglik_median)))
    lst.append('loglik of MAP: {:.2f}'.format(calc_loglikMAP()))
    lst.append('loglik of mean: {:.2f}'.format(calc_loglik_mean()))
    lst.append('loglik of median: {:.2f}'.format(calc_loglik_median()))

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

def plot_incidences(ax=None, color=blue, title=country_name ):
    if ax is None: fig, ax = plt.subplots()

    np.random.seed(10)
    num_simulations = 128
    daily_cases = []
    for _ in range(num_simulations):
        idx = np.random.choice(sample.shape[0])
        
        θ = sample[idx, :]

        y = model.generate_daily_cases(θ)
        daily_cases.append(y)

    t = np.arange(ndays)
    if title:
        plt.title(title)
    plt.plot(t, np.median(daily_cases,axis=0), color=color)
    plt.plot(t, incidences, '.', color=red)

    try:
        τ = model.τ #model with fixed tau
    except:
        try:    
            ind = var_names.index('τ') 
            τ_posterior = sample[:,ind].astype(int)
            τ = τ_posterior.mean()
            plt.axvline(τ, color=purple, linewidth=1, linestyle='--')
        except:
            None #not all models have τ. It's okey
    else:
        plt.axvline(τ, color=purple, linewidth=1, linestyle='--')

    ax.axvline(first_NPI, color='k', ls='--', alpha=0.25)
    ax.axvline(last_NPI, color='k', ls='--', alpha=0.25)
    
    days = list(range(0, ndays, round(ndays/10)))
    labels = [τ_to_string(d) for d in days]
    plt.xticks(days,labels,rotation=90);
    
    plt.ylabel('Daily cases')
    plt.xlabel('Day')
    sns.despine()
    return ax
