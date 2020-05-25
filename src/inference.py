import urllib.request
import os
from datetime import timedelta, datetime
import sys
from multiprocessing import Pool # TODO maybe use ProcessPoolExecutor?

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
import argparse
from shutil import copyfile, copytree, rmtree
from enum import IntEnum
from model.normal_prior_model import NormalPriorModel
from model.uniform_prior_model import UniformPriorModel
from model.no_tau_model import NoTauModel
from model.fixed_tau_model import FixedTauModel

np.random.seed(10)    
now = datetime.now().strftime('%Y-%m-%d')

Td1 = 9
Td2 = 6
seed_max = 3000

def get_model_class(model_type):
    if model_type == 1:
        return UniformPriorModel
    elif model_type == 2:
        return NormalPriorModel
    elif model_type == 3:
        return NoTauModel
    elif model_type == 4:
        return FixedTauModel
    else:
        return None

params_bounds = {
    'Z' : (2, 5),
    'D' : (2, 5),
    'μ' : (0.2, 1),
    'β' : (0.8, 1.5),
    'α1' : (0.02, 1),
    'λ'  : (0, 1),
    'α2' : (0.02, 1),
    'E0' : (0, seed_max),
    'Iu0' : (0, seed_max),
    'Δt0' : (1,5) #how much zeros before the first incident
}

def get_first_NPI_date(country_name):
    country_name = 'United Kingdom' if country_name == 'United_Kingdom' else country_name
    df = pd.read_csv('../data/NPI_dates.csv',parse_dates=['First','Last'])
    return df[df['Country']==country_name]['First'].iloc[0].to_pydatetime()

def get_last_NPI_date(country_name):
    country_name = 'United Kingdom' if country_name == 'United_Kingdom' else country_name
    df = pd.read_csv('../data/NPI_dates.csv',parse_dates=['First','Last'])
    return df[df['Country']==country_name]['Last'].iloc[0].to_pydatetime()


def find_start_day(cases_and_dates):
    #looks for the last 0 0 sequence pattern
    arr = np.array(cases_and_dates['cases'])
    ind = len(arr)-list(zip(arr, arr[1:]))[::-1].index((0,0))
    # return cases_and_dates.iloc[ind-1]['date']
    zeros = params_bounds['Δt0'][1]
    return cases_and_dates.iloc[ind-zeros]['date'].to_pydatetime()


def τ_to_string(τ, start_date):
    return (pd.to_datetime(start_date) + timedelta(days=τ)).strftime('%b %d')

def log_posterior(θ, model):
    logpri = model.log_prior(θ)  
    if np.isinf(logpri): 
        return logpri   

    assert not np.isnan(logpri), (logpri, θ)
    loglik = model.log_likelihood(θ)
    assert not np.isnan(loglik), (loglik, θ)
    logpost = logpri + loglik
    return logpost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('country_name')
    parser.add_argument('-s', '--steps',type=int,help='you can provide number of iteration steps, othewise the default is taken')
    parser.add_argument('-w', '--walkers',type=int,help='you can provide number of walkers, othewise the default is taken')
    parser.add_argument('-c', '--cores',type=int,help='by default 1 core')
    parser.add_argument('-d', '--ver-desc',type=str,help='short description of the version - will be part of the dir name')
    parser.add_argument('-m', '--tau-model',type=int,help='1 - uniform prior, 2 (default) - normal prior, 3 - no tau, 4 - fixed tau (on lockdown date)')
    parser.add_argument('--up-to-date',type=str, help='you can provide the last date (including). The format yyyy-mm-dd')
    args = parser.parse_args()
    country_name = args.country_name
    cores = args.cores
    ver_desc = '-'+args.ver_desc if args.ver_desc else ''
    model_type = args.tau_model if args.tau_model else 2
    ModelClass = get_model_class(model_type)
    if not os.path.exists('../data'):
        os.mkdir('../data')

    if country_name=='Wuhan':
        df = pd.read_csv('../data/Incidence.csv')
        df['date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['cases'] = df[country_name]
        df = df[::-1] # TODO why?
        N = pd.read_csv('../data/pop.csv', index_col='City').loc[country_name].values[0]
    else:
        # url = 'https://github.com/ImperialCollegeLondon/covid19model/raw/v1.0/data/COVID-19-up-to-date.csv'
        url = 'https://raw.githubusercontent.com/ImperialCollegeLondon/covid19model/master/data/COVID-19-up-to-date.csv'
        fname = '../data/COVID-19-up-to-date.csv'
        if not os.path.exists(fname):
            urllib.request.urlretrieve(url, fname)
        df = pd.read_csv(fname, encoding='iso-8859-1')
        df['date'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
        df = df[df['countriesAndTerritories'] == country_name]
        N = df.iloc[0]['popData2018']

    cases_and_dates = df.iloc[::-1][['cases','date']]
    if args.up_to_date:
        cases_and_dates = cases_and_dates[cases_and_dates['date']<=args.up_to_date]
    start_date = find_start_day(cases_and_dates)
    X = np.array(cases_and_dates[cases_and_dates['date'] >= start_date]['cases'])
    model = ModelClass(country_name, X, start_date, N, get_first_NPI_date(country_name), get_last_NPI_date(country_name), params_bounds, Td1, Td2)

    ndim = len(model.var_names)
    nwalkers = 50
    if args.walkers:
        nwalkers = args.walkers
    nsteps = 75000
    if args.steps:
        nsteps = args.steps

    guesses = np.array([model.guess_one() for _ in range(nwalkers)])
    if cores and cores!=1:
        with Pool(cores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[model], pool=pool)
            sampler.run_mcmc(guesses, nsteps, progress=True);
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[model])
        sampler.run_mcmc(guesses, nsteps, progress=True);

    params = [nsteps, ndim, int(N), Td1, Td2, int(model_type)]

    output_folder = '../output-tmp/{}{}/inference'.format(now,ver_desc) #tmp folder is not for production
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename =  '{}.npz'.format(country_name)
    filename = os.path.join(output_folder, filename)
    print('filling logliks')
    priors = [model.log_prior(s) for s in sampler.chain.reshape(-1, ndim)]
    logliks = sampler.lnprobability.reshape(-1) - priors
    print(filename)
    
    np.savez_compressed(
        filename,
        chain=sampler.chain,
        lnprobability=sampler.lnprobability, #log_posteriors
        logliks=logliks,
        incidences=X, # TODO maybe save as X=X
        # autocorr=autocorr,
        params=params, 
        var_names=model.var_names,
        start_date=str(start_date)
    )

    # we persist the source code of the current files for each experiment
    if not os.path.exists(os.path.join(output_folder,'src')):
        os.mkdir(os.path.join(output_folder,'src'))
    copyfile(sys.argv[0], os.path.join(output_folder,'src', sys.argv[0])) 
    model_path = os.path.join(output_folder,'src', 'model')
    if os.path.exists(model_path): 
        rmtree(model_path)
    copytree('model', model_path)
