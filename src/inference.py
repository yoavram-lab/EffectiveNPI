import urllib.request
import os
from datetime import timedelta, datetime
import sys
from multiprocessing import Pool # TODO maybe use ProcessPoolExecutor?

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
import pandas as pd
from scipy.integrate import odeint
import scipy.stats
from scipy.stats import truncnorm, randint
import emcee
import argparse
from shutil import copyfile
from enum import IntEnum

class NormalPriorModel:
    def __init__(self, country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds):
        self.country_name = country_name
        self.X = X
        self.start_date = start_date
        self.last_NPI_date = last_NPI_date
        self.get_first_NPI_date = first_NPI_date
        self.N = N
        self.params_bounds = params_bounds
        self.var_names = ['Z', 'D', 'μ', 'β', 'α1', 'λ', 'α2', 'E0', 'Iu0','Δt0','τ']
        self.τ_prior = self.__get_τ_prior()


    def log_likelihood(self, θ):
        Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ = θ
        X = self.X
        N = self.N

        τ = int(τ) # for explanation see https://github.com/dfm/emcee/issues/150
        Δt0 = int(Δt0)

        total_zeros = self.params_bounds['Δt0'][1]
        unrellevant_zeros = total_zeros - Δt0
        τ = τ - unrellevant_zeros
        X = X[unrellevant_zeros:]
        ndays = len(X)

        S, E, Ir, Iu, R, Y = simulate(Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ, ndays, N)
        p1 = 1/Td1
        p2 = 1/Td2
        Xsum = X.cumsum() 
        n = Y[1:] - Xsum[:-1] 
        n = np.maximum(0, n)
        p = ([p1] * τ + [p2] * (ndays - τ))[1:]

        loglik = scipy.stats.poisson.logpmf(X[Δt0:], n[Δt0-1:] * p[Δt0-1:])
        return loglik.mean()


    def log_prior(self, θ):
        Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ = θ
        τ = int(τ)
        Δt0 = int(Δt0)
        if self.__in_bounds(Z=Z, D=D, μ=μ, β=β, α1=α1, λ=λ, α2=α2, E0=E0, Iu0=Iu0, Δt0=Δt0):
            # if τ_model==TauModel.uniform_prior:
            #     return τ_prior.logpmf(τ)
            return self.τ_prior.logpdf(τ)
        else:
            return -np.inf


    def guess_one(self):
        while True:
            res = self.__prior()
            if np.isfinite(self.log_likelihood(res)):
                return res

    def __get_τ_prior(self):
        ndays = len(self.X)
        # if τ_model==TauModel.uniform_prior:
        #     return randint(params_bounds['Δt0'][1], ndays) #[including,not-including]

        last_τ = (self.last_NPI_date - self.start_date).days
        first_τ = (self.get_first_NPI_date - self.start_date).days

        lower = self.params_bounds['Δt0'][1]
        upper = ndays - 2
        μ = (last_τ + first_τ) / 2
        σ = (last_τ - first_τ) / 2
        σ = 5 if σ<5 else σ 
        # μ = last_τ
        # σ = 5
        return truncnorm(
            (lower - μ) / σ, (upper - μ) / σ, loc=μ, scale=σ)

    def __prior(self):
        params_bounds = self.params_bounds
        Z = uniform(*params_bounds['Z'])
        D = uniform(*params_bounds['D'])
        μ = uniform(*params_bounds['μ'])
        β = uniform(*params_bounds['β'])
        α1 = uniform(*params_bounds['α1'])
        λ = uniform(*params_bounds['λ'])
        α2 = uniform(*params_bounds['α2'])
        E0 = uniform(*params_bounds['E0'])
        Iu0 = uniform(*params_bounds['Iu0'])
        Δt0 = randint(params_bounds['Δt0'][0],params_bounds['Δt0'][1]+1).rvs() #+1 because randint don't include the upper value
        τ = self.τ_prior.rvs()

        return Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ


    def __in_bounds(self, **params):
        bounds = [self.params_bounds[p] for p in params]
        for val,(lower,higher) in zip(params.values(),bounds):
            if not lower<=val<=higher:
                return False
        return True

class TauModel(IntEnum):
    uniform_prior = 1
    normal_prior = 2

np.random.seed(10)    
now = datetime.now().strftime('%Y-%m-%d')

Td1 = 9
Td2 = 6
seed_max = 3000


def get_first_NPI_date(country_name):
    country_name = 'United Kingdom' if country_name == 'United_Kingdom' else country_name
    df = pd.read_csv('../data/NPI_dates.csv',parse_dates=['First','Last'])
    return df[df['Country']==country_name]['First'].iloc[0].to_pydatetime()

def get_last_NPI_date(country_name):
    country_name = 'United Kingdom' if country_name == 'United_Kingdom' else country_name
    df = pd.read_csv('../data/NPI_dates.csv',parse_dates=['First','Last'])
    return df[df['Country']==country_name]['Last'].iloc[0].to_pydatetime()


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


def find_start_day(cases_and_dates):
    #looks for the last 0 0 sequence pattern
    arr = np.array(cases_and_dates['cases'])
    ind = len(arr)-list(zip(arr, arr[1:]))[::-1].index((0,0))
    # return cases_and_dates.iloc[ind-1]['date']
    zeros = params_bounds['Δt0'][1]
    return cases_and_dates.iloc[ind-zeros]['date'].to_pydatetime()


def τ_to_string(τ, start_date):
    return (pd.to_datetime(start_date) + timedelta(days=τ)).strftime('%b %d')


def ode(v, t, Z, D, α, β, μ, N):
    S, E, Ir, Iu, Y = v
    return [
        -β * S * Ir / N - μ * β * S * Iu / N,
        +β * S * Ir / N + μ * β * S * Iu / N - E / Z,
         α * E / Z - Ir / D,
        (1-α) * E / Z - Iu / D,
         α * E / Z # accumulated reported infections
    ]


def simulate_one(Z, D, μ, β, α, y0, ndays, N):
    sol = odeint(ode, y0, np.arange(ndays), args=(Z, D, α, β, μ, N))
    S, E, Ir, Iu, Y = sol.T
    return S, E, Ir, Iu, Y


def simulate(Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ, ndays, N):
    τ = int(τ)
    Ir0 = 0
    S0 = N - E0 - Ir0 - Iu0
    init = [S0, E0, Ir0, Iu0, Ir0]
    sol1 = simulate_one(Z, D, μ, β, α1, init, τ, N)
    sol1 = np.array(sol1)
    sol2 = simulate_one(Z, D, μ, λ*β, α2, sol1[:, -1], ndays - τ, N)
    
    S, E, Ir, Iu, Y = np.concatenate((sol1, sol2), axis=1)
    R = N - (S + E + Ir + Iu)
    return S, E, Ir, Iu, R, Y


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
    parser.add_argument('-d', '--ver_desc',type=str,help='short description of the version - will be part of the dir name')
    parser.add_argument('-m', '--tau_model',type=int,help='1 - uniform prior, 2 - wide prior')
    args = parser.parse_args()
    country_name = args.country_name
    cores = args.cores
    ver_desc = '-'+args.ver_desc if args.ver_desc else ''
    τ_model = TauModel(args.tau_model) if args.tau_model else TauModel.uniform_prior

    if not os.path.exists('../data'):
        os.mkdir('../data')

    if country_name=='Wuhan':
        df = pd.read_csv('../data/Incidence.csv')
        df['date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['cases'] = df[country_name]
        df = df[::-1] # TODO why?
        N = pd.read_csv('../data/pop.csv', index_col='City').loc[country_name].values[0]
    else:
        url = 'https://github.com/ImperialCollegeLondon/covid19model/raw/v1.0/data/COVID-19-up-to-date.csv'
        fname = '../data/COVID-19-up-to-date.csv'
        if not os.path.exists(fname):
            urllib.request.urlretrieve(url, fname)
        df = pd.read_csv(fname, encoding='iso-8859-1')
        df['date'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
        df = df[df['countriesAndTerritories'] == country_name]
        N = df.iloc[0]['popData2018']

    cases_and_dates = df.iloc[::-1][['cases','date']]
    start_date = find_start_day(cases_and_dates)
    X = np.array(cases_and_dates[cases_and_dates['date'] >= start_date]['cases'])
    model = NormalPriorModel(country_name, X, start_date, N, get_first_NPI_date(country_name), get_last_NPI_date(country_name), params_bounds)

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

    params = [nsteps, ndim, int(N), Td1, Td2, int(τ_model)]

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
    copyfile(sys.argv[0], os.path.join(output_folder, sys.argv[0])) # we persist the source code of the current file for each experiment

