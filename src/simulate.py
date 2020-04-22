# TODO change filename to inference?
# TODO remove any function that is not used here?
# TODO rename to "model.py"?
import urllib.request
import os
from datetime import timedelta, datetime
import sys
from multiprocessing import Pool # TODO maybe use ProcessPoolExecutor?

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform, randint
import pandas as pd
from scipy.integrate import odeint
import scipy.stats
import emcee


Td1 = 9
Td2 = 6
seed_max = 3000


def find_start_day(cases_and_dates):
    last_date = None
    last_is_not_zero = False
    for r in cases_and_dates.iterrows():
        curr_cases = r[1]['cases']
        if last_is_not_zero and curr_cases != 0:
			break
        last_is_not_zero = False if curr_cases == 0 else True
        if curr_cases == 0:
            last_date = r[1]['date']
            last_is_not_zero = False
        else:
            last_is_not_zero = True
    return last_date


def prior():
    Z = uniform(2, 5)
    D = uniform(2, 5)
    μ = uniform(0.2, 1)
    β = uniform(0.6, 1.5)
    α1 = uniform(0.02, 0.8)
    λ = uniform(0, 1)
    α2 = uniform(0.02, 0.8)
    E0, Iu0 = uniform(0, seed_max, size=2)
    τ = randint(2, ndays-2)
    return Z, D, μ, β, α1, λ, α2, E0, Iu0, τ


def ode(y, t, Z, D, α, β, μ):
    S, E, Ir, Iu = y
    return [
        -β * S * Ir / N - μ * β * S * Iu / N,
        +β * S * Ir / N + μ * β * S * Iu / N - E / Z,
         α * E / Z - Ir / D,
        (1-α) * E / Z - Iu / D
    ]


def simulate_one(Z, D, μ, β, α, y0, ndays):
        sol = odeint(ode, y0, np.arange(ndays), args=(Z, D, μ, β, α))
        S, E, Ir, Iu = sol.T
        return S, E, Ir, Iu


def simulate(Z, D, μ, β, α1, λ, α2, E0, Iu0, τ):
    τ = int(τ)
    Ir0 = 0
    S0 = N - E0 - Ir0 - Iu0
    y0 = [S0, E0, Ir0, Iu0]
    y1 = simulate_one(Z, D, μ, β, α1, y0, τ)
    y2 = simulate_one(Z, D, μ, λ*β, α2, np.array(y1)[:,-1], ndays - τ)
    
    S, E, Ir, Iu = np.concatenate((y1, y2), axis=1)
    R = N - (S + E + Ir + Iu)
    return S, E, Ir, Iu, R


def log_prior(θ):
    Z, D, μ, β1, α1, λ, α2, E0, Iu0,τ = θ
    if (2 <= Z <=5) and (2 <= D <= 5) and (0.2 <= μ <= 1) and (0.8 <= β1 <= 1.5) and (0.02 <= α1 <= 1) and (0 <= λ <= 1) and (0.02 <= α2 <= 1) and (0 < E0 < seed_max) and (0 < Iu0 < seed_max) and (1<=τ<=ndays-1):
            return 0 # flat prior
    else:
        return -np.inf


def log_likelihood(θ, X):
    Z, D, μ, β, α1, λ, α2, E0, Iu0, τ = θ
    τ = int(τ)
    
    # TODO the following is implemented in log_prior and should not be here
    case = (2 <= Z <=5) and (2 <= D <= 5) and (0.2 <= μ <= 1) and (0.8 <= β <= 1.5) and (0.02 <= α1 <= 1) and (0 <= λ <= 1) and (0.02 <= α2 <= 1) and (0 < E0 < seed_max) and (0 < Iu0 < seed_max) and (1<=τ<=ndays-1)
    if not case:
        return -np.inf
    
    S, E, Ir, Iu, R = simulate(*θ)
    p1 = 1/Td1
    p2 = 1/Td2
    Y1 = α1 * E[:τ] / Z
    Y2 = α2 * E[τ:ndays] / Z
    Y = np.concatenate((Y1, Y2))
    Ysum = Y.cumsum()
    Xsum = X.cumsum() 
    n = Ysum[1:] - Xsum[:-1] 
    n = np.maximum(0, n)
    p = ([p1] * τ + [p2] * (ndays - τ))[1:]
    loglik = scipy.stats.poisson.logpmf(X[1:], n * p)
    return loglik.mean()


def log_posterior(θ, X):
    logpri = log_prior(θ)     
    assert not np.isnan(logpri), (logpri, θ)
    loglik = log_likelihood(θ, X)
    assert not np.isnan(loglik), (loglik, θ)
    logpost = logpri + loglik
    return logpost


if __name__ == '__main__':
	country = sys.argv[1] # TODO if there is no argv[1], print usage message

	if country=='Wuhan':
	    df = pd.read_csv('../data/Incidence.csv')
	    df['date'] = pd.to_datetime(df['Date'], dayfirst=True)
	    df['cases'] = df['Wuhan']
	    df = df[::-1] # TODO why?
	    N = pd.read_csv('../data/pop.csv', index_col='City').loc['Wuhan'].values[0]
	else:
	    url = 'https://github.com/ImperialCollegeLondon/covid19model/raw/v1.0/data/COVID-19-up-to-date.csv'
	    fname = '../data/COVID-19-up-to-date.csv'
	    if not os.path.exists(fname):
	        urllib.request.urlretrieve(url, fname)
	    df = pd.read_csv(fname, encoding='iso-8859-1')
	    df['date'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
	    df = df[df['countriesAndTerritories']==country]
	    N = df.iloc[0]['popData2018']

	cases_and_dates = df.iloc[::-1][['cases','date']]
	start_date = find_start_day(cases_and_dates)
	X = np.array(cases_and_dates[cases_and_dates['date']>=start_date]['cases'])
	ndays = len(X)

	var_names = ['Z', 'D', 'μ', 'β', 'α1', 'λ', 'α2', 'E0', 'Iu0','τ']
	ndim = len(var_names)
	nwalkers = 50
	nsteps = 500 * 3 * 50
	nsteps = 10 * 50 # TODO remove this line or the former
	# nburn = nsteps // 2
	guesses = np.array([prior() for _ in range(nwalkers)])

	np.random.seed(10)
	# with Pool() as pool: # TODO maybe use ProcessPoolExecutor?
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[X])
	sampler.run_mcmc(guesses, nsteps, progress=True);

	params = [nsteps, ndim, int(N), Td1, Td2]
	np.savez_compressed(
		# TODO pwd is 'src', save to ../output/<country>/<date>_<time>_inference.npz
		'./outputs/test/data/{}.npz'.format(country), 
		chain=sampler.chain, # TODO what about the burnin?
		incidences=X, # TODO maybe save as X=X
		params=params, 
		var_names=var_names,
		start_date=str(start_date)
	)

