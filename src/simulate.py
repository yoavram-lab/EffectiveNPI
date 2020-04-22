import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request
import os
from scipy.integrate import odeint
from scipy.stats import poisson
from scipy import stats
from datetime import timedelta, datetime
import sys

country = sys.argv[1]

def find_start_day(cases_and_dates):
    last_date = None
    last_is_not_zero = False
    for r in cases_and_dates.iterrows():
        curr_cases = r[1]['cases']
        if last_is_not_zero and curr_cases != 0:
                break
        last_is_not_zero= False if curr_cases == 0 else True
        if curr_cases == 0:
            last_date = r[1]['date']
            last_is_not_zero = False
        else:
            last_is_not_zero = True
    return last_date
def tau_to_string(tau):
    return (pd.to_datetime(start_date)+timedelta(days=tau)).strftime('%B %d')

url = 'https://github.com/ImperialCollegeLondon/covid19model/raw/v1.0/data/COVID-19-up-to-date.csv'
fname = '../data/COVID-19-up-to-date'
if not os.path.exists(fname):
    urllib.request.urlretrieve(url, fname)
df = pd.read_csv(fname, encoding='iso-8859-1')
df['date'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
df = df[df['countriesAndTerritories']==country]
N = df.iloc[0]['popData2018']
Td1 = 9
Td2 = 6
seed_max = 3000

cases_and_dates = df.iloc[::-1][['cases','date']]
start_date = find_start_day(cases_and_dates)
incidences = np.array(cases_and_dates[cases_and_dates['date']>=start_date]['cases'])
ndays = len(incidences)




from scipy.integrate import odeint
from scipy.stats import poisson, gamma, geom
from numpy.random import uniform, randint

def Re(α, β, μ, D):
    return α * β * D + (1-α) * μ * β * D

# parameters
def prior():
    Z = uniform(2, 5)
    D = uniform(2, 5)
    μ = uniform(0.2, 1)
    β1 = uniform(0.6, 1.5)
    α1 = uniform(0.02, 0.8)
    λ = uniform(0, 1)
    α2 = uniform(0.02, 0.8)
    E0, Iu0 = uniform(0, seed_max, size=2)
    tau = randint(2, ndays-2)
    return Z, D, μ, β1, α1, λ, α2, E0, Iu0, tau

def ode(y, t, Z, D, α, β, μ):
    S, E, Ir, Iu = y
    return [
        -β * S * Ir / N - μ * β * S * Iu / N,
        +β * S * Ir / N + μ * β * S * Iu / N - E / Z,
         α * E / Z - Ir / D,
        (1-α) * E / Z - Iu / D
    ]
def simulate(Z, D, μ, β1, α1, λ, α2, E0, Iu0,tau):
    tau=int(tau)
    def simulate_one(Z, D, μ, β, α, y0, ndays):
        sol = odeint(ode, y0, np.arange(ndays), args=(Z, D, μ, β, α))
        S, E, Ir, Iu = sol.T
        return S, E, Ir, Iu
    Ir0 = 0
    S0 = N - E0 - Ir0 - Iu0
    y0 = [S0, E0, Ir0, Iu0]
    y1 = simulate_one(Z, D, μ, β1, α1, y0, tau)
    y2 = simulate_one(Z, D, μ, λ*β1, α2, np.array(y1)[:,-1], ndays-tau)
    
    S, E, Ir, Iu = np.concatenate((y1,y2),axis=1)
    R = N - (S + E + Ir + Iu)
    return S, E, Ir, Iu, R



import emcee
import scipy.stats
def log_prior(θ):
    Z, D, μ, β1, α1, λ, α2, E0, Iu0,tau = θ
    if (2 <= Z <=5) and (2 <= D <= 5) and (0.2 <= μ <= 1) and (0.8 <= β1 <= 1.5) and (0.02 <= α1 <= 1) and (0 <= λ <= 1) and (0.02 <= α2 <= 1) and (0 < E0 < seed_max) and (0 < Iu0 < seed_max) and (1<=tau<=ndays-1):
            return 0 # flat prior
    else:
        return -np.inf
    
def log_likelihood(θ, C):
    Z, D, μ, β1, α1, λ, α2, E0, Iu0, tau = θ
    tau=int(tau)
    
    case = (2 <= Z <=5) and (2 <= D <= 5) and (0.2 <= μ <= 1) and (0.8 <= β1 <= 1.5) and (0.02 <= α1 <= 1) and (0 <= λ <= 1) and (0.02 <= α2 <= 1) and (0 < E0 < seed_max) and (0 < Iu0 < seed_max) and (1<=tau<=ndays-1)
    if not case:
        return -np.inf
    
    S, E, Ir, Iu, R = simulate(*θ)
    p1 = 1/Td1 # Td = E[X], X~Geom(p)   
    p2 = 1/Td2
    I1 = α1 *E[:tau]/Z
    I2 = α2 *E[tau:ndays]/Z
    I = np.concatenate((I1,I2))
    Isum = I.cumsum() # total infections
    Csum = C.cumsum() # total reports
    n = Isum[1:] - Csum[:-1] # total infections yet to be reported
    n = np.maximum(0, n)
    p = ([p1]*tau+[p2]*(ndays-tau))[1:]
    loglik = scipy.stats.poisson.logpmf(C[1:], n*p) # X ~ Poi(n), Y|X=n ~ Bin(n,p) -> X ~ Poi(np)
    return loglik.mean()

def log_posterior(θ, incidence):
    logpri = log_prior(θ)     
    assert not np.isnan(logpri), (logpri, θ)
    loglik = log_likelihood(θ, incidence)
    assert not np.isnan(loglik), (loglik, θ)
    logpost = logpri + loglik
    return logpost

var_names = ['Z', 'D', 'μ', 'β1', 'α1', 'λ', 'α2', 'E0', 'Iu0','tau']
ndim, nwalkers = len(var_names), 50
nsteps = 500*3*50
nsteps = 10*50
# nburn = nsteps // 2
guesses = np.array([prior() for _ in range(nwalkers)])

from multiprocessing import Pool

np.random.seed(10)
# with Pool() as pool:
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[incidences])
sampler.run_mcmc(guesses, nsteps, progress=True);

from datetime import datetime
params = [nsteps,ndim,int(N),Td1,Td2]
np.savez_compressed('./outputs/test/data/{}.npz'.format(country), chain=sampler.chain, incidences=incidences, params=params, var_names=var_names,
                   start_date=str(start_date))

