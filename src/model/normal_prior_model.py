from scipy.stats import truncnorm, randint
from numpy.random import uniform
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import scipy.stats


class NormalPriorModel:
    def __init__(self, country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2):
        self.country_name = country_name
        self.X = X
        self.start_date = start_date
        self.last_NPI_date = last_NPI_date
        self.get_first_NPI_date = first_NPI_date
        self.N = N
        self.params_bounds = params_bounds
        self.Td1 = Td1
        self.Td2 = Td2
        self.var_names = ['Z', 'D', 'μ', 'β', 'α1', 'λ', 'α2', 'E0', 'Iu0','Δt0','τ']
        self.τ_prior = self._get_τ_prior()


    def log_likelihood(self, θ):
        Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ = θ
        X = self.X
        Td1 = self.Td1
        Td2 = self.Td2

        τ = int(τ) # for explanation see https://github.com/dfm/emcee/issues/150
        Δt0 = int(Δt0)

        total_zeros = self.params_bounds['Δt0'][1]
        unrellevant_zeros = total_zeros - Δt0
        τ = τ - unrellevant_zeros
        X = X[unrellevant_zeros:]
        ndays = len(X)

        S, E, Ir, Iu, R, Y = self.simulate(Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ, ndays)
        p1 = 1/Td1
        p2 = 1/Td2
        Xsum = X.cumsum() 
        n = Y[1:] - Xsum[:-1] 
        n = np.maximum(1, n)
        p = ([p1] * τ + [p2] * (ndays - τ))[1:]

        loglik = scipy.stats.poisson.logpmf(X[Δt0:], n[Δt0-1:] * p[Δt0-1:])
        return loglik.mean()


    def log_prior(self, θ):
        Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ = θ
        τ = int(τ)
        Δt0 = int(Δt0)
        if self._in_bounds(Z=Z, D=D, μ=μ, β=β, α1=α1, λ=λ, α2=α2, E0=E0, Iu0=Iu0, Δt0=Δt0):
            return self.τ_prior.logpdf_or_pmf(τ)
        else:
            return -np.inf


    def guess_one(self):
        while True:
            res = self._prior()
            if np.isfinite(self.log_likelihood(res)):
                return res

    def _get_τ_prior(self):
        ndays = len(self.X)
        last_τ = (self.last_NPI_date - self.start_date).days
        first_τ = (self.get_first_NPI_date - self.start_date).days

        lower = self.params_bounds['Δt0'][1]
        upper = ndays - 2
        μ = (last_τ + first_τ) / 2
        σ = (last_τ - first_τ) / 2
        σ = 5 if σ<5 else σ 
        # μ = last_τ
        # σ = 5
        res = truncnorm((lower - μ) / σ, (upper - μ) / σ, loc=μ, scale=σ)
        res.logpdf_or_pmf = res.logpdf
        return res

    def _prior(self):
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


    def _in_bounds(self, **params):
        bounds = [self.params_bounds[p] for p in params]
        for val,(lower,higher) in zip(params.values(),bounds):
            if not lower<=val<=higher:
                return False
        return True

    def __ode(self, v, t, Z, D, α, β, μ):
        N = self.N
        S, E, Ir, Iu, Y = v
        return [
            -β * S * Ir / N - μ * β * S * Iu / N,
            +β * S * Ir / N + μ * β * S * Iu / N - E / Z,
             α * E / Z - Ir / D,
            (1-α) * E / Z - Iu / D,
             α * E / Z # accumulated reported infections
        ]

    def _simulate_one(self, Z, D, μ, β, α, y0, ndays):
        sol = odeint(self.__ode, y0, np.arange(ndays), args=(Z, D, α, β, μ))
        S, E, Ir, Iu, Y = sol.T
        return S, E, Ir, Iu, Y

    def simulate(self, Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ, ndays):
        N = self.N
        τ = int(τ)
        Ir0 = 0
        S0 = N - E0 - Ir0 - Iu0
        init = [S0, E0, Ir0, Iu0, Ir0]
        sol1 = self._simulate_one(Z, D, μ, β, α1, init, τ)
        sol1 = np.array(sol1)
        sol2 = self._simulate_one(Z, D, μ, λ*β, α2, sol1[:, -1], ndays - τ)

        S, E, Ir, Iu, Y = np.concatenate((sol1, sol2), axis=1)
        R = N - (S + E + Ir + Iu)
        return S, E, Ir, Iu, R, Y

    def generate_daily_cases(self, θ):
        Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ = θ
        Δt0 = int(Δt0)
        τ = int(τ)
        total_zeros = self.params_bounds['Δt0'][1]
        unrellevant_zeros = total_zeros - Δt0
        τ = τ - unrellevant_zeros
        θ = Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ

        S, E, Ir, Iu, R, Y = self.simulate(*θ,len(self.X)-unrellevant_zeros)
        p1 = 1/self.Td1
        p2 = 1/self.Td2 
        C = np.zeros_like(Y)
        for t in range(1, len(C)):
            p = p1 if t<τ else p2
            n = Y[t] - C[:t].sum()
            n = max(0,n)
            C[t] = np.random.poisson(n * p)     

        return [0]*unrellevant_zeros + list(C)


