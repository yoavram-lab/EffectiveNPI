from model.normal_prior_model import NormalPriorModel
from scipy.stats import truncnorm, randint
import numpy as np
from numpy.random import uniform
import scipy.stats

class NormalPriorNegativeBinModel(NormalPriorModel):
    def __init__(self, country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2):
        super().__init__(country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2)
        self.var_names = ['k', 'Z', 'D', 'μ', 'β', 'α1', 'λ', 'α2', 'E0', 'Iu0','Δt0','τ']

    def _confirmed_cases_dist_logpmf(self, X, Y):
        p=self.k/(self.k+Y)
        return scipy.stats.nbinom(X, self.k, p)

    def _prior(self):
        params_bounds = self.params_bounds
        k = uniform(*params_bounds['k'])
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

        return k, Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ

    def log_prior(self, θ):
        k, Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ = θ
        τ = int(τ)
        Δt0 = int(Δt0)
        if self._in_bounds(k=k, Z=Z, D=D, μ=μ, β=β, α1=α1, λ=λ, α2=α2, E0=E0, Iu0=Iu0, Δt0=Δt0):
            return self.τ_prior.logpdf_or_pmf(τ)
        else:
            return -np.inf

    def log_likelihood(self, θ):
        k, Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ = θ
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

        S, E, Ir, Iu, R, Y = self._simulate(Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0, τ, ndays)
        p1 = 1/Td1
        p2 = 1/Td2
        Xsum = X.cumsum() 
        n = Y[1:] - Xsum[:-1] 
        n = np.maximum(1, n)
        p = ([p1] * τ + [p2] * (ndays - τ))[1:]

        p= k/(k+n[Δt0-1:] * p[Δt0-1:])
        loglik = scipy.stats.nbinom.logpmf(X[Δt0:], k, p)
        return loglik.mean()

    
