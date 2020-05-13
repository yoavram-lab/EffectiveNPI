from model.normal_prior_model import NormalPriorModel
from scipy.stats import truncnorm, randint
import numpy as np
import scipy.stats

class NoTauModel(NormalPriorModel):
    def __init__(self, country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2):
        super().__init__(country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2)
        self.var_names = ['Z', 'D', 'μ', 'β', 'α1', 'λ', 'α2', 'E0', 'Iu0','Δt0']


    def log_prior(self, θ):
        Z, D, μ, β, α1, E0, Iu0, Δt0 = θ
        τ = int(τ)
        Δt0 = int(Δt0)
        if self.__in_bounds(Z=Z, D=D, μ=μ, β=β, α1=α1, E0=E0, Iu0=Iu0, Δt0=Δt0):
            return 0
        else:
            return -np.inf


    def log_likelihood(self, θ):
        Z, D, μ, β, α1, E0, Iu0, Δt0 = θ
        X = self.X
        Td1 = self.Td1
        Δt0 = int(Δt0)

        total_zeros = self.params_bounds['Δt0'][1]
        unrellevant_zeros = total_zeros - Δt0
        X = X[unrellevant_zeros:]
        ndays = len(X)

        S, E, Ir, Iu, R, Y = self.simulate(Z, D, μ, β, α1, E0, Iu0, Δt0, ndays)
        p1 = 1/Td1
        Xsum = X.cumsum() 
        n = Y[1:] - Xsum[:-1] 
        n = np.maximum(0, n)
        p = ([p1] * ndays)[1:]

        loglik = scipy.stats.poisson.logpmf(X[Δt0:], n[Δt0-1:] * p[Δt0-1:])
        return loglik.mean()

    def simulate(self, Z, D, μ, β, α1, E0, Iu0, Δt0, ndays):
        N = self.N
        Ir0 = 0
        S0 = N - E0 - Ir0 - Iu0
        init = [S0, E0, Ir0, Iu0, Ir0]
        S, E, Ir, Iu, Y = self.__simulate_one(Z, D, μ, β, α1, init, ndays)
        R = N - (S + E + Ir + Iu)
        return S, E, Ir, Iu, R, Y