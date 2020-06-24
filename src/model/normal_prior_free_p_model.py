from model.normal_prior_model import NormalPriorModel
from scipy.stats import truncnorm, randint
import numpy as np
from numpy.random import uniform


class NormalPriorFreepModel(NormalPriorModel):
    def __init__(self, country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2):
        super().__init__(country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2)
        self.var_names = self.var_names + ['Td1','Td2']
        self.Td1 = None
        self.Td2 = None

    def log_prior(self, θ):
        Td1, Td2 = θ[-2:]
        res = super().log_prior(θ[:-2])
        if self._in_bounds(Td1=Td1, Td2=Td2):
            return res
        else:
            return -np.inf

    def log_likelihood(self, θ):
        self.Td1, self.Td2 = θ[-2:]
        return super().log_likelihood(θ[:-2])

    def _prior(self):
        res = super()._prior()
        Td1 = uniform(*self.params_bounds['Td1'])
        Td2 = uniform(*self.params_bounds['Td2'])
        return (*res,Td1,Td2)

    def generate_daily_cases(self, θ):
        self.Td1, self.Td2 = θ[-2:]
        return super().generate_daily_cases(θ[:-2])