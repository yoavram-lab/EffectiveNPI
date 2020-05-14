from model.normal_prior_model import NormalPriorModel
import numpy as np

class FixedTauModel(NormalPriorModel):
    def __init__(self, country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2):
        super().__init__(country_name, X, start_date, N, last_NPI_date, first_NPI_date, params_bounds, Td1, Td2)
        self.var_names = ['Z', 'D', 'μ', 'β', 'α1', 'λ', 'α2', 'E0', 'Iu0','Δt0'] #no 
        self.τ = (last_NPI_date - start_date).days
    
    def _prior(self):
        res = super()._prior()
        return res[:-1] #same but without τ
    
    def log_prior(self, θ):
        Z, D, μ, β, α1, λ, α2, E0, Iu0, Δt0 = θ
        Δt0 = int(Δt0)
        if self._in_bounds(Z=Z, D=D, μ=μ, β=β, α1=α1, λ=λ, α2=α2, E0=E0, Iu0=Iu0, Δt0=Δt0):
            return 0
        else:
            return -np.inf

    def log_likelihood(self, θ):
        return super().log_likelihood((*θ,self.τ))
    
    def generate_daily_cases(self, θ):
        return super().generate_daily_cases((*θ,self.τ))

    