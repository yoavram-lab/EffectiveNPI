from model.normal_prior_model import NormalPriorModel
from scipy.stats import truncnorm, randint

class UniformPriorModel(NormalPriorModel):

    def _get_τ_prior(self):
        ndays = len(self.X)
        res = randint(self.params_bounds['Δt0'][1], ndays) #[including,not-including]
        res.logpdf_or_pmf = res.logpmf
        return res