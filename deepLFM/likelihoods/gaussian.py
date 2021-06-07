import torch
import numpy as np
from . import likelihood

class Gaussian(likelihood.Likelihood):
    
    def log_cond_prob(self, output, latent, log_var):
        log_cond_prob = - 0.5 * np.log(2 * np.pi) - 0.5 * log_var - 0.5 * torch.square(output - latent) / torch.exp(log_var)
        return log_cond_prob

    def predict(self, latent):
        return latent