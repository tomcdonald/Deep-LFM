import abc

class Likelihood:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_cond_prob(self, output, latent):
        """
        Likelihoods must compute log[p(y|F)], where y are outputs & F are latent values.
        """
        raise NotImplementedError("Implemented by subclass.")

    @abc.abstractmethod
    def predict(self, latent):
        raise NotImplementedError("Implemented by subclass.")