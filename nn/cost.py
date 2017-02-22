import theano, theano.tensor as T
from .module import Module

class DiagNormalNLL(Module):
    def __init__(self, **kwargs):
        super(DiagNormalNLL, self).__init__(**kwargs)

    def forward(self, (obs, mu, log_sigma)):
        nll = 0.5 * T.sum(log_sigma, axis=1) + \
              0.5 * T.sum(T.sqr((obs - mu) / (1e-6 + T.exp(log_sigma))), axis=1)
        return nll

class DiagNormalKLD(Module):
    def __init__(self, **kwargs):
        super(DiagNormalKLD, self).__init__(**kwargs)

    def forward(self, input):
        mu, logvar = input
        kld_element = logvar + 1. - T.sqr(mu) - T.exp(logvar) 
        kld = -0.5 * T.sum(kld_element, axis=1)

        return kld
