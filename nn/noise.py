import numpy as np
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

from .module import Module

class Dropout(Module):
    def __init__(self, drop_prob, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        assert drop_prob >= 0 and drop_prob < 1, 'dropout rate should be in [0, 1)'
        self.drop_prob = drop_prob
        
        self.trng = RandomStreams(np.random.randint(1, 2147462579))

    def forward(self, input):
        # Using theano constant to prevent upcasting
        retain_prob = T.constant(1.) - self.drop_prob

        # Random binomial mask
        mask = self.trng.binomial(input.shape, p=retain_prob, dtype=input.dtype)

        # Rescale output during training so that no rescale needed during evaluation
        dropped = input * mask / retain_prob

        # Set output using ifelse
        self.output = ifelse(self.train * self.drop_prob, dropped, input)

        return self.output

    def infer(self, input):
        return input

class ReparamDiagGaussian(Module):
    def __init__(self, **kwargs):
        super(ReparamDiagGaussian, self).__init__(**kwargs)
        
        self.trng = RandomStreams(np.random.randint(1, 2147462579))

    def forward(self, input):
        # The input must be a tuple
        assert isinstance(input, tuple), 'ReparamDiagGaussian expects a tuple input'

        # Unpack input into mu and logvar
        mu, logvar = input

        # Sample noise from standard Gaussian
        epsilon = self.trng.normal(size=mu.shape, avg=0.0, std=1.0, dtype=mu.dtype)

        # Reparameterization
        output = epsilon * T.exp(logvar * 0.5) + mu

        return output

# TODO: Gaussian noise
