import theano, theano.tensor as T
from theano.ifelse import ifelse
import lasagne

from .module import Module

class BatchNorm(Module):
    def __init__(self, input_size, g=lasagne.init.Constant(1.), **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.input_size = input_size

        shift = kwargs.get('shift', lasagne.init.Constant(0.))
        self.add_parameter('shift', shift, (self.input_size))

        scale = kwargs.get('scale', lasagne.init.Constant(1.))
        self.add_parameter('scale', scale, (self.input_size))

        avg_mean = kwargs.get('avg_mean', lasagne.init.Constant(0.))
        self.add_state('avg_mean', avg_mean, (self.input_size))
        
        avg_var = kwargs.get('avg_var', lasagne.init.Constant(1.))
        self.add_state('avg_var', avg_var, (self.input_size))

        self.obversed_means = []
        self.obversed_vars  = []

    def updates(self):
        """There are two possible ways to update the states in BN.
            
           For now, the way that is order invariant is used
        """
        #### sequential update (order sensitive)
        # new_avg_mean = self.avg_mean.clone()
        # for m in self.obversed_means:
        #     new_avg_mean = 0.9 * new_avg_mean + 0.1 * m
        # new_avg_var  = self.avg_var.clone()
        # for v in self.obversed_vars:
        #     new_avg_var = 0.9 * new_avg_var + 0.1 * v

        #### update once (order invariant)
        new_avg_mean = 0.9 * self.avg_mean + 0.1 * (sum(self.obversed_means) / len(self.obversed_means))
        new_avg_var  = 0.9 * self.avg_var  + 0.1 * (sum(self.obversed_vars)  / len(self.obversed_vars))
        
        self._updates['avg_mean'] = (self.avg_mean, new_avg_mean)
        self._updates['avg_var'] = (self.avg_var, new_avg_var)

        return super(BatchNorm, self).updates()

    def forward(self, input):
        if input.ndim == 4:
            self.sum_axes = (0,2,3)
            self.dim_args = ['x',0,'x','x']
        else:
            self.sum_axes = 0
            self.dim_args = ['x',0]

        dimshuffle_mean = self.avg_mean.dimshuffle(*self.dim_args)
        dimshuffle_stdv = T.sqrt(1e-6 + self.avg_var).dimshuffle(*self.dim_args)

        # normalized features during inference
        norm_features_infer = (input - dimshuffle_mean) / dimshuffle_stdv

        # normalized features during training
        batch_mean = T.mean(input, axis=self.sum_axes).flatten()
        centered_input = input-batch_mean.dimshuffle(*self.dim_args)
        batch_var  = T.mean(T.square(centered_input),axis=self.sum_axes).flatten()
        batch_stdv = T.sqrt(1e-6 + batch_var)
        norm_features_train = centered_input / batch_stdv.dimshuffle(*self.dim_args)

        # state updates during training
        self.obversed_means.append(batch_mean)
        self.obversed_vars.append(T.cast((0.1*input.shape[0]) / (input.shape[0]-1), theano.config.floatX) * batch_var)

        # train vs infer
        norm_features = ifelse(self.train, norm_features_train, norm_features_infer)

        # rescale and shift the normalized features using trainable scale & shift parameters
        output = norm_features * self.scale.dimshuffle(*self.dim_args)
        output += self.shift.dimshuffle(*self.dim_args)

        return output

    def __repr__(self):
        return '{name} ({input_size})'.format(**self.__dict__)

