import theano, theano.tensor as T
from theano.ifelse import ifelse
import lasagne

from .module import Module

class BatchNorm(Module):
    def __init__(self, input_size, **kwargs):
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

        # moving average
        mvavg_mean_dimsf = self.avg_mean.dimshuffle(*self.dim_args)
        mvavg_stdv_dimsf = T.sqrt(1e-6 + self.avg_var).dimshuffle(*self.dim_args)

        # normalized features during inference
        norm_features_infer = (input - mvavg_mean_dimsf) / mvavg_stdv_dimsf

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

    def infer(self, input):
        # moving average
        mvavg_mean_dimsf = self.avg_mean.dimshuffle(*self.dim_args)
        mvavg_stdv_dimsf = T.sqrt(1e-6 + self.avg_var).dimshuffle(*self.dim_args)

        # normalized features during inference
        norm_features = (input - mvavg_mean_dimsf) / mvavg_stdv_dimsf

        # rescale and shift the normalized features using trainable scale & shift parameters
        output = norm_features * self.scale.dimshuffle(*self.dim_args)
        output += self.shift.dimshuffle(*self.dim_args)
        
        return output

    def __repr__(self):
        return '{name} ({input_size})'.format(**self.__dict__)

class BatchRenorm(Module):
    def __init__(self, input_size, max_r=1., max_d=0., **kwargs):
        super(BatchRenorm, self).__init__(**kwargs)
        self.input_size = input_size
        self.max_r = max_r
        self.max_d = max_d

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

        return super(BatchRenorm, self).updates()

    def forward(self, input):
        if input.ndim == 4:
            self.sum_axes = (0,2,3)
            self.dim_args = ['x',0,'x','x']
        else:
            self.sum_axes = 0
            self.dim_args = ['x',0]

        #### inference
        # normalized features during inference
        mvavg_mean_dimsf = self.avg_mean.dimshuffle(*self.dim_args)
        mvavg_stdv_dimsf = T.sqrt(1e-6 + self.avg_var).dimshuffle(*self.dim_args)
        norm_feats_infer = (input - mvavg_mean_dimsf) / mvavg_stdv_dimsf

        #### training
        # (1) batch statistics
        # batch_mean
        batch_mean = T.mean(input, axis=self.sum_axes).flatten()
        batch_mean_dimsf = batch_mean.dimshuffle(*self.dim_args)

        # centered_input has zero mean
        centered_input = input - batch_mean_dimsf
        
        # batch_var and batch_stdv
        batch_var = T.mean(T.square(centered_input), axis=self.sum_axes).flatten()
        batch_var_dimsf = batch_var.dimshuffle(*self.dim_args)
        batch_stdv_dimsf = T.sqrt(1e-6 + batch_var_dimsf)
        norm_feats_train = centered_input / batch_stdv_dimsf
        
        # (2) affine transformation
        r = theano.gradient.disconnected_grad(
                T.clip(batch_stdv_dimsf / mvavg_stdv_dimsf, 1. / self.max_r, self.max_r))
        d = theano.gradient.disconnected_grad(
                T.clip((batch_mean_dimsf - mvavg_mean_dimsf) / mvavg_stdv_dimsf, -self.max_d, self.max_d))
        renorm_feats_train = norm_feats_train * r + d

        # state updates during training
        self.obversed_means.append(batch_mean)
        self.obversed_vars.append(T.cast((0.1*input.shape[0]) / (input.shape[0]-1), theano.config.floatX) * batch_var)

        # train vs infer
        norm_features = ifelse(self.train, renorm_feats_train, norm_feats_infer)

        # rescale and shift the normalized features using trainable scale & shift parameters
        output = norm_features * self.scale.dimshuffle(*self.dim_args)
        output += self.shift.dimshuffle(*self.dim_args)

        return output

    def infer(self, input):
        # moving average
        mvavg_mean_dimsf = self.avg_mean.dimshuffle(*self.dim_args)
        mvavg_stdv_dimsf = T.sqrt(1e-6 + self.avg_var).dimshuffle(*self.dim_args)

        # normalized features during inference
        norm_features = (input - mvavg_mean_dimsf) / mvavg_stdv_dimsf

        # rescale and shift the normalized features using trainable scale & shift parameters
        output = norm_features * self.scale.dimshuffle(*self.dim_args)
        output += self.shift.dimshuffle(*self.dim_args)
        
        return output

    def __repr__(self):
        return '{name} (size={input_size}, max_r={max_r}, max_d={max_d})'.format(**self.__dict__)
