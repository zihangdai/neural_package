import theano, theano.tensor as T
import lasagne

if theano.config.device.startswith('cuda'):
    from theano.gpuarray.dnn import GpuDnnConvDesc, GpuDnnConvGradI, dnn_conv, dnn_pool
    from theano.gpuarray.basic_ops import gpu_contiguous, gpu_alloc_empty
else:
    from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI, dnn_conv, dnn_pool
    from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty

from .module import Module

class Convolutional(Module):
    def __init__(self, filter_size, num_filters, num_channels,
                 stride=(1, 1), padding=(0, 0), use_bias=True, **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        assert isinstance(filter_size, tuple),\
            'filter_size should be a tuple, %s provided' % type(filter_size)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding

        weight = kwargs.get('weight', lasagne.init.Normal())
        self.add_parameter('weight', weight, (num_filters, num_channels) + filter_size)

        if use_bias:
            bias = kwargs.get('bias', lasagne.init.Constant(0.))
            self.add_parameter('bias', bias, (num_filters))

    def forward(self, input):
        output = dnn_conv(input, self.weight,
                          subsample=self.stride,
                          border_mode=self.padding)

        if self.use_bias:
            output += self.bias.dimshuffle('x', 0, 'x', 'x')

        return output

    def __repr__(self):
        s = ('{name} ({num_channels} -> {num_filters}, filter_size={filter_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        s += ', use_bias={use_bias})'
        return s.format(**self.__dict__)

class Deconvolutional(Module):
    """A deconvolutional (transposed convolutional) layer, which is the inverse of
    a convolutional layer defined by the same arguments. Therefore, it takes an input with `num_filters`
    channels and produces `num_channels` channels in output. Note that only specific cases are supported
    (same padding and even input sizes). TODO: generalize it."""

    def __init__(self, filter_size, num_filters, num_channels,
                 stride=(1, 1), padding=(0, 0), use_bias=True, **kwargs):
        super(Deconvolutional, self).__init__(**kwargs)

        assert isinstance(filter_size, tuple),\
            'filter_size should be a tuple, %s provided' % type(filter_size)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding

        weight = kwargs.get('weight', lasagne.init.Normal())
        # This is the shape of the corresponding convolutional kernel
        self.add_parameter('weight', weight, (num_filters, num_channels) + filter_size)

        if self.use_bias:
            bias = kwargs.get('bias', lasagne.init.Constant(0.))
            self.add_parameter('bias', bias, (num_channels))

    def forward(self, input):
        """Only works for same padding!"""
        img = gpu_contiguous(input)
        kerns = gpu_contiguous(self.weight)

        if theano.config.device.startswith('cuda'):
            # new theano GPU backend.
            alloc = gpu_alloc_empty(None, theano.config.floatX)
            out = alloc(img.shape[0], kerns.shape[1], img.shape[2]*self.stride[0], img.shape[3]*self.stride[1])
            desc = GpuDnnConvDesc(border_mode=self.padding, subsample=self.stride,
                                  conv_mode='conv')(out.shape)
        else:
            # old theano GPU backend
            out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*self.stride[0], img.shape[3]*self.stride[1])
            desc = GpuDnnConvDesc(border_mode=self.padding, subsample=self.stride,
                                  conv_mode='conv')(out.shape, kerns.shape)
        output = GpuDnnConvGradI()(kerns, img, out, desc)

        if self.use_bias:
            output += self.bias.dimshuffle('x', 0, 'x', 'x')

        return output

    def __repr__(self):
        s = ('{name} ({num_channels} <- {num_filters}, filter_size={filter_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        s += ', use_bias={use_bias})'
        return s.format(**self.__dict__)
