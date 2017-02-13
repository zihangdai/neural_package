import theano, theano.tensor as T
import lasagne

from .module import Module

class Linear(Module):
    def __init__(self, input_size, output_size, use_bias=True, **kwargs):
        super(Linear, self).__init__(**kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias

        weight = kwargs.get('weight', lasagne.init.Normal())
        self.add_parameter('weight', weight, (input_size, output_size))

        if use_bias:
            bias = kwargs.get('bias', lasagne.init.Normal())
            self.add_parameter('bias', bias, (output_size))

    def forward(self, input):
        output = T.dot(input, self.weight)
        if self.use_bias:
            output += self.bias

        return output

    def __repr__(self):
        return '{name} ({input_size} -> {output_size}, use_bias={use_bias})'.format(**self.__dict__)

# TODO: Bilinear