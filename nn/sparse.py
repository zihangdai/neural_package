import theano, theano.tensor as T
import lasagne

from .module import Module

class Embedding(Module):
    def __init__(self, input_size, output_size, padding_value=None,
                 max_norm=None, norm_type=None, **kwargs):
        super(Embedding, self).__init__(**kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.padding_value = padding_value
        self.max_norm = max_norm
        self.norm_type = norm_type

        weight = kwargs.get('weight', lasagne.init.Normal())
        self.add_parameters('weight', weight, (input_size, output_size))

    def forward(self, input):
        output_shape = [input.shape[i]
                        for i in range(input.ndim)] + [self.output_size]

        output = self.weight[input.flatten()].reshape(output_shape)

        return output