import theano
import theano.tensor as T
from theano.gpuarray import dnn
from theano.gpuarray.type import gpuarray_shared_constructor
import lasagne
from module import Module

# rnn_mode : {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'}
class RNN(Module):
    def __init__(self, rnn_mode, input_size, hidden_size, num_layers,
                 bidirectional=False, skip_input_projection=False, **kwargs):
        super(RNN, self).__init__(**kwargs)

        self.rnn_mode    = rnn_mode
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        if bidirectional:
            self.direction_mode = 'bidirectional'
            self.num_directions = 2
        else:
            self.direction_mode = 'unidirectional'
            self.num_directions = 1
        if skip_input_projection:
            assert self.input_size == self.hidden_size, \
                'input_size must be equal to the hidden_size when skip'
            self.input_mode = 'skip'
        else:
            self.input_mode = 'linear'

        self.rnn_core = dnn.RNNBlock(theano.config.floatX, self.hidden_size, self.num_layers, 
                                     rnn_mode=self.rnn_mode, input_mode=self.input_mode, 
                                     direction_mode=self.direction_mode)
        self.param_size = self.rnn_core.get_param_size([1, self.input_size])
        params_cudnn_initializer = kwargs.get('params_cudnn', lasagne.init.Normal(0.01))
        params_cudnn = gpuarray_shared_constructor(params_cudnn_initializer((self.param_size)))
        self.add_parameter('params_cudnn', params_cudnn, (self.param_size))

    def create_decoder(self):
        # create a decoder RNN using parameters of current cudnn.RNN
        for idx in range(self.num_layers):
            insize = self.input_size if idx == 0 else self.hidden_size
            dnn_params = self.rnn_core.split_params(self.params_cudnn, idx, [1, insize])
            W_ii, b_ii, W_if, b_if, W_ic, b_ic, W_io, b_io = dnn_params[:8] 
            W_hi, b_hi, W_hf, b_hf, W_hc, b_hc, W_ho, b_ho = dnn_params[8:] 

    def forward(self, input, init_hidden=None):
        if init_hidden is None:
            init_hidden = self.init_hidden_state(input.shape[1])
        if self.rnn_mode == 'lstm':
            output, last_h, last_c = self.rnn_core.apply(self.params_cudnn, input, init_hidden[0], init_hidden[1])
            last_hidden = (last_h, last_c)
        else:
            output, last_hidden = self.rnn_core.apply(self.params_cudnn, input, init_hidden)

        return output, last_hidden

    def init_hidden_state(self, batch_size):
        if self.rnn_mode == 'lstm':
            init_hidden = (T.alloc(0., self.num_layers*self.num_directions, batch_size, self.hidden_size), 
                           T.alloc(0., self.num_layers*self.num_directions, batch_size, self.hidden_size))
        else:
            init_hidden =  T.alloc(0., self.num_layers*self.num_directions, batch_size, self.hidden_size)

        return init_hidden

    def __repr__(self):
        tmp_str = ('cudnn.{rnn_mode} (input_size = {input_size}, '
                   'hidden_size = {hidden_size}, num_layers = {num_layers}'
                   .format(**self.__dict__))
        if self.num_directions == 2:
            tmp_str += ', bidirectional = True'
        if self.input_mode == 'skip':
            tmp_str += ', skip_input_projection = True'
        tmp_str += ')'
        return tmp_str

if __name__ == '__main__':
    bidirectional = False
    rnn = RNN('lstm', 3, 2, 1, bidirectional=bidirectional)
    print rnn
    print rnn.get_parameters()

    input = T.ones((12,1,3))
    output, last_hidden = rnn.forward(input)
    if bidirectional:
        print output[-1].eval()[:,:2] 
        print output[0].eval()[:,2:] 
        last_h = last_hidden[0].eval()
        print last_h[0]
        print last_h[1]
    else:
        print output.eval().shape
        print last_hidden[0].eval().shape
        print last_hidden[1].eval().shape
