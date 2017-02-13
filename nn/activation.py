import theano, theano.tensor as T

from .module import Module

class Expression(Module):
    def __init__(self, func, **kwargs):
        super(Expression, self).__init__(**kwargs)
        self.func = func

    def forward(self, input):
        output = self.func(input)
        return output

    def __repr__(self):
        return '{name} ()'.format(**self.__dict__)

Tanh    = lambda : Expression(T.tanh, name='Tanh')
ReLU    = lambda : Expression(T.nnet.relu, name='ReLU')
Sigmoid = lambda : Expression(T.nnet.sigmoid, name='Sigmoid')
Softmax = lambda : Expression(T.nnet.softmax, name='Softmax')

class LeakyRectify(Module):
    def __init__(self, leak=0.2, **kwargs):
        super(LeakyRectify, self).__init__(**kwargs)
        self.leak = leak

    def forward(self, input):
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * input + f2 * abs(input)

    def __repr__(self):
        return '{name} (leak={leak})'.format(**self.__dict__)