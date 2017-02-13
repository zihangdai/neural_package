from collections import OrderedDict
import numpy as np
import theano, theano.tensor as T

def _addindent(s_, num_spaces):
    s = s_.split('\n')
    # dont do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Module(object):
    def __init__(self, **kwargs):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        
        self._states = OrderedDict()
        self._updates = OrderedDict()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.train = theano.shared(1, name='train_flag')

    #### Mode: train vs infer
    # Set the module to evaluate mode
    def evaluate(self):
        self.train.set_value(0)

    # Set the module to training mode
    def training(self):
        self.train.set_value(1)

    #### Forward function to be implemented by each functioning module
    def forward(self, input):
        raise NotImplementedError

    #### Add parameter, state or module to the Module
    def init_shared_variable(self, tensor, shape, name=None):
        # Pass in a callable function which uses the shape to return np.ndarray
        if callable(tensor):
            tensor = tensor(shape)
        # Pass in np.ndarray as the initial value
        elif isinstance(tensor, np.ndarray):
            assert shape == tensor.shape, \
                'provided np.ndarray tensor shape does not match expected shape'
        # Pass in theano.shared valiable for parameter share
        elif isinstance(tensor, theano.Variable):
            assert np.all(shape == tensor.shape.eval()), \
                'provided theano shared tensor shape does not match expected shape'
            return tensor
        # Pass in NpzFile
        elif isinstance(tensor, np.lib.npyio.NpzFile):
            assert name in tensor, '%s is not in the archive'
            tensor_dict = tensor
            tensor = tensor_dict[name]
        # No other types supported for now
        else:
            raise ValueError('tensor type not understood')
        
        # Cast np.ndarray into floatX
        return theano.shared(tensor.astype(theano.config.floatX), name=name)

    def add_parameter(self, name, param, shape):
        """Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        self._parameters[name] = self.init_shared_variable(param, shape, 
                                        name='_'.join([self.name, name]))

    def add_state(self, name, state, shape):
        """Adds a persistent state to the module.

        This is typically used to register a state that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the persistent state.

        States can be accessed as attributes using given names.

        """
        if '_states' not in self.__dict__:
            raise AttributeError(
                "cannot assign state before Module.__init__() call")
        
        self._states[name] = self.init_shared_variable(state, shape,
                                    name='_'.join([self.name, name]))

    def add_module(self, name, module):
        if '_modules' not in self.__dict__:
            raise AttributeError(
                "cannot assign module before Module.__init__() call")
        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module):
            raise TypeError("{} is not a Module subclass".format(type(module)))
        
        self._modules[name] = module

    #### Sub-module management: self.children() + self = self.modules()
    def children(self):
        """Returns an iterator over children modules."""
        memo = set()
        for module in self._modules.values():
            if module is not None and module not in memo:
                memo.add(module)
                yield module

    def modules(self, memo=None):
        """Returns an iterator over children modules plus itself."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield self
            for module in self.children():
                for m in module.modules(memo):
                    yield m

    #### Parameters management
    def parameters(self, memo=None):
        """Returns an iterator over module parameters that are
        not in the memo set.
        """
        if memo is None:
            memo = set()
        for p in self._parameters.values():
            if p is not None and p not in memo:
                memo.add(p)
                yield p
        for module in self.children():
            for p in module.parameters(memo):
                yield p

    def get_parameters(self, memo=None):
        """Returns an list of module parameters that are
        not in the memo list.
        """
        if memo is None:
            return [p for p in self.parameters()]
        else:
            return [p for p in self.parameters(set(memo))]

    #### States management (anologous to those for parameters)
    def states(self, memo=None):
        """Returns an iterator over module states that are
        not in the memo set.
        """
        if memo is None:
            memo = set()
        for s in self._states.values():
            if s is not None and s not in memo:
                memo.add(s)
                yield s
        for module in self.children():
            for s in module.states(memo):
                yield s

    def get_states(self, memo=None):
        """Returns an list of module states that are
        not in the memo list.
        """
        if memo is None:
            return [s for s in self.states()]
        else:
            return [s for s in self.states(set(memo))]

    #### States management (we don't check duplicate for updates)
    def updates(self):
        for u in self._updates.values():
            yield u
        for module in self.children():
            for u in module.updates():
                yield u

    def get_updates(self):
        return [u for u in self.updates()]

    #### Recursive apply some function to all self.modules
    def apply(self, fn):
        for module in self.modules():
            module.apply(fn)
        return self

    #### Overwrite __(get|set|del)attr__ functions to enable acess
    #### to parameters, states and modules by __(get|set)attr__
    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_states' in self.__dict__:
            _states = self.__dict__['_states']
            if name in _states:
                return _states[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        """There are two things __setattr__ must make sure
        
           1. if the name has been occuppied _states or _parameters, 
           then the type of value must be correct
           2. if the name is not occuppied and the value is a Module, 
           set value as a child module, i.e. self._modules[name] = value
        """
        if hasattr(self, '_parameters') and name in self._parameters:
            if not isinstance(value, theano.Variable):
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(theano.Variable expected)"
                                .format(type(value), name))
            self.add_parameter(name, value, self._parameters[name].shape.eval())
        elif hasattr(self, '_states') and name in self._states:
            if not isinstance(value, theano.Variable):
                raise TypeError("cannot assign '{}' as state '{}' "
                                "(theano.Variable expected)"
                                .format(type(value), name))
            self.add_state(name, value, self._states[name].shape.eval())
        else:
            if hasattr(self, '_modules') and isinstance(value, Module):
                self._modules[name] = value
            else:
                object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._states:
            del self._states[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def __repr__(self):
        tmpstr = self.name + ' (\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  ('  + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr