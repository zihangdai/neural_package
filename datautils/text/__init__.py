from .vocab import *
from .utils import *
from .datasets import *
from .datareader import *
from .data_iterator import *

try:
    import torch
    torch_support = True
except ImportError:
    import warnings
    warnings.warn('Fail to import torch. No torch support') 
    torch_support = False

if torch_support:
    from .utils_torch import *
