#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import inspect
from typing import Callable
import warnings

__all__ = ['arity']

class Operator:
    def __init__(self, symbol, function, in_params):
        self.symbol = symbol
        self.function = function
        self.arity = in_params 
        
    def __call__(self, *args):
        if self.arity == 1:
            args = [args[0]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.function(*args)
    
    def name(self):
        return self.symbol

    def __name__(self):
        return self.symbol

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol

def arity(op: Operator) -> int:
    """Return the number of expected parameter of the function"""
    return op.arity