import warnings

__all__ = ['arity']

class Operator:
    def __init__(self, function, in_params):
        self.function = function
        self.arity = in_params 
        
    def __call__(self, *args):
        if self.arity == 1:
            args = [args[0]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.function(*args)

def arity(op: Operator) -> int:
    """Return the number of expected parameter of the function"""
    return op.arity