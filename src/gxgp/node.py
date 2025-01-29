#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import numbers
import warnings
from typing import Callable
import numpy as np

from .draw import draw
from .utils import arity, Operator

__all__ = ['Node']


class Node:
    _func: Callable
    _successors: tuple['Node']
    _arity: int
    _str: str
    _mse: float

    def __init__(self, node=None, successors=None, *, name=None):
        if isinstance(node, Operator):
            def _f(*_args, **_kwargs):
                return node(*_args)

            self._func = _f
            self._successors = tuple(successors)
            self._arity = arity(node)
            assert self._arity is None or len(tuple(successors)) == self._arity, (
                "Panic: Incorrect number of children."
                + f" Expected {len(tuple(successors))} found {arity(node)}"
            )
            self._leaf = False
            assert all(isinstance(s, Node) for s in successors), "Panic: Successors must be `Node`"
            self._successors = tuple(successors)
            if name is not None:
                self._str = node
            else:
                self._str = node
        elif isinstance(node, numbers.Number):
            self._func = eval(f'lambda **_kw: {node}')
            self._successors = tuple()
            self._arity = 0
            self._str = f'{node:g}'
        elif isinstance(node, str):
            self._func = eval(f'lambda *, {node}, **_kw: {node}')
            self._successors = tuple()
            self._arity = 0
            self._str = str(node)
        else:
            assert False

    def __call__(self, **kwargs):
        return self._func(*[c(**kwargs) for c in self._successors], **kwargs)
        
    def __str__(self):
        return self.long_name

    def __len__(self):
        return 1 + sum(len(c) for c in self._successors)

    @property
    def value(self):
        return self()

    @property
    def arity(self):
        return self._arity

    @property
    def successors(self):
        return list(self._successors)

    @successors.setter
    def successors(self, new_successors):
        self._successors = tuple(new_successors)

    @property
    def is_leaf(self):
        return not self._successors

    @property
    def short_name(self):
        return self._str

    @property
    def long_name(self):
        if self.is_leaf:
            return self.short_name
        else:
            return f'{self.short_name}(' + ', '.join(c.long_name for c in self._successors) + ')'

    @property
    def subtree(self):
        result = set()
        _get_subtree(result, self)
        return result
    
    @property
    def depth(self):
        if self.is_leaf:
            return 1
        return 1 + max(c.depth for c in self._successors)
    
    def is_operator(self) -> bool:
        return isinstance(self.short_name, Operator)
    
    def is_terminal(self) -> bool:
        return self.is_leaf
    
    def set_mse(self, mse: float):
        self._mse = mse

    @property
    def mse(self):
        return self._mse

    def draw(self):
        try:
            return draw(self)
        except Exception as msg:
            warnings.warn(f"Drawing not available ({msg})", UserWarning, 2)
            return None
    


def _get_subtree(bunch: set, node: Node):
    bunch.add(node)
    for c in node._successors:
        _get_subtree(bunch, c)


