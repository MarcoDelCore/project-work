import numbers
from typing import Callable

from .utils import arity, Operator

__all__ = ['Node']

class Node:
    _func: Callable
    _successors: list['Node']
    _arity: int
    _str: str
    _mse: float

    def __init__(self, node=None, successors=None, *, name=None):
        if isinstance(node, Operator):
            def _f(*_args, **_kwargs):
                return node(*_args)

            self._func = _f
            self._successors = list(successors)
            self._arity = arity(node)
            assert self._arity is None or len(list(successors)) == self._arity, (
                "Panic: Incorrect number of children."
                + f" Expected {len(list(successors))} found {arity(node)}"
            )
            self._leaf = False
            assert all(isinstance(s, Node) for s in successors), "Panic: Successors must be `Node`"
            self._successors = list(successors)
            if name is not None:
                self._str = name
            else:
                self._str = node
        elif isinstance(node, numbers.Number):
            self._func = eval(f'lambda **_kw: {node}')
            self._successors = list()
            self._arity = 0
            self._str = f'{node:g}'
        elif isinstance(node, str):
            self._func = eval(f'lambda *, {node}, **_kw: {node}')
            self._successors = list()
            self._arity = 0
            self._str = str(node)
        else:
            assert False, "Invalid node type"

    def __call__(self, **kwargs):
        stack = [(self, False)]  # (node, visited)
        result_stack = []

        while stack:
            node, visited = stack.pop()
            if visited:
                # Se il nodo è già stato visitato, calcola il risultato
                args = [result_stack.pop() for _ in range(node.arity)]
                result_stack.append(node._func(*args, **kwargs))
            else:
                # Se il nodo non è stato visitato, aggiungi i suoi successori allo stack
                stack.append((node, True))
                for child in reversed(node._successors):
                    stack.append((child, False))

        return result_stack[0]

    def __str__(self):
        return self.long_name

    def __len__(self):
        stack = [self]
        count = 0

        while stack:
            node = stack.pop()
            count += 1
            for child in node._successors:
                stack.append(child)

        return count

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
        self._successors = list(new_successors)

    @property
    def is_leaf(self):
        return not self._successors

    @property
    def short_name(self):
        return self._str

    @property
    def long_name(self):
        stack = [(self, False)]
        result_stack = []

        while stack:
            node, visited = stack.pop()
            if visited:
                if node.is_leaf:
                    result_stack.append(node.short_name)
                else:
                    args = [result_stack.pop() for _ in range(node.arity)]
                    result_stack.append(f'{node.short_name}(' + ', '.join(args) + ')')
            else:
                stack.append((node, True))
                for child in reversed(node._successors):
                    stack.append((child, False))

        return result_stack[0]

    @property
    def subtree(self):
        stack = [self]
        result = set()

        while stack:
            node = stack.pop()
            result.add(node)
            for child in node._successors:
                stack.append(child)

        return result

    @property
    def depth(self):
        stack = [(self, 1)]
        max_depth = 0

        while stack:
            node, current_depth = stack.pop()
            max_depth = max(max_depth, current_depth)
            for child in node._successors:
                stack.append((child, current_depth + 1))

        return max_depth

    def set_mse(self, mse: float):
        self._mse = mse

    @property
    def mse(self):
        return self._mse
