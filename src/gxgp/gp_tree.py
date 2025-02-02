#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import numbers
import random
from typing import Collection
import numpy as np

from .node import Node
from .utils import arity, Operator

__all__ = ['TreeGP']

class TreeGP:
    def __init__(self, variables: int | Collection, constants: int | Collection, *, seed=42):

        numpy_operators = [
            Operator("np.add", np.add, 2),
            Operator("np.subtract", np.subtract, 2),
            Operator("np.multiply", np.multiply, 2),
            Operator("np.divide", np.divide, 2),
            Operator("np.power", np.power, 2),
            Operator("np.exp", np.exp, 1),
            Operator("np.sqrt", np.sqrt, 1),
            Operator("np.sin", np.sin, 1),
            Operator("np.cos", np.cos, 1),
            Operator("np.tan", np.tan, 1),
            Operator("np.log", np.log, 1),
            Operator("np.abs", np.abs, 1),
            Operator("np.negative", np.negative, 1),
            Operator("np.reciprocal", np.reciprocal, 1),
            Operator("np.square", np.square, 1),
            Operator("np.cbrt", np.cbrt, 1),
            Operator("np.log1p", np.log1p, 1),
        ]
        """    Operator("np.expm1", np.expm1, 1),
            Operator("np.sinh", np.sinh, 1),
            Operator("np.cosh", np.cosh, 1),
            Operator("np.tanh", np.tanh, 1),
            Operator("np.arcsin", np.arcsin, 1),
            Operator("np.arccos", np.arccos, 1),
            Operator("np.arctan", np.arctan, 1),
            Operator("np.arcsinh", np.arcsinh, 1),
            Operator("np.arccosh", np.arccosh, 1),
            Operator("np.arctanh", np.arctanh, 1),
            Operator("np.maximum", np.maximum, 2),
            Operator("np.minimum", np.minimum, 2),
        ]"""
        
        self.operators = numpy_operators
        if isinstance(variables, int):
            self.variables = [TreeGP.default_variable(i) for i in range(variables)]
        else:
            self.variables = list(variables)

        self.constants = constants
        self.seed = seed

    @staticmethod
    def default_variable(i: int) -> str:
        return f'x{i}'

    def create_individual(self, max_depth=3):
        def generate_subtree(current_depth):
            constants = []
            if isinstance(self.constants, int):
                constants = [random.uniform(-1, 1) for _ in range(self.constants)]
            else:
                constants = list(self.constants)

            if current_depth >= max_depth or (random.random() < 0.2 and current_depth > 1):
                # Generate a leaf node (variable or constant)
                leaf = random.choice(self.variables + constants)
                return Node(leaf, None)

            else:
                # Generate an internal node (operator)
                operator = random.choice(self.operators)
                num_children = arity(operator)

                children = [generate_subtree(current_depth + 1) for _ in range(num_children)]

                if operator.name() == 'np.sqrt':
                    while (isinstance(children[0].short_name, numbers.Number) and float(children[0].short_name) < 0):
                        children[0] = generate_subtree(current_depth + 1)  # Regenerate only first child

                elif operator.name() == 'np.power':
                    # Avoid negative base with non-integer exponent (to avoid complex numbers)
                    while (isinstance(children[0].short_name, numbers.Number) and float(children[0].short_name) < 0
                        and isinstance(children[1].short_name, numbers.Number) and not children[1].short_name.is_integer()):
                        children[0] = generate_subtree(current_depth + 1)  # Regenerate base
                        children[1] = generate_subtree(current_depth + 1)  # Regenerate exponent

                elif operator.name() == 'np.divide':
                    while (isinstance(children[1].short_name, numbers.Number) and float(children[1].short_name) == 0):
                        children[1] = generate_subtree(current_depth + 1)  # Regenerate denominator

                return Node(operator, children)

        return generate_subtree(0)
    

    def random_operator(self) -> Operator:
        return random.choice(self.operators)
    
    def random_terminal(self) -> str:
        constants = []
        if isinstance(self.constants, int):
            constants = [random.uniform(-1, 1) for _ in range(self.constants)]
        else:
            constants = list(self.constants)
        return random.choice(self.variables+constants)
    
    def is_operator(self) -> bool:
        return self.value in self.operators

    def is_terminal(self) -> bool:
        return not self.is_operator()