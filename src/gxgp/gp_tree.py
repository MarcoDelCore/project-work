import random
from typing import Collection

from .node import Node
from .utils import arity, Operator

__all__ = ['TreeGP']

class TreeGP:
    def __init__(self, operators: dict[str, Operator], variables: int | Collection, constants: int | Collection):
        
        self.operators = operators
        if isinstance(variables, int):
            self.variables = [TreeGP.default_variable(i) for i in range(variables)]
        else:
            self.variables = list(variables)

        self.constants = constants

    @staticmethod
    def default_variable(i: int) -> str:
        return f'x{i}'

    def create_individual(self, max_depth=3):
        constants = [random.uniform(-1, 1) for _ in range(self.constants)] if isinstance(self.constants, int) else list(self.constants)

        stack = [] 
        root = None 

        # Stack format: (depth, parent_node, child_index)
        stack.append((0, None, None))

        while stack:
            depth, parent, child_index = stack.pop()

            if depth >= max_depth:
                leaf = self.random_terminal()
                node = Node(leaf, None)

            else:
                operator, operator_name = self.random_operator()
                num_children = arity(operator)
                children = [Node(0)] * num_children # Placeholder for children

                node = Node(operator, children, name=operator_name)

                for i in range(num_children - 1, -1, -1): 
                    stack.append((depth + 1, node, i))

            if parent is not None and child_index is not None:
                parent._successors[child_index] = node
            else:
                root = node 

        return root


    def random_operator(self) -> Operator:
        operator_name = random.choice(list(self.operators.keys()))
        return self.operators[operator_name], operator_name
    
    def random_terminal(self) -> str:
        constants = []
        if isinstance(self.constants, int):
            constants = [random.uniform(-1, 1) for _ in range(self.constants)]
        else:
            constants = list(self.constants)
        return random.choice(self.variables+constants)