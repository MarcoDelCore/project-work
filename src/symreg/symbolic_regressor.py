from copy import deepcopy
import random
from typing import Collection
from gxgp.gp_common import xover_swap_subtree
from gxgp import TreeGP, Node
import numpy as np

from gxgp.utils import arity
from joblib import Parallel, delayed


class SymbolicRegressor:
    def __init__(self,
                 operators: Collection, 
                 population_size=500,
                 generations=1000,
                 tournament_size=100,
                 randomness=0.3,
                 stopping_criteria=0.00001,
                 max_depth=7,
                 parsimony_coefficient=0.001,
                 p_crossover=0.7,
                 p_subtree_mutation=0.1,
                 p_hoist_mutation=0.1,
                 p_point_mutation=0.1,
                 max_samples=1.0):
        
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.randomness = randomness
        self.stopping_criteria = stopping_criteria
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.max_samples = max_samples
        self.operators = operators
        
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')
    

    def generate_population(self, X, y):
        """Generate a population of valid individuals."""
        depths = list(range(2, self.max_depth + 1))
        
        if len(self.population) == 0:
            print("Generating initial population...")
            while len(self.population) < self.population_size:
                depth = random.choice(depths)
                individual = self.treeGp.create_individual(depth)
                try:
                    mse = compute_mse(individual, X, y, max_samples=self.max_samples)
                    individual.set_mse(mse)
                    self.population.append(individual)
                except Exception as e:
                    continue  # If invalid, retry
        else:
            num_to_generate = int(self.population_size * self.randomness)
            while len(self.population) < num_to_generate:
                depth = random.choice(depths)
                individual = self.treeGp.create_individual(depth)
                try:
                    mse = compute_mse(individual, X, y, max_samples=self.max_samples)
                    individual.set_mse(mse)
                    self.population.append(individual)
                except Exception:
                    continue
        return self.population
    
    def evaluate_fitness(self, X, y, generation):
        """Compute the fitness of individuals considering parsimony."""
        parsimony_coeff = self.parsimony_coefficient * (generation / self.generations)
        
        for individual in self.population:
            individual.fitness = individual.mse + parsimony_coeff * individual.depth
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness)
        
        # Update best individual
        if self.population[0].mse < self.best_fitness:
            self.best_fitness = self.population[0].mse
            self.best_individual = self.population[0]
            return True
        else:
            return False
        

    def tournament_selection(self):
        """ Selects the best individuals for evolution """
        sorted_population = sorted(self.population, key=lambda x: x.mse)
        self.population = sorted_population[:self.tournament_size]
    
    def evolve_population(self, X, y):
        """Apply crossover and mutation to generate new individuals."""
        
        while len(self.population) < self.population_size:
            r = random.random()
            if r < self.p_crossover:
                parent1, parent2 = random.sample(self.population, 2)
                child = xover_swap_subtree(parent1, parent2)
            elif r < self.p_crossover + self.p_subtree_mutation:
                parent = random.choice(self.population)
                child = self.subtree_mutation(parent)
            elif r < self.p_crossover + self.p_subtree_mutation + self.p_hoist_mutation:
                parent = random.choice(self.population)
                child = self.hoist_mutation(parent)
            else:
                parent = random.choice(self.population)
                child = self.point_mutation(parent)
            
            try:
                mse = compute_mse(child, X, y, max_samples=self.max_samples)
                child.set_mse(mse)
                self.population.append(child)
            except Exception:
                continue  # Skip invalid individuals
    
    def fit(self, X, y):
        self.treeGp = TreeGP(self.operators, X.shape[1], 15)
        stagnation_counter = 0

        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}")
            
            if stagnation_counter == 10 or (generation > 0 and generation % 50 == 0):
                print("Injecting randomness...")
                num_to_mutate = int(self.tournament_size * 0.3)
                self.population = self.population[:num_to_mutate]
                stagnation_counter = 0
            
            self.generate_population(X, y)
            
            self.evolve_population(X, y)
            
            if self.evaluate_fitness(X, y, generation):
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            print(f"Stagnation counter: {stagnation_counter}")
            
            print(f"Best MSE: {self.best_fitness}")
            print(f"Best Individual: {self.best_individual}")
            
            if self.best_fitness <= self.stopping_criteria:
                print("Stopping criteria met. Training complete.")
                break
            
            self.tournament_selection()
        
        return self.best_individual


    def subtree_mutation(self, tree: Node) -> Node:
        """
        Performs subtree mutation on a given tree by replacing a randomly selected 
        subtree with a newly generated subtree of the same depth.

        This mutation helps introduce structural diversity and explore new solutions 
        in the search space.

        Args:
            tree (Node): The tree to mutate.

        Returns:
            Node: A new tree with the subtree mutation applied.
        """

        # Create a deep copy of the tree to avoid modifying the original tree
        offspring = deepcopy(tree)

        successors = None
        # Randomly select a node with at least one child (i.e., non-leaf node)
        while not successors:
            node = random.choice(list(offspring.subtree))
            successors = node.successors  # Ensure the selected node has children

        # Select a random child index to replace
        i = random.randrange(len(successors))
        depth = successors[i].depth  # Preserve the depth of the subtree being replaced

        # Generate a new subtree with the same depth
        new_subtree = self.treeGp.create_individual(depth)

        # Replace the selected child with the newly generated subtree
        successors[i] = new_subtree
        node.successors = successors  # Update the node's children

        return offspring


    def hoist_mutation(self, tree: Node) -> Node:
        """
        Applies hoist mutation to the given tree. This mutation selects a random 
        subtree and promotes one of its child nodes to replace it, reducing tree 
        depth and potentially improving generalization.

        Args:
            tree (Node): The input tree to mutate.

        Returns:
            Node: A new tree with the hoist mutation applied.
        """
        offspring = deepcopy(tree)

        successors = offspring.successors
        i = random.randrange(len(successors))
        node = successors[i]
        if node.is_leaf:
            return offspring
        else:
            j = random.randrange(len(node.successors))
            new_node = deepcopy(node.successors[j])
            successors[i] = new_node
            offspring.successors = successors
        return offspring
    

    def point_mutation(self, tree_node: Node) -> Node:
        """
        Performs point mutation on a given tree by randomly selecting a node and 
        replacing its operator or terminal with a new one while preserving its 
        structure.

        Point mutation introduces small variations in the tree without drastically 
        altering its shape, helping maintain diversity in the population.

        Args:
            tree_node (Node): The input tree to mutate.

        Returns:
            Node: A new tree with the point mutation applied.
        """

        # Create a deep copy of the tree to avoid modifying the original tree
        offspring = deepcopy(tree_node)

        successors = None
        # Randomly select a node that has at least one child (i.e., non-leaf node)
        while successors is None or len(successors) == 0:
            node = random.choice(list(offspring.subtree))
            successors = node.successors  # Ensure the selected node has children

        # Select a random child node to mutate
        i = random.randrange(len(successors))
        node_to_mutate = successors[i]

        # If the selected node is a leaf, replace its terminal value
        if node_to_mutate.is_leaf:
            new_node = Node(self.treeGp.random_terminal(), node_to_mutate.successors)
        
        else:
            # If it's an operator node, replace it with another operator of the same arity
            new_op = None
            while new_op is None or arity(new_op) != node_to_mutate.arity:
                new_op = self.treeGp.random_operator()
            new_node = Node(new_op, node_to_mutate.successors)

        # Replace the selected node with the mutated node
        successors[i] = deepcopy(new_node)
        node.successors = successors  # Update the node's children

        return offspring


def compute_mse(individual: Node, X, Y, max_samples=None):
    """
    Computes the Mean Squared Error (MSE) for a given individual (tree-based model) 
    on the dataset (X, Y). The function evaluates the individual's predictions and 
    penalizes invalid or extreme values to ensure stability.

    Args:
        individual (Node): The symbolic regression model (tree) to evaluate.
        X (numpy.ndarray): Input features.
        Y (numpy.ndarray): Target values.
        max_samples (float, optional): A fraction (0 < max_samples <= 1) of data points 
                                       to use for evaluation. If None, all data points are used.

    Returns:
        float: The computed MSE value. If the computation fails or results in an 
               unstable value, a high penalty (1e20) is returned.
    """

    # Generate an array of indices corresponding to the dataset
    indices = np.arange(len(X))
    
    # If max_samples is a fraction, randomly select a subset of the dataset
    if max_samples and max_samples < 1.0:
        indices = np.random.choice(indices, int(len(X) * max_samples), replace=False)
    
    # Extract the selected samples
    X_batch = X[indices]
    Y_batch = Y[indices]

    y_pred = []  # Stores predicted values

    # Iterate over each sample in the batch
    for row in X_batch:
        # Create a dictionary mapping variable names (e.g., "x0", "x1", ...) to their values
        variables = {f"x{i}": val for i, val in enumerate(row)}
        
        try:
            # Compute the predicted value using the tree model
            pred = individual(**variables)
            
            # Penalize non-numeric or extreme values to avoid instability
            if not np.isfinite(pred) or np.abs(pred) > 1e20:
                pred = np.inf  # Assign a high penalty value for unstable models
        
        except Exception as e:
            raise e  # Re-raise exceptions for debugging
        
        y_pred.append(pred)

    y_pred = np.array(y_pred)  # Convert predictions to a NumPy array

    # Ensure all predictions are finite; replace NaN or infinite values with large penalties
    if not np.all(np.isfinite(y_pred)):
        y_pred = np.nan_to_num(y_pred, nan=np.inf, posinf=np.inf, neginf=-np.inf)

    # Compute MSE with error handling
    try:
        mse = np.square(Y_batch - y_pred).mean()
        
        # If MSE is not finite or is too large, assign a high penalty value
        if not np.isfinite(mse) or mse > 1e20:
            mse = 1e20  
    except Exception:
        mse = 1e20  # Assign penalty in case of computational errors

    return mse  # Return the computed or penalized MSE


def simplify(node: Node) -> Node:
    """
    Simplifies a given expression tree by evaluating constant expressions 
    and applying algebraic simplifications.

    This function recursively simplifies nodes, reducing redundancy and 
    improving readability of the final expression.

    Args:
        node (Node): The root node of the expression tree.

    Returns:
        Node: The simplified tree.
    """

    # If the node is a leaf (constant or variable), return it as is
    if node.is_leaf:
        return node

    # Recursively simplify the successors (child nodes)
    simplified_successors = [simplify(child) for child in node.successors]

    # Attempt to evaluate constant expressions
    try:
        # Convert child nodes to numeric values
        values = [float(child.short_name) for child in simplified_successors]
        
        # Clip values to prevent overflow/underflow
        clamped_values = [np.clip(val, -1e300, 1e300) for val in values]

        # Handle special cases of mathematical operations
        if node.short_name == 'np.sqrt' and clamped_values[0] < 0:
            print(f"Error: Attempted sqrt({clamped_values[0]}), which is invalid.")
            return Node(node='NaN')  # Return NaN for invalid sqrt
        
        if node.short_name == 'np.power' and clamped_values[0] < 0 and not float(clamped_values[1]).is_integer():
            print(f"Error: Attempted power({clamped_values[0]}, {clamped_values[1]}), which results in a complex number.")
            return Node(node='NaN')  # Return NaN for invalid power operations
        
        if node.short_name == 'np.divide' and clamped_values[1] == 0:
            print(f"Error: Attempted divide({clamped_values[0]}, 0), which is invalid.")
            return Node(node='inf')  # Return infinity for division by zero

        # Evaluate the operation if it's a known function
        result = node._func(*clamped_values) if node._func else None

        # If the result is finite, replace the node with a constant
        if result is not None and np.isfinite(result):
            return Node(node=result)

    except (ValueError, TypeError, ZeroDivisionError):
        pass  # Ignore errors and continue with algebraic simplifications

    # Apply algebraic simplifications
    if node.short_name == 'np.add':
        # x + 0 -> x
        simplified_successors = [child for child in simplified_successors if child.short_name != '0']
        if not simplified_successors:
            return Node(node=0)
        if len(simplified_successors) == 1:
            return simplified_successors[0]

    elif node.short_name == 'np.multiply':
        # x * 1 -> x, x * 0 -> 0
        if any(child.short_name == '0' for child in simplified_successors):
            return Node(node=0)  # If multiplying by 0, result is 0
        simplified_successors = [child for child in simplified_successors if child.short_name != '1']
        if len(simplified_successors) == 1:
            return simplified_successors[0]

    elif node.short_name == 'np.subtract':
        # x - 0 -> x
        if len(simplified_successors) == 2 and simplified_successors[1].short_name == '0':
            return simplified_successors[0]
        # x - x -> 0
        if len(simplified_successors) == 2 and simplified_successors[0].short_name == simplified_successors[1].short_name:
            return Node(node=0)

    elif node.short_name == 'np.divide':
        # x / 1 -> x
        if len(simplified_successors) == 2 and simplified_successors[1].short_name == '1':
            return simplified_successors[0]
        # x / 0 -> 'inf'
        if len(simplified_successors) == 2 and simplified_successors[1].short_name == '0':
            return Node(node='inf')  # Handle division by zero

    elif node.short_name == 'np.power':
        base, exp = simplified_successors
        # x ** 0 -> 1
        if exp.short_name == '0':
            return Node(node=1)
        # x ** 1 -> x
        if exp.short_name == '1':
            return base
        # 0 ** x -> 0 (for x > 0)
        if base.short_name == '0' and float(exp.short_name) > 0:
            return Node(node=0)

    # If no simplifications applied, return a new node with simplified successors
    return Node(node=node.short_name, successors=simplified_successors)
