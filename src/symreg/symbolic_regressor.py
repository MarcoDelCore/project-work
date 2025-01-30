from copy import deepcopy
import numbers
import random
from gxgp.gp_common import xover_swap_subtree
from gxgp import TreeGP, Node
import numpy as np

from gxgp.utils import arity
from joblib import Parallel, delayed


class SymbolicRegressor:
    def __init__(self, 
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
        
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')
        print('SymbolicRegressor initialized')
    
    

    def generate_population_parallel(self, X, y, n_threads=4):
        """Parallelized population generation"""
        def generate_individual(depth_range, treeGp):
            """Generate a single valid individual with its MSE"""
            while True:
                depth = random.choice(depth_range)
                individual = treeGp.create_individual(depth)
                try:
                    mse = compute_mse(individual, X, y, max_samples=1.0)
                    individual.set_mse(mse)
                    return individual
                except Exception:
                    continue  # If invalid, retry
        depths = list(range(2, self.max_depth + 1))
        num_per_thread = self.population_size - len(self.population) // n_threads

        # Parallel execution
        new_population = Parallel(n_jobs=n_threads)(
            delayed(generate_individual)(depths, self.treeGp) for _ in range(self.population_size - len(self.population))
        )

        self.population.extend(new_population)
        return self.population
    
    def evaluate_fitness(self, X, y, generation):
        """ Computes the fitness of individuals considering parsimony """
        parsimony_coeff = self.parsimony_coefficient * (generation / self.generations)
        def fitness(ind):
            return ind.mse + parsimony_coeff * ind.depth
        
        fitness_results = Parallel(n_jobs=6)(delayed(fitness)(ind) for ind in self.population)
        self.population = [ind for _, ind in sorted(zip(fitness_results, self.population), key=lambda x: x[0])]
        
        # Update best individual
        if self.population[0].mse < self.best_fitness:
            self.best_fitness = self.population[0].mse
            self.best_individual = self.population[0]
            return True
        else:
            return False
        

    def tournament_selection(self):
        """ Selects the best individuals for crossover & mutation """
        sorted_population = sorted(self.population, key=lambda x: x.mse)
        self.population = sorted_population[:self.tournament_size]
    
    def evolve_population(self, X, y):
        """ Applies crossover and mutation to generate new individuals """
        new_population = []
        while len(new_population) < int((self.population_size - len(self.population))*(1-self.randomness)):
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
                new_population.append(child)
            except:
                pass  # Skip invalid individuals

        self.population.extend(new_population)
        self.population = sorted(self.population, key=lambda x: x.mse)
    
    def fit(self, X, y):
        self.treeGp = TreeGP(X.shape[1], 15)
        stagnation_counter = 0
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}")
            self.generate_population_parallel(X, y)


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

            if stagnation_counter > 10 or (generation > 0 and generation % 10 == 0):
                print("Stagnation detected. Applying mutations...")
                num_to_mutate = int(self.tournament_size * 0.3)
                self.population = self.population[:num_to_mutate]
                stagnation_counter = 0
                print(f"Keep only {len(self.population)} individuals.")
            
            self.evolve_population(X, y)
        
        return self.best_individual


    def subtree_mutation(self, tree: Node) -> Node:
        """
        Esegue una mutazione di un sottoalbero.
        
        :param tree: Nodo radice dell'albero.
        :param max_depth: Profondità massima del nuovo sottoalbero generato.
        :return: Una copia dell'albero con un sottoalbero mutato.
        """
        # Crea una copia dell'albero per evitare modifiche in-place
        offspring = deepcopy(tree)
        
        successors = None
        while not successors:
            node = random.choice(list(offspring.subtree))
            successors = node.successors
        
        i = random.randrange(len(successors))
        depth = successors[i].depth
        new_subtree = self.treeGp.create_individual(depth)
        successors[i] = new_subtree
        node.successors = successors
        return offspring
    

    def hoist_mutation(self, tree: Node) -> Node:
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
        offspring = deepcopy(tree_node)

        successors = None
        while successors is None or len(successors) == 0:
            node = random.choice(list(offspring.subtree))
            successors = node.successors
        
        i = random.randrange(len(successors))
        node_to_mutate = successors[i]
        if node_to_mutate.is_leaf:
            new_node = Node(self.treeGp.random_terminal(), node_to_mutate.successors)
        else:
            new_op = None
            while new_op is None or arity(new_op) != node_to_mutate.arity:
                new_op = self.treeGp.random_operator()
            new_node = Node(new_op, node_to_mutate.successors)
        successors[i] = deepcopy(new_node)
        node.successors = successors

        return offspring


def compute_mse(individual: Node, X, Y, max_samples=None):
    indices = np.arange(len(X))
    if max_samples and max_samples < 1.0:
        indices = np.random.choice(indices, int(len(X) * max_samples), replace=False)
    X_batch = X[indices]
    Y_batch = Y[indices]

    y_pred = []
    for row in X_batch:
        variables = {f"x{i}": val for i, val in enumerate(row)}
        try:
            pred = individual(**variables)
            # Controllo per valori non numerici o estremi
            if not np.isfinite(pred) or np.abs(pred) > 1e20:
                pred = np.inf  # Penalizza modelli instabili
        except Exception as e:
            raise e
        y_pred.append(pred)

    y_pred = np.array(y_pred)

    # Controllo finale per y_pred
    if not np.all(np.isfinite(y_pred)):
        y_pred = np.nan_to_num(y_pred, nan=np.inf, posinf=np.inf, neginf=-np.inf)

    # Calcolo del MSE con controllo di validità
    try:
        mse = np.square(Y_batch - y_pred).mean()
        if not np.isfinite(mse) or mse > 1e20:  # Penalizza MSE instabile o troppo alto
            mse = 1e20
    except Exception:
        mse = 1e20  # Penalizza errori nel calcolo del MSE

    return mse

def simplify(node: Node) -> Node:
    """Simplifies the given tree by evaluating constant expressions and applying simplification rules."""
    if node.is_leaf:
        return node  # A leaf node cannot be further simplified.

    # Recursively simplify the successors
    simplified_successors = [simplify(child) for child in node.successors]

    # Check if all successors are numeric constants
    try:
        values = [float(child.short_name) for child in simplified_successors]
        # Prevent overflow/underflow
        clamped_values = [np.clip(val, -1e300, 1e300) for val in values]

        if node.short_name == 'np.sqrt' and clamped_values[0] < 0:
            print(f"Error: Attempted sqrt({clamped_values[0]}), which is invalid.")
            return Node(node='NaN')  # Optional: Handle by returning NaN or a placeholder
        
        if node.short_name == 'np.power' and clamped_values[0] < 0 and not float(clamped_values[1]).is_integer():
            print(f"Error: Attempted power({clamped_values[0]}, {clamped_values[1]}), which results in a complex number.")
            return Node(node='NaN')  # Optional: Return NaN or keep it unsimplified
        
        if node.short_name == 'np.divide' and clamped_values[1] == 0:
            print(f"Error: Attempted divide({clamped_values[0]}, 0), which is invalid.")
            return Node(node='inf')  # Optional: Handle gracefully

        result = node._func(*clamped_values) if node._func else None

        # Ensure the result is finite
        if result is not None and np.isfinite(result):
            return Node(node=result)  # Replace with a constant node
    except (ValueError, TypeError, ZeroDivisionError):
        pass

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
            return Node(node=0)  # If multiplying by 0, return 0
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
        # x / 0 -> 'inf' or handle gracefully
        if len(simplified_successors) == 2 and simplified_successors[1].short_name == '0':
            return Node(node='inf')  # Optional: Handle division by zero

    elif node.short_name == 'np.power':
        base, exp = simplified_successors
        if exp.short_name == '0':
            return Node(node=1)  # x ** 0 -> 1
        if exp.short_name == '1':
            return base  # x ** 1 -> x
        if base.short_name == '0' and float(exp.short_name) > 0:
            return Node(node=0)  # 0 ** x -> 0 (if x > 0)

    # If no simplifications applied, return new node with simplified successors
    return Node(node=node.short_name, successors=simplified_successors)