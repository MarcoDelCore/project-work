from copy import deepcopy
import random
from gxgp.gp_common import xover_swap_subtree
from gxgp import TreeGP, Node
import numpy as np

from gxgp.utils import arity


class SymbolicRegressor():
    __population = []
    __fitness    = float('inf')
    __best_indiv = None

    def __init__(self, 
                 population_size=1000,
                 generations=200,
                 tournament_size=100,
                 stopping_criteria=0.00001,
                 max_depth=7,
                 parsimony_coefficient=0.01,
                 p_crossover=0.7,
                 p_subtree_mutation=0.1,
                 p_hoist_mutation=0.1,
                 p_point_mutation=0.1,
                 max_samples=1.0,):
        
        self.__population_size = population_size
        self.__generations = generations
        self.__tournament_size = tournament_size
        self.__stopping_criteria = stopping_criteria
        self.__max_depth = max_depth
        self.__parsimony_coefficient = parsimony_coefficient
        self.__p_crossover = p_crossover
        self.__p_subtree_mutation = p_subtree_mutation
        self.__p_hoist_mutation = p_hoist_mutation
        self.__p_point_mutation = p_point_mutation
        self.__max_samples = max_samples

        print('SymbolicRegressor initialized')
    
    def generate_population(self, X, y):
        self.treeGp = TreeGP(X.shape[1], 15)
        depths = list(range(2, self.__max_depth + 1))
        random.shuffle(depths) 
        while len(self.__population) < self.__population_size:
            depth = depths[len(self.__population) % len(depths)]
            individual = self.treeGp.create_individual(depth)
            is_valid = True

            try:
                mse = compute_mse(individual, X, y, max_samples=self.__max_samples)
                individual.set_mse(mse)
            except Exception as e:
                is_valid = False

            if is_valid:
                self.__population.append(individual)
        return self.__population

    def fit(self, X, y):
        """
        Addestra il modello sul dataset fornito.
        :param X: Array di input (dataset).
        :param y: Array di output (valori target).
        :return: La formula ottenuta dal train.
        """
        self.treeGp = TreeGP(X.shape[1], 15)
        stagnation_counter = 0
        
        for generation in range(self.__generations):
            print(f"\nGeneration {generation + 1}")

            self.generate_population(X, y)

            self.__population.sort(key=lambda x: x.mse)
            # Aggiorna il miglior individuo
            current_mse = self.__population[0].mse
            if current_mse < self.__fitness:
                self.__fitness = current_mse
                self.__best_indiv = self.__population[0]
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            print(f"Best MSE: {self.__fitness}")
            print(f"Best Individual: {self.__best_indiv}")

            # Arresta se la fitness raggiunge il criterio di stopping
            if self.__fitness <= self.__stopping_criteria:
                print("Stopping criteria met. Training complete.")
                break

            if generation == self.__generations - 1:
                print("Training complete. Maximum generations reached.")
                break
            # Riavvia parzialmente la popolazione in caso di stagnazione
            if stagnation_counter > 10 or (generation > 0 and generation % 10 == 0):
                print("Stagnation detected. Resetting part of the population...")
                num_to_reset = int(self.__population_size * 0.3)
                for _ in range(num_to_reset):
                    depth = random.randint(2, self.__max_depth)
                    new_individual = self.treeGp.create_individual(depth)
                    self.__population.append(new_individual)
                stagnation_counter = 0

            if self.__fitness < 10e6:
                # Seleziona la popolazione per il torneo
                self.__population = self.__population[:self.__tournament_size]

                # Genera una nuova popolazione
                new_population = []
                while len(new_population) < self.__population_size * 0.6:
                    r = random.random()
                    if r < self.__p_crossover:
                        parent1, parent2 = random.sample(self.__population, 2)
                        child = xover_swap_subtree(parent1, parent2)
                    elif r < self.__p_crossover + self.__p_subtree_mutation:
                        parent = random.choice(self.__population)
                        child = self.subtree_mutation(parent)
                    elif r < self.__p_crossover + self.__p_subtree_mutation + self.__p_hoist_mutation:
                        parent = random.choice(self.__population)
                        child = self.hoist_mutation(parent)
                    elif r < self.__p_crossover + self.__p_subtree_mutation + self.__p_hoist_mutation + self.__p_point_mutation:
                        parent = random.choice(self.__population)
                        child = self.point_mutation(parent)
                    else:
                        # Aggiungi mutazione puntuale di riserva
                        parent = random.choice(self.__population)
                        child = self.point_replace(parent)
                    mse = compute_mse(child, X, y)
                    child.set_mse(mse)
                    new_population.append(child)

                self.__population.extend(new_population)
            else:
                print("Skipping crossover and mutation due to high MSE")
                self.__population = []

        return self.__best_indiv


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

