from gxgp.utils import Operator
from symreg.symbolic_regressor import SymbolicRegressor
from symreg.symbolic_regressor import simplify
import numpy as np

n = 2
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
            Operator("np.expm1", np.expm1, 1),
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
        ]

problem = np.load(f"./data/problem_{n}.npz")
X = problem["x"]
y = problem["y"]

print(f"Solving problem {n}...")
print(f"X shape: {X.shape}")

sr = SymbolicRegressor(operators=numpy_operators, population_size=20, tournament_size=8, generations=50, stopping_criteria=0)
res = sr.fit(X, y)
print()
print("Training complete.")
print(f"Besto solution: {simplify(res)}")