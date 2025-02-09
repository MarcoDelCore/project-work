from gxgp.utils import Operator
from symreg.symbolic_regressor import SymbolicRegressor
import numpy as np

n = 4
numpy_operators = {
    "np.add": Operator(np.add, 2),
    "np.subtract": Operator(np.subtract, 2),
    "np.multiply": Operator(np.multiply, 2),
    "np.divide": Operator(np.divide, 2),
    "np.power": Operator(np.power, 2),
    "np.exp": Operator(np.exp, 1),
    "np.sqrt": Operator(np.sqrt, 1),
    "np.sin": Operator(np.sin, 1),
    "np.cos": Operator(np.cos, 1),
    "np.tan": Operator(np.tan, 1),
    "np.log": Operator(np.log, 1),
    "np.abs": Operator(np.abs, 1),
    "np.negative": Operator(np.negative, 1),
    "np.reciprocal": Operator(np.reciprocal, 1),
    "np.square": Operator(np.square, 1),
    "np.cbrt": Operator(np.cbrt, 1),
    "np.log1p": Operator(np.log1p, 1),
    "np.expm1": Operator(np.expm1, 1),
    "np.sinh": Operator(np.sinh, 1),
    "np.cosh": Operator(np.cosh, 1),
    "np.tanh": Operator(np.tanh, 1),
    "np.arcsin": Operator(np.arcsin, 1),
    "np.arccos": Operator(np.arccos, 1),
    "np.arctan": Operator(np.arctan, 1),
    "np.arcsinh": Operator(np.arcsinh, 1),
    "np.arccosh": Operator(np.arccosh, 1),
    "np.arctanh": Operator(np.arctanh, 1),
    "np.maximum": Operator(np.maximum, 2),
    "np.minimum": Operator(np.minimum, 2),
}

problem = np.load(f"./data/problem_{n}.npz")
X = problem["x"]
y = problem["y"]

print(f"Solving problem {n}...")
print(f"X shape: {X.shape}")

sr = SymbolicRegressor(operators=numpy_operators)
res = sr.fit(X, y)
print()
print("Training complete.")
print(f"Besto solution: {sr.simplify(res)}")