from symreg.symbolic_regressor import SymbolicRegressor
from symreg.symbolic_regressor import simplify
import numpy as np

n = 6

problem = np.load(f"./data/problem_{6}.npz")
X = problem["x"].T
y = problem["y"]

print(f"Solving problem {n}...")
print(X.shape, y.shape)

sr = SymbolicRegressor(stopping_criteria=0.0001, randomness=0.6)
res = sr.fit(X, y)
print(simplify(res))