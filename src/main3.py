from symreg.symbolic_regressor import SymbolicRegressor
from symreg.symbolic_regressor import simplify
import numpy as np

problem = np.load("./data/problem_5.npz")
X = problem["x"].T
y = problem["y"]

print(X.shape, y.shape)

sr = SymbolicRegressor(max_depth=3, parsimony_coefficient=0, stopping_criteria=0)
res = sr.fit(X, y)
print(simplify(res))