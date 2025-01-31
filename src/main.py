from symreg.symbolic_regressor import SymbolicRegressor
from symreg.symbolic_regressor import simplify
import numpy as np

problem = np.load("./data/problem_4.npz")
X = problem["x"].T
y = problem["y"]

print(X.shape, y.shape)

sr = SymbolicRegressor()
res = sr.fit(X, y)
print(simplify(res))