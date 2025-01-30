from symreg.sym2 import SymbolicRegressor
from symreg.sym2 import simplify
import numpy as np

problem = np.load("./data/problem_6.npz")
X = problem["x"].T
y = problem["y"]

print(X.shape, y.shape)

sr = SymbolicRegressor()
res = sr.fit(X, y)
print(simplify(res))