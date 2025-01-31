from symreg.sym2 import SymbolicRegressor
from symreg.sym2 import simplify
import numpy as np

problem = np.load("./data/problem_4.npz")
X = problem["x"].T
y = problem["y"]

print(X.shape, y.shape)

sr = SymbolicRegressor(stopping_criteria=0)
res = sr.fit(X, y)
print(simplify(res))