from symreg.symbolic_regressor import SymbolicRegressor
import numpy as np

problem = np.load("./data/problem_3.npz")
X = problem["x"].T
y = problem["y"]

print(X.shape, y.shape)

sr = SymbolicRegressor(max_depth=5, stopping_criteria=0, generations=10)
res = sr.fit(X, y)
print(res.simplify())