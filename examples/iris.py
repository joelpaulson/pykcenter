import torch
from sklearn.datasets import load_iris
from pykcenter.kcenter_bb import solve_branch_bound

X = load_iris(return_X_y=True)[0]
X = X[:, :4]
X = torch.tensor(X)    
X = X.T

k = 3
centers, UB, calcInfo = solve_branch_bound(X, k)