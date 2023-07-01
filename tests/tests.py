import torch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import sys
sys.path.append('../pykcenter')
from pykcenter.kcenter_bb import solve_branch_bound

# generate data
n_samples = 1000
random_state = 42
k = 3
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=k)
X = torch.tensor(X)
X = X.T

# run profiling to see how much each function costs
import cProfile
cProfile.run('solve_branch_bound(X, k)')