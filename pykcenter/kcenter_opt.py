import torch
from scipy.spatial.distance import cdist

def obj_assign(centers, X):
    d, n = X.shape
    dmat = torch.cdist(X.to(torch.float64).T, centers.to(torch.float64).T, p=2)**2
    costs, _ = torch.min(dmat, dim=1)
    return torch.max(costs)

def init_bound(X, d, k, lower=None, upper=None):
    lower_data = torch.min(X, axis=1).values
    upper_data = torch.max(X, axis=1).values
    lower_data = lower_data.repeat(k, 1).T
    upper_data = upper_data.repeat(k, 1).T

    if lower is None:
        lower = lower_data
        upper = upper_data
    else:
        lower = torch.max(lower, lower_data)
        upper = torch.min(upper, upper_data)
        lower = torch.max(lower, upper-1e-4)
        upper = torch.min(upper, lower+1e-4)

    return lower, upper

def sel_closest_centers(centers, X):
    d, k = centers.shape
    t_ctr = torch.zeros(d, k)
    dmat = torch.cdist(X.t(), centers.t(), p=2)**2
    for j in range(k):
        a = torch.argmin(dmat[:,j])
        t_ctr[:,j] = X[:,a]
    return t_ctr