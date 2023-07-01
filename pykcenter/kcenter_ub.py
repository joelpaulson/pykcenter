import torch
import random
from scipy.spatial.distance import cdist
from torch.nn.functional import pairwise_distance
from pykcenter.kcenter_opt import obj_assign
from pykcenter.kcenter_opt import sel_closest_centers

def max_dist(X1, X2):
    # max distance from array X1 to array X2
    dist_matrix = cdist(X1.T, X2.T)
    dist = dist_matrix.min(axis=1)
    c, a = dist.max(), dist.argmax()
    return c, a  # c is the max distance, a is the index.

def fft(X, k):
    d, n = X.size()
    centers = torch.zeros(d, k, device=X.device)
    z0 = random.randint(0,n-1)
    centers[:, 0] = X[:, z0]
    dist = torch.empty(n, device=X.device)
    dist_matrix = torch.empty(n, 1, device=X.device)
    for j in range(2, k+1):
        if j == 2:
            dist = pairwise_distance(X.t(), centers[:, j-2].view(1, -1))
        else:
            dist_matrix = pairwise_distance(X.t(), centers[:, j-2].view(1, -1))
            dist = torch.min(dist, dist_matrix.view(-1))
        _, a = torch.max(dist, dim=0)
        centers[:, j-1] = X[:, a]
    return centers

def fft_FBBT(X, k, lower, upper):
    d, n = X.size()
    centers = torch.zeros(d, k+1)
    centers[:, 0] = (lower+upper)/2
    dist = torch.empty(n)
    dist_matrix = torch.empty(n, 1)
    for j in range(2, k+2):
        if j == 2:
            dist = ((X - centers[:, j-2].unsqueeze(1))**2).sum(dim=0)
        else:
            pairwise_distances = ((X - centers[:, j-2].unsqueeze(1))**2).sum(dim=0)
            pairwise_distances = pairwise_distances.unsqueeze(1)
            pairwise_distances = torch.cat([pairwise_distances, dist_matrix[:, 0:]], dim=1)
            dist = pairwise_distances.min(dim=1).values
        _, a = torch.max(dist, dim=0)
        centers[:, j-1] = X[:, a]
    return centers[:, 1:].to(torch.float64)

def getUpperBound(X, k, lower, upper, tol=0):
    UB = float('inf')
    centers = None
    for tr in range(100):
        t_ctr = fft(X, k)
        t_UB = obj_assign(t_ctr, X)
        if tol <= UB - t_UB:
            UB = t_UB
            centers = t_ctr
    d, n = X.shape
    t_centers = (lower + upper) / 2
    t_ctr = sel_closest_centers(t_centers, X)
    for tr in range(100):
        t_UB = obj_assign(t_ctr, X)
        if tol <= UB - t_UB:
            UB = t_UB
            centers = t_ctr
        inds = torch.randperm(n)[:k]
        t_ctr = X[:, inds]
    return centers, UB