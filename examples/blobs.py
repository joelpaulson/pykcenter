import torch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pykcenter.kcenter_bb import solve_branch_bound

# generate data
n_samples = 1000
random_state = 42
k = 3
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=k)
X = torch.tensor(X)
X = X.T

# run branch and bound method
centers, UB, calcInfo = solve_branch_bound(X, k)

# calculate worst-case distance using assigned centers to verify calculated UB
dist_c = []
for c in centers.T:
  dist_c += [torch.sum((X - c.reshape((-1,1)))**2, dim=0)]
ind = torch.argmin(torch.stack(dist_c, dim=0), dim=0)
wc = torch.zeros(k)
for i in range(k):
  wc[i] = torch.max( torch.sum((X[:,ind==i].T - centers[:,i].T)**2, dim=1) )
wc_max = torch.max(wc)
print(f"The worst-case distance to all clusters is: {wc.detach().numpy()}")
print(f"The worst-case distance overall is: {wc_max.detach().numpy()}")

# plot the clusters
fig, ax = plt.subplots()
if X.shape[0] == 2:
  for i in range(k):
    X_i = X[:,ind==i]
    wc_samp = torch.argmax( torch.sum((X_i.T - centers[:,i].T)**2, dim=1) )
    ax.scatter(X[0,ind==i].numpy(), X[1,ind==i].numpy())
    ax.plot([centers[0,i], X_i[0,wc_samp].numpy()], [centers[1,i], X_i[1,wc_samp].numpy()], c='black', linewidth=1)
    # plt.scatter(X_i[0,wc_samp].numpy(), X_i[1,wc_samp].numpy(), c='yellow')
  ax.scatter(centers[0,:], centers[1,:], c='black', s=10)
  plt.show()