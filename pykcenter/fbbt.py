import torch
import random
import numpy  as np
from scipy.spatial import distance
from pykcenter.kcenter_ub import max_dist
from pykcenter.kcenter_ub import fft_FBBT

def select_init_seeds(X, k, UB):
    d, n = X.shape
    init_seeds = torch.zeros((d,k))
    init_seeds_ind = np.zeros(k, dtype=np.int32) # these samples are indexed starting from 0
    z = random.randint(0, n-1)  # randomly select a sample as the beginning
    init_seeds[:,0] = X[:,z]
    init_seeds_ind[0] = z # index corresponding to first cluster (we label cluster from 1 to k, so equal to index+1)

    diffclst = 1
    for j in range(1, k+1):
        _, a = max_dist(X.detach().numpy(), init_seeds[:, :j].detach().numpy())
        x = X[:, a]
        find_ = True
        for i in range(j):
            if torch.norm(x - init_seeds[:, i],2)**2 < 4*UB:
                find_ = False
                break
        if not find_:
            diffclst = 0
            break
        else:
            init_seeds[:, j-1] = x
            init_seeds_ind[j-1] = a

    return diffclst, init_seeds_ind

def select_seeds(X, k, UB, assign, n_trial=1):
    print("UB:   ", UB.detach().numpy())
    d, n = X.shape
    iter_ = 0
    diffclst = 0
    init_seeds_ind = []
    while iter_ <= 100 and diffclst == 0:
        diffclst, init_seeds_ind = select_init_seeds(X, k, UB)
        iter_ += 1
    print(f"try {iter_} times to select seeds")
    findseed = False
    if diffclst == 1:
        print("Find seeds successfully.")
        findseed = True
    else:
        print("Can not find seeds after 100 times.")

    seeds_ind = [[] for _ in range(k)]
    
    if diffclst == 1:
        for clst in range(1,k+1):
            seeds_ind[clst-1].append(init_seeds_ind[clst-1])
        seeds_ind = expand_seeds(X, k, UB, seeds_ind, n_trial)
        updateassign(assign, seeds_ind, k)
    
    return findseed

def updateassign(assign, seeds_ind, k):
    n = assign.size(0)
    for clst in range(1,k+1):
        for i in range(len(seeds_ind[clst-1])):
            ind = seeds_ind[clst-1][i]
            if assign[ind] != -1:
                assign[ind] = clst

def seedInAllCluster(seeds_ind):
    return all([len(seeds_ind[clst-1]) > 0 for clst in range(1,len(seeds_ind)+1)])

def expand_seeds(X, k, UB, seeds_ind, n_trial=1):
    d,n = X.shape
    check = seedInAllCluster(seeds_ind)
    if not check:
        print("error! need to have at least one seed per cluster first!")
        return None

    inner_assign = torch.zeros(n, dtype=torch.int64)
    seeds_mask = torch.zeros(n, k, dtype=torch.bool)
    for clst in range(1, k+1):
        seeds_mask[:, clst-1] = (torch.tensor(seeds_ind[clst-1]) == torch.arange(n).unsqueeze(1)).any(dim=1)
    inner_assign[seeds_mask.any(dim=1)] = torch.nonzero(seeds_mask, as_tuple=True)[1] + 1

    init_seeds = torch.zeros((d,k))
    for clst in range(1,k+1):
        init_seeds[:,clst-1] = X[:, seeds_ind[clst-1][0]]

    for t in range(1,n_trial+1):
        for s in range(n):
            if inner_assign[s] == 0:
                includ = torch.zeros(k, dtype=torch.bool)
                x = X[:,s]
                for clst in range(1,k+1):
                    d = distance.euclidean(x, init_seeds[:,clst-1])**2
                    if d < 4*UB:
                        includ[clst-1] = True
                if includ.sum() == 1:
                    c = torch.nonzero(includ)[0][0] + 1
                    seeds_ind[c-1].append(s)
                    inner_assign[s] = c

        for clst in range(1,k+1):
            ix = random.randint(0,len(seeds_ind[clst-1])-1)
            init_seeds[:,clst-1] = X[:, seeds_ind[clst-1][ix]]

    num = sum(len(seeds_ind[clst]) for clst in range(0,k))
    print("Expand #  seeds ", num)
    return seeds_ind

def divideclusternorm(X, UB, k, lower, upper, assign, center_cand, max_nseeds_c = 20):
    d, n = X.size()
    lwr = lower.clone()
    upr = upper.clone()
    for clst in range(1,k+1):
        old_center_cand_clst = center_cand[:, clst-1]
        oldset = torch.nonzero(old_center_cand_clst).flatten()

        seeds_ind_c =  (assign == clst).nonzero().flatten()   #seeds_ind[clst]

        if len(seeds_ind_c) <= max_nseeds_c:
            nseed_c = len(seeds_ind_c)
            seeds_id = seeds_ind_c[:nseed_c]
            seeds =  X[:, seeds_id]
        else:         
            seeds = fft_FBBT(X[:, seeds_ind_c], max_nseeds_c, lower[:,clst-1], upper[:,clst-1])
        dmat = torch.cdist(seeds.T.detach(), X[:, oldset].T.detach(), p=2).pow(2)

        num = 0
        center_cand[:, clst-1] = False
        for j in range(len(oldset)):
            s = oldset[j]
            if (X[:, s] >= lower[:, clst-1]).sum() == d and (X[:, s] <= upper[:, clst-1]).sum() == d:
                if (dmat[:, j] > UB).sum() == 0: #(dmat[:, j] - tol > UB).sum()
                    center_cand[s, clst-1] = True    
                    num += 1
                    if num == 1:
                        lwr[:, clst-1] = X[:, s]
                        upr[:, clst-1] = X[:, s]
                    else:
                        for i in range(d):
                            if lwr[i, clst-1] > X[i, s]:
                                lwr[i, clst-1] = X[i, s]
                            if upr[i, clst-1] < X[i, s]:
                                upr[i, clst-1] = X[i, s]

        # If there is at least one candidate center, update the lower and upper bounds
        if num > 0:
            s = torch.nonzero(center_cand[:, clst-1], as_tuple=True)[0][0]
            lwr[:, clst-1] = X[:, s]
            upr[:, clst-1] = X[:, s]
            if num > 1:
                lwr[:, clst-1] = torch.min(lwr[:, clst-1], X[:, center_cand[:, clst-1]].min(dim=1).values)
                upr[:, clst-1] = torch.max(upr[:, clst-1], X[:, center_cand[:, clst-1]].max(dim=1).values)

        if num == 0:
            return None, None
        elif num == 1:    
            s = torch.nonzero(center_cand[:, clst-1]).flatten()[0]
            if assign[s] == 0:
                assign[s] = clst
          
    return lwr, upr

def fbbt_base(X, k, node, UB, max_nseeds_c=20):
    d, n = X.shape
    assign = node.assign
    center_cand = node.center_cand

    lwr = node.lower.clone()
    upr = node.upper.clone()
    ra = np.sqrt(UB)

    mask = (assign != 0) & (assign != -1)
    clst = assign[mask] - 1
    X_ = X[:, mask]
    lwr_ = lwr[:, clst]
    upr_ = upr[:, clst]    
    lwr_diff = lwr_ < X_ - ra
    lwr_mask = lwr_diff.any(dim=1)
    upr_diff = upr_ > X_ + ra
    upr_mask = upr_diff.any(dim=1)
    lwr_[lwr_mask] = (X_ - ra)[lwr_mask]
    upr_[upr_mask] = (X_ + ra)[upr_mask]
    lwr[:, clst] = lwr_
    upr[:, clst] = upr_

    for clst in range(1,k+1):
        if torch.sum(lwr[:, clst-1] <= upr[:, clst-1]) != d:
            print("Delete this node")  # intersection is empty, delete this node
            return None, None

    lwr, upr = divideclusternorm(X, UB, k, lwr, upr, assign, center_cand, max_nseeds_c)  # given three seeds, divide the dataset according to distance<=UB
    if lwr is None and upr is None:
        print("Delete this node")  # intersection is empty, delete this node
        return None, None

    return lwr, upr
