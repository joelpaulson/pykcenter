import torch
from pykcenter.fbbt import expand_seeds
from pykcenter.fbbt import updateassign

def getGlobalLowerBound(nodeList): # if LB same, choose the first smallest one
    LB = 1e15
    nodeid = 1
    for idx, n in enumerate(nodeList):
        #println("remaining ", idx,  "   ", n.LB)
        if n.LB < LB:
            LB = n.LB
            nodeid = idx
    return LB, nodeid

def getLowerBound_analytic_basic_FBBT(X, k, node, UB):
    d, n = X.shape
    lwr = node.lower
    upr = node.upper
    assign = node.assign
    firstcheck = seedInAllCluster_LB(assign,k)  # If every cluster has seeds, firstcheck = true.
    addSeed = False

    LB_s_array, all_LB_s_array = getLB_FBBT(X, k, d, lwr, upr, assign) # changed to be vectorized
    all_LB_s_array = all_LB_s_array.T
    condition = (assign == 0)
    all_LB_s_array_filtered = all_LB_s_array[:, condition]
    includ = all_LB_s_array_filtered <= UB
    includ_count = torch.sum(includ, dim=0)
    condition1 = (includ_count == 1)
    indices = torch.nonzero(includ_count == 1)
    if condition1.any():
        c = indices[0][0]
        assign[condition][condition1] = c + 1
        addSeed = True

    node_LB = torch.tensor(node.LB) if not isinstance(node.LB, torch.Tensor) else node.LB
    LB = torch.max(torch.max(LB_s_array), node_LB)

    UB_S, all_UB_S = getUB(X, k, d, lwr, upr, assign) # changed to be vectorized so it is faster
    # check condition for assigning values
    condition = (assign != -1)    
    # filter the relevant tensors based on the condition
    UB_s_filtered = UB_S[condition]
    all_UB_s_filtered = all_UB_S[condition, :]
    all_LB_s_array_filtered = all_LB_s_array[:, condition]
    # check UB_s condition and update assign
    condition_1 = (UB_s_filtered < LB)
    condition_2 = (assign[condition] == 0)
    assign[condition][condition_1] = -1
    min_LB_s, ind = torch.min(all_LB_s_array_filtered, dim=0)
    all_LB_s_array_filtered[ind] = 1e16
    condition_3 = (all_UB_s_filtered[torch.arange(0,UB_s_filtered.shape[0]).unsqueeze(-1), ind.unsqueeze(-1)].squeeze() <= torch.min(all_LB_s_array_filtered, dim=0)[0])
    assign[condition][condition_2 & condition_3] = ind[condition_2 & condition_3] + 1    

    if addSeed:
        secondcheck = seedInAllCluster_LB(assign, k)
        if not firstcheck and secondcheck:
            seeds_ind = haveSeedInAllCluster_LB(assign, k)
            seeds_ind = expand_seeds(X, k, UB, seeds_ind)
            assign = updateassign(assign, seeds_ind, k)
            print("expand in LB")

    return torch.max(LB_s_array)

def seedInAllCluster_LB(assign, k):
    non_zero_idx = torch.nonzero((assign != 0) & (assign != -1), as_tuple=False).squeeze(1)
    covered = torch.unique(assign[non_zero_idx]).size(0)
    return covered == k

def haveSeedInAllCluster_LB(assign, k):
    mask = (assign != 0) & (assign != -1)
    seeds_ind = torch.nonzero(mask, as_tuple=False)[:, 0]
    seeds_clst = assign[mask] - 1
    seeds = [[] for clst in range(k)]
    for i in range(len(seeds_ind)):
        seeds[seeds_clst[i]].append(seeds_ind[i])
    return seeds

def getLB_FBBT(V, k, d, lwr, upr, clst_assigned):
    # V:    d x n
    # lwr:  d x k
    # upr:  d x k
    n = V.shape[1]
    mask1 = clst_assigned == -1
    mask2 = clst_assigned == 0
    mask3 = ~(mask1 | mask2)
    best_LB = torch.zeros(n).to(V)
    all_LB = torch.zeros((n,k)).to(V)
    if mask1.any():
        all_LB[mask1,:] = torch.nan
    if mask2.any():
        V_expanded = V[:,mask2].unsqueeze(1).repeat((1, k, 1)) # d x k x n_mask2
        lwr_expanded = lwr.unsqueeze(-1).repeat(1, 1, V_expanded.shape[-1]) # d x k x n_mask2
        upr_expanded = upr.unsqueeze(-1).repeat(1, 1, V_expanded.shape[-1]) # d x k x n_mask2
        mu_expanded = torch.median(torch.stack([lwr_expanded, V_expanded, upr_expanded], dim=3), dim=3)[0]
        all_LB[mask2,:] = (torch.norm(mu_expanded - V_expanded, p=2, dim=0)**2).T
        best_LB[mask2], _ = torch.min(all_LB[mask2,:], dim=1)
    if mask3.any():
        mu = torch.median(torch.stack([lwr[:,clst_assigned[mask3]-1], V[:,mask3], upr[:,clst_assigned[mask3]-1]], dim=0), dim=0)[0]
        all_LB[mask3,:] = torch.nan
        best_LB[mask3] = (torch.norm(mu - V[:,mask3], p=2, dim=0)**2)
    return best_LB, all_LB

def getUB(V, k, d, lwr, upr, clst_assigned):
    n = V.shape[1]
    mask1 = clst_assigned == -1
    mask2 = clst_assigned == 0
    mask3 = ~(mask1 | mask2)
    best_UB = torch.zeros(n).to(V)
    all_UB = torch.zeros((n,k)).to(V)
    if mask1.any():
        all_UB[mask1,:] = torch.nan
    if mask2.any():
        V_expanded = V[:,mask2].unsqueeze(1).repeat((1, k, 1))
        lwr_expanded = lwr.unsqueeze(-1).repeat(1, 1, V_expanded.shape[-1])
        upr_expanded = upr.unsqueeze(-1).repeat(1, 1, V_expanded.shape[-1])
        mu_expanded = UB_sol(lwr_expanded, V_expanded, upr_expanded)
        all_UB[mask2,:] = (torch.norm(mu_expanded - V_expanded, p=2, dim=0)**2).T
        best_UB[mask2], _ = torch.min(all_UB[mask2,:], dim=1)
    if mask3.any():
        mu = UB_sol(lwr[:,clst_assigned[mask3]-1], V[:,mask3], upr[:,clst_assigned[mask3]-1])
        all_UB[mask3,:] = torch.nan
        best_UB[mask3] = (torch.norm(mu - V[:,mask3], p=2, dim=0)**2)
    return best_UB, all_UB

def UB_sol(l, x, u):
    return torch.where(torch.abs(x - l) > torch.abs(x - u), l, u)
