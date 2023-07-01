from pykcenter.Nodes import Node
import torch

def select_var_max_range(node):
    dif = node.upper - node.lower
    ind = torch.argmax(dif)
    return ind // dif.shape[1], ind % dif.shape[1]

def branch(X, node_list, bVarIdx, bVarIdy, bValue, node, node_LB, k, symmtrc_breaking=0):
    d, n = X.shape
    lower = node.lower.clone()
    upper = node.upper.clone()
    upper[bVarIdx, bVarIdy] = bValue # split from this variable at bValue
    if symmtrc_breaking == 1:
        for j in range(1, k): # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
            if upper[0, k-j-1] >= upper[0, k-j]:
                upper[0, k-j-1] = upper[0, k-j]
    if torch.sum(lower <= upper) == d * k:
        left_node = Node(lower, upper, node.level+1, node_LB, node.assign.clone(), node.center_cand.clone())
        node_list.append(left_node)
    
    lower = node.lower.clone()
    upper = node.upper.clone()
    lower[bVarIdx, bVarIdy] = bValue
    if symmtrc_breaking == 1:
        for j in range(2, k+1): # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
            if lower[0, j-1] <= lower[0, j-1-1]:
                lower[0, j-1] = lower[0, j-1-1]
    if torch.sum(lower <= upper) == d * k:
        right_node = Node(lower, upper, node.level+1, node_LB, node.assign.clone(), node.center_cand.clone())
        node_list.append(right_node)