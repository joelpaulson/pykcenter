## imports
import torch
import time
from pykcenter.branch import select_var_max_range
from pykcenter.branch import branch
from pykcenter.fbbt import select_seeds
from pykcenter.fbbt import fbbt_base
from pykcenter.kcenter_lb import getLowerBound_analytic_basic_FBBT
from pykcenter.kcenter_lb import getGlobalLowerBound
from pykcenter.kcenter_opt import sel_closest_centers
from pykcenter.kcenter_opt import obj_assign
from pykcenter.kcenter_opt import init_bound
from pykcenter.kcenter_ub import getUpperBound
from pykcenter.Nodes import Node


### helper functions
def time_finish(seconds): # function to record the finish time point
    return int(10**9 * seconds + time.time_ns())

def getUnionBound(nodeList):
    lower = None
    upper = None    
    for idx, nd in enumerate(nodeList):
        if idx == 0:
            lower = nd.lower.clone()
            upper = nd.upper.clone()
        else:
            lower = torch.min(lower, nd.lower)
            upper = torch.max(upper, nd.upper)
    return lower, upper

def can_center_or_assign(assign, center_cand):
    n = assign.size(0)
    zeros_mask = (assign == -1) & (center_cand.sum(dim=1) == 0)
    remain = torch.ones(n, dtype=torch.bool)
    remain[zeros_mask] = False
    return remain

def randomUB(X, lwr, upr, UB, centers, ntr=5):
    ctr = centers
    d, k = lwr.size()
    t_centers = (lwr+upr)/2
    for tr in range(1, ntr+1):
        t_ctr = sel_closest_centers(t_centers, X)
        t_UB = obj_assign(t_ctr, X)
        if (t_UB < UB):
            UB = t_UB
            ctr = t_ctr
        if tr !=ntr:
            t_centers = torch.rand(d, k, device=X.device) * (upr - lwr) + lwr
    return ctr, UB


### function to perform the entire branch and bound process
def solve_branch_bound(X, k, symmtrc_breaking=0, maxiter=5000000, tol=1e-6, mingap=1e-3, time_lapse=4*3600):
  # scale the data
  x_max = torch.max(X) # max value after transfer to non-zero value
  tnsf_max = False
  if x_max > 20:
      tnsf_max = True
      X = X/(x_max*0.05)

  # create root node
  d, n = X.size()
  lower, upper = init_bound(X, d, k)
  max_LB = 1e15 # used to save the best lower bound at the end (smallest but within the mingap)
  centers, UB = getUpperBound(X, k, lower, upper, tol)
  root = Node(lower, upper, -1, -1e15, torch.zeros(n, dtype=torch.int64), torch.ones((n, k), dtype=torch.bool))

  # start seed process
  findseeds = select_seeds(X, k, UB, root.assign, 10)
  if not findseeds:
    symmtrc_breaking = 1
  else:
    oldboxSize = torch.sum(upper - lower)  # torch.norm(upper - lower, p=2)**2 / k
    print("sum:   ", oldboxSize.detach().numpy())
    oldUB = UB
    root_LB = -1e15
    stuck = 0
    for t in range(1, 41):
        print("trial      ", t, "   fbbt ")
        root.lower, root.upper = fbbt_base(X, k, root, UB, 50)
        boxSize = torch.sum(root.upper - root.lower)  # torch.norm(root.upper - root.lower, p=2)**2 / k
        print("sum:   ", boxSize.detach().numpy())
        centers, UB = randomUB(X, root.lower, root.upper, UB, centers, 10)
        # print("UB  ", UB)
        root_LB = getLowerBound_analytic_basic_FBBT(X, k, root, UB)
        root.LB = root_LB

        if boxSize / oldboxSize >= 0.99 and UB / oldUB >= 0.999:
            stuck += 1
        else:
            stuck = 0
        if stuck == 2:
            break
        oldboxSize = boxSize
        oldUB = UB

        if (UB - root_LB) <= mingap * min(abs(root_LB), abs(UB)):
            Gap = (UB - root_LB) / min(abs(root_LB), abs(UB))
            print("LB   ", root_LB.detach().numpy(), "  UB  ", UB.detach().numpy(), " Gap ", Gap.detach().numpy())

            ctr_dist = torch.cdist(centers.T, centers.T, p=2)**2
            min_ctr_dist = ctr_dist[0, 1]
            for i in range(k):
                for j in range(k):
                    if i != j and ctr_dist[i, j] < min_ctr_dist:
                        min_ctr_dist = ctr_dist[i, j]
            rate = min_ctr_dist / UB
            print("rate = minimum distance between centers/UB =   ", rate.detach().numpy())

            # transfer back to original value of optimal value
            if tnsf_max:
                UB = UB * (x_max * 0.05) ** 2

            print("UB:  ", UB.detach().numpy())

            return centers, UB, None

        remain = can_center_or_assign(root.assign, root.center_cand)
        print("# of elements deleted", n - torch.sum(remain).detach().numpy())
        print("# of elements remain", torch.sum(remain).detach().numpy())
        if torch.sum(remain) / n <= 0.8:
            X = X[:, remain]
            root.assign = root.assign[remain]
            root.center_cand = root.center_cand[remain, :]
            n = torch.sum(remain)

  # groups is not initalized, will generate at the first iteration after the calculation of upper bound
  nodeList = [root]
  iter = 0
  print("iter ", " left ", " lev  ", "       LB       ", "       UB      ", "      gap   ")

  # get program end time point
  end_time = time_finish(time_lapse) # the branch and bound process ends after 6 hours

  #####inside main loop##################################
  calcInfo = [] # initial space to save calcuation information
  while nodeList:
      # we start at the branch(node) with lowest Lower bound
      LB, nodeid = getGlobalLowerBound(nodeList) # Here the LB is the best LB and also node.LB of current iteration
      node = nodeList[nodeid]
      del nodeList[nodeid]
      
      # so currently, the global lower bound corresponding to node, LB = node.LB, groups = node.groups
      if iter%10 == 0:
          print("%-6d %-6d %-10d %-16.4f %-14.4f %-10.4f %s" % (iter, len(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))*100, "%"))
      
      # save calcuation information for result demostration
      calcInfo.append([iter, len(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])
      
      # time stamp should be checked after the retrival of the results
      if (iter == maxiter) or (time.time_ns() >= end_time):
          break

      ############# iteratively bound tightening #######################
      iter += 1   

      # use FBBT
      node.lower, node.upper = fbbt_base(X, k, node, UB)
      if node.lower is None and node.upper is None:
          continue

      node_LB = LB
      ##### LB ###############
      # println("LB:  ")
      # getLowerBound with closed-form expression
      if (UB - node_LB) <= mingap * min(abs(node_LB), abs(UB)):
          print(f"analytic LB {node_LB} >=UB {UB}")
      else:
        # use FBBT
        node_LB = getLowerBound_analytic_basic_FBBT(X, k, node, UB)    
      ##### UB ####################################
      ##### get upper bound from random centers
      # println("UB:  ")
      oldUB = UB
      centers, UB = randomUB(X, node.lower, node.upper, UB, centers)
      if (UB < oldUB):
          # the following code delete branch with lb close to the global upper bound
          delete_nodes = []
          for idx, nd in enumerate(nodeList):
              if (UB - nd.LB) <= mingap * torch.min(torch.abs(UB), torch.abs(nd.LB)):
                  delete_nodes.append(idx)
          for idx in sorted(delete_nodes, reverse=True):
              del nodeList[idx]

      if iter % 50 == 0:
          remain = None
          print("root node reduce size")
          if len(nodeList) >= 500:
              root.lower, root.upper = getUnionBound(nodeList)
              for t in range(2):
                  # print("trial ", t, " fbbt ")
                  root.lower, root.upper = fbbt_base(X, k, root, UB, 50)
                  boxSize = torch.sum(root.upper-root.lower)  # np.linalg.norm(root.upper-root.lower, ord=2)**2/k
                  print("sum of root: ", boxSize.detach().numpy())
                  root.LB = max(root.LB, LB)
                  root_LB = getLowerBound_analytic_basic_FBBT(X, k, root, UB)
                  # print("root LB: ", root.LB)
                  root.LB = max(root.LB, root_LB)

                  if (UB-root_LB) <= mingap*min(abs(root_LB), abs(UB)):
                      print("LB ", root_LB.detach().numpy(), " UB ", UB.detach().numpy())
                      # transfer back to original value of optimal value
                      if tnsf_max:
                          UB = UB * (x_max*0.05)**2
                      print("UB: ", UB.detach().numpy())

                      return centers, UB, None
                  remain = can_center_or_assign(root.assign, root.center_cand)
                  # print("# of elements deleted", n - np.sum(remain))
                  # print("# of elements remain", np.sum(remain))
              else:
                  for idx, nd in enumerate(nodeList):
                      node_remain = can_center_or_assign(nd.assign, nd.center_cand)
                      if idx == 0:
                          remain = node_remain
                      else:
                          remain = (remain | node_remain)
                      node_remain = can_center_or_assign(node.assign, node.center_cand)
                      remain = (remain | node_remain)
                      # print("node: ", idx, " LB ", nd.LB, " level ", nd.level, " # of elements deleted ", n - np.sum(node_remain))
                      # print("# of elements deleted", n - np.sum(remain))

              if torch.sum(remain) / n <= 0.8:
                  print(n - torch.sum(remain).detach().numpy(), " of elements deleted")
                  print(torch.sum(remain).detach().numpy(), " of elements remain")
                  n = torch.sum(remain)
                  X = X[:, remain]
                  root.assign = root.assign[remain]
                  root.center_cand = root.center_cand[remain, :]
                  for idx, nd in enumerate(nodeList):
                      nd.assign = nd.assign[remain]
                      nd.center_cand = nd.center_cand[remain, :]
                  node.assign = node.assign[remain]
                  node.center_cand = node.center_cand[remain, :]

      if (UB - node_LB) <= mingap * min(abs(node_LB), abs(UB)):
          if node_LB < max_LB:
              max_LB = node_LB
      else:
          bVarIdx, bVarIdy = select_var_max_range(node)
          bValue = (node.upper[bVarIdx, bVarIdy] + node.lower[bVarIdx, bVarIdy]) / 2
          branch(X, nodeList, bVarIdx, bVarIdy, bValue, node, node_LB, k, symmtrc_breaking)

  if not nodeList:
      print("all node solved")
      # save final calcuation information
      calcInfo.append([iter, len(nodeList), max_LB, UB, (UB-max_LB)/min(abs(max_LB), abs(UB))])
  else:
      max_LB = calcInfo[-1][3]
      print(max_LB)
  print("solved nodes:  ",iter)

  print("{:<25d}{:<14.4e}{:<20.4e}{:<8.4f}{} \n".format(iter, max_LB, UB, (UB-max_LB)/min(abs(max_LB),abs(UB))*100, "%"))
  print("centers = ")
  print(f"{centers.detach().numpy()}")

  ctr_dist = torch.cdist(centers.T, centers.T, p=2)**2
  min_ctr_dist = ctr_dist[0,1]
  for i in range(k):
      for j in range(k):
          if i != j and ctr_dist[i,j] < min_ctr_dist:
              min_ctr_dist = ctr_dist[i,j]
  rate = min_ctr_dist/UB
  print("rate = minimum distance between centers/UB =   ", rate.detach().numpy())

  # transfer back to original value of optimal value
  if tnsf_max:
      UB = UB * (x_max*0.05)**2

  print("UB:  ", UB.detach().numpy())

  return centers, UB, calcInfo