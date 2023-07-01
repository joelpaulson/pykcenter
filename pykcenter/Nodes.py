class Node:
    def __init__(self, lower=None, upper=None, level=-1, LB=-1e15, assign=None, center_cand=None):
        self.lower = lower
        self.upper = upper
        self.level = level
        self.LB = LB
        self.assign = assign
        self.center_cand = center_cand