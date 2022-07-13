import sys
import numpy as np

import treespaces.framework.waldspace

from treespaces.tools.structure_and_split import Split
from treespaces.tools.wald import Tree

n = 3
s0 = Split(n=n, part1=[0], part2=[1, 2])
s1 = Split(n=n, part1=[1], part2=[0, 2])
s2 = Split(n=n, part1=[2], part2=[0, 1])

splits = {s0, s1, s2}

b1 = [0.2, 0.2, 4]
b2 = [4, 0.2, 0.2]

tree1 = Tree(n=n, splits=splits, b=b1)
tree2 = Tree(n=n, splits=splits, b=b2)

print(f"Compute the means between the trees")
print(f"Tree 1 with structure {tree1.st} and edge lengths {tree1.b} or edge weights {tree1.x} and ")
print(f"tree 2 with structure {tree2.st} and edge lengths {tree2.b} or edge weights {tree2.x}.")

ws1 = treespaces.framework.waldspace.WaldSpace(geometry="bhv")
mean_bhv = ws1.g.s_path_t(p=tree1, q=tree2, t=0.5)

print(f"The mean in BHV space is ")
print(f"Tree with structure {mean_bhv.st} and edge lengths {mean_bhv.b} or edge weights {mean_bhv.x}.")

ws2 = treespaces.framework.waldspace.WaldSpace(geometry="waldspace")

proj_args = {'btol': 10**-10, 'gtol': 10**-10, 'ftol': 10**-15}
mean_wald = ws2.g.s_path(p=tree1, q=tree2, alg="straightening-ext", n_points=11, n_iter=1,
                         proj_args=proj_args)[5]

print(f"The mean in wald space is ")
print(f"Tree with structure {mean_wald.st} and edge lengths {mean_wald.b} or edge weights {mean_wald.x}.")

print("The correlation between leaf 0 and 2 is ")
print(f"Tree1: {(1 - tree1.x[0])*(1 - tree1.x[2])}.")
print(f"Tree2: {(1 - tree2.x[0])*(1 - tree2.x[2])}.")
print(f"In wald space,  mean: {(1 - mean_wald.x[0])*(1 - mean_wald.x[2])}.")
print(f"In BHV space, mean  : {(1 - mean_bhv.x[0])*(1 - mean_bhv.x[2])}.")

print("The correlation between leaf 0 and 1 is ")
print(f"Tree1: {(1 - tree1.x[0])*(1 - tree1.x[1])}.")
print(f"Tree2: {(1 - tree2.x[0])*(1 - tree2.x[1])}.")
print(f"In wald space,  mean: {(1 - mean_wald.x[0])*(1 - mean_wald.x[1])}.")
print(f"In BHV space, mean  : {(1 - mean_bhv.x[0])*(1 - mean_bhv.x[1])}.")

print("The correlation between leaf 1 and 2 is ")
print(f"Tree1: {(1 - tree1.x[1])*(1 - tree1.x[2])}.")
print(f"Tree2: {(1 - tree2.x[1])*(1 - tree2.x[2])}.")
print(f"In wald space,  mean: {(1 - mean_wald.x[1])*(1 - mean_wald.x[2])}.")
print(f"In BHV space, mean  : {(1 - mean_bhv.x[1])*(1 - mean_bhv.x[2])}.")
