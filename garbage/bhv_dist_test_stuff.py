import scipy.linalg as la
import numpy as np

from treespaces.spaces.treespace_bhv import TreeSpaceBhv

from tools.structure_and_split import Split, Structure
from treespaces.tools.wald import Tree

n = 4
s0 = Split(n=n, part1=[0], part2=[1, 2, 3])
s1 = Split(n=n, part1=[1], part2=[0, 2, 3])
s2 = Split(n=n, part1=[2], part2=[0, 1, 3])
s3 = Split(n=n, part1=[3], part2=[0, 1, 2])
sm1 = Split(n=n, part1=[0, 1], part2=[2, 3])
sm2 = Split(n=n, part1=[0, 2], part2=[1, 3])
sm3 = Split(n=n, part1=[0, 3], part2=[1, 2])


bhv = TreeSpaceBhv

splits1 = sorted([(s0, 1), (s1, 4), (s2, 2), (s3, 10), (sm1, 1)])
splits2 = sorted([(s0, 4), (s1, 9), (s2, 7), (s3, 1), (sm2, 4)])

t1, t2 = tuple(Tree(n=n, splits=[s[0] for s in sp], b=np.array([s[1] for s in sp])) for sp in [splits1, splits2])

n_points = 100
path_ = bhv.s_path(p=t1, q=t2, n_points=100)

for p, q in zip(path_[:5], [t1.dist + t*(t2.dist - t1.dist) for t in np.linspace(0, 1, n_points)][:5]):
    print("-"*10)
    print(p.dist)
    print(q)

# for p in []
