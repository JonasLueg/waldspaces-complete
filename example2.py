
import numpy as np

import treespaces.framework.waldspace

from treespaces.tools.structure_and_split import Split, Structure
from treespaces.tools.wald import Tree

n = 4
s0 = Split(n=n, part1=[0], part2=[1, 2, 3])
s1 = Split(n=n, part1=[1], part2=[0, 2, 3])
s2 = Split(n=n, part1=[2], part2=[0, 1, 3])
s3 = Split(n=n, part1=[3], part2=[0, 1, 2])
sm1 = Split(n=n, part1=[0, 1], part2=[2, 3])
sm2 = Split(n=n, part1=[0, 2], part2=[1, 3])
sm3 = Split(n=n, part1=[0, 3], part2=[1, 2])


ws = treespaces.framework.waldspace.TreeSpdAf1(geometry="waldspace")
n_points = 11

for s in [sm1, sm2, sm3]:
    splits = {s0, s1, s2, s3, s}
    st = Structure(n=n, partition=(tuple(i for i in range(n)),), split_collection=(tuple(splits),))

    start_path = []
    for t in np.linspace(0, 1, n_points):
        x = [0.2 + 0.5 * t]*len(splits)
        x[st.where(s)] = t * (1 - t)
        start_path.append(Tree(n=n, splits=splits, x=x))

    print(f"Start to calculate the shortest path between trees with edge "
          f"weights {start_path[0].x} and {start_path[-1].x}, respectively, and both with the same structure")
    print(f"{start_path[0].st}.")

    proj_args = {'btol': 10 ** -7, 'gtol': 10 ** -15, 'ftol': 10 ** -15, 'method': 'global-descent'}

    shortest_path = ws.g.s_path(p=start_path[0], q=start_path[-1], alg="straightening-ext",
                                n_points=n_points, n_iter=5, proj_args=proj_args)

    print(f"The shortest path is then:")
    for p in shortest_path:
        print(p.x)
