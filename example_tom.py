import sys
import numpy as np
import os
import pickle
import time
import ntpath

from treespaces.framework.waldspace import WaldSpace
from treespaces.tools.wald import Wald, Tree
from treespaces.tools.structure_and_split import Split, Structure

import treespaces.visualization.tools_plot as ptools
import treespaces.framework.tools_numerical as ntools
import treespaces.framework.tools_io as iotools


folder_name = ntpath.splitext(ntpath.basename(__file__))[0]
try:
    os.mkdir(path=folder_name)
except FileExistsError:
    pass


n = 3
partition = ((0, 1, 2),)
split_collection = [set([Split(n=n, part1=[i], part2=set(list(range(n))).difference({i})) for i in range(n)])]
st = Structure(n=n, partition=partition, split_collection=split_collection)

wald1 = Wald(n=n, st=st, x=[0.05, 0.2, 0.9])
wald2 = Wald(n=n, st=st, x=[0.9, 0.05, 0.05])

n_points = 10

# affine - invariant stuff is computed here.
ws = WaldSpace(geometry='affine-invariant')
proj_args = {'method': 'local', 'btol': 10**-10}
af_start_path = ws.g.s_path(p=wald1, q=wald2, alg='naive', n_points=n_points, proj_args=proj_args)
af_straightened_path = ws.g.s_path(p=wald1, q=wald2, alg='straightening-ext', start_path=af_start_path, proj_args=proj_args,
                                   n_iter=10)
print(ntools.energy(af_start_path, geometry=ws.g))
print(ntools.energy(af_straightened_path, geometry=ws.g))


# euclidean stuff is computed here.
ws = WaldSpace(geometry='euclidean')
proj_args = {'method': 'local', 'btol': 10**-10}
euc_start_path = ws.g.s_path(p=wald1, q=wald2, alg='naive', n_points=n_points, proj_args=proj_args)
euc_straightened_path = ws.g.s_path(p=wald1, q=wald2, alg='straightening-ext', start_path=euc_start_path, proj_args=proj_args,
                                    n_iter=10)
print(ntools.energy(euc_start_path, geometry=ws.g))
print(ntools.energy(euc_straightened_path, geometry=ws.g))


# euclidean stuff is computed here.
ws = WaldSpace(geometry='bhv')
bhv_path = ws.g.s_path(p=wald1, q=wald2, n_points=n_points)
print(ntools.energy(bhv_path, geometry=ws.g))

for i, p in enumerate(af_straightened_path):
    ptools.plot_wald(wald=p, root='a', fn=os.path.join(folder_name, f"af1_path_{iotools.f00(i)}.png"))

for i, p in enumerate(bhv_path):
    ptools.plot_wald(wald=p, root='a', fn=os.path.join(folder_name, f"bhv_path_{iotools.f00(i)}.png"))

for i, p in enumerate(euc_straightened_path):
    ptools.plot_wald(wald=p, root='a', fn=os.path.join(folder_name, f"euc_path_{iotools.f00(i)}.png"))

# plot everything.
# fig = ptools.Plot3Dims(alpha=0.3)
# fig.pass_curves([af_start_path, euc_start_path], labels=["af symmetric", "euc symmetric"])
# fig.pass_curves([af_straightened_path, euc_straightened_path], labels=["af straightened", "euc straightened"])
# fig.pass_curves([bhv_path], labels=["bhv geodesic"])
# fig.show()
