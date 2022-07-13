"""
Test the different algorithms for computing geodesics for three leaves with histograms for depicting the entropy of the
different discretized paths.
"""

import os
import sys
import pickle
import time
import ntpath
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

# DANGER ALERT: this works only, when this file is in the folder "/examples/distances_leaves_3/example1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 3
partition = ((0, 1, 2),)
split_collection = [[Split(n=n, part1=(0,), part2=(1, 2)),
                     Split(n=n, part1=(1,), part2=(0, 2)),
                     Split(n=n, part1=(2,), part2=(0, 1))]]

name_handle = ntpath.splitext(ntpath.basename(__file__))[0]

st = Structure(n=n, partition=partition, split_collection=split_collection)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

# the weights are associated according to this order of the splits: ((0, 2)|(1,), (0, 1)|(2,), (0,)|(1, 2))
coordinates = [[0.95, 0.95, 0.97], [0.97, 0.95, 0.95], [0.95, 0.97, 0.95]]
coordinates = tuple([tuple(_x) for _x in coordinates])
walds = [Wald(n=n, st=st, x=_x) for _x in coordinates]

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')

# ----- COMPUTATION OF THE PATHS FROM THE CONSTRUCTED WALDS TO THE BOUNDARY (STAR STRATUM) ----- #
n_points_on_path = 25
n_iter_straightening = 8
# these are the parameters used for every projection from ambient space onto the grove
proj_args = {'gtol': 10 ** -12, 'ftol': 10 ** -15, 'btol': 10 ** -5}
params = {'n_points': n_points_on_path, 'n_iter': n_iter_straightening, 'proj_args': proj_args}

method = 'straightening-ext'  # ['naive', 'symmetric', 'variational', 'straightening-ext']
names = {'naive': 'NP', 'symmetric': 'SSP', 'straightening-ext': 'EPS', 'straightening-int': 'IPS'}

# ----- THIS FLAG DETERMINES IF THE GEODESICS ARE COMPUTED AGAIN OR LOADED FROM DATA FILES INSTEAD -----
try:
    x_dict = pickle.load(file=open(os.path.join(f"{name_handle}_data", f"x_dict.p"), "rb"))
except FileNotFoundError:
    x_dict = {}

m = len(x_dict)
# look up the x_dict if for those coordinates the triangle was already computed once
catches = [_x for _x in x_dict.keys() if set(_x) == set(coordinates)]

if catches:  # in this case, no need to compute anything, load the already computed paths
    print(f"Load the geodesics.")
else:
    print(f"Calculate the geodesics with the {method} method.")
    start = time.time()
    paths = tuple([tuple(ws.g.s_path(p=walds[0], q=walds[1], alg=method, **params)),
                   tuple(ws.g.s_path(p=walds[1], q=walds[2], alg=method, **params)),
                   tuple(ws.g.s_path(p=walds[2], q=walds[0], alg=method, **params))])
    end = time.time()
    print(f"Calculation took {np.round(end - start, 2)} seconds.")
    print(f"Store the data that has been calculated.")
    pickle.dump(obj=paths, file=open(os.path.join(f"{name_handle}_data", f"data_{m}.p"), "wb"))
    x_dict[coordinates] = paths
    pickle.dump(obj=x_dict, file=open(os.path.join(f"{name_handle}_data", f"x_dict.p"), "wb"))


def angle_sum(path_pq, path_qs, path_sp):
    a_u = (path_pq[0].x - path_pq[1].x) / ws.g.s_norm(v=path_pq[0].x - path_pq[1].x, p=path_pq[0])
    a_v = (path_sp[-1].x - path_sp[-2].x) / ws.g.s_norm(v=path_sp[-1].x - path_sp[-2].x, p=path_sp[-1])
    a = np.arccos(ws.g.s_inner(u=a_u, v=a_v, p=path_pq[0]))

    b_u = (path_pq[-1].x - path_pq[-2].x) / ws.g.s_norm(v=path_pq[-1].x - path_pq[-2].x, p=path_pq[-1])
    b_v = (path_qs[0].x - path_qs[1].x) / ws.g.s_norm(v=path_qs[0].x - path_qs[1].x, p=path_qs[0])
    b = np.arccos(ws.g.s_inner(u=b_u, v=b_v, p=path_qs[0]))

    c_u = (path_qs[-1].x - path_qs[-2].x) / ws.g.s_norm(v=path_qs[-1].x - path_qs[-2].x, p=path_qs[-1])
    c_v = (path_sp[0].x - path_sp[1].x) / ws.g.s_norm(v=path_sp[0].x - path_sp[1].x, p=path_sp[0])
    c = np.arccos(ws.g.s_inner(u=c_u, v=c_v, p=path_sp[0]))
    return np.round((a + b + c)/2/np.pi*360, 4)


x_dict = pickle.load(file=open(os.path.join(f"{name_handle}_data", f"x_dict.p"), "rb"))
m = len(x_dict)
# -------------------- plot all triangles with angle sum in the legend -------------------------- #
sp0, sp1, sp2 = 2, 0, 1

fig = plt.figure()
ax = fig.gca(projection='3d')
# prepare the axes
ax.set_xlim(0.0, 1.0)
ax.set_xlabel(r"$\lambda_0$, split: " + repr(st.split_collection[0][sp0]))
ax.set_ylim(0.0, 1.0)
ax.set_ylabel(r"$\lambda_1$, split: " + repr(st.split_collection[0][sp1]))
ax.set_zlim(0.0, 1.0)
ax.set_zlabel(r"$\lambda_2$, split: " + repr(st.split_collection[0][sp2]))
# draw cube
r = [0, 1]
for s, e in it.combinations(np.array(list(it.product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == (r[1] - r[0]):
        ax.plot(*zip(s, e), color="black", lw=1)


for i in range(m):
    paths = pickle.load(file=open(os.path.join(f"{name_handle}_data", f"data_{i}.p"), "rb"))
    triangle_path = paths[0] + paths[1] + paths[2]
    print(triangle_path)
    xs = [x.x[sp0] for x in triangle_path]
    ys = [x.x[sp1] for x in triangle_path]
    zs = [x.x[sp2] for x in triangle_path]
    ax.plot(xs=xs, ys=ys, zs=zs, lw=1, label=f"angle sum = {angle_sum(paths[0], paths[1], paths[2])}")

ax.legend()
plt.show()
plt.savefig(os.path.join(f"{name_handle}_png", f"{name_handle}_cube_with_triangles.png"), dpi=200)
plt.clf()
plt.cla()
plt.close(fig)
