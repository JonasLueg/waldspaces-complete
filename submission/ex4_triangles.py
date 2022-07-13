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
import operator
import itertools as it
import matplotlib.pyplot as plt
import matplotlib as mpl

# # DANGER ALERT: this works only, when this file is in the folder "/examples/distances_leaves_3/example1.py"
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from treespaces.tools.wald import Wald
from treespaces.tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import WaldSpace
import treespaces.visualization.tools_plot as ptools

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 3
partition = ((0, 1, 2),)
sp0 = Split(n=n, part1=(0,), part2=(1, 2))
sp1 = Split(n=n, part1=(1,), part2=(0, 2))
sp2 = Split(n=n, part1=(2,), part2=(0, 1))
split_collection = [[sp0, sp1, sp2]]

folder = ntpath.splitext(ntpath.basename(__file__))[0]
data_folder = os.path.join(folder, "data")

try:
    os.mkdir(path=folder)
except FileExistsError:
    pass

try:
    os.mkdir(path=data_folder)
except FileExistsError:
    pass

st = Structure(n=n, partition=partition, split_collection=split_collection)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

val = 0.35
coordinates = [[1, val, val], [val, 1, val], [val, val, 1]]
p, q, s = tuple(Wald(n=n, st=st, x=_x) for _x in coordinates)

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
ws = WaldSpace(geometry='wald')

# ----- COMPUTATION OF THE PATHS FROM THE CONSTRUCTED WALDS TO THE BOUNDARY (STAR STRATUM) ----- #
n_points_on_path = 17
proj_args = {'gtol': 10 ** -9, 'ftol': 10 ** -9, 'btol': 10 ** -8}
params = {'n_points': n_points_on_path, 'method': 'global-straighten-extend', 'proj_args': proj_args}

# ----- THIS FLAG DETERMINES IF THE GEODESICS ARE COMPUTED AGAIN OR LOADED FROM DATA FILES INSTEAD -----
try:
    catalogue = pickle.load(file=open(os.path.join(data_folder, 'catalogue.p'), "rb"))
except FileNotFoundError:
    catalogue = dict()

# del catalogue[(p, q, s)]

if (p, q, s) in catalogue:  # in this case, no need to compute anything, load the already computed paths
    paths = catalogue[(p, q, s)]
else:
    print(f"Calculate the geodesics.")
    start = time.time()
    paths = tuple([tuple(ws.g.s_path(p, q, **params)),
                   tuple(ws.g.s_path(q, s, **params)),
                   tuple(ws.g.s_path(s, p, **params))])
    params['method'] = 'global-path-straightening'
    params['n_iter'] = 10
    print(f"Straighten the geodesics.")
    paths = tuple([tuple(ws.g.s_path(_path[0], _path[-1], start_path=_path, **params)) for _path in paths])
    end = time.time()
    print(f"Calculation took {np.round(end - start, 2)} seconds.")
    print(f"Store the data that has been calculated.")
    catalogue[(p, q, s)] = paths
    pickle.dump(obj=catalogue, file=open(os.path.join(data_folder, 'catalogue.p'), "wb"))


def angle(u, v, _p):
    u = ws.g.s_lift_vector(u, _p)
    v = ws.g.s_lift_vector(v, _p)
    _p = ws.g.s_lift(_p)
    _dummy = ws.g.a_inner(u, v, _p) / ws.g.a_norm(u, _p) / ws.g.a_norm(v, _p)
    return 360 / 2 / np.pi * np.arccos(_dummy)


def angles(path_pq, path_qs, path_sp):
    u = path_pq[1].corr - path_pq[0].corr
    v = path_sp[-2].corr - path_sp[-1].corr
    _a = angle(u=u, v=v, _p=path_pq[0])
    u = path_pq[-2].corr - path_pq[-1].corr
    v = path_qs[1].corr - path_qs[0].corr
    _b = angle(u=u, v=v, _p=path_qs[0])
    u = path_qs[-2].corr - path_qs[-1].corr
    v = path_sp[1].corr - path_sp[0].corr
    _c = angle(u=u, v=v, _p=path_sp[0])
    return _a, _b, _c


def angle_sum(path_pq, path_qs, path_sp):
    _a, _b, _c = angles(path_pq, path_qs, path_sp)
    return np.round((_a + _b + _c), 2)


# sort the order of the triangle according to their angle sum.
catalogue = sorted(catalogue.items(), key=lambda item: angle_sum(*item[1]))
# -------------------- plot all triangles with angle sum in the legend -------------------------- #
fig = ptools.Plot3Embedded()
colors = [next(fig.ax._get_lines.prop_cycler)['color'] for _ in catalogue]
for i, (((p, q, s), paths), color) in enumerate(zip(catalogue, colors)):
    fig.pass_point_family(points=[p, q, s], color=color, marker='.')
    fig.pass_curve_family(curves=paths, color=color)
    if p.x[2] < 0.7:
        fig.pass_point(point=p, marker='', text=r"$\lambda_u=$" + f"{np.round(p.x[2], 2)}",
                       offset=(-0.15, -0.15, -0.02), textcolor=color)
fig.ax.view_init(elev=32., azim=-45)
fig.savefig(fn=os.path.join(folder, "triangles_embedded.pdf"))
fig.close()

fig = plt.figure()
xs = [p.x[2] for (p, q, s), paths in catalogue]
ys = [angle_sum(*paths) for (p, q, s), paths in catalogue]
plt.scatter(xs, ys, color=colors)
plt.xlim((0, 1))
plt.xlabel(r"$\lambda_u$,\hspace{0.5cm}$u = 1,2,3$.")
plt.ylabel(r"angle sum in $~^\circ$")
plt.savefig(fname=os.path.join(folder, "triangles_angle_sums.pdf"))
plt.close()
