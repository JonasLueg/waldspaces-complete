"""
Test the different algorithms for computing geodesics for three leaves with histograms for depicting the entropy of the
different discretized paths.
"""

import os
import sys
import ntpath
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# DANGER ALERT: this works only, when this file is in the folder "/examples/distances_leaves_3/example1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1

# ----- THIS FLAG DETERMINES IF THE GEODESICS ARE COMPUTED AGAIN OR LOADED FROM DATA FILES INSTEAD -----
calculate = True

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 3
partition = ((0, 1, 2),)
split_collection = [[Split(n=n, part1=(0,), part2=(1, 2)),
                     Split(n=n, part1=(1,), part2=(0, 2)),
                     Split(n=n, part1=(2,), part2=(0, 1))]]

name_handle = ntpath.splitext(ntpath.basename(__file__))[0]

st = Structure(n=n, partition=partition, split_collection=split_collection)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")
# ((0, 2)|(1,), (0, 1)|(2,), (0,)|(1, 2))

# the weights are associated according to this order of the splits: ((0, 2)|(1,), (0, 1)|(2,), (0,)|(1, 2))
coordinates = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]]

wald1 = Wald(n=n, st=st, x=np.array(coordinates[0]))
wald2 = Wald(n=n, st=st, x=np.array(coordinates[1]))

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')


# ----- DEFINITION OF THE ENTROPY OF A GEODESICS DISTRIBUTION OF POINTS ALONG THE PATH ----- #


def entropy(_path, _ideal_value):
    """ Measures how equidistant the points in _path are. Zero means they are perfectly equidistant. """
    dists = [ws.g.a_dist(p=_path[_i], q=_path[_i + 1], squared=False) for _i in range(0, len(_path) - 1)]
    return np.sum([x * np.log(x / _ideal_value) for x in dists])


# ----- COMPUTATION OF THE PATHS FROM THE CONSTRUCTED WALDS TO THE BOUNDARY (STAR STRATUM) ----- #
n_points_on_path = 20
n_iter_straightening = 15
params = {}
# these are the parameters used for every projection from ambient space onto the grove
proj_args = {'gtol': 10 ** -12, 'ftol': 10 ** -15, 'btol': 10 ** -6}
# the parameters for each method are set here:
params['straightening-ext'] = {'n_points': n_points_on_path, 'n_iter': n_iter_straightening, 'proj_args': proj_args}

# if we want to calculate new, do it now

print(f"\nCalculate the geodesics with the straightening extrinsic method.")
path = tuple(ws.g.s_path(p=wald1, q=wald2, alg='straightening-ext', **params))


# ----- PLOT THE CALCULATED DATA ----- #
sp0, sp1, sp2 = 2, 0, 1

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim(0.0, 1.0)
ax.set_xlabel(r"$\lambda_1$")
ax.set_ylim(0.0, 1.0)
ax.set_ylabel(r"$\lambda_2$")
ax.set_zlim(0.0, 1.0)
ax.set_zlabel(r"$\lambda_3$")
# draw cube
r = [0, 1]
for s, e in it.combinations(np.array(list(it.product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == (r[1] - r[0]):
        ax.plot3D(*zip(s, e), color="black", lw=1)

xs = [x.x[sp0] for x in path]
ys = [x.x[sp1] for x in path]
zs = [x.x[sp2] for x in path]
ax.plot(xs=xs, ys=ys, zs=zs, lw=1, label="shortest path")
ax.scatter(xs=0.1, ys=0.1, zs=0.9, label="(0.1, 0.1, 0.9)")
ax.scatter(xs=0.1, ys=0.9, zs=0.1, label="(0.1, 0.9, 0.1)")

plt.legend()
plt.show()
