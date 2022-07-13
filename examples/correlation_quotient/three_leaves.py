"""
Compare the correlation quotient geometry with the original affine-invariant geometry.
"""

import os
import pickle
import time
import ntpath
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tools.wald import Tree
from tools.structure_and_split import Split
import treespaces.framework.tools_numerical as numtools

# import all the different geometries
from treespaces.spaces.treespace_corr_quotient import TreeSpaceCorrQuotient
from treespaces.spaces.treespace_spd_af1 import TreeSpdAf1
from treespaces.spaces.treespace_bhv import TreeSpaceBhv
from treespaces.spaces.treespace_spd_euclidean import TreeSpdEuclidean

# ----- THIS FLAG DETERMINES IF THE GEODESICS ARE COMPUTED AGAIN OR LOADED FROM DATA FILES INSTEAD -----
calculate = True
name_handle = ntpath.splitext(ntpath.basename(__file__))[0]

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 3
sp0 = Split(n=n, part1=(0,), part2=(1, 2))
sp1 = Split(n=n, part1=(1,), part2=(0, 2))
sp2 = Split(n=n, part1=(2,), part2=(0, 1))
splits = [sp0, sp1, sp2]

# the weights are associated according to this order of the splits: ((0, 2)|(1,), (0, 1)|(2,), (0,)|(1, 2))
coordinates = [[0.1, 0.9, 0.07], [0.4, 0.05, 0.8]]

# construct all the walds using the class Tree for convenience
walds = [Tree(n=n, splits=splits, x=_x) for _x in coordinates]
# the respective wald structure for all the walds
st = walds[0].st

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
geometries = [TreeSpdAf1, TreeSpaceBhv, TreeSpaceCorrQuotient, TreeSpdEuclidean]
geometries_names = ["Affine-Invariant", "BHV-Space", "Correlation-Quotient", "Euclidean"]
colors = ['red', 'blue', 'orange', 'black']  # , 'green', 'black']
# ----- COMPUTATION OF THE PATHS FROM THE CONSTRUCTED WALDS TO THE BOUNDARY (STAR STRATUM) ----- #
n_points_on_path = 21
n_iter_straightening = 10

# these are the parameters used for every projection from ambient space onto the grove
projection_args = {'gtol': 10 ** -12, 'ftol': 10 ** -15, 'btol': 10 ** -8, 'method': 'global-descent'}
algorithm_args = {'n_points': n_points_on_path, 'n_iter': n_iter_straightening, 'proj_args': projection_args}


# if we want to calculate new, do it now
paths = {}
times = {}
start_paths = {}
if calculate:
    for g, g_name in zip(geometries, geometries_names):
        print(f"\nCalculate the geodesics with respect to the geometry {g_name}.")
        start = time.time()
        start_paths[g_name] = g.s_path(p=walds[0], q=walds[1], alg="global-symmetric", **algorithm_args)
        paths[g_name] = tuple(g.s_path(p=walds[0], q=walds[1], start_path=start_paths[g_name],
                                       alg="global-straightening", **algorithm_args))
        end = time.time()
        times[g_name] = np.round(end - start, 2)
        print(f"Calculation took {times[g_name]} seconds.")
        print(f"Store the data that has been calculated.")
        pickle.dump(obj=(paths[g_name], times[g_name]),
                    file=open(os.path.join(f"{name_handle}", f"data_{g_name}.p"), "wb"))
else:
    print(f"Load the already stored data since the flag 'calculate' is False.")
    for g_name in geometries_names:
        paths[g_name], times[g_name] = pickle.load(file=open(os.path.join(f"{name_handle}", f"data_{g_name}.p"), "rb"))


# ----- PLOT THE CALCULATED DATA ----- #
print("Start constructing the plots.".center(80, "-"))

print("1) Plot the paths in the 3d cube.".center(60, "-"))
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

for g_name, color in zip(geometries_names, colors):
    _path = paths[g_name]
    print([_p.st for _p in _path])
    print(f"{st.where(sp0)}, {st.where(sp1)}, {st.where(sp2)}")
    xs = [_p.x[st.where(sp0)] for _p in _path]
    ys = [_p.x[st.where(sp1)] for _p in _path]
    zs = [_p.x[st.where(sp2)] for _p in _path]
    ax.scatter(xs=xs, ys=ys, zs=zs, lw=1, label=g_name, color=color, alpha=0.5)

# plot p
ax.text(walds[0].x[st.where(sp0)], walds[0].x[st.where(sp1)], walds[0].x[st.where(sp2)], r"$W_1$", color='black')
ax.text(walds[1].x[st.where(sp0)], walds[1].x[st.where(sp1)], walds[1].x[st.where(sp2)], r"$W_2$", color='black')

# remove multiple labels
# (from https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend)
handles, labels = plt.gca().get_legend_handles_labels()
i = 1
while i < len(labels):
    if labels[i] in labels[:i]:
        del (labels[i])
        del (handles[i])
    else:
        i += 1

plt.legend(handles, labels)
plt.savefig(os.path.join(f"{name_handle}", f"plot_geodesics_3d.png"), dpi=200)
plt.show()
plt.clf()
plt.cla()
plt.close(fig)

print("2) Histograms of distance between pair-wise consecutive points.".center(60, "-"))
lengths = {g_name: np.round(g.length(path_=paths[g_name]), 6) for g, g_name in zip(geometries, geometries_names)}
# NOW PLOT THE HISTOGRAMS OF THE GEODESICS 1:
for g, g_name, color in zip(geometries, geometries_names, colors):
    path = paths[g_name]
    try:
        data = [g.a_dist(p=path[_i], q=path[_i + 1]) for _i in range(0, len(path) - 1)]
    except NotImplementedError:
        data = [g.s_dist(p=path[_i], q=path[_i + 1]) for _i in range(0, len(path) - 1)]
    plt.hist(x=data, color=color, label=f"{g_name}: length={lengths[g_name]}, time={times[g_name]}s", alpha=0.2)
    plt.legend()
    plt.title(f"Distances between consecutive points")
    plt.savefig(os.path.join(f"{name_handle}", f"{g_name}_plot_consecutive_distances.png"), dpi=200)
    plt.clf()
    plt.cla()
    plt.close(fig)

for g, g_name in zip(geometries, geometries_names):
    print(f"The length in geometry {g_name} is {g.length(paths[g_name])}.")
    print(f"The entropy in geometry {g_name} is {numtools.entropy(_path=paths[g_name], geometry=g)}.")
    print(f"Time needed in geometry {g_name} is {times[g_name]}.")
    print(f"The energy of the method {g_name} is {numtools.energy(_path=paths[g_name], geometry=g)}.")
