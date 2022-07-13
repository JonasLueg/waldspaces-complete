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
from mpl_toolkits.mplot3d import Axes3D

# DANGER ALERT: this works only, when this file is in the folder "/examples/distances_leaves_3/example1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1
import treespaces.framework.tools_numerical as numtools

# ----- THIS FLAG DETERMINES IF THE GEODESICS ARE COMPUTED AGAIN OR LOADED FROM DATA FILES INSTEAD -----
calculate = True

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 3
partition = ((0, 1, 2),)
sp0 = Split(n=n, part1=(0,), part2=(1, 2))
sp1 = Split(n=n, part1=(1,), part2=(0, 2))
sp2 = Split(n=n, part1=(2,), part2=(0, 1))
split_collection = [[sp0, sp1, sp2]]

name_handle = ntpath.splitext(ntpath.basename(__file__))[0]

st = Structure(n=n, partition=partition, split_collection=split_collection)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

# the weights are associated according to this order of the splits: ((0, 2)|(1,), (0, 1)|(2,), (0,)|(1, 2))
coordinates = [[0.1, 0.9, 0.07], [0.4, 0.05, 0.8]]

walds = [Wald(n=n, st=st, x=np.array(_x)) for _x in coordinates]

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
geometry = "euclidean"
name_handle += "_euclidean"
ws = TreeSpdAf1(geometry=geometry)

# ----- COMPUTATION OF THE PATHS FROM THE CONSTRUCTED WALDS TO THE BOUNDARY (STAR STRATUM) ----- #
n_points_on_path = 15
n_iter_variational = 5
n_iter_straightening = 5
params = {}
# these are the parameters used for every projection from ambient space onto the grove
proj_args = {'gtol': 10 ** -12, 'ftol': 10 ** -15, 'btol': 10 ** -8, 'method': 'global-descent'}
# the parameters for each method are set here:
params['naive'] = {'n_points': n_points_on_path, 'proj_args': proj_args}
params['symmetric'] = {'n_points': n_points_on_path, 'proj_args': proj_args}
# params['variational'] = {'n_points': n_points_on_path, 'n_iter': n_iter_variational, 'proj_args': proj_args}
params['straightening-ext'] = {'n_points': n_points_on_path, 'n_iter': n_iter_straightening, 'proj_args': proj_args}
params['straightening-int'] = {'n_points': n_points_on_path, 'n_iter': n_iter_straightening, 'max_step': 1,
                               'atol': 10 ** -2}

# just take all methods that have their parameters defined in the dictionary params
methods = params.keys()  # ['naive', 'symmetric', 'variational', 'straightening-ext']
names = {'naive': 'NP', 'symmetric': 'SP', 'straightening-ext': 'EPS', 'straightening-int': 'IPS'}

# if we want to calculate new, do it now
paths = {}
times = {}
if calculate:
    for method in methods:
        print(f"\nCalculate the geodesics with the {method} method.")
        start = time.time()
        paths[method] = tuple(ws.g.s_path(p=walds[0], q=walds[1], alg=method, **params[method]))
        end = time.time()
        times[method] = np.round(end - start, 2)
        print(f"Calculation took {times[method]} seconds.")
        print(f"Store the data that has been calculated.")
        pickle.dump(obj=(paths[method], times[method]),
                    file=open(os.path.join(f"{name_handle}_data", f"{name_handle}_data_points_{method}.p"), "wb"))
else:
    print(f"Load the already stored data since the flag 'calculate' is False.")
    for method in methods:
        paths[method], times[method] = pickle.load(file=open(os.path.join(f"{name_handle}_data",
                                                                          f"{name_handle}_data_points_{method}.p"),
                                                             "rb"))

# ----- PLOT THE CALCULATED DATA ----- #

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

colors = ['red', 'blue', 'green', 'orange', 'black']
colors = {method: colors[i] for i, method in enumerate(methods)}

for method in methods:
    _path = paths[method]
    print([_p.st for _p in _path])
    print(f"{st.where(sp0)}, {st.where(sp1)}, {st.where(sp2)}")
    xs = [_p.x[st.where(sp0)] for _p in _path]
    ys = [_p.x[st.where(sp1)] for _p in _path]
    zs = [_p.x[st.where(sp2)] for _p in _path]
    ax.scatter(xs=xs, ys=ys, zs=zs, lw=1, label=names[method], color=colors[method], alpha=0.5)

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
plt.savefig(os.path.join(f"{name_handle}_png", f"{name_handle}_geodesics_3d.png"), dpi=200)
plt.show()
plt.clf()
plt.cla()
plt.close(fig)

lengths = {method: np.round(ws.g.length(path_=paths[method]), 6) for method in methods}
# NOW PLOT THE HISTOGRAMS OF THE GEODESICS 1:
for method in methods:
    if method in []:
        continue
    path = paths[method]
    data = [ws.g.a_dist(p=path[_i], q=path[_i + 1]) for _i in range(0, len(path) - 1)]
    plt.hist(x=data, color=colors[method], label=f"{names[method]}: length={lengths[method]}, time={times[method]}s",
             alpha=0.2)
plt.legend()
plt.title(f"Distances between consecutive points")
plt.savefig(os.path.join(f"{name_handle}_png", f"{name_handle}_distances_histogram.png"), dpi=200)
plt.clf()
plt.cla()
plt.close(fig)

# NOW PLOT THE HISTOGRAMS OF THE GEODESICS 2:
for method in methods:
    if method in ['symmetric']:
        continue
    path = paths[method]
    data = [ws.g.a_dist(p=path[_i], q=path[_i + 1]) for _i in range(0, len(path) - 1)]
    plt.hist(x=data, color=colors[method], label=f"{names[method]}: length={lengths[method]}, time={times[method]}s",
             alpha=0.2)
plt.legend()
plt.title(f"Distances between consecutive points")
plt.savefig(os.path.join(f"{name_handle}_png", f"{name_handle}_distances_histogram_without_symmetric.png"), dpi=200)
plt.clf()
plt.cla()
plt.close(fig)

# NOW PLOT THE HISTOGRAMS OF THE GEODESICS 3:
for method in methods:
    if method in ['symmetric', 'naive']:
        continue
    path = paths[method]
    data = [ws.g.a_dist(p=path[_i], q=path[_i + 1]) for _i in range(0, len(path) - 1)]
    plt.hist(x=data, color=colors[method], label=f"{names[method]}: length={lengths[method]}, time={times[method]}s",
             alpha=0.2)
plt.legend()
plt.title(f"Distances between consecutive points")
plt.savefig(
    os.path.join(f"{name_handle}_png", f"{name_handle}_distances_histogram_without_symmetric_without_naive.png"),
    dpi=200)
plt.clf()
plt.cla()
plt.close(fig)

print(paths['symmetric'])
for method in methods:
    print(f"The lengths of the method {method} is {ws.g.length(paths[method])}.")
    print(f"The entropy of the method {method} is {numtools.entropy(_path=paths[method], waldspace=ws)}.")
    print(f"Time needed of the method {method} is {times[method]}.")
    # print(f"The angles of the method {method} are {numtools.angles(_path=paths[method], waldspace=ws)}.")

for method in methods:
    print(f"The energy of the method {method} is {numtools.energy(_path=paths[method], geometry=ws)}.")
