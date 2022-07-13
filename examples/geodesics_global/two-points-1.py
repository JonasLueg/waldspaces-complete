import os
import sys
import time
import ntpath
import pickle
import numpy as np
import matplotlib.pyplot as plt

# DANGER ALERT: this works only, when this file is in the folder "/examples/cobwebs/cobweb1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import treespaces.framework.tools_numerical as numtools
from tools.wald import Wald, Tree
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import WaldSpace

_dir_name = ntpath.splitext(ntpath.basename(__file__))[0]

n = 5
ws = WaldSpace(geometry='killing')
partition = ((0, 1, 2, 3, 4),)

sp01 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
sp34 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
sp12 = Split(n=n, part1=(1, 2), part2=(0, 3, 4))
sp03 = Split(n=n, part1=(0, 3), part2=(1, 2, 4))

sp0 = Split(n=n, part1=(0,), part2=(1, 2, 3, 4))
sp1 = Split(n=n, part1=(1,), part2=(0, 2, 3, 4))
sp2 = Split(n=n, part1=(2,), part2=(0, 1, 3, 4))
sp3 = Split(n=n, part1=(3,), part2=(0, 1, 2, 4))
sp4 = Split(n=n, part1=(4,), part2=(0, 1, 2, 3))

splits1 = [[sp0, sp1, sp2, sp3, sp4, sp01, sp34]]
splits2 = [[sp0, sp1, sp2, sp3, sp4, sp12, sp03]]
splits3 = [[sp0, sp1, sp2, sp3, sp4, sp34, sp12]]

st1 = Structure(n=n, partition=partition, split_collection=splits1)
st2 = Structure(n=n, partition=partition, split_collection=splits2)
st3 = Structure(n=n, partition=partition, split_collection=splits3)


def make_wald(x, st):
    return Wald(n=n, st=st, x=x)


def make_start_path(n_points, wald1: Wald, wald2: Wald, proj_args=None, kind='cone'):
    """ Construct the start path, importantly, assumes that n_points is odd. """
    if kind == 'cone':  # cone path.
        x1_list = [np.array([_x * (1 - t) if _i in [st1.where(sp01), st1.where(sp34)] else _x
                             for _i, _x in enumerate(wald1.x)])
                   for t in np.linspace(start=0, stop=1, endpoint=False, num=n_points // 2)]

        x2_list = [np.array([_x * (1 - t) if _i in [st2.where(sp03), st2.where(sp12)] else _x
                             for _i, _x in enumerate(wald2.x)])
                   for t in np.linspace(start=0, stop=1, endpoint=False, num=n_points // 2)][::-1]

        _path_1 = tuple([make_wald(x=_x, st=st1) for _x in x1_list])
        _path_2 = tuple([make_wald(x=_x, st=st2) for _x in x2_list])
        _path_m = Wald(n=n, st=st1, x=np.array([0 if _i in [st1.where(sp01), st1.where(sp34)] else _x
                                                for _i, _x in enumerate(wald1.x)]))
        return _path_1 + (_path_m,) + _path_2

    if kind == 'symmetric':
        if proj_args is None:
            proj_args = dict()
        print(wald1.st)
        print(wald2.st)
        return ws.g.s_path(p=wald1, q=wald2, n_points=n_points, alg='global-symmetric', proj_args=proj_args)

    if kind == 'circular':
        x1_list = [np.array([_x * (1 - t) if _i in [5] else _x for _i, _x in enumerate(wald1.x)])
                   for t in np.linspace(start=0, stop=1, endpoint=False, num=n_points // 3)]

        x2_list = [np.array([_x * (1 - t) if _i in [5] else _x for _i, _x in enumerate(wald2.x)])
                   for t in np.linspace(start=0, stop=1, endpoint=False, num=n_points // 3)][::-1]

        def _cos(t): return np.cos(np.pi / 2 * t)

        def _sin(t): return np.sin(np.pi / 2 * t)

        x3_list = [np.array(
            [_cos(t) * wald1.x[4] if _i in [2] else _sin(t) * wald2.x[2] if _i in [5] else _x for _i, _x in
             enumerate(wald2.x)])
                   for t in np.linspace(start=0, stop=1, endpoint=True, num=n_points - 2 * (n_points // 3))[::-1]]

        _path_1 = tuple([make_wald(x=_x, st=st1) for _x in x1_list])
        _path_2 = tuple([make_wald(x=_x, st=st2) for _x in x2_list])
        _path_m = tuple([make_wald(x=_x, st=st3) for _x in x3_list])

        return _path_1 + _path_m + _path_2


proj_args = {'gtol': 10 ** -18, 'ftol': 10 ** -18, 'btol': 10 ** -6, 'method': 'global-descent'}


def compute_or_load_geodesics(n_points, kind, calculate, wald1=None, wald2=None, iterations=10):
    # ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
    _fn = os.path.join(f"{_dir_name}", "data", f"{_dir_name}_n_{n_points}_kind_{kind}.p")
    paths = None
    try:
        print(f"Load the data.")
        paths = list(pickle.load(file=open(_fn, "rb")))
        if not calculate:
            return paths
    except FileNotFoundError as e:
        if not calculate:
            print("File was not found and calculate was false, STOP.")
            raise e

    # we cannot be here if calculate was False, so assume it is True

    if paths is None:
        print("Start fresh with a new starting path.")
        _path = make_start_path(n_points=n_points, wald1=wald1, wald2=wald2, proj_args=proj_args, kind=kind)
        paths = [_path]
    else:
        print(f"Continue to iterate over the loaded data (already {len(paths) - 1} iterations have been performed.")

    # print(paths)
    # start the iterations.
    total = 0
    for i in range(iterations):
        print(f"Start with {i + 1}{numtools.postfix(i + 1)} iteration (out of {iterations}).")
        start = time.time()
        _path = ws.g.s_path(p=wald1, q=wald2, alg='global-straightening', n_points=n_points, start_path=paths[-1],
                            n_iter=1, proj_args=proj_args)
        end = time.time()
        total += end - start
        print(f"Time taken: {np.round(end - start, 2)} seconds. Total: {np.round(total, 2)} seconds.")
        paths.append(tuple(_path))

    print(f"Store the data that has been calculated.")
    pickle.dump(obj=tuple(paths), file=open(_fn, "wb"))
    return paths


# the two endpoints of the geodesics
_x1 = {**{s: 0.095 for s in [sp0, sp1, sp2, sp3, sp4]}, **{sp01: 0.39, sp34: 0.25}}
_x1 = np.array([_x1[s] for s in st1.split_collection[0]])

_x2 = {**{s: 0.095 for s in [sp0, sp1, sp2, sp3, sp4]}, **{sp03: 0.39, sp12: 0.25}}
_x2 = np.array([_x2[s] for s in st2.split_collection[0]])

wald1 = Wald(n=n, st=st1, x=_x1)
wald2 = Wald(n=n, st=st2, x=_x2)

# number of points on a geodesic
n_points = 31
# which start path to choose. (cone, symmetric, circular)
kind = 'symmetric'
# just plot stuff or also calculate?
calculate = False
# if calculation should happen, how many iterations?
iterations = 50

# computations or loading data happens here.
paths = compute_or_load_geodesics(n_points=n_points, kind=kind, calculate=calculate,
                                  wald1=wald1, wald2=wald2, iterations=iterations)

# for path in paths:
#     print([p.x for p in path])

# np.set_printoptions(precision=5)
# for i, p in enumerate(paths):
#     print(f"Energy of path {i} is {numtools.energy(p, ws.g)}.")

naive_path = ws.g.s_path(p=wald1, q=wald2, proj_args=proj_args, n_points=n_points, alg='global-naive')
# print(f"Energy of naive path is {numtools.energy(naive_path, ws.g)}.")

# parameters for plotting
slice = 20
alpha = 0.6
scatter = False

# for storing the plots.
_fn = os.path.join(f"{_dir_name}", f"{_dir_name}_n_{n_points}_kind_{kind}.pdf")
lengths = [np.round(ws.g.length(path_=path), 4) for path in paths]
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
for j, path in enumerate(paths[:-1]):
    if j % slice != 0:
        continue
    coords = numtools.give_bhv_coords_n5_interior_splits(path, sp01, sp34, sp12, sp03)
    label = f"PS: Iteration {'' if j > 9 else ' '}{j}."
    if scatter:
        plt.scatter(*coords, alpha=alpha, label=label)
    else:
        plt.plot(*coords, alpha=alpha, label=label)
# plot the last iteration
coords = numtools.give_bhv_coords_n5_interior_splits(paths[-1], sp01, sp34, sp12, sp03)
label = f"PS: Iteration {len(paths) - 1}."
alpha = 1
if scatter:
    plt.scatter(*coords, alpha=alpha, label=label, color='black')
else:
    plt.plot(*coords, alpha=alpha, label=label, color='black')

# plot the naive path
coords = numtools.give_bhv_coords_n5_interior_splits(naive_path, sp01, sp34, sp12, sp03)
label = f"Naive Projection (NP)."
alpha = 1
if scatter:
    plt.scatter(*coords, alpha=alpha, label=label, color='red')
else:
    plt.plot(*coords, alpha=alpha, label=label, color='red')


# plot the symmetric path
# coords = numtools.give_bhv_coords_n5_interior_splits(paths[0], sp01, sp34, sp12, sp03)
# label = f"Successive Projection (SP)."
# alpha = 1
# if scatter:
#     plt.scatter(*coords, alpha=alpha, label=label, color='blue')
# else:
#     plt.plot(*coords, alpha=alpha, label=label, color='blue')


plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlim((-0.6, 0.6))
plt.ylim((-0.6, 0.6))
plt.title("Comparison of three Algorithms")
plt.legend()
plt.savefig(_fn)
plt.clf()
plt.cla()
plt.close(fig)

# The lengths of the paths.
# plot lengths of the paths:
_fn = os.path.join(f"{_dir_name}", f"{_dir_name}_n_{n_points}_kind_{kind}_lengths.pdf")
fig = plt.figure()
ax = fig.add_subplot(111)
k_list = [k for k in np.arange(start=0, stop=len(paths))]
lengths = [ws.g.length(path_=_path) for _path in paths]
plt.plot(k_list, lengths, label="Path Straightening (PS)")
plt.plot(k_list, [ws.g.length(naive_path) for _ in paths], label="Naive Projection (NP)")
plt.plot(k_list, [ws.g.length(paths[0]) for _ in paths], label="Successive Projection (SP)")

ax.set_xlabel("iteration")
ax.set_ylabel("length of path")
plt.title("Lengths of the different paths")
plt.legend()
plt.savefig(_fn)
plt.clf()
plt.cla()
plt.close(fig)


# The energy of the paths.
# plot energies of the paths:
_fn = os.path.join(f"{_dir_name}", f"{_dir_name}_n_{n_points}_kind_{kind}_energy.pdf")
fig = plt.figure()
ax = fig.add_subplot(111)
k_list = [k for k in np.arange(start=0, stop=len(paths))]
energies = [numtools.energy(_path=_path, geometry=ws.g) for _path in paths]
plt.plot(k_list, energies, label="Path Straightening (PS)")
plt.plot(k_list, [numtools.energy(_path=naive_path, geometry=ws.g) for _ in paths], label="Naive Projection (NP)")
plt.plot(k_list, [numtools.energy(_path=paths[0], geometry=ws.g) for _ in paths], label="Successive Projection (SP)")

ax.set_xlabel("iteration")
ax.set_ylabel("energy of path")
plt.title("Energies of the different paths")
plt.legend()
plt.savefig(_fn)
plt.clf()
plt.cla()
plt.close(fig)
