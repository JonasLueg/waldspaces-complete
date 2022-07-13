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
from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1

_dir_name = ntpath.splitext(ntpath.basename(__file__))[0]

n = 5
ws = TreeSpdAf1(geometry='killing')
partition = ((0, 1, 2, 3, 4),)

sp0 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
sp1 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
sp2 = Split(n=n, part1=(0, 2), part2=(1, 3, 4))
sp3 = Split(n=n, part1=(1, 4), part2=(0, 2, 3))

split_collection1 = [[Split(n=n, part1=(0,), part2=(1, 2, 3, 4)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3, 4)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3, 4)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2, 4)),
                      Split(n=n, part1=(4,), part2=(0, 1, 2, 3)),
                      sp0,
                      sp1]]
st1 = Structure(n=n, partition=partition, split_collection=split_collection1)

split_collection2 = [[Split(n=n, part1=(0,), part2=(1, 2, 3, 4)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3, 4)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3, 4)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2, 4)),
                      Split(n=n, part1=(4,), part2=(0, 1, 2, 3)),
                      sp2,
                      sp3]]
st2 = Structure(n=n, partition=partition, split_collection=split_collection2)


def make_wald(x, st):
    """Indices of splits: st1: [sp0, sp1] at [5, 4], st2: [sp2, sp3] at [2, 5], st3: [sp1, sp2] at [5, 2]"""
    return Wald(n=n, st=st, x=x)


def make_start_path(n_points, wald1: Wald, wald2: Wald, proj_args=None, kind='infty'):
    """ Construct the start path, importantly, assumes that n_points is odd. """
    if kind == 'infty':  # cone path.
        x1_list = [np.array([t + _x * (1 - t) if _i in [6] else _x for _i, _x in enumerate(wald1.x)])
                   for t in np.linspace(start=0, stop=1, endpoint=False, num=n_points // 2)]
        x2_list = [np.array([t + _x * (1 - t) if _i in [6] else _x for _i, _x in enumerate(wald2.x)])
                   for t in np.linspace(start=0, stop=1, endpoint=False, num=n_points // 2)][::-1]
        _path_1 = tuple([make_wald(x=_x, st=st1) for _x in x1_list])
        _path_2 = tuple([make_wald(x=_x, st=st2) for _x in x2_list])
        _path_m = Wald(n=n, st=st1, x=np.array([1 if _i in [6] else _x for _i, _x in enumerate(wald1.x)]))
        return _path_1 + (_path_m,) + _path_2

    if kind == 'symmetric':
        if proj_args is None:
            proj_args = dict()
        return ws.g.s_path(p=wald1, q=wald2, n_points=n_points, alg='global-symmetric', proj_args=proj_args)


def compute_or_load_geodesics(n_points, kind, calculate, wald1=None, wald2=None, iterations=10):
    # ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
    _fn = os.path.join(f"{_dir_name}", "data", f"{_dir_name}_n_{n_points}_kind_{kind}.p")
    paths = None
    try:
        print(f"Load the data.")
        paths = list(pickle.load(file=open(_fn, "rb")))
        # paths = paths[:-3]
        if not calculate:
            return paths
    except FileNotFoundError as e:
        if not calculate:
            print("File was not found and calculate was false, STOP.")
            raise e

    # paths = [[Wald(n=w.n, st=w.st, x=w.x) for w in path] for path in paths]
    # we cannot be here if calculate was False, so assume it is True
    proj_args = {'gtol': 10 ** -20, 'ftol': 10 ** -20, 'btol': 10 ** -10, 'method': 'global-descent'}

    if paths is None:
        print("Start fresh with a new starting path.")
        _path = make_start_path(n_points=n_points, wald1=wald1, wald2=wald2, proj_args=proj_args, kind=kind)
        paths = [_path]
    else:
        print(f"Continue to iterate over the loaded data (already {len(paths) - 1} iterations have been performed.")

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


def plot_geodesics(paths, slice=2, scatter=True, kind='cone', alpha=0.7):
    # for storing the plots.
    _fn = os.path.join(f"{_dir_name}", f"{_dir_name}_n_{n_points}_kind_{kind}.png")
    lengths = [np.round(ws.g.length(path_=path), 4) for path in paths]
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for j, path in enumerate(paths[:-1]):
        if j % slice != 0:
            continue
        coords = numtools.give_coords_n5_interior_splits(path, sp0, sp1, sp2, sp3)
        label = f"It. {j}. Length = {lengths[j]}."
        if scatter:
            plt.scatter(*coords, alpha=alpha, label=label)
        else:
            plt.plot(*coords, alpha=alpha, label=label)
    # plot the last iteration
    coords = numtools.give_coords_n5_interior_splits(paths[-1], sp0, sp1, sp2, sp3)
    label = f"It. {len(paths) - 1}. Length = {lengths[-1]}."
    if scatter:
        plt.scatter(*coords, alpha=alpha, label=label, color='black')
    else:
        plt.plot(*coords, alpha=alpha, label=label, color='black')

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.title("Iterations of the EPS Algorithm")
    plt.legend()
    plt.savefig(_fn, dpi=700)
    plt.clf()
    plt.cla()
    plt.close(fig)

    k_list = [k for k in np.arange(start=0, stop=len(paths[0]))]
    _fn = os.path.join(f"{_dir_name}", f"{_dir_name}_n_{n_points}_kind_{kind}_pendant_lengths.png")
    fig = plt.figure()
    for sp in [Split(n=n, part1=(0,), part2=(1, 2, 3, 4)),
               Split(n=n, part1=(1,), part2=(0, 2, 3, 4)),
               Split(n=n, part1=(2,), part2=(0, 1, 3, 4)),
               Split(n=n, part1=(3,), part2=(0, 1, 2, 4)),
               Split(n=n, part1=(4,), part2=(0, 1, 2, 3))]:
        coords = [p.x[np.argmin([_ != sp for _ in p.st.split_collection[0]])] for p in paths[-1]]
        # print(coords)
        label = f"split {sp}"
        if scatter:
            plt.plot(k_list, coords, alpha=alpha, label=label)
        else:
            plt.plot(k_list, coords, alpha=alpha, label=label)

    plt.title("Pendant edge lengths.")
    plt.legend()
    plt.savefig(_fn, dpi=700)
    plt.clf()
    plt.cla()
    plt.close(fig)


if __name__ == "__main__":
    # the two endpoints of the geodesics
    _x1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    _x2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    print(st1)
    print(st2)
    wald1 = make_wald(x=_x1, st=st1)
    wald2 = make_wald(x=_x2, st=st2)

    # number of points on a geodesic
    n_points = 45
    # which start path to choose.
    kind = 'infty'
    # just plot stuff or also calculate?
    calculate = True
    # if calculation should happen, how many iterations?
    iterations = 1

    # computations or loading data happens here.
    # path = make_start_path(n_points, wald1, wald2, proj_args=None, kind='infty')
    # for p in path:
    #     print(f"{p.st}\n{p}")
    paths = compute_or_load_geodesics(n_points=n_points, kind=kind, calculate=calculate,
                                      wald1=wald1, wald2=wald2, iterations=iterations)

    # parameters for plotting
    slice_ = 10
    alpha = 0.4
    scatter = True
    plot_geodesics(paths, slice=slice_, scatter=scatter, kind=kind, alpha=alpha)
