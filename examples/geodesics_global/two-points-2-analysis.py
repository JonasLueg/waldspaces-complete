import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# DANGER ALERT: this works only, when this file is in the folder "/examples/cobwebs/cobweb1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import treespaces.framework.tools_numerical as numtools
from tools.structure_and_split import Split
from treespaces.framework.waldspace import TreeSpdAf1

_dir_name = 'two-points-2'
n_points = 45
ws = TreeSpdAf1(geometry='killing')

kinds = ['cone', 'circular', 'symmetric']
colors = {'cone': 'red', 'circular': 'blue', 'symmetric': 'black'}
labels = {'cone': 'cone path', 'circular': 'round path', 'symmetric': 'SP path'}

# load all geodesics
n_iter = 20
paths = dict()
for kind in kinds:
    paths[kind] = pickle.load(open(os.path.join(f"{_dir_name}", "data", f"{_dir_name}_n_{n_points}_kind_{kind}.p"), "rb"))
    paths[kind] = paths[kind][:n_iter]
k_list = [k for k in np.arange(start=0, stop=n_iter)]


# plot the geodesics themselves
n = 5
sp0 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
sp1 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
sp2 = Split(n=n, part1=(1, 2), part2=(0, 3, 4))
sp3 = Split(n=n, part1=(0, 3), part2=(1, 2, 4))
scatter = True
alpha = 0.7
_slice = 60
for kind in kinds:
    _fn = os.path.join(f"{_dir_name}", f"{_dir_name}_n_{n_points}_kind_{kind}.png")
    lengths = [np.round(ws.g.length(path_=path), 4) for path in paths[kind]]
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for j, path in enumerate(paths[kind][:-1]):
        if j % _slice != 0:
            continue
        coords = numtools.give_coords_n5_interior_splits(path, sp0, sp1, sp2, sp3)
        label = f"It. {(6 - 2*len(str(j)))*' '}{j}. Length = {lengths[j]}."
        if scatter:
            plt.scatter(*coords, alpha=alpha, label=label)
        else:
            plt.plot(*coords, alpha=alpha, label=label)
    # plot the last iteration
    coords = numtools.give_coords_n5_interior_splits(paths[kind][-1], sp0, sp1, sp2, sp3)
    label = f"It. {len(paths[kind]) - 1}. Length = {lengths[-1]}."
    alpha = 0.7
    if scatter:
        plt.scatter(*coords, alpha=alpha, label=label, color=colors[kind])
    else:
        plt.plot(*coords, alpha=alpha, label=label, color=colors[kind])

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlim((-0.5, 0.5))
    plt.ylim((-0.5, 0.5))
    ax.set_xlabel(r"$\lambda_6$")
    ax.set_ylabel(r"$\lambda_7$")
    ax.text(0.40, 0.21, r"$W_1$", color='black')
    ax.text(-0.23, -0.41, r"$W_2$", color='black')
    plt.title(labels[kind])
    plt.legend()
    plt.savefig(_fn, dpi=350)
    plt.clf()
    plt.cla()
    plt.close(fig)


# plot lengths of the paths:
fig = plt.figure()
ax = fig.add_subplot(111)
lengths = dict()
for kind in kinds:
    lengths[kind] = [ws.g.length(path_=_path) for _path in paths[kind]]
    plt.plot(k_list, lengths[kind], label=labels[kind], color=colors[kind])
ax.set_xlabel("iteration")
ax.set_ylabel("length of path")
plt.title("length")
plt.legend()
plt.savefig(os.path.join(f"{_dir_name}", f"{_dir_name}_paths_n_{n_points}_lengths.png"), dpi=700)
plt.clf()
plt.cla()
plt.close(fig)


# plot entropies of the paths:
fig = plt.figure()
entropy = dict()
for kind in kinds:
    entropy[kind] = [numtools.entropy(_path=_path, waldspace=ws) for _path in paths[kind]]
    plt.plot(k_list, entropy[kind], label=labels[kind], color=colors[kind])
ax.set_xlabel("iteration")
ax.set_ylabel("entropy of path")
plt.title("entropy")
plt.legend()
plt.savefig(os.path.join(f"{_dir_name}", f"{_dir_name}_paths_n_{n_points}_entropy.png"), dpi=700)
plt.clf()
plt.cla()
plt.close(fig)

# plot angles of the latest iterations:
fig = plt.figure()
angles = dict()
for kind in kinds:
    angles[kind] = numtools.angles(_path=paths[kind][-1], waldspace=ws)
    plt.hist(angles[kind], alpha=0.3, label=labels[kind], color=colors[kind])
plt.title("Angles of paths.")
plt.legend()
plt.savefig(os.path.join(f"{_dir_name}", f"{_dir_name}_paths_n_{n_points}_hist_angles.png"), dpi=700)
plt.clf()
plt.cla()
plt.close(fig)


# plot entropies of the paths:
fig = plt.figure()
energy = dict()
for kind in kinds:
    energy[kind] = [numtools.energy(_path=_path, geometry=ws) for _path in paths[kind]]
    plt.plot(k_list, energy[kind], label=labels[kind], color=colors[kind])
ax.set_xlabel("iteration")
ax.set_ylabel("energy of path")
plt.title("energy")
plt.legend()
plt.savefig(os.path.join(f"{_dir_name}", f"{_dir_name}_paths_n_{n_points}_energy.png"), dpi=700)
plt.clf()
plt.cla()
plt.close(fig)


# def plot_geodesics(paths, slice=2, scatter=True, kind='cone', alpha=0.7):
#     # for storing the plots.
#     _fn = os.path.join(f"{_dir_name}", f"{_dir_name}_n_{n_points}_kind_{kind}.png")
#     lengths = [np.round(ws.g.length(path_=path), 4) for path in paths]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, aspect='equal')
#     for j, path in enumerate(paths[:-1]):
#         if j % slice != 0:
#             continue
#         coords = numtools.give_coords_n5_interior_splits(path, sp0, sp1, sp2, sp3)
#         label = f"It. {j}. Length = {lengths[j]}."
#         if scatter:
#             plt.scatter(*coords, alpha=alpha, label=label)
#         else:
#             plt.plot(*coords, alpha=alpha, label=label)
#     # plot the last iteration
#     coords = numtools.give_coords_n5_interior_splits(paths[-1], sp0, sp1, sp2, sp3)
#     label = f"It. {len(paths) - 1}. Length = {lengths[-1]}."
#     alpha = 0.7
#     if scatter:
#         plt.scatter(*coords, alpha=alpha, label=label, color='black')
#     else:
#         plt.plot(*coords, alpha=alpha, label=label, color='black')
#
#     plt.axhline(0, color='black')
#     plt.axvline(0, color='black')
#     plt.xlim((-1, 1))
#     plt.ylim((-1, 1))
#     plt.title("Iterations of the EPS Algorithm")
#     plt.legend()
#     plt.savefig(_fn, dpi=700)
#     plt.clf()
#     plt.cla()
#     plt.close(fig)
#
#
# if __name__ == "__main__":
#     # the two endpoints of the geodesics
#     _x1 = np.array([0.095, 0.095, 0.095, 0.095, 0.25, 0.39, 0.095])
#     _x2 = np.array([0.095, 0.095, 0.25, 0.095, 0.095, 0.39, 0.095])
#     wald1 = make_wald(x=_x1, st=st1)
#     wald2 = make_wald(x=_x2, st=st2)
#
#     # number of points on a geodesic
#     n_points = 25
#     # which start path to choose.
#     kind = 'symmetric'
#     # just plot stuff or also calculate?
#     calculate = True
#     # if calculation should happen, how many iterations?
#     iterations = 150
#
#     # computations or loading data happens here.
#     paths = compute_or_load_geodesics(n_points=n_points, kind=kind, calculate=calculate,
#                                       wald1=wald1, wald2=wald2, iterations=iterations)
#
#     # parameters for plotting
#     slice_ = 50
#     alpha = 0.6
#     scatter = True
#     plot_geodesics(paths, slice=slice_, scatter=scatter, kind=kind, alpha=alpha)
