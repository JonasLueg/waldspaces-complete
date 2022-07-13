import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# DANGER ALERT: this works only, when this file is in the folder "/examples/cobwebs/cobweb1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import treespaces.framework.tools_numerical as numtools
from tools.structure_and_split import Split
from treespaces.framework.waldspace import WaldSpace
from tools.wald import Wald, Tree
from tools.structure_and_split import Split, Structure


n = 5
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

# the two endpoints of the geodesics
_x1 = {**{s: 0.095 for s in [sp0, sp1, sp2, sp3, sp4]}, **{sp01: 0.39, sp34: 0.25}}
_x1 = np.array([_x1[s] for s in st1.split_collection[0]])

_x2 = {**{s: 0.095 for s in [sp0, sp1, sp2, sp3, sp4]}, **{sp03: 0.39, sp12: 0.25}}
_x2 = np.array([_x2[s] for s in st2.split_collection[0]])


wald1 = Wald(n=n, st=st1, x=_x1)
wald2 = Wald(n=n, st=st2, x=_x2)

_dir_name = 'two-points-1'
n_points = 31
ws = WaldSpace(geometry='killing')


proj_args = {'gtol': 10 ** -18, 'ftol': 10 ** -18, 'btol': 10 ** -6, 'method': 'global-descent'}
print("Compute the naive path.")
naive_path = ws.g.s_path(p=wald1, q=wald2, proj_args=proj_args, n_points=n_points, alg='global-naive')
print("Done computing the naive path")

kinds = ['cone', 'symmetric']  # '['cone', 'circular', 'symmetric']
colors = {'cone': 'red', 'circular': 'blue', 'symmetric': 'black'}
labels = {'cone': 'cone path', 'circular': 'round path', 'symmetric': 'SP path'}

# load all geodesics
n_iter = 201
paths = dict()
for kind in kinds:
    paths[kind] = pickle.load(open(os.path.join(f"{_dir_name}", "data", f"{_dir_name}_n_{n_points}_kind_{kind}.p"), "rb"))
    paths[kind] = paths[kind][:n_iter]
k_list = [k for k in np.arange(start=0, stop=n_iter)]


# plot entropies of the paths:
fig = plt.figure()
ax = fig.add_subplot(111)
energy = dict()
for kind in kinds:
    energy[kind] = [numtools.energy(_path=_path, geometry=ws.g) for _path in paths[kind]]
    plt.plot(k_list[:len(energy[kind])], energy[kind], label=labels[kind], color=colors[kind])
    plt.plot(k_list, [numtools.energy(_path=paths[kind][0], geometry=ws.g) for _ in k_list])
plt.plot(k_list, [numtools.energy(_path=naive_path, geometry=ws.g) for _ in k_list])
ax.set_xlabel("iteration")
ax.set_ylabel("energy of path")
# plt.title("energy")
plt.legend()
plt.savefig(os.path.join(f"{_dir_name}", f"{_dir_name}_paths_n_{n_points}_energy.pdf"), dpi=700)
plt.clf()
plt.cla()
plt.close(fig)


# # plot the geodesics themselves
# n = 5
# sp0 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
# sp1 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
# sp2 = Split(n=n, part1=(1, 2), part2=(0, 3, 4))
# sp3 = Split(n=n, part1=(0, 3), part2=(1, 2, 4))
# sp_0 = Split(n=n, part1=(0,), part2=(1, 2, 3, 4))

# scatter = True
# alpha = 0.7
# _slice = 60
# for kind in kinds:
#     _fn = os.path.join(f"{_dir_name}", f"{_dir_name}_n_{n_points}_kind_{kind}.png")
#     lengths = [np.round(ws.g.length(path_=path), 4) for path in paths[kind]]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, aspect='equal')
#     for j, path in enumerate(paths[kind][:-1]):
#         if j % _slice != 0:
#             continue
#         coords = numtools.give_coords_n5_interior_splits(path, sp0, sp1, sp2, sp3)
#         label = f"It. {(6 - 2*len(str(j)))*' '}{j}. Length = {lengths[j]}."
#         if scatter:
#             plt.scatter(*coords, alpha=alpha, label=label)
#         else:
#             plt.plot(*coords, alpha=alpha, label=label)
#     # plot the last iteration
#     coords = numtools.give_coords_n5_interior_splits(paths[kind][-1], sp0, sp1, sp2, sp3)
#     label = f"It. {len(paths[kind]) - 1}. Length = {lengths[-1]}."
#     alpha = 0.7
#     if scatter:
#         plt.scatter(*coords, alpha=alpha, label=label, color=colors[kind])
#     else:
#         plt.plot(*coords, alpha=alpha, label=label, color=colors[kind])
#
#     plt.axhline(0, color='black')
#     plt.axvline(0, color='black')
#     plt.xlim((-0.5, 0.5))
#     plt.ylim((-0.5, 0.5))
#     ax.set_xlabel(r"$\lambda_6$")
#     ax.set_ylabel(r"$\lambda_7$")
#     ax.text(0.40, 0.21, r"$W_1$", color='black')
#     ax.text(-0.23, -0.41, r"$W_2$", color='black')
#     plt.title(labels[kind])
#     plt.legend()
#     plt.savefig(_fn, dpi=350)
#     plt.clf()
#     plt.cla()
#     plt.close(fig)
#
#
# # plot lengths of the paths:
# fig = plt.figure()
# ax = fig.add_subplot(111)
# lengths = dict()
# for kind in kinds:
#     m = 17
#     for i in range(len(paths[kind][-1][m].x)):
#         lengths[kind] = [_path[m].x[i] for _path in paths[kind]]
#         plt.plot(k_list, lengths[kind], label=f"Split {paths[kind][-1][m].st.split_collection[0][i]}")
# ax.set_xlabel("iteration")
# ax.set_ylabel(f"length of {sp_0}")
# plt.title("length")
# plt.legend()
# plt.savefig(os.path.join(f"{_dir_name}", f"{_dir_name}_paths_n_{n_points}_length_of_splits.png"), dpi=700)
# plt.clf()
# plt.cla()
# plt.close(fig)





