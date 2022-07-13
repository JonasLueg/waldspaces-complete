import os
import sys
import ntpath
import pickle
import numpy as np
import matplotlib.pyplot as plt

# DANGER ALERT: this works only, when this file is in the folder "/examples/cobwebs/cobweb1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1

str_name = ntpath.splitext(ntpath.basename(__file__))[0]
str_dir = str_name[:-2]

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 5
partition = ((0, 1, 2, 3, 4),)
split_collection1 = [[Split(n=n, part1=(0,), part2=(1, 2, 3, 4)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3, 4)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3, 4)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2, 4)),
                      Split(n=n, part1=(4,), part2=(0, 1, 2, 3)),
                      Split(n=n, part1=(0, 1), part2=(2, 3, 4)),
                      Split(n=n, part1=(3, 4), part2=(0, 1, 2))]]

split_collection2 = [[Split(n=n, part1=(0,), part2=(1, 2, 3, 4)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3, 4)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3, 4)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2, 4)),
                      Split(n=n, part1=(4,), part2=(0, 1, 2, 3)),
                      Split(n=n, part1=(0, 3), part2=(1, 2, 4)),
                      Split(n=n, part1=(1, 2), part2=(0, 3, 4))]]

st1 = Structure(n=n, partition=partition, split_collection=split_collection1)
st2 = Structure(n=n, partition=partition, split_collection=split_collection2)
print(f"The structure 1 of the forest has one component and contains the splits\n{st1.split_collection[0]}.")
print(f"The structure 2 of the forest has one component and contains the splits\n{st2.split_collection[0]}.")

# ----- CONSTRUCTION OF THE WALD SPACE WITH OUR GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')

x1 = np.array([0.095, 0.095, 0.095, 0.095, 0.25, 0.8, 0.095])
x2 = np.array([0.095, 0.095, 0.25, 0.095, 0.095, 0.8, 0.095])

wald1 = Wald(n=n, st=st1, x=x1)
wald2 = Wald(n=n, st=st2, x=x2)
# list the interior splits and their respective indices in the structure.
sp0 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
sp1 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
sp2 = Split(n=n, part1=(0, 3), part2=(1, 2, 4))
sp3 = Split(n=n, part1=(1, 2), part2=(0, 3, 4))

# print(f"The first wald is the one with structure\n{wald1.st}\nand coordinates\n{wald1.x}.")
# print(f"The second wald is the one with structure\n{wald2.st}\nand coordinates\n{wald2.x}.")
calculate = True
n_points = 15
iterations = 10

proj_args = {'gtol': 10 ** -8, 'ftol': 10 ** -10, 'btol': 10 ** -6, 'method': 'global-search'}

try:
    print(f"Load the already stored data.")
    paths = list(pickle.load(file=open(os.path.join(f"{str_dir}_data", f"{str_name}_paths_n_{n_points}.p"), "rb")))
except FileNotFoundError:
    print(f"No previous path done, so start with symmetric algorithm path.")
    start_path = ws.g.s_path(p=wald1, q=wald2, alg='global-symmetric', n_points=n_points, proj_args=proj_args)
    paths = [start_path]

if calculate:
    for i in range(iterations):
        paths.append(tuple(ws.g.s_path(p=wald1, q=wald2, alg='global-straightening-2',
                                       n_points=n_points, start_path=paths[-1], n_iter=1, proj_args=proj_args)))
        print(f"{len(paths)}{'th' if not len(paths) % 10 in [1, 2, 3] else ['st', 'nd', 'rd'][len(paths) % 10 - 1]}"
              f" iteration done.")
    print(f"Store the data that has been calculated.")
    pickle.dump(obj=tuple(paths),
                file=open(os.path.join(f"{str_dir}_data", f"{str_name}_paths_n_{n_points}.p"), "wb"))

interior = {sp0, sp1, sp2, sp3}
print(interior)


def give_coords(_path):
    coords_x = []
    coords_y = []
    for _wald in _path:
        _x, _y = 0, 0
        if sp0 in _wald.st.split_collection[0]:
            _x = _wald.x[np.argmin([sp != sp0 for sp in _wald.st.split_collection[0]])]
        if sp1 in _wald.st.split_collection[0]:
            _y = _wald.x[np.argmin([sp != sp1 for sp in _wald.st.split_collection[0]])]
        if sp2 in _wald.st.split_collection[0]:
            _y = -_wald.x[np.argmin([sp != sp2 for sp in _wald.st.split_collection[0]])]
        if sp3 in _wald.st.split_collection[0]:
            _x = -_wald.x[np.argmin([sp != sp3 for sp in _wald.st.split_collection[0]])]
        coords_x.append(_x)
        coords_y.append(_y)
    return coords_x, coords_y


def entropy(_path):
    """ Measures how equidistant the points in _path are. Zero means they are perfectly equidistant. """
    _ideal_value = ws.g.length(path_=_path)
    dists = [ws.g.a_dist(p=_path[_i], q=_path[_i + 1], squared=False) for _i in range(0, len(_path) - 1)]
    return np.sum([x * np.log((len(_path) - 1) * x / _ideal_value) for x in dists])


idx = -1
pa = paths[idx]
eq = [ws.g.a_dist(pa[i], pa[i + 1], squared=False) for i in range(len(pa) - 1)]
print(eq)

# for wald in pa:
#     print(wald.x)

# plot the lengths of the paths
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
alpha = 0.7
for j, path in enumerate(paths):
    if j % 40 == 0:
        plt.scatter(*give_coords(_path=path), alpha=alpha,
                    label=f"k = {j}, length = {np.round(ws.g.length(path_=path), 2)}.")
plt.scatter(*give_coords(_path=paths[idx]), alpha=0.7,
            label=f"k = {len(paths)}, length = {np.round(ws.g.length(path_=paths[idx]), 2)}.")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.title("Shortest path iterations.")
plt.legend()
plt.savefig(os.path.join(f"{str_dir}_png", f"{str_name}_paths_n_{n_points}.png"), dpi=700)
plt.clf()
plt.cla()
plt.close(fig)

k_list = [k for k in np.arange(start=0, stop=len(paths))]
l_list = [ws.g.length(path_=path) for path in paths]
e_list = [entropy(_path=path) for path in paths]

print(e_list)

fig = plt.figure()

plt.plot(k_list, l_list, label="lengths")
plt.plot(k_list, e_list, label="entropies")
plt.title("Lengths of paths.")
plt.legend()
plt.savefig(os.path.join(f"{str_dir}_png", f"{str_name}_paths_n_{n_points}_lengths.png"), dpi=700)
plt.clf()
plt.cla()
plt.close(fig)
