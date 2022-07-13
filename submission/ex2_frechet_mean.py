import os
import sys
import pickle
import time
import ntpath
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import WaldSpace
import treespaces.framework.tools_numerical as numtools
import treespaces.framework.tools_io as iotools
import treespaces.visualization.tools_plot as ptools


# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 4
partition = ((0, 1, 2, 3),)
sp0 = Split(n=n, part1=(0,), part2=(1, 2, 3))
sp1 = Split(n=n, part1=(1,), part2=(0, 2, 3))
sp2 = Split(n=n, part1=(2,), part2=(1, 0, 3))
sp3 = Split(n=n, part1=(3,), part2=(1, 2, 0))
sp01 = Split(n=n, part1=(0, 1), part2=(2, 3))
sp12 = Split(n=n, part1=(1, 2), part2=(0, 3))
sp02 = Split(n=n, part1=(0, 2), part2=(1, 3))

split_collection01 = [[sp0, sp1, sp2, sp3, sp01]]
split_collection12 = [[sp0, sp1, sp2, sp3, sp12]]
split_collection02 = [[sp0, sp1, sp2, sp3, sp02]]

folder_name = ntpath.splitext(ntpath.basename(__file__))[0]
try:
    os.mkdir(path=folder_name)
except FileExistsError:
    pass

try:
    os.mkdir(path=os.path.join(folder_name, "data"))
except FileExistsError:
    pass

st01 = Structure(n=n, partition=partition, split_collection=split_collection01)
st12 = Structure(n=n, partition=partition, split_collection=split_collection12)
st02 = Structure(n=n, partition=partition, split_collection=split_collection02)

for st in [st01, st12, st02]:
    print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

pendant_length = 0.3
interior_length = 0.1
# the weights are associated according to this order of the splits
# ((0, 1, 2)|(3,), (0, 1, 3)|(2,), (0, 2, 3)|(1,), (0,)|(1, 2, 3), (0, 1)|(2, 3))
x01 = [pendant_length, pendant_length, pendant_length, pendant_length, interior_length]
wald01 = Wald(n=n, st=st01, x=x01)

# ((0, 1, 2)|(3,), (0, 1, 3)|(2,), (0, 2, 3)|(1,), (0,)|(1, 2, 3), (0, 3)|(1, 2))
x12 = [pendant_length, pendant_length, pendant_length, pendant_length, interior_length]
wald12 = Wald(n=n, st=st12, x=x12)

# ((0, 1, 2)|(3,), (0, 1, 3)|(2,), (0, 2, 3)|(1,), (0, 2)|(1, 3), (0,)|(1, 2, 3))
x02 = [pendant_length, pendant_length, pendant_length, interior_length, pendant_length]
wald02 = Wald(n=n, st=st02, x=x02)


# ----- FLAG DETERMINING WHETHER THEY SHOULD BE SOMETHING COMPUTED OR JUST LOADED ----- #
calculate = False
wiggled = False

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
geometry = "affine-invariant"
ws = WaldSpace(geometry=geometry)

# parameters:
proj_args = {'gtol': 10 ** -6, 'ftol': 10 ** -6, 'btol': 10 ** -7, 'method': 'global-descent'}

params = {'proj_args': proj_args}

np.random.seed(111)
iterations = 100

str_pen = f"{np.round(pendant_length, 4)}".replace(".", "-")
str_int = f"{np.round(interior_length, 4)}".replace(".", "-")

fn = f"001_mean_pen_{str_pen}_int_{str_int}_it_{iterations}_{'wiggled_' if wiggled else ''}{wiggled if wiggled else ''}"
fn_data = os.path.join(folder_name, "data", f"{fn}.p")


def frechet_mean(walds, _iter=10):
    _mean_sequence = []
    j = 0
    perm = np.arange(0, len(walds))
    np.random.shuffle(perm)
    _mean = walds[perm[j]]
    _mean_sequence += [_mean]
    j += 1
    for i in range(2, _iter + 2):
        print(f"Iteration {i-1}.")
        if j == len(walds):
            j = 0
            perm = np.arange(0, len(walds))
            np.random.shuffle(perm)
        _next = walds[perm[j]]
        j += 1
        _n_points = int(2*(i + 1))
        current_path = ws.g.s_path(p=_mean, q=_next, alg='global-straighten-extend', n_points=_n_points, **params)
        print(f"------Calculating path from point\n{_mean.st}\n{_mean.x}\nto the point\n{_next.st}\n{_next.x}\n------.")
        _index = np.argmin([np.abs(k/(len(current_path) - 1) - 1/i) for k in range(len(current_path))])
        print(f"Distance from mean to next point is {ws.g.length(current_path)}.")
        _mean_sequence.append(current_path[_index])
        _mean = current_path[_index]
        print("New mean is")
        print(_mean.st)
        print(_mean.x)
    return _mean_sequence


if calculate:
    start = time.time()
    mean_sequence = frechet_mean(walds=[wald01, wald02, wald12], _iter=iterations)
    end = time.time()
    t = np.round(end - start, 2)
    print(f"Calculation took {t} seconds.")
    print(f"Store the data that has been calculated.")
    pickle.dump(obj=(tuple(mean_sequence), t), file=open(fn_data, "wb"))
else:
    print(f"Load the already stored data since the flag 'calculate' is False.")
    mean_sequence, t = pickle.load(file=open(fn_data, "rb"))


# ----- PLOT THE CALCULATED DATA! ----- #
vis_path = os.path.join(folder_name, f"vis_{fn}")
try:
    os.mkdir(path=vis_path)
except FileExistsError:
    pass

print(f"Calculation took {np.round(t / 60, 2)} minutes.")

ptools.plot_wald(wald01, labels={0: "1", 1: "2", 2: "3", 3: "4"}, root="a",
                 fn=os.path.join(vis_path, f"00_wald01.png"))
ptools.plot_wald(wald12, labels={0: "1", 1: "2", 2: "3", 3: "4"}, root="a",
                 fn=os.path.join(vis_path, f"00_wald12.png"))
ptools.plot_wald(wald02, labels={0: "1", 1: "2", 2: "3", 3: "4"}, root="a",
                 fn=os.path.join(vis_path, f"00_wald02.png"))

for i, p in enumerate(mean_sequence):
    ptools.plot_wald(p, labels={0: "1", 1: "2", 2: "3", 3: "4"}, root="a",
                     fn=os.path.join(vis_path, f"10_mean_{iotools.f00(i)}.png"))
