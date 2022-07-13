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
st00 = Structure(n=n, partition=partition, split_collection=[[sp0, sp1, sp2, sp3]])

for st in [st01, st12, st02]:
    print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

pendant_length = 0.3
interior_length = 0.1
# the weights are associated according to this order of the splits
# ((0, 1, 2)|(3,), (0, 1, 3)|(2,), (0, 2, 3)|(1,), (0,)|(1, 2, 3), (0, 1)|(2, 3))
x01 = [pendant_length, pendant_length, pendant_length, pendant_length, 0.15]
wald01 = Wald(n=n, st=st01, x=x01)

# ((0, 1, 2)|(3,), (0, 1, 3)|(2,), (0, 2, 3)|(1,), (0,)|(1, 2, 3), (0, 3)|(1, 2))
x12 = [pendant_length, pendant_length, pendant_length, pendant_length, interior_length]
wald12 = Wald(n=n, st=st12, x=x12)

# ((0, 1, 2)|(3,), (0, 1, 3)|(2,), (0, 2, 3)|(1,), (0, 2)|(1, 3), (0,)|(1, 2, 3))
x02 = [pendant_length, pendant_length, pendant_length, interior_length, pendant_length]
wald02 = Wald(n=n, st=st02, x=x02)


def compute_frechet_functional(w):
    print(f"Calculating Frechet functional of {w}.")
    paths = [ws.g.s_path(p=w, q=w_data, alg='global-straighten-extend', n_points=n_points, **params)
             for w_data in [wald01, wald02, wald12]]
    return np.sum([ws.g.length(_path)**2 for _path in paths])


# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
geometry = "affine-invariant"
ws = WaldSpace(geometry=geometry)


# CALCULATION NO. 1
# parameters:
# ----- FLAG DETERMINING WHETHER THEY SHOULD BE SOMETHING COMPUTED OR JUST LOADED ----- #
calculate_1 = False
proj_args = {'gtol': 10 ** -6, 'ftol': 10 ** -6, 'btol': 10 ** -7, 'method': 'global-descent'}
params = {'proj_args': proj_args}
eps = 10**-1
n_points = 65

fn = f"optimal_pendants_zoom"
fn_data = os.path.join(folder_name, "data", f"{fn}.p")

if calculate_1:
    start = time.time()
    # step 1: find out the optimal pendant edge length for interior = 0.
    pendants = np.linspace(0.2, 0.4, 100)
    walds = [Wald(n=n, st=st00, x=[_x]*4) for _x in pendants]
    frechet_values = [compute_frechet_functional(_w) for _w in walds]
    end = time.time()
    t = np.round(end - start, 2)
    print(f"Store the data that has been calculated.")
    pickle.dump(obj=(tuple(pendants), tuple(frechet_values), t), file=open(fn_data, "wb"))
else:
    print(f"Load the already stored data since the flag 'calculate' is False.")
    pendants, frechet_values, t = pickle.load(file=open(fn_data, "rb"))

# ----- PLOT THE CALCULATED DATA! ----- #
vis_path = os.path.join(folder_name, f"vis_{fn}")
try:
    os.mkdir(path=vis_path)
except FileExistsError:
    pass
print(f"Calculation took {np.round(t / 60, 2)} minutes.")

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(pendants, frechet_values, color='blue', lw=1.2)

plt.xlim(0.0, 1.0)
plt.xticks(np.linspace(start=0, stop=1, num=11, endpoint=True))
ax.set(xlabel=r"pendant edge length $\lambda$", ylabel=r"$F(\lambda)$")
plt.legend()
plt.savefig(os.path.join(vis_path, "optimal_pendants.pdf"))


# CALCULATION NO. 2
# parameters:
# ----- FLAG DETERMINING WHETHER THEY SHOULD BE SOMETHING COMPUTED OR JUST LOADED ----- #
calculate_2 = False
proj_args = {'gtol': 10 ** -8, 'ftol': 10 ** -8, 'btol': 10 ** -7, 'method': 'global-descent'}
params = {'proj_args': proj_args}
eps = 10**-4
n_points = 9

fn = f"optimal_interior"
fn_data = os.path.join(folder_name, "data", f"{fn}.p")

print(f"argmin lengths is {pendants[np.argmin(frechet_values)-3:np.argmin(frechet_values)+3]}")
print(f"minimum values is {frechet_values[np.argmin(frechet_values)-3:np.argmin(frechet_values)+3]}.")

if calculate_2:
    start = time.time()
    opt_pen = pendants[np.argmin(frechet_values)]
    interior = np.linspace(0, 10**-2, 10)
    # walds = [Wald(n=n, st=st00, x=[opt_pen]*4)] + [Wald(n=n, st=st01, x=[opt_pen]*4 + [_x]) for _x in interior]
    walds01 = [Wald(n=n, st=st01, x=[opt_pen, opt_pen, opt_pen, opt_pen, _x]) for _x in interior]
    walds02 = [Wald(n=n, st=st02, x=[opt_pen, opt_pen, opt_pen, _x, opt_pen]) for _x in interior]
    walds12 = [Wald(n=n, st=st12, x=[opt_pen, opt_pen, opt_pen, opt_pen, _x]) for _x in interior]
    # interior = np.hstack((0, interior))  # insert 0 length interior edge s.t. len(interior) = len(walds)
    print("compute Frechet values for 01")
    frechet_values_01 = [compute_frechet_functional(w=_w) for _w in walds01]
    print("compute Frechet values for 02")
    frechet_values_02 = [compute_frechet_functional(w=_w) for _w in walds02]
    print("compute Frechet values for 12")
    frechet_values_12 = [compute_frechet_functional(w=_w) for _w in walds12]
    end = time.time()
    t = np.round(end - start, 2)
    print(f"Store the data that has been calculated.")
    pickle.dump(obj=(tuple(interior), tuple(frechet_values_01), tuple(frechet_values_02), tuple(frechet_values_12), t),
                file=open(fn_data, "wb"))
else:
    print(f"Load the already stored data since the flag 'calculate' is False.")
    interior, frechet_values_01, frechet_values_02, frechet_values_12, t = pickle.load(file=open(fn_data, "rb"))
print(f"Calculation took {np.round(t / 60, 2)} minutes.")

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(interior, frechet_values_01, color='blue', lw=1.2, label="tree structure 0,1 vs 2,3")
ax.plot(interior, frechet_values_02, color='red', lw=1.2, label="tree structure 0,2 vs 1,3")
ax.plot(interior, frechet_values_12, color='orange', lw=1.2, label="tree structure 1,2 vs 0,3")

xmax = interior[-1]
plt.xlim(0.0, xmax)
plt.xticks(np.linspace(start=0, stop=xmax, num=11, endpoint=True))
ax.set(xlabel=r"interior edge length $\lambda$", ylabel=r"$F(\lambda)$")
plt.legend()
plt.savefig(os.path.join(vis_path, "optimal_interior.pdf"))

