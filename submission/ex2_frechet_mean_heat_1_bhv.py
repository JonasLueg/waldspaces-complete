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
x01 = [pendant_length, pendant_length, pendant_length, pendant_length, interior_length]
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
    return np.sum([ws.g.length(_path) ** 2 for _path in paths])


# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
geometry = "bhv"
ws = WaldSpace(geometry=geometry)

# CALCULATION NO. 1
# parameters:
# ----- FLAG DETERMINING WHETHER THEY SHOULD BE SOMETHING COMPUTED OR JUST LOADED ----- #
calculate = True
proj_args = {'gtol': 10 ** -7, 'ftol': 10 ** -7, 'btol': 10 ** -7, 'method': 'global-descent'}
params = {'proj_args': proj_args}
n_points = 33

fn = f"frechet_functional_heat"
fn_data = os.path.join(folder_name, "data", f"{fn}.p")

if calculate:
    start = time.time()
    # step 1: find out the optimal pendant edge length for interior = 0.
    interiors = np.linspace(0.297, 0.303, 20)
    pendants = np.linspace(0, 10 ** -3, 20)
    ints, pens = np.meshgrid(interiors, pendants)
    z = tuple(tuple(compute_frechet_functional(Wald(n=n, st=st01, x=[_pen, _pen, _pen, _pen, _int]))
                    for _pen, _int in zip(row_pen, row_int)) for row_pen, row_int in zip(ints, pens))
    end = time.time()
    t = np.round(end - start, 2)
    print(f"Store the data that has been calculated.")
    pickle.dump(obj=(tuple(interiors), tuple(pendants), z, t), file=open(fn_data, "wb"))
else:
    print(f"Load the already stored data since the flag 'calculate' is False.")
    interiors, pendants, z, t = pickle.load(file=open(fn_data, "rb"))

# ----- PLOT THE CALCULATED DATA! ----- #
vis_path = os.path.join(folder_name, f"vis_{fn}")
try:
    os.mkdir(path=vis_path)
except FileExistsError:
    pass
print(f"Calculation took {np.round(t / 60, 2)} minutes.")

ints, pens = np.meshgrid(interiors, pendants)
z_min, z_max = 0, np.max(z)

fig, ax = plt.subplots()
c = ax.pcolormesh(ints, pens, z, shading='auto')
# ax.set_title(r"\Large$d_{\mathcal{W}_2}\big(W_1, W_2\big)$")
# set the limits of the plot to the limits of the data
# ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.xlabel("interior")
plt.ylabel("pendant")
plt.savefig(os.path.join(vis_path, f"{fn}.pdf"))

# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)
# ax.plot(pendants, frechet_values, color='blue', lw=1.2)
#
# plt.xlim(0.0, 1.0)
# plt.xticks(np.linspace(start=0, stop=1, num=11, endpoint=True))
# ax.set(xlabel=r"pendant edge length $\lambda$", ylabel=r"$F(\lambda)$")
# plt.legend()
# plt.savefig(os.path.join(vis_path, "optimal_pendants.pdf"))
#
#
# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)
# ax.plot(interior, frechet_values_01, color='blue', lw=1.2, label="tree structure 0,1 vs 2,3")
# ax.plot(interior, frechet_values_02, color='red', lw=1.2, label="tree structure 0,2 vs 1,3")
# ax.plot(interior, frechet_values_12, color='orange', lw=1.2, label="tree structure 1,2 vs 0,3")
#
# xmax = interior[-1]
# plt.xlim(0.0, xmax)
# plt.xticks(np.linspace(start=0, stop=xmax, num=11, endpoint=True))
# ax.set(xlabel=r"interior edge length $\lambda$", ylabel=r"$F(\lambda)$")
# plt.legend()
# plt.savefig(os.path.join(vis_path, "optimal_interior.pdf"))

print(z)