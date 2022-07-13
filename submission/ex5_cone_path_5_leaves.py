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
from treespaces.framework.waldspace import WaldSpace
import framework.tools_numerical as numtools

folder = ntpath.splitext(ntpath.basename(__file__))[0]
data_folder = os.path.join(folder, "data")

try:
    os.mkdir(path=folder)
except FileExistsError:
    pass

try:
    os.mkdir(path=data_folder)
except FileExistsError:
    pass


# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 5
partition = ((0, 1, 2, 3, 4),)
sp0 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
sp1 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
sp2 = Split(n=n, part1=(1, 2), part2=(0, 3, 4))
sp3 = Split(n=n, part1=(0, 3), part2=(1, 2, 4))

split_collection1 = [[Split(n=n, part1=(0,), part2=(1, 2, 3, 4)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3, 4)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3, 4)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2, 4)),
                      Split(n=n, part1=(4,), part2=(0, 1, 2, 3)),
                      sp0,
                      sp1]]

split_collection2 = [[Split(n=n, part1=(0,), part2=(1, 2, 3, 4)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3, 4)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3, 4)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2, 4)),
                      Split(n=n, part1=(4,), part2=(0, 1, 2, 3)),
                      sp3,
                      sp2]]

st1 = Structure(n=n, partition=partition, split_collection=split_collection1)
st2 = Structure(n=n, partition=partition, split_collection=split_collection2)
print(f"The structure 1 of the forest has one component and contains the splits\n{st1.split_collection[0]}.")
print(f"The structure 2 of the forest has one component and contains the splits\n{st2.split_collection[0]}.")

# ----- CONSTRUCTION OF THE WALD SPACE WITH OUR GEOMETRY ----- #
ws = WaldSpace(geometry='wald')

x1 = np.array([0.095, 0.095, 0.3, 0.095, 0.095, 0.1, 0.095])
x2 = np.array([0.095, 0.095, 0.2, 0.095, 0.095, 0.2, 0.095])

wald1 = Wald(n=n, st=st1, x=x1)
wald2 = Wald(n=n, st=st2, x=x2)


# --------- PARAMETERS ---------------- #
calculate = False  # if False then load data, else compute the data
n_points = 17  # should be power of 2 + 1 ...
proj_args = {'gtol': 10 ** -8, 'ftol': 10 ** -10, 'btol': 10 ** -6, 'method': 'global-search'}

params1 = {'proj_args': proj_args, 'n_iter': 10, 'n_points': n_points}
params2 = {'proj_args': proj_args, 'n_points': n_points, 'alg': 'global-straighten-extend'}

filename = f"002_geodesics_{n_points}"
data_file = os.path.join(data_folder, f"{filename}.p")

if calculate:
    # start with symmetric projection then straightening
    print("Calculate starting path for global straightening.")
    start_path = ws.g.s_path(p=wald1, q=wald2, alg='global-symmetric', **params1)
    print("Calculate global straightening path.")
    path1 = ws.g.s_path(p=wald1, q=wald2, alg='global-straightening', start_path=start_path, **params1)
    print("Calculate global straightening and extend path.")
    path2 = ws.g.s_path(p=wald1, q=wald2, **params2)
    print(f"Store the data that has been calculated.")
    pickle.dump(obj=(start_path, path1, path2), file=open(data_file, "wb"))
else:
    try:
        print(f"Load the already stored data.")
        start_path, path1, path2 = list(pickle.load(file=open(data_file, "rb")))
    except FileNotFoundError:
        raise FileNotFoundError(f"File with name {data_file} does not exist, either typo or not calculated yet.")


# plot the paths in the square
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
start_path_coords = numtools.give_bhv_coords_n5_interior_splits(start_path, sp0, sp1, sp2, sp3)
path1_coords = numtools.give_bhv_coords_n5_interior_splits(path1, sp0, sp1, sp2, sp3)
path2_coords = numtools.give_bhv_coords_n5_interior_splits(path2, sp0, sp1, sp2, sp3)

plt.scatter(*start_path_coords, color='red', label=f"sym, e={np.round(numtools.energy(start_path, ws.g), 4)}")
plt.scatter(*path1_coords, color='blue', label=f"p-s, e={np.round(numtools.energy(path1, ws.g), 4)}")
plt.scatter(*path2_coords, color='black', label=f"p-s-e, e={np.round(numtools.energy(path2, ws.g), 4)}")

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlim((-0.5, 0.5))
plt.ylim((-0.5, 0.5))
plt.title("Geodesics")
plt.legend()
plt.savefig(os.path.join(folder, f"{filename}.png"), dpi=300)
plt.clf()
plt.cla()
plt.close(fig)
