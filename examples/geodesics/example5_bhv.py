import os
import sys
import ntpath
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import WaldSpace

dir_handle = ntpath.splitext(ntpath.basename(__file__))[0]
name_handle = dir_handle + "-p1"

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ---- #
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

split_collection3 = [[Split(n=n, part1=(0,), part2=(1, 2, 3, 4)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3, 4)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3, 4)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2, 4)),
                      Split(n=n, part1=(4,), part2=(0, 1, 2, 3)),
                      Split(n=n, part1=(3, 4), part2=(0, 1, 2)),
                      Split(n=n, part1=(1, 2), part2=(0, 3, 4))]]

st1 = Structure(n=n, partition=partition, split_collection=split_collection1)
st2 = Structure(n=n, partition=partition, split_collection=split_collection2)
st3 = Structure(n=n, partition=partition, split_collection=split_collection3)

print(f"The structure 1 of the forest has one component and contains the splits\n{st1.split_collection[0]}.")
print(f"The structure 2 of the forest has one component and contains the splits\n{st2.split_collection[0]}.")
print(f"The structure 3 of the forest has one component and contains the splits\n{st3.split_collection[0]}.")

# ----- CONSTRUCTION OF THE WALD SPACE WITH OUR GEOMETRY ----- #
ws = WaldSpace(geometry='bhv')

x1 = np.array([0.095, 0.095, 0.095, 0.095, 0.25, 0.39, 0.095])
x2 = np.array([0.095, 0.095, 0.26, 0.095, 0.095, 0.41, 0.095])
x3 = np.array([0.095, 0.095, 0.26, 0.095, 0.095, 0.26, 0.095])

wald1 = Wald(n=n, st=st1, x=x1)
wald2 = Wald(n=n, st=st2, x=x2)
# list the interior splits and their respective indices in the structure.
sp0 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
sp1 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
sp2 = Split(n=n, part1=(0, 3), part2=(1, 2, 4))
sp3 = Split(n=n, part1=(1, 2), part2=(0, 3, 4))

calculate = False
# n_points - 1 needs to be divisible by 3.
n_points = 16

geodesic = ws.g.s_path(p=wald1, q=wald2, n_points=n_points)

