import os
import sys
import pickle
import time
import ntpath
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from treespaces.tools.wald import Wald
from treespaces.tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import WaldSpace
import treespaces.framework.tools_numerical as numtools
import treespaces.framework.tools_io as iotools
import treespaces.visualization.tools_plot as ptools

# ----- THIS FLAG DETERMINES IF THE GEODESICS ARE COMPUTED AGAIN OR LOADED FROM DATA FILES INSTEAD -----
calculate = False

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 3
partition = ((0, 1, 2),)
sp0 = Split(n=n, part1=(0,), part2=(1, 2))
sp1 = Split(n=n, part1=(1,), part2=(0, 2))
sp2 = Split(n=n, part1=(2,), part2=(0, 1))
split_collection = [[sp0, sp1, sp2]]

folder_name = ntpath.splitext(ntpath.basename(__file__))[0]
try:
    os.mkdir(path=folder_name)
except FileExistsError:
    pass

try:
    os.mkdir(path=os.path.join(folder_name, "data"))
except FileExistsError:
    pass

st = Structure(n=n, partition=partition, split_collection=split_collection)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

# the weights are associated according to this order of the splits: ((0, 2)|(1,), (0, 1)|(2,), (0,)|(1, 2))
coordinates = [[0.1, 0.9, 0.07], [0.3, 0.1, 0.9]]

walds = [Wald(n=n, st=st, x=np.array(_x)) for _x in coordinates]

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
ws = WaldSpace(geometry="affine-invariant")
bhv = WaldSpace(geometry="bhv")
# ----- COMPUTATION OF THE PATHS FROM THE CONSTRUCTED WALDS TO THE BOUNDARY (STAR STRATUM) ----- #
n_points_on_path = 65

# these are the parameters used for every projection from ambient space onto the grove
proj_args = {'gtol': 10 ** -9, 'ftol': 10 ** -10, 'btol': 10 ** -7, 'method': 'global-descent'}
# the parameters for each method are set here:
params = {'n_points': n_points_on_path, 'proj_args': proj_args}

if calculate:
    start = time.time()
    calc_path = tuple(ws.g.s_path(p=walds[0], q=walds[1], alg='global-straighten-extend', **params))
    paths = [calc_path]
    for _i in range(5):
        paths.append(tuple(ws.g.s_path(p=walds[0], q=walds[1], alg='global-straightening', start_path=paths[-1],
                                       proj_args=proj_args, n_iter=1)))
    end = time.time()
    t = np.round(end - start, 2)
    print(f"Calculation took {t} seconds.")
    print(f"Store the data that has been calculated.")
    pickle.dump(obj=(tuple(paths), t), file=open(os.path.join(f"{folder_name}", "data",
                                                              f"path_{n_points_on_path}_points.p"), "wb"))
else:
    print(f"Load the already stored data since the flag 'calculate' is False.")
    paths, t = pickle.load(file=open(os.path.join(f"{folder_name}", "data",
                                                  f"path_{n_points_on_path}_points.p"), "rb"))


# compute the BHV space geodesic.
bhv_geodesic = bhv.g.s_path(p=walds[0], q=walds[1], n_points=n_points_on_path)
ws_geodesic = paths[-1]
# determine parts of wald space geodesic that are on the boundary
on_boundary = [not np.all(p.x > 0) for p in paths[-1]]
first_index, last_index = on_boundary.index(True), len(on_boundary) - on_boundary[::-1].index(True)
ws_geodesic_int = [ws_geodesic[:first_index + 1], ws_geodesic[last_index - 1:]]
ws_geodesic_bnd = ws_geodesic[first_index:last_index]

# ----- PLOT THE CALCULATED DATA ----- #
fig = ptools.Plot3Coordinates(bounds=True)
fig.pass_curve_family(curves=ws_geodesic_int, lw=2, label="wald space geodesic interior", color="red")
fig.pass_curve(curve=ws_geodesic_bnd, lw=2, label="wald space geodesic boundary", color="brown")
fig.pass_curve(curve=bhv_geodesic, lw=2, label="bhv space geodesic", color="blue")
names = [r"\large$W^{(1)}$", r"\large$W^{(2)}$"]
fig.pass_points(points=walds, marker='.', text=names)
fig.pass_point(point=(0, 0, 0), marker='.', text=r"(0, 0, 0)", color='black')
fig.pass_point(point=(0, 0, 1), marker='.', text=r"(0, 0, 1)", color='black')
fig.pass_point(point=(0, 1, 0), marker='.', text=r"(0, 1, 0)", color='black')
fig.ax.view_init(elev=18., azim=77)
fig.savefig(os.path.join(folder_name, f"geodesic_3d_{n_points_on_path}_points.pdf"))
# fig.show()
fig.close()

fig = ptools.Plot3Embedded(bounds=True, alpha=0.2)
fig.pass_curve_family(curves=ws_geodesic_int, lw=2, label="wald space geodesic interior", color="red")
fig.pass_curve(curve=ws_geodesic_bnd, lw=2, label="wald space geodesic boundary", color="brown")
fig.pass_curve(curve=bhv_geodesic, lw=2, label="bhv space geodesic", color="blue")
names = [r"\large$W^{(1)}$", r"\large$W^{(2)}$"]
fig.pass_points(points=walds, marker='.', text=names)
fig.ax.view_init(elev=23., azim=135)
fig.savefig(os.path.join(folder_name, f"geodesic_3d_emb_{n_points_on_path}_points.pdf"))
# fig.show()
fig.close()

# try:
#     os.mkdir(path=os.path.join(folder_name, f"vis_{n_points_on_path}_points"))
# except FileExistsError:
#     pass

# for i, p in enumerate(calc_path):
#     ptools.plot_wald(p, labels={0: "1", 1: "2", 2: "3"}, root="a",
#                      fn=os.path.join(folder_name, f"vis_{n_points_on_path}_points",
#                                      f"graph_{iotools.f00(i)}.png"))
