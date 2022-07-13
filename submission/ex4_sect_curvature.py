import os
import sys
import ntpath
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la

# DANGER ALERT: this works only, when this file is in the folder "/examples/distances_leaves_3/example1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from treespaces.tools.wald import Wald
from treespaces.tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import WaldSpace

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 3
partition = ((0, 1, 2),)
split_collection = [[Split(n=n, part1=(0,), part2=(1, 2)),
                     Split(n=n, part1=(1,), part2=(0, 2)),
                     Split(n=n, part1=(2,), part2=(0, 1))]]

folder_name = ntpath.splitext(ntpath.basename(__file__))[0]
try:
    os.mkdir(path=folder_name)
except FileExistsError:
    pass
#
# try:
#     os.mkdir(path=os.path.join(folder_name, "plots"))
# except FileExistsError:
#     pass

st = Structure(n=n, partition=partition, split_collection=split_collection)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
ws = WaldSpace(geometry='wald')
curv_symbols = ws.g.s_sectional_curvature(st=st)


def sectional_curvature_waldspace(u, v, p):
    if np.allclose(u, v):
        return 0
    curv_x = curv_symbols(x=p.x)
    numerator = np.dot(u, curv_x).dot(v)
    denominator = ws.g.s_inner(u, u, p) * ws.g.s_inner(v, v, p) - ws.g.s_inner(u, v, p) ** 2
    return numerator / denominator


def compute_plane_repr(phi, psi):
    """ phi is between 0 and 2pi and psi is between 0 and pi/2. """
    normal = np.array([np.sin(phi) * np.cos(psi), np.sin(phi) * np.sin(psi), np.cos(phi)])
    plane1 = np.array([np.cos(psi - np.pi / 2), np.sin(psi - np.pi / 2), 0])
    if np.sum(normal * plane1) > 10 ** -3:
        print("Something's wrong here.")
    plane2 = np.cross(normal, plane1)
    return plane1, plane2


eps = 0.001
int01 = np.linspace(0, 1, 75, endpoint=True)
interval = np.linspace(eps, 0.99, 30, endpoint=True)
# interval = [0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]

n_planes = 20
eps = 10 ** -2
psi_angles = np.linspace(eps, 2 * np.pi, num=4*n_planes, endpoint=False)
phi_angles = np.linspace(-eps, -np.pi / 2, num=n_planes, endpoint=False)
planes = [compute_plane_repr(phi, psi) for phi in phi_angles for psi in psi_angles]

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# for u, v in planes:
#     ax.plot(xs=[u[0], 0, v[0], u[0]], ys=[u[1], 0, v[1], u[1]], zs=[u[2], 0, v[2], u[2]])
# plt.show()

sect_curv_along_curve_min = []
sect_curv_along_curve_max = []

for i, t in enumerate(interval):
    print(f"{i+1}/{len(interval)}. Compute for t = {t}.")
    p = Wald(n=n, st=st, x=[t, t, t])
    sect_curvs = [sectional_curvature_waldspace(u=u, v=v, p=p) for u, v in planes]
    # print(sect_curvs)
    sect_curv_along_curve_max.append(np.max(sect_curvs))
    sect_curv_along_curve_min.append(np.min(sect_curvs))
#
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(interval, sect_curv_along_curve_min, color='blue', lw=1.2, label="minimum")
ax.plot(interval, sect_curv_along_curve_max, color='red', lw=1.2, label="maximum")
ax.set_yscale('symlog')

# print(np.round(np.array(sect_curv_along_curve_min), 6))
# # this is to delete every second y-tick label (as things get messy when it's log scale)
fig.canvas.draw()
labels = [item.get_text() for item in ax.get_yticklabels()]
offset = 1 if ((len(labels) + 1)//2) % 2 == 1 else 0
for i in range(len(labels)):
    if (i + offset) % 2 == 0:
        labels[i] = ""
ax.set_yticklabels(labels)
#
ax.plot(int01, [0 for _ in int01], label="constant 0", lw=1.2)
plt.xlim(0.02, 1.02)
ax.set(xlabel=r"${a}$", ylabel="",
       title=r"sectional curvatures for $W = [T,\lambda]$ with $\lambda=(a, a, a)$")
plt.legend()
plt.savefig(os.path.join(folder_name, "equal_edge_lengths.pdf"))

# t_list = np.linspace(0.01, 0.99, 100, endpoint=False)
# coordinates = [np.array([t, t, t]) for t in t_list]
# walds = [Wald(n=n, st=st, x=x) for x in coordinates]
#
# u = np.array([1, 0, 0])
# v = np.array([0, 1, 0])
#
# curvatures = [sectional_curvature_waldspace(u, v, p) for p in walds]
#
# curvatures_in_spd = [np.array([[sectional_curvature_spdspace(ws.g.s_lift_vector(u, p), ws.g.s_lift_vector(v, p),
#                                                              ws.g.s_lift(p)) for v in np.eye(m)] for u in np.eye(m)])[
#                          0, 1] for p in walds]
# print(curvatures_in_spd)
#
# fig, ax = plt.subplots()
# ax.plot(t_list, curvatures, linewidth=1, label="intrinsic")
# ax.plot(t_list, curvatures_in_spd, linewidth=1, label="extrinsic")
# ax.plot(t_list, [0 for _ in t_list], label="line at 0")
#
# plt.xlim(t_list[0], t_list[-1])
# plt.ylim(-20, 0.5)
# ax.set(xlabel=f"Coordinates value", ylabel=f"Sectional curvatures",
#        title=f"Curvatures at walds with equal coordinates.")
# plt.legend()
# fig.savefig("sectional_curvatures_wald.png", dpi=200)
# plt.show()
