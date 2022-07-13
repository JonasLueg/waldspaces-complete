import os
import sys
import ntpath
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import scipy.linalg as la

# DANGER ALERT: this works only, when this file is in the folder "/examples/distances_leaves_3/example1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import WaldSpace
import treespaces.tools.tools_vector_spaces as vtools

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
    numerator = ws.g.s_inner(u, curv_x.dot(v), p)
    numerator = np.dot(u, curv_x).dot(v)
    denominator = ws.g.s_inner(u, u, p) * ws.g.s_inner(v, v, p) - ws.g.s_inner(u, v, p) ** 2
    # print("New sectional curvature")
    # print(numerator)
    # print(denominator)
    # print(ws.g.s_inner(u, u, p)*ws.g.s_inner(v, v, p))
    # print(ws.g.s_inner(u, v, p) ** 2)
    print(denominator)
    return numerator / denominator


def compute_plane_repr(phi, psi):
    """ phi is between 0 and 2pi and psi is between 0 and pi/2. """
    normal = np.array([np.sin(phi)*np.cos(psi), np.sin(phi)*np.sin(psi), np.cos(phi)])
    plane1 = np.array([np.cos(psi - np.pi/2), np.sin(psi - np.pi/2), 0])
    if np.sum(normal * plane1) > 10**-3:
        print("Something's wrong here.")
    plane2 = np.cross(normal, plane1)
    return plane1, plane2


def angle(u, v, p):
    u = ws.g.s_lift_vector(u, p)
    v = ws.g.s_lift_vector(v, p)
    return 360 / 2 / np.pi * np.arccos(ws.g.a_inner(u, v, ws.g.s_lift(p)))


def ricci(index, basis, p):
    orth_basis = vtools.gram_schmidt(vectors=basis, dot=ws.g.s_inner, p=p)
    print(orth_basis)
    return np.sum([sectional_curvature_waldspace(u=orth_basis[index], v=v, p=p)
                   for i, v in enumerate(orth_basis) if i != index])


x = [0.99, 0.99, 0.99]
p = Wald(n, st=st, x=x)

basis = [e for e in np.eye(3)]
basis = [np.array(e) for e in [[0, 1, 1], [1, 0, 1], [1, 1, 0]]]
ricci_curv = ricci(0, basis, p)
print(ricci_curv)
