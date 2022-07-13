import scipy.linalg as la
import numpy as np

from treespaces.spaces.treespace_spd_af1 import TreeSpdAf1
from treespaces.spaces.embed_corr_in_spd_quotient import CorrQuotient

import tools.tools_vector_spaces as vtools
from tools.structure_and_split import Split, Structure
from tools.wald import Wald

np.set_printoptions(precision=3)


def sect_curv(v, w, p):
    p_inv = la.inv(p)
    nominator = np.trace(p_inv.dot(v).dot(p_inv).dot(w).dot(p_inv).dot(v.dot(p_inv).dot(w) - w.dot(p_inv).dot(v)))
    denominator = TreeSpdAf1.a_inner(v, v, p) * TreeSpdAf1.a_inner(w, w, p)
    return nominator / denominator


def sect_curv2(v, w, p):
    """ Computes the sectional curvature of the tangent plane spanned by v, w at the point p. """
    if np.allclose(v, w):
        return 0
    p_inv = la.inv(p)
    _x = np.dot(v, p_inv)
    _y = np.dot(w, p_inv)
    _mixed = np.dot(_x, _y)
    nominator = np.trace(np.dot(_mixed, _mixed) - np.dot(_x, _x).dot(np.dot(_y, _y)))
    denominator = TreeSpdAf1.a_inner(v, v, p) * TreeSpdAf1.a_inner(w, w, p) - TreeSpdAf1.a_inner(v, w, p) ** 2
    return nominator / denominator /2


n = 3

p = np.eye(n)
#p = np.ones((n, n))
#np.fill_diagonal(p, 0)
#p *= 0.5
#p += np.eye(n)*2
# p[1, 1] += 10

print(f"p is\n{p}")

np.random.seed(10)

basis_diagonal = list()
basis_offdiag = list()

noise = False

for i in range(n):
    dum = np.random.standard_normal((n, n))/10 if noise else np.zeros((n, n))
    e_ii = np.zeros((n, n)) + (dum + dum.T)
    e_ii[i, i] = 1
    basis_diagonal.append(e_ii)

for i in range(n):
    for j in range(n):
        if i < j:
            dum = np.random.standard_normal((n, n))/10 if noise else np.zeros((n, n))
            e_ij = np.zeros((n, n)) + (dum + dum.T)
            e_ij[i, j] = e_ij[j, i] = 1
            basis_offdiag.append(e_ij)

basis = basis_diagonal + basis_offdiag

# print(basis)
# basis = vtools.gram_schmidt(vectors=basis, p=p, dot=TreeSpdAf1.a_inner)

print("Basis")
for i, b in enumerate(basis):
    print(f"Basis element {i}:\n{b}.")

print("Curvature")
# print(basis)
curv = np.array([[sect_curv2(e_i, e_j, p) for e_j in basis] for e_i in basis])
print(curv)
