""" Reconstruction of the cobwebs that we have seen in our paper. """

import numpy as np
import scipy.linalg as la

import treespaces.tools.tools_forest_representation as ftools
from tools.wald import Wald
from treespaces.tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1

# ----- CONSTRUCTION OF THE STARTING POINT ----- #
n = 5
partition = ((0, 1, 2, 3, 4),)
split_collection = [[Split(n=n, part1=(1, 2, 3, 4), part2=(0,)),
                     Split(n=n, part1=(0, 2, 3, 4), part2=(1,)),
                     Split(n=n, part1=(0, 1, 3, 4), part2=(2,)),
                     Split(n=n, part1=(0, 1, 2, 4), part2=(3,)),
                     Split(n=n, part1=(0, 1, 2, 3), part2=(4,)),
                     Split(n=n, part1=(2, 3, 4), part2=(0, 1)),
                     Split(n=n, part1=(3, 4), part2=(0, 1, 2))]]

st = Structure(partition=partition, split_collection=split_collection, n=n)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

# the coordinates of our forest in Nye notation (wald space coordinates):
x = 0.88 * np.ones(7)
x[1] = 0.5
# the point from which we compute the geodesics:
p = Wald(n=n, st=st, x=x)

# ----- CONSTRUCTION OF THE WALD SPACE WITH OUR GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')

# christoffel = ws.g.w_christoffel(st=st)
# print(np.array(christoffel(x=x)))

sect = ws.g.s_sectional_curvature(st=st)
sect_x = sect(x=x)
curv_x = ws.g.s_curvature(st=st)(x=x)
_gradients_x = ws.g.s_gradient(st)(x)
_chart_x_inv = la.inv(ws.g.s_chart(st)(x))

_m_gradients_x = [_chart_x_inv.dot(grad) for grad in _gradients_x]
_gram_matrix = np.array([[np.sum(grad_i * grad_j.T) for grad_i in _m_gradients_x] for grad_j in _m_gradients_x])
_inv_gram_matrix = la.inv(_gram_matrix)

m = len(x)

# for i in range(m):
#     for j in range(m):
#         for k in range(m):
#             for s in range(m):
#                 if i <= j and k <= s and curv_x[i, j, k, s] > 10**-5:
#                     print((i, j, k, s))

ricci_symbols = np.array([[np.sum([[curv_x[i, j, k, s] * _inv_gram_matrix[s, j] for s in range(m)] for j in range(m)])
                           for k in range(m)] for i in range(m)])
scalar_curvature = 1 / m / (m - 1) * np.sum(
    [[ricci_symbols[i, k] * _inv_gram_matrix[i, k] for k in range(m)] for i in range(m)])

print(scalar_curvature)


def curvature(u, v):
    denominator = ws.g.s_norm(v=u, p=p, sq=True) * ws.g.s_norm(v=v, p=p, sq=True) - ws.g.s_inner(u=u, v=v, p=p)
    numerator = np.sum([[u_i ** 2 * v_j ** 2 * sect_x[i][j] for i, u_i in enumerate(u)] for j, v_j in enumerate(v)])
    return numerator / denominator


basis = np.eye(len(x))
_basis = basis.copy()
basis[0] = _basis[1]
basis[1] = _basis[0]
# print(basis)
basis = ftools.vector_space_tools.gram_schmidt(vectors=basis, dot=ws.g.s_inner, p=p)

# print(basis)
curvatures = np.array([[curvature(u=e_i, v=e_j) if i != j else 0
                        for i, e_i in enumerate(basis)] for j, e_j in enumerate(basis)])

for i in range(len(x)):
    for j in range(len(x)):
        if curvatures[i, j] > 0 and i < j:
            print(f"{st.split_collection[0][i]} vs {st.split_collection[0][j]}")

# print(curvatures)
