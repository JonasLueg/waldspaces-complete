""" Reconstruction of the cobwebs that we have seen in our paper. """

import os
import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# DANGER ALERT: this works only, when this file is in the folder "/examples/cobwebs/cobweb1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1

# ----- CONSTRUCTION OF THE STARTING POINT ----- #
n = 3
partition = ((0, 1, 2),)
split_collection = [[Split(n=n, part1=(1, 2), part2=(0,)),
                     Split(n=n, part1=(0, 2), part2=(1,)),
                     Split(n=n, part1=(0, 1), part2=(2,))]]

st = Structure(partition=partition, split_collection=split_collection, n=n)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

# the coordinates of our forest in Nye notation (wald space coordinates):
x = 0.5 * np.ones(3)

# the point from which we compute the geodesics:
p = Wald(n=n, st=st, x=x)

# ----- CONSTRUCTION OF THE WALD SPACE WITH OUR GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')

eps = 10 ** - 3
x_list = [np.array([0.5, 0.5, y]) for y in np.arange(start=eps, stop=0.96, step=eps)]
x_curvs = []
for x in x_list:
    curv_x = ws.g.s_curvature(st=st)(x=x)
    _gradients_x = ws.g.s_gradient(st)(x)
    _chart_x_inv = la.inv(ws.g.s_chart(st)(x))

    _m_gradients_x = [_chart_x_inv.dot(grad) for grad in _gradients_x]
    _gram_matrix = np.array([[np.sum(grad_i * grad_j.T) for grad_i in _m_gradients_x] for grad_j in _m_gradients_x])
    _inv_gram_matrix = la.inv(_gram_matrix)

    m = len(x)

    ricci_symbols = np.array(
        [[np.sum([[curv_x[i, j, k, s] * _inv_gram_matrix[s, j] for s in range(m)] for j in range(m)])
          for k in range(m)] for i in range(m)])
    scalar_curvature = 1 / m / (m - 1) * np.sum(
        [[ricci_symbols[i, k] * _inv_gram_matrix[i, k] for k in range(m)] for i in range(m)])
    x_curvs.append(scalar_curvature)

plt.plot([x[2] for x in x_list], x_curvs)
plt.plot([0, 1], [0, 0])
plt.xlim(0, 1)
plt.xlabel("length of each edge in Nye notation")
plt.ylabel("scalar curvature")
plt.title("Scalar curvature at star trees, 3 leaves.")
plt.savefig("scalar_curvature_3_leaves_one_edge.png")
