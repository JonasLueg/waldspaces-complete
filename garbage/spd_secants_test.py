import scipy.linalg as la
import numpy as np

from treespaces.spaces.spd_af1 import SpdAf1
from treespaces.spaces.embed_corr_in_spd_quotient import CorrQuotient

from tools.structure_and_split import Split, Structure
from tools.wald import Wald



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
x1 = [0.1, 0.1, 0.2, 0.7, 0.7, 0.4, 0.7]
x2 = [0.7, 0.65, 0.76, 0.9, 0.7, 0.4, 0.1]
x3 = [0.7, 0.7, 0.78, 0.7, 0.7, 0.45, 0.7]

p1, p2, p3 = tuple(Wald(n=n, st=st, x=x) for x in [x1, x2, x3])


def s_angle(u, v, p, g):
    u = u / g.s_norm(u, p)
    v = v / g.s_norm(v, p)
    return 360 / 2 / np.pi * np.arccos(g.s_inner(u=u, v=v, p=p))


def a_angle(u, v, p, g):
    u = u / g.a_norm(u, p)
    v = v / g.a_norm(v, p)
    return 360 / 2 / np.pi * np.arccos(g.a_inner(u=u, v=v, p=p))


g = CorrQuotient
d = np.array([0.1, 0.3, 0.4, 0.8, 0.1])

v = 100*g.s_proj_vector(g.a_log(p2.corr, p3.corr), p3.corr)
np.fill_diagonal(v, -0.5)

np.set_printoptions(precision=5)


p = 2*p1.corr
q = p2.corr


v1 = (q - p) / la.norm(q - p)

num = 100

for i, qt in enumerate([q + t*(p - q) for t in np.linspace(0, 0.9999, num)]):
    w2 = SpdAf1.log(qt, p)
    v2 = w2 / la.norm(w2)
    print(f"Spd direction for iteration {i}/{num} is\n{v2}.")

print(f"Euclidean direction is\n{v1}.")

