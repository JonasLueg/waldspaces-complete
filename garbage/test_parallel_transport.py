""" Test parallel transport, if it is correct... """

import os
import sys
import numpy as np

# DANGER ALERT: this works only, when this file is in the folder "/examples/distance_to_star_stratum/star_stratum1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
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
# (0, 2, 3, 4)|(1,), (0, 1, 3, 4)|(2,), (0, 1, 2, 4)|(3,), (0, 1, 2, 3)|(4,),
# (0, 1, 2)|(3, 4), (0, 1)|(2, 3, 4), (0,)|(1, 2, 3, 4)

coordinates = [[0.5, 0.5, 0.5, 0.5, 0.8, 0.1, 0.5],
               [0.5, 0.5, 0.5, 0.5, 0.1, 0.9, 0.5],
               [0.1, 0.4, 0.2, 0.8, 0.9, 0.9, 0.9]]

walds = [Wald(n=n, st=st, x=np.array(_x)) for _x in coordinates]

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')

p = walds[0]
q = walds[1]
r = walds[2]

# u = ws.g.a_log(q=q, p=p)
# v = ws.g.a_log(q=r, p=p)
#
# inner = ws.g.a_inner(u=u, v=v, p=p)
#
# u_ = ws.g.a_trans(v=u, p=p, q=q)
# v_ = ws.g.a_trans(v=v, p=p, q=q)
#
# inner_ = ws.g.a_inner(u=u_, v=v_, p=q)
#
# print(f"Inner product of u, v before transport was {inner}.")
# print(f"Inner product of u, v after  transport is  {inner_}.")

u = ws.g.s_proj_vector(v=ws.g.a_log(q=q, p=p), p=p)
v = ws.g.s_proj_vector(v=ws.g.a_log(q=r, p=p), p=p)

inner = ws.g.s_inner(u=u, v=v, p=p)


# u_ = ws.g.a_trans(v=u, p=p, q=q)
# v_ = ws.g.a_trans(v=v, p=p, q=q)
#
# inner_ = ws.g.a_inner(u=u_, v=v_, p=q)

print(f"Inner product of u, v before transport was {inner}.")
# print(f"Inner product of u, v after  transport is  {inner_}.")










