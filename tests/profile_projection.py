import cProfile
import pstats
import numpy as np

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1

# p = np.array([[1, 0.01, 0.01, 0.01], [0.01, 1, 0.01, 0.01], [0.01, 0.01, 1, 0.01], [0.01, 0.01, 0.01, 1]])

# ----- CONSTRUCTION OF THE POINTS IN THE INTERIOR OF THE FULLY DIMENSIONAL GROVE ----- #
n = 4
partition = ((0, 1, 2, 3),)
split_collection1 = [[Split(n=n, part1=(0,), part2=(1, 2, 3)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2)),
                      Split(n=n, part1=(0, 1), part2=(2, 3))]]

split_collection2 = [[Split(n=n, part1=(0,), part2=(1, 2, 3)),
                      Split(n=n, part1=(1,), part2=(0, 2, 3)),
                      Split(n=n, part1=(2,), part2=(0, 1, 3)),
                      Split(n=n, part1=(3,), part2=(0, 1, 2)),
                      Split(n=n, part1=(0, 2), part2=(1, 3))]]

st1 = Structure(n=n, partition=partition, split_collection=split_collection1)
st2 = Structure(n=n, partition=partition, split_collection=split_collection2)

# ----- CONSTRUCTION OF THE WALD SPACE WITH OUR GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')

x1 = np.array([0.5, 0.99, 0.7, 0.98, 0.98])
x2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

wald1 = Wald(n=n, st=st1, x=x1)
wald2 = Wald(n=n, st=st2, x=x2)

# x0 = x1 - np.random.rand(len(x1)) / 10
x0 = np.array([0.4025917, 0.95477172, 0.67290398, 0.92991521, 0.90936795])


def do_proj():
    return ws.g.s_proj(p=wald1.corr, st=st1, x0=x0, btol=10 ** -5, gtol=10 ** -10, ftol=10 ** -10,
                       method='global-descent')


if __name__ == "__main__":
    cProfile.run("do_proj()", "restats")
    p = pstats.Stats("restats")
    p.print_stats()
