# DOES NOT WORK SINCE SPLIT ARGUMENTS HAVE BEEN REITERATED.

import numpy as np

from tools.wald import Wald
from tools.structure_and_split import Structure, Split


n = 5
partition = ((1, 2, 0), (3, 4))
split_collection = [(Split(n=(2,), part1=n, part2=(0, 1)), Split(n=(0,), part1=n, part2=(1, 2))),
                    (Split(n=(4,), part1=n, part2=(3,)),)]

struct = Structure(partition=partition, split_collection=split_collection, n=n)
x = np.array([0.1, 0.4, 0.8])

p = Wald(n=n, st=struct, x=x)

print(p.st)
print(p.x)
print(p.dist)
print(p.corr)

q = Wald(n=n, dist=p.dist)

print(q.st)
print(q.corr)

r = Wald(n=n, corr=p.corr)
print(r.st)
print(r.dist)
print(r.x)

z = Wald(n=n, corr=np.eye(n))
print(z.dist)
print(z.st)
print(z.x)
