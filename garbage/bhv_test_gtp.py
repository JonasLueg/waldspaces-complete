import numpy as np

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1
import treespaces.tools.tools_bhv_gtp_algorithm as gtp

ws = TreeSpdAf1(geometry='bhv')

n = 5
partition = ((0, 1, 2, 3, 4),)

sp0 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
sp1 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
sp2 = Split(n=n, part1=(1, 2), part2=(0, 3, 4))
sp3 = Split(n=n, part1=(0, 3), part2=(1, 2, 4))
pendants = {Split(n=n, part1=(i,), part2=tuple(j for j in range(n) if j != i)) for i in range(n)}

print("small test for gtp trees with distinct support")
# splits_a = [sp0, sp1]
# splits_b = [sp2, sp3]
weights_a = {sp0: 1, sp1: 2}
weights_b = {sp2: 2, sp3: 1}
support = gtp.gtp_trees_with_distinct_support(weights_a, weights_b)
print(support)

print("small test for gpt trees with common support")
splits_a = [sp0, sp1]
splits_b = [sp2, sp3]
# pendants_a = {Split(n=n, part1=(i,), part2=tuple(j for j in range(n) if j != i)) for i in range(n) if i != 2}
# pendants_b = {Split(n=n, part1=(i,), part2=tuple(j for j in range(n) if j != i)) for i in range(n) if i != 3}
# weights_a = {**{s: 1 for s in pendants_a}, **{s: 1 for s in splits_a}}
# weights_b = {**{s: 1 for s in pendants_b}, **{s: 1 for s in splits_b}}

weights_a = {sp0: 1, sp1: 2}
weights_b = {sp2: 2, sp3: 1}
st_a = Structure(n=n, partition=partition, split_collection=(tuple(weights_a.keys()),))
st_b = Structure(n=n, partition=partition, split_collection=(tuple(weights_b.keys()),))

print(st_a)
print(st_b)
b_a = [weights_a[s] for s in st_a.unravel(st_a.split_collection)]
b_b = [weights_b[s] for s in st_b.unravel(st_b.split_collection)]
p = Wald(n=n, st=st_a, b=b_a)
q = Wald(n=n, st=st_b, b=b_b)
print(f"Wald p: {p.st} with lengths {p.b}.")
print(f"Wald q: {q.st} with lengths {q.b}.")
dist = ws.g.s_dist(p=p, q=q)
print(dist)


print("small test for gtp trees with common support, n = 6, from Owen and Provan paper.")
n = 6
partition = ((0, 1, 2, 3, 4, 5),)

sp0 = Split(n=n, part1=(1, 2), part2=(0, 3, 4, 5))
sp1 = Split(n=n, part1=(3, 4), part2=(0, 1, 2, 5))
sp2 = Split(n=n, part1=(0, 5), part2=(1, 2, 3, 4))

sp3 = Split(n=n, part1=(0, 1), part2=(2, 3, 4, 5))
sp4 = Split(n=n, part1=(2, 3), part2=(0, 1, 4, 5))
sp5 = Split(n=n, part1=(4, 5), part2=(0, 1, 2, 3))

sp6 = Split(n=n, part1=(0,), part2=(1, 2, 3, 4, 5))

pendants_a = {Split(n=n, part1=(i,), part2=tuple(j for j in range(n) if j != i)): np.random.randint(1, 10)
              for i in range(n) if i != 2}
pendants_b = {Split(n=n, part1=(i,), part2=tuple(j for j in range(n) if j != i)): np.random.randint(1, 10)
              for i in range(n) if i != 3}
weights_p = {**{sp0: 4, sp1: 10}, **pendants_a}  # {sp6: 1}  #  sp2: 3
weights_q = {**{sp3: 10, sp4: 4, sp5: 3}, **pendants_b}

st_p = Structure(n=n, partition=partition, split_collection=(tuple(weights_p.keys()),))
st_q = Structure(n=n, partition=partition, split_collection=(tuple(weights_q.keys()),))

b_p = [weights_p[s] for s in st_p.unravel(st_p.split_collection)]
b_q = [weights_q[s] for s in st_q.unravel(st_q.split_collection)]

p = Wald(n=n, st=st_p, b=b_p)
q = Wald(n=n, st=st_q, b=b_q)

print(f"Wald p: {p.st} with lengths {p.b}.")
print(f"Wald q: {q.st} with lengths {q.b}.")

dist = ws.g.s_dist(p=p, q=q)

print(dist)

for t in np.linspace(0, 1, 10):
    r = ws.g.s_path_t(p=p, q=q, t=t)
    print(f"{r.st} with {[np.round(x, 2) for x in r.b]}")

print("another round")

for r in ws.g.s_path(p=p, q=q, n_points=10):
    print(f"{r.st} with {[np.round(x, 2) for x in r.b]}")
