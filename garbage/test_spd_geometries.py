import numpy as np

from treespaces.spaces.treespace_bhv import TreeSpaceBhv
from treespaces.spaces.treespace_spd_euclidean import TreeSpdEuclidean as SpdEucl
from treespaces.spaces.treespace_spd_wasserstein import TreeSpdWasserstein as SpdWasserstein
from treespaces.spaces.spd_af1 import SpdAf1


from tools.structure_and_split import Split
from treespaces.tools.wald import Tree


import visualization.tools_plot as ptools

n = 4
s0 = Split(n=n, part1=[0], part2=[1, 2, 3])
s1 = Split(n=n, part1=[1], part2=[0, 2, 3])
s2 = Split(n=n, part1=[2], part2=[0, 1, 3])
s3 = Split(n=n, part1=[3], part2=[0, 1, 2])
sm1 = Split(n=n, part1=[0, 1], part2=[2, 3])
sm2 = Split(n=n, part1=[0, 2], part2=[1, 3])
sm3 = Split(n=n, part1=[0, 3], part2=[1, 2])


a = np.array([[1, 0.8], [0.8, 1]])

b = np.array([[1, 0.2], [0.2, 1]])

path_ = SpdAf1.path(p=a, q=b, n_points=10)

boundary_options = {"lw": 0.5}
plot = ptools.PlotSpd2Dims(boundary=True, boundary_options=boundary_options)
plot.pass_curve(curve=path_, lw=0.8, label="path between a and b llolz")
plot.pass_points([a, b])
plot.show()


bhv = TreeSpaceBhv

splits1 = sorted([(s0, 1), (s1, 4), (s2, 2), (s3, 2), (sm1, 1)])
splits2 = sorted([(s0, 4), (s1, 2), (s2, 1), (s3, 1), (sm1, 0.5)])

t1, t2 = tuple(Tree(n=n, splits=[s[0] for s in sp], b=np.array([s[1] for s in sp])) for sp in [splits1, splits2])

# print(t1.corr)
# print(t2.corr)

# for z in SpdEucl.a_path(p=t1, q=t2):
#     print(z)
#
# print(SpdEucl.length(path_=SpdEucl.a_path(p=t1, q=t2)))
# print(SpdEucl.a_dist(t1, t2))
#
# path_ = SpdEucl.s_path(p=t1, q=t2)
# # print(path_)
#
# a = SpdWasserstein.a_inner(0.5*t1.corr, t2.corr, 2*t1.corr)
# print(a)


# v = np.eye(n)
# a = t1.corr
# print(a)
# b = SpdWasserstein.a_exp(v, a)
# print(b)
# w = SpdWasserstein.a_log(q=b, p=a)
# print(w)

print(t1.x)
a = t1.corr

# b = SpdWasserstein.s_proj(p=a, st=t1.st, ftol=10**-15, gtol=10**-15)

b = t2.corr

v = SpdWasserstein.a_log(q=b, p=a)
print(v)
print(SpdWasserstein.a_dist(a, b))
print(SpdWasserstein.a_norm(v, a))

bb = SpdWasserstein.a_exp(v, a)
print(np.allclose(b, bb))


for m in SpdEucl.s_path(t1, t2, alg="symmetric", n_points=5):
    print(m.x)
