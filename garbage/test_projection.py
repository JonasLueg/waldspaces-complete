import os
import sys
import time
import numpy as np

# DANGER ALERT: this works only, when this file is in the folder "/examples/cobwebs/cobweb1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
print(f"The structure 1 of the forest has one component and contains the splits\n{st1.split_collection[0]}.")
print(f"The structure 2 of the forest has one component and contains the splits\n{st2.split_collection[0]}.")

# ----- CONSTRUCTION OF THE WALD SPACE WITH OUR GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')

x1 = np.array([0.5, 0.99, 0.7, 0.98, 0.98])
x2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

wald1 = Wald(n=n, st=st1, x=x1)
wald2 = Wald(n=n, st=st2, x=x2)

# path_ = ws.g.w_path(p=wald2, q=wald1, alg='global-search')
# print(path_)

print(f"Expected output: {wald1} with structure \n{wald1.st}.")
x0 = x1 - np.random.rand(len(x1)) / 10
print(x0)
start = time.time()
proj_p = ws.g.s_proj(p=wald1.corr, st=st1, x0=x0, btol=10 ** -5, gtol=10 ** -10, ftol=10 ** -10,
                     method='global-descent')
end = time.time()
print(f"Global-descent took {np.round(end - start, 3)} seconds.")
print(f"Output is {proj_p} with structure\n{proj_p.st}.")
print(proj_p)

print(ws.g.a_dist(p=wald1, q=proj_p))
# start = time.time()
# proj_p = ws.g.w_proj(p=wald1.corr, st=st1, x0=x2*9, btol=10 ** -8, gtol=10 ** -8, ftol=10 ** -8,
#                      method='global-search')
# end = time.time()
# print(f"Global-search took {np.round(end - start, 3)} seconds.")


#
# print(proj_p.st)
# print(proj_p.x)

# p = np.array([[1., 0.81, 0.729, 0.729],
#               [0.81, 1., 0.729, 0.729],
#               [0.729, 0.729, 1., 0.81],
#               [0.729, 0.729, 0.81, 1.]])
#
# # [0.1*5] is the point we are searching here.
# print(ws.g.w_proj(p=p, st=st1, btol=10 ** -7, gtol=10 ** -15, ftol=10 ** -10))
# print(ws.g.w_proj(p=p, st=st2, btol=10 ** -7, gtol=10 ** -15, ftol=10 ** -10))
#
# proj_q = ws.g.w_proj(p=p, st=st2, x0=np.array([0.3] * 5), btol=10 ** -7, gtol=10 ** -15, ftol=10 ** -10,
#                      global_search=True)
#
# # give index of split where the entry of the vector is zero
# idx = np.where(proj_q.x == 0)[0]
# if len(idx) >= 1:
#     sp0 = st2.split_collection[0][idx[0]]
#     st_n2, st_n1 = tools.give_nni_candidates(st=st2, sp=sp0)
#
#     split_lengths = {s: proj_q.x[i] for i, s in enumerate(st2.split_collection[0])}
#     print(split_lengths)
#
# x_n1 = np.array([split_lengths[s] if s in st2.split_collection[0] else 10 ** -5 for s in st_n1.split_collection[0]])
#
#     print("next projection")
#     proj_r = ws.g.w_proj(p=p, st=st_n1, x0=x_n1, btol=10 ** -10, gtol=10 ** -15, ftol=10 ** -10, global_search=True)
#     print(proj_r)

# print(proj_q)
# q = np.array([[1.00000269, 0.05354611, 0.01140092, 0.01140092],
#               [0.05354611, 1.00000269, 0.01140092, 0.01140092],
#               [0.01140092, 0.01140092, 1.00000269, 0.05354611],
#               [0.01140092, 0.01140092, 0.05354611, 1.00000269]])
#
# proj_q = ws.g.w_proj(q, x0=np.array([0.8]*len(proj_p.x)), st=st, btol=10 ** -10, gtol=10 ** -15, ftol=10 ** -12)
# print(proj_q)


# # ----- CONSTRUCTION OF THE WALD SPACE WITH OUR GEOMETRY ----- #
# ws = WaldSpace(geometry='wald')
#
# x1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
# x2 = np.array([0.18, 0.3, 0.1, 0.9, 0.1])
#
# wald1 = Wald(n=n, st=st1, x=x1)
# q = wald1.corr # + np.eye(N=n)*0.1
# # wald2 = Wald(n=n, st=st2, x=x2)
#
# proj_st = st1
# start = Wald(n=n, st=proj_st, x=x2)
#
# proj_params = {'method': 'local', 'btol': 10**-4, 'gtol': 10**-15, 'ftol': 10**-13}
# v = ws.g.a_log(q=q, p=start)
#
#
# def f(t):
#     return ws.g.a_dist(p=ws.g.w_proj(p=ws.g.a_exp(v=t*v, p=start), st=proj_st, **proj_params), q=q, squared=True)
#
#
# res = minimize(fun=f, x0=np.array(1), method='Nelder-Mead', options={'gtol': 10**-3})
# print(res.x)
# print(np.sqrt(res.fun))
#
# m = 15
# _path = [ws.g.a_exp(v=t*v, p=start) for t in np.linspace(0, 2, m)]
# _proj = [ws.g.w_proj(p=p, st=proj_st, **proj_params) for p in _path]
# for p in _proj:
#     print(p)
# dists = [ws.g.a_dist(p=ws.g.w_proj(p=p, st=proj_st, **proj_params), q=q, squared=False) for p in _path]
#
# # plot the lengths of the paths
# fig = plt.figure()
# numbers = np.arange(start=0, stop=len(_path))
# plt.plot(np.linspace(0, 2, m), dists)
# plt.scatter(res.x, np.sqrt(res.fun))
# print(dists)
#
# plt.show()
