from tools.structure_and_split import Structure, Split
from tools.wald import Wald
from spaces.treespace_spd_af1 import TreeSpdAf1 as Spd
import numpy as np
import matplotlib.pyplot as plt


def plot_hist(g):
    numbers = [Spd.a_dist(p=g[_i], q=g[_i+1], squared=False) for _i in range(len(g) - 1)]
    plt.hist(numbers)
    plt.title("Distances")
    plt.xlabel("Value")
    plt.ylabel("Occurrences")
    plt.show()
    return


n = 5
partition = ((1, 2, 0), (3, 4))
split_collection = [(Split(n=n, part1=(2,), part2=(0, 1)), Split(n=n, part1=(0,), part2=(1, 2))),
                    (Split(n=n, part1=(4,), part2=(3,)),)]

struct = Structure(partition=partition, split_collection=split_collection, n=n)
print(struct.split_collection)

# _p = np.eye(5)
# x = Spd.w_proj(_p=_p, structure=struct)

x = np.array([0.3, 0.4, 0.2])
chart = Spd.s_chart(st=struct)
y = chart(x=x)
print(y)

v = np.array([0.1, -0.1, -0.1])
p = Wald(st=struct, n=n, x=x)

t_max = 2
q1 = Spd.s_exp(v=v, p=p, t_max=t_max, max_step=10 ** -3)
q2 = Spd.s_exp(v=v, p=p, t_max=t_max, max_step=10 ** -3)

n_points = 30

g_naive = Spd.s_path(p=p, q=q1, n_points=n_points, alg='naive')
# plot_hist(g=g_naive)
g_symmetric = Spd.s_path(p=p, q=q1, n_points=n_points, alg='symmetric')
# plot_hist(g=g_symmetric)

print(p.x)
print(q1.x)
print(f"The length of the naive geodesic: {Spd.length(path_=g_naive)}.")
print(f"The length of the symme geodesic: {Spd.length(path_=g_symmetric)}.")

g_variational = Spd.s_path(p=p, q=q1, n_points=n_points, alg='variational', n_iter=30)
# plot_hist(g=g_variational)
print(f"The length of the varia geodesic: {Spd.length(path_=g_variational)}.")
