import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm, Normalize

matplotlib.rcParams['text.usetex'] = True

from treespaces.framework.waldspace import WaldSpace
from treespaces.tools.wald import Wald
from treespaces.tools.structure_and_split import Split, Structure


def p(x):
    return np.sqrt(2) * np.sqrt(1 + (1 - x) ** 2)


def dist(x, y):
    c1 = np.sqrt(2) * np.log((1 - y + p(y) / np.sqrt(2)) / (1 - x + p(x) / np.sqrt(2)))
    a1 = ((p(x) + (1 - x)) ** 2 - 1) / ((p(x) - (1 - x)) ** 2 - 1)
    a2 = ((p(y) + (1 - y)) ** 2 - 1) / ((p(y) - (1 - y)) ** 2 - 1)
    c2 = np.log(a1 / a2) / 2
    return np.abs(c1 + c2)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 18,
})

s = Split(n=2, part1=[0], part2=[1])
st = Structure(n=2, partition=((0, 1),), split_collection=((s,),))
st_inf = Structure(n=2, partition=((0,), (1,)), split_collection=((), ()))
w_inf = Wald(n=2, st=st, x=[1])
x_list = np.linspace(0.1, 0.999, endpoint=True, num=20)


geometries = ['wald', 'euclidean', 'correlation']
for geometry in geometries:
    print(f"Compute stuff in geometry {geometry}.")
    ws = WaldSpace(geometry=geometry)
    paths = [ws.g.s_path(p=Wald(n=2, st=st, x=np.array([x])), q=w_inf, n_points=50,
                         alg="symmetric") for x in x_list]
    lengths = [ws.g.length(path) for path in paths]
    plt.plot(x_list, lengths, label=geometry)

plt.xlim(left=0, right=1)
plt.ylim(bottom=0, top=None)
plt.xlabel(r"$\lambda$")
plt.legend()
plt.savefig("n2_comparison.pdf")
plt.close()

# eps = 10 ** -4
# lambda1 = np.linspace(eps, 1, endpoint=True, num=50)
# # generate 2 2d grids for the x & y bounds
# y, x = np.meshgrid(lambda1, lambda1)
#
# z = dist(x, y)
# for i in range(len(lambda1)):
#     for j in range(len(lambda1)):
#         if i >= j:
#             z[i, j] = np.nan
#
# z_min, z_max = 0, np.abs(z).max()
#
# fig, ax = plt.subplots()
# c = ax.pcolormesh(x, y, z, norm=LogNorm())
# ax.set_title(r"\Large$d_{\mathcal{W}_2}\big(W_1, W_2\big)$")
# # set the limits of the plot to the limits of the data
# ax.axis([x.min(), x.max(), y.min(), y.max()])
# fig.colorbar(c, ax=ax)
# plt.xlabel(r"\Large$\lambda_1$")
# plt.ylabel(r"\Large$\lambda_2$")
# plt.savefig("waldspace-2-dims-distance.pdf")
