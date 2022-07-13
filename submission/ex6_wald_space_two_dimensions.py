import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm, Normalize
matplotlib.rcParams['text.usetex'] = True

eps = 10 ** -4
lambda1 = np.linspace(eps, 1, endpoint=True, num=300)


def p(x):
    return np.sqrt(2) * np.sqrt((1 - x) ** 2 + 1)


def dist(x, y):
    c1 = np.log((1 - y + p(y) / np.sqrt(2)) / (1 - x + p(x) / np.sqrt(2)))
    a1 = ((p(x) + (1 - x)) ** 2 - 1) / ((p(x) - (1 - x)) ** 2 - 1)
    a2 = ((p(y) + (1 - y)) ** 2 - 1) / ((p(y) - (1 - y)) ** 2 - 1)
    c2 = np.log(a1 / a2) / 2 / np.sqrt(2)
    return np.abs(c1 + c2)


plt.plot(lambda1, dist(1, lambda1))
plt.ylim(0.0, plt.ylim()[1])
plt.xlabel(r"\Large$\lambda_1$")
plt.ylabel(r"\Large$d_{\mathcal{W}_2}(W_1, I_2)$")
plt.savefig("waldspace-2-dims-distance_to_inf.pdf")
plt.close()


eps = 10 ** -4
lambda1 = np.linspace(eps, 1, endpoint=True, num=50)
# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(lambda1, lambda1)

z = dist(x, y)
for i in range(len(lambda1)):
    for j in range(len(lambda1)):
        if i >= j:
            z[i, j] = np.nan

z_min, z_max = 0, np.abs(z).max()

fig, ax = plt.subplots()
c = ax.pcolormesh(x, y, z, norm=LogNorm())
ax.set_title(r"\Large$d_{\mathcal{W}_2}\big(W_1, W_2\big)$")
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.xlabel(r"\Large$\lambda_1$")
plt.ylabel(r"\Large$\lambda_2$")
plt.savefig("waldspace-2-dims-distance.pdf")
