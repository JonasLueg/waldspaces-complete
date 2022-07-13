import os
import sys
import ntpath
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import treespaces.visualization.tools_plot as ptools

folder_name = ntpath.splitext(ntpath.basename(__file__))[0]
try:
    os.mkdir(path=folder_name)
except FileExistsError:
    pass


int01 = np.linspace(10 ** -5, 1, 30, endpoint=True)
rho12, rho13 = np.meshgrid(int01, int01)
surf_up1 = np.minimum(rho12 / rho13, 1)
surf_up1[rho12 / rho13 > rho13 / rho12] = np.nan
surf_up2 = np.minimum(rho13 / rho12, 1)
surf_up2[rho12 / rho13 < rho13 / rho12] = np.nan
surf_down = rho12 * rho13


font = {'family': 'serif',
        'size': 16}
plt.rc('font', **font)
plt.rc('text', **{'usetex': True})

alpha = 0.75
fig = ptools.Plot3Embedded(alpha=alpha, bounds=False)

fig.ax.plot_wireframe(rho12, rho13, surf_up1, color='lightcoral', lw=0.5, alpha=alpha,
                      label=r"$\rho_{12} = \rho_{13}\rho_{23}$")
fig.ax.plot_wireframe(rho12, rho13, surf_up2, color='cornflowerblue', lw=0.5, alpha=alpha,
                      label=r"$\rho_{13} = \rho_{12}\rho_{23}$")
fig.ax.plot_wireframe(rho12, rho13, surf_down, color='gold', lw=0.5, alpha=alpha,
                      label=r"$\rho_{23} = \rho_{12}\rho_{13}$")

fig.ax.plot(xs=[0, 1], ys=[0, 1], zs=[1, 1], color='black', lw=1, alpha=1)
fig.ax.plot(xs=[0, 1], zs=[0, 1], ys=[1, 1], color='black', lw=1, alpha=1)
fig.ax.plot(zs=[0, 1], ys=[0, 1], xs=[1, 1], color='black', lw=1, alpha=1)
fig.ax.text(x=0, y=0, z=0, s="(0, 0, 0)")
fig.ax.text(x=1, y=1, z=1, s="(1, 1, 1)")
fig.ax.text(x=1, y=0, z=0, s="(1, 0, 0)")
fig.ax.text(x=0, y=1, z=0, s="(0, 1, 0)")
fig.ax.text(x=0, y=0, z=1, s="(0, 0, 1)")
fig.ax.text(x=0.5, y=-0.1, z=0.02, s=r"$\rho_{12}$")
fig.ax.text(x=-0.05, y=0.5, z=0.02, s=r"$\rho_{13}$")
fig.ax.text(x=-0.03, y=-0.03, z=0.5, s=r"$\rho_{23}$")


# forest 1,2 vs 3
fig.ax.plot([0, 1], [0, 0], zs=[0, 0], label=r'$\rho_{13} = \rho_{23} = 0$', color='red', lw=2)
# forest 1,3 vs 2
fig.ax.plot([0, 0], [0, 1], zs=[0, 0], label=r'$\rho_{12} = \rho_{23} = 0$', color='blue', lw=2)
# forest 2,3 vs 1
fig.ax.plot([0, 0], [0, 0], zs=[0, 1], label=r'$\rho_{12} = \rho_{13} = 0$', color='green', lw=2)
# complete forest 1 vs 2 vs 3
fig.ax.scatter([0], [0], [0], label=r'$\rho_{12} = \rho_{13} = \rho_{23} = 0$', color='black', s=5)

# fig.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
fig.ax.grid(False)

# Hide axes ticks
fig.ax.set_axis_off()
fig.ax.set_aspect('auto')
fig.legend(prop={'size': 14}, loc='upper left', ncol=2, fancybox=True)
handles, labels = fig.ax.get_legend_handles_labels()
fig.ax.get_legend().remove()

# save first view:
fig.ax.view_init(elev=42., azim=49)
fig.savefig(os.path.join(folder_name, "cone_front.pdf"), dpi=200, transparent=True)

# save second view:
fig.ax.view_init(elev=19., azim=171)
fig.savefig(os.path.join(folder_name, "cone_side.pdf"), dpi=200, transparent=True)

# save third view:
fig.ax.view_init(elev=-27., azim=-125)
fig.savefig(os.path.join(folder_name, "cone_back.pdf"), dpi=200, transparent=True)
fig.close()


for handle in handles:
    handle.set_alpha(1)
    handle.set_lw(2)
fig = plt.figure()
fig.legend(handles, labels, ncol=4, prop={'size': 16}, fancybox=True)
fig.savefig(os.path.join(folder_name, "cone_legend.pdf"), dpi=200, bbox_inches='tight', transparent=True)
plt.close()
