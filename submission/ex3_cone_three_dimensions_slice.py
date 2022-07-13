import os
import sys
import pickle
import time
import ntpath
import imageio
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import treespaces.framework.tools_io as iotools
import treespaces.visualization.tools_plot as ptools

folder_name = ntpath.splitext(ntpath.basename(__file__))[0]
try:
    os.mkdir(path=folder_name)
except FileExistsError:
    pass

n_frames = "custom"
try:
    os.mkdir(path=os.path.join(folder_name, f"frames_{n_frames}"))
except FileExistsError:
    pass

try:
    os.mkdir(path=os.path.join(folder_name, f"frames_{n_frames}_pdf"))
except FileExistsError:
    pass

# global settings for changing plots
font = {'family': 'serif',
        'size': 16}
plt.rc('font', **font)
plt.rc('text', **{'usetex': True})

# we insert custom boundaries and compute them here:
int01 = np.linspace(10 ** -5, 1, 30, endpoint=True)
rho12, rho13 = np.meshgrid(int01, int01)
surf_up1 = np.minimum(rho12 / rho13, 1)
surf_up1[rho12 / rho13 > rho13 / rho12] = np.nan
surf_up2 = np.minimum(rho13 / rho12, 1)
surf_up2[rho12 / rho13 < rho13 / rho12] = np.nan
surf_down = rho12 * rho13

# start plotting the slices.
alpha = 0.4
handles, labels = None, None
# levels = (1 - np.linspace(10 ** -2, 1 - 10 ** -3, n_frames, endpoint=False) ** 2)[::-1]
levels = [0.2, 0.87, 0.997]

for i, level in enumerate(levels):
    print(f"Creating slice {i + 1}/{len(levels)} for level {np.round(level, 5)}...")
    level_text = str(np.round(level, 5))
    fig = ptools.Plot3Embedded(alpha=alpha, bounds=False)
    # title=r"\qquad slice $B_a$ for $a = $" + f"{np.round(level, 5)}" + ".")
    fig.ax.plot_wireframe(rho12, rho13, surf_up1, color='orange', lw=0.5, alpha=alpha,
                          label=r"boundary for $\rho_{12} = \rho_{13}\rho_{23}$ or $\rho_{13} = \rho_{12}\rho_{23}$ "
                                r"or $\rho_{23} = \rho_{12}\rho_{13}$")
    fig.ax.plot_wireframe(rho12, rho13, surf_up2, color='orange', lw=0.5, alpha=alpha,
                          label=None)
    fig.ax.plot_wireframe(rho12, rho13, surf_down, color='orange', lw=0.5, alpha=alpha,
                          label=None)
    fig.ax.text(x=0, y=0, z=0, s="(0, 0, 0)")
    fig.ax.text(x=1, y=1, z=1, s="(1, 1, 1)")
    fig.ax.text(x=1, y=0, z=0, s="(1, 0, 0)")
    fig.ax.text(x=0, y=1, z=0, s="(0, 1, 0)")
    fig.ax.text(x=0, y=0, z=1, s="(0, 0, 1)")
    fig.ax.text(x=0.5, y=-0.1, z=0.02, s=r"$\rho_{12}$")
    fig.ax.text(x=-0.05, y=0.5, z=0.02, s=r"$\rho_{13}$")
    fig.ax.text(x=-0.03, y=-0.03, z=0.5, s=r"$\rho_{23}$")
    # forest 1,2 vs 3
    fig.ax.plot([0, 1], [0, 0], zs=[0, 0], color='black', lw=1)
    # forest 1,3 vs 2
    fig.ax.plot([0, 0], [0, 1], zs=[0, 0], color='black', lw=1)
    # forest 2,3 vs 1
    fig.ax.plot([0, 0], [0, 0], zs=[0, 1], color='black', lw=1)
    # complete forest 1 vs 2 vs 3
    fig.ax.scatter([0], [0], [0], color='black', s=3)
    # plot the slices
    fig.plot_slice(level=level, m=150, label="slice")
    fig.ax.grid(False)
    fig.ax.set_axis_off()
    fig.ax.set_aspect('auto')
    if i == 0:
        fig.legend()
        handles, labels = fig.ax.get_legend_handles_labels()
        fig.ax.get_legend().remove()
    fig.ax.view_init(elev=19., azim=97)
    fn = os.path.join(folder_name, f"frames_{n_frames}", f"{iotools.f00(i)}.png")
    fig.savefig(fn, dpi=200, transparent=True)
    fn = os.path.join(folder_name, f"frames_{n_frames}_pdf", f"{iotools.f00(i)}.pdf")
    fig.savefig(fn, dpi=200, transparent=True)
    fig.close()


for handle in handles:
    handle.set_alpha(1)
    handle.set_lw(2)
fig = plt.figure()
fig.legend(handles, labels, ncol=4, prop={'size': 16}, fancybox=True)
fig.savefig(os.path.join(folder_name, f"frames_{n_frames}_pdf", "legend.pdf"),
            dpi=200, bbox_inches='tight', transparent=True)
plt.close()


# print(f"Create the GIF ...")
# with imageio.get_writer(os.path.join(folder_name, f"slices_frames_{n_frames}.gif"), mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)
