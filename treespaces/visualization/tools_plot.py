# external imports
import io
import numpy as np
import itertools as it
from collections import abc
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio import Phylo

from treespaces.tools.wald import Wald
import treespaces.framework.tools_io as iotools

matplotlib.rcParams['text.usetex'] = True
# matplotlib.style.use('seaborn')


def plot_wald(wald: Wald, labels=None, fn=None, round_decimals=5, root='0'):
    graph = iotools.out_newick(wald=wald, labels=labels, round_decimals=round_decimals)
    tree = Phylo.read(io.StringIO(graph), format='newick')
    tree.rooted = True
    # TODO: find a decent graphical representation, rooting at midpoint might be unstable for larger trees.
    # for leaf in tree.get_terminals():
    #     print(leaf)
    # print(leaf for leaf in tree.get_terminals())
    # clade = [leaf for leaf in tree.get_terminals() if leaf.name == root][0]
    # tree.root_with_outgroup(clade)
    tree.root_at_midpoint()
    iotools.plot_tree(tree, fn=fn)
    return


class PlotSpd2Dims(object):
    """
    Plotting functions for elements of spd space in 2 dimensions.

    For two dimensions, elements of the spd space look like the following:

    p = [[a, c], [c, b]] with three parameters such that a,b > 0 and a*b > c**2.

    Thus we can embed the space into the Euclidean space with 3 dimensions, where
    the x-axis and y-axis are a and b, respectively, and the z-axis is parameter c.
    """

    def __init__(self, boundary=True, **kwargs):
        """ Initialize the plotting procedure with or without boundary of spd space."""
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        boundary_options = kwargs['boundary_options'] if 'boundary_options' in kwargs else dict()
        if boundary:
            self._plot_boundary(**boundary_options)

    def pass_curve(self, curve, lw=1, label=None, color=None):
        xs = [p[0, 0] for p in curve]
        ys = [p[1, 1] for p in curve]
        zs = [p[0, 1] for p in curve]
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        self.ax.plot3D(xs=xs, ys=ys, zs=zs, lw=lw, label=label, color=color)

    def pass_curves(self, curves, lw=1, label=None, color=None):
        """ Plot all curves (can be matrices or forests) given in a list."""
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        self.pass_curve(curves[0], lw=lw, label=label, color=color)
        [self.pass_curve(curve, lw=lw, label=None, color=color) for curve in curves[1:]]

    def pass_point(self, point, marker='o', label=None, color=None):
        xs = [point[0, 0]]
        ys = [point[1, 1]]
        zs = [point[0, 1]]
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        self.ax.scatter(xs=xs, ys=ys, zs=zs, marker=marker, label=label, color=color)

    def pass_points(self, points, marker='o', label=None, color=None):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        self.pass_point(points[0], marker=marker, label=label, color=color)
        [self.pass_point(point, marker=marker, label=None, color=color) for point in points[1:]]

    def pass_wireframe(self, x, y, z, color, lw, label=None, rstride=10, cstride=10, alpha=0.5):
        self.ax.plot_wireframe(x, y, z, color=color, lw=lw, rstride=rstride, cstride=cstride, alpha=alpha, label=label)

    def _plot_boundary(self, ax_labels=True, b_max=2, color='#137f83', lw=0.05, rstride=10, cstride=10, alpha=0.5):
        x, y = np.meshgrid(np.linspace(0, b_max, 200), np.linspace(0, b_max, 200))
        z_low, z_up = -np.sqrt(x * y), np.sqrt(x * y)
        self.ax.plot_wireframe(x, y, z_low, color=color, lw=lw, rstride=rstride, cstride=cstride, alpha=alpha)
        self.ax.plot_wireframe(x, y, z_up, color=color, lw=lw, rstride=rstride, cstride=cstride, alpha=alpha)
        if ax_labels is not None:
            if type(ax_labels) is bool:
                ax_labels = [r"$p_{11}$", r"$p_{22}$", r"$p_{12}=p_{21}$"]
            self.ax.set_xlabel(ax_labels[0])
            self.ax.set_ylabel(ax_labels[1])
            self.ax.set_zlabel(ax_labels[2])

    def text(self, point, text):
        xs = point[0, 0]
        ys = point[1, 1]
        zs = point[0, 1]
        self.ax.text(xs, ys, zs, text, color='black')

    @staticmethod
    def show(leg=False):
        if leg:
            plt.legend()
        plt.show()

    @staticmethod
    def savefig(fn, dpi=200):
        plt.savefig(fn, dpi=dpi)


class Plot3Embedded(object):
    """
    Consider three leaves, then the forest matrices look like

    R = [[1, a, b], [a, 1, c], [b, c, 1]], for a,b,c in [0,1)

    with the conditions:
    a >= b*c
    b >= a*c
    c >= a*b,

    transform those into surface conditions
    1) c <= a/b
    2) c <= b/a
    3) c >= a*b.

    We want to plot these three parameters and look how the parameter space
    looks like.
    """

    def __init__(self, bounds=True, forests_marked=True, title="", alpha=0.2):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax.set_box_aspect([1, 1, 1])
        self.points = []
        self.curves = []
        self.ax.text2D(0.05, 0.95, title, transform=self.ax.transAxes)

        if bounds:
            int01 = np.linspace(10 ** -5, 1, 100, endpoint=True)
            a, b = np.meshgrid(int01, int01)
            surf_up = np.minimum(np.minimum(a / b, b / a), 1)
            surf_down = a * b
            self.ax.plot_wireframe(a, b, surf_up, color='orange', lw=0.3, alpha=alpha)
            self.ax.plot_wireframe(a, b, surf_down, color='orange', lw=0.3, alpha=alpha)
            self.ax.set_xlim(-0.1, 1.1)
            self.ax.set_xlabel(r"$\rho_{12}$")
            self.ax.set_ylim(-0.1, 1.1)
            self.ax.set_ylabel(r"$\rho_{13}$")
            self.ax.set_zlim(-0.1, 1.1)
            self.ax.set_zlabel(r"$\rho_{23}$")

        if forests_marked:
            zeros = np.zeros((2,))
            int01 = np.linspace(10 ** -5, 1, 2, endpoint=True)
            # forest 1,2 vs 3
            a, b, c = int01, zeros, zeros
            self.ax.plot(a, b, zs=c, label='', color='black', lw=1)
            # forest 1,3 vs 2
            a, b, c = zeros, int01, zeros
            self.ax.plot(a, b, zs=c, label='', color='black', lw=1)
            # forest 2,3 vs 1
            a, b, c = zeros, zeros, int01
            self.ax.plot(a, b, zs=c, label='', color='black', lw=1)
            # complete forest 1 vs 2 vs 3
            a, b, c = [0], [0], [0]
            self.ax.scatter(a, b, zs=c, label='', color='black', s=3)

    def pass_point(self, point: [Wald, np.ndarray, abc.Iterable], marker='o', label='', text='', color=None,
                   offset=(0, 0, 0), textcolor=None):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        if isinstance(point, Wald):
            x, y, z = tuple(point.corr[(0, 0, 1), (1, 2, 2)])
        elif isinstance(point, np.ndarray):
            x, y, z = tuple(point[(0, 0, 1), (1, 2, 2)])
        else:
            x, y, z = tuple(point)
        if text:
            textcolor = color if textcolor is None else textcolor
            self.ax.text(x + offset[0], y + offset[1], z + offset[2], text, color=textcolor)
        self.ax.scatter(x, y, z, marker=marker, label=label, color=color)

    def pass_points(self, points, marker='o', label=None, text=None, color=None):
        params = [[param] * len(points) if not isinstance(param, list) else param
                  for param in [marker, label, text, color]]
        for point, _m, _l, _t, _c in zip(points, *params):
            self.pass_point(point, _m, _l, _t, _c)

    def pass_curve(self, curve, lw=1, label=None, color=None, alpha=1):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        if all(isinstance(p, Wald) for p in curve):
            xs, ys, zs = [p.corr[0, 1] for p in curve], [p.corr[0, 2] for p in curve], [p.corr[1, 2] for p in curve]
        else:
            xs, ys, zs = [p[0, 1] for p in curve], [p[0, 2] for p in curve], [p[1, 2] for p in curve]
        self.ax.plot3D(xs=xs, ys=ys, zs=zs, lw=lw, label=label, color=color, alpha=alpha)

    def pass_curves(self, curves, lw=1, label=None, color=None, alpha=1):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        params = [[param] * len(curves) if not isinstance(param, list) else param
                  for param in [lw, label, color, alpha]]
        for curve, _lw, _l, _c, _a in zip(curves, *params):
            self.pass_curve(curve, _lw, _l, _c, _a)

    def pass_curve_family(self, curves, lw=1, label=None, color=None, alpha=1):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        self.pass_curves(curves=curves[:-1], lw=lw, label=None, color=color, alpha=alpha)
        self.pass_curve(curve=curves[-1], lw=lw, label=label, color=color, alpha=alpha)

    def pass_point_family(self, points, marker='o', label=None, text=None, color=None):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        self.pass_points(points=points[:-1], marker=marker, label=None, text=text, color=color)
        self.pass_point(point=points[-1], marker=marker, label=label, text=text, color=color)

    def plot_slice(self, level=0.5, m=150, color='blue', alpha=0.4, lw=1, cstride=2, rstride=2, label=None):
        """ Plots the slice at level between 0 and 1 into the 3d figure of the wald space. """
        # zero dimensional boundary of slice
        xs, ys, zs = [level, 0, 0], [0, level, 0], [0, 0, level]
        self.ax.scatter(xs=xs, ys=ys, zs=zs, color=color)
        # one dimensional boundary of slice
        interval01 = np.linspace(10 ** -3, 1, m, endpoint=False)
        y_dim = [(x ** 2 - np.sqrt(x ** 4 - 4 * (x ** 2 - x) * (1 - level))) / 2 / (x ** 2 - x) for x in interval01]
        xs = [1 - x if 0 <= x <= 1 else np.nan for x in interval01]
        ys = [1 - y if 0 <= y <= 1 else np.nan for y in y_dim]
        zs = [(1 - x) * (1 - y) if 0 <= (1 - x) * (1 - y) <= 1 else np.nan for x, y in zip(interval01, y_dim)]
        self.ax.plot3D(xs=xs, ys=ys, zs=zs, lw=lw, color=color)
        self.ax.plot3D(xs=xs, ys=zs, zs=ys, lw=lw, color=color)
        self.ax.plot3D(xs=zs, ys=ys, zs=xs, lw=lw, color=color, label=label)
        # three dimensional slice.
        a, b = np.meshgrid(interval01, interval01)
        c = 1 - (1 - level) / (1 - a) / (1 - b)
        for i in range(m):
            for j in range(m):
                if c[i, j] > np.minimum(a[i, j] / b[i, j], b[i, j] / a[i, j]):
                    a[i, j] = b[i, j] = c[i, j] = np.nan
                if c[i, j] < a[i, j] * b[i, j]:
                    a[i, j] = b[i, j] = c[i, j] = np.nan
        self.ax.plot_wireframe(a, b, c, color=color, lw=lw, alpha=alpha, cstride=cstride, rstride=rstride)
        return

    # def pass_tangent_vectors(self, vectors, p, length=0.2):
    #     """ Plot all tangent vectors at the point with some length. """
    #     lines = [[p + t * v for t in np.linspace(0, length, 3)] for v in vectors]
    #     self.pass_curves(lines)

    @staticmethod
    def show():
        plt.legend()
        plt.show()

    @staticmethod
    def legend(**kwargs):
        leg = plt.legend(**kwargs)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_lw(2)
        return leg

    @staticmethod
    def savefig(fn, **kwargs):
        plt.savefig(fname=fn, **kwargs)

    @staticmethod
    def close():
        plt.cla()
        plt.clf()
        plt.close()


class Plot3Coordinates(object):
    """
    For N = 3, wald space has at most three dimensions, and we can plot the coordinates.
    """

    def __init__(self, bounds=True, forests_marked=True, title="", alpha=0.2):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.text2D(0.05, 0.95, title, transform=self.ax.transAxes)
        # self.ax.set_xlim(0.0, 1.0)
        self.ax.set_xlabel(r"$\lambda_1$")
        # self.ax.set_ylim(0.0, 1.0)
        self.ax.set_ylabel(r"$\lambda_2$")
        # self.ax.set_zlim(0.0, 1.0)
        self.ax.set_zlabel(r"$\lambda_3$")
        if bounds:
            # draw cube
            r = [0, 1]
            for s, e in it.combinations(np.array(list(it.product(r, r, r))), 2):
                if np.sum(np.abs(s - e)) == (r[1] - r[0]):
                    self.ax.plot3D(*zip(s, e), color="black", lw=1)

    def pass_point(self, point: [Wald, abc.Iterable], marker='o', label='', text='', color=None):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        x, y, z = tuple(point.x) if isinstance(point, Wald) else tuple(point)
        if text:
            self.ax.text(x, y, z, text, color='black')
        self.ax.scatter(x, y, z, marker=marker, label=label, color=color)

    def pass_points(self, points, marker='o', label=None, text=None, color=None):
        params = [[param] * len(points) if not isinstance(param, list) else param
                  for param in [marker, label, text, color]]
        for point, _m, _l, _t, _c in zip(points, *params):
            self.pass_point(point, _m, _l, _t, _c)

    def pass_curve(self, curve, lw=1, label=None, color=None, alpha=1):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        if all(isinstance(p, Wald) for p in curve):
            xs, ys, zs = [p.x[0] for p in curve], [p.x[1] for p in curve], [p.x[2] for p in curve]
        else:
            xs, ys, zs = [p[0] for p in curve], [p[1] for p in curve], [p[2] for p in curve]
        self.ax.plot3D(xs=xs, ys=ys, zs=zs, lw=lw, label=label, color=color, alpha=alpha)

    def pass_curves(self, curves, lw=1, label=None, color=None, alpha=1):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        params = [[param] * len(curves) if not isinstance(param, list) else param
                  for param in [lw, label, color, alpha]]
        for curve, _lw, _l, _c, _a in zip(curves, *params):
            self.pass_curve(curve, _lw, _l, _c, _a)

    def pass_curve_family(self, curves, lw=1, label=None, color=None, alpha=1):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        self.pass_curves(curves=curves[:-1], lw=lw, label=None, color=color, alpha=alpha)
        self.pass_curve(curve=curves[-1], lw=lw, label=label, color=color, alpha=alpha)

    def pass_point_family(self, points, marker='o', label=None, text=None, color=None):
        color = color if color is not None else next(self.ax._get_lines.prop_cycler)['color']
        self.pass_points(points=points[:-1], marker=marker, label=None, text=text, color=color)
        self.pass_point(point=points[-1], marker=marker, label=label, text=text, color=color)

    @staticmethod
    def show():
        plt.legend()
        plt.show()

    @staticmethod
    def savefig(fn, dpi=300):
        plt.legend()
        plt.savefig(fn, dpi=dpi)

    @staticmethod
    def close():
        plt.cla()
        plt.clf()
        plt.close()


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_xlim(0.0, 1.0)
# ax.set_xlabel(r"$\lambda_1$")
# ax.set_ylim(0.0, 1.0)
# ax.set_ylabel(r"$\lambda_2$")
# ax.set_zlim(0.0, 1.0)
# ax.set_zlabel(r"$\lambda_3$")
#
#
# xs = [_p.x[st.where(sp0)] for _p in calc_path]
# ys = [_p.x[st.where(sp1)] for _p in calc_path]
# zs = [_p.x[st.where(sp2)] for _p in calc_path]
# ax.plot(xs=xs, ys=ys, zs=zs, lw=1, label='Shortest path', color='blue', alpha=1)
#
# # plot p
# ax.text(walds[0].x[st.where(sp0)], walds[0].x[st.where(sp1)], walds[0].x[st.where(sp2)], r"\large$W^{(1)}$", color='black')
# ax.text(walds[1].x[st.where(sp0)], walds[1].x[st.where(sp1)], walds[1].x[st.where(sp2)], r"\large$W^{(2)}$", color='black')
#
# # remove multiple labels
# # (from https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend)
# handles, labels = plt.gca().get_legend_handles_labels()
# i = 1
# while i < len(labels):
#     if labels[i] in labels[:i]:
#         del (labels[i])
#         del (handles[i])
#     else:
#         i += 1
#
# plt.legend(handles, labels)
# plt.savefig(os.path.join(folder_name, f"geodesic_3d_{n_points_on_path}_points.pdf"))
# plt.clf()
# plt.cla()
# plt.close(fig)

# remove multiple labels
# (from https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend)
# handles, labels = plt.gca().get_legend_handles_labels()
# i = 1
# while i < len(labels):
#     if labels[i] in labels[:i]:
#         del (labels[i])
#         del (handles[i])
#     else:
#         i += 1
#
# plt.legend(handles, labels)