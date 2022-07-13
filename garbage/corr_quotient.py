import numpy as np

from visualization.tools_plot import PlotSpd2Dims
from treespaces.spaces.embed_corr_in_spd_quotient import CorrQuotient

boundary_options = {'lw': 0.5}
plt = PlotSpd2Dims(boundary=True, boundary_options=boundary_options)

unit = np.eye(2)
plt.pass_point(unit, label=None)
plt.text(point=unit, text=r"$I$")

corr_space = [np.array([[1, -1], [-1, 1]]), np.array([[1, 1], [1, 1]])]
plt.pass_curve(corr_space, lw=1.5, label="correlation space")

c1 = np.array([[1, 0.5], [0.5, 1]])
c2 = np.array([[1, -0.7], [-0.7, 1]])

plt.pass_point(point=c1, label=None)
plt.text(point=c1, text=r'$C_1$')
plt.pass_point(point=c2, label=None)
plt.text(point=c2, text=r'$C_2$')


geodesic_c1_c2 = CorrQuotient.a_path(p=c1, q=c2, n_points=20)
plt.pass_curve(curve=geodesic_c1_c2, lw=1, label=r"geodesic between $C_1$ and $C_2$")

low = 0.1
up = 1.4
fine = 100

dx, dy = np.meshgrid(np.linspace(low, up, fine), np.linspace(low, up, fine))
dx = dx.flatten()
dy = dy.flatten()

c1_fiber = np.array([CorrQuotient.grp_action(p=c1, d=np.array([dx[i], dy[i]])) for i in range(dx.shape[0])])

xp = c1_fiber[:, 0, 0].reshape((fine, fine))
yp = c1_fiber[:, 1, 1].reshape((fine, fine))
zp = c1_fiber[:, 0, 1].reshape((fine, fine))

plt.pass_wireframe(xp, yp, zp, color="green", lw=0.2, alpha=0.8, rstride=2, cstride=2, label=r"fiber of $C_1$")


c2_fiber = np.array([CorrQuotient.grp_action(p=c2, d=np.array([dx[i], dy[i]])) for i in range(dx.shape[0])])

xp = c2_fiber[:, 0, 0].reshape((fine, fine))
yp = c2_fiber[:, 1, 1].reshape((fine, fine))
zp = c2_fiber[:, 0, 1].reshape((fine, fine))

plt.pass_wireframe(xp, yp, zp, color="red", lw=0.2, alpha=0.8, rstride=2, cstride=2, label=r"fiber of $C_2$")

# q2 = CorrQuotient.optimize_position(q=c2, p=c1)
# plt.pass_point(q2, label=None)
# plt.text(point=q2, text=r"$Q_2$")
#
# geodesic_c1_q2 = CorrQuotient.a_path(p=c1, q=q2, n_points=20)
# plt.pass_curve(curve=geodesic_c1_q2, lw=1, label=r"geodesic between $C_1$ and $Q_2$")
#
#
# q1 = CorrQuotient.optimize_position(q=c1, p=c2)
# plt.pass_point(q1, label=None)
# plt.text(point=q1, text=r"$Q_1$")
#
# geodesic_c2_q1 = CorrQuotient.a_path(p=c2, q=q1, n_points=20)
# plt.pass_curve(curve=geodesic_c2_q1, lw=1, label=r"geodesic between $C_2$ and $Q_1$")

plt.show(leg=True)

