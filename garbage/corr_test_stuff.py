import scipy.linalg as la
import numpy as np

from treespaces.spaces.treespace_bhv import TreeSpaceBhv
from treespaces.spaces.treespace_spd_af1 import TreeSpdAf1
from treespaces.spaces.embed_corr_in_spd_quotient import CorrQuotient

import framework.tools_numerical as numtools
import tools.tools_vector_spaces as vtools
from tools.structure_and_split import Split, Structure
from tools.wald import Wald

import matplotlib.pyplot as plt


# ----- CONSTRUCTION OF THE STARTING POINT ----- #
n = 5
partition = ((0, 1, 2, 3, 4),)
split_collection = [[Split(n=n, part1=(1, 2, 3, 4), part2=(0,)),
                     Split(n=n, part1=(0, 2, 3, 4), part2=(1,)),
                     Split(n=n, part1=(0, 1, 3, 4), part2=(2,)),
                     Split(n=n, part1=(0, 1, 2, 4), part2=(3,)),
                     Split(n=n, part1=(0, 1, 2, 3), part2=(4,)),
                     Split(n=n, part1=(2, 3, 4), part2=(0, 1)),
                     Split(n=n, part1=(3, 4), part2=(0, 1, 2))]]

st = Structure(partition=partition, split_collection=split_collection, n=n)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")

# the coordinates of our forest in Nye notation (wald space coordinates):
x1 = [0.1, 0.1, 0.2, 0.7, 0.7, 0.4, 0.7]
x2 = [0.7, 0.65, 0.76, 0.9, 0.7, 0.4, 0.1]
x3 = [0.7, 0.7, 0.78, 0.7, 0.7, 0.45, 0.7]

p1, p2, p3 = tuple(Wald(n=n, st=st, x=x) for x in [x1, x2, x3])


def s_angle(u, v, p, g):
    u = u / g.s_norm(u, p)
    v = v / g.s_norm(v, p)
    return 360 / 2 / np.pi * np.arccos(g.s_inner(u=u, v=v, p=p))


def a_angle(u, v, p, g):
    u = u / g.a_norm(u, p)
    v = v / g.a_norm(v, p)
    return 360 / 2 / np.pi * np.arccos(g.a_inner(u=u, v=v, p=p))


g = CorrQuotient
d = np.array([0.1, 0.3, 0.4, 0.8, 0.1])

v = 100*g.s_proj_vector(g.a_log(p2.corr, p3.corr), p3.corr)
np.fill_diagonal(v, -0.5)

np.set_printoptions(precision=12)


def test_group_actions():
    d = np.abs(np.random.randn(n))*2
    p = p1.corr

    def act1():
        return np.dot(np.diag(d), p).dot(np.diag(d))

    def act2():
        return (d * p).T * d

    def act3():
        return np.multiply(np.multiply(d[:, None], p), d[None, :])

    return act1, act2, act3


# act1, act2, act3 = test_group_actions()
# print(f"{np.allclose(act1(), act2())}, {np.allclose(act2(), act3())}, {np.allclose(act3(), act1())}!")
#
# import timeit
#
# for i, act in enumerate([act1, act2, act3]):
#     print(f"Time needed for 10000 executions, method act{i+1}: {timeit.timeit(act, number=10000)}.")
#
# print("ergo, the act2 method is best.")

# p is p1
# p = p1.corr
# print(v)
# print(f"p is {p}.")
#
# v_horizontal = g.s_proj_vector(v, p)
# v_vertical = v - v_horizontal
#
# print(v_horizontal)
# w_horizontal = g.diff_grp_action(v_horizontal, d)
# print(w_horizontal)
#
# print(v_vertical)
# w_vertical = g.diff_grp_action(v_vertical, d)
# print(w_vertical)
#
# q = g.grp_action(p, d)
# print(f"Scalar product between horizontal and vertical: {g.a_inner(u=v_horizontal, v=v_vertical, p=p)}.")
# print(f"Scalar product between horizontal and vertical but at q: {g.a_inner(u=w_horizontal, v=w_vertical, p=q)}.")
#
# r1 = g.grp_action(g.a_exp(v_horizontal, p=p), d)
# print(r1)
# r2 = g.a_exp(g.diff_grp_action(v=v_horizontal, d=d), p=g.grp_action(p=p, d=d))
# print(r2)
#
# print("\n"*3)
# print("EXPERIMENT 2".center(80, '-'))
# q = 2*p3.corr
#
# p = 2*p
# d = g.find_optimal_position(p=p, q=q, ftol=10**-9, gtol=10**-8)
#
# print(d)
# print(la.eigvalsh(p))
# print(la.eigvalsh(q))
# print(np.sqrt(la.eigvalsh(p)/la.eigvalsh(q)))
# print(np.sqrt(np.real(la.eigvals(p.dot(la.inv(q))))))
# print(np.real(la.eigvals(q.dot(la.inv(p)))))
# grad = la.logm(np.diag(d).dot(la.inv(p)).dot(np.diag(d)).dot(q))
#
# np.fill_diagonal(grad, 0)
#
# print(grad)
# print(g.diff_proj(v=grad, p=p))
# lift_grad = g.s_lift_vector(grad, p)
# print(lift_grad)
# proj_grad = g.s_proj_vector(v=lift_grad, p=p)
# print(proj_grad)

# a = np.reshape(list(range(9)), (3, 3))
# print(a)
# d = np.array([10, 100, 1000])
#
# print(a * d)
# print((a.T * d).T)
# print((a * d).T * d)


# v = g.s_log(q=q, p=p, ftol=10**-20, gtol=10**-20)
#
# print(v)
#
# r = g.s_exp(v=v, p=p)
#
# print(g.s_dist(q, r))
# print(g.s_dist(p, q))
#
# d_opt = g.find_optimal_position(p=p, q=q)
# print(g.grp_action(q, d=d_opt))
# q_opt = g.optimize_position(p=p, q=q)
#
# print(q_opt)
# print(r)


p = p1.corr
print(p)
q = p2.corr
i = np.eye(p.shape[0])

# print(p)

d_opt = CorrQuotient.find_optimal_position(p=p, q=q, ftol=10**-20, gtol=10**-20)
d_opt_inv = CorrQuotient.find_optimal_position(p=q, q=p, ftol=10**-20, gtol=10**-20)
print(np.allclose(d_opt, 1/d_opt_inv))
# q_opt = CorrQuotient.optimize_position(q=q, p=p, ftol=10**-15, gtol=10**-15)

# print(np.dot(la.inv(p), q_opt))
# print(np.dot(q_opt, la.inv(p)))

# eig_p = la.eigvalsh(p)
# eig_q = la.eigvalsh(q)

# eig_p, u = la.eigh(p)
# eig_q, v = la.eigh(q)
#
# print(f"Eigenvalues of p: {eig_p}")
# print(f"Eigenvalues of q: {eig_q}")
# print(f"Optimal d**2 for identity: {d_opt}")


print(f"The proposals for d.".center(50, "-"))
print(f"Optimal d = {d_opt}.")

# print(np.diag(u.dot(np.diag(np.sqrt(eig_p))).dot(u.T).dot(v).dot(np.diag(np.sqrt(eig_q)**-1)).dot(v.T)))
# print(np.diag(np.dot(la.sqrtm(p), la.inv(la.sqrtm(q)))))

sqrtp = np.array(la.sqrtm(p))
q_inv = np.array(la.inv(q))

eig_pq, u = la.eigh(sqrtp.dot(la.inv(q)).dot(p))

attempt1 = np.exp(np.diag(0.5*np.array(la.logm(sqrtp.dot(q_inv).dot(sqrtp)))))
print(f"Attempt 1 = {attempt1}.")

attempt2 = np.exp(np.diag(0.5*np.array(la.logm(np.dot(q_inv, p)))))
print(f"Attempt 2 = {attempt2}.")

attempt3 = np.diag(la.expm(0.5*np.diag(np.diag(la.logm(np.dot(q_inv, p))))))
print(f"Attempt 3 = {attempt3}.")

target_gradient = CorrQuotient._find_optimal_position_target_gradient(p, q, la.inv(p))

print("Evaluation".center(50, "-"))
target, gradient = target_gradient(_d=d_opt)
target_opt = target

print(f"Optimal d, target = {target}.")
print(f"Gradient = {gradient}.")

target, gradient = target_gradient(_d=attempt1)
print(f"Attempt 1, target = {target}.")
print(f"Gradient = {gradient}.")

target, gradient = target_gradient(_d=attempt2)
print(f"Attempt 2, target = {target}.")
print(f"Gradient = {gradient}.")

target, gradient = target_gradient(_d=attempt3)
print(f"Attempt 3, target = {target}.")
print(f"Gradient = {gradient}.")


print("\n")
print(f"Optimal d = {d_opt}.")
print(f"My guess  = {attempt2}.")
eps = 10**-7
eps_list = np.linspace(-eps, eps, num=20)
# print(eps_list)

d_considered = d_opt
results_opt = []
for i in range(len(d_opt)):
    eps_i = np.eye(len(d_opt))[i]
    d_list = [d_considered + eps_i*a for a in eps_list]
    results_opt.append([target_gradient(_d=_d)[0] for _d in d_list])

d_considered = attempt2
results_guess = []
for i in range(len(d_opt)):
    eps_i = np.eye(len(d_opt))[i]
    d_list = [d_considered + eps_i*a for a in eps_list]
    results_guess.append([target_gradient(_d=_d)[0] for _d in d_list])


y_min = np.min([np.min([np.min(x) for x in results_opt]), np.min([np.min(x) for x in results_guess])])
y_max = np.max([np.max([np.max(x) for x in results_opt]), np.max([np.max(x) for x in results_guess])])


fig = plt.figure()
for i in range(len(results_opt)):
    plt.plot(eps_list, results_opt[i], label=f"{i}{numtools.postfix(i)} coordinate")
plt.legend()
plt.ylim([y_min, y_max])
plt.savefig("corr_min_d_optimal.pdf")
plt.clf()


fig = plt.figure()
for i in range(len(results_guess)):
    plt.plot(eps_list, results_guess[i], label=f"{i}{numtools.postfix(i)} coordinate")
plt.gca().set_ylim(bottom=y_min, top=y_max)
plt.legend()
plt.ylim([y_min, y_max])
plt.savefig("corr_min_d_guess.pdf")
plt.clf()




# dummy = np.dot(la.sqrtm(q), np.diag(d)).dot(p)
# print(dummy*la.inv(dummy.T))

# v = CorrQuotient.a_log(q, p)
# print(v)
#
# w = CorrQuotient.s_proj_vector(v, p)
# print(w)
#
# w_ = CorrQuotient.hor(v, p)
# print(w_)
# print(w_.dot(la.inv(p)))
# print(la.inv(p).dot(w_))
# print(w_.dot(la.inv(p)) + la.inv(p).dot(w_))
# p = p1.corr
# q = g.grp_action(p, d)
#
# exp_q = g.a_exp(v, q)
# exp_p = g.a_exp(v, p)
#
#
# dist = g.s_dist(10*exp_p, 5*exp_q)
# dist2 = g.s_dist(p, q, btol=10**-12, tol=10**-10)
# print(dist)
# print(dist2)
#
#
# q_sq = la.sqrtm(q)
#
# test_a = np.dot(q_sq, la.logm(np.dot(q_sq, np.diag(d)).dot(la.inv(p)).dot(np.diag(d)).dot(q_sq))).dot(la.inv(q_sq))
# test_b = la.logm(np.dot(q, np.diag(d)).dot(la.inv(p)).dot(np.diag(d)))
#
# # test_a = la.logm(p.dot(q).dot(la.inv(p)))
# # test_b = p.dot(la.logm(q)).dot(la.inv(p))
# print(test_a)
# print(test_b)
# print(np.allclose(test_a, test_b))

# for each geometry, compute the triangles.
# print("Start computing angles in spd space.")
# g = SpdAf1
# v12 = g.a_log(p2, p1)
# v21 = g.a_log(p1, p2)
# v23 = g.a_log(p3, p2)
# v32 = g.a_log(p2, p3)
# v31 = g.a_log(p1, p3)
# v13 = g.a_log(p3, p1)
#
# angles = [a_angle(v12, v13, p1, g), a_angle(v23, v21, p2, g), a_angle(v31, v32, p3, g)]
# print(sum(angles))
#
# print("Start computing angles for the quotient space of correlation matrices in spd space.")
# g = CorrQuotient
# path12 = g.s_path(p=p1.corr, q=p2.corr, n_points=100)
# path23 = g.s_path(p=p2.corr, q=p3.corr, n_points=100)
# path31 = g.s_path(p=p3.corr, q=p1.corr, n_points=100)
# v12 = path12[1] - path12[0]
# v21 = path12[-2] - path12[-1]
# v23 = path23[1] - path23[0]
# v32 = path23[-2] - path23[-1]
# v31 = path31[1] - path31[0]
# v13 = path31[-2] - path31[-1]
#
# angles = [s_angle(v12, v13, p1.corr, g), s_angle(v23, v21, p2.corr, g), s_angle(v31, v32, p3.corr, g)]
# print(sum(angles))
#
# print("Start computing angles for the embedding of wald space into the spd space.")
# g = SpdAf1
# path12 = g.s_path(p=p1, q=p2, n_points=12, n_iter=10)
# path23 = g.s_path(p=p2, q=p3, n_points=12, n_iter=10)
# path31 = g.s_path(p=p3, q=p1, n_points=12, n_iter=10)
# v12 = path12[1].x - path12[0].x
# v21 = path12[-2].x - path12[-1].x
# v23 = path23[1].x - path23[0].x
# v32 = path23[-2].x - path23[-1].x
# v31 = path31[1].x - path31[0].x
# v13 = path31[-2].x - path31[-1].x
#
# angles = [s_angle(v12, v13, p1, g), s_angle(v23, v21, p2, g), s_angle(v31, v32, p3, g)]
# print(sum(angles))
#
# print("Start computing angles for the bhv space for sanity check.")
# g = BhvSpace
# path12 = g.s_path(p=p1, q=p2, n_points=10)
# path23 = g.s_path(p=p2, q=p3, n_points=10)
# path31 = g.s_path(p=p3, q=p1, n_points=10)
# v12 = path12[1].b - path12[0].b
# v21 = path12[-2].b - path12[-1].b
# v23 = path23[1].b - path23[0].b
# v32 = path23[-2].b - path23[-1].b
# v31 = path31[1].b - path31[0].b
# v13 = path31[-2].b - path31[-1].b
#
# angles = [s_angle(v12, v13, p1, g), s_angle(v23, v21, p2, g), s_angle(v31, v32, p3, g)]
# print(sum(angles))
