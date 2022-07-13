
import time
import itertools as it
import numpy as np
import scipy.linalg as la
import scipy.optimize

from treespaces.spaces.spd_af1 import SpdAf1
from treespaces.spaces.embed_corr_in_spd_quotient import CorrQuotient
from treespaces.spaces.treespace_spd import TreeSpaceSpd
from tools.structure_and_split import Split, Structure


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

chart = TreeSpaceSpd.s_chart(st=st)


def random_tree_matrix():
    x = np.random.uniform(0.01, 0.99, 7)
    return chart(x)


def random_positive_def_matrix(n):
    m = np.zeros((n, n))
    for i, j in it.combinations(range(n), 2):
        if i <= j:
            m[i, j] = m[j, i] = np.abs(np.maximum(-1, np.minimum(1, np.random.standard_normal())))
    m = la.expm(m)
    if not np.allclose(m, m.T):
        raise ValueError(f"Something went wrong creating a symmetric matrix: {m}.")
    if not np.alltrue(la.eigvalsh(m) > 0):
        raise ValueError(f"Something went wrong creating a positive definite matrix {m}.")
    return m


def opt1(p, q, **kwargs):
    if np.allclose(p, q):
        return np.ones(p.shape[0])

    target_and_gradient = CorrQuotient._find_optimal_position_target_gradient(p=p, q=q, p_inv=la.inv(p))

    x0 = kwargs['x0'] if 'x0' in kwargs else np.ones(p.shape[0])
    ftol = kwargs['ftol'] if 'ftol' in kwargs else 10 ** -15
    gtol = kwargs['gtol'] if 'gtol' in kwargs else 10 ** -15
    btol = kwargs['btol'] if 'btol' in kwargs else 10 ** -15
    bounds = ((btol, np.inf),) * p.shape[0]
    res = scipy.optimize.minimize(target_and_gradient, x0=x0, jac=True, method='L-BFGS-B', bounds=bounds,
                                  options={'ftol': ftol, 'gtol': gtol})
    # store the results
    return res.x


def opt2(p, q, **kwargs):
    if np.allclose(p, q):
        return np.ones(p.shape[0])

    target_and_gradient = CorrQuotient._find_optimal_position_target_gradient(p=p, q=q, p_inv=la.inv(p))

    # qq = la.inv(la.sqrtm(q))
    x0 = np.exp(np.diag(0.5*np.array(la.logm(np.dot(la.inv(q), p)))))
    # print(x0)
    ftol = kwargs['ftol'] if 'ftol' in kwargs else 10 ** -15
    gtol = kwargs['gtol'] if 'gtol' in kwargs else 10 ** -15
    btol = kwargs['btol'] if 'btol' in kwargs else 10 ** -15
    bounds = ((btol, np.inf),) * p.shape[0]
    res = scipy.optimize.minimize(target_and_gradient, x0=x0, jac=True, method='L-BFGS-B', bounds=bounds,
                                  options={'ftol': ftol, 'gtol': gtol})
    # store the results
    # print(res.x)
    return res.x


def opt3(p, q, **kwargs):
    if np.allclose(p, q):
        return np.ones(p.shape[0])

    target_and_gradient = CorrQuotient._find_optimal_position_target_gradient(p=p, q=q, p_inv=la.inv(p))

    # qq = la.inv(la.sqrtm(q))
    x0 = np.exp(np.diag(0.5*np.array(la.logm(np.dot(la.inv(q), p)))))
    # print(x0)
    ftol = kwargs['ftol'] if 'ftol' in kwargs else 10 ** -15
    gtol = kwargs['gtol'] if 'gtol' in kwargs else 10 ** -15
    btol = kwargs['btol'] if 'btol' in kwargs else 10 ** -15
    bounds = ((btol, np.inf),) * p.shape[0]
    res = scipy.optimize.minimize(target_and_gradient, x0=x0, jac=True, method='L-BFGS-B', bounds=bounds,
                                  options={'ftol': ftol, 'gtol': gtol})
    # store the results
    # print(res.x)
    return res.x


def measure_time(method, **args):
    start = time.time()
    method(**args)
    end = time.time()
    return end - start


def time_opts(p, q, **kwargs):
    return np.array([measure_time(opt, p=p, q=q, **kwargs) for opt in [opt1, opt2, opt3]])


n = 5
iterations = 100
times = np.zeros(3)
for i in range(iterations):
    # p, q = CorrQuotient.s_proj(random_positive_def_matrix(n)), CorrQuotient.s_proj(random_positive_def_matrix(n))
    p, q = random_tree_matrix(), random_tree_matrix()
    times += time_opts(p, q)

print(times)

# print(random_tree_matrix())
