import numpy as np
import scipy.linalg as la


def random_symmetric(_n):
    mat = np.random.random((_n, _n))
    mat = mat + mat.T
    return mat


def random_positive(_n):
    mat = np.random.random((_n, _n))
    mat = mat + mat.T
    mat = mat @ mat
    return mat


def first_divided_difference(f, f_prime, vector):
    m = len(vector)
    mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                mat[i, i] = f_prime(vector[i])
            else:
                mat[i, j] = (f(vector[i]) - f(vector[j])) / (vector[i] - vector[j])
    return mat


def exp_differential(point, direction, n_terms=10):
    point_power = [np.linalg.matrix_power(point, k) for k in range(n_terms)]
    terms = np.array([np.sum([point_power[k - j - 1] @ direction @ point_power[j]
                              for j in range(k)], axis=0) / np.math.factorial(k)
                      for k in range(1, n_terms+1)])
    return np.sum(terms, axis=0)


def exp_diff_invariant(point, direction):
    s, u = la.eigh(point)
    fdd = first_divided_difference(np.exp, np.exp, s)
    return u @ (fdd * (u.T @ direction @ u)) @ u.T


def exp_diff_invariant_inverse(point, direction):
    s, u = la.eigh(point)
    fdd = first_divided_difference(np.log, lambda x: 1/x, s)
    return u.T @ (fdd * (u.T @ direction @ u)) @ u.T


def log_diff_invariant(point, direction):
    s, u = la.eigh(point)
    fdd = first_divided_difference(np.log, lambda x: 1/x, s)
    return u @ (fdd * (u.T @ direction @ u)) @ u.T


np.set_printoptions(precision=10)
np.random.seed(1000)
n = 4
X = random_symmetric(n)
Y = random_symmetric(n)
n_terms = 25

diff = exp_differential(X, Y, n_terms)
diff2 = exp_diff_invariant(X, Y)


with np.printoptions(precision=10, suppress=True):
    print(np.array(diff).dtype)
    print(np.array(diff))
    print(diff.shape)
    print(diff2)
    print(diff2.shape)


# P = random_positive(n)
# X = random_symmetric(n)
#
#
# logdiff = log_diff_invariant(point=P, direction=X)
# expdiff = exp_diff_invariant(point=la.logm(P))
# print(logdiff)
