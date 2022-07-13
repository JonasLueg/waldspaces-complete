"""
For this particular geometry, the ambient space is the space of strictly positive definite real symmetric matrices
with diagonal 1, i.e. the correlation matrices, which has the geometry of the quotient space.
"""

# system imports
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.linalg as la
import numpy.linalg as na

# wald space imports
from treespaces.spaces.embedding import Embedding
from treespaces.spaces.spd_af1 import SpdAf1


class CorrQuotient(Embedding):
    # methods of the ambient space ('a' stands for 'ambient'):
    # ---------------------------------------------------------------
    @classmethod
    def a_dist(cls, p, q, squared=False):
        """ Computes the ambient distance between p and q. """
        p, q = cls.s_lift(p), cls.s_lift(q)
        return SpdAf1.dist(p, q, squared)

    @classmethod
    def a_path_t(cls, p, q, t):
        """ Computes the geodesic between _p and q at portion t (where t is between 0 and 1) in ambient space. """
        p, q = cls.s_lift(p), cls.s_lift(q)
        return SpdAf1.path_t(p, q, t)

    @classmethod
    def a_path(cls, p, q, **kwargs):
        """ Computes the geodesic with n_points points between p and q in ambient space. """
        p, q = cls.s_lift(p), cls.s_lift(q)
        return SpdAf1.path(p, q, **kwargs)

    @classmethod
    def a_inner(cls, u, v, p):
        """ Computes the ambient riemannian metric of vectors u and v at point p. """
        p = cls.s_lift(p)
        return SpdAf1.inner(u, v, p)

    @classmethod
    def a_exp(cls, v, p):
        """ Computes the ambient riemannian exponential of vector v at point x. """
        p = cls.s_lift(p)
        return SpdAf1.exp(v, p)

    @classmethod
    def a_log(cls, q, p):
        """ Computes the ambient riemannian logarithm of point q base p (outcome is a vector at point p). """
        p, q = cls.s_lift(p), cls.s_lift(q)
        return SpdAf1.log(q, p)

    @classmethod
    def a_trans(cls, v, p, q):
        """ Computes the ambient parallel transport of the vector v from point p to point q. """
        p, q = cls.s_lift(p), cls.s_lift(q)
        return SpdAf1.trans(v, p, q)

    # methods of the embedded space ('s' in the methods stands for 'subspace').
    # ---------------------------------------------------------------
    @staticmethod
    def s_lift(p):
        """ 'Lifts' the correlation matrix into the ambient space. """
        return p

    @classmethod
    def s_lift_vector(cls, v, p):
        return cls.hor(cls.grp_action(v, np.sqrt(np.diag(p))), p)

    @classmethod
    def s_proj(cls, p):
        d = np.sqrt(np.diag(p)) ** -1
        return cls.grp_action(p, d)

    @classmethod
    def s_proj_vector(cls, v, p):
        """ Projects the symmetric matrix (vector in spd space) onto the horizontal space. """
        return cls.diff_proj(v=cls.hor(v=v, p=p), p=p)

    @classmethod
    def s_dist(cls, p, q, **kwargs):
        squared = kwargs['squared'] if 'squared' in kwargs else False
        return cls.a_dist(p=p, q=cls.optimize_position(p=p, q=q, **kwargs), squared=squared)

    @classmethod
    def s_path_t(cls, p, q, t: float, **kwargs):
        return cls.s_proj(cls.s_exp(v=t * cls.s_log(q, p, **kwargs), p=p))

    @classmethod
    def s_path(cls, p, q, **kwargs):
        if 'times' in kwargs:
            times = kwargs['times']
        else:
            n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            times = np.linspace(0, 1, n_points)
        q_opt = cls.optimize_position(q=q, p=p, **kwargs)
        log_pq = cls.s_log(q=q_opt, p=p, opt=True)
        return tuple(cls.s_proj(cls.s_exp(v=t * log_pq, p=p)) for t in times)

    @classmethod
    def s_inner(cls, u, v, p):
        p_inv = la.inv(p)
        m = la.inv(np.eye(p.shape[0]) + p * p_inv)
        dummy = np.inner(np.diag(np.dot(p_inv, u)), np.dot(m, np.diag(np.dot(p_inv, v))))
        return cls.a_inner(u, v, p) - 2 * dummy

    @classmethod
    def s_exp(cls, v, p):
        """ v needs to be in the tangent space of p which represents the tangent space of the equivalence class [p]"""
        return cls.s_proj(cls.a_exp(v=cls.hor(v, p), p=p))

    @classmethod
    def s_log(cls, q, p, **kwargs):
        opt = kwargs['opt'] if 'opt' in kwargs else False
        if opt:
            return cls.diff_proj(cls.a_log(q=q, p=p), p)
        else:
            return cls.diff_proj(cls.a_log(q=cls.optimize_position(q=q, p=p, **kwargs), p=p), p)

    @staticmethod
    def s_trans(v, p, q, **kwargs):
        raise NotImplementedError

    @staticmethod
    def hor(v, p):
        p_inv = la.inv(p)
        a = la.inv(np.eye(p.shape[0]) + p * p_inv).dot(np.diag(p_inv.dot(v)))
        return v - p * (a[:, None] + a[None, :])

    @classmethod
    def diff_proj(cls, v, p):
        """ Computes the differential of the projection. """
        _d = 1 / np.diag(p)
        _d_sqrt = np.sqrt(_d)
        _a = _d * np.diag(v)
        return cls.grp_action(p=v - 0.5 * (_a * p + (_a * p.T).T), d=_d_sqrt)

    @staticmethod
    def grp_action(p, d):
        return (d * p).T * d

    @staticmethod
    def diff_grp_action(v, d):
        """ d is a vector, v is a matrix. """
        return (d * v).T * d

    @classmethod
    def optimize_position(cls, q, p, **kwargs):
        """ Maps q into optimal position to p. """
        if np.allclose(p, q):
            return p
        return cls.grp_action(q, cls.find_optimal_position(p=p, q=q, **kwargs))

    @classmethod
    def find_optimal_position(cls, p, q, **kwargs):
        """ Computes d such that p and q are in optimal position. """

        if np.allclose(p, q):
            return np.ones(p.shape[0])

        target_and_gradient = cls._find_optimal_position_target_gradient(p=p, q=q, p_inv=la.inv(p))

        # solve minimization problem:
        x0 = np.exp(np.diag(0.5 * np.array(la.logm(np.dot(la.inv(q), p)))))
        ftol = kwargs['ftol'] if 'ftol' in kwargs else 10 ** -10
        gtol = kwargs['gtol'] if 'gtol' in kwargs else 10 ** -10
        btol = kwargs['btol'] if 'btol' in kwargs else 10 ** -10
        bounds = ((btol, np.inf),) * p.shape[0]
        res = scipy.optimize.minimize(target_and_gradient, x0=x0, jac=True, method='L-BFGS-B', bounds=bounds,
                                      options={'ftol': ftol, 'gtol': gtol})
        # store the results
        return res.x

    @classmethod
    def _find_optimal_position_target_gradient(cls, p, q, p_inv):
        """ Computes the gradient and target of function that is minimized when computing the optimal position. """

        def target_gradient(_d):
            _target = SpdAf1.dist(p, cls.grp_action(q, _d), squared=True)
            dummy = SpdAf1.ssqrtm(q).dot(np.diag(_d)).dot(p_inv).dot(np.diag(_d)).dot(SpdAf1.ssqrtm(q))
            eig_qdpd, u = la.eigh(dummy)
            m = na.multi_dot(arrays=[la.sqrtm(q), u, np.diag(np.log(eig_qdpd)), u.T, la.inv(la.sqrtm(q))])
            _gradient = 4 * np.diag(m) / _d
            # IMPORTANT: /_d makes it np.array
            # print(f"D: {_d}, target: {_target}, gradient: {_gradient}.")
            return _target, _gradient

        return target_gradient
