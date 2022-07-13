"""
For this particular geometry, the ambient space is the space of strictly positive definite real symmetric matrices.
The riemannian metric on this ambient space is the so-called killing metric, or affine-invariant metric, or
scaled-frobenius metric.

"""

# system imports
import numpy as np
import scipy.linalg as la
import itertools as it


# wald space imports
from treespaces.spaces.treespace_spd import TreeSpaceSpd
from treespaces.tools.structure_and_split import Structure

# global variables
CURVATURE = dict()
SECTIONAL_CURVATURE = dict()


class TreeSpdEuclidean(TreeSpaceSpd):
    # methods of the ambient space ('a' stands for 'ambient'):
    # ---------------------------------------------------------------
    @classmethod
    def a_dist(cls, p, q, squared=False):
        """ Computes the ambient distance between p and q. """
        p, q = cls.s_lift(p), cls.s_lift(q)
        _dist = np.sum((p - q)**2) if squared else np.sqrt(np.sum((p - q)**2))
        return _dist

    @classmethod
    def a_path_t(cls, p, q, t):
        """ Computes the geodesic between _p and q at portion t (where t is between 0 and 1) in ambient space. """
        p, q = cls.s_lift(p), cls.s_lift(q)
        return (1 - t)*p + t*q

    @classmethod
    def a_path(cls, p, q, **kwargs):
        """ Computes the geodesic with n_points points between p and q in ambient space. """
        p, q = cls.s_lift(p), cls.s_lift(q)
        if 'times' in kwargs:
            times = kwargs['times']
        else:
            n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            times = np.linspace(0, 1, n_points)
        _v = q - p
        return [p + t*_v for t in times]

    @classmethod
    def a_inner(cls, u, v, p):
        """ Computes the ambient riemannian metric of vectors u and v at point p. """
        return np.sum(u*v)

    @classmethod
    def a_exp(cls, v, p):
        """ Computes the ambient riemannian exponential of vector v at point x. """
        p = cls.s_lift(p)
        return p + v

    @classmethod
    def a_log(cls, q, p):
        """ Computes the ambient riemannian logarithm of point q base p (outcome is a vector at point p). """
        p, q = cls.s_lift(p), cls.s_lift(q)
        return q - p

    @classmethod
    def a_trans(cls, v, p, q):
        """ Computes the ambient parallel transport of the vector v from point p to point q. """
        return v

    # methods of the embedded space ('s' in the methods stands for 'subspace').
    # ---------------------------------------------------------------

    @staticmethod
    def s_log(q, p, **kwargs):
        raise NotImplementedError("This method is not implemented yet. Better not rely on this in your algorithms.")

    @staticmethod
    def s_trans(v, p, q, **kwargs):
        raise NotImplementedError("This method is not implemented yet. Better not rely on this in your algorithms.")

    @classmethod
    def s_christoffel(cls, st: Structure):
        _chart = cls.s_chart(st=st)
        _gradient = cls.s_gradient(st=st)

        def _christoffel(x, **kwargs):
            """ Input is a flat vector or list x (Nye parametrization). """
            _chart_x = _chart(x) if 'chart_x' not in kwargs else kwargs['chart_x']
            _gradients_x = _gradient(x) if 'gradient_x' not in kwargs else kwargs['gradient_x']
            # TODO: CAUTION: here, the transpose is actually relevant! A.dot(B) is not always symmetric if A and B are.
            _gram_matrix = np.array([[np.sum(grad_i * grad_j) for grad_i in _gradients_x] for grad_j in _gradients_x])
            _gram_matrix_inv = la.inv(_gram_matrix)

            # compute the second derivatives: taking derivative of two edges from different components, the hessian is 0
            _m_gradients_x_raveled = st.ravel(x=_gradients_x)
            _sep = st.sep
            _hessian = [[np.zeros(shape=_chart_x.shape) for _ in _gradients_x] for _ in _gradients_x]
            for k, splits in enumerate(st.split_collection):
                for (i, sp_A), (j, sp_B) in it.combinations(enumerate(splits), 2):
                    # transform the indices into global indices
                    i, j = _sep[k] + i, _sep[k] + j
                    a_outer, b_outer = sp_A.directed_away_from_split(sp_B), sp_B.directed_away_from_split(sp_A)
                    for u, v in it.product(a_outer, b_outer):
                        _hessian[i][j][u, v] = _hessian[i][j][v, u] = _chart_x[u, v] / (1 - x[i]) / (1 - x[j])
                    _hessian[i][j] = _hessian[j][i]

            # now for the christoffel symbols:
            _dummy = [[np.array([np.sum(_hessian[i][j] * p_k) for p_k in _gradients_x]) for i in range(len(x))]
                      for j in range(len(x))]

            # TODO: exploit symmetry in i, j here. might double the speed...
            _symbols = [[[np.sum(gm * _dummy[i][j]) for i in range(len(x))] for j in range(len(x))]
                        for gm in _gram_matrix_inv]
            return _symbols

        return _christoffel

    @classmethod
    def s_curvature(cls, st: Structure):
        raise NotImplementedError("This method is not implemented yet.")
        # if st in CURVATURE:
        #     _curvature = CURVATURE[st]
        # else:
        #     _curvature = ftools.compute_curvature_symbols(st=st, chart=cls.s_chart(st=st),
        #                                                   gradient=cls.s_gradient(st=st))
        # return _curvature

    @classmethod
    def s_sectional_curvature(cls, st: Structure):
        raise NotImplementedError("This method is not implemented yet.")
        # if st in SECTIONAL_CURVATURE:
        #     _sectional_curvature = SECTIONAL_CURVATURE[st]
        # else:
        #     _sectional_curvature = ftools.compute_sectional_curvature_symbols(st=st, chart=cls.s_chart(st=st),
        #                                                                       gradient=cls.s_gradient(st=st))
        # return _sectional_curvature

    # @staticmethod
    # def _proj_onto_tangent_cone(v, x):
    #     """ Projects v to the tangent cone if x are coordinates on the boundary. """
    #     v = [v[i] if 0 < x_i < 1 or (x_i == 1 and v[i] <= 0) or (x_i == 0 and v[i] >= 0) else 0
    #          for i, x_i in enumerate(x)]
    #     return np.array(v)

    @classmethod
    def _proj_target_gradient(cls, p, st, **kwargs):
        """ Compute functional: squared distance of _p to the grove at coordinates _x, and its gradient."""
        chart, gradient = cls.s_chart_and_gradient(st=st)

        def target_gradient(_x):
            _corr = chart(_x)
            _target = cls.a_dist(_corr, p, squared=True)
            _target_gradient = np.array([2*np.sum((_corr - p)*grad) for grad in gradient(_x)])
            return _target, _target_gradient

        return target_gradient
