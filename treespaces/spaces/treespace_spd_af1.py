"""
For this particular geometry, the ambient space is the space of strictly positive definite real symmetric matrices.
The riemannian metric on this ambient space is the so-called killing metric, or affine-invariant metric, or
scaled-frobenius metric.

"""

# system imports
import numpy as np
import scipy.linalg as la

# wald space imports
from treespaces.spaces.spd_af1 import SpdAf1
from treespaces.spaces.treespace_spd import TreeSpaceSpd
from treespaces.tools.structure_and_split import Structure
import treespaces.tools.tools_forest_representation as ftools

# global variables
CURVATURE = dict()
SECTIONAL_CURVATURE = dict()


class TreeSpdAf1(TreeSpaceSpd):
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
    def s_log(q, p, **kwargs):
        raise NotImplementedError("This method is not implemented yet. Better not rely on this in your algorithms.")

    @staticmethod
    def s_trans(v, p, q, **kwargs):
        raise NotImplementedError("This method is not implemented yet. Better not rely on this in your algorithms.")

    @classmethod
    def s_christoffel(cls, st: Structure):
        _chart = cls.s_chart(st=st)
        _gradient = cls.s_gradient(st=st)
        _christoffel = ftools.compute_christoffel_symbols(st=st, chart=_chart, gradient=_gradient)
        return _christoffel

    @classmethod
    def s_curvature(cls, st: Structure):
        if st in CURVATURE:
            _curvature = CURVATURE[st]
        else:
            _curvature = ftools.compute_curvature_symbols(st=st, chart=cls.s_chart(st=st),
                                                          gradient=cls.s_gradient(st=st))
        return _curvature

    @classmethod
    def s_sectional_curvature(cls, st: Structure):
        # if st in SECTIONAL_CURVATURE:
        #     _sectional_curvature = SECTIONAL_CURVATURE[st]
        # else:
        _sectional_curvature = ftools.compute_sectional_curvature_symbols(st=st, chart=cls.s_chart(st=st),
                                                                          gradient=cls.s_gradient(st=st))
        return _sectional_curvature

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
        p_inv = kwargs['p_inv'] if 'p_inv' in kwargs else la.inv(p)
        p_sqrt = SpdAf1.ssqrtm(p)
        p_inv_sqrt = SpdAf1.ssqrtm(p_inv)

        def target_gradient(_x):
            _corr = chart(_x)
            # print(f"Eigenvalues of p: {la.eigvalsh(p)}.")
            # print(f"We have p**-1 = {la.inv(p)}.")
            # print(f"Eigvals of inverse = {la.eigvalsh(la.inv(p))}.")
            # print(f"In target, gradient, we have p = \n{p}, \ncorr = \n{_corr}.")
            _target = cls.a_dist(_corr, p, squared=True)
            dummy = la.logm(p_inv_sqrt.dot(_corr).dot(p_inv_sqrt))

            _target_gradient = np.array(
                [0.5 * np.trace(np.dot(dummy, p_sqrt).dot(la.inv(_corr).dot(grad)).dot(p_inv_sqrt))
                 for grad in gradient(_x)])
            return _target, _target_gradient

        return target_gradient
