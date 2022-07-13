"""
This class contains the class Embedding, which is the foundation of all classes that explicitly define the geometry.

It is an abstract class.
"""
# external imports
import abc
import numpy as np


class Embedding(object):
    __metaclass__ = abc.ABCMeta

    # general methods that independent of ambient space or embedded space:
    # ---------------------------------------------------------------
    @classmethod
    def length(cls, path_):
        """ Computes the length of a path by summing up pair-wise distances. """
        return sum(cls.a_dist(cls.s_lift(path_[i]), cls.s_lift(path_[i + 1]), squared=False)
                   for i in range(len(path_) - 1))

    # methods of the ambient space ('a' stands for 'ambient'):
    # ---------------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def a_dist(cls, p, q, squared=False):
        """ Computes the ambient distance between p and q. """
        pass

    @classmethod
    @abc.abstractmethod
    def a_path_t(cls, p, q, t):
        """ Computes the geodesic between p and q at portion t (where t is between 0 and 1) in ambient space. """
        pass

    @staticmethod
    @abc.abstractmethod
    def a_path(p, q, **kwargs):
        """ Computes the geodesic with n_points points between p and q in ambient space. """
        pass

    @staticmethod
    @abc.abstractmethod
    def a_inner(u, v, p):
        """ Computes the ambient riemannian metric of vectors u and v at point p. """
        pass

    @classmethod
    def a_norm(cls, v, p, squared=False):
        """ Computes the ambient norm of vector v at point _p. """
        return np.sqrt(cls.a_inner(u=v, v=v, p=p)) if not squared else cls.a_inner(u=v, v=v, p=p)

    @staticmethod
    @abc.abstractmethod
    def a_exp(v, p):
        """ Computes the ambient riemannian exponential of vector v at point p. """
        pass

    @staticmethod
    @abc.abstractmethod
    def a_log(q, p):
        """ Computes the ambient riemannian logarithm of point q base p (outcome is a vector at point p). """
        pass

    @staticmethod
    @abc.abstractmethod
    def a_trans(v, p, q):
        """ Computes the ambient parallel transport of the vector v from point p to point q. """
        pass

    # methods of the embedded space ('s' in the methods stands for 'subspace').
    # ---------------------------------------------------------------
    @staticmethod
    @abc.abstractmethod
    def s_lift(p):
        """ 'Lifts' the element in subspace into the ambient space. """
        pass

    @staticmethod
    @abc.abstractmethod
    def s_lift_vector(v, p):
        """ 'Lifts' the vector in tangent space of subspace into the tangent space of the ambient space. """
        pass

    @staticmethod
    @abc.abstractmethod
    def s_proj(p):
        """ Computes the orthogonal projection from the ambient space onto the subspace. """
        pass

    @staticmethod
    @abc.abstractmethod
    def s_proj_vector(v, p):
        """ Projects the vector v in ambient space down onto the tangent space of the subspace at point p. """
        pass

    @staticmethod
    @abc.abstractmethod
    def s_dist(p, q):
        pass

    @staticmethod
    @abc.abstractmethod
    def s_path_t(p, q, t):
        pass

    @staticmethod
    @abc.abstractmethod
    def s_path(p, q):
        """ Computes the geodesic from point p to point q with n_points points."""
        pass

    @staticmethod
    @abc.abstractmethod
    def s_inner(u, v, p):
        pass

    @classmethod
    def s_norm(cls, v, p, sq=False):
        """ Computes the norm of vector v (tangent space in subset) at point _p. """
        return np.sqrt(cls.s_inner(u=v, v=v, p=p)) if not sq else cls.s_inner(u=v, v=v, p=p)

    @staticmethod
    @abc.abstractmethod
    def s_exp(v, p):
        pass

    @staticmethod
    @abc.abstractmethod
    def s_log(q, p):
        pass

    @staticmethod
    @abc.abstractmethod
    def s_trans(v, p, q):
        pass
