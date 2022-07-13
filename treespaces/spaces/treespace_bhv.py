# external imports
import numpy as np

# package imports
from treespaces.tools.wald import Tree, Wald
from treespaces.spaces.treespace import TreeSpace
import treespaces.tools.tools_bhv_gtp_algorithm as gtp

Trees = [Tree, Wald]

# an ambient space for the bhv space is not (yet?) defined, as it has dimension exponentially increasing with number of
# leaves.


class TreeSpaceBhv(TreeSpace):
    @classmethod
    def length(cls, path_):
        """ Computes the length of a path by summing up pair-wise distances. """
        return sum(cls.s_dist(path_[i], path_[i + 1], squared=False) for i in range(len(path_) - 1))

    # methods of the ambient space ('a' stands for 'ambient'):
    # ---------------------------------------------------------------

    @classmethod
    def a_dist(cls, p, q, squared=False):
        raise NotImplementedError(f"Method a_dist not implemented.")

    @classmethod
    def a_path_t(cls, p, q, t):
        raise NotImplementedError(f"Method a_path_t not implemented.")

    @staticmethod
    def a_path(p, q, **kwargs):
        raise NotImplementedError(f"Method a_path not implemented.")

    @staticmethod
    def a_inner(u, v, p):
        raise NotImplementedError(f"Method a_inner not implemented.")

    @staticmethod
    def a_exp(v, p):
        raise NotImplementedError(f"Method a_exp not implemented.")

    @staticmethod
    def a_log(q, p):
        raise NotImplementedError(f"Method a_log not implemented.")

    @staticmethod
    def a_trans(v, p, q):
        raise NotImplementedError(f"Method a_trans not implemented.")

    # methods of the embedded space ('s' in the methods stands for 'subspace').
    # ---------------------------------------------------------------
    @staticmethod
    def s_lift(p: Trees):
        """ 'Lifts' the wald into the ambient space. """
        raise NotImplementedError(f"Method s_lift not implemented.")

    @staticmethod
    def s_lift_vector(v, p: Trees):
        raise NotImplementedError(f"Method s_lift_vector not implemented.")

    @staticmethod
    def s_proj(p, **kwargs) -> Trees:
        raise NotImplementedError(f"Method s_proj not implemented.")

    @classmethod
    def s_proj_vector(cls, v, p: Trees):
        raise NotImplementedError(f"Method s_proj_vector not implemented.")

    @staticmethod
    def s_dist(p: Trees, q: Trees, **kwargs):
        if len(p.st.split_collection) > 1 or len(q.st.split_collection) > 1:
            raise ValueError("Either p or q are not trees but forests.")
        # construct the split representations of the trees
        splits_p = {s: p.b[i] for i, s in enumerate(p.st.split_collection[0])}
        splits_q = {s: q.b[i] for i, s in enumerate(q.st.split_collection[0])}
        return gtp.gtp_dist(n=p.n, splits_a=splits_p, splits_b=splits_q)

    @staticmethod
    def s_path_t(p: Trees, q: Trees, t: float, **kwargs):
        if len(p.st.split_collection) > 1 or len(q.st.split_collection) > 1:
            raise ValueError("Either p or q are not trees but forests.")
        if t == 1:  # later we will divide by (1 - t), so exclude that case here already
            return q
        if t == 0:
            return p
        # construct the split representations of the trees
        splits_p = {s: p.b[i] for i, s in enumerate(p.st.split_collection[0])}
        splits_q = {s: q.b[i] for i, s in enumerate(q.st.split_collection[0])}
        return gtp.gtp_point_on_geodesic(n=p.n, splits_a=splits_p, splits_b=splits_q, t=t)

    @classmethod
    def s_path(cls, p: Trees, q: Trees, **kwargs):
        if len(p.st.split_collection) > 1 or len(q.st.split_collection) > 1:
            raise ValueError("Either p or q are not trees but forests.")
        if 'times' in kwargs:
            times = kwargs['times']
        else:
            n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            times = np.linspace(0, 1, n_points)
        # construct the split representations of the trees
        splits_p = {s: p.b[i] for i, s in enumerate(p.st.split_collection[0])}
        splits_q = {s: q.b[i] for i, s in enumerate(q.st.split_collection[0])}
        # those parameters will not change over the course of the geodesic
        common_splits_p, common_splits_q, supports = gtp.gtp_trees_with_common_support(p.n, splits_p, splits_q)
        _params = {'common_splits_a': common_splits_p, 'common_splits_b': common_splits_q, 'supports': supports}
        # calculate the path
        _path = tuple(gtp.gtp_point_on_geodesic(n=p.n, splits_a=splits_p, splits_b=splits_q, t=t, **_params)
                      for t in times)
        return _path

    @staticmethod
    def s_inner(u, v, p: Trees, **kwargs):
        return np.inner(u, v)

    @staticmethod
    def s_exp(v, p: Trees, **kwargs):
        return Tree(n=p.n, st=p.st, b=p.b + v)

    @staticmethod
    def s_log(q, p, **kwargs):
        return q.b - p.b

    @staticmethod
    def s_trans(v, p, q, **kwargs):
        return v
