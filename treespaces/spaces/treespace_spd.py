# external imports

import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.linalg as la

# external imports
import abc

# internal imports
from treespaces.tools.wald import Wald
from treespaces.spaces.treespace import TreeSpace
from treespaces.tools.structure_and_split import Structure

import treespaces.tools.tools_vector_spaces as vtools
import treespaces.tools.tools_forest_representation as ftools

# global variables
CHARTS = dict()
GRADIENTS = dict()
CHRISTOFFEL = dict()
CURVATURE = dict()
SECTIONAL_CURVATURE = dict()
FULLY_RESOLVED_SPLIT_SETS = None


class TreeSpaceSpd(TreeSpace):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def s_lift(p):
        try:
            return p.corr
        except AttributeError:
            return p

    @classmethod
    def s_lift_vector(cls, v, p: Wald):
        """ 'Lifts' the vector from grove into ambient vector space. """
        # TODO: is there a smarter way to do this?
        try:
            assert v.shape == p.corr.shape
            return v
        except AssertionError:
            return np.array(np.sum([v[i] * b_i for i, b_i in enumerate(cls.s_gradient(st=p.st)(x=p.x))], axis=0))

    @classmethod
    def s_proj(cls, p, **kwargs) -> Wald:
        """ Computes the orthogonal projection from the ambient space onto the subset of forests.

        :param p: The point that is projected onto the subset of forests. Is an element of the corresponding ambient
            space (e.g. positive definite matrix for the wald space).

        :Keyword Arguments:
            * *st* (``Structure``) --
                The structure of :class:`Structure` determining on which grove we want to project onto.
            * *btol* (``float``) --
                The tolerance as to what is considered already being on the boundary. A `float`, numbers that are
                closer than this to a boundary are mapped onto the boundary.
            * *x0* (``np.ndarray``) --
                Initial guessed coordinates. Defaults to constant vector filled with 0.5s.
            * *gtol* (``float``) --
                Terminates if the projected gradient < ``gtol``. Defaults to 1e-05.
            * *ftol* (``float``) --
                Terminates if the relative function value descent < ``ftol``. Defaults to 2.22e-09.
            * *btol* (``float``) --
                Tolerance for the boundary. Coordinates must be entry-wise >= ``btol`` far away from the boundary.
            * *method* (``bool``) --
                A flag that determines whether the projection should be done globally.
                The structure ``st`` is then the grove where the search starts.
        """
        if 'st' not in kwargs:
            raise NotImplementedError(f"Projection methods need ``st`` to be defined, i.e. a proposal for Structure.")
        if 'btol' not in kwargs:
            kwargs['btol'] = 10 ** -8
        method = kwargs['method'] if 'method' in kwargs else 'local'
        if method == 'local':
            return cls._proj_method_local(p=p, **kwargs)
        elif method == 'global-search':
            return cls._proj_method_global_search(p=p, **kwargs)
        elif method == 'global-descent':
            return cls._proj_method_global_descent(p=p, **kwargs)
        else:
            raise NotImplementedError(f"The projection method '{method}' is not implemented. Typo?")

    @classmethod
    def s_proj_vector(cls, v, p: Wald):
        """ Projects the symmetric matrix (vector in spd space) onto the tangent space of the structure (a vector). """
        ob = vtools.gram_schmidt(vectors=[e_i for e_i in np.eye(N=len(p.x))], dot=cls.s_inner, p=p)
        # becomes a vector of the tangent space of the grove
        v_proj = vtools.vector_space_projection(v=v, basis=ob, dot=cls.a_inner, p=p, lift=cls.s_lift,
                                                lift_vector=cls.s_lift_vector)
        return v_proj

    @classmethod
    def s_dist(cls, p, q, **kwargs):
        path_ = cls.s_path(p, q, **kwargs)
        return cls.length(path_=path_)

    @classmethod
    def s_path_t(cls, p, q, t, **kwargs):
        path_ = cls.s_path(p, q, **kwargs)
        return path_[int(t * len(path_))]

    # @classmethod
    # def s_path(cls, p: Wald, q: Wald, **kwargs):
    #     try:
    #         return super().s_path(p=p, q=q, **kwargs)
    #     except NotImplementedError:
    #         raise NotImplementedError(f"Apparently, this algorithm named '{kwargs['alg']}' is not implemented yet.")

    @classmethod
    def s_inner(cls, u, v, p: Wald, **kwargs):
        """ Computes the inner product (riemannian metric) of subset between two vectors u, v at a point _p. """
        _gradient_x = kwargs['gradient_x'] if 'gradient_x' in kwargs else cls.s_gradient(st=p.st)(x=p.x)
        # print("Inside s_inner")
        # print(_gradient_x)
        u_lift = np.sum([u[k] * grad for k, grad in enumerate(_gradient_x)], axis=0)
        v_lift = np.sum([v[k] * grad for k, grad in enumerate(_gradient_x)], axis=0)
        # this method just computes the push-forward by formula. TODO: one can possibly implement it more cleverly.
        return cls.a_inner(u=u_lift, v=v_lift, p=cls.s_lift(p=p))

    @classmethod
    def s_exp(cls, v, p: Wald, **kwargs):
        """ Compute the riemannian exponential in subset geometry of vector (numpy array) v at point _p.

        :param v: The direction in which we travel. Must be a vector of length = #edges in forest.
        :param p: The point at which we start. Must be of :class:`Wald`.
        :param kwargs: See below.

        :Keyword Arguments:
            * *atol* (``float``) --
                Absolute tolerance. Solver keeps local error estimates less than atol + rtol * abs(y).
            * *rtol* (``float``) --
                Relative tolerance. Solver keeps local error estimates less than atol + rtol * abs(y).
            * *btol* (``float``) --
                Boundary tolerance. Boundary is reached if values are closer to boundary (0 or 1) than btol.
            * *t_max* (``float``) --
                Maximum time traveled, integration won't continue beyond. Defaults to 1 (length of v is important here).
            * *max_step* (``float``) --
                The maximum allowed step size. If set to np.inf (default), solver determines this value).
            * *info* (``boolean``) --
                A flag that determines whether the output is with more detail or not.

        """
        st = p.st
        m = len(p.x)
        if m == 0:  # in this case, we have the pure forest (no edges at all)
            return p
        if st in CHRISTOFFEL:
            _christoffel = CHRISTOFFEL[st]
        else:
            _christoffel = cls.s_christoffel(st=st)

        # ----- partial differentiation
        def fun(_, y):
            """t is the time and y = (w, v) is the weight and direction (they are coupled)."""
            # y[m:] = SpdKilling._proj_onto_tangent_cone(v=y[m:], x=y[:m])
            gamma = _christoffel(x=y[:m])  # note that gamma[k][j][i] = \Gamma_{ij}^k.

            _eq = np.zeros(y.shape)  # start to compute the returning value of the function
            _eq[:m] = y[m:]

            for k in range(m):
                _eq[m + k] = -np.sum([[gamma[k][j][i] * y[m + i] * y[m + j] for i in range(m)] for j in range(m)])
            return _eq

        # this event function determines to stop if the boundary is reached or beyond.
        btol = kwargs['btol'] if 'btol' in kwargs else 10 ** -10

        def event(_, y):
            return 1 - 2 / (1 - 2 * btol) * np.max(np.abs(y[:m] - 0.5))

        event.terminal = True

        # initialize the parameters for the minimization. If not given, take default values specified below.
        y0 = np.concatenate([p.x, v])  # here, the y's are tangent bundle elements: coordinates plus vector concatenated
        t_span = (0, kwargs['t_max']) if 't_max' in kwargs else (0, 1)
        defaults = {'atol': 10 ** -6, 'rtol': 10 ** -3, 'max_step': 10 ** -2, 'y0': y0, 't_span': t_span}
        params = {param: kwargs[param] if param in kwargs else defaults[param] for param in defaults.keys()}
        params['first_step'] = params['max_step']
        res = scipy.integrate.solve_ivp(fun=fun, events=event, **params)  # bunch object; second entry is result.
        _x = np.minimum(np.maximum(0, res.y[:m, -1]), 1)  # this code works despite inspection claiming differently.
        info = kwargs['info'] if 'info' in kwargs else False
        if info:
            _points = [Wald(x=_z, st=st, n=p.n) for _z in res.y[:m, :].T]
            _vectors = [_v for _v in res.y[m:, :].T]
            _times = [_t for _t in res.t]
            return _points, _vectors, _times
        else:
            return Wald(st=st, x=_x, n=p.n)

    @staticmethod
    def s_chart(st: Structure):
        if st in CHARTS:
            _chart = CHARTS[st]
        else:
            _chart = ftools.compute_chart(st=st)
            CHARTS[st] = _chart
        return _chart

    @classmethod
    def s_gradient(cls, st: Structure):
        # if st in GRADIENTS:
        #     _gradient = GRADIENTS[st]
        # else:
        _gradient = ftools.compute_chart_gradient(st=st, chart=cls.s_chart(st=st))
        # GRADIENTS[st] = _gradient
        return _gradient

    @staticmethod
    def s_chart_and_gradient(st: Structure):
        if st in CHARTS:
            _chart = CHARTS[st]
        else:
            _chart = ftools.compute_chart(st=st)
            CHARTS[st] = _chart
        if st in GRADIENTS:
            _gradient = GRADIENTS[st]
        else:
            _gradient = ftools.compute_chart_gradient(st=st, chart=_chart)
            GRADIENTS[st] = _gradient
        return _chart, _gradient

    @classmethod
    @abc.abstractmethod
    def s_christoffel(cls, st: Structure):
        """ Returns a map that maps a wald to the Christoffel symbols at that respective point. """

    @staticmethod
    @abc.abstractmethod
    def _proj_target_gradient(p, st, **kwargs):
        """ Compute functional: squared distance of _p to the grove at coordinates _x, and its gradient."""

    @classmethod
    def _proj_method_local(cls, p, st: Structure, **kwargs) -> Wald:
        """ This method projects onto a fixed structure 'st' by solving a classic minimization problem. """
        if len(st.partition) == st.n:
            return Wald(n=st.n, corr=np.eye(st.n))
        p_inv = la.inv(p)
        # the target function that we will minimize.
        target_and_gradient = cls._proj_target_gradient(p=p, p_inv=p_inv, st=st)
        # the number of edges/splits in the forest
        n_splits = np.sum(a=[len(splits) for splits in st.split_collection], dtype=int)
        # default parameters
        default_params = {'x0': np.repeat(0.5, n_splits), 'btol': 10 ** -10, 'ftol': 2.22e-09, 'gtol': 10 ** -5}
        # initialize the parameters for the projection.
        proj_params = {par: kwargs[par] if par in kwargs else default_params[par] for par in default_params.keys()}
        btol = proj_params['btol']
        bounds = [(btol, 1 - btol)] * n_splits
        # print(target_and_gradient(_x=np.array([0.5, 0.99, 0.7, 0.98, 0.98])))
        # print(proj_params)
        # finally, minimize.
        # print(f"Projection of p = \n{p}.")
        res = scipy.optimize.minimize(target_and_gradient, proj_params['x0'], jac=True, method='L-BFGS-B',
                                      bounds=bounds, tol=None,
                                      options={'gtol': proj_params['gtol'], 'ftol': proj_params['ftol']})
        # if x is close to boundary, treat as if it was on boundary
        # print(res)
        if res.status != 0:
            raise ValueError("Projection failed!")
        x = [_x if btol < _x < 1 - btol else 0 if _x <= btol else 1 for _x in res.x]
        if np.allclose(np.eye(st.n), Wald(st=st, x=x, n=st.n).corr):
            print(x)
        return Wald(st=st, x=x, n=st.n)

    @classmethod
    def _proj_method_global_search(cls, p, st: Structure, btol=10 ** -10, **kwargs) -> Wald:
        """ Searches over all possible fully resolved tree structures and projects onto each of them. """
        # NOT RECOMMENDED for leaves > 5 as it has SUPER-EXPONENTIAL RUNNING TIME in number of leaves
        global FULLY_RESOLVED_SPLIT_SETS
        if FULLY_RESOLVED_SPLIT_SETS is None:
            FULLY_RESOLVED_SPLIT_SETS = list(ftools.make_structures(n=st.n))
        # project onto each structure with original parameters, but the method must be 'local'
        kwargs['method'] = 'local'
        proposals = [cls.s_proj(p=p, st=_st, btol=btol, **kwargs) for _st in FULLY_RESOLVED_SPLIT_SETS]
        dists = np.array([cls.a_dist(p=p, q=wald, squared=True) for wald in proposals])
        return proposals[int(np.argmin(dists))]

    @classmethod
    def _proj_method_global_descent(cls, p, st: Structure, btol=10 ** -10, **kwargs) -> Wald:
        """ Starts in structure 'st' and wanders to the minimum across structures minimizing distance to p. """
        if len(st.partition) > 1:
            raise NotImplementedError("This projection is only defined for trees. ")

        def _search_in_grove(wald0: Wald):
            """ Searches for the best point in one grove. """
            _stop_gradient = False
            _stop_boundary = False
            _params = kwargs.copy()
            _params['x0'] = wald0.x
            _params['method'] = 'local'
            _q = cls.s_proj(p=p, st=st, btol=btol, **_params)
            if not np.all(_q.x != 0):
                _stop_boundary = True
            else:
                _stop_gradient = True
            return _q, _stop_gradient, _stop_boundary

        # back to the algorithm
        ruled_out_splits = set()

        x = kwargs['x0'] if 'x0' in kwargs else np.repeat(0.5, len(st.split_collection[0]))
        while True:
            q, stop_gradient, stop_boundary = _search_in_grove(wald0=Wald(n=st.n, st=st, x=x))
            if stop_gradient:
                return q
            # if not stop gradient, then it must have been stop boundary.
            # rule out all splits that have converged to zero
            splits = [st.split_collection[0][_i] for _i in np.where(q.x == 0)[0]]
            ruled_out_splits = ruled_out_splits | set(splits)
            np.set_printoptions(precision=5)
            # print(f"{q.st} and {q.x}.")
            # determine the next structure to be investigated:
            neighbours = [_st for _st, _sp in ftools.give_nni_candidates(st=q.st, sp=splits[0]) if
                          _sp not in ruled_out_splits]
            # print(f"Neighbors are {neighbours}.")
            # assuming we are not exactly on a degenerate tree
            if len(neighbours) == 0:
                return q

            st_new = neighbours[0]
            sp_x = {s: q.x[i] for i, s in enumerate(st.split_collection[0])}
            # with the new split, start in the 'middle' of the grove.
            x = np.array([sp_x[s] if s in st.split_collection[0] else 0.4 for s in st_new.split_collection[0]])
            st = st_new
