"""
This class contains the class TreeSpace, which is the foundation of all classes that explicitly define the geometry.

It is an abstract class.
"""

# external imports
import abc
import numpy as np

# internal imports
from treespaces.tools.wald import Wald, Tree
from treespaces.spaces.embedding import Embedding

Trees = [Wald, Tree]


class TreeSpace(Embedding):
    __metaclass__ = abc.ABCMeta

    # methods of the embedded space ('s' in the methods stands for 'subspace').
    # ---------------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def s_proj(cls, p, **kwargs) -> Trees:
        """ Computes the orthogonal projection from the ambient space onto the subset of forests. """
        pass

    @classmethod
    def s_path(cls, p: Trees, q: Trees, **kwargs):
        """ Computes the geodesic from point p to point q with n_points points."""
        alg = kwargs[
            'alg'] if 'alg' in kwargs else 'naive' if p.st == q.st else 'global-symmetric'

        if alg == 'naive':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            return [cls.s_proj(x, st=p.st, x0=p.x, **_proj_args) for x in
                    cls.a_path(p=p, q=q, n_points=_n_points)]

        if alg == 'symmetric':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            n_half = _n_points // 2
            g, h = [p], [q]
            for i in range(1, n_half):
                g.append(cls.s_proj(
                    cls.a_path_t(p=g[i - 1], q=h[i - 1], t=1 / (_n_points - 2 * i + 1)),
                    st=p.st,
                    x0=g[i - 1].x, **_proj_args))
                h.append(cls.s_proj(
                    cls.a_path_t(p=h[i - 1], q=g[i - 1], t=1 / (_n_points - 2 * i + 1)),
                    st=p.st,
                    x0=h[i - 1].x, **_proj_args))
            if _n_points % 2 != 0:  # add a point s.t. the total number of points in the geodesic is n_points
                g.append(cls.s_proj(cls.a_path_t(p=g[-1], q=h[-1], t=0.5), st=p.st,
                                    x0=g[-1].x, **_proj_args))
            return g + h[::-1]

        if alg == 'straightening-ext':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _n_iter = kwargs['n_iter'] if 'n_iter' in kwargs else 10
            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            start_path = kwargs['start_path'] if 'start_path' in kwargs else None

            def _update(path_):
                velocities = [0.5 * (
                            cls.a_log(q=path_[_i], p=p_i) + cls.a_log(q=path_[_i + 2],
                                                                      p=p_i))
                              for _i, p_i in enumerate(path_[1:-1])]
                _path_upd = [
                    cls.s_proj(p=cls.a_exp(v=velocities[_i], p=p_i), st=p.st, x0=p_i.x,
                               **_proj_args)
                    for _i, p_i in enumerate(path_[1:-1])]
                return [p] + _path_upd + [q]

            # start with the naive path
            if start_path is None:
                paths = [cls.s_path(p=p, q=q, alg='naive', n_points=_n_points,
                                    proj_args=_proj_args)]
            else:
                paths = [start_path]
            for i in range(_n_iter):
                paths += [_update(path_=paths[-1])]
            return paths[-1]

        if alg == 'straightening-int':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _n_iter = kwargs['n_iter'] if 'n_iter' in kwargs else 10

            def _update(path_):
                velocities = [
                    cls.a_log(q=path_[_i], p=p_i) + cls.a_log(q=path_[_i + 2], p=p_i)
                    for _i, p_i in enumerate(path_[1:-1])]
                velocities = [0.5 * cls.s_proj_vector(v=velocities[_i], p=p_i)
                              for _i, p_i in enumerate(path_[1:-1])]
                _path_upd = [cls.s_exp(v=velocities[_i], p=p_i, **kwargs) for _i, p_i in
                             enumerate(path_[1:-1])]
                return [p] + _path_upd + [q]

            # start with the naive path
            paths = [cls.s_path(p=p, q=q, alg='naive', n_points=_n_points)]
            for i in range(_n_iter):
                paths += [_update(path_=paths[-1])]
            return paths[-1]

        if alg == 'variational':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _n_iter = kwargs['n_iter'] if 'n_iter' in kwargs else 10
            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'

            def update(path_):
                _m = [cls.a_path_t(p=path_[_i], q=path_[_i + 1], t=0.5) for _i in
                      range(_n_points - 1)]
                derivs = [0.5 * (cls.a_log(q=path_[_i + 1], p=_m[_i]) - cls.a_log(
                    q=path_[_i], p=_m[_i]))
                          for _i in range(_n_points - 1)]
                derivs2 = [cls.a_trans(v=derivs[_i], p=_m[_i], q=path_[_i]) -
                           cls.a_trans(v=derivs[_i - 1], p=_m[_i - 1], q=path_[_i])
                           for _i in range(1, _n_points - 1)]
                _path_upd = [
                    cls.s_proj(p=cls.a_exp(v=derivs2[_i - 1], p=path_[_i]), st=p.st,
                               x0=path_[_i].x, **_proj_args)
                    for _i in range(1, _n_points - 1)]
                return [p] + _path_upd + [q]

            # start with the naive path
            paths = [cls.s_path(p=p, q=q, alg='naive', n_points=_n_points,
                                proj_args=_proj_args)]
            for i in range(_n_iter):
                paths += [update(path_=paths[-1])]
            return paths[-1]

        if alg == 'global-naive':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            if 'method' not in _proj_args:
                _proj_args['method'] = 'global-descent'
            return [cls.s_proj(x, st=p.st, x0=p.x, **_proj_args) for x in
                    cls.a_path(p=p, q=q, n_points=_n_points)]

        if alg == 'global-symmetric':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            if 'method' not in _proj_args:
                _proj_args['method'] = 'global-descent'
            n_half = _n_points // 2
            g, h = [p], [q]
            for i in range(1, n_half):
                print(f"Iteration {i}.")
                g.append(cls.s_proj(
                    cls.a_path_t(p=g[i - 1], q=h[i - 1], t=1 / (_n_points - 2 * i + 1)),
                    st=g[i - 1].st,
                    x0=g[i - 1].x, **_proj_args))
                h.append(cls.s_proj(
                    cls.a_path_t(p=h[i - 1], q=g[i - 1], t=1 / (_n_points - 2 * i + 1)),
                    st=h[i - 1].st,
                    x0=h[i - 1].x, **_proj_args))
            if _n_points % 2 != 0:  # add a point s.t. the total number of points in the geodesic is n_points
                g.append(cls.s_proj(cls.a_path_t(p=g[-1], q=h[-1], t=0.5), st=g[-1].st,
                                    x0=g[-1].x, **_proj_args))
            return g + h[::-1]

        if alg == 'global-straightening':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _n_iter = kwargs['n_iter'] if 'n_iter' in kwargs else 10
            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            if 'method' not in _proj_args:
                _proj_args['method'] = 'global-descent'
            _start_path = kwargs['start_path'] if 'start_path' in kwargs else None

            def _update(path_):
                velocities = [0.5 * (
                            cls.a_log(q=path_[_i], p=p_i) + cls.a_log(q=path_[_i + 2],
                                                                      p=p_i))
                              for _i, p_i in enumerate(path_[1:-1])]
                _path_upd = [
                    cls.s_proj(p=cls.a_exp(v=velocities[_i], p=p_i), st=p.st, x0=p_i.x,
                               **_proj_args)
                    for _i, p_i in enumerate(path_[1:-1])]
                return [p] + _path_upd + [q]

            # start with the naive path
            paths = [_start_path]
            for i in range(_n_iter):
                print(f"Inside iteration algorithm: iteration {i + 1}/{_n_iter}.")
                paths += [_update(path_=paths[-1])]
            return paths[-1]

        if alg == 'global-straightening-2':
            _n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            _n_iter = kwargs['n_iter'] if 'n_iter' in kwargs else 10

            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            if 'method' not in _proj_args:
                _proj_args['method'] = 'global-descent'
            _start_path = kwargs['start_path'] if 'start_path' in kwargs else None

            def _update(path_):
                # outgoing directions from x_i to x_{i-1} (minus) and x_{i+1} (plus).
                v_minus = [cls.a_log(q=path_[_i], p=p_i) for _i, p_i in
                           enumerate(path_[1:-1])]
                v_plus = [cls.a_log(q=path_[_i + 2], p=p_i) for _i, p_i in
                          enumerate(path_[1:-1])]
                # outgoing directions projected onto the tangent space of the grove.
                u_minus = [cls.s_proj_vector(v=_v, p=path_[_i + 1]) for _i, _v in
                           enumerate(v_minus)]
                u_plus = [cls.s_proj_vector(v=_v, p=path_[_i + 1]) for _i, _v in
                          enumerate(v_plus)]
                # computed velocities:
                vel_minus = [
                    cls.a_norm(v=v_minus[_i], p=p_i.corr) / cls.s_norm(v=u_minus[_i],
                                                                       p=p_i) * u_minus[
                        _i]
                    for _i, p_i in enumerate(path_[1:-1])]
                vel_plus = [
                    cls.a_norm(v=v_plus[_i], p=p_i.corr) / cls.s_norm(v=u_plus[_i],
                                                                      p=p_i) * u_plus[
                        _i]
                    for _i, p_i in enumerate(path_[1:-1])]
                velocities = [
                    0.5 * cls.s_lift_vector(v=vel_minus[_i] + vel_plus[_i], p=p_i)
                    for _i, p_i in enumerate(path_[1:-1])]

                _path_upd = [
                    cls.s_proj(p=cls.a_exp(v=velocities[_i], p=p_i), st=p.st, x0=p_i.x,
                               **_proj_args)
                    for _i, p_i in enumerate(path_[1:-1])]
                return [p] + _path_upd + [q]

            # start with the naive path
            paths = [_start_path]
            for i in range(_n_iter):
                print(f"Inside iteration algorithm: iteration {i + 1}/{_n_iter}.")
                paths += [_update(path_=paths[-1])]
            return paths[-1]

        if alg == 'global-straighten-extend':
            if 'n_points' not in kwargs and 'n_iter' not in kwargs:
                _n_iter = 5
                raise UserWarning("Neither no. points nor no. iterations was specified,"
                                  " defaulted to 5 iterations,"
                                  "yielding 33 points on the geodesic.")
            elif 'n_iter' not in kwargs:
                _n_iter = int(np.ceil(np.log2(kwargs['n_points'] - 1)))
            else:
                _n_iter = kwargs['n_iter']

            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            elif 'method' not in _proj_args:
                _proj_args['method'] = 'global-descent'

            def _extend(path_):
                _midpoints = [
                    cls.s_proj(cls.a_path_t(p=_p, q=_q, t=0.5), st=_p.st, x0=_p.x,
                               **_proj_args)
                    for _p, _q in zip(path_[:-1], path_[1:])]
                _merged = [None] * (len(path_) + len(_midpoints))
                _merged[::2] = path_
                _merged[1::2] = _midpoints
                return _merged

            def _update(path_):
                _grad = [0.5 * (
                            cls.a_log(q=path_[_i], p=p_i) + cls.a_log(q=path_[_i + 2],
                                                                      p=p_i))
                         for _i, p_i in enumerate(path_[1:-1])]
                _path_upd = [
                    cls.s_proj(p=cls.a_exp(v=_grad[_i], p=p_i), st=p.st, x0=p_i.x,
                               **_proj_args)
                    for _i, p_i in enumerate(path_[1:-1])]
                return [p] + _path_upd + [q]

            # start with the naive path
            paths = [[p, q]]
            for i in range(_n_iter):
                paths.append(_update(path_=_extend(path_=paths[-1])))
            return paths[-1]

        if alg == 'extend-straighten':
            if 'n_points' not in kwargs and 'n_iter' not in kwargs:
                _n_iter = 5
                raise UserWarning(
                    "Neither no. points nor no. iterations was specified, defaulted to 5 iterations,"
                    "yielding 33 points on the geodesic.")
            elif 'n_iter' not in kwargs:
                _n_iter = int(np.ceil(np.log2(kwargs['n_points'] - 1)))
            else:
                _n_iter = kwargs['n_iter']
            _j_iter = kwargs['j_iter'] if 'j_iter' in kwargs else 2
            _proj_args = kwargs['proj_args'] if 'proj_args' in kwargs else {}
            if p.n == 3:
                _proj_args['method'] = 'local'
            elif 'method' not in _proj_args:
                _proj_args['method'] = 'global-descent'

            def _extend(path_):
                _midpoints = [
                    cls.s_proj(cls.a_path_t(p=_p, q=_q, t=0.5), st=_p.st, x0=_p.x,
                               **_proj_args)
                    for _p, _q in zip(path_[:-1], path_[1:])]
                _merged = [None] * (len(path_) + len(_midpoints))
                _merged[::2] = path_
                _merged[1::2] = _midpoints
                return _merged

            def straighten(path_):
                _grad = [0.5 * (
                            cls.a_log(q=path_[_i], p=p_i) + cls.a_log(q=path_[_i + 2],
                                                                      p=p_i))
                         for _i, p_i in enumerate(path_[1:-1])]
                _path_upd = [
                    cls.s_proj(p=cls.a_exp(v=_grad[_i], p=p_i), st=p.st, x0=p_i.x,
                               **_proj_args)
                    for _i, p_i in enumerate(path_[1:-1])]
                return [p] + _path_upd + [q]

            # start with the symmetric projection with 5 points.
            paths = [cls.s_path(p=p, q=q, n_points=5, proj_args=_proj_args)]
            for i in range(_n_iter):
                print(f"outer iteration: {i}")
                ext_path = _extend(path_=paths[-1])
                for j in range(_j_iter):
                    # print(f"inner iteration: {j}")
                    ext_path = straighten(path_=ext_path)
                paths.append(ext_path)
            return paths[-1]
        raise NotImplementedError(f"The algorithm '{alg}' is not implemented (yet?).")

    @staticmethod
    @abc.abstractmethod
    def s_exp(v, p: Trees, **kwargs):
        pass
