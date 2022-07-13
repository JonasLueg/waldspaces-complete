import numpy as np


def postfix(number):
    if number % 10 in [1, 2, 3] and number not in [11, 12, 13]:
        return ["st", "nd", "rd"][number % 10 - 1]
    else:
        return "th"


def give_coords_n5_interior_splits(_path, sp0, sp1, sp2, sp3):
    """ Returns the coordinates of the walds with 5 leaves in 2 dimensions; the values of the interior splits are taken.

    {sp0, sp1}, {sp1, sp2} and {sp2, sp3} are compatible, {sp0, sp2}, {sp1, sp3} are not.
    """
    coords_x = []
    coords_y = []
    for _wald in _path:
        _x, _y = 0, 0
        if sp0 in _wald.st.split_collection[0]:
            _x = _wald.x[_wald.st.where(sp0)]
        if sp1 in _wald.st.split_collection[0]:
            _y = _wald.x[_wald.st.where(sp1)]
        if sp2 in _wald.st.split_collection[0]:
            _x = -_wald.x[_wald.st.where(sp2)]
        if sp3 in _wald.st.split_collection[0]:
            _y = -_wald.x[_wald.st.where(sp3)]
        coords_x.append(_x)
        coords_y.append(_y)
    return coords_x, coords_y


def give_bhv_coords_n5_interior_splits(_path, sp0, sp1, sp2, sp3):
    """ Returns the coordinates of the walds with 5 leaves in 2 dimensions; the values of the interior splits are taken.

    {sp0, sp1}, {sp1, sp2} and {sp2, sp3} are compatible, {sp0, sp2}, {sp1, sp3} are not.
    """
    coords_x = []
    coords_y = []
    for _wald in _path:
        _x, _y = 0, 0
        if sp0 in _wald.st.split_collection[0]:
            _x = _wald.b[_wald.st.where(sp0)]
        if sp1 in _wald.st.split_collection[0]:
            _y = _wald.b[_wald.st.where(sp1)]
        if sp2 in _wald.st.split_collection[0]:
            _x = -_wald.b[_wald.st.where(sp2)]
        if sp3 in _wald.st.split_collection[0]:
            _y = -_wald.b[_wald.st.where(sp3)]
        coords_x.append(_x)
        coords_y.append(_y)
    return coords_x, coords_y


def entropy(_path, geometry):
    """ Measures how equidistant the points in _path are. Zero means they are perfectly equidistant. """
    _ideal_value = geometry.length(path_=_path)
    try:
        dists = [geometry.a_dist(p=_path[_i], q=_path[_i + 1], squared=False) for _i in range(0, len(_path) - 1)]
    except NotImplementedError:
        dists = [geometry.s_dist(p=_path[_i], q=_path[_i + 1], squared=False) for _i in range(0, len(_path) - 1)]
    return np.sum([x * np.log((len(_path) - 1) * x / _ideal_value) for x in dists])


def energy(_path, geometry):
    try:
        dists = [geometry.a_dist(p=_path[_i], q=_path[_i + 1], squared=True) for _i in range(0, len(_path) - 1)]
    except NotImplementedError:
        dists = [geometry.s_dist(p=_path[_i], q=_path[_i + 1], squared=True) for _i in range(0, len(_path) - 1)]
    return 0.5 * np.sum(dists)


def angles(_path, waldspace):
    _angles = []
    for i, wald in enumerate(_path):
        if i == 0 or i == len(_path) - 1:
            continue
        # ambient space logs in riemannian fashion
        v_minus = waldspace.g.a_log(q=_path[i - 1].corr, p=wald.corr)
        v_plus = waldspace.g.a_log(q=_path[i + 1].corr, p=wald.corr)
        # project vectors onto wald space tangent space
        u_minus = waldspace.g.s_proj_vector(v=v_minus, p=wald)
        u_plus = waldspace.g.s_proj_vector(v=v_plus, p=wald)
        # normalize the vectors to have norm 1 in wald space tangent space
        u_minus = u_minus / waldspace.g.s_norm(v=u_minus, p=wald)
        u_plus = u_plus / waldspace.g.s_norm(v=u_plus, p=wald)
        # compute the angle in degrees.
        _angles.append(360 / 2 / np.pi * np.arccos(waldspace.g.s_inner(u=u_minus, v=u_plus, p=wald)))
    return _angles
