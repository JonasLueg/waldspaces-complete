import numpy as np
import itertools as it
import scipy.linalg as la
from treespaces.tools.structure_and_split import Split, Structure


def equivalence_partition(group, relation):
    """Partitions a set of objects into equivalence classes
    Taken from
    https://stackoverflow.com/questions/38924421/is-there-a-standard-way-to-partition-an-interable-into-equivalence-classes-given

    Args:
        group: collection of objects to be partitioned
        relation: equivalence relation. I.e. relation(o1,o2) evaluates to True
            if and only if o1 and o2 are equivalent

    Returns: classes, partitions
        classes: A sequence of sets. Each one is an equivalence class
        partitions: A dictionary mapping objects to equivalence classes
    """
    classes = []
    partitions = {}
    for o in group:  # for each object
        # find the class it is in
        found = False
        for c in classes:
            if relation(next(iter(c)), o):  # is it equivalent to this class?
                c.add(o)
                partitions[o] = c
                found = True
                break
        if not found:  # it is in a new class
            classes.append({o})
            partitions[o] = classes[-1]
    classes = tuple(map(tuple, classes))
    partitions = {key: tuple(item) for key, item in partitions.items()}
    return classes, partitions


def compute_dist_from_structure_and_coordinates(st: Structure, x):
    """ Computes the distance matrix from split representation by taking sum over lengths of splits on paths. """
    _dist = np.full(shape=(st.n, st.n), fill_value=np.inf)
    np.fill_diagonal(a=_dist, val=0)  # diagonal is zero
    # TODO: get the case where x is empty...!!!
    _ell = [np.maximum(0, np.real(-np.log(1 - _x))) if _x < 1 else np.inf for _x in x]
    # compute all other values
    for i, labels in enumerate(st.partition):  # for each component
        for u, v in it.combinations(labels, 2):
            _dist[u, v] = np.sum([_ell[j + st.sep[i]]
                                  for j, sp in enumerate(st.split_collection[i]) if
                                  sp.separates(u, v)])
            _dist[v, u] = _dist[u, v]
    return _dist


def compute_structure_from_dist(dist, btol=10 ** -10):
    _n = dist.shape[0]
    """ Computes the split representation of a forest characterized by a distance matrix 
    that can have entries = inf."""
    _partition, _ = equivalence_partition(group=range(_n),
                                          relation=lambda i, j: dist[i, j] < np.inf)
    # for each component, compute the splits in that TREE
    _split_collection = [
        _compute_splits_from_sub_dist(sub_dist=dist, sub_labels=labels, btol=btol)
        for labels in _partition]
    return Structure(n=_n, partition=_partition, split_collection=_split_collection)


def _compute_splits_from_sub_dist(sub_dist, sub_labels, btol=10 ** -10):
    """ Computes the split representation of a single tree characterized by a distance matrix.

    The label set might be smaller than the dimension of dist, in which case, canonically, the sub-matrix is taken.
    """
    _n = sub_dist.shape[0]
    # test if we have only one label, then we have zero splits.
    # note that sub_dist might be bigger!
    if len(sub_labels) == 1:
        return tuple()

    # plan is the following:
    # generate all pairs, i.e. sets of the form {i, j}
    # compute all 2-splits, i.e. splits of the form {i, j}|{k, l}
    # compute their length and take only those with positive length
    # (i.e. they are compatible with forest structure)
    # the pair that appears most often in those 2-splits must be a cherry
    # the cherry is reduced to a single leaf with a new label and the process is
    # repeated
    # after that, all new labels are resolved back to their original leaves,
    # giving the tree

    used_keys = set(sub_labels)
    key_dict = {u: {u} for u in used_keys}
    # important: pairs is a list for it to be compatible with sorting statements
    pairs = [{u, v} for u, v in it.product(used_keys, repeat=2) if u < v]
    # for technical reasons: we have new keys, and a maximum of 2*N - 3, and need to
    # compare those splits, thus w.r.t. n
    _m = 2 * _n - 3
    pair_splits = {Split(n=_m, part1=p1, part2=p2) for p1, p2 in
                   it.product(pairs, repeat=2) if not (p1 & p2)}
    _ps_lengths = {sp: compute_length_of_split_from_dist(split=sp, dist=sub_dist) for sp
                   in pair_splits}
    pair_splits = {sp for sp in pair_splits if _ps_lengths[sp] > btol}

    # start with the pendant edges, although not all of them need to exist!!!
    splits = {Split(n=_m, part1={u}, part2=used_keys - {u}) for u in used_keys}
    _s_lengths = {sp: compute_length_of_split_from_dist(split=sp, dist=sub_dist) for sp
                  in splits}
    splits = {sp for sp in splits if _s_lengths[sp] > btol}

    while pair_splits:
        # compute how often a pair appears, take the most frequent one as the next
        # cherry to reduce
        frequencies = [len([sp for sp in pair_splits if sp.contains(pair)]) for pair in
                       pairs]
        # most frequent pair is next cherry (set of two used keys)
        next_cherry = set(pairs[np.argsort(frequencies)[-1]])
        # the cherry determines a unique split, it's saved in its unresolved state
        # (identify with new key)
        splits.add(Split(n=_m, part1=next_cherry, part2=(used_keys - next_cherry)))
        new_key = max(key_dict.keys()) + 1
        # update the pair_splits. some pairs are vanishing
        # (namely those that have the next_cherry as one part).
        pair_splits = {
            Split(n=_m,
                  part1=[new_key if u in next_cherry else u for u in sp.part1],
                  part2=[new_key if u in next_cherry else u for u in sp.part2]
                  )
            for sp in pair_splits if not sp.contains(next_cherry)}
        # store the new key that now encodes the two labels contained in the cherry
        key_dict[new_key] = set.union(*[key_dict[k] for k in next_cherry])
        # update the label set with the new key and removing the labels contained
        # in the cherry
        used_keys = used_keys - next_cherry | {new_key}
        # update the pairs that are contained in the set of pair_splits
        pairs = [{new_key if u in next_cherry else u for u in p}
                 for p in pairs if next_cherry != set(p)
                 ]
        # eliminate doubles (frozenset is needed since a set must contain hashable
        # types only).
        pairs = list(map(set, set(map(frozenset, pairs))))

    # finally, a star tree is what's left, we add those splits to the tree_splits
    # as well
    # they might very well be already contained in tree_splits
    splits = splits | {Split(n=_m, part1=used_keys - {u}, part2={u}) for u in used_keys}

    # now we can convert the splits consisting of unresolved keys back to splits
    # containing only original labels
    # and again eliminate doubles: that's why we use the set.
    splits = {Split(n=_n, part1=set.union(*[key_dict[u] for u in sp.part2]),
                    part2=set.union(*[key_dict[u] for u in sp.part1]))
              for sp in splits}
    _s_lengths = {sp: compute_length_of_split_from_dist(split=sp, dist=sub_dist) for sp
                  in splits}
    splits = sorted([sp for sp in splits if _s_lengths[sp] > btol])
    return splits


def compute_length_of_split_from_dist(split: Split, dist):
    """ Gives back the length of a split according to the distance matrix dist."""
    if not split:  # if one part is empty, define this as minus infinity.
        return -np.inf

    pairs1 = list(it.combinations(split.part1, 2) if len(split.part1) > 1 else [
        (split.part1[0],) * 2])
    pairs2 = list(it.combinations(split.part2, 2) if len(split.part2) > 1 else [
        (split.part2[0],) * 2])
    # take care of infinite cases. some splits might have infinite length
    # (although we generally want to avoid that)
    return 0.5 * np.min(
        [[np.inf if np.isinf(dist[p1[0], p2[0]]) or np.isinf(dist[p1[1], p2[1]])
          else dist[p1[0], p2[0]] + dist[p1[1], p2[1]] - dist[p1] - dist[p2]
          for p1 in pairs1] for p2 in pairs2])


def give_nni_candidates(st: Structure, sp: Split):
    """ Gives the two neighboring fully resolved structures for a split that is supposed to be zero. """
    # assume that split 'sp' splits into (set_a + set_b) vs. (set_c + set_d)
    set_a, set_b, set_c, set_d = set(), set(), set(), set()
    set_ab, set_cd = set(sp.part1), set(sp.part2)
    rest_dict = {s: set(s.directed_away_from_split(sp)) for s in st.split_collection[0]
                 if s != sp}
    empty_count = 4  # stop if all sets (set_a, ..., set_d) are filled with something.
    for s in sorted(rest_dict, key=lambda k: len(rest_dict[k]), reverse=True):
        rest = rest_dict[s]
        if rest.issubset(set_ab):
            if set_a.issubset(rest):
                set_a = rest
                empty_count -= 1
            elif rest.issubset(set_a):
                continue
            elif set_b.issubset(rest):
                set_b = rest
                empty_count -= 1
            elif rest.issubset(set_b):
                continue
        else:
            if set_c.issubset(rest):
                set_c = rest
                empty_count -= 1
            elif rest.issubset(set_c):
                continue
            elif set_d.issubset(rest):
                set_d = rest
                empty_count -= 1
            elif rest.issubset(set_d):
                continue
        if not empty_count:
            break
    sp1 = Split(n=sp.n, part1=set_a | set_c, part2=set_b | set_d)
    sp2 = Split(n=sp.n, part1=set_a | set_d, part2=set_b | set_c)
    split_collection1 = [[sp1] + [s for s in st.split_collection[0] if s != sp]]
    split_collection2 = [[sp2] + [s for s in st.split_collection[0] if s != sp]]
    st1 = Structure(n=st.n, partition=st.partition, split_collection=split_collection1)
    st2 = Structure(n=st.n, partition=st.partition, split_collection=split_collection2)
    return [(st1, sp1), (st2, sp2)]


def make_splits(n):
    """ Generates all possible splits of a collection."""
    if n <= 1:
        raise ValueError(f"n must be greater or equal than 2, current value is n={n}.")
    if n == 2:
        yield Split(n=n, part1=[0], part2=[1])
    else:
        for split in make_splits(n=n - 1):
            yield Split(n=n, part1=split.part1, part2=split.part2 + (n - 1,))
            yield Split(n=n, part1=split.part1 + (n - 1,), part2=split.part2)
        yield Split(n=n, part1=list(range(n - 1)), part2=[n - 1])


def make_structures(n):
    """ Generates all possible sets of compatible splits of a collection.

    This only works well for len(collection) < 8.
    """
    if n <= 1:
        raise ValueError("The collection must have 2 elements or more.")
    if n in [2, 3]:
        yield Structure(n=n, partition=(tuple(range(n)),),
                        split_collection=(make_splits(n),))
    else:
        pendant_split = Split(n=n, part1=[n - 1], part2=list(range(n - 1)))
        for st in make_structures(n - 1):
            for s in st.split_collection[0]:
                new_split_set = (pendant_split,)
                a, b = set(s.part1), set(s.part2)
                for t in st.split_collection[0]:
                    c, d = set(t.part1), set(t.part2)
                    if t != s:
                        if a.issubset(d) or b.issubset(d):
                            new_split_set += (
                                Split(n=n, part1=t.part1, part2=t.part2 + (n - 1,)),)
                        else:
                            new_split_set += (
                                Split(n=n, part1=t.part2, part2=t.part1 + (n - 1,)),)
                    else:
                        new_split_set += (
                            Split(n=n, part1=s.part1, part2=s.part2 + (n - 1,)),)
                        new_split_set += (
                            Split(n=n, part1=s.part2, part2=s.part1 + (n - 1,)),)
                yield Structure(n=n, partition=(tuple(range(n)),),
                                split_collection=(new_split_set,))


def compute_chart(st: Structure):
    """ Computes the chart of a grove with structure `st`. Returns map that takes coordinates.

    Arguments:
        :param Structure st: The forest structure.

    Returns:
        A map from vector of length 'no. of splits' to the correlation matrices (numpy array).
    """

    def _chart(x):
        """ Input is a flat vector or list x (Nye parametrization). """
        _w = st.ravel(x=x)
        _corr = np.zeros((st.n, st.n))
        for i, d in enumerate(st.leaf_paths):
            for (
                        u,
                        v), split_list in d.items():  # assumes each index appears only once!
                _corr[u, v] = np.prod([1 - _w[i][k] for k in split_list])
                _corr[v, u] = _corr[u, v]
        np.fill_diagonal(a=_corr, val=1)
        return _corr

    return _chart


def compute_chart_gradient(st: Structure, chart):
    """ Computes the gradient of the chart of a grove with structure `st`. Returns map that takes coordinates. """

    def _chart_gradient(x):
        """ Input is a flat vector or list x (Nye parametrization). """
        _x_with_zeros = [[y if i != k else 0 for i, y in enumerate(x)] for k in
                         range(len(x))]
        _corr_gradient = [st.support[k] * -chart(xk) for k, xk in
                          enumerate(_x_with_zeros)]
        return _corr_gradient

    return _chart_gradient


def compute_christoffel_symbols(st: Structure, chart, gradient):
    """ Computes the christoffel symbols of a grove with structure `st`. Returns map that takes coordinates. """

    def _christoffel_symbols(x, **kwargs):
        """ Input is a flat vector or list x (Nye parametrization). """
        _chart_x = chart(x) if 'chart_x' not in kwargs else kwargs['chart_x']
        _chart_x_inv = la.inv(_chart_x) if 'chart_x_inv' not in kwargs else kwargs[
            'chart_x_inv']
        _gradients_x = gradient(x) if 'gradient_x' not in kwargs else kwargs[
            'gradient_x']
        # TODO: the line below might be optimized, since gradients might be quite sparse (generally are)
        _m_gradients_x = [_chart_x_inv.dot(grad) for grad in _gradients_x]
        # TODO: CAUTION: here, the transpose is actually relevant! A.dot(B) is not always symmetric if A and B are.
        _gram_matrix = np.array(
            [[np.sum(grad_i * grad_j.T) for grad_i in _m_gradients_x] for grad_j in
             _m_gradients_x])
        # TODO: take care of the case where we have 0 edges => the matrix has dimensions 0 x 0.
        _gram_matrix_inv = la.inv(_gram_matrix)

        # compute the second derivatives: taking derivative of two edges from different components, the hessian is 0.
        _m_gradients_x_raveled = st.ravel(x=_m_gradients_x)
        _sep = st.sep
        _hessian = [[np.zeros(shape=_chart_x.shape) for _ in _m_gradients_x] for _ in
                    _m_gradients_x]
        for k, splits in enumerate(st.split_collection):
            for (i, sp_A), (j, sp_B) in it.combinations(enumerate(splits), 2):
                # transform the indices into global indices
                i, j = _sep[k] + i, _sep[k] + j
                a_outer, b_outer = sp_A.directed_away_from_split(
                    sp_B), sp_B.directed_away_from_split(sp_A)
                for u, v in it.product(a_outer, b_outer):
                    _hessian[i][j][u, v] = _hessian[i][j][v, u] = _chart_x[u, v] / (
                            1 - x[i]) / (1 - x[j])
                # multiply with _p^{-1} from the left!
                _hessian[i][j] = _hessian[j][i] = _chart_x_inv.dot(_hessian[i][j])
        _x_with_zeros = [
            [[y if k != i and k != j else 0 for k, y in enumerate(x)] for j in
             range(len(x))]
            for i in range(len(x))]
        _hessian = [st.support[i] * st.support[j] * chart(xij) for i, xi in
                    enumerate(_x_with_zeros) for j, xij in
                    enumerate(xi)]
        print(_hessian)
        # now for the christoffel symbols:
        _dummy = [
            [2 * _hessian[i][j] - np.dot(p_i, p_j) - np.dot(p_j, p_i) for i, p_i in
             enumerate(_m_gradients_x)]
            for j, p_j in enumerate(_m_gradients_x)]
        _dummy2 = [
            [np.array([np.sum(_dummy[i][j] * p_k) for p_k in _m_gradients_x]) for i in
             range(len(x))]
            for j in range(len(x))]
        # TODO: exploit symmetry in i, j here. might double the speed...
        _symbols = [[[0.5 * np.sum(gm * _dummy2[i][j]) for i in range(len(x))] for j in
                     range(len(x))]
                    for gm in _gram_matrix_inv]
        return _symbols

    return _christoffel_symbols


def compute_curvature_symbols(st: Structure, chart, gradient):
    """ Computes the curvature symbols of a grove with structure `st`. Returns a map that takes coordinates. """

    def _curvature_symbols(x, **kwargs):
        """ Input is a flat vector or list x (Nye parametrization). """
        _chart_x = chart(x) if 'chart_x' not in kwargs else kwargs['chart_x']
        _chart_x_inv = la.inv(_chart_x) if 'chart_x_inv' not in kwargs else kwargs[
            'chart_x_inv']
        _gradients_x = gradient(x) if 'gradient_x' not in kwargs else kwargs[
            'gradient_x']
        # TODO: the line below might be optimized, since gradients might be quite sparse (generally are)
        _m_gradients_x = [_chart_x_inv.dot(grad) for grad in _gradients_x]
        # TODO: CAUTION: here, the transpose is actually relevant! A*B is not always symmetric if A and B are.
        _gram_matrix = np.array(
            [[np.sum(grad_i * grad_j.T) for grad_i in _m_gradients_x] for grad_j in
             _m_gradients_x])
        # TODO: take care of the case where we have 0 edges => the matrix has dimensions 0 x 0.
        _gram_matrix_inv = la.inv(_gram_matrix)
        # compute the second derivatives: taking derivative of two edges from different components, the hessian is 0.
        _sep = st.sep
        _hessian = [[np.zeros(shape=_chart_x.shape) for _ in _m_gradients_x] for _ in
                    _m_gradients_x]
        for k, splits in enumerate(st.split_collection):
            for (i, sp_A), (j, sp_B) in it.combinations(enumerate(splits), 2):
                # transform the indices into global indices
                i, j = _sep[k] + i, _sep[k] + j
                a_outer, b_outer = sp_A.directed_away_from_split(
                    sp_B), sp_B.directed_away_from_split(sp_A)
                for u, v in it.product(a_outer, b_outer):
                    _hessian[i][j][u, v] = _hessian[i][j][v, u] = _chart_x[u, v] / (
                            1 - x[i]) / (1 - x[j])
                # multiply with _p^{-1} from the left!
                _hessian[i][j] = _hessian[j][i] = _chart_x_inv.dot(_hessian[i][j])

        _dummy = np.array([[[np.sum(
            2 * _hessian[i][j] * r_h.T - r_j.dot(r_i) * r_h.T - r_i.dot(r_j) * r_h.T)
            for j, r_j in enumerate(_m_gradients_x)]
            for i, r_i in enumerate(_m_gradients_x)]
            for r_h in _m_gradients_x])
        _part1 = np.array(
            [[[[np.sum([[_gram_matrix_inv[a, h] * _dummy[j, k, a] * _dummy[i, s, h]
                         for a in range(len(x))] for h in range(len(x))])
                for s in range(len(x))]
               for k in range(len(x))]
              for j in range(len(x))]
             for i in range(len(x))])
        _part2 = np.array(
            [[[[np.sum([[_gram_matrix_inv[a, h] * _dummy[i, k, a] * _dummy[j, s, h]
                         for a in range(len(x))] for h in range(len(x))])
                for s in range(len(x))]
               for k in range(len(x))]
              for j in range(len(x))]
             for i in range(len(x))])
        _part3 = np.array([[[[np.sum(
            2 * _hessian[i][k] * _hessian[j][s].T - 2 * _hessian[j][k] * _hessian[i][
                s].T - _hessian[i][k].dot(
                r_s) * r_j.T + _hessian[j][k].dot(r_s) * r_i.T - _hessian[i][k].dot(
                r_j) * r_s.T + _hessian[j][k].dot(
                r_i) * r_s.T - _hessian[j][s].dot(r_i) * r_k.T - _hessian[j][s].dot(
                r_k) * r_i.T + _hessian[i][s].dot(
                r_j) * r_k.T + _hessian[i][s].dot(r_k) * r_j.T)
            for s, r_s in enumerate(_m_gradients_x)]
            for k, r_k in enumerate(_m_gradients_x)]
            for j, r_j in enumerate(_m_gradients_x)]
            for i, r_i in enumerate(_m_gradients_x)])
        _symbols = 0.25 * (_part1 - _part2) + 0.5 * _part3
        return _symbols

    return _curvature_symbols


def compute_sectional_curvature_symbols(st: Structure, chart, gradient):
    """ Computes the curvature symbols of a grove with structure `st`. Returns a map that takes coordinates. """

    def _sectional_curvature_symbols(x, **kwargs):
        """ Input is a flat vector or list x (Nye parametrization). """

        def _tr(a, b):
            d1 = np.sum(a * b.T)
            d2 = np.trace(np.dot(a, b))
            if not np.allclose(d1, d2):
                raise ValueError("Calculation of trace(A*B) went wrong.")
            return d1

        _chart_x = chart(x) if 'chart_x' not in kwargs else kwargs['chart_x']
        # print(f"value x is = {x}")
        _chart_x_inv = la.inv(_chart_x) if 'chart_x_inv' not in kwargs else kwargs[
            'chart_x_inv']
        _gradients_x = gradient(x) if 'gradient_x' not in kwargs else kwargs[
            'gradient_x']
        # TODO: the line below might be optimized, since gradients might be quite sparse (generally are)
        # print(f"gradient = {_gradients_x[0]}")
        _m_gradients_x = [_chart_x_inv.dot(grad) for grad in _gradients_x]
        # print("NEW COMPUTATION OF CURVATURE SYMBOLS.")
        # TODO: CAUTION: here, the transpose is actually relevant! A*B is not always symmetric if A and B are.
        _gram_matrix = np.array(
            [[np.sum(grad_i * grad_j.T) for grad_i in _m_gradients_x] for grad_j in
             _m_gradients_x])
        # TODO: take care of the case where we have 0 edges => the matrix has dimensions 0 x 0.
        _gram_matrix_inv = la.inv(_gram_matrix)
        # print("Inside curvature")
        # print(f"inv gram matrix = {np.round(_gram_matrix_inv, 6)}.")
        # compute the second derivatives: taking derivative of two edges from different components, the hessian is 0.
        _sep = st.sep
        # _hessian = [[np.zeros(shape=_chart_x.shape) for _ in _m_gradients_x] for _ in _m_gradients_x]
        # for k, splits in enumerate(st.split_collection):
        #     for (i, sp_A), (j, sp_B) in it.combinations(enumerate(splits), 2):
        #         # transform the indices into global indices
        #         i, j = _sep[k] + i, _sep[k] + j
        #         a_outer, b_outer = sp_A.directed_away_from_split(sp_B), sp_B.directed_away_from_split(sp_A)
        #         for u, v in it.product(a_outer, b_outer):
        #             _hessian[i][j][u, v] = _hessian[i][j][v, u] = _chart_x[u, v] / (1 - x[i]) / (1 - x[j])
        #         # multiply with _p^{-1} from the left!
        #         _hessian[i][j] = _hessian[j][i] = _chart_x_inv.dot(_hessian[i][j])
        #
        _x_with_zeros = [
            [[y if k != i and k != j else 0 for k, y in enumerate(x)] for j in
             range(len(x))]
            for i in range(len(x))]
        # print(st.support[0])
        _hessian = [
            [_chart_x_inv.dot(
                st.support[i] * st.support[j] * chart(xij)) if i != j else np.zeros(
                _chart_x.shape)
             for j, xij in enumerate(xi)] for i, xi in enumerate(_x_with_zeros)]
        # print(_m_gradients_x)

        for i in range(len(_m_gradients_x)):
            if not np.allclose(_hessian[i][i], np.zeros(_hessian[i][i].shape)):
                raise ValueError("diagonal must be zero.")

        _dummy = np.array(
            [[[_tr(2 * _hessian[i][j] - np.dot(r_j, r_i) - np.dot(r_i, r_j), r_h)
               for j, r_j in enumerate(_m_gradients_x)]
              for i, r_i in enumerate(_m_gradients_x)]
             for r_h in _m_gradients_x])
        _part1 = 0.25 * np.array(
            [[np.sum([[_gram_matrix_inv[a, h] * _dummy[a, i, j] * _dummy[h, i, j]
                       for a in range(len(x))] for h in range(len(x))])
              for j in range(len(x))] for i in range(len(x))])
        _part2 = np.array(
            [[np.sum([[_gram_matrix_inv[a, h] * _dummy[a, i, i] * _dummy[h, j, j]
                       for a in range(len(x))] for h in range(len(x))])
              for j in range(len(x))] for i in range(len(x))])
        _part3 = np.array([[np.sum(
            r_i.dot(r_j) * _hessian[i][j].T + r_j.dot(r_i) * _hessian[i][j].T - 2 *
            _hessian[i][j] * _hessian[i][j].T)
            for j, r_j in enumerate(_m_gradients_x)] for i, r_i in
            enumerate(_m_gradients_x)])
        # print(f"Inv P = {_chart_x_inv}.")
        # print(f"Hessian[0][1] = {st.support[0] * st.support[1] * chart(_x_with_zeros[0][1])}.")
        # print(f"the inv_ * grads = {_m_gradients_x[0]}")
        # print(f"Hessian = {_hessian}.")
        # print(f"part1 = {_part1},\npart2 = {_part2},\npart3 = {_part3}.")
        _symbols = _part1 - _part2 + _part3
        return _symbols

    return _sectional_curvature_symbols
