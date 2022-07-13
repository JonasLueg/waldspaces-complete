# external imports
import operator
import numpy as np
import networkx as nx
import itertools as it

# package imports
from treespaces.tools.structure_and_split import Split
from treespaces.tools.wald import Tree


def gtp_dist(n, splits_a, splits_b):
    # compute common splits and also the supports (Owen&Provan paper terminology) of the non-common splits
    common_splits_a, common_splits_b, supports = gtp_trees_with_common_support(n,
                                                                               splits_a,
                                                                               splits_b)
    # distance between common splits is just euclidean
    sq_dist_common = sum(
        (common_splits_a[s] - common_splits_b[s]) ** 2 for s in common_splits_a.keys())
    # now compute the distances for each subtree
    sq_dist_parts = sum((np.sqrt(sum(splits_a[s] ** 2 for s in a)) + np.sqrt(
        sum(splits_b[s] ** 2 for s in b))) ** 2
                        for supp_a, supp_b in supports.values() for a, b in
                        zip(supp_a, supp_b))
    return np.sqrt(sq_dist_common + sq_dist_parts)


def gtp_point_on_geodesic(n, splits_a, splits_b, t, **kwargs):
    if t == 0:
        return Tree(n=n, splits=[s for s, _ in splits_a.items()],
                    b=[x for _, x in splits_a.items()])
    if t == 1:
        return Tree(n=n, splits=[s for s, _ in splits_b.items()],
                    b=[x for _, x in splits_b.items()])

    if 'common_splits_a' in kwargs and 'common_splits_b' in kwargs and 'supports' in kwargs:
        common_splits_a = kwargs['common_splits_a']
        common_splits_b = kwargs['common_splits_b']
        supports = kwargs['supports']
    else:
        common_splits_a, common_splits_b, supports = gtp_trees_with_common_support(n,
                                                                                   splits_a,
                                                                                   splits_b)

    # the edge lengths of the resulting wald for the common splits:
    common_splits = {s: (1 - t) * common_splits_a[s] + t * common_splits_b[s] for s in
                     common_splits_a}
    # compute which non-common splits are in the resulting wald plus edge lengths.
    distinct_splits = dict()
    t_ratio = (t / (1 - t)) ** 2
    # keep the ratios squared for saving computational power of making square root (that is why t_ratio is squared)
    ratios = {
        part: [sum(splits_a[s] ** 2 for s in a) / sum(splits_b[s] ** 2 for s in b) for
               a, b in zip(supp_a, supp_b)]
        for part, (supp_a, supp_b) in supports.items()}

    # basically implementing Theorem 2.4 from Owen & Provan paper, for each subtree
    for part, (supp_a, supp_b) in supports.items():
        index = np.argmax([t_ratio <= _r for _r in ratios[part] + [np.inf]])
        splits_bb = {s: splits_b[s] * (t - (1 - t) * np.sqrt(_r))
                     for b_k, _r in zip(supp_b[:index], ratios[part][:index]) for s in
                     b_k}
        splits_aa = {s: splits_a[s] * (1 - t - t / np.sqrt(_r))
                     for a_k, _r in zip(supp_a[index:], ratios[part][index:]) for s in
                     a_k}
        distinct_splits = {**distinct_splits, **splits_aa, **splits_bb}

    splits_t = list({**common_splits, **distinct_splits}.items())
    splits_t.sort(key=operator.itemgetter(0))  # sort by splits
    return Tree(n=n, splits=[s for s, _ in splits_t], b=[x for _, x in splits_t])


def gtp_trees_with_common_support(n, splits_a, splits_b):
    """
    Computes the geodesic tree path problem (GTP) for two dicts of splits, weights;
    common splits possible. """
    print(splits_a)
    print(splits_b)
    # the set of pendant splits
    pendants = {Split(n=n, part1=[i], part2=set(range(n)) - {i}) for i in range(n)}
    common_splits = set(splits_a.keys()) & set(splits_b.keys())

    # TODO i would not need all unmatched edges, just the ones where the opposing set
    #      would be empty (in a subtree)
    #      basically: if a component in subtree_a was empty but component in subtree_b
    #      not, the alg by Owen & Provan
    #      would not work I think. in this case, we could just add a zero length edge
    #      from subtree_b to subtree_a
    #      without care and the algorithm would work again. right now we fill up
    #      everything from subtree_b to
    #      subtree_a that is compatible with everything in subtree_a.

    # check if non-common splits from a are compatible with b and vice versa;
    # those need to be taken care of!
    # note that this only happens if a or b is a degenerate tree,
    # but we want to include this case.
    # note that also PENDANT EDGES can be in here if one is not contained in the other!
    distinct_a, distinct_b = set(splits_a.keys()) - common_splits, set(
        splits_b.keys()) - common_splits
    unmatched_a = {s for s in distinct_a if
                   np.alltrue([t.compatible_with(s) for t in distinct_b])}
    unmatched_b = {s for s in distinct_b if
                   np.alltrue([t.compatible_with(s) for t in distinct_a])}

    # the trees we cut into subtrees (including unmatched and common splits,
    # NO PENDANT EDGES)
    pure_a = (set(splits_a.keys()) | unmatched_b) - pendants
    pure_b = (set(splits_b.keys()) | unmatched_a) - pendants

    # TODO: one can easier check the partitions into subtrees by checking which pairs
    #  of labels are separated?
    cut_splits = (common_splits | unmatched_a | unmatched_b) - pendants
    print(unmatched_a)
    print(unmatched_b)
    print(common_splits - pendants)
    trees_a = gtp_cut_tree_at_splits(n=n, splits=pure_a, cut_splits=cut_splits)
    trees_b = gtp_cut_tree_at_splits(n=n, splits=pure_b, cut_splits=cut_splits)
    # note that keys of both dictionaries must be the same by construction!
    print("gtp common support")
    print(trees_a)
    print(trees_b)
    # next compute the supports (Owen & Provan paper terminology) of the subtrees, and forget about empty subtrees
    # todo: subtrees might be empty for a but not for b!!!
    supports = {
        part: gtp_trees_with_distinct_support({s: splits_a[s] for s in trees_a[part]},
                                              {s: splits_b[s] for s in trees_b[part]})
        for part in trees_a.keys() if trees_a[part] and trees_b[part]}
    # now we have computed everything regarding change of tree structures.

    # common edges for each tree WITH edge weights (unmatched edges get weight 0 in the other tree)
    common_splits = common_splits | unmatched_a | unmatched_b
    common_splits_a = {s: splits_a[s] if s in splits_a else 0 for s in common_splits}
    common_splits_b = {s: splits_b[s] if s in splits_b else 0 for s in common_splits}
    return common_splits_a, common_splits_b, supports


def gtp_cut_tree_at_splits(n, splits, cut_splits):
    """ A tree, given by splits, is cut at all edges in cut_splits.

    Starting with the partition that consists of all labels and is assigned all splits,
    the tree is successively cut into parts by the splits in cut_splits.
    Accordingly, the set of labels is cut successively into parts and the set of all
    splits is also cut successively into the respective parts.

    Parameters
    ----------
    n : int
        The number of labels in the tree.
    splits : list of Split
        The tree given via its splits. Each split corresponds to an edge.
    cut_splits : list of Split
        A subset of splits, the edges at which the tree is cut.

    Returns
    -------
    partition : dict of tuple, tuple
        A dictionary, where the keys form a partition of the set of labels (0,...,n-1),
        and each key is assigned the tuple of splits that are part of the subtree that
        the respective set of labels is spanning.
    """
    partition = {tuple(range(n)): splits}

    for cut in cut_splits:
        try:
            labels, subtree = [(_, subtree) for _, subtree in partition.items()
                               if cut in subtree][0]
        except IndexError:
            continue
        splits = set(subtree) - {cut}
        part1, part2 = set(labels) & set(cut.part1), set(labels) & set(cut.part2)
        # TODO : potential error here!!! -> the subtrees are not splitted correctly.
        #           should be checking for if the part of s that is towards the cut
        #           split is equal to the part1 as defined above. all those splits are
        #           then subtree1....!!!
        subtree1 = {s for s in splits if part1.issubset(set(s.directed_to_split(cut)))}
        subtree2 = splits - subtree1

        partition.pop(labels)
        partition = {**partition,
                     tuple(part1): tuple(subtree1),
                     tuple(part2): tuple(subtree2)
                     }
    return partition


# old code, for saving.
# partition = {tuple(range(n)): splits}
# # successively cut the tree into subtrees
# for cut in cut_splits:
#     dummy = [(part, subtree) for part, subtree in partition.items() if
#              cut in subtree]
#     if dummy:
#         part, subtree = dummy[0]
#     else:
#         continue
#     # print(f"{part}, subtree={subtree}")
#     # cut it in two parts according to the split 'cut'
#     part1, part2 = set(part) & set(cut.part1), set(part) & set(cut.part2)
#     subtree1 = tuple(s for s in subtree if
#                      s.contains(part2) and not s.contains(part1) and s != cut)
#     subtree2 = tuple(s for s in subtree if
#                      s.contains(part1) and not s.contains(part2) and s != cut)
#     # print(f"part1={part1}, part2={part2}, subtree1={subtree1},
#     subtree2={subtree2}.")
#     # put those two subtrees back in partition and leave out the old subtree
#     partition = {_part: _subtree for _part, _subtree in partition.items() if
#                  _part != part}
#     # print(partition)
#     partition = {**partition, tuple(part1): subtree1, tuple(part2): subtree2}
# return partition


def gtp_trees_with_distinct_support(splits_a, splits_b, **kwargs):
    """Compute the support that corresponds to a geodesic for disjoint split sets.

    This is essentially the GTP algorithm from [1], starting with a cone path and
    iteratively updating the support, solving in each iteration an extension problem for
    each support pair.

    The Extension Problem gives a minimum cut of a graph and two-set partitions C1 and
    C2 of A, and D1 and D2 of B, respectively. If the value of the minimum cut is
    greater or equal to one minus some tolerance, then the support pair (A,B) is split
    into (C1,D1) and (C2,D2).

    Parameters
    ----------
    splits_a : dict of Split, float
        The splits in A and their respective lengths.
    splits_b : dict of Split, float
        The splits in B and their respective lengths.

    Returns
    -------
    support_a : tuple of tuple
        The support partition of A corresponding to a geodesic.
    support_b : tuple of tuple
        The support partition of B corresponding to a geodesic.
    """
    tol = kwargs.pop('tol', 10**-10)
    old_support_a = (tuple(splits_a.keys()),)
    old_support_b = (tuple(splits_b.keys()),)
    weights_a = {split: splits_a[split] ** 2 for split in splits_a}
    weights_b = {split: splits_b[split] ** 2 for split in splits_b}
    while 1:
        new_support_a, new_support_b = tuple(), tuple()
        for pair_a, pair_b in zip(old_support_a, old_support_b):
            pair_a_w = {s: weights_a[s] for s in pair_a}
            pair_b_w = {s: weights_b[s] for s in pair_b}
            value, c1, c2, d1, d2 = gtp_solve_extension_problem(pair_a_w, pair_b_w)
            if value >= 1 - tol:
                new_support_a += (pair_a,)
                new_support_b += (pair_b,)
            else:
                new_support_a += (c1, c2)
                new_support_b += (d1, d2)
        if len(new_support_a) == len(old_support_a):
            return new_support_a, new_support_b
        else:
            old_support_a, old_support_b = new_support_a, new_support_b


def gtp_solve_extension_problem(sq_splits_a, sq_splits_b):
    """Solve the extension problem in [1] for sets of splits with squared weights.

    Solving the min weight vertex cover with respect to the incompatibility graph in
    the Extension Problem in [1] is equivalent to solving the minimum cut problem for
    the following directed graph with edges that have 'capacities'.
    The set of vertices are the splits in A, the splits in B, a sink and a source node.
    The source is connected to all splits in A, each edge has the normalized squared
    weight of the split it is attached to. Analogously, each split in B is connected to
    the sink and the corresponding edge has normalized squared weight of the split in B.
    Finally, each split in A is attached to a split in B whenever the splits are not
    compatible. The edge is given infinite capacity.

    The minimum cut returns the two-set partition (V, V_bar) of the set of vertices and
    its value, that is the sum of all capacities of edges from V to V_bar, such that the
    source is in V and the sink is in V_bar.

    If the value is larger or equal than one (possibly with respect to some tolerance),
    then a geodesic is found and there is no need to update anything.
    Else, the sets A and B are separated into sets
    C_1 = A intersection V_bar, C_2 = A intersection V,
    D_1 = B intersection V_bar, D_2 = B intersection V.
    Then, the new support is (i.e. A and B are replaced with) (C_1, C_2) and (D_1, D_2)
    (here, the notation from [1], GTP algorithm is used).

    Parameters
    ----------
    sq_splits_a : dict of Split, float
        Dictionary of splits in A with squared length associated to each split.
    sq_splits_b : dict of Split, float
        Dictionary of splits in B with squared length associated to each split.

    Returns
    -------
    value : float
        The value of the minimum cut.
    c1 : set of Split
        First part of A that it is split into.
    c2 : set of Split
        Second part of A that it is split into.
    d1 : set of Split
        First part of B that it is split into.
    d2 : set of Split
        Second part of B that it is split into.
    """
    total_a, total_b = sum(sq_splits_a.values()), sum(sq_splits_b.values())
    graph = nx.DiGraph()
    for split, weight in sq_splits_a.items():
        graph.add_edge('source', split, capacity=weight / total_a)
    for split, weight in sq_splits_b.items():
        graph.add_edge(split, 'sink', capacity=weight / total_b)
    for split_a, split_b in it.product(sq_splits_a.keys(), sq_splits_b.keys()):
        if not split_a.compatible_with(split_b):
            graph.add_edge(split_a, split_b)

    min_value, (v, v_bar) = nx.minimum_cut(graph, 'source', 'sink')
    a = set(sq_splits_a.keys())
    b = set(sq_splits_b.keys())
    v = set(v)
    v_bar = set(v_bar)
    return min_value, tuple(a & v_bar), tuple(a & v), tuple(b & v_bar), tuple(b & v)
