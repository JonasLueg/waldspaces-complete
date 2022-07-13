"""
The specification of this file is that it is not permitted to import other package-level files (in order to avoid
circular dependencies).
"""

# system imports
import functools
import numpy as np
import itertools as it


@functools.total_ordering
class Split(object):
    """ A split {part1, part2} with respect to the set {0,...,n-1} is a two-set partition of this set.

    * We also allow splits to be a two-set partition of a smaller subset of {0,...,n-1}.
    * If those splits are actually appearing, give the parameter n (the cardinality of the original superset), since
      only then the __hash__ function works properly for comparison of arbitrary splits.

    Parameters
    ----------
    part1, part2 :: can be tuples, lists or sets that are subsets of {0,...,n-1} and disjoint.
    n :: the number of leaves in the tree or forest.
    """
    def __init__(self, n, part1, part2):
        # sort both parts and convert them into tuples
        part1, part2 = tuple(np.sort(list(part1))), tuple(np.sort(list(part2)))
        if set(part1) & set(part2):
            raise ValueError(f"A split consists of disjoint sets, those are not: {part1}, {part2}.")
        # the next if-clauses make sure that we store the parts in a unique fixed way
        if part1 and part2:
            self._part1 = part1 if part1[0] < part2[0] else part2
            self._part2 = part2 if part1[0] < part2[0] else part1
        elif not part1:
            self._part1 = part2
            self._part2 = tuple()
        elif not part2:
            self._part1 = part1
            self._part2 = tuple()
        else:
            self._part1 = tuple()
            self._part2 = tuple()
        self._n = n  # I decided that this must be given by the user for total clarity

    @property
    def part1(self):
        """ This is the attribute part1 (basically a getter). This can be accessed by someone, but not changed. """
        return self._part1

    @property
    def part2(self):
        """ This is the attribute part2 (basically a getter). This can be accessed by someone, but not changed. """
        return self._part2

    @property
    def n(self):
        """ The number of leaves this split is seen in context with (might be a sub-split, say for a subtree)."""
        return self._n

    def restr(self, subset):
        """ The restriction of a split to a subtree determined by the labels in the subset. """
        return Split(n=self.n, part1=set(self.part1) & subset, part2=set(self.part2) & subset)

    def contains(self, subset):
        """ Returns True if subset is contained in either part1 or part2. subset needs to be a set."""
        return subset.issubset(set(self.part1)) or subset.issubset(set(self.part2))

    def separates(self, u, v):
        """ Returns True if u and v are not in the same part, else returns False. u and v are labels. """
        if type(u) == type(v) == int:
            return (u in self.part1 and v in self.part2) or (u in self.part2 and v in self.part1)
        else:
            return (set(u).issubset(set(self.part1)) and set(v).issubset(set(self.part2))) \
                   or (set(v).issubset(set(self.part1)) and set(u).issubset(set(self.part2)))

    def directed_to_split(self, other):
        """ Gives back the part of this split that is directed to the other split."""
        return self.part1 if set(self.part1) & set(other.part1) and set(self.part1) & set(other.part2) else self.part2

    def directed_away_from_split(self, other):
        """ The subtree is spanned by labels given in subtree, IMPORTANT: self is not part of that subtree."""
        return self.part2 if set(self.part1) & set(other.part1) and set(self.part1) & set(other.part2) else self.part1

    def compatible_with(self, other):
        """ Checks whether this split is compatible with another split. """
        p1, p2 = set(self.part1), set(self.part2)
        o1, o2 = set(other.part1), set(other.part2)
        return not np.all([p1 & o1, p1 & o2, p2 & o1, p2 & o2])  # only compatible if at least one intersection is empty

    def __eq__(self, other):
        """ The equal function. """
        # return set(self.part1) == set(other.part1) and set(self.part2) == set(self.part2)
        return self.__hash__() == other.__hash__()

    def __lt__(self, other):
        """ Less than function (<). """
        return self.__hash__() < other.__hash__()

    def __hash__(self):
        """ The hash function is defined with respect to self.n, so sub-splits are distinguishable from splits, too."""
        return hash((self._part1, self._part2))

    def __str__(self):
        """ String representation of the split. """
        return str((self.part1, self.part2))

    def __repr__(self):
        return f"{self.part1}|{self.part2}"

    def __bool__(self):
        """ Returns False only if both parts are empty sets. """
        return bool(self.part1) and bool(self.part2)


class Structure(object):
    """ Contains a forest structure. """
    def __init__(self, partition, split_collection, n):
        """ Initialize a structure with """
        self._n = n
        # sort the sets that are part of the partition, and sort the partition itself.
        partition = [tuple(sorted(x)) for x in partition]
        sort_key = np.argsort([part[0] for part in partition])
        self._partition = tuple([partition[key] for key in sort_key])
        assert set.union(*[set(x) for x in partition]) == set(range(n)), \
            f"The partition {partition} is not a partition of (0, ..., {n-1})."
        # sort the collections of splits according to the partition, sort the splits in each set of splits as well.
        self._split_collection = tuple([tuple(sorted(split_collection[key])) for key in sort_key])
        self._leaf_paths = None
        self._separators = None
        self._support = None
        return

    @property
    def n(self):
        """ The number of leaves this split is seen in context with (might be a sub-split, say for a subtree)."""
        return self._n

    @property
    def partition(self):
        """ The partition of the leaves into subsets, each of which is the leaf set of a component of a forest. """
        return self._partition

    @property
    def split_collection(self):
        """ The tuple containing the tuples of splits of the respective components of the forest, order as partition."""
        return self._split_collection

    @property
    def leaf_paths(self):
        """ Returns a list of dictionaries, where each dictionary contains for leaves u, v the splits on their path. """
        if self._leaf_paths is None:
            self._leaf_paths = [{(u, v): [k for k, s in enumerate(splits) if s.separates(u, v)]
                                 for u, v in it.combinations(self.partition[i], r=2)}
                                for i, splits in enumerate(self.split_collection)]
        return self._leaf_paths

    @property
    def support(self):
        if self._support is None:
            # for i-th component and k-th split in that component,
            # support[i][k][u, v] is True, if that split separates u and v.
            _support = [[np.zeros((self.n, self.n), dtype=bool) for _ in splits] for splits in self.split_collection]
            for i, d in enumerate(self.leaf_paths):
                for (u, v), split_list in d.items():
                    for k in split_list:
                        _support[i][k][u, v] = True
                        _support[i][k][v, u] = True
            self._support = self.unravel(_support)
        return self._support

    @property
    def sep(self):
        if self._separators is None:
            self._separators = [0] + list(np.cumsum([len(splits) for splits in self.split_collection], dtype=int))
        return self._separators

    def where(self, s):
        """ Gives the index (unraveled) of the split s in the structure. """
        return int(np.argmin([o != s for o in self.unravel(self.split_collection)]))

    def ravel(self, x):
        return [x[i:j] for i, j in zip(self.sep[:-1], self.sep[1:])]

    @staticmethod
    def unravel(x):
        return [y for z in x for y in z]

    def __str__(self):
        """ Prints the structure as a string representation. """
        return str(self.split_collection)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """ Determines if two structures are equal. """
        # if partitions are not equal, then structures are not equal.
        if self.partition != other.partition:
            return False
        # two structures must contain exactly the same splits to be equal.
        return self.split_collection == other.split_collection

    def __le__(self, other):
        """ Less than function (<). """
        xs = [set(x) for x in self.partition]
        ys = [set(y) for y in other.partition]
        # ----- check out condition 1. of the partial ordering -----
        try:
            x_to_y = {i: [j for j, y in enumerate(ys) if x.issubset(y)][0] for i, x in enumerate(xs)}
        except IndexError:  # in this case, x is not a refinement of y.
            return False
        # ----- check out condition 2. of the partial ordering -----
        try:
            for i, splits in enumerate(self.split_collection):
                # check if the splits are contained in the other restricted splits of the corresponding component.
                restr_other_splits = {sp_y.restr(subset=xs[i]) for sp_y in other.split_collection[x_to_y[i]]}
                assert set(splits).issubset(restr_other_splits)
        except AssertionError:
            return False
        # ----- check out condition 3. of the partial ordering -----
        try:
            for j, y in enumerate(ys):
                xs_in_y = [x for i, x in enumerate(xs) if x_to_y[i] == j]
                for x1, x2 in it.combinations(xs_in_y, r=2):
                    assert len([sp for sp in other.split_collection[j] if sp.separates(x1, x2)]) != 0
        except AssertionError:
            return False

        # ----- all conditions are satisfied -> it is indeed a partial ordering! -----
        return True

    def __gt__(self, other):
        """ Strictly greater than (>). """
        return other < self

    def __ge__(self, other):
        """ Greater than or equal function (>=). """
        return other <= self

    def __lt__(self, other):
        """ Less than or equal function (<=). """
        return not self == other and self <= other

    def __hash__(self):
        """ Computes the hash of the structure. """
        return hash(str(self))
