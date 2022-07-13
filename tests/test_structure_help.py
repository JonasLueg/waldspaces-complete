def make_splits(collection):
    """ Generates all possible splits of a collection."""
    if len(collection) <= 1:
        raise ValueError("The collection must have 2 elements or more.")
    if len(collection) == 2:
        yield ((collection[0],), (collection[1],))
    else:
        last = collection[-1]
        for split in make_splits(collection[:-1]):
            yield (split[0], split[1] + (last,))
            yield (split[0] + (last,), split[1])
        yield (tuple(collection[:-1]), (last,))


def make_split_sets(collection):
    """ Generates all possible sets of compatible splits of a collection.

    This only works well for len(collection) < 8 (time-constraint).
    Importantly, all split sets with the maximum number of splits are generated.
    """
    if len(collection) <= 1:
        raise ValueError("The collection must have 2 elements or more.")
    if len(collection) in [2, 3]:
        yield tuple(make_splits(collection))
    else:
        last = collection[-1]
        pendant_split = ((last,), tuple(collection[:-1]))
        for split_set in make_split_sets(collection[:-1]):
            for s in split_set:
                new_split_set = (pendant_split,)
                a, b = set(s[0]), set(s[1])
                for t in split_set:
                    c, d = set(t[0]), set(t[1])
                    if t != s:
                        if a.issubset(d) or b.issubset(d):
                            new_split_set += ((t[0], t[1] + (last,),),)
                        else:
                            new_split_set += ((t[1], t[0] + (last,),),)
                    else:
                        new_split_set += ((s[0], s[1] + (last,),),)
                        new_split_set += ((s[1], s[0] + (last,),),)
                yield new_split_set


def partition(collection):
    if len(collection) == 1:
        yield (tuple(collection),)
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            print(smaller)
            _dummy = smaller[:n] + ((first,) + subset,) + smaller[n+1:]
            yield tuple(sorted(_dummy))
        # put `first` in its own subset
        _dummy = ((first,),) + smaller
        yield tuple(sorted(_dummy))


def generate_splits_subsets(splits):
    if len(splits) == 0:
        yield tuple()
        return

    if len(splits) == 1:
        yield tuple(splits)
        yield tuple()
        return

    # recursion principle is missing.
    pass


if __name__ == "__main__":
    res = make_split_sets([0, 1, 2, 3])
    print(list(res))
