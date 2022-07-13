import numpy as np
import itertools as it
from Bio import Phylo
import matplotlib.pyplot as plt

from tools.structure_and_split import Split


def f00(number):
    if number < 10:
        return f"0{number}"
    else:
        return str(number)


def read_newick(file):
    return Phylo.parse(file, format='newick')


def plot_tree(tree, fn=None, **kwargs):
    fig = plt.figure()
    diag = fig.add_subplot(111)

    branch_length_min_label = kwargs.pop('min_label', 0.02)

    def branch_labels(c):
        if c.branch_length is None:
            return None
        elif c.branch_length > branch_length_min_label:
            return '%.4f' % c.branch_length
        else:
            return None

    tree.ladderize()
    Phylo.draw(tree, branch_labels=branch_labels, do_show=False, axes=diag)
    xlim_left, xlim_right = kwargs.pop('xlim', (-0.02, 0.5))

    plt.xlim(left=xlim_left)
    plt.xlim(right=xlim_right)
    plt.ylabel("labels")
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn, dpi=300)
    plt.close()
    return


def to_distance_matrix(tree, coding=None):
    """Create a distance matrix (NumPy array) from clades/branches in tree.

    coding is list of tuples of form (i, name), where name is the name of a leaf...

    A cell (i,j) in the array is the length of the branch between allclades[i]
    and allclades[j], if a branch exists, otherwise infinity.
    """
    parents = all_parents(tree)
    leaf_clades = lookup_by_names(tree)
    taxa = list(leaf_clades.keys())
    if coding is None:
        coding = {i: name for i, name in enumerate(taxa)}
    else:
        if not (set(coding.values()) == set(taxa)):
            raise ValueError("The taxa of the tree and the coding are not the same.")

    n = len(taxa)
    distmat = np.zeros((n, n))
    clades = {i: leaf_clades[leaf_name] for i, leaf_name in coding.items()}
    for i in range(n):
        for j in range(n):
            if i < j:
                distmat[i, j] += clades[i].branch_length + clades[j].branch_length
                parent_i = parents[clades[i]]
                parent_j = parents[clades[j]]
                while clades[i] not in parent_j.find_clades():
                    distmat[i, j] += parent_j.branch_length
                    parent_j = parents[parent_j]
                while clades[j] not in parent_i.find_clades():
                    distmat[i, j] += parent_i.branch_length
                    parent_i = parents[parent_i]
    distmat += distmat.T
    return distmat, coding


def prune_tree(tree, taxa):
    """ Prunes the tree such that only the given taxa are left. """
    pruned_taxa = {t.name for t in tree.get_terminals()} - set(taxa)
    for taxon in pruned_taxa:
        tree.prune(taxon)
    return tree


def all_parents(tree):
    parents = {}
    for clade in tree.find_clades(order="level"):
        for child in clade:
            parents[child] = clade
    return parents


def lookup_by_names(tree):
    names = {}
    for clade in tree.find_clades():
        if clade.name:
            if clade.name in names:
                raise ValueError("Duplicate key: %s" % clade.name)
            names[clade.name] = clade
    return names


def wald_graph_representation(wald):
    """ Graph representation consists of nodes and edges. """
    # TODO this only works if wald is a tree.
    # each node is uniquely determined by a key and its representation by Buneman (1971) is stored in this dictionary
    splits = wald.st.split_collection[0]
    # compute the node representation (Buneman's representation 1971) for the first two nodes incident to first split
    sp0 = splits[0]
    node_half = tuple([split.directed_to_split(sp0) for split in splits[1:]])
    nodes = ((sp0.part1,) + node_half, (sp0.part2,) + node_half,)

    # for each split, construct the node incident to the split that is further away from the very first split
    for i, split in enumerate(splits[1:]):
        node_half1 = tuple([sp.directed_to_split(split) for sp in splits[0:i + 1]])
        node_half2 = tuple([sp.directed_to_split(split) for sp in splits[i + 2:]])
        nodes += (node_half1 + (split.directed_away_from_split(sp0),) + node_half2,)

    edges = dict()
    # if two nodes coincide in all but one coordinate, they are linked by an edge
    for u, v in it.combinations(nodes, r=2):
        diff = list(set(u) ^ set(v))
        if len(diff) == 2:
            sp0 = Split(n=wald.n, part1=diff[0], part2=diff[1])
            edges[(u, v)] = np.abs(-np.log(1 - wald.x[np.argmin([sp != sp0 for sp in splits])]))
    return nodes, edges


def out_graph(wald, string=True, reduced=True):
    """ Returns a graph-theoretical representation, returns string if True, else: returns (nodes, edges)."""
    nodes, edges = wald_graph_representation(wald)

    interior_node_ascii_counter = 97
    nodes_names = dict()
    for node in nodes:
        intersection = list(set.intersection(*[set(x) for x in node]))
        if len(intersection) == 0:
            nodes_names[node] = chr(interior_node_ascii_counter)
            interior_node_ascii_counter += 1
        else:
            nodes_names[node] = str(intersection[0])
    if reduced:
        nodes = [nodes_names[node] for node in nodes]
        edges = {tuple([nodes_names[node] for node in edge]): weight for edge, weight in edges.items()}
    if not string:
        return nodes, edges, nodes_names
    else:
        if reduced:
            graph_string = ""
            for node in nodes:
                graph_string += f"{node}, "
            graph_string = graph_string[:-2] + "; "
            for edge, weight in edges.items():
                graph_string += f"{edge}: {np.round(weight, 6)}, "
            return graph_string[:-2]
        else:
            graph_string = ""
            changed_nodes = {node: tuple([tuple([x + 1 for x in c]) for c in node]) for node in nodes}
            for node in nodes:
                graph_string += f"{changed_nodes[node]}, "
            graph_string = graph_string[:-2] + "; "
            for edge, weight in edges.items():
                graph_string += f"{tuple([changed_nodes[node] for node in edge])}: {np.round(weight, 6)}, "
            return graph_string[:-2]


def out_newick(wald, labels=None, round_decimals=5):
    n = wald.n
    if labels is None:
        labels = {i: str(i) for i in range(n)}
    nodes, edges, nodes_names = out_graph(wald, string=False, reduced=True)
    # we basically hang up the tree at the interior vertex attached to the leaf '0'.
    root = [[v for v in edge if v != '0'][0] for edge in edges if '0' in edge][0]
    return f"{_newick_recursive(root=root, edges=edges, labels=labels, round_decimals=round_decimals)};"


def _newick_recursive(root, edges, labels, round_decimals):
    incident = [edge for edge in edges.keys() if root in edge]
    if len(incident) == 0:
        return f"{labels[int(root)]}"
    else:
        _roots = [[x for x in edge if x is not root][0] for edge in incident]
        _edges = [edges.copy() for _ in incident]
        _weights = [_edges[i].pop(edge) for i, edge in enumerate(incident)]
        if round_decimals != 0:
            _weights = [np.round(x, decimals=round_decimals) for x in _weights]
        _strings = [f"{_newick_recursive(_roots[i], _edges[i], labels, round_decimals)}:{weight}"
                    for i, weight in enumerate(_weights)]
        return f"({', '.join(_strings)})"
