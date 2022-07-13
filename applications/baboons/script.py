import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time
from Bio import Phylo

from treespaces.tools.wald import Wald
from treespaces.framework.waldspace import WaldSpace
import treespaces.framework.tools_io as io_tools

file = os.path.join('data', 'autosomal-500k-iqtree.treefile')


# try to load the data from somewhere else first, that should be faster.
fn = os.path.join('data', 'two_trees.p')
try:
    walds_labels = pickle.load(file=open(fn, "rb"))
except FileNotFoundError:
    # store all trees in pickle so we can load them faster
    generator = io_tools.read_newick(file=file)
    i = 0
    trees = []
    while i < 2:
        trees.append(next(generator))
        i += 1

    for i, tree in enumerate(trees):
        io_tools.plot_tree(tree,
                           fn=os.path.join('graphics', f"tree_{100 + i}_test.pdf"),
                           xlim=(-0.0005, 0.011), min_label=0.001)
    coding = {i: leaf.name for i, leaf in enumerate(trees[0].get_terminals())}
    n = len(coding)
    dists_labels = [io_tools.to_distance_matrix(tree=tree, coding=coding) for tree in trees]
    walds_labels = [(Wald(n=n, dist=dist), taxa) for dist, taxa in dists_labels]
    pickle.dump(obj=tuple(walds_labels), file=open(fn, "wb"))

plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'lines.linewidth': 0.7})

# for i, (wald, taxa) in enumerate(walds_labels):
#     print("print a wald")
#     for sp, weight in zip(wald.st.unravel(wald.st.split_collection), wald.x):
#         print(f"{sp}: {weight}.")

time0 = time.time()
plot_tree = True
coding = walds_labels[0][1]
trees = []
for wald, taxa in walds_labels:
    newick_string = io_tools.out_newick(wald, labels=taxa, round_decimals=15)
    handle = io.StringIO(newick_string)
    tree = Phylo.read(handle, "newick")
    trees.append(tree)

for i, tree in enumerate(trees):
    tree.ladderize()
    if plot_tree:
        io_tools.plot_tree(tree,
                           fn=os.path.join('graphics', f"tree_{100 + i}_conv.pdf"),
                           xlim=(-0.0005, 0.011), min_label=0.001)
    tree.rooted = True
    clade = [leaf for leaf in tree.get_terminals() if leaf.name == "PD_0067"][0]
    tree.root_with_outgroup(clade)
    tree.ladderize()
    if plot_tree:
        io_tools.plot_tree(tree,
                           fn=os.path.join('graphics',
                                           f"tree_rooted_{100 + i}_conv.pdf"),
                           xlim=(-0.0005, 0.017), min_label=0.001)
time1 = time.time()
print(f"Plotting time: {np.round((time1 - time0), 5)} seconds.")

walds = [wald for wald, taxa in walds_labels]
ws = WaldSpace(geometry='bhv')

n_points = 33
# j_iter = 2
fn = os.path.join('data', f'geodesic_bhv_{n_points}.p')
try:
    geodesic = pickle.load(file=open(fn, "rb"))
    raise FileNotFoundError
except FileNotFoundError:
    time0 = time.time()
    geodesic = ws.g.s_path(p=walds[0], q=walds[1], n_points=n_points,
                           alg='global-symmetric')
    time1 = time.time()
    print(f"Time for calculation of geodesic: {np.round((time1 - time0) / 60, 5)} "
          f"minutes.")
    pickle.dump(obj=tuple(geodesic), file=open(fn, "wb"))

print(len(geodesic))
print("Print the trees on the geodesic.")
trees = []
for wald in geodesic:
    newick_string = io_tools.out_newick(wald, labels=coding, round_decimals=15)
    handle = io.StringIO(newick_string)
    tree = Phylo.read(handle, "newick")
    trees.append(tree)

for i, tree in enumerate(trees):
    tree.rooted = True
    clade = [leaf for leaf in tree.get_terminals() if leaf.name == "PD_0067"][0]
    tree.root_with_outgroup(clade)
    tree.ladderize()
    io_tools.plot_tree(tree,
                       fn=os.path.join('graphics', f"path_{n_points}_bhv_tree_rooted"
                                                   f"_{100 + i}.png"),
                       xlim=(-0.0005, 0.017), min_label=0.001)

print("Finished")

# for wald in geodesic:
#     print("print a wald")
#     for sp, weight in zip(wald.st.unravel(wald.st.split_collection), wald.x):
#         print(f"{sp}: {weight}.")
# ws = TreeSpdAf1(geometry='waldspace')
# # wald_list = list()
# # for corr in corr_list:
# #     wald = ws.g.w_proj(p=corr, st=st1, btol=10**-10, method='global-search')
# #     wald_list.append(wald)
# #
# # fn = os.path.join("4_136_237", f"trees.p")
# # pickle.dump((tuple(wald_list), tuple([(i, val) for i, val in taxa.items()])), file=open(fn, "wb"))
#
# fn = os.path.join("4_136_237", f"trees.p")
# wald_list, taxa = pickle.load(file=open(fn, "rb"))
# taxa = {i: taxon for i, taxon in taxa}
#
# for wald in wald_list:
#     print(wald.st)
#     print([-np.log(1 - _x) for _x in wald.x])
# # for i, wald in enumerate(wald_list):
# #     print("new wald:")
# #     str_wald = str(wald.st)
# #     for j, taxon in taxa.items():
# #         str_wald = str_wald.replace(str(j), taxon)
# #     print(str_wald)
# #     print(wald.st)
# #     print([-np.log(1 - _x) for _x in wald.x])
#
# # start with path between 4 and 136
# n_points = 25
# n_iter = 10
# proj_args = {'gtol': 10 ** -10, 'ftol': 10 ** -15, 'btol': 10 ** -10, 'method': 'global-search'}
#
# iterations = 15
# # find out until what iteration we have already computed:
# cur_iteration = 0
# try:
#     for i in range(1, iterations):
#         cur_iteration = i
#         fn = os.path.join("4_136_237", f"path{i}.p")
#         p = pickle.load(file=open(fn, "rb"))
# except FileNotFoundError:
#     pass
#
# fn = os.path.join("4_136_237", f"path{cur_iteration - 1}.p")
# cur_path = pickle.load(file=open(fn, "rb"))
#
# calculate = False
#
# if calculate:
#     for i in range(cur_iteration, iterations):
#         print(f"Frechet Mean Iteration: {i}.")
#         p = cur_path[int((n_points - 1) / i)]
#         q = wald_list[i % 3]
#         start_path = ws.g.s_path(p=p, q=q, n_points=n_points, alg='global-symmetric', proj_args=proj_args)
#         cur_path = ws.g.s_path(p=p, q=q, n_points=n_points, alg='global-straightening',
#                                start_path=start_path, proj_args=proj_args, n_iter=10)
#         fn = os.path.join("4_136_237", f"path{i}.p")
#         pickle.dump(tuple(cur_path), file=open(fn, "wb"))
#
# # for i in range(1, iterations):
# #     fn = os.path.join("4_136_237", f"path{i}.p")
# #     cur_path = pickle.load(file=open(fn, "rb"))
# #     wald = cur_path[0]
# #     print(f"Iteration {i}.")
# #     print(wald.st)
# #     print(wald.x)
#
#
# # means = list()
# # for i in range(1, 18):
# #     fn = os.path.join("4_136_237", f"path{i}.p")
# #     print(int((n_points - 1) / i))
# #     means.append(pickle.load(file=open(fn, "rb"))[int((n_points - 1) / i)])
# #
# # for mean in means:
# #     print(mean.x)
#
# # print([ws.g.a_dist(p=means[_i], q=means[_i + 1], squared=False) for _i in range(len(means) - 1)])
# # print(ws.g.length(ws.g.w_path(p=means[-2], q=means[-1], alg='global-symmetric', n_points=15, proj_args=proj_args)))
#
# fn = os.path.join("4_136_237", f"path{iterations - 1}.p")
# cur_path = pickle.load(file=open(fn, "rb"))
# wald = cur_path[int((n_points - 1) / iterations)]
# # tol_path = ws.g.w_path(p=cur_path[0], q=wald, alg='global-symmetric', n_points=25, proj_args=proj_args)
# # tol = ws.g.length(tol_path)
# # print(tol)
#
# print(wald.st)
# print(wald.x)
# nodes, edges = io_tools.wald_graph_representation(wald=wald)
# # graph = io.out_graph(wald=wald, string=False, reduced=True)
# graph = io_tools.out_newick(wald=wald, labels=taxa, round_decimals=8)
# tree = Phylo.read(io.StringIO(graph), format='newick')
#
# tree.rooted = True
# # find clade with leaf "Pf":
# clade = [leaf for leaf in tree.get_terminals() if leaf.name == "Pf"][0]
# tree.root_with_outgroup(clade)
#
# io_tools.plot_tree(tree, fn=os.path.join("4_136_237", f"frechet_mean_waldspace1.png"))
