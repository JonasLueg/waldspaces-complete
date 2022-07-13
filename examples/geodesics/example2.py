""" Reconstruction of the distances to the star strata that we have seen in our paper. """

import os
import sys
import pickle
import time
import ntpath
import numpy as np
import matplotlib.pyplot as plt

# DANGER ALERT: this works only, when this file is in the folder "/examples/distance_to_star_stratum/star_stratum1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tools.wald import Wald
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import TreeSpdAf1
import treespaces.framework.tools_numerical as numtools

# ----- CONSTRUCTION OF THE STARTING POINT ----- #
n = 5
name_handle = ntpath.splitext(ntpath.basename(__file__))[0]

partition = ((0, 1, 2, 3, 4),)
split_collection = [[Split(n=n, part1=(1, 2, 3, 4), part2=(0,)),
                     Split(n=n, part1=(0, 2, 3, 4), part2=(1,)),
                     Split(n=n, part1=(0, 1, 3, 4), part2=(2,)),
                     Split(n=n, part1=(0, 1, 2, 4), part2=(3,)),
                     Split(n=n, part1=(0, 1, 2, 3), part2=(4,)),
                     Split(n=n, part1=(2, 3, 4), part2=(0, 1)),
                     Split(n=n, part1=(3, 4), part2=(0, 1, 2))]]
st = Structure(partition=partition, split_collection=split_collection, n=n)
print(f"The structure of the forest has one component and contains the splits\n{st.split_collection[0]}.")
# (0, 2, 3, 4)|(1,), (0, 1, 3, 4)|(2,), (0, 1, 2, 4)|(3,), (0, 1, 2, 3)|(4,),
# (0, 1, 2)|(3, 4), (0, 1)|(2, 3, 4), (0,)|(1, 2, 3, 4)

coordinates = [[0.5, 0.5, 0.5, 0.5, 0.8, 0.1, 0.5], [0.5, 0.5, 0.5, 0.5, 0.1, 0.9, 0.5]]

walds = [Wald(n=n, st=st, x=np.array(_x)) for _x in coordinates]

# ----- CONSTRUCTION OF THE WALD SPACE WITH EMBEDDING GEOMETRY ----- #
ws = TreeSpdAf1(geometry='wald')


# ----- DEFINITION OF THE ENTROPY OF A GEODESICS DISTRIBUTION OF POINTS ALONG THE PATH ----- #

def entropy(_path, _ideal_value):
    """ Measures how equidistant the points in _path are. Zero means they are perfectly equidistant. """
    dists = [ws.g.a_dist(p=_path[_i], q=_path[_i + 1], squared=False) for _i in range(0, len(_path) - 1)]
    return np.sum([x * np.log(x / _ideal_value) for x in dists])


def length_of_paths(_calculate):
    max_n = 10
    n_points_on_path = np.arange(start=4, stop=max_n, step=3, dtype=int)
    n_iter = 5
    params = {}

    # these are the parameters used for every projection from ambient space onto the grove
    proj_args = {'gtol': 10 ** -12, 'ftol': 10 ** -15, 'btol': 10 ** -5, 'method': 'global-search'}
    # the parameters for each method are set here:
    params['naive'] = {'proj_args': proj_args}
    params['symmetric'] = {'proj_args': proj_args}
    params['straightening-ext'] = {'n_iter': n_iter, 'proj_args': proj_args}
    params['straightening-int'] = {'n_iter': n_iter, 'max_step': 10 ** -1}

    # just take all methods that have their parameters defined in the dictionary params
    methods = params.keys()  # ['naive', 'symmetric', 'variational', 'straightening-1']
    names = {'naive': 'NP', 'symmetric': 'SSP', 'straightening-ext': 'EPS', 'straightening-int': 'IPS'}

    # if we want to calculate new, do it now
    paths = {}
    if _calculate:
        for method in methods:
            print(f"\nCalculate geodesics with the {method} method.")
            start = time.time()
            paths[method] = []
            print("n = ", end='', flush=True)
            for m in n_points_on_path:
                print(f"{m}{', ' if m < n_points_on_path[-1] else '.'}", end='', flush=True)
                params[method]['n_points'] = m
                paths[method].append(tuple(ws.g.s_path(p=walds[0], q=walds[1], alg=method, **params[method])))
            print("", flush=True)
            paths[method] = tuple(paths[method])
            end = time.time()
            print(f"Calculation took {np.round(end - start, 2)} seconds.")
            print(f"Store the data that has been calculated.")
            pickle.dump(obj=paths[method],
                        file=open(os.path.join(f"{name_handle}_data",
                                               f"{name_handle}_data_{max_n}points_{method}.p"), "wb"))
    else:
        print(f"Load the already stored data since the flag 'calculate' is False.")
        for method in methods:
            paths[method] = pickle.load(
                file=open(os.path.join(f"{name_handle}_data",
                                       f"{name_handle}_data_{max_n}points_{method}.p"), "rb"))

    colors = ['red', 'blue', 'green', 'orange', 'black']
    colors = {method: colors[i] for i, method in enumerate(methods)}

    sp0 = 5
    sp1 = 4

    fig, ax = plt.subplots()
    for method in methods:
        _path = paths[method][-1]
        ax.scatter([wald.x[sp0] for wald in _path], [wald.x[sp1] for wald in _path],
                   label=names[method], linewidth=1, color=colors[method])

    ax.set_xlabel(r"$\lambda_6$")
    ax.set_ylabel(r"$\lambda_7$")
    plt.scatter(walds[0].x[sp0], walds[0].x[sp1], color='black')
    plt.scatter(walds[1].x[sp0], walds[1].x[sp1], color='black')
    ax.text(walds[0].x[sp0], walds[0].x[sp1], r"$W_1$", color='black')
    ax.text(walds[1].x[sp0], walds[1].x[sp1], r"$W_2$", color='black')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    ax.set_aspect('equal', adjustable='box')
    fig.savefig(os.path.join(f"{name_handle}_png", f"{name_handle}_paths_in_2d.png"), dpi=200)

    # plot the lengths of the paths
    fig = plt.figure()

    lengths = {method: [ws.g.length(path_=_path) for _path in paths[method]] for method in methods}
    for method in methods:
        plt.scatter(n_points_on_path, lengths[method], color=colors[method], label=names[method])

    plt.legend()
    plt.title("length")
    plt.savefig(os.path.join(f"{name_handle}_png", f"{name_handle}_lengths_given_points.png"), dpi=200)
    plt.clf()
    plt.cla()
    plt.close(fig)

    # plot the entropy of the paths
    fig = plt.figure()

    ideal_values = [ws.g.length(_path) / (n_points_on_path[_i] - 1)
                    for _i, _path in enumerate(paths['straightening-ext'])]
    entropies = {method: [entropy(_path=_path, _ideal_value=ideal_values[_i]) for _i, _path in enumerate(paths[method])]
                 for method in methods}
    for method in methods:
        if method == 'symmetric':
            continue
        plt.scatter(n_points_on_path, entropies[method], color=colors[method], label=names[method])

    plt.legend()
    plt.title(f"entropy")
    plt.savefig(os.path.join(f"{name_handle}_png", f"{name_handle}_entropy_given_points.png"), dpi=200)
    plt.clf()
    plt.cla()
    plt.close(fig)

    # plot the energy of the paths
    fig = plt.figure()

    energies = {method: [numtools.energy(_path=_path, geometry=ws) for _i, _path in enumerate(paths[method])]
                for method in methods}
    for method in methods:
        plt.scatter(n_points_on_path, energies[method], color=colors[method], label=names[method])

    plt.legend()
    plt.title(f"energy")
    plt.savefig(os.path.join(f"{name_handle}_png", f"{name_handle}_energy_given_points.png"), dpi=200)
    plt.clf()
    plt.cla()
    plt.close(fig)


if __name__ == "__main__":
    calculate = True
    length_of_paths(_calculate=calculate)
