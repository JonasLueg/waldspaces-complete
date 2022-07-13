import os
import sys
import numpy as np

import treespaces.tools.tools_forest_representation as ftools

for n in range(2, 10):
    splits = list(ftools.make_splits(n))
    print(f"For n = {n}, number of splits should be {2**(n-1) - 1}, and are {len(splits)}.")


for n in range(2, 8):
    structures = list(ftools.make_structures(n))
    if n == 2:
        n_true = 1
    else:
        n_true = int(np.prod(np.arange(1, 2*n - 5 + 1)[::2]))
    if n < 5:
        print([str(st) for st in structures])
    print(f"For n = {n}, number of fully resolved trees should be {n_true}, and is {len(structures)}.")

