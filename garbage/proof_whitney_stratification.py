import os
import sys
import time
import ntpath
import pickle
import numpy as np
import matplotlib.pyplot as plt

# DANGER ALERT: this works only, when this file is in the folder "/examples/cobwebs/cobweb1.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import treespaces.framework.tools_numerical as numtools
from tools.wald import Wald, Tree
from tools.structure_and_split import Split, Structure
from treespaces.framework.waldspace import WaldSpace
import treespaces.tools.tools_forest_representation as ftools


_dir_name = ntpath.splitext(ntpath.basename(__file__))[0]

n = 5
ws = WaldSpace(geometry='killing')
partition = ((0, 1, 2, 3, 4),)

sp01 = Split(n=n, part1=(0, 1), part2=(2, 3, 4))
sp34 = Split(n=n, part1=(3, 4), part2=(0, 1, 2))
sp12 = Split(n=n, part1=(1, 2), part2=(0, 3, 4))
sp03 = Split(n=n, part1=(0, 3), part2=(1, 2, 4))

sp0 = Split(n=n, part1=(0,), part2=(1, 2, 3, 4))
sp1 = Split(n=n, part1=(1,), part2=(0, 2, 3, 4))
sp2 = Split(n=n, part1=(2,), part2=(0, 1, 3, 4))
sp3 = Split(n=n, part1=(3,), part2=(0, 1, 2, 4))
sp4 = Split(n=n, part1=(4,), part2=(0, 1, 2, 3))

splits1 = [[sp0, sp1, sp2, sp3, sp4, sp01, sp34]]
splits2 = [[sp0, sp1, sp2, sp3, sp4, sp12, sp03]]
splits3 = [[sp0, sp1, sp2, sp3, sp4, sp34, sp12]]

st1 = Structure(n=n, partition=partition, split_collection=splits1)
st2 = Structure(n=n, partition=partition, split_collection=splits2)
st3 = Structure(n=n, partition=partition, split_collection=splits3)

phi = ftools.compute_chart_gradient(st=st1, chart=ftools.compute_chart(st=st1))
print(st1)

print(phi(x=[0.5, 1, 1, 0.5, 0.5, 0.5, 0.5]))
