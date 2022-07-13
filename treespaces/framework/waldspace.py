""" This class enables the construction of different spaces with different spaces. """

# internal imports
from treespaces.spaces.treespace_spd_af1 import TreeSpdAf1
from treespaces.spaces.treespace_spd_euclidean import TreeSpdEuclidean
from treespaces.spaces.treespace_bhv import TreeSpaceBhv
from treespaces.spaces.treespace_corr_quotient import TreeSpaceCorrQuotient

# global variables
LIST_OF_GEOMETRIES = [None, TreeSpdAf1, TreeSpaceBhv, TreeSpdEuclidean, TreeSpaceCorrQuotient]


class WaldSpace(object):
    def __init__(self, geometry: str):
        self.g: LIST_OF_GEOMETRIES = None
        if geometry == 'bhv':
            self.g = TreeSpaceBhv
        elif geometry in ['wald', 'waldspace', 'killing', 'affine-invariant', 'scaled-frobenius']:
            self.g = TreeSpdAf1
        elif geometry in ['euclidean']:
            self.g = TreeSpdEuclidean
        elif geometry in ['correlation', 'quotient', 'correlation-quotient']:
            self.g = TreeSpaceCorrQuotient
        else:
            raise ValueError("This geometry does not exist.")
