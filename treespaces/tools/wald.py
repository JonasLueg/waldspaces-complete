""" Contains the Wald class, which is the universal class for characterizing trees and forests, i.e. walds. """

import numpy as np

import treespaces.tools.tools_forest_representation as ftools
from treespaces.tools.structure_and_split import Structure


class Wald(object):
    """ Contains a wald. These are the points contained in wald space. """

    def __init__(self, n, **kwargs):
        """ The constructor for walds. There are several possibilities on how to construct a wald. """
        self._n: int = n
        self._dist: [np.ndarray, None] = None
        self._corr: [np.ndarray, None] = None
        self._st: [Structure, None] = None
        self._x: [np.ndarray, None] = None
        self._b: [np.ndarray, None] = None

        # start constructing stuff and pull the arguments out of kwargs.
        if 'st' in kwargs and 'x' in kwargs:
            self._st = kwargs['st']
            self._x = np.array(kwargs['x'])
        elif 'corr' in kwargs and 'st' in kwargs:
            self._st = kwargs['st']
            self._corr = kwargs['corr']
            np.fill_diagonal(a=self._corr, val=1)
        elif 'corr' in kwargs:
            self._corr = kwargs['corr']
            np.fill_diagonal(a=self._corr, val=1)
        elif 'dist' in kwargs and 'st' in kwargs:
            self._st = kwargs['st']
            self._dist = kwargs['dist']
            np.fill_diagonal(a=self._dist, val=0)
        elif 'dist' in kwargs:
            self._dist = kwargs['dist']
            np.fill_diagonal(a=self._dist, val=0)
        elif 'st' in kwargs and 'b' in kwargs:
            self._st = kwargs['st']
            self._b = np.array(kwargs['b'])
        else:
            raise RuntimeError("The arguments passed to the Wald constructor are not sufficient.")
        # TODO: finish the class Wald according to the needs of the algorithms.
        return

    @property
    def st(self) -> Structure:
        """ The structure of the wald. Is of class Structure. """
        if self._st is not None:
            pass
        elif self._dist is not None or self._corr is not None:
            self._st = ftools.compute_structure_from_dist(dist=self.dist, btol=10 ** -10)
        else:  # we should never reach this case due to design of the constructor, but you never know.
            raise AttributeError("Cannot compute the structure from the given information of the Wald.")
        return self._st

    @property
    def x(self) -> np.ndarray:
        """ The (flat) vector containing the edge weights in Nye notation. """
        if self._x is not None:
            return self._x
        elif self._corr is not None or self._dist is not None:  # self._st is constructed if necessary (we call self.st)
            _ells = np.array([ftools.compute_length_of_split_from_dist(sp, dist=self.dist)
                              for sp in self.st.unravel(self.st.split_collection)])
            self._x = np.maximum(0, np.minimum(1, 1 - np.exp(-_ells)))
        elif self._b is not None:
            self._x = np.maximum(0, np.minimum(1, 1 - np.exp(-self.b)))
        return self._x

    @property
    def b(self) -> np.ndarray:
        """ The (flat) vector containing the edge lengths in Bhv notation. """
        if self._b is not None:
            return self._b
        else:
            return -np.log(1 - self._x)

    @property
    def n(self):
        """ The number of labels in the wald. """
        return self._n

    @property
    def corr(self):
        if self._corr is not None:
            pass
        elif self._corr is None and self.dist is not None:
            self._corr = np.maximum(0, np.minimum(1, np.exp(-self.dist)))  # avoid ambiguities and restrict on [0,1]
            np.fill_diagonal(a=self._corr, val=1)
        elif self._corr is None and (self.st is not None and self.x is not None):
            self._corr = ftools.compute_chart(st=self.st)(self.x)
        else:  # we should never reach this case due to design of the constructor, but you never know.
            raise AttributeError("Cannot compute corr from the given information of the Wald.")
        return self._corr

    @property
    def dist(self):
        if self._dist is not None:
            return self._dist
        elif self._dist is None and self._corr is not None:
            with np.errstate(divide='ignore'):  # ignore warning 'divide by zero', this is intended to be fine, -np.inf.
                self._dist = np.maximum(0, -np.log(self.corr))
        elif self._dist is None and (self.st is not None and self.x is not None):
            self._dist = ftools.compute_dist_from_structure_and_coordinates(st=self.st, x=self.x)
        elif self._dist is None and (self.st is not None and self.b is not None):
            self._dist = ftools.compute_dist_from_structure_and_coordinates(st=self.st, x=self.x)
        else:  # we should never reach this case due to design of the constructor, but you never know.
            raise AttributeError("Cannot compute dist from the given information of the Wald.")
        return self._dist

    def __hash__(self):
        return hash((self.st, tuple(self.x)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return str(self.x)


class Tree(Wald):
    def __init__(self, n, **kwargs):
        if 'splits' in kwargs:
            # let Structure handle the sorting of the splits
            kwargs['st'] = Structure(n=n, partition=(tuple(i for i in range(n)),), split_collection=(kwargs['splits'],))
        super().__init__(n=n, **kwargs)
        self._splits = None

    @property
    def splits(self):
        if self._splits is None:
            self._splits = self.st.split_collection[0]
        return self._splits
