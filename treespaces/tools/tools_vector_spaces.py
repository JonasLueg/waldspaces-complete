import numpy as np


def gram_schmidt(vectors, dot, p):
    """ Computes a basis from a given linearly independent collection of vectors wrt a dot product at point _p."""
    basis = [None] * len(vectors)  # basis of orthonormal elements
    v = vectors  # shortage of notation
    if len(v) == 0:
        return []
    basis[0] = v[0] / np.sqrt(dot(v[0], v[0], p))
    for i in range(1, len(v)):
        basis[i] = v[i] - np.sum([b * dot(b, v[i], p) for b in basis[:i]], axis=0)
        basis[i] = basis[i] / np.sqrt(dot(basis[i], basis[i], p))
    return basis


def vector_space_projection(v, basis, dot, p, lift, lift_vector):
    """ Projects a vector (or matrix) onto the subspace spanned by basis with respect to dot. """
    if len(basis) == 0:
        return np.zeros(p.x.shape)
    return np.sum([dot(v, lift_vector(b, p), lift(p)) * b for b in basis], axis=0)


def vector_basis_representation(v, basis, dot, p):
    """ Returns a representation of v in basis. """
    if len(basis) == 0:
        return np.zeros(v.shape)
    return np.array([dot(v, b, p) for b in basis])
