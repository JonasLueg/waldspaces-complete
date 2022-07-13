import numpy as np
import scipy.linalg as la


class SpdAf1(object):
    @classmethod
    def dist(cls, p, q, squared=False):
        """ Computes the distance between p and q. """
        p_inv_sqrt = cls.ssqrtm(la.inv(p))
        eigenvalues = la.eigvalsh(np.dot(p_inv_sqrt, np.dot(q, p_inv_sqrt)))
        sq_distance = np.sum(np.log(eigenvalues) ** 2)
        return sq_distance if squared else np.sqrt(sq_distance)

    @classmethod
    def path_t(cls, p, q, t):
        """ Computes the geodesic between p and q at portion t (where t is between 0 and 1). """
        root_p = cls.ssqrtm(p)
        inv_root_p = la.inv(root_p)
        direction = la.logm(inv_root_p.dot(q).dot(inv_root_p))
        return np.dot(root_p, la.expm(t * direction).dot(root_p))

    @classmethod
    def path(cls, p, q, **kwargs):
        """ Computes the geodesic with n_points points between p and q. """
        if 'times' in kwargs:
            times = kwargs['times']
        else:
            n_points = kwargs['n_points'] if 'n_points' in kwargs else 20
            times = np.linspace(0, 1, n_points)
        root_p = cls.ssqrtm(p)
        inv_root_p = la.inv(root_p)
        direction = la.logm(inv_root_p.dot(q).dot(inv_root_p))
        _path = [np.dot(root_p, la.expm(t * direction).dot(root_p)) for t in times]
        return _path

    @classmethod
    def inner(cls, u, v, p):
        """ Computes the riemannian metric of vectors u and v at point p. """
        # TODO: make all methods such that **kwargs is possible and we can feed the inverse of p already. do this
        #       package wide.
        inv_p = la.inv(p)
        return np.trace(np.dot(u, inv_p).dot(v).dot(inv_p))

    @classmethod
    def exp(cls, v, p):
        """ Computes the riemannian exponential of vector v at point p. """
        root_p = cls.ssqrtm(p)
        inv_root_p = la.inv(root_p)
        dummy = la.expm(np.dot(inv_root_p, v).dot(inv_root_p))
        return np.dot(root_p, dummy).dot(root_p)

    @classmethod
    def log(cls, q, p):
        """ Computes the riemannian logarithm of point q base p (outcome is a vector at point p). """
        root_p = cls.ssqrtm(p)
        inv_root_p = la.inv(root_p)
        dummy = la.logm(np.dot(inv_root_p, q).dot(inv_root_p))
        return np.dot(root_p, dummy).dot(root_p)

    @classmethod
    def trans(cls, v, p, q):
        """ Computes the parallel transport of the vector v from point p to point q. """
        e = cls.ssqrtm(q.dot(la.inv(p)))
        return e.dot(v).dot(e.T)

    @staticmethod
    def ssqrtm(p):
        """ Safe matrix square root. Does not produce machine precision complex numbers like the scipy routine. """
        # this one avoids complex numbers by assuming that the matrix is hermitian.
        # s, u = la.eigh(a)
        # return u.dot(np.sqrt(np.diag(s))).dot(u.T)
        # this one is just plain and simple taking the real part, should basically result in the same matrix.
        sqrt_p = np.real(la.sqrtm(p))
        # sanity check: if something goes wrong with just taking the real part, I will know.
        # note that just taking the real part without sanity check is very dangerous as you might never know there is an
        # actual error happening and the square root was taken somehow in a wrong way!

        if not np.allclose(p, np.dot(sqrt_p, sqrt_p), atol=10**-5):
            print(np.max(np.abs(sqrt_p @ sqrt_p - p)))
            raise ValueError(f"Something went wrong when computing the square root of\n{p}.")
        return sqrt_p


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    p = np.array([[1, 0.5], [0.5, 1]])
    q = np.array([[1, 0.8], [0.8, 1]])
    midpoint = SpdAf1.path_t(p=p, q=q, t=0.5)
    print(f"The midpoint of p = \n{p} and q = \n{q} is m = \n{midpoint}.")
    print(f"The distance from p to q is {SpdAf1.dist(p=p, q=q, squared=False)}.")
    v = SpdAf1.log(q=q, p=p)
    print(f"The direction that one must go into if one wants to move from p to q is v = \n{v}.")
    print(f"The length of this vector is {np.sqrt(SpdAf1.inner(u=v, v=v, p=p))}.")
    print(f"Why not move into the direction of v starting at p? Let's try!")
    r = SpdAf1.exp(v=v, p=p)
    print(f"Sanity check: Is q = r satisfied? Answer: {'Yes' if np.allclose(q, r) else 'No'}.")
    print(f"Sanity check whether we move if we feed the Riemannian exponential the zero direction:")
    w = np.zeros(p.shape)
    print(f"Is p = Exp_p(0)? Answer: {'Yes' if np.allclose(p, SpdAf1.exp(v=w, p=p)) else 'No'}.")
    print(f"Certainly a surprise. :-).")