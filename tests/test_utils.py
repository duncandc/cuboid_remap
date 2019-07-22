"""
"""
import numpy as np
from cuboid_remap.utils import triple_scalar_product


__all__ = ('test_triple_scalar_product', )


def test_triple_scalar_product():
    """
    """

    u = np.array([7, 3, 2])
    v = np.array([1, 1, 1])
    z = np.array([3, 5, 7])

    a = triple_scalar_product(u,v,z)
    b = np.linalg.det(np.dstack([u,v,z]))[0]

    assert np.allclose(a, b)


