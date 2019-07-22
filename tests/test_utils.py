"""
"""
import numpy as np
from cuboid_remap.utils import vec3, dot, square, length, triple_scalar_product


__all__ = ('test_1', 'test_2', 'test_3')


def test_vec3():
    """
    """

    u = vec3()
    assert [u.x,u.y,u.z]==[0,0,0]
    assert [u[0],u[1],u[2]]==[0,0,0]


    u = vec3([0,0,0])
    assert [u.x,u.y,u.z]==[0,0,0]
    assert [u[0],u[1],u[2]]==[0,0,0]

    u = vec3([1,22,333])
    assert [u.x,u.y,u.z]==[1,22,333]
    assert [u[0],u[1],u[2]]==[1,22,333]


def test_dot():
    """
    """

    u = vec3(7, 3, 2)
    v = vec3(1, 1, 1)

    assert dot(u,v) == np.dot(u,v)


def test_square():
    """
    """

    u = vec3(7, 3, 2)
    v = vec3(1, 1, 1)

    assert square(u) == np.sum(u**2)
    assert square(v) == np.sum(v**2)


def test_length():
    """
    """

    u = vec3(7, 3, 2)
    v = vec3(1, 1, 1)

    assert length(u) == np.sqrt(np.sum(u**2))
    assert length(v) == np.sqrt(np.sum(v**2))


def test_triple_scalar_product():
    """
    """

    u = vec3(7, 3, 2)
    v = vec3(1, 1, 1)
    z = vec3(3, 5, 7)

    a = triple_scalar_product(u,v,z)
    b = np.linalg.det(np.dstack([u,v,z]))[0]

    assert np.allclose(a, b)


