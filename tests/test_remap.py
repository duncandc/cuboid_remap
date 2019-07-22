"""
"""

from __future__ import print_function
import numpy as np
from cuboid_remap.remap import remap

Npts = 10**2
test_coords = np.random.random((Npts,3))

def test_1():
    
    u1 = [1,0,0]
    u2 = [0,1,0]
    u3 = [0,0,1]
    result = remap(test_coords, u1, u2, u3)
    
    assert np.shape(result)==(Npts,3), print(np.shape(result))


def test_2():

    u1=[2,1,0]
    u2=[0,0,1]
    u3=[1,0,0]
    result = remap(test_coords, u1, u2, u3)
    
    assert np.shape(result)==(Npts,3), print(np.shape(result))


