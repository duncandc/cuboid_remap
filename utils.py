"""
utility classes and functions
"""

import sys
from math import *
import numpy as np


class vec3(np.ndarray):
    """
    a simple 3D vector class
    """
    def __new__(cls, *args):
        """
        create a new vec3 object

        Parameters
        ----------
        v : array_like, optional
            length 3 array defining a 3-vector.
            if not passed, v is set to [0,0,0] 
        
        Notes
        -----
        instead of passing an array_like object, v, to intialize, 
        you may pass three floats.
        """
        a = np.ndarray.__new__(vec3, (3,), float)
        
        if len(args) == 0:
            a[0] = a[1] = a[2] = 0
        elif len(args) == 1:
            v = args[0]
            a[0] = v[0]
            a[1] = v[1]
            a[2] = v[2]
        elif len(args) == 3:
            print(args)
            a[0] = args[0]
            a[1] = args[1]
            a[2] = args[2]
        else:
            raise RuntimeError
        
        return a

    def _getx(self): 
        return self[0]

    def _gety(self): 
        return self[1]

    def _getz(self): 
        return self[2]

    def _setx(self, value):
        self[0] = value

    def _sety(self, value):
        self[1] = value

    def _setz(self, value):
        self[2] = value
    
    # define property attributes
    x = property(_getx, _setx)  # (fget, fset, fdel, doc_string)
    y = property(_gety, _sety)  # (fget, fset, fdel, doc_string)
    z = property(_getz, _setz)  # (fget, fset, fdel, doc_string)


def dot(u, v):
    """
    vector dot product
    """
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def square(v):
    """
    vector magnitude squared
    """
    return v[0]**2 + v[1]**2 + v[2]**2


def length(v):
    """
    vector magnitude
    """
    return sqrt(square(v))


def triple_scalar_product(u, v, w):
    """
    triple scalar product of three vectors
    """
    return u[0]*(v[1]*w[2] - v[2]*w[1]) +\
           u[1]*(v[2]*w[0] - v[0]*w[2]) +\
           u[2]*(v[0]*w[1] - v[1]*w[0])

def gcd(*args):
    """
    greatest common integer divisor

    parameters
    ----------
    a : int

    b : int

    ...

    n : int
    """
    if len(args)==1:
        return args[0]
    elif len(args)==2:
        a = args[0]
        b = args[1]

        if(a < 0):
            a = -a

        if(b < 0):
            b = -b

        while(b != 0):
            tmp = b
            b = a % b
            a = tmp
        return a
    else:
        # recursive call
        a = args[0]
        b = gcd(*args[1:])
        return gcd(a,b)


def coprime_triples(max_int, min_int=0, method='effecient'):
    """
    return all integer coprime triples within a range

    Parameters
    ----------
    max_int : int
        maximum integer in the range.

    min_int : int, optional
        minimum integer in the range.
        default is 0

    Returns 
    -------
    d : dictionary
        A dictionary of coprime triples.  The keys of the dictionary
        are the sorted integers stored in a tuple. The associated values
        are the number of times this triple was encountered in the algorithm
    """

    d = {}

    if method == 'brute_force':
        """
        loop through all possible integer combinations
        """
        for i in range(min_int, max_int+1):
            for j in range(min_int, max_int+1):
                for k in range(min_int, max_int+1):
                    if gcd(i,j,k)==1:
                        x = min(i, j, k)  # smallest
                        z = max(i, j, k)  # largest
                        y = (i + j + k) - (x + z)  # middle
                        key = (x,y,z)
                        try:
                            d[key] += 1
                        except KeyError:
                            d[key] = 1
    elif method == 'effecient':
        """
        short circuit loop when encountering a coprime double 
        """
        for i in range(min_int, max_int+1):
            for j in range(min_int, max_int+1):
                # if a pair is coprime, a triple must be coprime
                if gcd(i,j)==1:
                    for k in range(min_int, max_int+1):
                        x = min(i, j, k)  # smallest
                        z = max(i, j, k)  # largest
                        y = (i + j + k) - (x + z)  # middle
                        key = (x,y,z)
                        try:
                            d[key] += 1
                        except KeyError:
                            d[key] = 1
                # if not, check to see if triple is coprime
                else:
                    for k in range(min_int, max_int+1):
                        if gcd(i,j, k)==1:
                            x = min(i, j, k)  # smallest
                            z = max(i, j, k)  # largest
                            y = (i + j + k) - (x + z)  # middle
                            key = (x,y,z)
                            try:
                                d[key] += 1
                            except KeyError:
                                d[key] = 1
        
    else:
        msg = ('method not recognized.')
        raise ValueError(msg)

    return d















