"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass


    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """


        def g(x):
            return f1(x) - f2(x)

        # Creating 100 intervals
        val = (b - a) / 100
        t = a + val
        x1, x2 = a, t
        X = []
        while t < b:
            # Checking if there might be an intersection point in this sub-interval.
            if g(a) * g(t) > 0:
                a += val
                t += val
                x1 = a
                x2 = t
                continue
            f1x1, f1x2 = g(x1), g(x2)

            # We will find the intersection point in this sub-interval using the Regula Falsi formula
            x3 = x2 - f1x2 * (x2 - x1) / (f1x2 - f1x1)
            f1x3 = g(x3)

            if abs(f1x3) < maxerr:
                X.append(x3)
                a += val
                t += val
                x1 = a
                x2 = t
            else:
                if f1x1 * f1x3 < 0:
                    x2 = x3
                else:
                    x1 = x3

        return X
##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
