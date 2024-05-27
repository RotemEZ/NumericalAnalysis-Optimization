"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def thomas_algo(self, a, b, c, d):
        # Using Thomas algorithm to find the value of A
        num = len(d)
        for i in range(1, num):
            t = a[i] / b[i - 1]
            b[i] = b[i] - t * c[i - 1]
            d[i] = d[i] - t * d[i - 1]
        x = np.zeros(num, dtype=object)
        x[-1] = d[-1] / b[-1]
        for i in range(num - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
        return x

    def cubic_bezier(self, a, b, c, d):
        # Creating cubic Bezier formula
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        xs = np.linspace(a, b, n)
        points = [np.array([x, f(x)]) for x in xs]

        sol_vec = np.zeros(n - 1, dtype=object)
        for i in range(1, n - 2): # Creating the solutions' vector
            sol_vec[i] = 2 * (2 * points[i] + points[i + 1])
        sol_vec[0] = points[0] + 2 * points[1]
        sol_vec[n - 2] = 8 * points[n - 2] + points[n - 1]

        # Creating the diagonals
        upper = np.zeros(n - 1, dtype=object)
        middle = np.zeros(n - 1, dtype=object)
        lower = np.zeros(n - 1, dtype=object)

        for i in range(n - 1):
            upper[i] = 1
            middle[i] = 4
            lower[i] = 1
        middle[0] = 2
        middle[n - 2] = 7
        lower[n - 2] = 2

        A = self.thomas_algo(lower, middle, upper, sol_vec)

        # Creating B
        B = np.zeros(n - 1, dtype=object)
        for i in range(n - 2):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 2] = (A[n - 2] + points[n - 1]) / 2

        lst = [self.cubic_bezier(points[i], A[i], B[i], points[i + 1]) for i in range(n - 1)]

        # Creating the function which we will return
        def g(x):
            r_0 = 0
            r_1 = 1
            idx = 0
            # Checking the interval that includes x
            for i in range(1, n):
                bracket = points[i][0]
                if x <= bracket:
                    idx = i
                    break
            x_0 = points[idx - 1][0]
            x_1 = A[idx - 1][0]
            x_2 = B[idx - 1][0]
            x_3 = points[idx][0]
            fun = lambda t: (-x_0 + 3 * x_1 - 3 * x_2 + x_3) * (t ** 3) + (3 * x_0 - 6 * x_1 + 3 * x_2) * (t ** 2) + (-3 * x_0 + 3 * x_1) * t + x_0 - x
            # Using Secant method
            f0 = fun(r_0)
            f1 = fun(r_1)
            if abs(f1) < abs(f0):
                z = r_1
            else:
                z = r_0
            while abs(fun(z)) > 0.0001:
                z = r_1 - ((fun(r_1) * (r_1 - r_0)) / (fun(r_1) - fun(r_0)))
                r_0 = r_1
                r_1 = z
            s = lst[idx - 1](z)
            sol = s[1]
            return sol

        return g

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
