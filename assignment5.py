"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
from sklearn.cluster import KMeans
import math


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, area):
        # Receiving the area of the shape and save it as a field
        self.area1 = area

    def contour(self, n: int):
        pass

    def area(self) -> np.float32:
        # This method return the shape's area
        return np.float32(self.area1)

class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        # Starting to check the area with 10 points
        n = 10
        area = 0
        error = float("inf")

        while error > maxerr and n < 10000:
            points = contour(n)
            xs = points[:, 0]
            ys = points[:, 1]
            # Shoelace formula
            sec_area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
            error = abs(sec_area - area)
            area = sec_area
            n *= 10
        return np.float32(area)



    def sort_points(self, points, reference_point):
        # Calculate the polar angle of each point
        def calc_polar_angle(point):
            x, y = point
            ref_x, ref_y = reference_point
            return math.atan2(y - ref_y, x - ref_x)

        # Sort the points based on their polar angles
        sorted_points = sorted(points, key=calc_polar_angle)
        return sorted_points
    
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        # replace these lines with your solution
        # Checking the time of calling one point from the contour
        start = time.perf_counter()
        point = sample()
        interval = time.perf_counter() - start
        # If one interval is bigger than the maxerr, then it's not enough for the area's calculations
        if interval > 0.97 * maxtime:
            return MyShape(0)
        points = [point]
        num = int((0.97 * maxtime - interval) / interval)
        for i in range(num):
            point = sample()
            points.append(point)

        # Finding the center of all the points
        center_point = (sum((point[0] for point in points)) / len(points),
               sum((point[1] for point in points)) / len(points))


        kmeans = KMeans(n_clusters=36, n_init=1)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_

        sorted_centers = self.sort_points(centers, center_point)
        # Using shoelace formula
        area = 0
        area += (sorted_centers[len(sorted_centers) - 1][0] * sorted_centers[0][1])
        area -= (sorted_centers[0][0] * sorted_centers[len(sorted_centers) - 1][1])
        for i in range(len(sorted_centers) - 1):
            area += (sorted_centers[i][0] * sorted_centers[i + 1][1])
            area -= (sorted_centers[i + 1][0] * sorted_centers[i][1])
        area = np.float32(0.5 * abs(area))
        return MyShape(area)







##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
