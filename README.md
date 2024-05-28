# Numerical Analysis and Optimization Project
This project focuses on implementing various numerical analysis and optimization techniques in Python. It covers topics such as interpolation, function intersections, definite integrals, curve fitting, and shape area computation.
## Files and Descriptions
1. **interpolation.py:**
Implements a method for interpolating a given function using cubic Bezier curves, aiming to minimize the interpolation error while maintaining reasonable running time complexity.

2. **function_intersections.py:**
Contains a method for finding the intersection points between two given functions within a specified range and with a maximum error tolerance, using the Regula Falsi method.

3. **area_between_curves.py:**
Implements methods for computing the area enclosed between two given functions and performing numerical integration using techniques like Simpson's rule.

4. **curve_fitting.py:**
Focuses on fitting a model function to noisy data points sampled from a given function, using the least-squares method and adhering to a specified running time constraint.

5. **shape_area.py:**
Contains methods for fitting a shape to noisy data points sampled from a contour and computing the area of the fitted shape using numerical integration techniques like the shoelace formula and the K-Means clustering algorithm.

6. **NA_Final_Task.pdf:**
This PDF file contains the detailed instructions and requirements for the Numerical Analysis final task, including theoretical questions and grading policies for each assignment.
## Requirements
- Python 3.x
- This project relies on the following Python packages: Numpy, time, random, collections, sklearn (for shape_area.py only).
  You can install the necessary packages using pip. For example:
  ```bash
  pip install numpy
## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/RotemEZ/NumericalAnalysis-Optimization.git
2. Run any of the Python files to see the implemented numerical methods and algorithms:
   ```bash
   python <file_name>.py
Replace <file_name> with the name of the script you wish to run.
## Acknowledgements
This project was completed as part of a numerical analysis and optimization course. Special thanks to the course instructor and the provided materials. The project received a score of 100 and was placed second in the class, in terms of performance and efficiency.
