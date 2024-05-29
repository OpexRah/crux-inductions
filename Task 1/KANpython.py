import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class KAN:
    def __init__(self, grid_points, degree = 3):
        """
        grid_points: grid where the B-spline will be defined
        degree: degree of the B-spline which is 3 by defualt (cubic Bspline)
        """

        self.grid_points = grid_points
        print(self.grid_points)
        self.degree = degree

        """
        Any BSpline can be defined as a linear combination of B-spline basis functions.
        B(t) = c0*B0(t) + c1*B1(t) + c2*B2(t) + ... + cn*Bn(t)
        where B(t) is the B-spline, B0(t), B1(t), B2(t), ..., Bn(t) are the basis functions
        and c0, c1, c2, ..., cn are the coefficients.
        
        The B-spline basis is always fixed for a given degree and grid points.
        So the only thing that changes is the coefficients (this is what the model learns)
        """
        self.bspline_basis = self.basis_functions()
        # Randomly initialize the coefficients
        self.coefficients = np.random.randn(len(self.bspline_basis))
    
    def basis_functions(self):
        """
        This function computes the B-spline basis functions for the given grid points
        """
        knots = np.concatenate((
            [self.grid_points[0]]*self.degree,
            self.grid_points,
            [self.grid_points[-1]]*self.degree
        )) # The extra points at the start and end are to ensure that the Bspline is fixed at the start and end points
        print(knots)
        basis = []
        for i in range(len(self.grid_points) + self.degree - 1):
            # Compute the B-spline basis function for the i-th segment
            b_spline = interpolate.BSpline.basis_element(knots[i:i + self.degree + 1])
            basis.append(b_spline)

        return basis

