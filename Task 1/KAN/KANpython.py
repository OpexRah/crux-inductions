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
        print(f"There are {len(basis)} basis B-Splines for the given grid configuration")
        return basis

    def forward_prop(self, x):
        """
        This function computes the B-spline at the given x and then multiplies it with the coefficients
        to return the output
        """
        output = 0
        for i in range(len(self.bspline_basis)):
            output += self.bspline_basis[i](x)*self.coefficients[i]

        return output
    
    def backward_prop(self, x, deriv):
        """
        This is the gradient decent part where we compute the gradient of the loss function with respect to the coefficients

        Loss Function: L
        Gradient of L with respect to coefficients: dL/dc_i = dL/dB * dB/dc_i (chain rule) where B is the spline function
        Since B is a linear combination of basis functions, dB/dc_i is the basis function itself
        So, dL/dc_i = dL/dB * B_i

        deriv: dL/dB
        x: input data to evaluate B
        """
        gradient = np.array([np.sum(deriv * self.bspline_basis[i](x)) for i in range(len(self.bspline_basis))])
        return gradient
    
    def train(self, x_train, y_train, learning_rate = 0.01, epochs = 2000):
        """
        grid: grid points for the B-spline
        x_train: input data
        y_train: output data generated by ideal function
        learning_rate: learning rate for gradient descent, default is 0.01
        epochs: number of iterations for training, default is 2000
        """

        for epoch in range(epochs):
            activations = self.forward_prop(x_train)
            deriv = 2 * (activations - y_train) / len(x_train)
            gradients = self.backward_prop(x_train, deriv)
            self.coefficients -= learning_rate * gradients

            mean_swquared_error = np.mean((y_train - activations)**2)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {mean_swquared_error}')
    
    def plot_splines(self, x_train, y_train = None, title= None, xlabel = None, ylabel = None, legend = False):
        """
        Takes the input data and plots the B-splines
        """
        plt.plot(x_train, self.forward_prop(x_train), label="Predicted output")
        if y_train is not None:
            plt.plot(x_train, y_train, label="Ground Truth", linestyle='dashed')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend:
            plt.legend()
        plt.show()
    
    def prune(self):
        self.coefficients = np.random.randn(len(self.bspline_basis))