import numpy as np
from scipy.sparse import diags

def make_grid(xmin=-10.0, xmax=10.0, N=1024):
    """"
    This function creates a line grid on x between xmin & xmax, which contain N points and they are evenly spaced
    """
    x = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    return x, dx

def second_derivative_matrix(N, dx):
    """
    central 3-point: (f_{i+1} - 2 f_i + f_{i-1}) / dx^2 this an approximation of the 1D Laplacian operator
    N is the number of points in the grid
    """
    main = -2.0 * np.ones(N) / dx**2
    off  = 1.0 * np.ones(N-1) / dx**2
    D2 = diags([off, main, off], offsets=[-1,0,1], format='csr')
    return D2

def infinite_well_potential(x, L):
    """
    Basically, this return a matrix that have zeros in every point of x in our grid, which can represent
    the infinite well potential
    """
    V = np.zeros_like(x)
    return V

def build_hamiltonian(D2, V):
    """
    :param D2: This is the 1D Laplacian operator
    :param V: This is the well potential
    :return: The Hamiltonian operator
    """
    H = (-0.5) * D2 + diags(V, 0, format='csr')
    return H
