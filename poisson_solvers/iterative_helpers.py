# Author: Kevin Howarth                                                         #
# Date: 1/12/2022                                                               #
# File: iterative_helpers.py                                                    #
# Description: Contains helper functions for constructing finite difference     #
# matrices and creating surface plots of solutions. These helper functions are  #
# used by the iterative solver routines as well as in the rest of the PIC code. #
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot3DSurface(ZList, titleString=None, bc="Poisson2DZeroBoundary"):

    """

    Helper function for plotting a 3D surface. If multiple data sets are provided, creates a subplot.
    Note that plotting is always done on the domain [-1, -1] to [1, 1].

    Parameters
    ----------
    ZList:  A length m list containing each data set to plot. All data sets must be either N-by-N arrays or length N^2
            vectors containing the same number of elements.
    titleString:  String containing the title, if any, to add to the plot. If ZList is a list, titleString must
                  also be a list containing the string to use for each subplot title.
    bc:  String specifying which boundary condition was used to generate the data set. This is necessary to determine
         how the data set should be padded to fit on the domain. Available cases include
                - "Poisson2DZeroBoundary" - zero Dirichlet boundary conditions
                - "Poisson2DPeriodic" - periodic Dirichlet boundary conditions

    Raises
    -------
    ValueError:  Raises ValueError if a 1D input could not be reshaped into a square 2D array.

    Returns
    -------
    None

    """

    m = len(ZList)

    # Perform padding, reshape if necessary, and find grid spacing based on boundary conditions
    if bc == "Poisson2DZeroBoundary":
        for mi in range(m):
            if ZList[mi].ndim == 1:
                try:
                    dim = int(np.sqrt(ZList[mi].shape[0]))
                    ZList[mi] = ZList[mi].reshape((dim, dim))
                except:
                    raise ValueError("One or more input could not be shaped into a square 2D array.")
            ZList[mi] = np.pad(ZList[mi], 1)

        gridSize = ZList[0].shape[0]
        meshSpacing = 2 / (gridSize - 1)
    else:
        for mi in range(m):
            if ZList[mi].ndim == 1:
                try:
                    dim = int(np.sqrt(ZList[mi].shape[0]))
                    ZList[mi] = ZList[mi].reshape((dim, dim))
                except:
                    raise ValueError("One or more input could not be shaped into a square 2D array.")
            ZList[mi] = np.pad(ZList[mi], 1, mode="wrap")[1:, 1:]

        gridSize = ZList[0].shape[0]
        meshSpacing = 2 / (gridSize - 1)

    # Create X and Y data, get axes, and plot
    X = np.arange(-1, 1 + meshSpacing, meshSpacing)
    Y = np.arange(-1, 1 + meshSpacing, meshSpacing)
    X, Y = np.meshgrid(X, Y)
    fig, axes = plt.subplots(1, m, subplot_kw={"projection": "3d"})
    if m > 1:
        for mi in range(m):
            axes[mi].plot_surface(X, Y, ZList[mi], cmap=cm.Blues)
            if titleString:
                axes[mi].set_title(titleString[mi])
    else:
        axes.plot_surface(X, Y, ZList[mi], cmap=cm.Blues)
        if titleString:
            axes.set_title(titleString)
    plt.grid()
    plt.show()

def get2DCenteredDifferenceMatrix(N, h, bc):

    """

    Function for constructing finite difference operator matrices. For a 2D grid, Poisson's equation can
    be discretized as:

        u_(i-1,j) + u_(i+1,j) - 4u_(i,j) + u_(i,j-1) + u_(i,j-1)
        -------------------------------------------------------- = -f_(i,j)
                                  h^2

    Since i,j = [1, 2, ... N], there are N^2 grid points. This creates a system of N^2 algebraic equations representing
    the discretized Poisson equation. This can be written as a large, sparse linear system of the form Ax = b. The A
    matrix contains the numerical coefficients of the second-order centered difference stencil. This function constructs
    this matrix for a Poisson grid of size N by N with either zero Dirichlet or periodic Dirichlet boundary conditions.

    Parameters
    ----------
    N:  Scalar int specifying the dimension of the computational grid
    h:  Scalar float specifying the mesh spacing of the computational grid
    bc:  String specifying which boundary condition to use. Available cases include
                - "Poisson2DZeroBoundary" - zero Dirichlet boundary conditions
                - "Poisson2DPeriodic" - periodic Dirichlet boundary conditions

    Raises
    -------
    ValueError:  ValueError is raised if an invalid boundary condition is supplied

    Returns
    -------
    A:  an N^2 by N^2 2D array containing the sparse finite differencing matrix.

    """

    if bc == "Poisson2DZeroBoundary":
        # For the zero boundary conditions we can use a nice trick involving kronecker products to construct the matrix
        A = -2 * np.eye(N) + np.diag(np.ones(N - 1), k=1) + np.diag(np.ones(N - 1), k=-1)
        A = np.kron(np.eye(N), A) + np.kron(A, np.eye(N))
    elif bc == "Poisson2DPeriodic":
        # In the periodic case, the matrix is still very sparse, but each row always has five nonzero entries, as a
        # point on the grid always has four orthogonal neighbors due to periodic wrapping.
        A = np.zeros((N**2, N**2))

        # mapping function from 2D indexing to 1D indexing
        def M(x, y):
            return y * N + x

        # loop through 2D coordinates and populate finite difference A matrix
        for i in range(N):
            for j in range(N):
                Ai = M(i, j)
                A[Ai, M(i, j - 1) % (N ** 2)] = 1.
                A[Ai, M(i - 1, j) % (N ** 2)] = 1.
                A[Ai, M(i, j)] = -4.
                A[Ai, M(i, j + 1) % (N ** 2)] = 1.
                A[Ai, M(i + 1, j) % (N ** 2)] = 1.
    else:
        raise ValueError(("Please provide a valid boundary condition case. Valid cases include 'Poisson2DZeroBoundary'"
                          " and 'Poisson2DPeriodic'."))

    return (1 / h ** 2) * A
