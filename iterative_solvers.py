# Authors: Kevin Howarth                                                        #
# Date: 1/7/2022                                                                #
# File: iterative_solvers.py                                                    #
# Description: Implementation of iterative methods for solving large systems of #
# linear equations, such as those arising in the finite-difference solution of  #
# the Poisson equation and other elliptic PDEs. The methods developed in this   #
# file are primarily meant to be imported and called by other functions, but    #
# some testing routines are available and can be accessed by calling this file  #
# via the command line with keyword arguments corresponding to the test you     #
# want to run. To see a list of all keywords and testing routines, simply call  #
# the file as:                                                                  #
#      >> python iterative_solvers.py                                           #
#################################################################################

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

###########################################################
#          HIGH-LEVEL TESTS AND HELPER FUNCTIONS          #
###########################################################

def convergenceTest(A, P):

    """

    Function for testing whether an iterative method will converge. Iterative methods
    are defined by a matrix splitting of a linear system:

        Ax = b
        Px + Ax = Px + b
        Px = (P - A)x + b

    This equation can be used as a stationary iteration, starting with some guess x0.
    It can be shown that the error equation for such an iteration is given by:

        e_k+1 = M@e_k

    Where the matrix M = I - inv(P)@A. To guarantee convergence, every eigenvalue of M
    must have magnitude less than or equal to 1. Further, the spectral radius, defined
    as the magnitude of the largest eigenvalue of M, sets the convergence rate of the
    method, and thus it is of special interest.

    Given an arbitrary matrix A and a preconditioning matrix P, this function checks
    whether an iteration based on P will be convergent and finds the spectral radius.

    Parameters
    ----------
    A:  n by n 2D np.array containing arbitrary values. Presumably it is large and sparse,
        eg. a finite difference coefficient matrix.
    P:  n by n 2D np.array containing the preconditioner to be used for iteration.

    Raises
    -------
    ValueError:  Value error is raised if input data shapes are not appropriate

    Returns
    -------
    isConvergent:  a boolean describing whether an iteration defined by P will converge
    spectralRadius:  a scalar float containing the spectral radius of M

    """

    # input checking
    if not A.shape[0] == A.shape[1] or not P.shape[0] == P.shape[1]:
        raise ValueError("ERROR: A and P must be square matrices.")

    if not A.shape == P.shape:
        raise ValueError("ERROR: A and P must have the same dimensions.")

    # build the M matrix
    n = A.shape[0]
    M = np.eye(n) - LA.inv(P)@A

    # get the eigenvalues of M
    w = LA.eigvals(M)

    # check magnitude of each eigenvalue
    spectralRadius = np.abs(w[0])
    for wi in w:
        if np.abs(wi) > spectralRadius:
            spectralRadius = np.abs(wi)

    if spectralRadius >= 1:
        return False, spectralRadius
    else:
        return True, spectralRadius

def plot3DSurface(ZList, titleString=None, bc="Poisson2DZeroBoundary"):

    """

    Helper function for plotting a 3D surface. If multiple data sets are provided, creates a subplot.
    Note that plotting is always done on the domain [-1, -1] to [1, 1].

    Parameters
    ----------
    ZList:  A length m list containing each data set to plot. All data sets must be N-by-N arrays of equal size.
    titleString:  String containing the title, if any, to add to the plot. If ZList is a list, titleString must
                  also be a list containing the string to use for each subplot title.
    bc:  String specifying which boundary condition was used to generate the data set. This is necessary to determine
         how the data set should be padded to fit on the domain. Available cases include
                - "Poisson2DZeroBoundary" - zero Dirichlet boundary conditions
                - "Poisson2DPeriodic" - periodic Dirichlet boundary conditions

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # Get system size
    m = len(ZList)
    gridSize = ZList[0].shape[0]

    # Get correct mesh spacing and perform padding based on boundary conditions
    if bc == "Poisson2DZeroBoundary":
        meshSpacing = 2 / (gridSize + 1)
        for mi in range(m):
            ZList[mi] = np.pad(ZList[mi], 1)
    else:
        meshSpacing = 2 / gridSize
        for mi in range(m):
            ZList[mi] = np.pad(ZList[mi], 1, mode="wrap")[1:, 1:]

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

def runIterativeMethod(N, itrList, method, bc, plotExtra=False):

    """

    Function for running iterative methods to test convergence with increasing iterations.
    A key metric for assessing the usefulness of an iterative method is how quickly, and how well, it converges
    to the direct solution. This handler function carries out a convergence test on the specified method with the
    specified boundary condition for a specified list of iterations. It first computes the direct solution of the linear
    system, then calculates the L2 norm of the error at each number of iterations.

    This method is used exclusively for studying convergence on a 2D Poisson problem on the domain [-1, -1] to [1, 1].

    Parameters
    ----------
    N:  Scalar int specifying the dimension of the Poisson grid
    itrList:  a 1D int array containing the list of numbers of iterations to calculate the error at.
    method:  String specifying which iterative method to use. Available cases include
                - "jacobi" - uses the Jacobi method
                - "GS" - uses the Gauss-Seidel method
                - "SOR" - Uses Successive Over-Relaxation
                - "SORCA" - Uses Successive Over-Relaxation with Chebyshev Acceleration
    bc:  String specifying which boundary condition to use. Available cases include
                - "Poisson2DZeroBoundary" - zero Dirichlet boundary conditions
                - "Poisson2DPeriodic" - periodic Dirichlet boundary conditions
    plotExtra:  boolean representing whether the source term and direct solution should be plotted

    Raises
    -------
    ValueError:  ValueError is raised if invalid method or boundary conditions are supplied.

    Returns
    -------
    errorList:  a 1D array containing the L2 norm of the error at each specified number of iterations

    """

    # Input checking
    if method not in ["jacobi", "GS", "SOR", "SORCA"]:
        raise ValueError("Please specify a valid iterative method. Valid methods include 'jacobi', 'GS', and 'SOR'.")
    elif bc not in ["Poisson2DZeroBoundary", "Poisson2DPeriodic"]:
        raise ValueError(("Please provide a valid boundary condition case. Valid cases include 'Poisson2DZeroBoundary'"
                          " and 'Poisson2DPeriodic'."))

    # First create the source term, depending on choice of boundary conditions. The square from [-1,1] to [1,1] is the
    # domain of choice. Note that the source and solution terms are only specified on the interior of the grid for the
    # zero boundary case, while they are specified on the interior of the grid as well as one boundary for the
    # periodic boundary case.

    # Mesh size
    dx = 2 / (N - 1)

    if bc == "Poisson2DZeroBoundary":
        plotTitleString1 = "Source Term, Zero Boundary"
        plotTitleString2 = "Direct Solution, Zero Boundary"
        NGrid = N - 2

        # Mapping function for grid point xi to real space, valid on either axis. Starts at -1 + dx to avoid
        # boundary points.
        def M(xi):
            return -1 + dx + xi * dx

        # Source is a step function with a large number of ones at the center of the grid, and zeroes elsewhere.
        bSource = np.zeros((NGrid, NGrid), dtype=float)
        for i in range(NGrid):
            for j in range(NGrid):
                if abs(M(i)) <= 0.5 and abs(M(j)) <= 0.5:
                    bSource[i, j] = 1.0

        # Construct the second-order centered-difference coefficient matrix for the inner grid domain.
        A_2DCenteredDiff = get2DCenteredDifferenceMatrix(NGrid, dx, bc=bc)

    elif bc == "Poisson2DPeriodic":
        plotTitleString1 = "Source Term, Periodic"
        plotTitleString2 = "Direct Solution, Periodic"
        NGrid = N - 1

        # Mapping function for grid point xi to real space, valid on either axis. Starts at -1 to include
        # boundary points.
        def M(xi):
            return -1 + xi * dx

        # Source is a cosine function scaled to be periodic on the domain.
        def periodicSource(xi, yi):
            return np.cos(M(xi) * np.pi) * np.cos(M(yi) * np.pi)

        bSource = np.zeros((NGrid, NGrid), dtype=float)
        for i in range(NGrid):
            for j in range(NGrid):
                bSource[i, j] = periodicSource(i, j)

        # Construct the second-order centered-difference coefficient matrix for the inner grid domain and boundary.
        A_2DCenteredDiff = get2DCenteredDifferenceMatrix(NGrid, dx, bc=bc)

    # Get the "exact" solution as obtained by direct solution of the finite difference system
    xExact = LA.solve(A_2DCenteredDiff, -1.*bSource.flatten())

    if argList[2] == "P":
        # Adding a correction term to ensure a fair comparison between the direct and iterative solutions. Both satisfy
        # Poisson's equation, but the direct solution ends up with an extra constant factor added in.
        xExact = xExact - np.average(xExact)

    # Plot source and solution if specified
    if plotExtra:
        if argList[2] == "NP":
            plot3DSurface([bSource, xExact.reshape((NGrid, NGrid))], [plotTitleString1, plotTitleString2], bc="Poisson2DZeroBoundary")
        elif argList[2] == "P":
            plot3DSurface([bSource, xExact.reshape((NGrid, NGrid))], [plotTitleString1, plotTitleString2], bc="Poisson2DPeriodic")

    # Now get the solution via the specified iteration, and observe how the error changes with increasing iterations.
    errorList = np.zeros_like(itrList, dtype=float)
    for itr in range(len(itrList)):
        if method == "jacobi":
            xIter = jacobiIter(bSource, dx, bc=bc, maxItr=itrList[itr])
        elif method == "GS":
            xIter = gaussSeidelIter(bSource, dx, bc=bc, maxItr=itrList[itr])
        elif method == "SOR":
            xIter = gaussSeidelIter(bSource, dx, bc=bc, maxItr=itrList[itr], SOR=True)
        elif method == "SORCA":
            xIter = gaussSeidelIter(bSource, dx, bc=bc, maxItr=itrList[itr], SOR=True, chebyshevAccel=True)

        # Take the squared 2 norm of the error as the error measure
        errorList[itr] = LA.norm(xIter.flatten() - xExact) ** 2

    return errorList

###########################################################
#                    ITERATION METHODS                    #
###########################################################

def jacobiIter(b, h, bc, maxItr=5000):

    """

    Function implementing the Jacobi method to solve various system structures. The
    Jacobi method applies a simple iteration which arises from choosing the
    preconditioning matrix as the diagonal part of A. Thus the iteration can be written:

        Px = (P - A)x + b
        x = (I - inv(D)@A)x + inv(D)b

    Since different matrix structures of A will arise depending on the system being solved,
    the boundary conditions, etc., this function implements multiple use cases which each
    imply the structure of A. Each case will apply a finite difference discretization
    to the Poisson equation, thus a step size h is a required input.

    Parameters
    ----------
    b:  2D np.array of length n by n containing the source term for the Poisson equation
    h:  Scalar float specifying the mesh spacing of the computational grid
    bc:  String specifying which solution case to use. Available cases include
                - "Poisson2DZeroBoundary" - solves Poisson equation on a 2D grid with zero Dirichlet boundary conditions
                - "Poisson2DPeriodic" - solves Poisson equation on a 2D grid with periodic Dirichlet boundary conditions
    maxItr: Scalar float specifying the number of iterations to use

    Raises
    -------
    ValueError:  ValueError is raised if an invalid boundary condition is supplied

    Returns
    -------
    x:  a 1D array containing the solution estimate of Ax = b obtained via Jacobi iteration

    """

    if bc not in ["Poisson2DZeroBoundary", "Poisson2DPeriodic"]:
        raise ValueError(("Please specify a valid boundary condition. Valid conditions include "
                          "\"Poisson2DZeroBoundary\" and \"Poisson2DPeriodic\""))

    # get system size
    n = b.shape[0]

    # pad the source term and initialize solution as all zeros
    b = np.pad(b, 1)
    x = np.zeros_like(b)
    xN = x.copy()

    # perform iteration
    itr = 0
    while itr < maxItr:
        # update all points actually represented on the grid using adjacent points
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                xN[i, j] = (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1] + h**2 * b[i, j]) / 4.

        # ensure that periodicity is enforced, if relevant
        if bc == "Poisson2DPeriodic":
            xN = np.pad(xN[1:n + 1, 1:n + 1], 1, mode="wrap")

        # update current matrix data and iterate
        x = xN.copy()
        itr += 1

    # remove padding on solution and return
    return x[1:-1, 1:-1]

def gaussSeidelIter(b, h, bc, maxItr=5000, SOR=False, chebyshevAccel=False):

    """

    Function implementing the Gauss-Seidel method to solve various system structures. The Gauss-Seidel method applies an
    iteration which arises from choosing the preconditioning matrix as the lower triangular part of A. Thus the
    iteration can be written:

        Px = (P - A)x + b
        x = (I - inv(L)@A)x + inv(L)b

    Since different matrix structures of A will arise depending on the system being solved, the boundary conditions,
    etc., this function implements multiple use cases which each imply the structure of A. Each case will apply a finite
    difference discretization to the Poisson equation, thus a step size h is a required input.

    The Gauss-Seidel algorithm is improved upon by the Successive Over-Relaxation (SOR) method, which uses the
    heuristic that if the direction from x_k to x_k+1 is already a good direction to move, then we may as well move
    further in that direction. A boolean input allows the user to specify whether SOR should be used instead.

    Note that this implementation uses the red-black ordering scheme, and the SOR parameter omega (notated w) is chosen
    using a formula given by J. W. Demmel's Applied Numerical Linear Algebra (SIAM, 1997). Alternatively, Chebyshev
    Acceleration as described in Hockney and Eastwood's Computer Simulation Using Particles (IOP 1988) may be used to
    update omega at each iteration such that the initial increase in error often observed with SOR is mitigated.

    Parameters
    ----------
    b:  2D np.array of length n by n containing the source term for the Poisson equation
    h:  Scalar float specifying the mesh spacing of the computational grid
    bc:  String specifying which solution case to use. Available cases include
                - "Poisson2DZeroBoundary" - solves Poisson equation on a 2D grid with zero Dirichlet boundary conditions
                - "Poisson2DPeriodic" - solves Poisson equation on a 2D grid with periodic Dirichlet boundary conditions
    maxItr:  Scalar float specifying the number of iterations to use
    SOR:  boolean describing whether the Successive Over-Relaxation algorithm should be used
    chebyshevAccel:  Boolean specifying whether Chebyshev Acceleration should be used to update omega at each iteration.
                     Note that Chebyshev Acceleration can only be used if SOR = True.

    Raises
    -------
    ValueError:  ValueError is raised if an invalid boundary condition is supplied or if chebyshevAccel = True when
                 SOR = False, which is an invalid use case.

    Returns
    -------
    x:  a 1D array containing the solution estimate of Ax = b obtained via Gauss-Seidel iteration

    """

    if bc not in ["Poisson2DZeroBoundary", "Poisson2DPeriodic"]:
        raise ValueError(("Please specify a valid boundary condition. Valid conditions include "
                          "\"Poisson2DZeroBoundary\" and \"Poisson2DPeriodic\""))
    elif (not SOR) and chebyshevAccel:
        raise ValueError("Chebyshev Acceleration may only be used with Successive Over-Relaxation.")

    # get system size
    n = b.shape[0]

    # pad the source term and initialize solution as all zeros
    b = np.pad(b, 1)
    x = np.zeros_like(b)

    if SOR:
        # get the over-relaxation parameter, using Chebyshev Acceleration method if needed
        if bc == "Poisson2DZeroBoundary":
            if chebyshevAccel:
                rho = 1 - 0.5*(np.pi ** 2 / n ** 2)
                w = 1.
            else:
                w = 2 / (1 + np.sin(np.pi / (n + 1)))
        else:
            if chebyshevAccel:
                # Note: there is no guarantee Chebyshev Acceleration will work for the periodic boundary condition, as
                # the spectral radius of Jacobi, rho, is actually greater than 1.
                rho = 0.999  # A guess near 1
                w = 1.
            else:
                w = 1.7  # a guess nearish to 2

    # perform iteration
    itr = 0
    while itr < maxItr:
        # update red nodes using old information. Careful indexing of the for loops visits only the red nodes.
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if (i + j) % 2 == 0:
                    if not SOR:
                        x[i, j] = (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1] + h ** 2 * b[i, j]) / 4.
                    else:
                        x[i, j] = (1 - w) * x[i, j] + \
                                  w * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1] + h ** 2 * b[i, j]) / 4.

        # ensure that periodicity is enforced, if relevant
        if bc == "Poisson2DPeriodic":
            x = np.pad(x[1:n + 1, 1:n + 1],1,mode="wrap")

        # Update omega, if relevant
        if chebyshevAccel:
            if itr == 0:
                w = 1 / (1 - 0.5 * rho ** 2)
            else:
                w = 1 / (1 - 0.25 * w * rho ** 2)

        # update black nodes using new information
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if (i + j) % 2 == 1:
                    if not SOR:
                        x[i, j] = (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1] + h ** 2 * b[i, j]) / 4.
                    else:
                        x[i, j] = (1 - w) * x[i, j] + \
                                  w * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1] + h ** 2 * b[i, j]) / 4.

        # again enforce periodicity
        if bc == "Poisson2DPeriodic":
            x = np.pad(x[1:n + 1, 1:n + 1], 1, mode="wrap")

        # Again update omega
        if chebyshevAccel:
            w = 1 / (1 - 0.25 * w * rho ** 2)

        itr += 1

    # remove padding on solution and return
    return x[1:-1, 1:-1]

if __name__ == '__main__':

    argList = sys.argv
    # argList = ['iterative_solvers.py', '--SOR','P','65']  # fake input for debugging

    if len(argList) == 1:
        print("Must use an extra argument to specify a test.")
        print("  '--convergence N' - run tests checking whether a few different iterative methods will converge for a")
        print("                      few Poisson cases. N is the system size.")
        print("  '--jacobi bc N' - test accuracy of the Jacobi method with increasing iterations. bc is the boundary")
        print("                    condition mode (P for periodic, NP for non-periodic), and N is the system size.")
        print("  '--GS bc N' - test accuracy of the Gauss-Seidel method with increasing iterations. bc is the boundary")
        print("                condition mode (P for periodic, NP for non-periodic), and N is the system size.")
        print("  '--SOR bc N' - test accuracy of the Successive Over-Relaxation method with increasing iterations.")
        print("                 bc is the boundary condition mode (P for periodic, NP for non-periodic), and N is")
        print("                 the system size.")
        print("  '--SORCA bc N' - test accuracy of the Successive Over-Relaxation method with Chebyshev Acceleration")
        print("                   for increasing iterations. bc is the boundary condition mode (P for periodic, NP")
        print("                   for non-periodic), and N is the system size.")
        print("  '--methodcomparison bc N' - test the convergence rate of each of the four iteration methods for a")
        print("                              given boundary condition bc and system size N.")
        print()
        print("Example usage: 'python iterative_solvers.py --convergence 10'")

    elif argList[1] == "--convergence":
        # tests for convergenceTest function
        if not len(argList) == 3:
            print("Must provide system size N.")
            print("Example usage: 'python iterative_solvers.py --convergence 10'")
            sys.exit()

        # construct a standard 2D centered-difference coefficient matrix for an N by N Poisson equation grid
        NInput = int(argList[2])
        A_centeredDiff2D = get2DCenteredDifferenceMatrix(NInput, 1, bc="Poisson2DZeroBoundary")

        # construct a 2D centered-difference coefficient matrix for N by N Poisson with periodic boundary conditions
        A_centeredDiff2DPeriodic = get2DCenteredDifferenceMatrix(NInput, 1, bc="Poisson2DPeriodic")

        # construct the conditioner matrix for Jacobi iteration - simply the diagonal part of A
        P_Jacobi = np.diag(np.diag(A_centeredDiff2D))
        P_JacobiPeriodic = np.diag(np.diag(A_centeredDiff2DPeriodic))

        # construct the conditioner matrix for Gauss-Seidel iteration - the lower triangular part of A
        P_GS = np.tril(A_centeredDiff2D)
        P_GSPeriodic = np.tril(A_centeredDiff2DPeriodic)

        # construct the conditioner matrix for Successive Over-Relaxation
        omega = 1.6  # a guess for omega, approaching 2
        P_SOR = np.diag(np.diag(A_centeredDiff2D)) + omega * np.tril(A_centeredDiff2D, k=-1)
        P_SORPeriodic = np.diag(np.diag(A_centeredDiff2DPeriodic)) + omega * np.tril(A_centeredDiff2DPeriodic, k=-1)

        # check convergence and spectral radii
        isConvergentJac, spectralRadJac = convergenceTest(A_centeredDiff2D, P_Jacobi)
        isConvergentGS, spectralRadGS = convergenceTest(A_centeredDiff2D, P_GS)
        isConvergentSOR, spectralRadSOR = convergenceTest(A_centeredDiff2D, P_SOR)
        isConvergentJacPeriodic, spectralRadJacPeriodic = convergenceTest(A_centeredDiff2DPeriodic, P_JacobiPeriodic)
        isConvergentGSPeriodic, spectralRadGSPeriodic = convergenceTest(A_centeredDiff2DPeriodic, P_GSPeriodic)
        isConvergentSORPeriodic, spectralRadSORPeriodic = convergenceTest(A_centeredDiff2DPeriodic, P_SORPeriodic)

        print("First check the 2D centered-difference case with zero Dirichlet boundary conditions:")
        print("  - Jacobi method")
        print("     - is convergent:",isConvergentJac)
        print("     - spectral radius:",spectralRadJac)
        print("  - Gauss-Seidel method:")
        print("     - is convergent:", isConvergentGS)
        print("     - spectral radius:",spectralRadGS)
        print("  - Successive Over-Relaxation method:")
        print("     - is convergent:", isConvergentSOR)
        print("     - spectral radius:", spectralRadSOR)
        print("")
        print("Then check the 2D centered-difference case with periodic Dirichlet boundary conditions.")
        print("  - Jacobi method")
        print("     - is convergent:", isConvergentJacPeriodic)
        print("     - spectral radius:", spectralRadJacPeriodic)
        print("  - Gauss-Seidel method:")
        print("     - is convergent:", isConvergentGSPeriodic)
        print("     - spectral radius:", spectralRadGSPeriodic)
        print("  - Successive Over-Relaxation method:")
        print("     - is convergent:", isConvergentSORPeriodic)
        print("     - spectral radius:", spectralRadSORPeriodic)

    elif argList[1] == "--jacobi":
        # tests for Jacobi iteration
        if not len(argList) == 4:
            print("Must specify boundary condition (P for periodic, NP for non-periodic) and system size N.")
            print("Example usage: 'python iterative_solvers.py --jacobi NP 65'")
            sys.exit()
        elif argList[2] not in ["P", "NP"]:
            print("Must specify boundary condition as either P for periodic or NP for non-periodic (zero boundary).")
            sys.exit()
        elif int(argList[3]) < 14:
            print("Must specify number of grid points as at least 14 in order to avoid matplotlib broadcasting errors.")
            sys.exit()

        if argList[2] == "P":
            BC = "Poisson2DPeriodic"
            titleStr = "Convergence of Jacobi Method, N = " + argList[3] + ", Periodic BC"
        elif argList[2] == "NP":
            BC = "Poisson2DZeroBoundary"
            titleStr = "Convergence of Jacobi Method, N = " + argList[3] + ", Zero BC"

        # set number of iterations and call runner function
        iterList = np.array([2 ** i for i in range(5, 13)], dtype=int)
        errList = runIterativeMethod(int(argList[3]), iterList, method="jacobi", bc=BC, plotExtra=True)

        # plot the error against number of iterations
        plt.semilogy(iterList,errList)
        plt.xlabel("# of Iterations")
        plt.ylabel("Sum of Squared Solution Error")
        plt.title(titleStr)
        plt.grid()
        plt.show()

    elif argList[1] == "--GS":
        # tests for Gauss-Seidel iteration
        if not len(argList) == 4:
            print("Must specify boundary condition (P for periodic, NP for non-periodic) and system size N.")
            print("Example usage: 'python iterative_solvers.py --GS NP 65'")
            sys.exit()
        elif argList[2] not in ["P", "NP"]:
            print("Must specify boundary condition as either P for periodic or NP for non-periodic (zero boundary).")
            sys.exit()
        elif int(argList[3]) < 14:
            print("Must specify number of grid points as at least 14 in order to avoid matplotlib broadcasting errors.")
            sys.exit()

        if argList[2] == "P":
            BC = "Poisson2DPeriodic"
            titleStr = "Convergence of Gauss-Seidel Method, N = " + argList[3] + ", Periodic BC"
        elif argList[2] == "NP":
            BC = "Poisson2DZeroBoundary"
            titleStr = "Convergence of Gauss-Seidel Method, N = " + argList[3] + ", Zero BC"

        # set number of iterations and call runner function
        iterList = np.array([2 ** i for i in range(5, 13)], dtype=int)
        errList = runIterativeMethod(int(argList[3]), iterList, method="GS", bc=BC, plotExtra=True)

        # plot the error against number of iterations
        plt.semilogy(iterList,errList)
        plt.xlabel("# of Iterations")
        plt.ylabel("Sum of Squared Solution Error")
        plt.title(titleStr)
        plt.grid()
        plt.show()

    elif argList[1] == "--SOR":
        # tests for Successive Over-Relaxation iteration
        if not len(argList) == 4:
            print("Must specify boundary condition (P for periodic, NP for non-periodic) and system size N.")
            print("Example usage: 'python iterative_solvers.py --SOR NP 65'")
            sys.exit()
        elif argList[2] not in ["P", "NP"]:
            print("Must specify boundary condition as either P for periodic or NP for non-periodic (zero boundary).")
            sys.exit()
        elif int(argList[3]) < 14:
            print("Must specify number of grid points as at least 14 in order to avoid matplotlib broadcasting errors.")
            sys.exit()

        if argList[2] == "P":
            BC = "Poisson2DPeriodic"
            titleStr = "Convergence of SOR Method, N = " + argList[3] + ", Periodic BC"
        elif argList[2] == "NP":
            BC = "Poisson2DZeroBoundary"
            titleStr = "Convergence of SOR Method, N = " + argList[3] + ", Zero BC"

        # set number of iterations and call runner function
        iterList = np.array([2 ** i for i in range(5, 13)], dtype=int)
        errList = runIterativeMethod(int(argList[3]), iterList, method="SOR", bc=BC, plotExtra=True)

        # plot the error against number of iterations
        plt.semilogy(iterList,errList)
        plt.xlabel("# of Iterations")
        plt.ylabel("Sum of Squared Solution Error")
        plt.title(titleStr)
        plt.grid()
        plt.show()

    elif argList[1] == "--SORCA":
        # tests for Successive Over-Relaxation iteration with Chebyshev Acceleration
        if not len(argList) == 4:
            print("Must specify boundary condition (P for periodic, NP for non-periodic) and system size N.")
            print("Example usage: 'python iterative_solvers.py --SORCA NP 65'")
            sys.exit()
        elif argList[2] not in ["P", "NP"]:
            print("Must specify boundary condition as either P for periodic or NP for non-periodic (zero boundary).")
            sys.exit()
        elif int(argList[3]) < 14:
            print("Must specify number of grid points as at least 14 in order to avoid matplotlib broadcasting errors.")
            sys.exit()

        if argList[2] == "P":
            BC = "Poisson2DPeriodic"
            titleStr = "Convergence of SOR Method with Chebyshev Accel., N = " + argList[3] + ", Periodic BC"
        elif argList[2] == "NP":
            BC = "Poisson2DZeroBoundary"
            titleStr = "Convergence of SOR Method with Chebyshev Accel., N = " + argList[3] + ", Zero BC"

        # set number of iterations and call runner function
        iterList = np.array([2 ** i for i in range(5, 12)], dtype=int)
        errList = runIterativeMethod(int(argList[3]), iterList, method="SORCA", bc=BC, plotExtra=True)

        # plot the error against number of iterations
        plt.semilogy(iterList, errList)
        plt.xlabel("# of Iterations")
        plt.ylabel("Sum of Squared Solution Error")
        plt.title(titleStr)
        plt.grid()
        plt.show()

    elif argList[1] == "--methodcomparison":
        # Run and compare convergence of each method
        if not len(argList) == 4:
            print("Must specify boundary condition (P for periodic, NP for non-periodic) and system size N.")
            print("Example usage: 'python iterative_solvers.py --SORCA NP 65'")
            sys.exit()
        elif argList[2] not in ["P", "NP"]:
            print("Must specify boundary condition as either P for periodic or NP for non-periodic (zero boundary).")
            sys.exit()
        elif int(argList[3]) < 14:
            print("Must specify number of grid points as at least 14 in order to avoid matplotlib broadcasting errors.")
            sys.exit()

        if argList[2] == "P":
            BC = "Poisson2DPeriodic"
            titleStr = "Method Convergences, N = " + argList[3] + ", Periodic BC"
        elif argList[2] == "NP":
            BC = "Poisson2DZeroBoundary"
            titleStr = "Method Convergences, N = " + argList[3] + ", Zero BC"

        # set number of iterations and call runner function
        iterList = np.array([2 ** i for i in range(5, 12)], dtype=int)
        errListJacobi = runIterativeMethod(int(argList[3]), iterList, method="jacobi", bc=BC, plotExtra=False)
        errListGS = runIterativeMethod(int(argList[3]), iterList, method="GS", bc=BC, plotExtra=False)
        errListSOR = runIterativeMethod(int(argList[3]), iterList, method="SOR", bc=BC, plotExtra=False)
        errListSORCA = runIterativeMethod(int(argList[3]), iterList, method="SORCA", bc=BC, plotExtra=False)

        # plot convergence
        plt.semilogy(iterList, errListJacobi, '--', label="Jacobi")
        plt.semilogy(iterList, errListGS, '--', label="GS")
        plt.semilogy(iterList, errListSOR, '--', label="SOR")
        plt.semilogy(iterList, errListSORCA, '--', label="SOR w/ Chebyshev")
        plt.xlabel("# of Iterations")
        plt.ylabel("Sum of Squared Solution Error")
        plt.title(titleStr)
        plt.grid()
        plt.legend()
        plt.show()

    else:
        print("Test keyword '" + argList[1] + "' not recognized.")
        print("Valid tests:")
        print("  '--convergence N' - run tests checking whether a few different iterative methods will converge for a")
        print("                      few Poisson cases. N is the system size.")
        print("  '--jacobi bc N' - test accuracy of the Jacobi method with increasing iterations. bc is the boundary")
        print("                    condition mode (P for periodic, NP for non-periodic), and N is the system size.")
        print("  '--GS bc N' - test accuracy of the Gauss-Seidel method with increasing iterations. bc is the boundary")
        print("                condition mode (P for periodic, NP for non-periodic), and N is the system size.")
        print("  '--SOR bc N' - test accuracy of the Successive Over-Relaxation method with increasing iterations.")
        print("                 bc is the boundary condition mode (P for periodic, NP for non-periodic), and N is")
        print("                 the system size.")
        print("  '--SORCA bc N' - test accuracy of the Successive Over-Relaxation method with Chebyshev Acceleration")
        print("                   for increasing iterations. bc is the boundary condition mode (P for periodic, NP")
        print("                   for non-periodic), and N is the system size.")
        print("  '--methodcomparison bc N' - test the convergence rate of each of the four iteration methods for a")
        print("                              given boundary condition bc and system size N.")
        print()
        print("Example usage: 'python iterative_solvers.py --convergence 10'")
