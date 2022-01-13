# Author: Kevin Howarth                                                         #
# Date: 1/12/2022                                                               #
# File: classical_iterative_solvers.py                                          #
# Description: Implementation of a few classical iterative methods for solving  #
# large systems of linear equations, such as those arising in the               #
# finite-difference solution of the Poisson equation and other elliptic PDEs.   #
# Methods implemented include Jacobi, Gauss-Seidel, Successive Over-Relaxation, #
# and SOR with Chebyshev Acceleration. This file also includes a test routine   #
# to determine whether a classical iterative method will converge for a given   #
# system size and boundary condition. To run the test, simply call the file     #
# with a specified system size. For example:                                    #
#      >> python classical_iterative_solvers.py 65                              #
#################################################################################

import numpy as np
from numpy import linalg as LA
from iterative_helpers import get2DCenteredDifferenceMatrix
import sys

###########################################################
#             CLASSICAL METHOD-SPECIFIC TESTS             #
###########################################################

def convergenceTest(A, P):

    """

    Function for testing whether a classical iterative method will converge. Classical
    iterative methods are defined by a matrix splitting of a linear system:

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


###########################################################
#                    ITERATION METHODS                    #
###########################################################

def jacobiIter(b, h, bc, maxItr=5000):

    """

    Function implementing the Jacobi method to solve various system structures. The
    Jacobi method applies a simple iteration which arises from choosing the
    preconditioning matrix as the diagonal part of A. Thus the iteration can be written:

        P @ x = (P - A) @ x + b
        x = (I - inv(D) @ A) @ x + inv(D) @ b

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
    maxItr:  Scalar float specifying the number of iterations to use

    Raises
    -------
    ValueError:  ValueError is raised if an invalid boundary condition is supplied

    Returns
    -------
    x:  a 2D array containing the solution estimate of Ax = b obtained via Jacobi iteration

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

        P @ x = (P - A) @ x + b
        x = (I - inv(L)@A) @ x + inv(L) @ b

    Since different matrix structures of A will arise depending on the system being solved, the boundary conditions,
    etc., this function implements multiple use cases which each imply the structure of A. Each case will apply a finite
    difference discretization to the Poisson equation, thus a step size h is a required input.

    The Gauss-Seidel algorithm is improved upon by the Successive Over-Relaxation (SOR) method, which uses the
    heuristic that if the direction from x_k to x_k+1 is already a good direction to move, then we may as well move
    further in that direction. A boolean input allows the user to specify whether SOR should be used instead.

    Note that this implementation uses the red-black ordering scheme, and the SOR parameter omega (notated w) is chosen
    using a formula given in (Demmel, 1997). Alternatively, Chebyshev Acceleration as described in (Hockney and
    Eastwood, 1988) may be used to update omega at each iteration such that the initial increase in error often observed
    with SOR is mitigated.

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
    x:  a 2D array containing the solution estimate of Ax = b obtained via Gauss-Seidel iteration

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

    if not len(argList) == 1:
        print("Must use (only) one extra argument to specify the system size.")
        print("Example usage: 'python classical_iterative_solvers.py 65'")
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