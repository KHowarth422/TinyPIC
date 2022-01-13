# Author: Kevin Howarth                                                         #
# Date: 1/12/2022                                                               #
# File: conjugate_gradient_solvers.py                                           #
# Description: Implementation of conjugate gradient methods for solving         #
# large systems of linear equations, such as those arising in the               #
# finite-difference solution of the Poisson equation and other elliptic PDEs.   #
# To see a list of all keywords and testing routines, simply run:               #
#      >> python conjugate_gradient_solvers.py                                  #
#################################################################################

import numpy as np
from numpy import linalg as LA

def conjugateGradient(b, h, bc, maxItr=100, tol=1e-8):

    """

    Function implementing the standard Conjugate Gradient algorithm as presented in (Iserles, 2008). The method of
    conjugate gradients formulates the solution of a linear system Ax = b as an optimization problem on f(x), where
    the unique minimum of

        f(x) = 1/2 * x.T @ A @ x - b.T @ x

    is the solution of the aforementioned linear system. A naive algorithm for solving a multivariate optimization
    problem is to start with an initial guess, choose a step direction as the steepest gradient direction at the
    current point, taking a step of some size, and repeating. This method, known as steepest descent,  converges
    very slowly. This motivates the conjugate gradient method, which chooses the step direction such that subsequent
    directions d_(k) and d_(k-1) are conjugate to one another, meaning that

        d_(k).T @ A @ d_(k-1).T = 0

    This choice of step direction, along with a neat choice of step size based on the residual at a current step size
    r_(k) = A @ x_(k) - b, leads to greatly improved convergence.

    Note that this implementation is tailored to solving Poisson's equation on a 2D grid, with either zero-boundary
    or periodic boundary conditions.

    Parameters
    ----------
    b:  2D np.array of length n by n containing the source term for the Poisson equation
    h:  Scalar float specifying the mesh spacing of the computational grid
    bc:  String specifying which solution case to use. Available cases include
                - "Poisson2DZeroBoundary" - solves Poisson equation on a 2D grid with zero Dirichlet boundary conditions
                - "Poisson2DPeriodic" - solves Poisson equation on a 2D grid with periodic Dirichlet boundary conditions
    maxItr:  Scalar float specifying the number of iterations to use
    tol:  Scalar float specifying the tolerance value for the residual at which to stop iterating

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
    bFlat = b.flatten()

    # Initialize solution guess, residual, step direction, and intermediate v vector
    x = np.zeros_like(bFlat)
    r = bFlat.copy()
    rOld = r.copy()
    d = r.copy()
    v = np.zeros_like(bFlat)

    # perform iteration
    itr = 0
    while itr < maxItr:
        # check norm of residual for convergence. if less than tolerance, stop iterating
        rNorm = LA.norm(r)
        rNormOld = LA.norm(rOld)
        if rNorm < tol:
            break

        # compute step direction on all iterations past the first
        if itr > 0:
            beta = rNorm ** 2 / rNormOld ** 2
            d = r + beta * d

        # calculate the matrix-vector product v_(k) = A @ d_(k)
        # a multiplication rule is used to perform this operation in O(n^2) time instead of O(n^4).
        # this can be done because the structure of the matrix A is known based on the problem and boundary conditions.
        # additionally, padding the matrix form of d allows for convenient programming of the multiplication rule.
        if bc == "Poisson2DZeroBoundary":
            dGrid = np.pad(np.reshape(d, (n, n)), 1)
        else:
            dGrid = np.pad(np.reshape(d, (n, n)), 1, mode="wrap")

        v = np.reshape(v, (n,n))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                v[i - 1, j - 1] = -dGrid[i - 1, j] - dGrid[i + 1, j] - dGrid[i, j - 1] - dGrid[i, j + 1] + 4 * dGrid[i, j]
        v = v.flatten() / h ** 2

        # calculate the step size, omega (notated here as w)
        w = rNorm ** 2 / np.dot(d, v)

        # form the new solution iteration and residual, increase counter, and repeat
        x = x + w * d
        rOld = r.copy()
        r = r - w * v
        itr += 1

    return x
