# Author: Kevin Howarth                                                         #
# Date: 1/12/2022                                                               #
# File: iterative_runner.py                                                     #
# Description: Contains runner function and convergence test routines for all   #
# iterative solution methods developed in this directory. To see a list of all  #
# keywords and testing routines, simply run:                                    #
#      >> python iterative_runner.py                                            #
#################################################################################

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from iterative_helpers import plot3DSurface, get2DCenteredDifferenceMatrix
from classical_iterative_solvers import jacobiIter, gaussSeidelIter
from conjugate_gradient_solvers import conjugateGradient
import sys

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
    if method not in ["jacobi", "GS", "SOR", "SORCA", "CG"]:
        raise ValueError(("Please specify a valid iterative method. Valid methods include 'jacobi', 'GS', 'SORCA',"
                          " 'SOR', and 'CG'."))
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

    if bc == "Poisson2DPeriodic":
        # Adding a correction term to ensure a fair comparison between the direct and iterative solutions. Both satisfy
        # Poisson's equation, but the direct solution ends up with an extra constant factor added in.
        xExact = xExact - np.average(xExact)

    # Plot source and solution if specified
    if plotExtra:
        if bc == "Poisson2DZeroBoundary":
            plot3DSurface([bSource, xExact.reshape((NGrid, NGrid))], [plotTitleString1, plotTitleString2], bc="Poisson2DZeroBoundary")
        elif bc == "Poisson2DPeriodic":
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
        elif method == "CG":
            xIter = conjugateGradient(bSource, dx, bc=bc, maxItr=itrList[itr])

        #plot3DSurface([xIter, xExact],["CG, itr = "+str(itrList[itr]), "xExact"])

        # Take the squared 2 norm of the error as the error measure
        errorList[itr] = LA.norm(xIter.flatten() - xExact) ** 2

    return errorList

if __name__ == '__main__':

    argList = sys.argv
    #argList = ['iterative_runner.py', '--CG','NP','65']  # fake input for debugging

    if len(argList) == 1:
        print("Must use an extra argument to specify a test. Available tests include:")
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
        print("  '--classicalcomparison bc N' - test the convergence rate of each of the four classical iteration")
        print("                                 methods for given boundary condition bc and system size N.")
        print("  '--CG bc N' - test accuracy of the basic conjugate gradient algorithm with increasing iterations. bc")
        print("                is the boundary condition and N is the system size.")
        print()
        print("Example usage: 'python classical_iterative_solvers.py --jacobi NP 65'")

    elif argList[1] == "--jacobi":
        # tests for Jacobi iteration
        if not len(argList) == 4:
            print("Must specify boundary condition (P for periodic, NP for non-periodic) and system size N.")
            print("Example usage: 'python classical_iterative_solvers.py --jacobi NP 65'")
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
            print("Example usage: 'python classical_iterative_solvers.py --GS NP 65'")
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
            print("Example usage: 'python classical_iterative_solvers.py --SOR NP 65'")
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
            print("Example usage: 'python classical_iterative_solvers.py --SORCA NP 65'")
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

    elif argList[1] == "--classicalcomparison":
        # Run and compare convergence of each method
        if not len(argList) == 4:
            print("Must specify boundary condition (P for periodic, NP for non-periodic) and system size N.")
            print("Example usage: 'python classical_iterative_solvers.py --SORCA NP 65'")
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

    elif argList[1] == "--CG":
        # tests for Conjugate Gradient method
        if not len(argList) == 4:
            print("Must specify boundary condition (P for periodic, NP for non-periodic) and system size N.")
            print("Example usage: 'python classical_iterative_solvers.py --CG NP 65'")
            sys.exit()
        elif argList[2] not in ["P", "NP"]:
            print("Must specify boundary condition as either P for periodic or NP for non-periodic (zero boundary).")
            sys.exit()
        elif int(argList[3]) < 14:
            print("Must specify number of grid points as at least 14 in order to avoid matplotlib broadcasting errors.")
            sys.exit()

        if argList[2] == "P":
            BC = "Poisson2DPeriodic"
            titleStr = "Convergence of CG Method, N = " + argList[3] + ", Periodic BC"
        elif argList[2] == "NP":
            BC = "Poisson2DZeroBoundary"
            titleStr = "Convergence of CG Method, N = " + argList[3] + ", Zero BC"

        # set number of iterations and call runner function
        iterList = np.array([2**i for i in range(3, 10)], dtype=int)
        errList = runIterativeMethod(int(argList[3]), iterList, method="CG", bc=BC, plotExtra=True)

        # plot the error against number of iterations
        plt.semilogy(iterList, errList)
        plt.xlabel("# of Iterations")
        plt.ylabel("Sum of Squared Solution Error")
        plt.title(titleStr)
        plt.grid()
        plt.show()

    else:
        print("Test keyword '" + argList[1] + "' not recognized.")
        print("Valid tests:")
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
        print("  '--classicalcomparison bc N' - test the convergence rate of each of the four classical iteration")
        print("                                 methods for agiven boundary condition bc and system size N.")
        print()
        print("Example usage: 'python classical_iterative_solvers.py --jacobi NP 65'")