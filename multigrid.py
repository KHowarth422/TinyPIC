# Authors: Kyle Fridberg, Kevin Howarth, Hari Raval,                            #
# Course: AM 205                                                                #
# File: multigrid.py                                                            #
# Description: Implementation of the multigrid method, an iterative method      #
# for solving linear systems of equations. To be used for solving the Poisson   #
# equation on a 2-Dimensional grid. This implementation restricts uses to       #
# grid sizings with odd numbers of points for convenience with interpolation    #
# and restrictions, for now.                                                    #
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
import sys

########################################
########## OPERATOR FUNCTIONS ##########
########################################

def S(R_GS, R_GS2, ui, bi):
    # Solution operator function. On a given grid level, the solution operator
    # returns a better approximation to the solution. In general this can be
    # any method for iterating to solve a linear system.

    # Return an iteration
    return R_GS @ ui +  R_GS2 @ bi

def R(b):
    # Restriction operator function. On a given grid level, the restriction operator
    # returns a restriction on the current level's source vector, b, such that the
    # restriction corresponds to the source vector on a coarser grid. This implementation
    # uses a weighted average of each point and its eight neighbors along with the known
    # periodic boundary conditions of the system.

    # First reshape from vector form to square grid, and create coarser grid
    dimCurrent = int(np.sqrt(len(b)))
    b = b.reshape((dimCurrent, dimCurrent))
    dimNew = int(dimCurrent/2)
    bCoarser = np.zeros((dimNew, dimNew), dtype=float)

    # Define restriction kernel
    RKernel = np.array([[1/16, 1/8, 1/16],
                        [1/8, 1/4, 1/8],
                        [1/16, 1/8, 1/16]], dtype=float)

    # Take weighted averages
    for i in range(dimNew):
        for j in range(dimNew):
            idx_i = range(2*i-1, 2*i+2)
            idx_j = range(2*j-1, 2*j+2)
            bCoarser[i,j] = np.tensordot(RKernel, b.take(idx_i,mode='wrap', axis=0).take(idx_j,mode='wrap',axis=1))

    # Return coarser matrix
    return bCoarser.flatten()

def T(u):
    # Interpolation operator function. On a given grid level, the interpolation
    # operator returns an interpolation of the current level's solution vector, u, such
    # that the interpolation corresponds to the solution vector on a finer grid. This
    # implementation uses bilinear interpolation specific to a 2D grid with an even square
    # side length and periodic boundary conditions.

    # Get grid sizes and create new solution vector
    dimCurrent = int(np.sqrt(len(u)))
    dimNew = 2*dimCurrent
    uNew = np.zeros(dimNew**2, dtype=float)

    # Perform sweeps for bilinear interpolation. This method populates the interpolated
    # solution vector in O(dimNew*dimCurrent) runtime, instead of the much worse runtime
    # for direct matrix multiplication with the interpolation matrix.
    for j in range(dimNew):
        # Horizontal Sweep
        for i in range(dimCurrent):
            # Get relevant locations in new and old solution vectors
            idxN = 2 * i * dimNew + j
            idxN2 = dimNew + idxN
            idxC = int(j/2 + i*dimCurrent)
            idxC2 = idxC + dimCurrent if not i == (dimCurrent-1) else idxC % dimCurrent

            if j % 2 == 0:
                # This is a shared point, simply copy it directly
                uNew[idxN] = u[idxC]

                # These are the points being used for vertical interpolation
                uNew[idxN2] = u[idxC] / 2
                uNew[idxN2] += u[idxC2] / 2
            else:
                # Perform Horizontal linear interpolation
                idxC_upper = idxC + 1 if not j == (dimNew-1) else idxC-(dimCurrent-1)
                uNew[idxN] = 0.5*(u[idxC] + u[idxC_upper])

                # These are two points being used for bilinear interpolation
                uNew[idxN2] = 0.25 * (u[idxC] + u[idxC_upper])

                # These are the other two points being used for bilinear interpolation
                idxC2_upper = idxC2 + 1 if not j == (dimNew - 1) else idxC2 - (dimCurrent-1)
                uNew[idxN2] += 0.25 * (u[idxC2] + u[idxC2_upper])

    return uNew

def T1(u):
    # Alternate interpolation operator function. On a given grid level, the interpolation
    # operator returns an interpolation of the current level's solution vector, u, such
    # that the interpolation corresponds to the solution vector on a finer grid. This
    # implementation uses bilinear interpolation for application to a 2-Dimensional grid
    # with an odd square side length.

    # First reshape from vector form to square grid, and create finer grid
    dimCurrent = int(np.sqrt(len(u)))
    u = u.reshape((dimCurrent, dimCurrent))
    dimNew = int(2*dimCurrent - 1)
    uFiner = np.zeros((dimNew, dimNew),dtype=float)

    # Perform horizontal sweep over odd-numbered columns
    for j in range(dimNew):
        if j % 2 == 0:
            # Points at even indices are just copied directly from the coarser solution
            uFiner[::2,j] = u[:,int(j/2)]
        else:
            # Points at odd indices are linearly interpolated from the coarser solution
            jj = int((j - 1) / 2)
            uFiner[::2,j] = 0.5*(u[:,jj] + u[:,jj+1])

    # Perform vertical sweep, only over odd-numbered rows
    for i in range(1, dimNew, 2):
        # vertical linear interpolation
        ii = int((i - 1) / 2)
        uFiner[i,::2] = (u[ii,:] + u[ii + 1,:])/2

        # bilinear averaging over a given row
        for j in range(1, dimNew, 2):
            jj = int((j - 1) / 2)
            uFiner[i, j] = (u[ii, jj] + u[ii+1,jj] + u[ii, jj+1] + u[ii+1,jj+1])/4

    return uFiner.flatten()

####################################
########## MATRIX GETTERS ##########
####################################

def getT(dim):
    # Returns the interpolation matrix of size (dim**2, (2*dim)**2) for the periodic
    # boundary condition case with even grid square side lengths. This matrix interpolates
    # a periodic grid of side length dim to the same grid with side length 2*dim, with twice
    # as many points, using bilinear interpolation.

    dimNew = 2*dim

    # Create horizontal interpolation sub-matrix
    TE = np.zeros((dimNew, dim), dtype=float)
    for j in range(dim):
        i = 2*j
        TE[i, j] = 1.
        TE[i+1, j] = 0.5
        TE[i+1, (j+1)%dim] = 0.5

    # Create overall interpolation matrix
    T = np.zeros((dimNew**2, dim**2), dtype=float)
    for j in range(0,dim**2,dim):
        i = 4*j

        # Horizontal interpolation block
        T[i:i+dimNew,j:j+dim] = TE

        # Half of vertical/bilinear block, shift downward
        i = i + dimNew
        T[i:i + dimNew,j:j + dim] = 0.5 * TE

        if i == (dimNew**2 - dimNew):
            # Last row, must wrap around
            T[i:i + dimNew, 0:dim] = 0.5 * TE
        else:
            # Not last row, standard positioning
            T[i:i + dimNew, j + dim:j + 2*dim] = 0.5 * TE

    return T

def getR(dim):
    # Returns the restriction matrix of size ((2*dim)**2, dim**2) for the periodic
    # boundary condition case with even grid square side lengths. This matrix restricts
    # a periodic grid of side length dim to the same grid with side length dim/2, with half
    # as many points, using reverse bilinear interpolation.

    # Luckily, the restriction matrix is just one fourth of the transpose of the interpolation matrix
    return 0.25*getT(int(dim/2)).T

def A(Ai):
    # Operator matrix constructor function. Given the differential operator matrix
    # on a finer level, return the matrix reduced to a coarser level.
    dimFiner = int(np.sqrt(Ai.shape[0]))
    dimCoarser = int(dimFiner/2)
    Ti = getT(dimCoarser)
    Ri = getR(dimFiner)
    return Ri@Ai@Ti

#########################################
########## MULTIGRID FUNCTIONS ##########
#########################################

def MultiStep(A_list, ui, bi, biList, iList, i, g, vDown, vUp, R_GS_list, R_GS2_list, isFirst=False):
    # Recursive step of the multi-grid algorithm V-cycle.
    # Inputs:
    #    A_list - A list containing the A matrix needed at each grid resolution.
    #    ui - The guess for the solution vector at the current grid resolution
    #    bi - The source vector at the current resolution
    #    biList - The list containing the original source vector restriction at all resolutions.
    #    iList - The list containing the grid levels still to be visited. iList[i]
    #            gives the current grid level to operate on.
    #    i - the index of iList we are at.
    #    g - the index of the coarsest grid level
    #    vDown - The number of solution operator iterations to perform on the way down
    #    vUp - The number of solution operator iterations to perform on the way up
    #    isFirst - a bool describing whether this is the first time MultiStep is being called
    #
    # Outputs:
    #    uiNew - A better solution at the current grid level.
    #

    # Get A matrix at current grid level.
    Ai = A_list[iList[i]]

    if iList[i] == g:
        try:
            # We are at the coarsest grid. Solve exactly.
            ui = np.linalg.solve(Ai, bi)
        except np.linalg.LinAlgError:
            # Matrix is singular, use ten iterations of Gauss-Seidel instead
            for j in range(10):
                ui = S(R_GS_list[iList[i]], R_GS2_list[iList[i]], ui, bi)

        if isFirst:
            # Start the first v-cycle
            return T(MultiStep(A_list, T(ui), biList[-2], biList, iList, i+1, g, vDown, vUp, R_GS_list, R_GS2_list))
        else:
            # This was the bottom of a V-cycle, return result
            return ui
    elif i == len(iList)-1:
        # We have finished. Return the current solution guess.
        return ui
    else:
        # We are inside a V-cycle. Continue recursive iteration.
        # Improve solution guess at current grid level on the way down.
        for j in range(vDown):
            ui = S(R_GS_list[iList[i]], R_GS2_list[iList[i]], ui, bi)

        # Compute residual at current grid level
        ri = bi - Ai@ui

        # Get increment on solution at current grid level by recursing to coarser grid.
        di = T(MultiStep(A_list, np.zeros(int((len(ui))/4)), R(ri), biList, iList, i+1, g, vDown, vUp, R_GS_list, R_GS2_list))

        # Increment solution at current grid level
        ui += di

        # Improve solution guess on the way back up
        for j in range(vUp):
            ui = S(R_GS_list[iList[i]], R_GS2_list[iList[i]], ui, bi)

        if iList[i - 1] > iList[i] and iList[i + 1] > iList[i]:
            # We are actually at the top of a V-cycle and must start a new one
            # Depth of the V-cycle that just finished
            depth = g - iList[i]

            # Index for starting point of next V-cycle
            iNew = 2 * depth + 1
            if not i + iNew == len(iList):
                # It is not the final V-cycle
                return T(MultiStep(A_list, T(ui), biList[iList[i + iNew]], biList, iList, i + iNew, g, vDown, vUp, R_GS_list, R_GS2_list))
            else:
                # The final V-cycle is finished!
                return ui
        else:
            # Return accurate solution guess
            return ui

def MultiRunner(A0, b0, A_list=[], b_list=[], R_GS_list=[], R_GS2_list=[]):
    # Handler for recursive MultiStep function.
    # Inputs:
    #    A0 - Finite difference matrix for 2D grid
    #    b0 - Source term containing values on 2D grid

    # Get dimension of grid
    b0 = b0.flatten()
    dimFinest = int(np.sqrt(len(b0)))

    if not (dimFinest != 0) and (dimFinest & (dimFinest-1) == 0):
        # We require that the dimension is equal to 2^n, so error out if this is not the case
        sys.exit("Error: Please provide a grid size equal to 2^n for some n")

    # Construct a list of all grid sizes to use. We will solve on a grid no smaller than (4 x 4)
    dimList = np.array([dimFinest], dtype=int)
    dimNew = int(dimFinest/2)
    while dimNew >= 4:
        dimList = np.append(dimList, dimNew)
        dimNew = int(dimNew/2)

    # Now create a list of indices describing a full-multigrid cycle (FMG). An FMG is composed of
    # V-cycles of increasing size, starting at the coarsest grid. This list contains directions
    # informing the recursive algorithm of which levels it should go to and in what order.
    FMG_list = np.array([len(dimList) - 1],dtype=int)
    for i in range(len(dimList)-2, -1, -1):
        # V-cycles of increasing size
        for j in range(i, len(dimList)):
            # Back down to coarsest grid
            FMG_list = np.append(FMG_list, j)
        for j in range(len(dimList)-2, i-1, -1):
            # Back up to finer grid within same cycle
            FMG_list = np.append(FMG_list, j)

    # Since construction of Ai may be expensive if we do it at each step over multiple V-cycles,
    # we construct all A matrices ahead of time. A0 has dimension (dimFinest**2 x dimFinest**2)
    if not A_list:
        A_list.append(A0)
        for i in range(1, len(dimList)):
            A_list.append(A(A_list[-1]))

    # Similarly, we can construct the b vector at each spacing.
    if not b_list:
        b_list.append(b0)
        for i in range(1, len(dimList)):
            b_list.append(R(b_list[-1]))

    # Similarly, construction/inversion of the matrix splits for Gauss-Seidel may be expensive
    # if done at each step. We can build these ahead of time as well.
    if not R_GS_list:
        for i in range(len(dimList)):
            Ai = A_list[i]
            U = -np.triu(Ai)
            L = -np.tril(Ai)
            D = np.diag(np.diag(Ai))
            R_GS2_list.append(np.linalg.inv(D - L))
            R_GS_list.append(R_GS2_list[-1] @ U)

    # Call the Multigrid function
    return MultiStep(A_list, np.zeros(16), b_list[-1], b_list, FMG_list,
                     i=0, g=len(dimList)-1, vDown=2, vUp=2,
                     R_GS_list=R_GS_list, R_GS2_list=R_GS2_list, isFirst=True)

if __name__ == '__main__':
    # Testing routines for each function
    # 2D Bilinear interpolation test
    T_test = np.array([[1, 1, 1, 1],
                       [1, 2, 2, 1],
                       [1, 2, 2, 1],
                       [1, 1, 1, 1]])
    print("T_test:")
    print(T_test)
    print()
    print("T_test interpolated:")
    print(T(T_test.flatten()).reshape(8,8))
    print()

    # 2D Restriction operator test.
    print("Restriction of T(T_test):")
    print(R(T(T_test.flatten())).reshape(4,4))
    print()

    # Interpolation matrix test
    print("Interpolation matrix for 4x4 --> 8x8:")
    plt.spy(getT(4))
    #plt.show()
    print()

    # Test that Interpolation matrix produces same result as interpolation operator function
    T_test_res = T(T_test.flatten())
    T_matx_res = getT(4) @ T_test.flatten()
    print("Interpolation matrix multiplication produces same result as operator function:")
    print(np.array_equal(T_test_res, T_matx_res))
    print()

    # Restriction matrix test
    print("Restriction matrix for 8x8 --> 4x4:")
    plt.spy(getR(8))
    #plt.show()
    print()

    # Test that Restriction matrix produces same result as restriction operator function
    T_large = T(T_test.flatten())
    R_test_res = R(T_large)
    R_matx_res = getR(8) @ T_large
    print("Restriction matrix multiplication produces same result as operator function:")
    print(np.array_equal(R_test_res, R_matx_res))
    print()

    # Test Operate matrix restriction function
    print("Restrict A from managing an 8x8 --> 4x4:")
    A_test = np.kron(np.ones((2,2)), T_test)
    A_test = np.kron(np.ones((8,8)), A_test)
    #print(A(A_test))
    # Will take this as working

    # Test the runner
    N = 64
    b0 = np.ones((64,64))
    A0 = np.ones((64**2, 64**2))
    MultiRunner(A0, b0)
