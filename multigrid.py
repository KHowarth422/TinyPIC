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

def S(u, b):
    # Solution operator function. On a given grid level, the solution operator
    # returns a better approximation to the solution. In general this can be
    # any method for solving a linear system. This implementation will use a
    # red-black Gauss-Seidel sweep.
    return 0

def R(b):
    # Restriction operator function. On a given grid level, the restriction operator
    # returns a restriction on the current level's source vector, b, such that the
    # restriction corresponds to the source vector on a coarser grid. This implementation
    # uses a weighted average of each point and its eight neighbors along with the known
    # periodic boundary conditions of the system.

    # First reshape from vector form to square grid, and create coarser grid
    dimCurrent = int(np.sqrt(len(b)))
    b = b.reshape((dimCurrent, dimCurrent))
    dimNew = int((dimCurrent + 1)/2)
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
    return bCoarser

def T(u):
    # Interpolation operator function. On a given grid level, the interpolation
    # operator returns an interpolation of the current level's solution vector, u, such
    # that the interpolation corresponds to the solution vector on a finer grid. This
    # implementation uses bilinear interpolation for application to a 2-Dimensional grid.

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

def A(Ai_m1, dimNew):
    # Operator matrix constructor function. Given the differential operator matrix
    # on a coarser level, return the matrix reduced to a finer level.
    dimCurrent = Ai_m1.shape[0]
    return 0

def MultiStep(Ai_m1, ui, bi, i, g, vDown, vUp):
    # Recursive step of the multi-grid algorithm V-cycle.
    # Inputs:
    #    ui - The guess for the solution vector at the current grid resolution
    #    bi - The restriction of the source vector at the current grid resolution
    #    i - The current grid level. Level 0 is the finest, level g is the coarsest
    #    dims - a 1D numpy array of len(g+1) containing the system dimension to use at
    #           each grid level.
    #    vDown - The number of solution operator iterations to perform on the way down
    #    vUp - The number of solution operator iterations to perform on the way up
    # Outputs:
    #    uiNew - A better solution at the current grid level.
    #

    # Get A matrix at current grid level.
    Ai = A(Ai_m1)

    if i == g:
        # We are at the coarsest grid. Solve exactly.
        uiNew = np.linalg.solve(Ai, bi)
        return uiNew
    else:
        # We are not at the coarsest grid. Continue recursive iteration.
        # Improve solution guess at current grid level on the way down.
        for j in range(vDown):
            ui = S(ui, bi)

        # Compute residual at current grid level
        ri = bi - Ai@ui

        # Get increment on solution at current grid level by recursing to coarser grid.
        di = T(MultiStep(Ai, np.zeros(int((len(Ai)+1)/2)), R(ri), i+1, g, vDown, vUp))

        # Increment solution at current grid level
        ui += di

        # Improve solution guess on the way back down
        for j in range(vUp):
            ui = S(ui, bi)

        # Return accurate solution guess
        return ui

if __name__ == '__main__':
    # Testing routines for each function
    # 2D Bilinear interpolation test
    T_test = np.array([[1, 1, 1],
                       [1, 2, 1],
                       [1, 1, 1]])
    print("T_test:")
    print(T_test)
    print()
    print("T_test interpolated:")
    print(T(T_test.flatten()).reshape(5,5))
    print()

    # 2D Restriction operator test.
    print("Restriction of T(T_test):")
    print(R(T(T_test.flatten())).reshape(3,3))
