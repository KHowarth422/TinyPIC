# Author: Kevin Howarth                                                         #
# Date: 1/17/2022                                                               #
# File: spectral_solvers.py                                                     #
# Description: Implementation of spectral methods for solving the Poisson       #
# equation on a 2D grid. The methods could easily be extended to other elliptic #
# PDEs, and with some effort to other classes of PDEs. In most cases periodic   #
# boundary conditions are required, although Chebyshev methods are able to      #
# retain spectral convergence while circumventing this restriction.             #
#################################################################################

import numpy as np

def spectralSolve(b):

    """

    Function implementing a standard spectral method as presented in (Iserles, 2008). Spectral methods are based on
    using the Fast Fourier Transform to find the Fourier coefficients of a sequence of discrete data. Then, integration
    may be performed as algebra in Fourier space to determine the Fourier coefficients of the corresponding solution
    term. The inverse FFT is then used to find the solution in real space.

    Spectral methods are advantageous due to the exponential decay in error with increasing number of grid points. The
    price of this incredible accuracy is that they are only applicable on systems with periodic boundary conditions.

    This function applies the spectral method specifically for the solution of Poisson's equation on a 2D grid with
    periodic boundary conditions. In other words, this function finds the solution u for which:

        Laplacian(u) = b

    Parameters
    ----------
    b:  2D np.array of length n by n containing the source term for the Poisson equation
    h:  Scalar float specifying the mesh spacing of the computational grid

    Raises
    -------
    None

    Returns
    -------
    u:  a 2D array containing the solution estimate of Laplacian(u) = b obtained via the spectral method

    """

    # get system size
    n = b.shape[0]

    # recover mesh spacing
    h = 2 / n

    # initialize Fourier coefficients of solution and source
    U = np.zeros_like(b, dtype=np.cdouble)
    B = np.fft.fft2(b)

    # get physical wavenumbers
    k = np.fft.fftfreq(n) / h

    # set (0, 0)th coefficient to 0, obeying normalization condition
    U[0, 0] = 0.

    # perform algebra to get remaining terms
    for k in range(n):
        for l in range(n):
            if k == 0 and l == 0:
                continue
            U[k, l] = -B[k, l] / (np.pi ** 2 * (k ** 2 + l ** 2))

    # invert the Fourier transform and return result
    u = np.fft.ifft2(U)
    return u
