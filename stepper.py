# File for performing the time-stepping on a grid
import numpy as np
from classes import Particle, Grid, qEle

def ChargeAssignmentStep(g, Nc):
    # Dimensionless Charge Assignment, see pg. 34 of Hockney & Eastwood

    # Initialize charge density accumulators
    for p in range(g.Ng-1):
        g.Charge[p] = -Nc

    # Accumulate charge density
    for j in range(len(g.Particles)):
        # locate nearest mesh point
        p = int(np.round(g.Particles[j].x[-1]))

        # increment charge density
        g.Charge[p] += 1

    # scale charge densities
    for p in range(g.Ng-1):
        g.Charge[p] = g.Charge[p] * g.wp()**2 * g.dt**2 / (2 * Nc)

def PoissonStep(g):
    # Solve Poisson's equation to get potential at every point on the grid.
    # See pg. 35 of Hockney & Eastwood. Note that the reference potential
    # is chosen such that the potential is 0 at the 0th grid point.

    # Eq. 2.56, compute potential at mesh point 1
    g.Potential[1] = np.sum([p*g.Charge[p] for p in range(g.Ng-1)])/g.Ng

    # compute potential at mesh point 2
    g.Potential[2] = g.Charge[1] + 2*g.Potential[1]

    # Compute remaining potentials
    for p in range(3, g.Ng - 1):
        g.Potential[p] = g.Charge[p-1] + 2*g.Potential[p-1] - g.Potential[p-2]

def EFieldStep(g):
    # Calculate the electric field at every point on the grid using known potentials.
    # Eq. 2-34, pg 32 of Hockney & Eastwood
    g.EField[0] = g.Potential[1] - g.Potential[-1]
    for p in range(g.Ng - 2):
        g.EField[p] = g.Potential[p-1] - g.Potential[p+1]
    g.EField[-1] = g.Potential[0] - g.Potential[-2]

def ForceInterpStep(g):
    # Dimensionless force interpolation for each particle
    for prt in g.Particles:
        prt.a.append(0)
        for p in range(g.Ng - 1):
            prt.a[-1] += g.W(prt.x[-1] - p)*g.EField[p]

def vStep(g):
    # Time-step velocity for each particle
    for prt in g.Particles:
        prt.v.append(prt.v[-1] + prt.a[-1])

def xStep(g):
    # Time-step position for each particle
    for prt in g.Particles:
        prt.x.append(prt.x[-1] + prt.v[-1])

        # enforce periodicity
        if prt.x[-1] >= 999:
            prt.x[-1] -= 999

def DiscreteModelStep(g, Nc):
    # Perform a single step through the discrete model on Grid g
    # Charge Assignment
    ChargeAssignmentStep(g, Nc)

    # Field Equations
    PoissonStep(g)
    EFieldStep(g)

    # Force Interpolation
    ForceInterpStep(g)

    # Equations of Motion
    vStep(g)
    xStep(g)

def RunDiscreteModel(g):
    # Time-step the discrete model from 0 to g.T

    # First define relevant parameters
    rho0 = 1  # background charge density from fixed ions [C*m^-3]
    Ns = 10  # number of electrons per super-particle [m^-2]
    Nc = rho0 * g.H / (Ns * qEle)  # average number of super-particles per cell

    # TODO: Check the four constraints on pg. 32 of Hockney & Eastwood

    # Now perform time-stepping
    t = 0
    while t < g.T:
        DiscreteModelStep(g, Nc)
        t += g.dt

if __name__ == '__main__':
    # Define an example grid
    L = 1e-7  # Grid length [m]
    Ng = 1000  # number of cells in grid
    dt = 1e-5  # time-step size [s]
    T = 1e-3  # ending simulation time [s]
    n0 = 1e3  # Background ion density [m^-3]

    G = Grid(L, Ng, dt, T, n0)

    # Define some example particles and add them to the grid
    p1 = Particle(ID="1", x0=100, v0=0)
    p2 = Particle(ID="2", x0=101, v0=0)
    p3 = Particle(ID="3", x0=102, v0=0)
    p4 = Particle(ID="4", x0=101.5, v0=0)
    p5 = Particle(ID="5", x0=102.5, v0=0)

    G.addParticle(p1)
    G.addParticle(p2)
    G.addParticle(p3)
    G.addParticle(p4)
    G.addParticle(p5)

    # Run the simulation and plot the results
    RunDiscreteModel(G)
    G.plotState1D()
