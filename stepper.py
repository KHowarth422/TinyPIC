# File containing algorithms for Electrostatic 1D Particle-in-Cell simulations
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from classes import Particle, Grid

def ChargeAssignmentStep(g, debug):
    # Dimensionless Charge Assignment, see pg. 34 of Hockney & Eastwood

    # Initialize charge density accumulators
    g.Charge = g.C["qBackground"]*np.ones_like(g.Charge)

    # Accumulate scaled charge density
    for j in range(len(g.Particles)):
        # locate nearest mesh point
        p = int(np.round(g.Particles[j].x[-1]))

        if debug:
            try:
                # increment charge density
                g.Charge[p] += g.C["delChg"]
            except IndexError:
                print("IndexError for particle with position:",g.Particles[j].x[-1])
        else:
            g.Charge[p] += g.C["delChg"]

    if debug:
        print("Check Charge Neutrality. Sum of charges:",np.sum(g.Charge))

def PoissonStep(g):
    # Solve Poisson's equation to get potential at every point on the grid.
    # See pg. 35 of Hockney & Eastwood. Note that the reference potential
    # is chosen such that the potential is 0 at the 0th grid point.

    # Eq. 2.56, compute potential at mesh point 1
    # Slight subtlety here because in Python we are representing mesh points [0 to Ng-1], whereas
    # in the textbook/Fortran they use base 1 indexing and represent mesh point [1 to Ng]
    g.Potential[1] = (np.sum([p*g.Charge[p] for p in range(1,g.Ng)]) + g.Ng*g.Charge[0])/g.Ng

    # compute remaining potentials
    for p in range(2, g.Ng):
        g.Potential[p] = g.Charge[p-1] + 2.*g.Potential[p-1] - g.Potential[p-2]


def EFieldStep(g):
    # Calculate the electric field at every point on the grid using known potentials.
    # Eq. 2-34, pg 32 of Hockney & Eastwood
    for p in range(g.Ng - 1):
        g.EField[p] = g.Potential[p+1] - g.Potential[p-1]
    g.EField[-1] = g.Potential[0] - g.Potential[-2]

def ForceInterpStep(g):
    # Dimensionless force interpolation for each particle
    for prt in g.Particles:
        # Get nearest mesh point
        pos = int(np.round(prt.x[-1]))

        # Take E-field at nearest mesh point
        prt.a.append(g.EField[pos])

def vStep(g):
    # Time-step velocity for each particle
    for prt in g.Particles:
        prt.v.append(prt.v[-1] + prt.a[-1])

def xStep(g):
    # Time-step position for each particle
    # TODO: Could add error checking here - eg. if a particle has moved more than
    #       a few grid lengths in a single time-step, the simulation is probably unstable
    for prt in g.Particles:
        prt.x.append(prt.x[-1] + prt.v[-1])
        # Enforce periodicity
        # Translate position outside the right end of the grid
        while int(np.round(prt.x[-1])) >= g.Ng:
            prt.x[-1] -= g.Ng

        # Translate position outside the left end of the grid
        while int(np.round(prt.x[-1])) < 0:
            prt.x[-1] += g.Ng

def DiscreteModelStep(g, debug):
    # Perform a single step through the discrete model on Grid g
    # Charge Assignment
    if debug:
        print("Begin Charge Assignment")
    ChargeAssignmentStep(g, debug)

    # Field Equations
    if debug:
        print("Begin Poisson Step")
    PoissonStep(g)

    if debug:
        print("Begin EField Step")
    EFieldStep(g)

    # Force Interpolation
    if debug:
        print("Begin Force Interpolation")
    ForceInterpStep(g)

    # Equations of Motion
    if debug:
        print("Begin vStep")
    vStep(g)

    if debug:
        print("Begin xStep")
    xStep(g)

def RunDiscreteModel(g, debug=False):
    # Time-step the discrete model from 0 to g.T

    # TODO: Kevin add checking for the four constraints on pg. 32 of Hockney & Eastwood

    # Perform time-stepping
    t = 0
    while t*g.dt < g.T:
        if debug:
            print()
            print("Start step",t)
        DiscreteModelStep(g, debug)
        t += 1

if __name__ == '__main__':
    # toggle debug mode
    debug = True

    # Define an example grid
    L = 64  # Grid length [m]
    Ng = 64  # number of cells in grid
    dt = 0.25  # time-step size [s]

    # Run for a few time-steps
    G = Grid(L=L, Ng=Ng, dt=0.25*dt, T=100*dt)

    # Define some example particles and add them to the grid
    p1 = Particle(ID="1", x0=0.5*Ng, v0=0)
    p2 = Particle(ID="2", x0=0.51*Ng, v0=0)
    p3 = Particle(ID="3", x0=0.52*Ng, v0=0)
    p4 = Particle(ID="4", x0=0.49*Ng, v0=0)
    p5 = Particle(ID="5", x0=0.48*Ng, v0=0)
    p6 = Particle(ID="6", x0=0.53*Ng, v0=0)

    G.addParticle(p1)
    G.addParticle(p2)
    G.addParticle(p3)
    G.addParticle(p4)
    G.addParticle(p5)
    G.addParticle(p6)

    # TODO: Hari simplify this please
    # set random state for reproducibility when debugging
    if debug:
        rs = RandomState(MT19937(SeedSequence(123456789)))
    else:
        rs = RandomState()

    # Add a big random distribution of particles
    for i in range(100):
        xi = rs.normal(0.6,0.05)
        G.addParticle(Particle(ID=str(i+7), x0=xi*Ng, v0=0))

    # Define an alternate grid
    G2 = Grid(L=64, Ng=64, dt=0.25, T=100*0.25)
    G2.addParticle(Particle("1", x0=10.))
    G2.addParticle(Particle("2", x0=26.))
    # G2.addParticle(Particle("2alt", x0=26., v0=3.))  # Can uncomment to try with extra particles
    # G2.addParticle(Particle("3", x0=43.))  # Replace 2 with 2alt and 3 to recreate "3Particle 1D.mp4"

    # Run the simulation and plot some results
    # Run with G to test many-particle case
    # Run with G2 to recreate results on pg 36 of Ch2 of Hockney & Eastwood
    #    - dt=0.25 recreates Fig. 2.3a as well as "2Particle 1D.mp4"
    #    - dt=1.0 recreates Fig. 2.3b
    #    - dt=2.25 recreates Fig. 2.3c
    RunDiscreteModel(G2, debug=debug)
    G2.plotCharge1D()
    G2.plotPotential1D()
    G2.plotEField1D()
    G2.plotState1D()
    # G2.animateState1D()
