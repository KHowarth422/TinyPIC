# File containing algorithms for Electrostatic 2D Particle-in-Cell simulations
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
from classes import Particle2D, Grid2D

def ChargeAssignmentStep(g, debug):
    # Dimensionless Charge Assignment, see pg. 34 of Hockney & Eastwood

    # Initialize charge density accumulators
    g.Charge = g.C["qBackground"]*np.ones_like(g.Charge)

    # Accumulate scaled charge density
    for j in range(len(g.Particles)):
        # locate nearest mesh point
        p0 = int(np.round(g.Particles[j].x_0[-1]))
        p1 = int(np.round(g.Particles[j].x_1[-1]))

        if debug:
            try:
                # increment charge density
                g.Charge[p0][p1] += g.C["delChg"]
            except IndexError:
                print("IndexError for particle with position:",g.Particles[j].x[-1])
        else:
            g.Charge[p0][p1] += g.C["delChg"]

    if debug:
        print("Check Charge Neutrality. Sum of charges:",np.sum(g.Charge))

def PoissonStep(g):
    A = np.zeros((g.Ng**2, g.Ng**2))
    for i in range(g.Ng**2):
        for j in range(g.Ng**2):
            if i == j: # set discretization coefficients with periodic BC
                A[i,j] = -4
                A[i,(j+1)%g.Ng**2] = 1
                A[i,(j+2)%g.Ng**2] = 1
                A[i,(j-1)%g.Ng**2] = 1 
                A[i,(j-2)%g.Ng**2] = 1
                break
        A[i,0] = 0 # enforce phi_{0,0} = 0

    rho = g.Charge.flatten()
    phi = np.linalg.solve(A,rho)
    g.Potential = np.reshape(phi,(g.Ng, g.Ng))


def EFieldStep(g):
    # Calculate the electric field at every point on the grid using known potentials.
    # Eq. 2-34, pg 32 of Hockney & Eastwood
    for i in range(g.Ng):
        for j in range(g.Ng):
            g.EField_0[i,j] = g.Potential[i,(j+1)%g.Ng] - g.Potential[i,(j-1)%g.Ng]

    for i in range(g.Ng):
        for j in range(g.Ng):
            g.EField_1[i,j] = g.Potential[(i+1)%g.Ng,j] - g.Potential[(i-1)%g.Ng,j]

def ForceInterpStep(g):
    # Dimensionless force interpolation for each particle
    for prt in g.Particles:
        # Get nearest mesh point
        pos0 = int(np.round(prt.x_0[-1]))
        pos1 = int(np.round(prt.x_1[-1]))

        # Take E-field at nearest mesh point
        prt.a_0.append(g.EField_0[pos0])
        prt.a_1.append(g.EField_1[pos1])

def vStep(g):
    # Time-step velocity for each particle
    for prt in g.Particles:
        prt.v_0.append(prt.v_0[-1] + prt.a_0[-1])
        prt.v_1.append(prt.v_1[-1] + prt.a_1[-1])

def xStep(g):
    # Time-step position for each particle
    # TODO: Could add error checking here - eg. if a particle has moved more than
    #       a few grid lengths in a single time-step, the simulation is probably unstable
    for prt in g.Particles:
        prt.x_0.append(prt.x_0[-1] + prt.v_0[-1])
        prt.x_1.append(prt.x_1[-1] + prt.v_1[-1])
        # Enforce periodicity
        # Translate position outside the right end of the grid
        prt.x_0[-1] = (prt.x_0[-1] + 1/2) % g.Ng - 1/2
        prt.x_1[-1] = (prt.x_1[-1] + 1/2) % g.Ng - 1/2

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
    debug = False

    # Define an example grid
    L = 128  # Grid length [m]
    Ng = 128  # number of cells in grid
    dt = 0.25  # time-step size [s]

    # Run for a few time-steps
    G = Grid2D(L=L, Ng=Ng, dt=0.25*dt, T=300*dt)

    # Define some example particles and add them to the grid
    p1 = Particle2D(ID="1", x0=0.1*Ng, v0=1)
    p2 = Particle2D(ID="2", x0=0.2*Ng, v0=0.6)
    p3 = Particle2D(ID="3", x0=0.3*Ng, v0=1)
    p4 = Particle2D(ID="4", x0=0.45*Ng, v0=-0.5)
    p5 = Particle2D(ID="5", x0=0.6*Ng, v0=2)
    p6 = Particle2D(ID="6", x0=0.8*Ng, v0=1)

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
    for i in range(10):
        xi = rs.normal(0.6,0.1)
        G.addParticle(Particle2D(ID=str(i+7), x0=xi*Ng, v0=0))

    # Define an alternate grid
    G2 = Grid2D(L=64, Ng=64, dt=0.25, T=100*0.25)
    G2.addParticle(Particle2D("1", x0=10., x1 = 10))
    G2.addParticle(Particle2D("2", x0=26., x1 = 26))
    # G2.addParticle(Particle2D("2alt", x0=26., v0=3.))  # Can uncomment to try with extra particles
    # G2.addParticle(Particle2D("3", x0=43.))  # Replace 2 with 2alt and 3 to recreate "3Particle 2D.mp4"

    # Run the simulation and plot some results
    # Run with G to test many-particle case
    # Run with G2 to recreate results on pg 36 of Ch2 of Hockney & Eastwood
    #    - dt=0.25 recreates Fig. 2.3a as well as "2Particle 2D.mp4"
    #    - dt=1.0 recreates Fig. 2.3b
    #    - dt=2.25 recreates Fig. 2.3c
    RunDiscreteModel(G2, debug=debug)
    #G2.plotCharge()
    #G2.plotPotential()
    #G2.plotEField()
    G2.plotState()
    #G.animateState()