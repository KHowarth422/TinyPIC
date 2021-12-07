# File containing functions for testing the accuracy of the PIC code
import numpy as np
import matplotlib.pyplot as plt
from classes import Particle1D, Grid1D
from stepper import RunDiscreteModel
import sys

def compareParticlePositions1D(G, GTrue):
    # Return the sum of absolute dimensionless distances between Grids G and GTrue. G and GTrue must contain
    # the same number of particles. GTrue is treated as the more accurate grid spacing for the purpose of
    # calculating relative error. This is mainly a helper function for gridMeshSpacingConvergence().
    distSum = 0
    for prt in range(len(G.Particles)):
        # Particle positions are in units of grid spacings, so must non-dimensionalize to compare
        distSum += np.abs(G.Particles[prt].x[-1]/G.Ng - GTrue.Particles[prt].x[-1]/GTrue.Ng)
    return distSum

def gridMeshSpacingConvergence1D(Ng, dt):
    # Run a mesh sizing convergence test for the PIC algorithm.
    # Using Ng as the smallest number of mesh points, run 10 simulations with finer and finer mesh sizes.
    # Accuracy is measured by comparing the final positions of each particle on the Grid to the
    # final positions of each particle on the Grid with twice the finest mesh spacing.
    # Simulations are run for 100 time-steps of size dt with identical initial conditions of
    # 1000 particles.
    # Inputs:
    #   Ng - A scalar float representing the lowest number of mesh points to use
    #   dt - A scalar float representing the time-step to use for all simulations.

    # First set up the coarsest grid, using 1000 particles and a grid length of 100.
    L = 100
    G = Grid1D(L, Ng, dt, T=100*dt)
    for i in range(1000):
        xi = np.random.normal(0.5*Ng,0.2*Ng)
        G.addParticle(Particle1D(ID=str(i), x0=xi, v0=0))

    # The given number of grid points is taken as the lowest. Consider 10 factors of 2
    # larger than this many points.
    Ng_list = np.array([Ng*2**n for n in range(10)])

    # Simulate once with a grid with twice as many mesh points as the highest. Must update some
    # grid parameters when changing the mesh spacing.
    G_largest = G.__copy__()
    G_largest.updateNg(Ng_list[-1]*2)
    RunDiscreteModel(G_largest)

    # Now run simulation for each grid with larger and larger number of points, and compare error
    # in position to G_largest
    pos_errors = np.zeros_like(Ng_list,dtype=float)
    for i in range(len(Ng_list)):
        Gi = G.__copy__()
        Gi.updateNg(Ng_list[i])
        RunDiscreteModel(Gi)
        pos_errors[i] = compareParticlePositions1D(Gi,G_largest)

    # Return the list of grid point numbers and the list of total position errors
    return Ng_list, pos_errors

def energyConservationTest(Ng, dt):
    # Test for energy conservation.
    # Create a grid with Ng grid points and step size dt, create 1000 particles, and
    # simulate for 100 time-steps.
    G = Grid1D(L=100, Ng=Ng, dt=dt, T=100*dt)
    for i in range(1000):
        xi = np.random.normal(0.5*Ng,0.2*Ng)
        G.addParticle(Particle1D(ID=str(i), x0=xi, v0=0))

    # Run the model and return the total kinetic energy at each time-step
    RunDiscreteModel(G)
    return G.getTotalKineticEnergy()

if __name__ == '__main__':
    argList = sys.argv
    if len(argList) == 1:
        print("Must use an extra argument to specify a test.")
        print("  '--mesh Ng dt' - run mesh size convergence test with number of grid points Ng and step size dt")
        print("  '--energy Ng dt' - run energy conservation test with number of grid points Ng and step size dt")
        print()
        print("Example usage: 'python accuracyTests.py --mesh 100 0.025'")
    elif argList[1] == "--mesh":
        # Run a convergence test and plot the results. NOTE: Usually takes a few minutes to run.
        if not len(argList) == 4:
            print("Must provide Ng and dt arguments.")
            print("Example usage: 'python accuracyTests.py --mesh 100 0.025'")
        Ng0 = int(argList[2])
        dt = float(argList[3])
        Ng_list, pos_errors = gridMeshSpacingConvergence1D(Ng0, dt)
        plt.loglog(Ng_list, pos_errors, 'o--',label="Errors")
        plt.loglog(Ng_list, 1e3 / Ng_list, '--',label="10^3/Ng")
        plt.legend()
        plt.xlabel("Number of Grid Points")
        plt.ylabel("Sum of Absolute Dimensionless Position Errors")
        plt.title("Convergence with Number of Grid Points, 1D Case, dt = "+str(dt))
        plt.grid()
        plt.show()
    elif argList[1] == "--energy":
        # Run energy conservation test
        if not len(argList) == 4:
            print("Must provide Ng and dt arguments.")
            print("Example usage: 'python accuracyTests.py --energy 100 0.25'")
        Ng = int(argList[2])
        dt = float(argList[3])
        TotalKE = energyConservationTest(Ng,dt)
        plt.plot(range(len(TotalKE)),TotalKE,'--o')
        plt.xlabel("Time-step")
        plt.ylabel("Total Kinetic Energy")
        plt.title("Total System Kinetic Energy over Time")
        plt.grid()
        plt.show()
    else:
        print("Accuracy test",argList[1],"not recognized.")
        print("Valid tests:")
        print("  '--mesh Ng dt' - run mesh size convergence test with number of grid points Ng and step size dt")
        print("  '--energy Ng dt' - run energy conservation test with number of grid points Ng and step size dt")
        print()
        print("Example usage: 'python accuracyTests.py --mesh 100 0.025'")