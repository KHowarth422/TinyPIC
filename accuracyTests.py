# File containing functions for testing the accuracy of the PIC code
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from classes import Particle1D, Grid1D
from stepper import RunDiscreteModel


def compareParticlePositions1D(G, GTrue):
    # Return the sum of absolute dimensionless distances between Grids G and GTrue. G and GTrue must contain
    # the same number of particles. GTrue is treated as the more accurate grid spacing for the purpose of
    # calculating relative error. This is mainly a helper function for gridMeshSpacingConvergence().
    distSum = 0
    for prt in range(len(G.Particles)):
        # Particle positions are in units of grid spacings, so must non-dimensionalize to compare
        distSum += (G.Particles[prt].x[-1]/G.Ng - GTrue.Particles[prt].x[-1]/GTrue.Ng)**2
    return distSum

def gridMeshSpacingConvergence1D(Ng, dt):
    # Run a mesh sizing convergence test for the PIC algorithm.
    # Using Ng as the smallest number of mesh points, run 10 simulations with finer and finer mesh sizes.
    # Accuracy is measured by comparing the final positions of each particle on the Grid to the
    # final positions of each particle on the Grid with twice the finest mesh spacing.
    # Simulations are run for 100 time-steps with identical initial conditions of 1000 particles.
    # The process is repeated three times using step sizes dt, dt/10, dt/100.
    # Inputs:
    #   Ng - A scalar float representing the lowest number of mesh points to use.
    #   dt - A scalar float representing the smallest time-step size to use.

    # First set up the coarsest grid, using 1000 particles and a grid length of 100.
    L = 100
    G = Grid1D(L, Ng, dt, T=100*dt)
    for i in range(1000):
        xi = np.random.normal(0.5*Ng,0.2*Ng)
        G.addParticle(Particle1D(ID=str(i), x0=xi, v0=0))

    # The given number of grid points is taken as the lowest. Consider 10 factors of 2
    # larger than this many points and 3 factors of 10 smaller than the given step size.
    Ng_list = np.array([Ng*2**n for n in range(10)])
    dt_list = np.array([dt/10**n for n in range(3)])

    # Simulate once with a grid with twice as many mesh points as the highest and one
    # tenth of the smallest time-step size.
    G_largest = G.__copy__()
    G_largest.updateNg(Ng_list[-1]*2)
    G_largest.updateDt(dt_list[-1])
    RunDiscreteModel(G_largest)

    # Now run simulation for each grid with larger and larger number of points, and compare error
    # in position to G_largest
    pos_errors = np.zeros((len(dt_list),len(Ng_list)),dtype=float)
    for Ngi in range(len(Ng_list)):
        for dti in range(len(dt_list)):
            Gi = G.__copy__()
            Gi.updateNg(Ng_list[Ngi])
            Gi.updateDt(dt_list[dti])
            RunDiscreteModel(Gi)
            pos_errors[dti,Ngi] = compareParticlePositions1D(Gi,G_largest)

    # Return all information
    return Ng_list, dt_list, pos_errors

def timeStepSizeConvergence1D(Ng, dt):
    # Run a time-step size convergence test for the PIC algorithm.
    # Using dt as the smallest time-step size, run 10 simulations with finer and finer step sizes.
    # Accuracy is measured by comparing the final positions of each particle on the Grid to the
    # final positions of each particle on the Grid with twice the finest step size.
    # Simulations are run for 100 time-steps with identical initial conditions of 1000 particles.
    # The process is repeated three times for Grids with numbers of points Ng, 10*Ng, 100*Ng.
    # Inputs:
    #   Ng - A scalar float representing the smallest number of mesh points to use.
    #   dt - A scalar float representing the smallest time-step size to use.
    # Outputs:
    #   dt_list - a 1D numpy array containing all time step sizes used
    #   Ng_list - a 1D numpy array containing all mesh sizes used
    #   pos_errors - a (10 by 3) 2D numpy array containing the total position error for each
    #                combination of time step size and mesh size

    # First set up the coarsest grid, using 1000 particles and a grid length of 100.
    L = 100
    G = Grid1D(L, Ng, dt, T=100*dt)
    for i in range(1000):
        xi = np.random.normal(0.5*Ng,0.2*Ng)
        G.addParticle(Particle1D(ID=str(i), x0=xi, v0=0))

    # The given time-step size is taken as the lowest. Consider 10 factors of 2
    # smaller than this time-step. Also consider 3 factors of 10 greater than
    # the smallest grid spacing.
    dt_list = np.array([dt/2**n for n in range(10)])
    Ng_list = np.array([Ng*10**n for n in range(3)])

    # Simulate once with a grid with half the smallest time-step size, and ten times
    #  # Todo: FIX THIS it is ending at a different time than other sims so obviously error
    #          will be huge
    G_smallest = G.__copy__()
    G_smallest.updateDt(dt_list[-1]/2)
    G_smallest.updateNg(Ng_list[-1]*10)
    RunDiscreteModel(G_smallest)

    # Now run simulation for each grid with smaller and smaller step size, and compare error
    # in position to G_smallest
    pos_errors = np.zeros((len(Ng_list),len(dt_list)),dtype=float)
    for Ngi in range(len(Ng_list)):
        for dti in range(len(dt_list)):
            Gi = G.__copy__()
            Gi.updateDt(dt_list[dti])
            Gi.updateNg(Ng_list[Ngi])
            RunDiscreteModel(Gi)
            pos_errors[Ngi,dti] = compareParticlePositions1D(Gi,G_smallest)

    # Return all information
    return dt_list, Ng_list, pos_errors

def energyConservationTest(Ng, dt):
    # Test for energy conservation.
    # Create a grid with Ng grid points and step size dt, create 100 particles, and
    # simulate for 100 time-steps.
    G = Grid1D(L=64, Ng=Ng, dt=dt, T=1000*dt)
    G.addParticle(Particle1D("1", x0=10.))
    G.addParticle(Particle1D("2", x0=26.))
    #for i in range(100):
    #    xi = np.random.normal(0.5*Ng,0.2*Ng)
    #    G.addParticle(Particle1D(ID=str(i), x0=xi, v0=0))



    # Run the model and return the total energy at each time-step
    RunDiscreteModel(G)
    return G.getTotalKineticEnergy() + G.PE

def NBodySimulationTimer(G, dt, T):
    # Given a Grid1D containing an array of Particle1Ds, run an N-body simulation
    # using the given parameters.
    # Inputs:
    #   G - Grid1D containing pre-populated array of particles, complete with initial conditions
    #   dt - Scalar containing the time-step size to use
    #   T - Scalar containing the time to end simulation
    # Outputs:
    #   runtime - Scalar containing the wall-clock time required to run the simulation

    start = time.perf_counter()
    t = 0
    while t < T:
        # Calculate the net force on each particle by checking the Couloumb force between
        # each pair of particles. Store forces in a vector.
        forceVector = np.zeros_like(G.Particles)
        for i in range(len(G.Particles)-1): # For each particle...
            for j in range(i+1, len(G.Particles)): # For each unique pair of particles...
                # Note - since this simulation is meant only to compare runtime and not accuracy,
                # the force will be arbitrarily scaled down.
                r_ij = G.Particles[i].x[-1] - G.Particles[j].x[-1]
                F_ij = G.C["eChg"]**2/r_ij**2

                # As all particles are electrons, the forces are repulsive
                if G.Particles[i].x[-1] > G.Particles[j].x[-1]:
                    forceVector[i] += F_ij
                    forceVector[j] -= F_ij
                else:
                    forceVector[j] += F_ij
                    forceVector[i] -= F_ij

        # Once forces are calculated, time-step each particle with Forward Euler
        for i in range(len(G.Particles)):
            # Time-stepping
            prt = G.Particles[i]
            prt.v.append(prt.v[-1] + dt*forceVector[i])
            prt.x.append(prt.x[-1] + dt*prt.v[-1])

            # Enforce periodicity
            # Roughly translate position outside the right end of the grid
            if int(np.round(prt.x[-1])) > G.Ng or int(np.round(prt.x[-1])) < 0:
                prt.x[-1] = prt.x[-1] % G.Ng

        # Increment timer
        t += dt

    end = time.perf_counter()
    return end-start

def NBodyRuntimeComparison(Np, Ng):
    # Test PIC runtime compared to direct N-body simulation runtime.
    # This test is not intended to compare the physical accuracy of each method, but
    # rather to compare the time complexity with increasing numbers of particles. Similarly
    # to the mesh spacing convergence test, the user provides the smallest number of particles,
    # and we run 10 simulations with more and more particles, tracking the wall clock time
    # of each simulation for both algorithms. Since time-step affects only accuracy and not
    # runtime, it will be kept constant and small, and both simulations will run for 10 time-steps.
    # Inputs:
    #   Np - Scalar value for the minimum number of particles to simulate with.
    #   Ng - 1D np.array containing the number of mesh points to use for PIC. If multiple
    #        are provided, simulations will be run on grids of each size.
    # Outputs:
    #   Np_list - 1D np.array containing each number of particles simulated with.
    #   NBodyRunTimes - 1D np.array containing run-times of the N-Body case for each number of particles
    #   PICRunTimes - len(Ng) by len(Np_list) 2D np.array containing run-times of the PIC algorithm with
    #                 varying numbers of grid points and particles.

    # Initialize a base grid
    dt = 0.025
    BaseGrid = Grid1D(L=100, Ng=100, dt=dt, T=10*dt)

    # Create array of numbers of particle to use
    Np_list = np.array([Np*2**n for n in range(8)])

    # Create arrays to store run-times
    NBodyRunTimes = np.zeros_like(Np_list, dtype=float)
    PICRunTimes = np.zeros((len(Ng), len(Np_list)), dtype=float)

    for i in range(len(Np_list)): # for each number of particles...
        # Reset and populate the Base Grid
        Npi = Np_list[i]
        BaseGrid.Particles = np.array([],dtype=Particle1D)
        for prt in range(Npi):
            xi = np.random.normal(0.5 * 100, 0.2 * 100)
            BaseGrid.addParticle(Particle1D(ID=str(prt), x0=xi, v0=0))

        # Run the N-Body case
        NBodyRunTimes[i] = NBodySimulationTimer(BaseGrid.__copy__(), dt, 10*dt)

        # Run for each PIC case
        for j in range(len(Ng)):
            # Create a grid copy with all particles, and update the number of grid points
            TempGrid = BaseGrid.__copy__()
            TempGrid.updateNg(Ng[j])

            # Run and time simulation
            start = time.perf_counter()
            RunDiscreteModel(TempGrid)
            end = time.perf_counter()

            # Record runtime
            PICRunTimes[j,i] = end - start

    return Np_list, NBodyRunTimes, PICRunTimes

def verifyTwoStreamInstability():
    # Show that the two-stream instability can be recreated in the PIC algorithm.
    # The two-stream instability is a kinetic instability that can occur as a result of
    # streams of particles flowing through one another. This function sets up a Grid1D
    # and simulates with just such a scenario.
    # Outputs:
    #    GUnstable - a Grid1D containing the results of simulating a two-stream instability.

    # Create the Grid
    L = 100
    Ng = 100
    dt = 0.25
    GUnstable = Grid1D(L=L, Ng=Ng, dt=dt, T=300*dt)

    # Create two beams of electrons distributed uniformly throughout the Grid.
    # One beam has positive initial velocity, the other has negative.
    vBeam = 3
    particlesPerBeam = 1000
    for p in range(particlesPerBeam):
        # Beam with positive velocity
        x1 = np.random.uniform(-0.5, Ng-0.5)
        prt1 = Particle1D(ID=str(p)+"_right",x0=x1,v0=vBeam)
        GUnstable.addParticle(prt1)

        # Beam with negative velocity
        x2 = np.random.uniform(-0.5, Ng - 0.5)
        prt2 = Particle1D(ID=str(p) + "_left", x0=x2, v0=-vBeam)
        GUnstable.addParticle(prt2)

    # Run the simulation and return the Grid
    RunDiscreteModel(GUnstable)
    return GUnstable

if __name__ == '__main__':
    argList = sys.argv
    #argList = ['accuracyTests.py', '--runtime','100','100','200','400']
    if len(argList) == 1:
        print("Must use an extra argument to specify a test.")
        print("  '--mesh Ng dt' - mesh size convergence test with number of grid points Ng and step size dt")
        print("  '--timestep Ng dt' - time-step size convergence test with number of grid points Ng and step size dt")
        print("  '--energy Ng dt' - energy conservation test with number of grid points Ng and step size dt")
        print("  '--runtime Np Ng1 Ng2 ... Ngn' - runtime comparison test between PIC and N-body algorithms.")
        print("                                   Np is the least number of particles to run with, and Ng1 through Ngn")
        print("                                   are the numbers of grid points to run PIC with.")
        print("  '--twostream' - test for existence of the two-stream instability.")
        print()
        print("Example usage: 'python accuracyTests.py --mesh 100 0.025'")
    elif argList[1] == "--mesh":
        # Run a convergence test and plot the results. NOTE: Usually takes a few minutes to run.
        if not len(argList) == 4:
            print("Must provide Ng and dt arguments.")
            print("Example usage: 'python accuracyTests.py --mesh 100 0.025'")
            sys.exit()

        # Parse inputs and run convergence test
        Ng0 = int(argList[2])
        dt = float(argList[3])
        Ng_list, dt_list, pos_errors = gridMeshSpacingConvergence1D(Ng0, dt)

        # Plot results
        for dti in range(len(dt_list)):
            plt.loglog(Ng_list, pos_errors[dti], 'o--',label="dt = "+str(dt_list[dti]))
        plt.loglog(Ng_list, 1e-2 / Ng_list,  'r--', linestyle=(0, (1, 1)), linewidth=1, label="1e-2/Ng")
        plt.legend(loc="center right")
        plt.xlabel("Number of Grid Points")
        plt.ylabel("Sum of Squared Dimensionless Position Errors")
        plt.title("Convergence with Number of Grid Points, 1D Case")
        plt.grid()
        plt.show()
    elif argList[1] == "--timestep":
        # Run a convergence test and plot the results. NOTE: Usually takes a few minutes to run.
        if not len(argList) == 4:
            print("Must provide Ng and dt arguments.")
            print("Example usage: 'python accuracyTests.py --timestep 10000 0.25'")
            sys.exit()

        # Parse inputs and run convergence test
        Ng = int(argList[2])
        dt0 = float(argList[3])
        dt_list, Ng_list, pos_errors = timeStepSizeConvergence1D(Ng, dt0)

        # Plot results
        for Ngi in range(len(Ng_list)):
            plt.loglog(dt_list, pos_errors[Ngi], 'o--',label="Ng = "+str(Ng_list[Ngi]))
        plt.loglog(dt_list, 1e9*dt_list**4, 'r--', linestyle=(0, (1, 1)), linewidth=1, label="1e9*dt^4")
        plt.legend()
        plt.xlabel("Time-step Size")
        plt.ylabel("Sum of Squared Dimensionless Position Errors")
        plt.title("Convergence with Time-step Size, 1D Case")
        plt.grid()
        plt.show()
    elif argList[1] == "--energy":
        # Run energy conservation test
        if not len(argList) == 4:
            print("Must provide Ng and dt arguments.")
            print("Example usage: 'python accuracyTests.py --energy 64 0.25'")
            sys.exit()

        # Parse inputs and run energy conservation test
        Ng = int(argList[2])
        dt = float(argList[3])
        TotalEnergy = energyConservationTest(Ng,dt)

        # Plot results
        plt.semilogy(range(len(TotalEnergy)-1),TotalEnergy[:-1],'.')
        plt.xlabel("Time-step")
        plt.ylabel("Total Energy")
        plt.title("Total System Energy over Time, dt = "+str(dt))
        plt.ylim([40, 20000])
        plt.grid()
        plt.show()
    elif argList[1] == "--runtime":
        # Runtime comparison test
        if len(argList) <= 3:
            print("Must provide Np and at least one Ng argument.")
            print("Example usage: 'python accuracyTests.py --runtime 100 1000 10000 100000 1000000'")
            sys.exit()

        # Parse inputs and run test
        Np = int(argList[2])
        Ng_list = np.zeros(len(argList)-3,dtype=int)
        for i in range(3,len(argList)):
            Ng_list[i-3] = int(argList[i])
        Np_list, NBodyRunTimes, PICRunTimes = NBodyRuntimeComparison(Np, Ng_list)

        # N-body Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.loglog(Np_list, NBodyRunTimes, 'b--o', linestyle=(0, (4, 4)), linewidth=3, label='N-Body')
        ax1.loglog(Np_list, 1e-5*Np_list**2, 'b--', linestyle=(0, (1, 1)), linewidth=1, label='C*N^2')
        ax1.legend()
        ax1.set_xlabel('Number of Particles')
        ax1.set_ylabel('Runtime [s]')
        ax1.set_title("N-Body Runtime")
        ax1.grid()

        # PIC Plots
        for i in range(len(Ng_list)):
            ax2.loglog(Np_list, PICRunTimes[i], '--^', linestyle=(0, (3, 6)), linewidth=2,
                       label="Ng = "+"{:.0e}".format(Ng_list[i]))
        ax2.loglog(Np_list, 1e-4 * Np_list, 'r--', linestyle=(0, (1, 1)), linewidth=1, label='C*N')
        ax2.legend()
        ax2.set_xlabel('Number of Particles')
        ax2.set_ylabel('Runtime [s]')
        ax2.set_title("PIC Runtime")
        ax2.grid()
        ax2.tick_params(labelleft=True)
        plt.setp(ax2.get_yticklabels(), visible=True)
        plt.tight_layout()
        plt.show()
    elif argList[1] == "--twostream":
        if not len(argList) == 2:
            print("No keyword arguments accepted for '--twostream' test.")
            print("Example usage: 'python accuracyTests.py --twostream'")
            sys.exit()
        # Get grid with Particles becoming unstable over time
        GUnstable = verifyTwoStreamInstability()

        # Plot snapshots of particles in v-x space after 5, 15, 50 time-steps
        t1 = 5
        t2 = 50
        t3 = 300
        for t in [t1, t2, t3]:
            for prt in GUnstable.Particles:
                plt.plot(prt.x[t],prt.v[t],'b.')
            plt.xlabel('Position [Grid Spacings]')
            plt.ylabel('Velocity [Grid Spacings/Plasma Period]')
            plt.title("Stream Velocities, Timestep = "+str(t))
            plt.grid()
            plt.show()
        GUnstable.animateState(animateVelocity=True)
    else:
        print("Test keyword '"+argList[1]+"' not recognized.")
        print("Valid tests:")
        print("  '--mesh Ng dt' - mesh size convergence test with number of grid points Ng and step size dt")
        print("  '--timestep Ng dt' - time-step size convergence test with number of grid points Ng and step size dt")
        print("  '--energy Ng dt' - energy conservation test with number of grid points Ng and step size dt")
        print("  '--runtime Np Ng1 Ng2 ... Ngn' - runtime comparison test between PIC and N-body algorithms.")
        print("                                   Np is the least number of particles to run with, and Ng1 through Ngn")
        print("                                   are the numbers of grid points to run PIC with.")
        print("  '--twostream' - test for existence of the two-stream instability.")
        print()
        print("Example usage: 'python accuracyTests.py --mesh 100 0.025'")