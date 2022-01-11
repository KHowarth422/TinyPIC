# Authors: Kyle Fridberg, Kevin Howarth, Hari Raval                             #
# Course: AM 205                                                                #
# File: stepper2D.py                                                            #
# Description: Primary algorithms for Electrostatic 2D                          #
# Particle-in-Cell simulations. This file actually runs the simulations by      #
# leveraging the classes created in classes2D.py                                #
#################################################################################

import numpy as np
from classes2D import Particle2D, Grid2D, C
from iterative_solvers import get2DCenteredDifferenceMatrix
import time

def ChargeAssignmentStep(g, debug):

    """

    Method to perform dimensionless Charge Assignment (pg. 34 of Hockney & Eastwood)

    Parameters
    ----------
    g: Grid2D object to perform charge assignments in
    debug : boolean variable indicating whether to enter debug mode to step through simulation diagnostics carefully

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # initialize charge density accumulators
    g.Charge = g.C["qBackground"] * np.ones_like(g.Charge)

    # accumulate scaled charge density
    for j in range(len(g.Particles)):
        # locate nearest mesh point
        px = int(np.round(g.Particles[j].x_0[-1]))
        py = int(np.round(g.Particles[j].x_1[-1]))

        if debug:
            try:
                # increment charge density
                g.Charge[py, px] += g.C["delChg"]
            except IndexError:
                print("IndexError for particle with position:", g.Particles[j].x[-1])
        else:
            g.Charge[py, px] += g.C["delChg"]

    # print sum of charges, if debug mode is activated
    if debug:
        print("Check Charge Neutrality. Sum of charges:", np.sum(g.Charge))


def PoissonStep(g):
  
    """

    Method to solve Poisson's equation to get potential at every point on the grid (pg. 35 of Hockney & Eastwood)

    ** NOTE 1: the reference potential is chosen such that the potential is 0 at the 0th grid point **

    ** NOTE 2: we use 0-based indexing so that we are representing mesh points [1 to Ng-1] (unlike Fortran)

    Parameters
    ----------
    g: Grid2D object to perform poisson steps on

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # finite difference matrix for 2D Poisson equation
    if not g.K:
        g.K = get2DCenteredDifferenceMatrix(g.Ng, 1, bc="Poisson2DPeriodic")

    # compute the updated potential by directly solving the linear system
    rho = g.Charge.flatten()
    phi = np.linalg.solve(g.K, rho)
    g.Potential = np.reshape(phi, (g.Ng, g.Ng))


def EFieldStep(g):

    """

    Method to calculate the electric field at every point on the grid using known potentials

    Parameters
    ----------
    g: Grid2D object to perform electric field calculations on

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # calculate electric field at every point on the grid using known potentials (Eq. 2-34, pg 32 of Hockney & Eastwood)
    for i in range(g.Ng):
        for j in range(g.Ng):
            g.EField_0[i, j] = g.Potential[i, (j + 1) % g.Ng] - g.Potential[i, (j - 1) % g.Ng]

    # calculate electric field at every point on the grid using known potentials (Eq. 2-34, pg 32 of Hockney & Eastwood)
    for i in range(g.Ng):
        for j in range(g.Ng):
            g.EField_1[i, j] = g.Potential[(i + 1) % g.Ng, j] - g.Potential[(i - 1) % g.Ng, j]


def ForceInterpStep(g):

    """

    Method to calculate the dimensionless force interpolation for each particle

    Parameters
    ----------
    g: Grid2D object to perform force interpolation calculations on

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # iterate over each particle in the grid to compute the force interpolation
    for prt in g.Particles:
        # get the nearest mesh point
        px = int(np.round(prt.x_0[-1]))
        py = int(np.round(prt.x_1[-1]))

        # extract the electric field at that nearest mesh point
        prt.a_0.append(g.EField_0[py, px])
        prt.a_1.append(g.EField_1[py, px])


def vStep(g):

    """

    Method to calculate the time-step velocity for each particle

    Parameters
    ----------
    g: Grid2D object to perform force interpolation calculations on

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # iterate over each particle in the grid to compute the time-step velocity
    for prt in g.Particles:
        prt.v_0.append(prt.v_0[-1] + prt.a_0[-1])
        prt.v_1.append(prt.v_1[-1] + prt.a_1[-1])


def xStep(g):

    """

    Method to calculate the time-step position for each particle

    Parameters
    ----------
    g: Grid2D object to perform force interpolation calculations on

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # iterate over each particle in the grid to compute the time-step position
    for prt in g.Particles:
        # enforce periodicity
        prt.x_0.append(prt.x_0[-1] + prt.v_0[-1])
        prt.x_1.append(prt.x_1[-1] + prt.v_1[-1])

        # translate the position outside the ends of the grid
        prt.x_0[-1] = (prt.x_0[-1] + 1 / 2) % g.Ng - 1 / 2
        prt.x_1[-1] = (prt.x_1[-1] + 1 / 2) % g.Ng - 1 / 2


def DiscreteModelStep(g, debug):

    """

    Method to perform a single step through the discrete model on Grid g

    Parameters
    ----------
    g: Grid2D object to perform force interpolation calculations on
    debug: boolean variable indicating whether to enter debug mode to step through simulation diagnostics carefully

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # print charge diagnostics if debug is set to TRUE
    if debug:
        print("Begin Charge Assignment")
    # perform charge assignments
    ChargeAssignmentStep(g, debug)

    # print field equation (poisson) diagnostics if debug is set to TRUE
    if debug:
        print("Begin Poisson Step")
        start = time.perf_counter()
    # perform poisson computations
    PoissonStep(g)
    if debug:
        end = time.perf_counter()
        print("Finished Poisson Step in "+str(end-start)+" s.")

    # print field equation (electric field) diagnostics if debug is set to TRUE
    if debug:
        end = time.perf_counter()
        print("Finished Poisson Step in " + str(end - start) + " s.")
    if debug:
        print("Begin EField Step")
    # perform electric field computations
    EFieldStep(g)

    # print force interpolation diagnostics if debug is set to TRUE
    if debug:
        print("Begin Force Interpolation")
    # perform force interpolation computations
    ForceInterpStep(g)

    # print equations of motion (velocity) diagnostics if debug is set to TRUE
    if debug:
        print("Begin vStep")
    # perform velocity computations
    vStep(g)

    # print equations of motion (position) diagnostics if debug is set to TRUE
    if debug:
        print("Begin xStep")
    # perform position computations
    xStep(g)


def RunDiscreteModel(g, debug=False):

    """

    Method to perform a time-step of the discrete model from 0 to g.T

    Parameters
    ----------
    g: Grid2D object to perform force interpolation calculations on
    debug: boolean variable indicating whether to enter debug mode to step through simulation diagnostics carefully

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # initialize time to time 0
    t = 0

    # perform the time-steps until the end simulation time is met
    while t * g.dt < g.T:
        # print diagnostics, if in debug-mode
        if debug:
            print()
            print("Start step", t)
        DiscreteModelStep(g, debug)
        t += 1


def check_particle_validity(particles):

    """

    Method to check the validity of the inputed particles to be simulated (as entered by the user)

    Parameters
    ----------
    particles: list of Particle2D objects to be simulated

    Raises
    -------
    TypeError: Type error is raised if objects in particles list are not all Particle2D
    ValueError: Value error is raised if an empty list of particles is passed in

    Returns
    -------
    None

    """

    # ensure the user is simulating at least one particle
    if particles is None or len(particles) < 1:
        raise ValueError("ERROR: At least one Particle2D object must be passed in")

    # iterate over each particle to ensure it is of Particle2D type, otherwise raise an illegal type error
    for particle in particles:
        if not isinstance(particle, Particle2D):
            raise TypeError("ERROR: All input particles must be Particle2D objects")

    # warn the user of simulating too few particles but do not raise an error
    if len(particles) < 2:
        print("WARNING: Particle simulation yields the best results with more than 1 particle ")


def check_general_user_input_validity(L, Ng, dt, T):

    """

    Method to check the validity of the inputed initial conditions of the simluation (as entered by the user)

    Parameters
    ----------
    L: length of simulation grid
    Ng: number of cells in grid
    dt: time step size
    T: simulation time

    Raises
    -------
    TypeError: Type error is raised if some or all of the initial conditions are not passed in
    ValueError: Value error is raised if data types or signs of initial conditions are not appropriate

    Returns
    -------
    None

    """

    # ensure the user is not passing in empty inital start conditions
    if L is None or Ng is None or dt is None or T is None:
        raise TypeError("ERROR: Grid Length, number of cells, time step, and simulation time can not be of Type None")

    # ensure that the grid length is an integer
    if not isinstance(L, int):
        raise ValueError("ERROR: Grid Length must be an integer")

    # ensure that the number of cells in the grid is an integer
    if not isinstance(Ng, int):
        raise ValueError("ERROR: Number of cells in grid must be an integer ")

    # ensure that the time step is a numeric value
    if not isinstance(dt, (float, int)):
        raise ValueError("ERROR: Time Step must be a float or integer ")

    # ensure that the simulation ending time is a numeric value
    if not isinstance(T, (float, int)):
        raise ValueError("ERROR: Simulation time must be a float or integer ")

    # ensure that the grid length is positive
    if L <= 0:
        raise ValueError("ERROR: Grid Length must be a positive value")

    # ensure that the number of cells in the grid is positive
    if Ng <= 0:
        raise ValueError("ERROR: Number of cells in grid must be a positive value")

    # ensure that the time step is positive
    if dt <= 0:
        raise ValueError("ERROR: Time Step must be a positive value")

    # ensure that the simulation time is positive
    if T <= 0:
        raise ValueError("ERROR: Simulation time must be a positive value")

    # ensure that mesh spacing is no greater in order than the Debye length
    if L / Ng > C["debyeLength"]:
        print(
            "WARNING: Mesh Spacing L/Ng is greater than Debye length. "
            " \n To increase physical accuracy, try using more grid points or a smaller grid.")

    # ensure that the timestep is small enough to capture the plasma frequency
    if dt * C["plasmaFreq"] > 2:
        print(
            "WARNING: Time-step dt may be too large compared to the plasma frequency. "
            "\n To increase physical accuracy, try decreasing the time-step size.")

    # ensure that the grid length is much less than the debeye length
    if L < 10 * C["debyeLength"]:
        print(
            "WARNING: Grid size L is not much larger than the Debye Length. "
            "\n To accurately resolve Debye shielding, try increasing the grid length.")


def run_simulations(L, Ng, dt, T, particles):

    """

    Method to run a 2D particle-in-cell simulation according to user-specified inputs

    Parameters
    ----------
    L: length of simulation grid
    Ng: number of cells in grid
    dt: time step size
    T: simulation end time
    particles: list of Particle2D objects to be simulated

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # start timer
    start = time.time()

    # check the validity of the user-inputted particles and initial conditions
    check_particle_validity(particles)
    check_general_user_input_validity(L, Ng, dt, T)

    # initialize a grid with inputted conditions
    G = Grid2D(L=L, Ng=Ng, dt=dt, T=T)

    # add all paricles to the grid
    for particle in particles:
        G.addParticle(particle)

    # Ensure that enough particles are present to resolve Debye shielding
    if len(G.Particles) < G.L:
        print(
            "WARNING: There may not be enough particles present to resolve Debye shielding. "
            "\n If attempting to represent plasma waves, try using more particles.")

    # run the discrete model
    RunDiscreteModel(G)

    # end the timer before plotting begins
    end = time.time()

    # plot the charge
    G.plotCharge()

    # plot the potential
    G.plotPotential()

    # plot the electric field
    G.plotEField()

    # plot the state of the particles
    G.plotState()

    # animate
    G.animateState()

    print(f"2D SIMULATION COMPLETED SUCCESSFULLY in {np.around(end - start, 2)} seconds!")
