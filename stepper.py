# Authors: Kyle Fridberg, Kevin Howarth, Hari Raval,                            #
# Course: AM 205                                                                #
# File: stepper.py                                                              #
# Description: Primary algorithms for Electrostatic 1D                          #
# Particle-in-Cell simulations. This file actually runs the simulations by      #
# leveraging the classes created in classes.py                                  #
#################################################################################

import numpy as np
from classes import Particle1D, Grid1D, C


def ChargeAssignmentStep(g, debug):

    """

    Method to perform dimensionless Charge Assignment (pg. 34 of Hockney & Eastwood)

    Parameters
    ----------
    g: Grid1D object to perform charge assignments in
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
        p = int(np.round(g.Particles[j].x[-1]))

        if debug:
            try:
                # increment charge density
                g.Charge[p] += g.C["delChg"]
            except IndexError:
                print("IndexError for particle with position:", g.Particles[j].x[-1])
        else:
            g.Charge[p] += g.C["delChg"]

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
    g: Grid1D object to perform poisson steps on

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # compute potential at mesh point 1 (Eq. 2.56 of Hockney & Eastwood)
    g.Potential[1] = (np.sum([p * g.Charge[p] for p in range(1, g.Ng)]) + g.Ng * g.Charge[0]) / g.Ng

    # compute the remaining potentials after the first
    for p in range(2, g.Ng):
        g.Potential[p] = g.Charge[p - 1] + 2. * g.Potential[p - 1] - g.Potential[p - 2]

    # Calculate the potential energy stored in the potential field and charge distribution
    PE = 0
    for p in range(g.Ng):
        PE += g.Charge[p] * g.Potential[p]
    g.PE = np.append(g.PE, PE * g.C["PEConversionFactor"])

def EFieldStep(g):

    """

    Method to calculate the electric field at every point on the grid using known potentials

    Parameters
    ----------
    g: Grid1D object to perform electric field calculations on

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # calculate electric field at every point on the grid using known potentials (Eq. 2-34, pg 32 of Hockney & Eastwood
    for p in range(g.Ng - 1):
        g.EField[p] = g.Potential[p + 1] - g.Potential[p - 1]

    g.EField[-1] = g.Potential[0] - g.Potential[-2]


def ForceInterpStep(g):

    """

    Method to calculate the dimensionless force interpolation for each particle

    Parameters
    ----------
    g: Grid1D object to perform force interpolation calculations on

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
        pos = int(np.round(prt.x[-1]))

        # extract the electric field at that nearest mesh point
        prt.a.append(g.EField[pos])


def vStep(g):

    """

    Method to calculate the time-step velocity for each particle

    Parameters
    ----------
    g: Grid1D object to perform force interpolation calculations on

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # iterate over each particle in the grid to compute the time-step velocity
    for prt in g.Particles:
        prt.v.append(prt.v[-1] + prt.a[-1])


def xStep(g):

    """

    Method to calculate the time-step position for each particle

    Parameters
    ----------
    g: Grid1D object to perform force interpolation calculations on

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
        prt.x.append(prt.x[-1] + prt.v[-1])

        # translate the position outside the right end of the grid
        while int(np.round(prt.x[-1])) >= g.Ng:
            prt.x[-1] -= g.Ng

        # translate the position outside the left end of the grid
        while int(np.round(prt.x[-1])) < 0:
            prt.x[-1] += g.Ng


def DiscreteModelStep(g, debug):

    """

    Method to perform a single step through the discrete model on Grid g

    Parameters
    ----------
    g: Grid1D object to perform force interpolation calculations on
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
    # perform poisson computations
    PoissonStep(g)

    # print field equation (electric field) diagnostics if debug is set to TRUE
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
    g: Grid1D object to perform force interpolation calculations on
    debug: boolean variable indicating whether to enter debug mode to step through simulation diagnostics carefully

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # TODO: Kevin add checking for the four constraints on pg. 32 of Hockney & Eastwood
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
    particles: list of Particle1D objects to be simulated

    Raises
    -------
    TypeError: Type error is raised if objects in particles list are not all Particle1D
    ValueError: Value error is raised if an empty list of particles is passed in

    Returns
    -------
    None

    """

    # ensure the user is simulating at least one particle
    if particles is None or len(particles) < 1:
        raise ValueError("ERROR: At least one Particle1D object must be passed in")

    # iterate over each particle to ensure it is of Particle1D type, otherwise raise an illegal type error
    for particle in particles:
        if not isinstance(particle, Particle1D):
            raise TypeError("ERROR: All input particles must be Particle1D objects")

    # warn the user of simulating too few particles but do not raise an error
    if len(particles) < 2:
        print("WARNING: Particle simulation yields the best results with more than 1 particle ")


def check_general_user_input_validity(L, Ng, dt):

    """

    Method to check the validity of the inputed initial conditions of the simluation (as entered by the user)

    Parameters
    ----------
    L: length of simulation grid
    Ng: number of cells in grid
    dt: time step size

    Raises
    -------
    TypeError: Type error is raised if some or all of the initial conditions are not passed in
    ValueError: Value error is raised if data types or signs of initial conditions are not appropriate

    Returns
    -------
    None

    """

    # ensure the user is not passing in empty inital start conditions
    if L is None or Ng is None or dt is None:
        raise TypeError("ERROR: Grid Length, Number of cells, and time step can not be of Type None")

    # ensure that the grid length is an integer
    if not isinstance(L, int):
        raise ValueError("ERROR: Grid Length must be an integer integer")

    # ensure that the number of cells in the grid is an integer
    if not isinstance(Ng, int):
        raise ValueError("ERROR: Number of cells in grid must be an integer ")

    # ensure that the time step is a numeric value
    if not isinstance(dt, (float, int)):
        raise ValueError("ERROR: Time Step must be a float or integer ")

    # ensure that the grid length is positive
    if L <= 0:
        raise ValueError("ERROR: Grid Length must be a positive value")

    # ensure that the number of cells in the grid is positive
    if Ng <= 0:
        raise ValueError("ERROR: Number of cells in grid must be a positive value")

    # ensure that the time step is positive
    if dt <= 0:
        raise ValueError("ERROR: Time Step must be a positive value")

    # ensure that mesh spacing is no greater in order than the Debye length
    if L/Ng > C["debyeLength"]:
        print("Warning: Mesh Spacing L/Ng is greater than Debye length. To increase physical accuracy, try using more grid points or a smaller grid.")

    # ensure that the timestep is small enough to capture the plasma frequency
    if dt*C["plasmaFreq"] > 2:
        print("Warning: Time-step dt may be too large compared to the plasma frequency. To increase physical accuracy, try decreasing the time-step size.")

    if L < 10*C["debyeLength"]:
        print("Warning: Grid size L is not much larger than the Debye Length. To accurately resolve Debye shielding, try increasing the grid length.")

def run_simulations(L, Ng, dt, particles, random_state):

    """

    Method to run a 1D particle-in-cell simulation according to user-specified inputs

    Parameters
    ----------
    L: length of simulation grid
    Ng: number of cells in grid
    dt: time step size
     particles: list of Particle1D objects to be simulated
    random_state: random number for simulations

    Raises
    -------
    None

    Returns
    -------
    None

    """

    # check the validity of the user-inputted particles and initial conditions
    check_particle_validity(particles)
    check_general_user_input_validity(L, Ng, dt)

    # initialize a grid with inputted conditions
    G = Grid1D(L=L, Ng=Ng, dt=0.25 * dt, T=300 * dt)

    # add all paricles to the grid
    for particle in particles:
        G.addParticle(particle)

    # if the user passed in a random state, use it otherwise create a truly random one
    if random_state is None:
        random_seed = np.random.RandomState()
    else:
        random_seed = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # add a random distribution of particles
    for i in range(10):
        xi = random_seed.normal(0.6, 0.1)
        G.addParticle(Particle1D(ID=str(i + 7), x0=xi * Ng, v0=0))

    # Ensure that enough particles are present to resolve Debye shielding
    if len(G.Particles) < G.L:
        print("Warning: There may not be enough particles present to resolve Debye shielding. If attempting to represent plasma waves, try using more particles.")

    # run the discrete model
    RunDiscreteModel(G)

    # plot the charge
    G.plotCharge()

    # plot the potential
    G.plotPotential()

    # plot the electric field
    G.plotEField()

    # plot the state of the particles
    G.plotState()

    # animate
    # G.animateState()
