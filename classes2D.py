# Authors: Kyle Fridberg, Kevin Howarth, Hari Raval                             #
# Course: AM 205                                                                #
# File: classes2D.py                                                            #
# Description: Class definitions for data structures necesary for               #
# Electrostatic 2D Particle-in-Cell simulations. The file contains the          #
# Particle2D class which represents a particle with some mass, charge,          #
# and kinematic signature and a Grid2D class which represents the 2D grid in    #
# which the particles live                                                      #
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# some global constants
# NOTE: For now these aren't used as everything is done non-dimensionally (see C below)
# qEle = 1.602176634e-19  # absolute value of the elementary charge [C]
# eMass = 9.10938356e-31  # electron mass [kg]
# eps0 = 8.8541878128e-12  # vacuum permittivity [kg^-1 * m^-3 * s^4 * A^2]

# dictionary of constants for the 2D two-particle case
C = {
    "Kb": 1.0,  # boltzmann Constant
    "eChg": -1.0,  # electron charge
    "eMass": 1.0,  # electron mass
    "eps0": 1.0,  # vacuum permittivity
    "rho0": 1.0,  # background particle density
    "T0": 1.0,  # particle distribution temperature
    "vd": 1.0  # particle distribution drift velocity
}

# update dictionary of constants to include additional three terms
C.update({"debyeLength": np.sqrt(C["eps0"] * C["Kb"] / (C["rho0"] * C["eChg"] ** 2))})
C.update({"plasmaFreq": np.sqrt(C["rho0"] * C["eChg"] ** 2 / (C["eps0"] * C["eMass"]))})
C.update({"vTh": C["debyeLength"] * C["plasmaFreq"]})


class Particle2D:

    """

    Class containing the definition of a particle in 2 dimensions; represents a
    particle with mass, charge, and kinematic signature

    Instance Variables
    ----------
    ID: string identifier / unique label of an individual particle
    X0: initial x position of particle in dimensionless units of intervals (default value 0)
    X1: initial y position of particle in dimensionless units of intervals (default value 0)
    v0: initial x velocity of particle in dimensionless units of cell widths per time-step (default value 0)
    v1: initial y velocity of particle in dimensionless units of cell widths per time-step (default value 0)
    a0: initial x acceleration of particle in dimensionless units of cell widths per time-step^2
    a1: initial y acceleration of particle in dimensionless units of cell widths per time-step^2


    Returns
    -------
    Particle2D object

    """

    def __init__(self, ID, x0=0, x1=0, v0=0, v1=0):
        # string identifier for a particle
        self.ID = ID
        # x position in dimensionless units of intervals [L * H^-1] = [m * m^-1]
        self.x_0 = [x0]
        # y position in dimensionless units of intervals [L * H^-1] = [m * m^-1]
        self.x_1 = [x1]
        # x velocity in dimensionless units of cell widths per time-step
        self.v_0 = [v0]
        # y velocity in dimensionless units of cell widths per time-step
        self.v_1 = [v1]
        # x acceleration in dimensionless units of cell widths per time-step^2
        self.a_0 = []
        # y acceleration in dimensionless units of cell widths per time-step^2
        self.a_1 = []

    def __copy__(self):

        """

        Method to create a deep copy of a Particle2D object

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        pNew: Particle2D object containing the same attributes which self contained

        """

        # create a deep copy of a particle
        pNew = Particle2D(ID=self.ID + "_copy")
        pNew.x_0 = self.x_0[:]
        pNew.x_1 = self.x_1[:]
        pNew.v_0 = self.v_0[:]
        pNew.v_1 = self.v_1[:]
        pNew.a_0 = self.a_0[:]
        pNew.a_1 = self.a_1[:]

        return pNew


class Grid2D:

    """

    Class containing the definition of a Grid2D class which represents the 2D grid in
    which a particle lives

    Instance Variables
    ----------
    L: length of grid for simulation
    Ng: grid spacing for simulation
    dt: time-step size
    T: ending simulation time
    Particles: list of all Particles in the grid
    Charge: dimensionless charge at all grid points
    Potential: dimensionless potential at all grid points
    EField_0: dimensionless electric field in horizontal direction at all grid points
    EField_1: dimensionless electric field in vertical direction at all grid points
    PE: potential energy at each time-step
    C: dictionary of physical values

    Returns
    -------
    Grid2D object

    """

    def __init__(self, L, Ng, dt, T):
        # grid length
        self.L = L
        # grid spacing
        self.H = L / Ng
        # number of grid points
        self.Ng = Ng
        # time-step size (plasma frequencies)
        self.dt = dt
        # ending simulation time
        self.T = T
        # list of all Particles in the grid
        self.Particles = np.array([], dtype=Particle2D)
        # dimensionless charge at all grid points
        self.Charge = np.zeros((Ng, Ng))
        # dimensionless potential at all grid points
        self.Potential = np.zeros((Ng, Ng))
        # dimensionless electric field in horizontal direction at all grid points
        self.EField_0 = np.zeros((Ng, Ng))
        # dimensionless electric field in vertical direction at all grid points
        self.EField_1 = np.zeros((Ng, Ng))

        # populate the dictionary with the correct values for plasmaFreqDT and qBackground
        self.C = C.copy()
        self.C.update({"plasmaFreqDT": self.dt * self.C["plasmaFreq"]})
        self.C.update({"qBackground": -self.C["plasmaFreqDT"] ** 2 / 2.})

    def __copy__(self):

        """

        Method to create a deep copy of a Grid2D object

        ** NOTE: Copying a grid object requires copying each particle contained in the grid separately **

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        GNew: Grid2D object containing the same attributes which the original object contained

        """

        Gnew = Grid2D(self.L, self.Ng, self.dt, self.T)
        # iterate over each particle and make a shallow copy of it
        for i in range(len(self.Particles)):
            Gnew.Particles = np.append(Gnew.Particles, self.Particles[i].__copy__())

        # update, in the new grid, the average particles per cell and the change in value
        Gnew.C.update({"avgParticlesPerCell": len(Gnew.Particles) / Gnew.Ng ** 2})
        Gnew.C.update(
            {"delChg": Gnew.C["plasmaFreqDT"] ** 2 * Gnew.Ng ** 2 / (2. * len(Gnew.Particles))})
        return Gnew

    def addParticle(self, p):

        """

        Method to add a particle to the grid. If the particle position is outside the grid,
        adjust the position until it is on the periodic image inside the grid

        ** NOTE: The valid range of positions is -0.5 <= x < Ng - 0.5, so that the nearest
        integer to any particle is a valid grid point index **

        Parameters
        ----------
        p: particle to be added to the grid

        Raises
        -------
        None

        Returns
        -------
        None

        """

        # adjust the particle position by moving to the left/right, if necessary
        p.x_0[0] = (p.x_0[0] + 1 / 2) % self.Ng - 1 / 2
        # adjust the particle position by moving to the up/down, if necessary
        p.x_1[0] = (p.x_1[0] + 1 / 2) % self.Ng - 1 / 2

        self.Particles = np.append(self.Particles, p)

        # after adding the particle, update the related parameters in the dictionary
        self.C.update({"avgParticlesPerCell": len(self.Particles) / self.Ng ** 2})
        self.C.update(
            {"delChg": self.C["plasmaFreqDT"] ** 2 * self.Ng ** 2 / (2. * len(self.Particles))})

    def updateNg(self, NgNew):

        """

        Method to update all parameters (including rescaling particle positions),
        which is needed to update to a new number of mesh points

        Parameters
        ----------
        NgNew: updated number of cells in the grid

        Raises
        -------
        None

        Returns
        -------
        None

        """

        # iterate over all particles in the grid and rescale to the appropriate functions
        for prt in range(len(self.Particles)):
            for xi in range(len(self.Particles[prt].x)):
                self.Particles[prt].x_0[xi] *= NgNew / self.Ng
                self.Particles[prt].x_1[xi] *= NgNew / self.Ng
            for vi in range(len(self.Particles[prt].v)):
                self.Particles[prt].v_0[vi] *= NgNew / self.Ng
                self.Particles[prt].v_1[vi] *= NgNew / self.Ng
            for ai in range(len(self.Particles[prt].a)):
                self.Particles[prt].a_0[ai] *= NgNew / self.Ng
                self.Particles[prt].a_1[ai] *= NgNew / self.Ng

        # update the parameters of the grid itself
        self.Ng = NgNew
        self.H = self.L / NgNew
        self.Charge = np.zeros((NgNew, NgNew))
        self.Potential = np.zeros((NgNew, NgNew))
        self.EField_0 = np.zeros((NgNew, NgNew))
        self.EField_1 = np.zeros((NgNew, NgNew))

        # update the dictionary constants based on the new number of grid points
        self.C.update({"avgParticlesPerCell": len(self.Particles) / NgNew ** 2})
        self.C.update({"delChg": self.C["plasmaFreqDT"] ** 2 * NgNew ** 2 / (2. * len(self.Particles))})

    def getTotalKineticEnergy(self):

        """

        Method to calculate and retrieve the total kinetic energy of all particles
        in the grid at each time-step up to the current/present

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        TotalKE: total kinetic energy of all particles at the current time-step

        """

        # return the total kinetic energy of all particles in the grid at each time-step up to the current/present
        TotalKE = np.zeros_like(self.Particles[0].v_0)
        # iterate over each time step
        for i in range(len(TotalKE)):
            # iterate over all particles per time step
            for prt in range(len(self.Particles)):
                TotalKE[i] += 0.5 * self.C["eMass"] * \
                              (self.Particles[prt].v_0[i] ** 2 + self.Particles[prt].v_1[i] ** 2)
        return TotalKE

    def plotState(self):

        """

        Method to plot the position of each particle as a function of time

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        None

        """

        # plot the position of each particle as a function of time
        for prt in self.Particles:
            plt.plot(prt.x_0, prt.x_1, 'o', label="Particle " + prt.ID)

        plt.xlim([-0.5, self.Ng - 0.5])
        plt.ylim([-0.5, self.Ng - 0.5])
        plt.xlabel("x-position")
        plt.ylabel("y-position")
        plt.title("Particle Trajectories")
        plt.grid()
        plt.show()

    def plotCharge(self):

        """

        Method to plot the charge at every point on the grid

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        None

        """

        X = range(self.Ng)
        X, Y = np.meshgrid(X, X)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, self.Charge)
        ax.set_xlabel("X Grid Position [Intervals]")
        ax.set_ylabel("Y Grid Position [Intervals]")
        ax.set_zlabel("Grid Charge")
        plt.grid()
        plt.show()

    def plotEField(self):

        """

        Method to plot the electric field at every point on the grid

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        None

        """

        plt.quiver(self.EField_0, self.EField_1)
        plt.xlabel("X Grid Position [Intervals]")
        plt.ylabel("Y Grid Position [Intervals]")
        plt.title("Grid EField")
        plt.grid()
        plt.show()

    def plotPotential(self):

        """

        Method to plot the potential at every point on the grid

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        None

        """

        X = range(self.Ng)
        X, Y = np.meshgrid(X, X)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, self.Potential)
        ax.set_xlabel("X Grid Position [Intervals]")
        ax.set_ylabel("Y Grid Position [Intervals]")
        ax.set_zlabel("Potential")
        ax.set_title("Grid Potential")
        plt.grid()
        plt.show()

    def animateState(self):

        """

        Method to create a video of the particle simulations

        ** NOTE 1: In order to generate a movie, users must install moviepy via 'pip3 install moviepy' **

        ** NOTE 2: Generating the movie takes a considerable amount of time,
        especially for larger numbers of particles **

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        None

        """

        # duration of the video
        duration = 20
        fps = 20

        # matplot subplot
        fig, ax = plt.subplots()

        # method to get frames
        def make_frame(t):
            # clear
            ax.clear()

            # plotting line
            for prt in self.Particles:
                ax.plot(prt.x_0[int(t * fps) % len(prt.x_0)], prt.x_1[int(t * fps) % len(prt.x_1)], 'o',
                        label="Particle " + prt.ID)
                ax.set_xlim([-0.5, self.Ng - 0.5])
                ax.set_ylim([-0.5, self.Ng - 0.5])

            # returning numpy image
            return mplfig_to_npimage(fig)

        # creating animation
        animation = VideoClip(make_frame, duration=duration)

        # displaying animation with auto play and looping
        animation.ipython_display(fps=fps, loop=False, autoplay=True)
