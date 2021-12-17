# Authors: Kyle Fridberg, Kevin Howarth, Hari Raval                             #
# Course: AM 205                                                                #
# File: classes.py                                                              #
# Description: Class definitions for data structures necesary for               #
# Electrostatic 1D Particle-in-Cell simulations. The file contains the          #
# Particle1D class which represents a particle with some mass, charge,          #
# and kinematic signature and a Grid1D class which represents the 1D grid in    #
# which the particles live                                                      #
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# NOTE: global contants listed below for reference (not used as everything is done in a unitless manner)
# qEle = 1.602176634e-19  <------> absolute value of the elementary charge [C]
# eMass = 9.10938356e-31  <------> electron mass [kg]
# eps0 = 8.8541878128e-12 <------> vacuum permittivity [kg^-1 * m^-3 * s^4 * A^2]

# dictionary of constants for the 1D two-particle case
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


class Particle1D:

    """

    Class containing the definition of a particle in 1 dimension; represents a
    particle with mass, charge, and kinematic signature

    Instance Variables
    ----------
    ID: string identifier / unique label of an individual particle
    X0: initial position of particle in dimensionless units of intervals (default value 0)
    v0: initial velocity of particle in dimensionless units of cell widths per time-step (default value 0)
    a: acceleration of particle in dimensionless units of cell widths per time-step^2

    Returns
    -------
    Particle1D object

    """

    def __init__(self, ID, x0=0, v0=0):
        # string identifier for a particle
        self.ID = ID
        # position in dimensionless units of intervals [L * H^-1] = [m * m^-1]
        self.x = [x0]
        # velocity in dimensionless units of cell widths per time-step
        self.v = [v0]
        # acceleration in dimensionless units of cell widths per time-step^2
        self.a = []

    def __copy__(self):

        """

        Method to create a deep copy of a Particle1D object

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        pNew: Particle1D object containing the same attributes which self contained

        """

        # create a deep copy of a particle
        pNew = Particle1D(ID=self.ID + "_copy")
        pNew.x = self.x[:]
        pNew.v = self.v[:]
        pNew.a = self.a[:]

        return pNew


class Grid1D:

    """

    Class containing the definition of a Grid1D class which represents the 1D grid in
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
    EField: dimensionless electric field at all grid points
    PE: potential energy at each time-step
    C: dictionary of physical values

    Returns
    -------
    Grid1D object

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
        self.Particles = np.array([], dtype=Particle1D)
        # dimensionless charge at all grid points
        self.Charge = np.zeros(Ng)
        # dimensionless potential at all grid points
        self.Potential = np.zeros(Ng)
        # dimensionless electric field at all grid points
        self.EField = np.zeros(Ng)
        # potential energy at each time-step
        self.PE = np.array([], dtype=float)

        # populate the dictionary with the correct values for plasmaFreqDT and qBackground
        self.C = C.copy()
        self.C.update({"plasmaFreqDT": self.dt * self.C["plasmaFreq"]})
        self.C.update({"qBackground": -self.C["plasmaFreqDT"] ** 2 / 2.})

    def __copy__(self):

        """

        Method to create a deep copy of a Grid1D object

        ** NOTE: Copying a grid object requires copying each particle contained in the grid separately **

        Parameters
        ----------
        None

        Raises
        -------
        None

        Returns
        -------
        GNew: Grid1D object containing the same attributes which the original object contained

        """

        Gnew = Grid1D(self.L, self.Ng, self.dt, self.T)
        # iterate over each particle and make a shallow copy of it
        for i in range(len(self.Particles)):
            Gnew.Particles = np.append(Gnew.Particles, self.Particles[i].__copy__())

        # update, in the new grid, the average particles per cell and the change in value
        Gnew.C.update({"avgParticlesPerCell": len(Gnew.Particles) / Gnew.Ng})
        Gnew.C.update({"delChg": Gnew.C["plasmaFreqDT"] ** 2 * Gnew.Ng / (2. * len(Gnew.Particles))})
        Gnew.C.update(({"PEConversionFactor": -16.0 * len(Gnew.Particles) / (Gnew.Ng * Gnew.C["plasmaFreqDT"] ** 2)}))
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

        # adjust the particle position by moving to the left, if necessary
        while int(np.round(p.x[0])) >= self.Ng:
            p.x[0] -= self.Ng

        # adjust the particle position by moving to the right, if necessary
        while int(np.round(p.x[0])) < 0:
            p.x[0] += self.Ng

        # after adding the particle, update the related parameters in the dictionary
        self.Particles = np.append(self.Particles, p)
        self.C.update({"avgParticlesPerCell": len(self.Particles) / self.Ng})
        self.C.update({"delChg": self.C["plasmaFreqDT"] ** 2 * self.Ng / (2. * len(self.Particles))})
        self.C.update(({"PEConversionFactor": -16.0 * len(self.Particles) / (self.Ng * self.C["plasmaFreqDT"] ** 2)}))

    def W(self, x):

        """

        Method to create the charge assignment function. This method performs the zeroth order interpolation
        function. (Eq. 2.28 on pg. 31 of Hockney & Eastwood)

        Parameters
        ----------
        x: position of the particle of the current article

        Raises
        -------
        None

        Returns
        -------
        1 or 0: Appropriate charge assignment of the particle

        """

        if np.abs(x) <= self.H / 2:
            return 1
        else:
            return 0

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
                self.Particles[prt].x[xi] *= NgNew / self.Ng
            for vi in range(len(self.Particles[prt].v)):
                self.Particles[prt].v[vi] *= NgNew / self.Ng
            for ai in range(len(self.Particles[prt].a)):
                self.Particles[prt].a[ai] *= NgNew / self.Ng

        # update the parameters of the grid itself
        self.Ng = NgNew
        self.H = self.L / NgNew
        self.Charge = np.zeros(NgNew)
        self.Potential = np.zeros(NgNew)
        self.EField = np.zeros(NgNew)

        # update the dictionary constants based on the new number of grid points
        self.C.update({"avgParticlesPerCell": len(self.Particles) / NgNew})
        self.C.update({"delChg": self.C["plasmaFreqDT"] ** 2 * NgNew / (2. * len(self.Particles))})
        self.C.update(({"PEConversionFactor": -16.0 * len(self.Particles) / (self.Ng * self.C["plasmaFreqDT"] ** 2)}))

    def updateDt(self, dtNew):

        """

        Method to update all parameters needed to update to a new time-step size. Additionally, updates
        ending simulation time so that the total number of time-steps remains the same

        Parameters
        ----------
        dtNew: updated timestep

        Raises
        -------
        None

        Returns
        -------
        None

        """

        # update relevant parameters
        numTimesteps = self.T/self.dt
        self.dt = dtNew
        self.T = numTimesteps*dtNew
        self.C.update({"plasmaFreqDT": self.dt * self.C["plasmaFreq"]})
        self.C.update({"qBackground": -self.C["plasmaFreqDT"] ** 2 / 2.})
        self.C.update({"delChg": self.C["plasmaFreqDT"] ** 2 * self.Ng / (2. * len(self.Particles))})
        self.C.update(({"PEConversionFactor": -16.0 * len(self.Particles) / (self.Ng * self.C["plasmaFreqDT"] ** 2)}))

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
        TotalKE = np.zeros_like(self.Particles[0].v)
        # iterate over each time step
        for i in range(len(TotalKE)):
            # iterate over all particles per time step
            for prt in range(len(self.Particles)):
                TotalKE[i] += (self.Particles[prt].v[i-1] + self.Particles[prt].v[i]) ** 2

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
            plt.plot(range(len(prt.x)), prt.x, 'o', label="Particle " + prt.ID)

        plt.xlabel("Time-step")
        plt.ylabel("Position [Intervals]")
        plt.title("Particle Positions vs. Time")
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

        plt.plot(range(self.Ng), self.Charge, 'o')
        plt.xlabel("Grid Position [Intervals]")
        plt.ylabel("Charge")
        plt.title("Grid Charge")
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

        plt.plot(range(self.Ng), self.EField, 'o')
        plt.xlabel("Grid Position [Intervals]")
        plt.ylabel("EField")
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

        plt.plot(range(self.Ng), self.Potential, 'o')
        plt.xlabel("Grid Position [Intervals]")
        plt.ylabel("Potential")
        plt.title("Grid Potential")
        plt.grid()
        plt.show()

    def animateState(self, animateVelocity=False):

        """

        Method to create a video of the particle simulations

        ** NOTE 1: In order to generate a movie, users must install moviepy via 'pip3 install moviepy' **

        ** NOTE 2: Generating the movie takes a considerable amount of time,
        especially for larger numbers of particles **

        Parameters
        ----------
        animateVelocity: debug option to determine whether to build animation video or not

        Raises
        -------
        None

        Returns
        -------
        None

        """

        # set the duration of the video
        if animateVelocity:
            duration = 20
        else:
            duration = 60
        # set the frames per second of the video
        fps = 20

        # create a matplotlib subplot to hold the frames
        fig, ax = plt.subplots()

        # create a method to retrieve the frames
        def make_frame(t):
            # clear the current frame
            ax.clear()

            # plot the line
            if not animateVelocity:
                for prt in self.Particles:
                    ax.plot(prt.x[int(t * fps) % len(prt.x)], 0, 'o', markersize=40, label="Particle " + prt.ID)
                    ax.set_xlim([-0.5, self.Ng - 0.5])
            else:
                for prt in self.Particles:
                    ax.plot(prt.x[int(t * fps) % len(prt.x)], prt.v[int(t * fps) % len(prt.v)], '.')
                    ax.set_xlim([-0.5, self.Ng - 0.5])
                    ax.set_ylim([-10, 10])
                ax.set_xlabel("Position [Grid Spacings]")
                ax.set_ylabel("Velocity [Grid Spacings/Plasma Period]")

            # return a numpy image
            return mplfig_to_npimage(fig)

        # create the overall animation
        animation = VideoClip(make_frame, duration=duration)

        # display the animation with auto play and no looping
        animation.ipython_display(fps=fps, loop=False, autoplay=True)
