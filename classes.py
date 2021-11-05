# File containing the grid and particle classes
import numpy as np
import matplotlib.pyplot as plt

# some global constants
qEle = 1.602176634e-19  # absolute value of the elementary charge [C]
eMass = 9.10938356e-31  # electron mass [kg]
eps0 = 8.8541878128e-12  # vacuum permittivity [kg^-1 * m^-3 * s^4 * A^2]

class Particle:
    # A class for representing a particle with some mass, charge, and kinematic signature
    def __init__(self, ID, x0=0, v0=0):
        # self.q = q  # Particle charge
        # self.m = m  # Particle mass
        self.ID = ID  # string identifier for particle
        self.x = [x0]  # Position in dimensionless units of intervals [L * H^-1] = [m * m^-1]
        self.v = [v0]  # Velocity in dimensionless units of cell widths per time-step
        self.a = []   # Acceleration in dimensionless units of cell widths per time-step^2

class Grid:
    # A class for representing the 1D grid in which the particles live
    def __init__(self, L, Ng, dt, T, n0):
        self.L = L  # grid length [m]
        self.H = L/Ng   # grid spacing [m]
        self.Ng = Ng  # number of grid points
        self.dt = dt  # time-step size [s]
        self.T = T  # ending simulation time
        self.Particles = []  # list of all Particles in the grid
        self.Charge = np.zeros(Ng - 1)  # Dimensionless charge at all grid points
        self.Potential = np.zeros(Ng - 1)  # Dimensionless potential at all grid points
        self.EField = np.zeros(Ng - 1)  # Dimensionless electric field at all grid points
        self.n0 = n0  # Background ion density TODO: can maybe estimate rho0 with this?

    def addParticle(self, p):
        self.Particles.append(p)

    def W(self, x):
        # Charge assignment function, Eq. 2.28 on pg. 31 of Hockney & Eastwood
        if np.abs(x) <= self.H/2:
            return 1
        else:
            return 0

    def wp(self):
        # Return the electron plasma frequency
        return np.sqrt(self.n0*qEle**2/(eps0*eMass))

    def plotState1D(self):
        # Plot the position of each particle as a function of time
        # Not worrying about units for now, just looking at behavior
        for prt in self.Particles:
            plt.plot(range(len(prt.x)), prt.x, '--', label="Particle "+prt.ID)

        plt.xlabel("Time-step")
        plt.ylabel("Position [Intervals]")
        plt.title("Particle Positions vs. Time")
        plt.grid()
        plt.legend(loc='best')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
