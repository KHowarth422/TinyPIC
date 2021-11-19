# File containing data structures for Electrostatic 1D Particle-in-Cell simulations
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# some global constants
# NOTE: For now these aren't used as everything is done non-dimensionally (see C below)
# qEle = 1.602176634e-19  # absolute value of the elementary charge [C]
# eMass = 9.10938356e-31  # electron mass [kg]
# eps0 = 8.8541878128e-12  # vacuum permittivity [kg^-1 * m^-3 * s^4 * A^2]

# dictionary of constants for the 1D two-particle case
# I am copying everything in the dictionary as directly as I can from the textbook
# In theory if you wanted to change everything to SI units instead of units of
# plasma frequencies in time, this is where you would enforce that change.
C = {
    "Kb": 1.0,  # Boltzmann Constant
    "eChg": -1.0,  # Electron charge
    "eMass": 1.0,  # Electron mass
    "eps0": 1.0,  # Vacuum permittivity
    "rho0": 1.0,  # Background particle density
    "T0": 1.0,  # Particle distribution temperature
    "vd": 1.0  # Particle distribution drift velocity
}
C.update({"debyeLength": np.sqrt(C["eps0"]*C["Kb"]/(C["rho0"]*C["eChg"]**2))})
C.update({"plasmaFreq": np.sqrt(C["rho0"]*C["eChg"]**2/(C["eps0"]*C["eMass"]))})
C.update({"vTh": C["debyeLength"]*C["plasmaFreq"]})

class Particle:
    # A class for representing a particle with some mass, charge, and kinematic signature
    def __init__(self, ID, x0=0, v0=0):
        self.ID = ID  # string identifier for particle. Currently not used for anything and may
                      # not be necessary. I thought it could possibly be useful to carry around
                      # an identifier, maybe not worth the extra memory especially when we scale
                      # to large numbers of particles.
        self.x = [x0]  # Position in dimensionless units of intervals [L * H^-1] = [m * m^-1]
        self.v = [v0]  # Velocity in dimensionless units of cell widths per time-step
        self.a = []   # Acceleration in dimensionless units of cell widths per time-step^2

class Grid:
    # A class for representing the 1D grid in which the particles live
    def __init__(self, L, Ng, dt, T):
        self.L = L  # grid length [m]
        self.H = L/Ng   # grid spacing [m]
        self.Ng = Ng  # number of grid points
        self.dt = dt  # time-step size [plasma frequencies]
        self.T = T  # ending simulation time
        self.Particles = []  # list of all Particles in the grid
                             # TODO: Particles should maybe be a numpy array instead? Does it matter?
                             #       I'm only using a list so I can do Particles.append() in Grid.addParticle()
        self.Charge = np.zeros(Ng)  # Dimensionless charge at all grid points
        self.Potential = np.zeros(Ng)  # Dimensionless potential at all grid points
        self.EField = np.zeros(Ng)  # Dimensionless electric field at all grid points

        # Populate the dictionary
        self.C = C.copy()
        self.C.update({"plasmaFreqDT": self.dt * self.C["plasmaFreq"]})
        self.C.update({"qBackground": -self.C["plasmaFreqDT"] ** 2 / 2.})

    def addParticle(self, p):
        # Add a particle to the grid. If the particle position is outside the grid, adjust
        # the position until it is on the periodic image inside the grid.
        # Note that the valid range of positions is -0.5 <= x < Ng - 0.5, so that the nearest
        # integer to any particle is a valid grid point index.

        while int(np.round(p.x[0])) >= self.Ng:
            p.x[0] -= self.Ng

        while int(np.round(p.x[0])) < 0:
            p.x[0] += self.Ng

        self.Particles.append(p)

        # after adding the particle, update the related parameters in the dictionary
        self.C.update({"avgParticlesPerCell": len(self.Particles)/self.Ng})
        self.C.update({"delChg": self.C["plasmaFreqDT"]**2 * self.Ng / (2. * len(self.Particles))})

    def W(self, x):
        # Charge assignment function, Eq. 2.28 on pg. 31 of Hockney & Eastwood
        # This is basically the zeroth order interpolation function
        if np.abs(x) <= self.H/2:
            return 1
        else:
            return 0

    def plotState1D(self):
        # Plot the position of each particle as a function of time
        for prt in self.Particles:
            plt.plot(range(len(prt.x)), prt.x, 'o', label="Particle "+prt.ID)

        plt.xlabel("Time-step")
        plt.ylabel("Position [Intervals]")
        plt.title("Particle Positions vs. Time")
        plt.grid()
        plt.show()

    def plotCharge1D(self):
        # Plot the charge at every point on the grid
        plt.plot(range(self.Ng),self.Charge,'o')
        plt.xlabel("Grid Position [Intervals]")
        plt.ylabel("Charge")
        plt.title("Grid Charge")
        plt.grid()
        plt.show()

    def plotEField1D(self):
        # Plot the charge at every point on the grid
        plt.plot(range(self.Ng),self.EField,'o')
        plt.xlabel("Grid Position [Intervals]")
        plt.ylabel("EField")
        plt.title("Grid EField")
        plt.grid()
        plt.show()

    def plotPotential1D(self):
        # Plot the potential at every point on the grid
        plt.plot(range(self.Ng), self.Potential, 'o')
        plt.xlabel("Grid Position [Intervals]")
        plt.ylabel("Potential")
        plt.title("Grid Potential")
        plt.grid()
        plt.show()

    def animateState1D(self):
        # duration of the video
        duration = 5
        fps = 20

        # matplot subplot
        fig, ax = plt.subplots()

        # method to get frames
        def make_frame(t):
            # clear
            ax.clear()

            # plotting line
            for prt in self.Particles:
                ax.plot(prt.x[int(t*fps)%len(prt.x)], 0, 'o', label="Particle " + prt.ID)
                ax.set_xlim([-0.5,63.5])

            # returning numpy image
            return mplfig_to_npimage(fig)

        # creating animation
        animation = VideoClip(make_frame, duration=duration)

        # displaying animation with auto play and looping
        animation.ipython_display(fps=fps, loop=False, autoplay=True)
