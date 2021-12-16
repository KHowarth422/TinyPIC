# File containing data structures for Electrostatic 2D Particle-in-Cell simulations
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from scipy.interpolate import RegularGridInterpolator

# some global constants
# NOTE: For now these aren't used as everything is done non-dimensionally (see C below)
# qEle = 1.602176634e-19  # absolute value of the elementary charge [C]
# eMass = 9.10938356e-31  # electron mass [kg]
# eps0 = 8.8541878128e-12  # vacuum permittivity [kg^-1 * m^-3 * s^4 * A^2]

# dictionary of constants for the 2D two-particle case
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

class Particle2D:
    # A class for representing a particle with some mass, charge, and kinematic signature
    def __init__(self, ID, x0 = 0, x1 = 0, v0 = 0, v1 = 0):
        self.ID = ID  # string identifier for particle. Currently not used for anything and may
                      # not be necessary. I thought it could possibly be useful to carry around
                      # an identifier, maybe not worth the extra memory especially when we scale
                      # to large numbers of particles.
        self.x_0 = [x0]  # Position in dimensionless units of intervals [L * H^-1] = [m * m^-1]
        self.x_1 = [x1]
        self.v_0 = [v0]  # Velocity in dimensionless units of cell widths per time-step
        self.v_1 = [v1]
        self.a_0 = []   # Acceleration in dimensionless units of cell widths per time-step^2
        self.a_1 = []

    def __copy__(self):
        # Shallow copy of a particle
        pNew = Particle2D(ID = self.ID+"_copy")
        pNew.x_0 = self.x_0[:]
        pNew.x_1 = self.x_1[:]
        pNew.v_0 = self.v_0[:]
        pNew.v_1 = self.v_1[:]
        pNew.a_0 = self.a_0[:]
        pNew.a_1 = self.a_1[:]
        return pNew

class Grid2D:
    # A class for representing the 2D grid in which the particles live
    def __init__(self, L, Ng, dt, T):
        self.L = L  # grid length [m]
        self.H = L/Ng   # grid spacing [m]
        self.Ng = Ng  # grid dimension is Ng x Ng
        self.dt = dt  # time-step size [plasma frequencies]
        self.T = T  # ending simulation time
        self.Particles = np.array([],dtype=Particle2D)  # list of all Particles in the grid
        self.Charge = np.zeros((Ng,Ng))  # Dimensionless charge at all grid points
        self.Potential = np.zeros((Ng,Ng))  # Dimensionless potential at all grid points
        self.EField_0 = np.zeros((Ng,Ng))  # Dimensionless electric field in horizontal direction at all grid points
        self.EField_1 = np.zeros((Ng,Ng))  # Dimensionless electric field in vertical direction at all grid points

        # Populate the dictionary
        self.C = C.copy()
        self.C.update({"plasmaFreqDT": self.dt * self.C["plasmaFreq"]})
        self.C.update({"qBackground": -self.C["plasmaFreqDT"] ** 2 / 2.})

    def __copy__(self):
        # Make a copy of the Grid. This must be done carefully, since the particles
        # contained by the Grid must be copied separately.
        Gnew = Grid2D(self.L, self.Ng, self.dt, self.T)
        for i in range(len(self.Particles)):
            Gnew.Particles = np.append(Gnew.Particles,self.Particles[i].__copy__())
        Gnew.C.update({"avgParticlesPerCell": len(Gnew.Particles)/Gnew.Ng**2})
        Gnew.C.update({"delChg": Gnew.C["plasmaFreqDT"]**2 * Gnew.Ng**2 / (2. * len(Gnew.Particles))}) #COME BACK TO
        return Gnew

    def addParticle(self, p):
        # Add a particle to the grid. If the particle position is outside the grid, adjust
        # the position until it is on the periodic image inside the grid.
        # Note that the valid range of positions is -0.5 <= x < Ng - 0.5, so that the nearest
        # integer to any particle is a valid grid point index.

        p.x_0[0] = (p.x_0[0] + 1/2) % self.Ng - 1/2
        p.x_1[0] = (p.x_1[0] + 1/2) % self.Ng - 1/2

        self.Particles = np.append(self.Particles, p)

        # after adding the particle, update the related parameters in the dictionary
        self.C.update({"avgParticlesPerCell": len(self.Particles)/self.Ng**2})
        self.C.update({"delChg": self.C["plasmaFreqDT"]**2 * self.Ng**2 / (2. * len(self.Particles))}) #COME BACK TO

    # def W(self, x): #COME BACK TO
    #     # Charge assignment function, Eq. 2.28 on pg. 31 of Hockney & Eastwood
    #     # This is basically the zeroth order interpolation function
    #     if np.abs(x) <= self.H/2:
    #         return 1
    #     else:
    #         return 0

    # def gridInterpolate(self, x):
    #     lat = [i for i in range(self.Ng)]
    #     xi, yi = np.meshgrid(lat, lat, indexing='ij', sparse=True)
    #     data = self.Charge[xi][yi]
    #     itp = RegularGridInterpolator((xi, yi), data, method='nearest')

    #     return itp


    def updateNg(self, NgNew):
        # Update all parameters needed to update to a new number of mesh points
        # This includes rescaling particle positions
        for prt in range(len(self.Particles)):
            for xi in range(len(self.Particles[prt].x)):
                self.Particles[prt].x_0[xi] *= NgNew/self.Ng
                self.Particles[prt].x_1[xi] *= NgNew/self.Ng
            for vi in range(len(self.Particles[prt].v)):
                self.Particles[prt].v_0[vi] *= NgNew/self.Ng
                self.Particles[prt].v_1[vi] *= NgNew/self.Ng
            for ai in range(len(self.Particles[prt].a)):
                self.Particles[prt].a_0[ai] *= NgNew/self.Ng
                self.Particles[prt].a_1[ai] *= NgNew/self.Ng
        self.Ng = NgNew
        self.H = self.L / NgNew
        self.Charge = np.zeros((NgNew, NgNew))
        self.Potential = np.zeros((NgNew, NgNew))
        self.EField_0 = np.zeros((NgNew, NgNew))
        self.EField_1 = np.zeros((NgNew, NgNew))
        self.C.update({"avgParticlesPerCell": len(self.Particles) / NgNew**2})
        self.C.update({"delChg": self.C["plasmaFreqDT"] ** 2 * NgNew**2 / (2. * len(self.Particles))}) # COME BACK TO

    def getTotalKineticEnergy(self):
        # Return the total kinetic energy of all particles in the Grid at each time-step
        # up to the present.
        TotalKE = np.zeros_like(self.Particles[0].v_0)
        for i in range(len(TotalKE)): # At each time-step...
            for prt in range(len(self.Particles)): # For each particle...
                TotalKE[i] += 0.5*self.C["eMass"]*(self.Particles[prt].v_0[i]**2 + self.Particles[prt].v_1[i]**2)
        return TotalKE

    ############################################
    ##### PLOTTING/VISUALIZATION FUNCTIONS ##### ## LATER ##
    ############################################

    def plotState(self):
        # Plot the position of each particle
        for prt in self.Particles:
            plt.plot(prt.x_0, prt.x_1, 'o', label="Particle "+prt.ID)
        plt.xlim([-0.5, self.Ng - 0.5])
        plt.ylim([-0.5, self.Ng - 0.5])
        plt.xlabel("x-position")
        plt.ylabel("y-position")
        plt.title("Particle Trajectories")
        plt.grid()
        plt.show()

    def plotCharge(self):
        # Plot the charge at every point on the grid
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
        # Plot the electric field as a vector field on the grid
        plt.quiver(self.EField_0, self.EField_1)
        plt.xlabel("X Grid Position [Intervals]")
        plt.ylabel("Y Grid Position [Intervals]")
        plt.title("Grid EField")
        plt.grid()
        plt.show()

    def plotPotential(self):
        # Plot the potential at every point on the grid
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
                ax.plot(prt.x_0[int(t*fps)%len(prt.x_0)], prt.x_1[int(t*fps)%len(prt.x_1)], 'o', label="Particle " + prt.ID)
                ax.set_xlim([-0.5, self.Ng - 0.5])
                ax.set_ylim([-0.5, self.Ng - 0.5])

            # returning numpy image
            return mplfig_to_npimage(fig)

        # creating animation
        animation = VideoClip(make_frame, duration=duration)

        # displaying animation with auto play and looping
        animation.ipython_display(fps=fps, loop=False, autoplay=True)
