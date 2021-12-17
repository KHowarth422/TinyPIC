# Authors: Kyle Fridberg, Kevin Howarth, Hari Raval                             #
# Course: AM 205                                                                #
# File: driver2D.py                                                             #
# Description: Run a 2D particle in cell simulation for a desired number of     #
# cells, inputted initial conditions, and a custom grid size. This file is the  #
# only one which the user should interact with to run their simulations. The    #
# required files to run this driver script are classes2D.py and stepper2D.py.   #
#################################################################################
from classes2D import Particle2D
from stepper2D import run_simulations
import csv
import tkinter as tk
import time
from tkinter import filedialog


def initiate_simulations():

    """

    Pass in input necessary to run particle in cell simulation and visualize simulation results

    ** NOTE: User is required to pass in the following inputs as requested by the program prompts **

    L: length of simulation grid (integer)
    Ng: number of cells in grid (integer)
    dt: time step size (integer or float)
    T: end simulation time
    particles: list of Particle1D objects to be simulated (list)

    Parameters
    ----------
    None

    Raises
    -------
    None

    Returns
    -------
    None

    Example
    --------
    # run the script
    >>> python3 driver.py
    # choose a grid length when prompted
    >>> L = 64
    # choose a number of cells to use in the grid when prompted
    >>> Ng = 64
    # choose a time step when prompted
    >>> dt = 0.125
     # choose a simulation end time (in seconds) when prompted
    >>> T = 25
    # select a formatted CSV file when prompted by the file browser menu
    >>> "select particle_2D_input_ex_2.csv from TinyPIC/sample_csv_inputs/two_dimensional_inputs/ "

    """

    # read in L, Ng, dt, and T
    L = int(input("Please enter the desired grid length (integer): "))
    Ng = int(input("Please enter the desired number of grid points: "))
    dt = float(input("Please enter the desired time-step: "))
    T = float(input("Please enter the desired simulation time: "))
    print("Please select the formatted particles csv file from the menu: ")
    time.sleep(1)
    particles = []

    # read in the input particles file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    # generate a list of input particles from the user formatted CSV
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                pass
            else:
                particles.append(Particle2D(ID=str(row[0]), x0=float(row[1]), x1=float(row[2]),
                                            v0=float(row[3]), v1=float(row[4])))

    # run the simulations
    run_simulations(L, Ng, dt, T, particles)


initiate_simulations()
