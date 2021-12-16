# Particle-in-Cell Method for Plasma Simulation

## Authors:

- Kyle Fridberg
- Kevin Howarth
- Hari Raval

## Course:

- AM 205

## Files:

- classes.py: Particle and Grid representations for 1D simulations
- stepper.py: Particle-in-cell algorithms for 1D simulations 
- driver.py: Wrapper script to execute program for 1D simulations
- classes2D.py: Particle and Grid representations for 2D simulations
- stepper2D.py: Particle-in-cell algorithms for 2D simulations
- driver2D.py: Wrapper script to execute program for 2D simulations
- accuracyTests.py: Empirical tests for validating particle simulation accuracy 

## Dependencies (Python 3 installation assumed):

- pip install numpy
- pip install moviepy


## How to Use:

1. Clone GitHub repository via git clone https://github.com/KHowarth422/TinyPIC.git
2. Generate a CSV of particles (see "Particles CSV Format" below)
3. run a driver script: >> python driver.py
4. Enter the the grid size when prompted
5. Enter the desired number of particles when prompted
6. Enter the time-step when prompted
7. Select the CSV file prompt created in 2. when file browser pops up
8. View simulated plots and animation

## Particles CSV Format

- All CSV files containing particle data should be set up with three columns and any desired number of rows. The first row should contain the headers associated with a particle's ID, position, and velocity. Each subsequent row contains the associated data for each particle. The particle ID must be a string and the position and velocity can be integer or float values.

- Below, we include a sample CSV set-up for 6 particles: 


| ParticleID | Position | Velocity |
| -----------| ---------|----------|
|     1      |   12.8   |   1.0    |
|     2      |   25.6   |   0.6    |
|     3      |   38.4   |   1.0    |
|     4      |   57.6   |   -0.5   |
|     5      |   76.8   |   2.0    |
|     6      |   102.4  |   1.0    |