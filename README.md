# Particle-in-Cell Method for Plasma Simulation

![The beauty of PIC!](https://github.com/KHowarth422/TinyPIC/blob/main/sample_csv_inputs/two_dimensional_inputs/swirl%202D.gif)

## Authors:

- Kyle Fridberg
- Kevin Howarth
- Hari Raval

## Course:

- AM 205

## Directory Structure:

- classes.py: Particle and Grid representations for 1D simulations
- stepper.py: Particle-in-cell algorithms for 1D simulations 
- driver.py: Wrapper script to execute program for 1D simulations
- classes2D.py: Particle and Grid representations for 2D simulations
- stepper2D.py: Particle-in-cell algorithms for 2D simulations
- driver2D.py: Wrapper script to execute program for 2D simulations
- sample_csv_inputs: Directory containing example input CSV files describing particles 
- accuracyTests.py: Empirical tests for validating particle simulation accuracy 

## Dependencies:

For the following dependencies, we assume the user has already installed Python 3

- pip install numpy
- pip install moviepy


## How to Use:

1. Clone GitHub repository: 'git clone https://github.com/KHowarth422/TinyPIC.git'
2. Generate a CSV of particles (see "Particles CSV Format" below)
3. run a driver script: 'python driver.py' (for 1D case) or 'python driver2D.py' (for 2D case)
4. Enter the the grid size when prompted
5. Enter the desired number of particles when prompted
6. Enter the time-step when prompted
7. Enter the simulation time when prompted
8. Select the CSV file prompt created in 2. when file browser pops up
9. View simulated plots and animation

## Particles CSV Format


### 1D Simulation Format

- All 1D simulation CSV files containing particle data should be set up with three columns and any desired number of rows. The first row should contain the headers associated with a particle's ID, position, and velocity. Each subsequent row contains the associated data for each particle. The particle ID must be a string and the position and velocity can be integer or float values.

- Below, we include a sample CSV set-up for 6 particles: 


| ParticleID | Position | Velocity |
| -----------| ---------|----------|
|     1      |   12.8   |   1.0    |
|     2      |   25.6   |   0.6    |
|     3      |   38.4   |   1.0    |
|     4      |   57.6   |   -0.5   |
|     5      |   76.8   |   2.0    |
|     6      |   102.4  |   1.0    |

- The directory sample_csv_inputs/one_dimensional_inputs/ contains sample .csv files

- Note: A sample file is provided containing particles initialized to simulate the two-stream instability.
However, we recommend that to simulate the two-stream instability, the user should run 
`python accuracyTests.py --twostream` as this will create an impressive animation in v-x phase space.


### 2D Simulation Format

- All 2D simulation CSV files containing particle data should be set up with five columns and any desired number of rows. The first row should contain the headers associated with a particle's ID, x position, y position, x velocity, and y velocity. Each subsequent row contains the associated data for each particle. The particle ID must be a string and the position and velocity can be integer or float values.

- Below, we include a sample CSV set-up for 3 particles: 


| ParticleID | Position X | Position Y | Velocity X | Velocity Y |
|------------|------------|------------|------------|------------|
|     1      |    10      |    32      |     1      |     0      |
|     2      |    32      |    32      |     0      |     0      |
|     3      |    15      |    45      |     0      |     1      |

- The directory sample_csv_inputs/two_dimensional_inputs/ contains sample .csv files. To recreate the gif shown at the
top of this README, use [particle_2D_input_ex_swirl.csv](https://github.com/KHowarth422/TinyPIC/blob/main/sample_csv_inputs/two_dimensional_inputs/particle_2D_input_ex_swirl.csv).