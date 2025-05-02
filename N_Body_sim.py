"""
N-Body Simulation using Euler, Runge-Kutta 4, and Yoshida Integrators
---------------------------------------------------------------------

Author: Michael Gonzalez
Date: May-01-2025

Description:
    This script simulates an N-body gravitational system in 3D space. 
    It supports multiple integrators (Euler, RK4, Yoshida), calculates 
    gravitational forces with softening.

Features:
    - Structured numpy array for body properties
    - Multiple integration methods
    - Core radius calculation based on mass distribution
    - Optional CSV input to load initial body conditions

Dependencies:
    - numpy
    - pandas

Usage:
    1. Create a CSV file (e.g., data1.csv) with the following columns:
        mass, x, y, z, v_x, v_y, v_z

    Special Note: This is an update simulation from a previous research project.  
"""

import numpy as np
import pandas as pd


class NBodySimulation:

    def __init__(self, num_bodies, G = 1 , softening = 0.1, dt = 0.1, integrator="euler"):
        """
        Initialize the N-body simulation.
        Parameters
        -------------
        num_bodies: Integer
            Number of bodies in the simulation
        G: float
            Gravitational constant, Default is 1
        softening: float
            Softening parameter to avoid numerical error. Default is 0.1
        dt: float
            Time step size, default is 0.1.
        integrator: string
            Integration method ("euler”, “rk4”, “yoshida”). Default is “euler”. 
        """
        
        # Store simulation parameters
        self.num_bodies = num_bodies
        self.G = G
        self.softening = softening
        self.dt = dt
        self.integrator = integrator.lower() # convert to lower case

    
        # Define the structured data types for each body. Each body has mass and a 3D vector for position and velocity.
        self.body_dtype = np.dtype([
            ('mass', float),
            ('position', float, (3,)),
            ('velocity', float, (3,))
        ])

        # Initialize structured array to hold all bodies.
        self.bodies = np.zeros(num_bodies, dtype=self.body_dtype)

        # Separate the arrays into mass, position, velocity and acceleration.
        self.masses = np.zeros(num_bodies)
        self.positions = np.zeros((num_bodies, 3))
        self.velocities = np.zeros((num_bodies, 3))
        self.accelerations = np.zeros((num_bodies, 3))

    def set_initial_conditions(self, masses, positions, velocities):
        """
        Set the initial conditions for the simulation.
        Parameters
        -------------
        masses: np.ndarray, shape (n, )
            Masses of each body.
        Positions: np.ndarray, shape (n, 3)
            Initial 3D positions of each body.
        velocities: np.array, shape (n, 3)
            Initial 3D positions of each body.
        """

        # Loop through each body and assign mass, position and velocity.
        for  i in range(self.num_bodies):
            self.bodies[i]['mass'] = masses[i]
            self.bodies[i]['position'] = positions[i]
            self.bodies[i]['velocity'] = velocities[i]

        # store data in arrays
        self.masses[:] = masses
        self.positions[:] = positions
        self.velocities[:] = velocities

    def calculate_acceleration(self, positions, velocities = None, full_derivative = False):
        """
        Calculate the gravitational acceleration for all the bodies in the system.

        Parameters
        ----------
        positions: np.ndarray
            Current positions of the bodies.
        velocities: np.ndarray
            Current velocities (requited if full_derivative is True).
        full_derivative: bool
            If True return concatenated [velocities, acceleration] array.

        Returns
        -------
        np.ndarray
            Acceleration or [velocities, acceleration] array.
        """

        accelerations = np.zeros_like(positions) # make sure accelerations array is the same shape of positions array and filled with zeros

        # calculate accelerations
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i != j: # skip if index is the same
                    r_ij = positions[j] - positions[i]
                    dist_sqr = np.sum(r_ij**2) + self.softening**2
                    inv_dist_cube = 1.0 / (dist_sqr * np.sqrt(dist_sqr))
                    
                    accelerations[i] += self.G * self.masses[j] * r_ij * inv_dist_cube # acc = G*M/r^3

        # Return accelerations or accelerations and velocities
        if full_derivative:
            if velocities is None:
                raise ValueError("Velocities need to be given for (blank) integrator")
            return np.concatenate([velocities, accelerations])
        else:
            return accelerations
        
    def euler(self):
        """
        Performs one integration step using Euler method.     
        """
        # Calculate accelerations
        accelerations = self.calculate_acceleration(self.positions)

        # Update velocities and positions
        self.velocities += accelerations * self.dt
        self.positions += self.velocities * self.dt

        # Update array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]


    def rung_kutta_4(self):
        """"
        Performs one integration step using Runge-Kutta 4th order integration. 
        """

        # Copy current states
        current_positions = self.positions.copy()
        current_velocities = self.velocities.copy()

        # Calculate k4 term
        k1_acc = self.calculate_acceleration(current_positions)
        k1_vel = current_velocities.copy()

        # Calculate k2 term
        k2_pos = current_positions + 0.5 * self.dt * k1_vel
        k2_acc = self.calculate_acceleration(k2_pos)
        k2_vel = current_velocities + 0.5 * self.dt *k1_acc
    
        # Calculate k3 term
        k3_pos = current_positions + 0.5 * self.dt * k2_vel
        k3_acc = self.calculate_acceleration(k3_pos)
        k3_vel = current_velocities + 0.5 * self.dt *k2_acc

        # Calculate k4 term
        k4_pos = current_positions + self.dt * k3_vel
        k4_acc = self.calculate_acceleration(k4_pos)
        k4_vel = current_velocities + self.dt *k3_acc

        # Update position and velocities
        self.positions += self.dt / 6.0 * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)
        self.velocities += self.dt / 6.0 * (k1_acc + 2 * k2_acc + 2 * k3_acc + k4_acc)

        # Update bodies array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]

    def yoshida(self):
        """
        Perform one step using the 4-stage Yoshida integrator.
        """
        # Yoshida coefficients
        cr2 = 2 ** (1/3)
        w0 = -cr2 / (2 - cr2)
        w1 = 1 / (2 - cr2)
        c1 = w1 / 2
        c2 = (w0 + w1) / 2

        # Copy current states
        positions = self.positions.copy()
        velocities = self.velocities.copy()

        # Step 1
        positions += self.dt * c1 * velocities
        accelerations = self.calculate_acceleration(positions)
        velocities += self.dt * w1 * accelerations

        # Step 2
        positions += self.dt * c2 * velocities
        accelerations = self.calculate_acceleration(positions)
        velocities += self.dt * w0 * accelerations
        
        # Step 3
        positions += self.dt * c2 * velocities
        accelerations = self.calculate_acceleration(positions)
        velocities += self.dt * w1 * accelerations

        # Step 4
        positions += self.dt * c1 * velocities

        # Update last variables
        self.positions = positions
        self.velocities = velocities

        # Update bodies array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]
    
    def choose_integrator(self):
        """
        Applies the selected integration method 
        """
        if self.integrator == "euler":
            self.euler()
        elif self.integrator == "rk4":
            self.rung_kutta_4()
        elif self.integrator == "yoshida":
            self.yoshida()
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")
        
    def run(self, frames):
        """
        Run the simulation for a given number of frames.

        Parameters
        ----------
        frames: integer
            Number of integration steps to simulate.

        Returns
        -------
        np.ndarray, shape (frames + 1, num_bodies, 3)
            History of body positions.
        """

        # Initialize history array and initial positions 
        history = []
        history.append(self.positions.copy())
        
        for i in range(frames):
            if self.integrator == "euler":
                self.euler()
            elif self.integrator == "rk4":
                self.rung_kutta_4()
            elif self.integrator == "yoshida":
                self.yoshida()
            else:
                raise ValueError(f"Unknown integrator: {self.integrator}")
            
            history.append(self.positions.copy())

        return np.array(history)
    
    @staticmethod
    def calculate_core_radius(positions, masses):
        """
        Calculate the center of mass and the core radius enclosing a given percentage of the total mass.
        
        Parameters
        ----------
        Positions: np.ndarray, shape (n. 3)
            Positions of the bodies.
        masses: np.ndarray, shape (n, )
            Masses of the bodies.

        Returns
        -------
        (center_of_mass, core_radius)
        """
        
        
        # Calculate the Center of mass of the system 
        total_mass = np.sum(masses) # find total mass
        center_of_mass = np.sum(positions * masses[:, np.newaxis], axis=0) / np.sum(masses) # find the center of mass
        
        #Calculate the distances of bodies from the center of mass 
        relative_positions = positions - center_of_mass # find distances from center of mass
        distances = np.linalg.norm(relative_positions, axis = 1) # store distances in an array

        # Sort the bodies by distance
        sorted_indices = np.argsort(distances) # sorted array closest to furthest 
        sorted_distances = distances[sorted_indices] # array of distances closest from furthest
        sorted_masses = masses[sorted_indices] # array of masses closest to furthest

        # Cumulative mass as move outward
        cumulative_mass = np.cumsum(sorted_masses)

        # Distance that encloses n% of the total mass
        n_mass = 0.6 * total_mass
        idx = np.searchsorted(cumulative_mass, n_mass) # first index that cummutuve mass >= n% of mass

        # Make sure index is not out of range
        if idx >= len(sorted_distances):
            idx = len(sorted_distances) - 1
        core_radius = sorted_distances[idx]
        
        return center_of_mass, core_radius



    @classmethod
    def load_data(cls, file_name, G = 1, softening = 0, dt = 0.1, integrator = "euler" ):
        """
        Load the particle data from CSV and create a simulation instance
        
        Parameters
        ----------
        file_name: string
            CSV file containing columns: mass, x, y, x, v_x, v_y, v_z.
        G: float
            Gravitational constant.
        softening: float
            Softening parameter.
        dt: float
            Time step.
        integrator: string
            Integration method ("euler, "rk4', "yoshida")

        Returns
        -------
        NBodySimulation
            Initialized simulation object.
        """

        df = pd.read_csv(file_name) # load data frame
    
        # Extract values as float arrays
        masses = df["mass"].to_numpy(dtype=float)
        positions = df[["x", "y", "z"]].to_numpy(dtype=float)
        velocities = df[["v_x", "v_y", "v_z"]].to_numpy(dtype=float)

        # Create and initialize simulation
        sim = cls(num_bodies = len(masses), G = G, softening = softening, dt = dt, integrator = integrator)
        sim.set_initial_conditions(masses, positions, velocities) # set inital conditions

        return sim