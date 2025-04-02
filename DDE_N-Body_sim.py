"""
Delayed Gravity N-Body Simulation using Euler, Runge-Kutta 4, and Yoshida Integrators
---------------------------------------------------------------------

Author: Michael Gonzalez
Date: Apr-01-2025

Description:
    This script simulates an N-body gravitational system in 3D space. 
    It supports multiple integrators (Euler, RK4, Yoshida), calculates 
    gravitational forces with softening, and visualizes the results with
    a 3D animated plot using matplotlib.

Features:
    - Structured numpy array for body properties
    - Multiple integration methods
    - Core radius calculation based on mass distribution
    - Optional CSV input to load initial body conditions
    - 3D animated visualization with colored trails

Dependencies:
    - numpy
    - pandas
    - matplotlib

Usage:
    1. Create a CSV file (e.g., data1.csv) with the following columns:
        mass, x, y, z, v_x, v_y, v_z
    2. Run the script. An animation will be saved as `nbody_simulation.mp4`.

    Special Note: This is an update simulation from a previous research project.  
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D

class NBodySimulation:

    def __init__(self, num_bodies, G = 1 , softening = 0.1, dt = 0.1, integrator="euler", gravity_speed = float('inf')):
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
        self.integrator = integrator.lower() # covert to lower case
        self.gravity_speed = gravity_speed
        self.current_time = 0.0

    
        # Define the structured data tpyes for each body. Each body has mass and a 3D vector for position and velocity
        self.body_dtype = np.dtype([
            ('mass', float),
            ('position', float, (3,)),
            ('velocity', float, (3,))
        ])

        # Initialize structured array to hold all bodies 
        self.bodies = np.zeros(num_bodies, dtype=self.body_dtype)

        # separate the arrays into mass, position, velocity and acceleration
        self.masses = np.zeros(num_bodies)
        self.positions = np.zeros((num_bodies, 3))
        self.velocities = np.zeros((num_bodies, 3))
        self.accelerations = np.zeros((num_bodies, 3))
        self.position_history = []
        self.time_history = []

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

        # Loop through each body and assign mass, position and velocity
        for  i in range(self.num_bodies):
            self.bodies[i]['mass'] = masses[i]
            self.bodies[i]['position'] = positions[i]
            self.bodies[i]['velocity'] = velocities[i]

        # Store data in arrays
        self.masses[:] = masses
        self.positions[:] = positions
        self.velocities[:] = velocities
        self.position_history.append(positions.copy())
        self.time_history.append(self.current_time)

    def get_delayed_position(self, i_position, j, current_time):
        """
        Compute the vector from body i to body j, accounting for the finite speed of gravity.

        Parameters
        ----------
        i_position: np.ndarray, shape (3,))
            The current position of body i
        j: integer
            Index of body j, the source of gravitational influence.
        current_time: float
            The simulation's current time.

        Returns
        -------
        r_ij : np.ndarray
            The vector pointing from body j to body i.
        inv_dist_cube : float
            The inverse cube of the softened distance between the two bodies, used in gravitational force calculations. If the delay condition is not met, returns 0 to represent no influence yet.
        """
        # If gravity speed if infinite, no delay
        if np.isinf(self.gravity_speed):
            r_ij = i_position - self.positions[j]
            dist_sqr = np.sum(r_ij**2) + self.softening**2
            inv_dist_cube = 1.0 / (dist_sqr * np.sqrt(dist_sqr))
            return r_ij, inv_dist_cube
        
        # Find position
        for fram_idx in range(len(self.time_history) -1, -1, -1):

            # Calculate time delay
            time_delay = current_time - self.time_history[fram_idx]

            # Get position
            j_position = self.position_history[fram_idx][j]

            # Calculate distnace
            r_ij = i_position - j_position
            dist_ij = np.sqrt(np.sum(r_ij**2) + self.softening**2)

            # Check if correct frame
            if time_delay >= dist_ij / self.gravity_speed:
                inv_dist_cube = 1.0 / (dist_ij**3)
                return r_ij, inv_dist_cube
            
        # If no suitable time frame, ooooof
        r_ij = i_position - self.position_history[0][j]
        return r_ij, 0.0


    def calculate_acceleration(self, positions, velocities = None, time = None, full_derivative = False):
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
        
        if positions is None:
            positions = self.positions

        if time is None:
            time = self.current_time

        accelerations = np.zeros_like(positions) # make sure array is the same shape and filled with zeros

        # Calculate accelerations
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i != j: # skip if index is the same
                    
                    r_ij, inv_dist_cube = self.get_delayed_position(positions[i], j, time)
                    
                    accelerations[i] += -self.G * self.masses[j] * r_ij * inv_dist_cube # acc = G*M/r^3

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
        self.current_time += self.dt

        # Update array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]

        # Add current position to history
        self.position_history.append(self.positions.copy())
        self.time_history.append(self.current_time)

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
        self.current_time += self.dt

        # Update bodies array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]

        self.position_history.append(self.positions.copy())
        self.time_history.append(self.current_time)

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

        # Update lass variables
        self.positions = positions
        self.velocities = velocities
        self.current_time += self.dt

        # Update bodies array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]

        self.position_history.append(self.positions.copy())
        self.time_history.append(self.current_time)
    
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

        # Clear existing history and start fresh
        self.position_history = [self.positions.copy()]
        self.time_history = [self.current_time]
        
        # Add initial positions
        history = []
        history.append(self.positions.copy())
        
        for i in range(frames):
            # Use the choose_integrator method
            self.choose_integrator()
            
            # Record positions after integration step
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

        # Find center of mass
        total_mass = np.sum(masses) # find total mass
        center_of_mass = np.sum(positions * masses[:, np.newaxis], axis=0) / np.sum(masses) # find the center of mass
        
        # Distances from center of mass
        reltive_positions = positions - center_of_mass # find distances from center of mass
        distances = np.linalg.norm(reltive_positions, axis = 1) # store distances in an array

        # Sort the bodies by distance
        sorted_indices = np.argsort(distances) # sorted array 
        sorted_distances = distances[sorted_indices] # array of distances closest from furthest
        sorted_masses = masses[sorted_indices] # array of masses closest to furthest

        # Cumulative mass as we move outward
        cumulative_mass = np.cumsum(sorted_masses)

        # Distance that encloses n% of the total mass
        n_mass = 0.6 * total_mass
        idx = np.searchsorted(cumulative_mass, n_mass) # first index that cumulative mass >= n% of mass

        # Make sure index is not out of range
        if idx >= len(sorted_distances):
            idx = len(sorted_distances) - 1
        core_radius = sorted_distances[idx]
        
        return center_of_mass, core_radius
    
    @classmethod
    def load_data(cls, file_name, G = 1, softening = 0, dt = 0.1, integrator = "euler", gravity_speed=float('inf')):

        df = pd.read_csv(file_name) # load data frame
    
        # Extract values as float arrays
        masses = df["mass"].to_numpy(dtype=float)
        positions = df[["x", "y", "z"]].to_numpy(dtype=float)
        velocities = df[["v_x", "v_y", "v_z"]].to_numpy(dtype=float)

        # Create and initialize simulation
        sim = cls(num_bodies=len(masses), G = G, softening = softening, dt = dt, integrator = integrator, gravity_speed = gravity_speed)

        sim.set_initial_conditions(masses, positions, velocities) # set initial conditions

        return sim

# Setup simulation
sim = NBodySimulation.load_data("data1.csv", dt=0.01, integrator='euler')

# Run simulation
position_history = sim.run(frames=500)

# Setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_box_aspect([1, 1, 1])

# Setup colors
num_bodies = position_history.shape[1]
colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'lime', 'teal']
colors = colors * ((num_bodies + len(colors) - 1) // len(colors))  # Repeat if not enough colors

# Create scatter points and tails
points = [ax.plot([], [], [], 'o', color = colors[i])[0] for i in range(num_bodies)]
tails = [Line3D([], [], [], color = colors[i]) for i in range(num_bodies)]
for tail in tails:
    ax.add_line(tail)

# Animation update functions
def update(frame):

    print(f"\rrendering frame: {frame}", end="") # show progress

    # Ger current position, get center of mass and core radius
    current_position = position_history[frame]
    center_of_mass, core_radius = NBodySimulation.calculate_core_radius(current_position, sim.masses)

    # Set plot limits 
    ax.set_xlim(center_of_mass[0] - core_radius, center_of_mass[0] + core_radius)
    ax.set_ylim(center_of_mass[1] - core_radius, center_of_mass[1] + core_radius)
    ax.set_zlim(center_of_mass[2] - core_radius, center_of_mass[2] + core_radius)

    # Update positions of points and tails
    for i in range(num_bodies):
        points[i].set_data([current_position[i, 0]], [current_position[i, 1]])
        points[i].set_3d_properties(current_position[i, 2])
        
        # The tail for the body (all previous frames)
        tails[i].set_data(position_history[:frame, i, 0], position_history[:frame, i, 1])
        tails[i].set_3d_properties(position_history[:frame, i, 2])

    return points + tails


# Run animation
anim = FuncAnimation(fig, update, frames=len(position_history), interval=10)
anim.save('nbody_simulation.mp4', writer='ffmpeg')