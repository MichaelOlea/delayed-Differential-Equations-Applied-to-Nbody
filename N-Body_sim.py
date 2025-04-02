import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D

class NBodySimulation:

    def __init__(self, num_bodies, G = 1 , softening = 0.1, dt = 0.1, integrator="euler"):
        """
        Initialize the N-body simulation.

        Parameters
        ----------
        num_bodies : int
        Number of bodies in the simulation.
        G : float, optional
        Gravitational constant. Default is 1.
        softening : float, optional
        Softening parameter to avoid division by zero at short distances. Default is 0.1.
        dt : float, optional
        Time step (delta t) for integration. Default is 0.1.
        integrator : str, optional
        Integration method to use (e.g., "euler", "rk4"). Default is "euler".
        """

        # Store simulation parameters
        self.num_bodies = num_bodies
        self.G = G
        self.softening = softening
        self.dt = dt
        self.integrator = integrator.lower() # covert to lower case

    
        # Define the structured data tpye for each body. Each body has mass and a 3D vector for position and velocity
        self.body_dtype = np.dtype([
            ('mass', float),
            ('position', float, (3,)),
            ('velocity', float, (3,))
        ])

        # Initialize structured array to hold all bodies 
        self.bodies = np.zeros(num_bodies, dtype=self.body_dtype)

        # seperate the arrays into mass, position, velocity and acceleration
        self.masses = np.zeros(num_bodies)
        self.positions = np.zeros((num_bodies, 3))
        self.velocities = np.zeros((num_bodies, 3))
        self.accelerations = np.zeros((num_bodies, 3))

    def set_initial_conditions(self, masses, positions, velocities):
        """
        Set initial conditions for simulation

        parameters
        ----------
        masses: np.ndarray, shape (n,)
            array of masses for each body
        positions: np.ndarray, shape (n,)
            initial 3D positions of each body
        velocities: np.ndarray, shape (n,)
            initial 3D velocities of each body
        """

        # loop through each body and assign mass, posittion and velocity
        for  i in range(self.num_bodies):
            self.bodies[i]['mass'] = masses[i]
            self.bodies[i]['position'] = positions[i]
            self.bodies[i]['velocity'] = velocities[i]

        # store data in arrays
        self.masses[:] = masses
        self.positions[:] = positions
        self.velocities[:] = velocities

    def calculate_acceleration(self, positions, velocities = None, time = None, full_derivative = False):
        
        accelerations = np.zeros_like(positions) # make sure array is the same shape and filled with zeros

        # calculate accelerations
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i != j: # skip if index is the same
                    r_ij = positions[j] - positions[i]
                    dist_sqr = np.sum(r_ij**2) + self.softening**2
                    inv_dist_cube = 1.0 / (dist_sqr * np.sqrt(dist_sqr))
                    
                    # acc = G*M/r^3
                    accelerations[i] += self.G * self.masses[j] * r_ij * inv_dist_cube

        # Return accelerations or accelerations and velocities
        if full_derivative:
            if velocities is None:
                raise ValueError("Velocities need to be given for (blank) integrator")
            return np.concatenate([velocities, accelerations])
        else:
            return accelerations
        
    def euler(self):
        """
        performs one integration step using Eulers method

        Updates each body's velocity and position based on the current
        acceleration and time step.        
        """
        # caculate accelerations
        accelerations = self.calculate_acceleration(self.positions)

        # Update velocities and positions
        self.velocities += accelerations * self.dt
        self.positions += self.velocities * self.dt

        # update array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]


    def rung_kutta_4(self):
        """"
        implmetion of 4th order runge kutta methods
        """

        # copy current states
        current_positions = self.positions.copy()
        current_velocities = self.velocities.copy()

        # calculate k4 term
        k1_acc = self.calculate_acceleration(current_positions)
        k1_vel = current_velocities.copy()

        # calculate k2 term
        k2_pos = current_positions + 0.5 * self.dt * k1_vel
        k2_acc = self.calculate_acceleration(k2_pos)
        k2_vel = current_velocities + 0.5 * self.dt *k1_acc
    
        # calculate k3 term
        k3_pos = current_positions + 0.5 * self.dt * k2_vel
        k3_acc = self.calculate_acceleration(k3_pos)
        k3_vel = current_velocities + 0.5 * self.dt *k2_acc

        # calculate k4 term
        k4_pos = current_positions + self.dt * k3_vel
        k4_acc = self.calculate_acceleration(k4_pos)
        k4_vel = current_velocities + self.dt *k3_acc

        # update position and velocities
        self.positions += self.dt / 6.0 * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)
        self.velocities += self.dt / 6.0 * (k1_acc + 2 * k2_acc + 2 * k3_acc + k4_acc)

        # update bodies array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]

    def yoshida(self):
        """
        Implmenation of yoshida integrator
        """
        # yoshida coefficients
        cr2 = 2 ** (1/3)
        w0 = -cr2 / (2 - cr2)
        w1 = 1 / (2 - cr2)
        c1 = w1 / 2
        c2 = (w0 + w1) / 2

        # copy current states
        positions = self.positions.copy()
        velocities = self.velocities.copy()

        # step 1
        positions += self.dt * c1 * velocities
        accelerations = self.calculate_acceleration(positions)
        velocities += self.dt * w1 * accelerations

        # step 2
        positions += self.dt * c2 * velocities
        accelerations = self.calculate_acceleration(positions)
        velocities += self.dt * w0 * accelerations
        
        # step 3
        positions += self.dt * c2 * velocities
        accelerations = self.calculate_acceleration(positions)
        velocities += self.dt * w1 * accelerations

        # step 4
        positions += self.dt * c1 * velocities

        # update lass variables
        self.positions = positions
        self.velocities = velocities

        # update bodies array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]
    
    def choose_integrator(self):
        """
        A general function that let you choose which integrator to apply (will adee more integrators later).
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
        update !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Run the simulations for a given number of frames

        Parameters
        ---------
        frames: intger
            Number of frames (time steps) to run simultion 

        Returns
        -------
        np.ndarray
            Position history with shape (frames, num_bodies, 3),
            the positions of all bodies over time.
        """

        history = []
        history.append(self.positions.copy()) # add initla positions 
        
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
    
    def calculate_core_radius(positions, masses):
        
        # find center of mass
        total_mass = np.sum(masses) # find total mass
        center_of_mass = np.sum(positions * masses[:, np.newaxis], axis=0) / np.sum(masses) # find the center of mass
        
        # Disances form center of mass
        reltive_positions = positions - center_of_mass # find distances from center of mass
        distances = np.linalg.norm(reltive_positions, axis = 1) # store distanes in an array

        # sort the bodies by distance
        sorted_indices = np.argsort(distances) # sorted array 
        sorted_distances = distances[sorted_indices] # array of distnaces closest from furthest
        sorted_masses = masses[sorted_indices] # array of masses clsest to furthest

        # cumulative mass as we move outward
        cumulative_mass = np.cumsum(sorted_masses)

        # distnace that encloses n% of the total mass
        n_mass = 0.6 * total_mass
        idx = np.searchsorted(cumulative_mass, n_mass) # first index that cummutuve mass >= n% of mass

        # make sure index is not out of range
        if idx >= len(sorted_distances):
            idx = len(sorted_distances) - 1
        core_radius = sorted_distances[idx]
        
        return center_of_mass, core_radius



    @classmethod
    def load_data(cls, file_name, G = 1, softening = 0, dt = 0.1, integrator = "euler" ):
        """
        Create a simulation instance by loading particle data from a CSV file.

        Parameters
        ----------
        file_name : string
            Name  CSV file containing columns: mass, x, y, z, v_x, v_y, v_z.
        G : float
            Gravitational constant. 
        softening : float, optional
            Softening parameter. 
        dt : float
            Time step for the simulation.
        integrator : string
            Integration method to use (e.g., "euler", "rk4")

        Returns
        -------
        NBodySimulation
            A fully initialized simulation object.
        """

        df = pd.read_csv(file_name) # load data frame
    
        # Extract values as float arrays
        masses = df["mass"].to_numpy(dtype=float)
        positions = df[["x", "y", "z"]].to_numpy(dtype=float)
        velocities = df[["v_x", "v_y", "v_z"]].to_numpy(dtype=float)

        # Creat and initialize simulation
        sim = cls(num_bodies = len(masses), G = G, softening = softening, dt = dt, integrator = integrator)
        sim.set_initial_conditions(masses, positions, velocities) # set inital conditions

        return sim

# setup simulation
sim = NBodySimulation.load_data("data1.csv", dt=0.01, integrator='yoshida')

# Run simulation
position_history = sim.run(frames=500)

# setup plot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_box_aspect([1, 1, 1])

# setup colors
num_bodies = position_history.shape[1]
colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'lime', 'teal']
colors = colors * ((num_bodies + len(colors) - 1) // len(colors))  # Repeat if not enough colors

# Creat scatter points and tails
points = [ax.plot([], [], [], 'o', color = colors[i])[0] for i in range(num_bodies)]
tails = [Line3D([], [], [], color = colors[i]) for i in range(num_bodies)]
for tail in tails:
    ax.add_line(tail)

# Animation update functions
def update(frame):

    print(f"\rrendering frame: {frame}", end="") # show progress

    # ge current position, get center of mass and core raduis
    current_position = position_history[frame]
    center_of_mass, core_radius = NBodySimulation.calculate_core_radius(current_position, sim.masses)

    # set plot limits 
    ax.set_xlim(center_of_mass[0] - core_radius, center_of_mass[0] + core_radius)
    ax.set_ylim(center_of_mass[1] - core_radius, center_of_mass[1] + core_radius)
    ax.set_zlim(center_of_mass[2] - core_radius, center_of_mass[2] + core_radius)

    # update positions of points and tails
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