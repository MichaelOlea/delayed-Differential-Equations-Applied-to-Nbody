import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D

class NBodySimulation:

    def __init__(self, num_bodies, G = 1 , softening = 0.1, dt = 0.1, integrator="euler", gravity_speed = float('inf')):

        # Store simulation parameters
        self.num_bodies = num_bodies
        self.G = G
        self.softening = softening
        self.dt = dt
        self.integrator = integrator.lower() # covert to lower case
        self.gravity_speed = gravity_speed
        self.current_time = 0.0

    
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
        self.position_history = []
        self.time_history = []

    def set_initial_conditions(self, masses, positions, velocities):

        # loop through each body and assign mass, posittion and velocity
        for  i in range(self.num_bodies):
            self.bodies[i]['mass'] = masses[i]
            self.bodies[i]['position'] = positions[i]
            self.bodies[i]['velocity'] = velocities[i]

        # store data in arrays
        self.masses[:] = masses
        self.positions[:] = positions
        self.velocities[:] = velocities
        self.position_history.append(positions.copy())
        self.time_history.append(self.current_time)

    def get_delayed_position(self, i_position, j, current_time):
        # if gravity speed if infinite, no dealy
        if np.isinf(self.gravity_speed):
            r_ij = i_position - self.positions[j]
            dist_sqr = np.sum(r_ij**2) + self.softening**2
            inv_dist_cube = 1.0 / (dist_sqr * np.sqrt(dist_sqr))
            return r_ij, inv_dist_cube
        
        # Find position based on speed of gravity
        for fram_idx in range(len(self.time_history) -1, -1, -1):

            # calc time delay
            time_delay = current_time - self.time_history[fram_idx]

            # get position
            j_position = self.position_history[fram_idx][j]

            # calc distnace
            r_ij = i_position - j_position
            dist_ij = np.sqrt(np.sum(r_ij**2) + self.softening**2)

            # Check if correct frame
            if time_delay >= dist_ij / self.gravity_speed:
                inv_dist_cube = 1.0 / (dist_ij**3)
                return r_ij, inv_dist_cube
            
        # if no suitbale time frame, ooooof
        r_ij = i_position - self.position_history[0][j]
        return r_ij, 0.0


    def calculate_acceleration(self, positions, velocities = None, time = None, full_derivative = False):
        
        if positions is None:
            positions = self.positions

        if time is None:
            time = self.current_time

        accelerations = np.zeros_like(positions) # make sure array is the same shape and filled with zeros

        # calculate accelerations
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i != j: # skip if index is the same
                    
                    r_ij, inv_dist_cube = self.get_delayed_position(positions[i], j, time)
                    
                    # acc = G*M/r^3
                    accelerations[i] += -self.G * self.masses[j] * r_ij * inv_dist_cube

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
        self.current_time += self.dt

        # update array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]

        # add current position to history
        self.position_history.append(self.positions.copy())
        self.time_history.append(self.current_time)

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
        self.current_time += self.dt

        # update bodies array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]

        self.position_history.append(self.positions.copy())
        self.time_history.append(self.current_time)

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
        self.current_time += self.dt

        # update bodies array
        for i in range(self.num_bodies):
            self.bodies[i]['position'] = self.positions[i]
            self.bodies[i]['velocity'] = self.velocities[i]

        self.position_history.append(self.positions.copy())
        self.time_history.append(self.current_time)
    
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
        # Clear existing history and start fresh
        self.position_history = [self.positions.copy()]
        self.time_history = [self.current_time]
        
        # Add initial positions
        history = []
        history.append(self.positions.copy())
        
        for i in range(frames):
            # Use the choose_integrator method instead of if-else
            self.choose_integrator()
            
            # Record positions after integration step
            history.append(self.positions.copy())

        return np.array(history)
    
    @classmethod
    def load_data(cls, file_name, G = 1, softening = 0, dt = 0.1, integrator = "euler", gravity_speed=float('inf')):

        df = pd.read_csv(file_name) # load data frame
    
        # Extract values as float arrays
        masses = df["mass"].to_numpy(dtype=float)
        positions = df[["x", "y", "z"]].to_numpy(dtype=float)
        velocities = df[["v_x", "v_y", "v_z"]].to_numpy(dtype=float)

        # Creat and initialize simulation
        sim = cls(num_bodies=len(masses), G = G, softening = softening, dt = dt, integrator = integrator, gravity_speed = gravity_speed)

        sim.set_initial_conditions(masses, positions, velocities) # set inital conditions

        return sim

# setup simulation
sim = NBodySimulation.load_data("data1.csv", dt=0.01, integrator='euler')

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

    print(f"\rrendering frame: {frame}", end="") # update which frame is being redered

    # find center of system to center camera
    current_position = position_history[frame]
    center_mass = current_position.mean(axis = 0)

    # Autoscale view
    max_distance = np.max(np.linalg.norm(current_position - center_mass, axis = 1))
    ax.set_xlim(center_mass[0] - max_distance, center_mass[0] + max_distance)
    ax.set_ylim(center_mass[1] - max_distance, center_mass[1] + max_distance)
    ax.set_zlim(center_mass[2] - max_distance, center_mass[2] + max_distance)

    for i in range(num_bodies):
        points[i].set_data([current_position[i, 0]], [current_position[i, 1]])
        points[i].set_3d_properties(current_position[i, 2])
        tails[i].set_data(position_history[:frame, i, 0], position_history[:frame, i, 1])
        tails[i].set_3d_properties(position_history[:frame, i, 2])
    return points + tails


# Run animation
anim = FuncAnimation(fig, update, frames=len(position_history), interval=10)
anim.save('nbody_simulation.mp4', writer='ffmpeg')