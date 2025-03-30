import numpy as np 

class NBodySimulation:

    def __init__(self, num_bodies, G = 1 , softening = 0.1, dt = 0.1):
        """
        Initializes the N-body simulation.

        parameters:
        num_bodies: int, the number of bodies in the simultion 
        G: float, Gravitaitonal constant
        softening: float, Softening parameter to prevent numerical errors
        dt: float, stands for delta time
        """
        self.num_bodies = num_bodies
        self.G = G
        self.softening = softening
        self.dt = dt
    
        # Define structured arrays
        self.body_dtype = np.dtype([
            ('mass', float),
            ('position', float, (3,)),
            ('velocity', float, (3,))
        ])

        # Define array for bodies
        self.bodies = np.zeros(num_bodies, dtype=self.body_dtype)

        # seperate the arrays into mass, position, velocity and acceleration
        self.masses = np.zeros(num_bodies)
        self.positions = np.zeros((num_bodies, 3))
        self.velocities = np.zeros((num_bodies, 3))
        self.accelerations = np.zeros((num_bodies, 3))

    def set_initial_conditions(self, masses, positions, velocities):
        pass

'''# small test to make sure all the arrays are in proper order
test_input = np.array([
    (1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    (2.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    (3.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
], dtype=[
    ('mass', float),
    ('position', float, (3,)),
    ('velocity', float, (3,))
])

num_bodies = 3
sim = NBodySimulation(num_bodies)
sim.bodies = test_input

# Print to confirm is correct
for i in range(num_bodies):
    print("Body mass:", sim.bodies[i]['mass'])
    print("Body position:", sim.bodies[i]['position'])
    print("Body velocity:", sim.bodies[i]['velocity'])
'''