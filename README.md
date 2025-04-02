# Delayed Differential Equations Applied to N-body Simulations

This repository contains code for simulating N-body gravitational systems with both traditional and delayed gravity implementations.

## Overview

This project investigates the effects of finite gravity propagation speed on N-body gravitational simulations. It includes two main simulation implementations:

1. **Standard N-body Simulation** (`N-Body_sim.py`): Traditional implementation where gravitational influences propagate instantaneously.

2. **Delayed Gravity N-body Simulation** (`DDE_N-Body_sim.py`): Advanced implementation that accounts for the finite speed of gravity propagation.

Both simulations support multiple numerical integration methods and produce 3D visualizations of the resulting celestial body movements.

## Features

- Multiple integration methods:
  - Euler (first-order)
  - 4th-order Runge-Kutta (RK4)
  - 4th-stage Yoshida symplectic integrator
- Gravitational force calculation with adjustable softening parameter
- Core radius calculation based on mass distribution
- CSV input for initial body conditions
- 3D animated visualization with colored trails
- Configurable time step and gravitational constant
- Finite speed of gravity propagation (in the DDE version)

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- ffmpeg (for saving animations)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/delayed-Differential-Equations-Applied-to-Nbody.git
cd delayed-Differential-Equations-Applied-to-Nbody

# Install required packages
pip install numpy pandas matplotlib
```

## Usage

### Input Data Format

Create a CSV file (e.g., `data1.csv`) with the following columns:
- `mass`: Mass of the body
- `x`, `y`, `z`: Initial position coordinates
- `v_x`, `v_y`, `v_z`: Initial velocity components

### Running Simulations

#### Standard N-body Simulation:

```python
# Inside N-Body_sim.py
sim = NBodySimulation.load_data("data1.csv", dt=0.01, integrator='yoshida')
position_history = sim.run(frames=500)
```

#### Delayed Gravity Simulation:

```python
# Inside DDE_N-Body_sim.py
sim = NBodySimulation.load_data("data1.csv", dt=0.01, integrator='euler', gravity_speed=10.0)
position_history = sim.run(frames=500)
```

## Integration Methods

1. **Euler**: Simple first-order method. Fast but less accurate for longer simulations.
2. **RK4**: Fourth-order Runge-Kutta method. Better accuracy than Euler but more computationally intensive.
3. **Yoshida**: Fourth-order symplectic integrator. Excellent energy conservation properties, ideal for long-term gravitational simulations.

## Parameters

- `num_bodies`: Number of bodies in the simulation
- `G`: Gravitational constant (default: 1)
- `softening`: Softening parameter to avoid numerical singularities (default: 0.1)
- `dt`: Time step size (default: 0.1)
- `integrator`: Integration method ("euler", "rk4", or "yoshida")
- `gravity_speed`: Speed of gravity propagation (only in DDE version, default: infinity)

## Output

The simulations generate an animated 3D visualization saved as `nbody_simulation.mp4`, showing the movements of all bodies with colored trails.

## Delayed Gravity Implementation

The delayed gravity implementation in `DDE_N-Body_sim.py` tracks the position history of all bodies and calculates gravitational influences based on the finite propagation speed of gravity. This means that each body responds to the gravitational influence of other bodies' positions from the past, based on their distance and the speed of gravity.

Key components:
- Position and time history tracking
- Delayed position calculation based on light/gravity travel time
- Influence weighting based on propagation delay

## Author

Michael Gonzalez (April 2025)

## Acknowledgments

Thank you professor Dylan Gustafson for guiding me through this process.
