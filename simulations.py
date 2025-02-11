# Romain Puech, 2024
# Simulations

import numpy as np
import random
import time
import os
from PIL import Image
import re

from dataset import save_scenario

# Fire and smoke
def sigmoid(x, scale=1.0):
    """
    Sigmoid function to normalize probabilities.
    """
    return 1 / (1 + np.exp(-x/scale)) * scale

def get_wind_effect(wind_direction, from_cell, to_cell):
    """
    Determine the wind effect between two cells.
    wind_direction: direction of the wind as a string ("left", "right", "up", "down", or diagonals).
    from_cell: (x, y) coordinates of the wildfire cell.
    to_cell: (x, y) coordinates of the neighboring cell.
    Returns 1 if wind favors the spread, -1 if wind opposes the spread, 0 otherwise.
    """
    direction_map = {
        "left": (0, -1),
        "right": (0, 1),
        "up": (-1, 0),
        "down": (1, 0),
        "up-left": (-1, -1),
        "up-right": (-1, 1),
        "down-left": (1, -1),
        "down-right": (1, 1),
    }
    dx, dy = to_cell[0] - from_cell[0], to_cell[1] - from_cell[1]
    
    # Wind-favored direction
    if (dx, dy) == direction_map.get(wind_direction, (None, None)):
        return 1
    # Wind-opposed direction
    elif (dx, dy) == tuple(-i for i in direction_map.get(wind_direction, (None, None))):
        return -1
    return 0

def get_wind_effect(wind_direction, from_cell, to_cell):
    """
    Determine the wind effect between two cells.
    wind_direction: direction of the wind as a string ("left", "right", "up", "down", or diagonals).
    from_cell: (x, y) coordinates of the wildfire cell.
    to_cell: (x, y) coordinates of the neighboring cell.
    Returns 1 if wind favors the spread, -1 if wind opposes the spread, 0 otherwise.
    """
    direction_map = {
        "left": (0, -1),
        "right": (0, 1),
        "up": (-1, 0),
        "down": (1, 0),
        "up-left": (-1, -1),
        "up-right": (-1, 1),
        "down-left": (1, -1),
        "down-right": (1, 1),
    }
    
    # Calculate the delta (dx, dy) from the from_cell to the to_cell
    dx, dy = to_cell[0] - from_cell[0], to_cell[1] - from_cell[1]
    
    # Get the wind direction vector from the direction map
    wind_dx, wind_dy = direction_map.get(wind_direction, (None, None))

    # Handle the case where the wind direction doesn't exist in the map
    if wind_dx is None or wind_dy is None:
        return 0

    # Wind-favored direction: check if the direction matches the wind's direction
    if (dx, dy) == (wind_dx, wind_dy):
        return 1
    
    # Wind-opposed direction: check if the direction is the opposite of the wind's direction
    elif (dx, dy) == (-wind_dx, -wind_dy):
        return -1
    
    # No effect
    return 0


def propagate_wildfire(grid, burn_time, burn_counters, p, wind_speed, wind_direction):
    """
    Simulate wildfire propagation for one time step.
    grid: NxN numpy array representing the current wildfire state (0: not burning, 1: burning, 2: burnt).
    burn_time: number of steps a cell burns before becoming burnt.
    burn_counters: NxN numpy array tracking burn time for each burning cell.
    p: base probability of fire spreading.
    wind_speed: strength of the wind.
    wind_direction: direction of the wind as a string ("left", "right", "up", "down", or diagonals).
    Returns a new grid representing the next state and updated burn_counters.
    """
    N = grid.shape[0]
    grid_t_plus_1 = grid.copy()  # Copy grid at the start of the timestep
    
   # Propagation step
    for x in range(N):
        for y in range(N):
            if grid[x, y] == 1:  # Only consider burning cells from the previous timestep
                # Check all neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip the current cell
                        
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < N and 0 <= ny < N and grid_t_plus_1[nx, ny] == 0:  # Valid neighbor
                            wind_effect = get_wind_effect(wind_direction, (x, y), (nx, ny))
                            spread_probability = sigmoid(p + wind_speed * wind_effect)
                            
                            # Fire spreads based on the adjusted probability
                            if np.random.rand() < spread_probability:
                                grid_t_plus_1[nx, ny] = 1

                # update burning time/ burnt status
                burn_counters[x, y] += 1
                if burn_counters[x, y] >= burn_time:
                    grid_t_plus_1[x, y] = 2

    return grid_t_plus_1, burn_counters

def propagate_smoke(fire_grid,smoke_grid, wind_speed, wind_direction, diffusion_percentage=1):
    """
    Simulate wildfire propagation for one time step.
    grid: NxN numpy array representing the current wildfire state (0: not burning, 1: burning, 2: burnt).
    burn_time: number of steps a cell burns before becoming burnt.
    burn_counters: NxN numpy array tracking burn time for each burning cell.
    p: base probability of fire spreading.
    wind_speed: strength of the wind.
    wind_direction: direction of the wind as a string ("left", "right", "up", "down", or diagonals).
    Returns a new grid representing the next state and updated burn_counters.
    """
    N = fire_grid.shape[0]
    smoke_grid_t_plus_1 = np.zeros_like(smoke_grid)  # Copy grid at the start of the timestep
    
    # Propagation step
    for x in range(N):
        for y in range(N):
            initial_smoke = smoke_grid[x, y]
            if fire_grid[x, y] == 1:
                # fire ads a fixed amount of smoke each turn
                initial_smoke += 5
            # wind spread
            wind_spread_percentage = max(1,0.00*wind_speed)
            wind_spread = wind_spread_percentage*initial_smoke
            initial_smoke -= wind_spread
            # diffuse to all neighbors
            spread = (initial_smoke*diffusion_percentage)/8
            initial_smoke -= initial_smoke*diffusion_percentage
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the current cell
                    
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N:  # Valid neighbor
                        wind_effect = get_wind_effect(wind_direction, (x, y), (nx, ny))
                        smoke_grid_t_plus_1[nx, ny] += spread
                        smoke_grid_t_plus_1[nx, ny] += wind_spread*max(0,wind_effect)
            smoke_grid_t_plus_1[x,y] += initial_smoke

    return smoke_grid_t_plus_1

def propagate_smoke_biased(fire_grid,smoke_grid, wind_speed, wind_direction, diffusion_percentage=0.4):
    """
    Simulate wildfire propagation for one time step.
    grid: NxN numpy array representing the current wildfire state (0: not burning, 1: burning, 2: burnt).
    burn_time: number of steps a cell burns before becoming burnt.
    burn_counters: NxN numpy array tracking burn time for each burning cell.
    p: base probability of fire spreading.
    wind_speed: strength of the wind.
    wind_direction: direction of the wind as a string ("left", "right", "up", "down", or diagonals).
    Returns a new grid representing the next state and updated burn_counters.
    """
    N = fire_grid.shape[0]
    smoke_grid_t_plus_1 = smoke_grid.copy()  # Copy grid at the start of the timestep
    
    # Propagation step
    for x in range(N):
        for y in range(N):
            if fire_grid[x, y] == 1:
                # fire ads a fixed amount of smoke each turn
                smoke_grid_t_plus_1[x, y] += 3
            # wind spread
            wind_spread_percentage = max(1,0.1*wind_speed)
            wind_spread = wind_spread_percentage*smoke_grid[x,y]
            smoke_grid_t_plus_1[x, y] -= wind_spread
            # diffuse to all neighbors
            spread = (smoke_grid_t_plus_1[x,y]*diffusion_percentage)/8
            smoke_grid_t_plus_1[x, y] -= spread*8
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the current cell
                    
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N:  # Valid neighbor
                        wind_effect = get_wind_effect(wind_direction, (x, y), (nx, ny))
                        smoke_grid_t_plus_1[nx, ny] += spread
                        smoke_grid_t_plus_1[nx, ny] += wind_spread*max(0,wind_effect)

    return smoke_grid_t_plus_1

from concurrent.futures import ThreadPoolExecutor

def propagate_smoke_parallel(fire_grid, smoke_grid, wind_speed, wind_direction, diffusion_percentage=0.4, n_threads=4):
    N = fire_grid.shape[0]
    chunk_size = N // n_threads
    thread_inputs = []
    
    # Helper function to calculate diffusion for a chunk
    def process_chunk(start_chunk_index, end_chunk_index, smoke_grid):
        chunk_height = end_chunk_index - start_chunk_index
        smoke_chunk = smoke_grid[start_chunk_index:end_chunk_index].copy()
        position_thread = "first" if start_chunk_index == 0 else "last" if end_chunk_index == N else None
        
        for x in range(0, chunk_height):
            for y in range(N):
                if fire_grid[start_chunk_index + x, y] == 1:
                    # fire ads a fixed amount of smoke each turn
                    smoke_chunk[x,y] += 3
                # wind spread
                wind_spread_percentage = max(1, 0.1 * wind_speed)
                wind_spread = wind_spread_percentage * smoke_grid[start_chunk_index + x,y]
                smoke_chunk[x,y] -= wind_spread
                # diffuse to all neighbors
                spread = (smoke_chunk[x,y] * diffusion_percentage) / 8
                smoke_chunk[x,y] -= spread * 8
                # update the smoke information of neighboring cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue # Skip the current cell

                        nx, ny = x + dx, y + dy
                        if 0 <= nx < chunk_height and 0 <= ny < N:
                            wind_effect = get_wind_effect(wind_direction, (start_chunk_index + x, y), (start_chunk_index + nx, ny))
                            smoke_chunk[nx, ny] += spread
                            smoke_chunk[nx, ny] += wind_spread * max(0, wind_effect)
        if position_thread == 'first': 
            return smoke_chunk[:-1]  # Remove padding before returning
        elif position_thread == 'last':
            return smoke_chunk[1:]
        return smoke_chunk[1:-1]  # Remove padding before returning

    # Split the grid into chunks with padding
    for i in range(n_threads):
        start = i * chunk_size
        end = start + chunk_size if i < n_threads - 1 else N
        thread_inputs.append([max(0, start - 1), min(N, end + 1)])
    
    # Process chunks in parallel
    results = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_chunk, start, end, smoke_grid) for start, end in thread_inputs]
        results = [f.result() for f in futures]

    # Merge chunks back into the final grid
    smoke_grid_t_plus_1 = np.vstack(results)
    return smoke_grid_t_plus_1


def propagate_smoke_parallel2(fire_grid, smoke_grid, wind_speed, wind_direction, diffusion_percentage=0.4, n_threads=4):
    N = fire_grid.shape[0]
    chunk_size = N // n_threads
    padded_chunks = []
    
    # Helper function to calculate diffusion for a chunk
    def process_chunk(chunk, start_idx):
        chunk_height = chunk.shape[0]
        smoke_chunk = chunk.copy()
        
        for x in range(1, chunk_height - 1):
            for y in range(N):
                if fire_grid[start_idx + x - 1, y] == 1:  # Offset by padding
                    smoke_chunk[x, y] += 3
                
                wind_spread_percentage = max(1, 0.1 * wind_speed)
                wind_spread = wind_spread_percentage * smoke_chunk[x, y]
                smoke_chunk[x, y] -= wind_spread

                spread = (smoke_chunk[x, y] * diffusion_percentage) / 8
                smoke_chunk[x, y] -= spread * 8

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue

                        nx, ny = x + dx, y + dy
                        if 0 <= ny < N:
                            wind_effect = get_wind_effect(wind_direction, (start_idx + x - 1, y), (start_idx + nx - 1, ny))
                            smoke_chunk[nx, ny] += spread
                            smoke_chunk[nx, ny] += wind_spread * max(0, wind_effect)
        
        return smoke_chunk[1:-1]  # Remove padding before returning

    # Split the grid into chunks with padding
    for i in range(n_threads):
        start = i * chunk_size
        end = start + chunk_size if i < n_threads - 1 else N
        chunk = smoke_grid[max(0, start - 1):min(N, end + 1)].copy()
        padded_chunks.append((chunk, start))
    
    # Process chunks in parallel
    results = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_chunk, chunk, start) for chunk, start in padded_chunks]
        results = [f.result() for f in futures]

    # Merge chunks back into the final grid
    smoke_grid_t_plus_1 = np.vstack(results)
    return smoke_grid_t_plus_1







#######


###### Initialize simulation parameters
p = -2.5  # Base fire spread logodds
burn_time = 10  # Number of steps before a burning cell becomes burnt

wind_speed = 3  # Wind strength
wind_direction = "right"  # Wind direction

diffusion_percentage = 0.4 # Percentage of smoke that diffuses to neighboring cells
n_drones = 5
#####

# def run_simulation(initial_wildfire_cells, horizon):
#     ###### Initialize grids
#     grid = np.zeros((N, N), dtype=int)
#     for x, y in initial_wildfire_cells:
#         grid[x, y] = 1
#     burn_counters = np.zeros((N, N), dtype=int) # for how long the cell has been burning
#     smoke_grid = np.zeros((N, N), dtype=float)
#     smoke_grid2 = np.zeros((N, N), dtype=float)

#     # drones = [Drone(*np.random.randint(0,N,2)) for _ in range(n_drones)]

#     # print("Initial Grid:")
#     #display_grid(grid,smoke_grid,drones,{'fire','smoke','drones'})
#     # save_grid_image(grid, smoke_grid, drones, {'smoke','fire','drones'}, 0)
#     #####

#     time_smoke = 0
#     time_fire = 0
#     time_smoke2 = 0

#     ####### Simulation loop
#     start = time.time()

#     drone_signals = []
#     for t in range(1,horizon):
#         #print(f"Time Step {t}")
#         # simulate wildfire and smoke spread
#         start_fire_time = time.time()
#         grid, burn_counters = propagate_wildfire(grid, burn_time, burn_counters, p, wind_speed, wind_direction)
#         time_fire += time.time() - start_fire_time
#         start_time_smoke = time.time()
#         smoke_grid = propagate_smoke(grid, smoke_grid, wind_speed, wind_direction)
#         time_smoke += time.time() - start_time_smoke

#     # save final result as a video
#     output_video = "wildfire_simulation.mp4"
#     # create_video_from_images(frames_per_image=3)
#     end = time.time()
#     #print(f"Simulation completed in {round(end-start)} seconds. Video '{output_video}' created.")

#     # Compute objective values
#     first_signal = 'UNDETECTED'
#     for i in range(len(drone_signals)):
#         if any(drone_signals[i]):
#             first_signal = i
#             break

#     # print(f"Fire detected after {first_signal} time steps")
#     print(f"{time_fire=}, {time_smoke=}, {time_smoke2=}")
#     return 1



def run_simulation(initial_wildfire_cells, horizon, N):
    """
    Run a wildfire simulation for a given horizon starting from initial cells.
    
    Args:
        initial_wildfire_cells (list): List of (x,y) tuples representing initial fire locations
        horizon (int): Number of time steps to simulate
    
    Returns:
        numpy.ndarray: horizonxNxN array representing the fire evolution
    """
    # Initialize grids
    wind_direction = random.choice(["right", "left", "up", "down", "up-left", "up-right", "down-left", "down-right"])
    grid = np.zeros((N, N), dtype=int)
    for x, y in initial_wildfire_cells:
        grid[x, y] = 1
    burn_counters = np.zeros((N, N), dtype=int)
    
    # Initialize the output array to store the simulation history
    simulation_history = np.zeros((horizon, N, N))
    simulation_history[0] = grid.copy()  # Store initial state
    
    # Simulation loop
    for t in range(1, horizon):
        # Update fire spread
        if not random.randint(0,9):
            wind_direction = random.choice(["right", "left", "up", "down", "up-left", "up-right", "down-left", "down-right"])
        grid, burn_counters = propagate_wildfire(grid, burn_time, burn_counters, p, wind_speed, wind_direction)
        
        # Store the current state
        simulation_history[t] = grid.copy()
    
    return simulation_history



#############

import numpy as np
def generate_ignition_map(N, correlation_level=0.7, defined_points=None):
    """
    Generate a normalized ignition probability map with spatial correlation and defined points.
    
    Args:
        N (int): Size of the square grid
        correlation_level (float): Between 0 and 1, how much each cell should be 
            influenced by its neighbors. Higher values mean more correlation.
        defined_points (list): List of ((x,y), value) tuples where value is the 
            raw value (before normalization) at position (x,y)
    """
    # Initialize the map with random values
    ignition_map = np.random.randint(0, 100, size=(N, N)).astype(float)
    
    # Create a mask for defined points and set their initial values
    defined_mask = np.zeros((N, N), dtype=bool)
    if defined_points is not None:
        for (x, y), value in defined_points:
            ignition_map[x, y] = value
            defined_mask[x, y] = True
    
    # Helper function to get average of neighbors
    def get_neighbor_average(arr, x, y):
        neighbors = []
        for dx in [_ for _ in range(-2,3)]:
            for dy in [_ for _ in range(-2,3)]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < N and 0 <= new_y < N:
                    neighbors.append(arr[new_x, new_y])
        return np.mean(neighbors)
    
    # Apply correlation multiple times to ensure good mixing
    num_iterations = 10
    for _ in range(num_iterations):
        new_map = ignition_map.copy()
        for x in range(N):
            for y in range(N):
                if not defined_mask[x, y]:
                    neighbor_avg = get_neighbor_average(ignition_map, x, y)
                    random_value = ignition_map[x, y]  # Use existing value as random component
                    new_map[x, y] = correlation_level * neighbor_avg + (1 - correlation_level) * random_value
        ignition_map = new_map
    
    # Ensure all values are non-negative
    ignition_map = np.maximum(ignition_map, 0)

    # replace the original fixed points
    
    # Normalize by dividing by the sum
    total = ignition_map.sum()
    ignition_map = ignition_map / total
    
    return ignition_map

def plot_ignition_map(ignition_map):
    """
    Plot the ignition map with a white-yellow-red colormap.
    
    Args:
        ignition_map (numpy.ndarray): NxN array of probabilities
    """
    import matplotlib.pyplot as plt
    
    N = ignition_map.shape[0]
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap from white to yellow to red
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]  # White to yellow to red
    n_bins = 100  # Number of color gradients
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Plot the heatmap
    im = plt.imshow(ignition_map, cmap=cmap, vmin=0, vmax= np.max(ignition_map))
    
    # Add colorbar
    plt.colorbar(im, label='Ignition Probability')
    
    # Add title and labels
    plt.title('Ignition Probability Map')
    plt.xlabel('Y coordinate')
    plt.ylabel('X coordinate')
    
    plt.show()



def sample_ignition(ignition_map):
    """
    Sample a point from the ignition map according to its probability distribution.
    
    Args:
        ignition_map (numpy.ndarray): NxN array of probabilities that sum to 1
    
    Returns:
        tuple: (x, y) coordinates of the sampled point
    """
    # Flatten the 2D array to 1D for sampling
    flat_probs = ignition_map.flatten()
    
    # Sample an index based on the probabilities
    flat_index = np.random.choice(len(flat_probs), p=flat_probs)
    
    # Convert flat index back to 2D coordinates
    N = ignition_map.shape[0]
    x = flat_index // N
    y = flat_index % N
    
    return (x, y)

###
from multiprocessing import shared_memory

def generate_scenario(shared_mem_name,shape,dtype, T, filename):
    """
    Generate a wildfire scenario by sampling a starting time and point, then running the simulation.
    
    Args:
        ignition_map (numpy.ndarray): NxN array of ignition probabilities
        T (int): Total time horizon
    
    Returns:
        tuple: (scenario, starting_time) where scenario is a TxNxN array and starting_time is an integer
    """
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    ignition_map = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    N = ignition_map.shape[0]
    starting_time = np.random.randint(0, T)
    empty_period = np.zeros((starting_time, N, N))
    starting_point = sample_ignition(ignition_map)
    simulation = run_simulation(
        initial_wildfire_cells=[starting_point],
        horizon=T-starting_time,
        N=N
    )
    scenario = np.concatenate([empty_period, simulation], axis=0)

    save_scenario(scenario, starting_time, filename)

    return scenario, starting_time
