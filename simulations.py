# Romain Puech, 2024
# Simulations

import numpy as np

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
                        if 0 <= nx < N and 0 <= ny < N and grid[nx, ny] == 0:  # Valid neighbor
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

def propagate_smoke(fire_grid,smoke_grid, wind_speed, wind_direction, diffusion_percentage=0.4):
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
