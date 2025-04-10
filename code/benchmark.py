from Drone import Drone
import time
import numpy as np
import tqdm
import os
import json
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import entropy as scipy_entropy

from dataset import load_scenario_npy, load_scenario_jpg, listdir_limited
from Strategy import SensorPlacementStrategy, DroneRoutingStrategy

def load_strategy(strategy_folder: str, strategy_file: str, class_name: str):
    """
    Dynamically loads a strategy class from a file.
    """
    strategy_path = os.path.join(strategy_folder, strategy_file)
    print(f"Looking for strategy file at: {strategy_path}")  # Add this line to debug
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"Strategy file {strategy_path} not found!")

    module_name = strategy_file.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise ImportError(f"Class {class_name} not found in {strategy_file}!")

    return getattr(module, class_name)


def get_automatic_layout_parameters(scenario:np.ndarray, input_dir:str=''):
    return {
        "N": scenario.shape[1],
        "M": scenario.shape[2],
        "max_battery_distance": -10,
        "max_battery_time": 20,
        "n_drones": 5,
        "n_ground_stations": 10,
        "n_charging_stations": 5,
        "input_dir": input_dir
    }

def return_no_custom_parameters():
    return {}

def build_custom_init_params(input_dir, layout_name):
    print(f"Building custom init params for {layout_name} in {input_dir}")
    base_path = '/'.join(input_dir.strip('/').split('/')[:-1])

    return {
        "burnmap_filename": f"{base_path}/burn_map.npy",
        "log_file": f"{base_path}/custom_params/{layout_name}.json",
        "call_every_n_steps": 5,               
        "optimization_horizon": 20      
    }

def get_burnmap_parameters(input_dir: str):
    return {
        "burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy"
    }

def compute_entropy(locations, grid_size):
    N, M = grid_size
    count_grid = np.zeros((N, M))
    for x, y in locations:
        count_grid[x, y] += 1

    prob_grid = count_grid.flatten()
    prob_grid = prob_grid / np.sum(prob_grid)
    prob_grid = prob_grid[prob_grid > 0]

    if len(prob_grid) == 0:
        return 0.0

    return scipy_entropy(prob_grid)

def run_drone_routing_strategy(drone_routing_strategy:DroneRoutingStrategy, sensor_placement_strategy:SensorPlacementStrategy, T:int, canonical_scenario:np.ndarray, automatic_initialization_parameters_function:callable, custom_initialization_parameters_function:callable, custom_step_parameters_function:callable, input_dir:str=''):
    """
    Run a drone routing strategy to create a logfile.
    """
    # 1. Get layout parameters
    if automatic_initialization_parameters_function is None:
        automatic_initialization_parameters = get_automatic_layout_parameters(canonical_scenario, input_dir)
    else:
        automatic_initialization_parameters = automatic_initialization_parameters_function(canonical_scenario, input_dir)
    
    if custom_initialization_parameters_function is not None:
        custom_initialization_parameters = custom_initialization_parameters_function(automatic_initialization_parameters['input_dir'])
    
    # 2. Get ground sensor locations
    ground_sensor_locations, charging_stations_locations =  sensor_placement_strategy(automatic_initialization_parameters, custom_initialization_parameters).get_locations()
    rows_ground, cols_ground = zip(*ground_sensor_locations)
    rows_charging, cols_charging = zip(*charging_stations_locations)

    # add computed positions to initialization parameters
    automatic_initialization_parameters["ground_sensor_locations"] = ground_sensor_locations
    automatic_initialization_parameters["charging_stations_locations"] = charging_stations_locations

    # 3. Initialize drones
    Routing_Strat = drone_routing_strategy(automatic_initialization_parameters, custom_initialization_parameters)
    drones = [Drone(x,y,state,charging_stations_locations,automatic_initialization_parameters["N"],automatic_initialization_parameters["M"], automatic_initialization_parameters["max_battery_distance"], automatic_initialization_parameters["max_battery_time"],automatic_initialization_parameters["max_battery_distance"]-1*(state=='fly'), automatic_initialization_parameters["max_battery_time"]-1*(state=='fly')) for (state,(x,y)) in Routing_Strat.get_initial_drone_locations()]
    drone_locations = [drone.get_position() for drone in drones]
    drone_batteries = [drone.get_battery() for drone in drones]
    drone_states = [drone.get_state() for drone in drones]

    # 4. Run the strategy
    for t in range(T):
        ### Move the drones

        # 1. Get the parameters
        custom_step_parameters = custom_step_parameters_function() # TODO: add weather parameters
        automatic_step_parameters = {
            "drone_locations": drone_locations,
            "drone_batteries": drone_batteries,
            "drone_states": drone_states,
            "t": t }
        
        # 2. Get the actions
        start_time = time.time()
        actions = Routing_Strat.next_actions(automatic_step_parameters, custom_step_parameters)

        # 3. Move the drones
        for drone_index, (drone, action) in enumerate(zip(drones, actions)):
            old_x, old_y = drone_locations[drone_index]
            new_x, new_y, new_distance_battery, new_time_battery, new_state = drone.route(action)

            drone_locations[drone_index] = (new_x, new_y)
            drone_batteries[drone_index] = (new_distance_battery, new_time_battery)
            drone_states[drone_index] = new_state
    print("Drone routing strategy finished")
    
    
    
    
def run_benchmark_scenario(scenario: np.ndarray, sensor_placement_strategy:SensorPlacementStrategy, drone_routing_strategy:DroneRoutingStrategy, custom_initialization_parameters:dict, custom_step_parameters_function:callable, starting_time:int=0, return_history:bool=False, custom_initialization_parameters_function:callable=None, automatic_initialization_parameters_function:callable=None):
    """
    Benchmark a routing and placement strategy on a single fire detection scenario.

    Args:
        scenario np.ndarray: grid states representing the fire progression over time.
        sensor_placement_strategy (function): Strategy function for placing ground sensors and charging stations.
        drone_routing_strategy (function): Strategy function for routing drones.
        layout_parameters (dict): Custom parameters given to the strategy at initialization.
        time_step_parameters_function (function): Function called at each time step. Returns a dict of custom_parameters given to the strategy.
        starting_time (int, optional): Time steps before the wildfire starts. Defaults to 0.
        return_history (bool, optional): If True, returns the history of drone positions. Defaults to False.

    Returns:
        tuple: Contains:
            - delta_t (int): Time steps taken to detect fire, or -1 if undetected
            - device (str): Which device detected the fire ('ground sensor', 'charging station', 'drone', or 'undetected')
            - history (tuple): If return_history=True, returns (drone_locations_history, ground_sensor_locations, charging_stations_locations)
    """

    # 1. Get layout parameters
    if automatic_initialization_parameters_function is None:
        automatic_initialization_parameters = get_automatic_layout_parameters(scenario)
    else:
        automatic_initialization_parameters = automatic_initialization_parameters_function(scenario)
    
    if custom_initialization_parameters_function is not None:
        custom_initialization_parameters = custom_initialization_parameters_function(automatic_initialization_parameters)

    # print(f"custom_initialization_parameters: {custom_initialization_parameters}")
    # print(f"automatic_initialization_parameters: {automatic_initialization_parameters}")

    # 2. Get ground sensor locations
    ground_sensor_locations, charging_stations_locations =  sensor_placement_strategy(automatic_initialization_parameters, custom_initialization_parameters).get_locations()
    rows_ground, cols_ground = zip(*ground_sensor_locations)
    rows_charging, cols_charging = zip(*charging_stations_locations)

    charging_stations_locations = {tuple(station) for station in charging_stations_locations}  # Convert to set of tuples

  
    # print(f"ground_sensor_locations: {ground_sensor_locations}")
    # print(f"charging_stations_locations: {charging_stations_locations}")

    # add computed positions to initialization parameters
    automatic_initialization_parameters["ground_sensor_locations"] = ground_sensor_locations
    automatic_initialization_parameters["charging_stations_locations"] = charging_stations_locations
    
    # 3. Initialize drones

    Routing_Strat = drone_routing_strategy(automatic_initialization_parameters, custom_initialization_parameters)
    # Print initial drone locations
    initial_drone_locations = Routing_Strat.get_initial_drone_locations()
    print(f"\nDEBUG: Initial Drone Locations: {initial_drone_locations}")

    drones = [Drone(x,y,state,charging_stations_locations,automatic_initialization_parameters["N"],automatic_initialization_parameters["M"], automatic_initialization_parameters["max_battery_distance"], automatic_initialization_parameters["max_battery_time"],automatic_initialization_parameters["max_battery_distance"]-1*(state=='fly'), automatic_initialization_parameters["max_battery_time"]-1*(state=='fly')) for (state,(x,y)) in Routing_Strat.get_initial_drone_locations()]
    drone_locations = [drone.get_position() for drone in drones]
    drone_batteries = [drone.get_battery() for drone in drones]
    drone_states = [drone.get_state() for drone in drones]
    drone_locations_history = None
    if return_history:
        drone_locations_history = [list(drone_locations)]

    # print(f"drone_locations: {drone_locations}")
    # print(f"drone_batteries: {drone_batteries}")

    # ========================

    # Initialize metrics
    execution_times = []
    drone_visited_cells = set(drone_locations)
    total_distance_traveled = 0
    drone_entropy_per_timestep = []
    sensor_entropy = compute_entropy(ground_sensor_locations, (automatic_initialization_parameters["N"], automatic_initialization_parameters["M"]))

    fire_size_cells = 0
    fire_size_percentage = 0
    # ========================

    t_found = 0
    device = 'undetected'
    

    for time_step in range(-starting_time,len(scenario)):
        if time_step >= 0: # The fire has started.
            # 1. Check if a fire is detected
            grid = scenario[time_step]
            
            if ground_sensor_locations:
                if (grid[rows_ground,cols_ground]==1).any():
                    device = 'ground sensor'
                    fire_size_cells = np.sum(grid == 1)
                    fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                    break

            if charging_stations_locations:
                if (grid[rows_charging,cols_charging]==1).any():
                    device = 'charging station'
                    fire_size_cells = np.sum(grid == 1)
                    fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                    break

            if drone_locations:
                rows, cols = zip(*drone_locations)
                if (grid[rows,cols]==1).any():
                    device = 'drone'
                    fire_size_cells = np.sum(grid == 1)
                    fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                    break

        # no fire detected, onto next time step
        t_found +=1

        ### Move the drones

        # 1. Get the parameters
        custom_step_parameters = custom_step_parameters_function() # TODO: add weather parameters
        automatic_step_parameters = {
            "drone_locations": drone_locations,
            "drone_batteries": drone_batteries,
            "drone_states": drone_states,
            "t": t_found }
        
        # 2. Get the actions
        start_time = time.time()
        actions = Routing_Strat.next_actions(automatic_step_parameters, custom_step_parameters)
        execution_times.append(time.time() - start_time)

        # 3. Move the drones
        for drone_index, (drone, action) in enumerate(zip(drones, actions)):
            old_x, old_y = drone_locations[drone_index]
            new_x, new_y, new_distance_battery, new_time_battery, new_state = drone.route(action)

            drone_locations[drone_index] = (new_x, new_y)
            drone_batteries[drone_index] = (new_distance_battery, new_time_battery)
            drone_states[drone_index] = new_state

            total_distance_traveled += abs(new_x - old_x) + abs(new_y - old_y)
            drone_visited_cells.add((new_x, new_y))
        # print("Drone actions: ", actions)
        # print("Drone batteries: ", drone_batteries)
        

        drone_entropy = compute_entropy(drone_locations, (automatic_initialization_parameters["N"], automatic_initialization_parameters["M"]))
        drone_entropy_per_timestep.append(drone_entropy)

        if return_history:
            drone_locations_history.append(tuple(drone_locations))
        # print(f"drone_locations: {drone_locations}")

    delta_t = t_found - starting_time if t_found != len(scenario) else -1
    avg_execution_time = np.mean(execution_times)
    avg_drone_entropy = np.mean(drone_entropy_per_timestep)
    percentage_map_explored = len(drone_visited_cells) / (automatic_initialization_parameters["N"] * automatic_initialization_parameters["M"]) * 100

    if device == 'undetected':
        final_grid = scenario[-1]
        fire_size_cells = np.sum(final_grid == 1)
        fire_size_percentage = fire_size_cells / (final_grid.shape[0] * final_grid.shape[1]) * 100

    results = {
        "delta_t": delta_t,
        "device": device,
        "avg_execution_time": avg_execution_time,
        "fire_size_cells": fire_size_cells,
        "fire_size_percentage": fire_size_percentage,
        "percentage_map_explored": percentage_map_explored,
        "total_distance_traveled": total_distance_traveled,
        "avg_drone_entropy": avg_drone_entropy,
        "sensor_entropy": sensor_entropy
    }

    return results, (drone_locations_history, ground_sensor_locations, charging_stations_locations) if return_history else ()


def listdir_txt_limited(input_dir, max_n_scenarii=None):
    """
    Generate paths to .txt files in the input directory, with optional limit.

    Args:
        input_dir (str): Directory path to scan for .txt files.
        max_n_scenarii (int, optional): Maximum number of files to yield. If None, yields all files.

    Yields:
        str: Full path to each .txt file found.
    """
    count = 0
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.txt'):
                yield input_dir + entry.name
                count += 1
                if max_n_scenarii is not None and count >= max_n_scenarii:
                    break

def listdir_npy_limited(input_dir, max_n_scenarii=None):
    """
    Generate paths to .npy files in the input directory, with optional limit.

    Args:
        input_dir (str): Directory path to scan for .npy files.
        max_n_scenarii (int, optional): Maximum number of files to yield. If None, yields all files.

    Yields:
        str: Full path to each .npy file found.
    """
    count = 0
    if not input_dir.endswith('/'):
        input_dir += '/'

    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.npy'):
                yield os.path.join(input_dir, entry.name)
                count += 1
                if max_n_scenarii is not None and count >= max_n_scenarii:
                    break

def listdir_folder_limited(input_dir, max_n_scenarii=None):
    """
    Generate paths to folders in the input directory, with optional limit.

    Args:
        input_dir (str): Directory path to scan for folders.
        max_n_scenarii (int, optional): Maximum number of folders to yield. If None, yields all folders.

    Yields:
        str: Full path to each folder found.
    """
    count = 0
    if not input_dir.endswith('/'):
        input_dir += '/'

    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_dir():
                yield os.path.join(input_dir, entry.name)
                count += 1
                if max_n_scenarii is not None and count >= max_n_scenarii:
                    break


def run_benchmark_scenarii(input_dir, ground_placement_strategy, drone_routing_strategy, ground_parameters, routing_parameters, max_n_scenarii=None):
    """
    Run parallel benchmarks on multiple scenarios using thread pooling.

    Args:
        input_dir (str): Directory containing scenario files.
        ground_placement_strategy (function): Strategy for placing ground sensors and charging stations.
        drone_routing_strategy (function): Strategy for controlling drone movements.
        ground_parameters (tuple): Parameters for ground placement strategy.
        routing_parameters (tuple): Parameters for routing strategy.
        max_n_scenarii (int, optional): Maximum number of scenarios to process. If None, processes all scenarios.

    Prints:
        Average time steps to fire detection and detection statistics by device type.
    """
    # TODO: add starting time
    raise NotImplementedError("Starting time is not implemented yet")
    if not input_dir.endswith('/'):
        input_dir += '/'

    iterable = listdir_txt_limited(input_dir, max_n_scenarii)

    M = len(os.listdir(input_dir)) if max_n_scenarii is None else max_n_scenarii
    
    def process_scenario(infile):
        start = GroundPlacementOptimization(10,10,100,"burn_maps/burn_map_1")
        print(start.get_locations())
        return 0,'undetected'
        # scenario, start_time = load_scenario(infile)
        # delta_t, device, _ = run_benchmark_scenario(scenario, start_time, ground_placement_strategy, 
        #                                           drone_routing_strategy, ground_parameters, routing_parameters)
        # return delta_t, device

    # Initialize counters
    delta_ts = 0
    fails = 0
    devices = {'ground sensor': 0, "charging station": 0, "drone": 0, 'undetected': 0}
    
    # Use ThreadPoolExecutor to parallelize scenario processing
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Use tqdm to show progress bar for parallel execution
        results = list(tqdm.tqdm(executor.map(process_scenario, iterable), total=M))
        
    # Process results
    for delta_t, device in results:
        if delta_t == -1:
            fails += 1
            delta_t = 0
        delta_ts += delta_t
        devices[device] += 1
    
    print(f"This strategy took on average {delta_ts/max(1,(M-fails))} time steps to find the fire.")
    for device in devices.keys():
        print(f"Fire found {round(devices[device]/M*100,2)}% of the time by {device}")

def run_benchmark_scenarii_sequential(input_dir, sensor_placement_strategy:SensorPlacementStrategy, drone_routing_strategy:DroneRoutingStrategy, custom_initialization_parameters_function:callable, custom_step_parameters_function:callable, starting_time:int=0, max_n_scenarii:int=None, file_format="npy"):
    """
    Run sequential benchmarks on multiple scenarios.

    Args:
        input_dir (str): Directory containing scenario files.
        sensor_placement_strategy (function): Strategy for placing ground sensors and charging stations.
        drone_routing_strategy (function): Strategy for controlling drone movements.
        custom_step_parameters_function (function): Function for custom step parameters.
        starting_time (int, optional): Time step at which the wildfire starts.
        return_history (bool, optional): If True, return the history of drone positions.
        custom_initialization_parameters_function (function, optional): Function for custom initialization parameters.
        max_n_scenarii (int, optional): Maximum number of scenarios to process. If None, processes all scenarios.
    
    Returns:
        dict: Metrics dictionary containing benchmark results.
    """
    if file_format not in ["npy", "jpg"]:
        raise ValueError("file_format must be 'npy' or 'jpg'")

    if not input_dir.endswith('/'):
        input_dir += '/'

    # choose the iterator and loader
    if file_format == "npy":
        iterable = listdir_npy_limited(input_dir, max_n_scenarii)
        load_scenario_fn = load_scenario_npy
    else:
        iterable = listdir_folder_limited(input_dir, max_n_scenarii)
        load_scenario_fn = load_scenario_jpg

    N_SCENARII = max_n_scenarii if max_n_scenarii else len(os.listdir(input_dir))
    automatic_initialization_parameters = None

    # Initialize counters
    delta_ts = 0
    fails = 0
    devices = {'ground sensor': 0, "charging station": 0, "drone": 0, 'undetected': 0}

    total_execution_times = []
    total_fire_sizes = []
    total_fire_percentages = []
    map_explored = []
    total_distances = []
    drone_entropies = []
    sensor_entropies = []

    # Extract layout name from input directory path
    layout_name = os.path.basename(os.path.dirname(input_dir))
    
    # Check the number of parameters the function accepts
    import inspect
    sig = inspect.signature(custom_initialization_parameters_function)
    param_count = len(sig.parameters)
    
    # Call the function with the appropriate number of parameters
    if param_count >= 2:
        custom_initialization_parameters = custom_initialization_parameters_function(input_dir, layout_name)
    else:
        custom_initialization_parameters = custom_initialization_parameters_function(input_dir)
    
    for file in tqdm.tqdm(iterable, total = N_SCENARII):
        #print(f"Processing scenario {file}")
        scenario = load_scenario_fn(file)
        if automatic_initialization_parameters is None:
            # Compute initialization parameters
            automatic_initialization_parameters = get_automatic_layout_parameters(scenario) #TODO compute them once only per layout rather than per scenario..
        results, _ = run_benchmark_scenario(
            scenario,
            sensor_placement_strategy,
            drone_routing_strategy,
            custom_initialization_parameters,
            custom_step_parameters_function,
            starting_time=starting_time
        )

        delta_t = results["delta_t"]
        device = results["device"]

        if delta_t == -1:
            fails += 1
            delta_t = 0

        delta_ts += delta_t
        devices[device] += 1

        total_execution_times.append(results["avg_execution_time"])
        total_fire_sizes.append(results["fire_size_cells"])
        total_fire_percentages.append(results["fire_size_percentage"])
        map_explored.append(results["percentage_map_explored"])
        total_distances.append(results["total_distance_traveled"])
        drone_entropies.append(results["avg_drone_entropy"])
        sensor_entropies.append(results["sensor_entropy"])
    
    # Calculate metrics
    avg_time_to_detection = delta_ts / max(1, (N_SCENARII - fails))
    device_percentages = {device: round(count / N_SCENARII * 100, 2) for device, count in devices.items()}
    avg_execution_time = np.mean(total_execution_times)
    avg_fire_size = np.mean(total_fire_sizes)
    avg_fire_percentage = np.mean(total_fire_percentages)
    avg_map_explored = np.mean(map_explored)
    avg_distance = np.mean(total_distances)
    avg_drone_entropy = np.mean(drone_entropies)
    avg_sensor_entropy = np.mean(sensor_entropies)
    
    # Create metrics dictionary
    metrics = {
        "avg_time_to_detection": avg_time_to_detection,
        "device_percentages": device_percentages,
        "avg_execution_time": avg_execution_time,
        "avg_fire_size": avg_fire_size,
        "avg_fire_percentage": avg_fire_percentage,
        "avg_map_explored": avg_map_explored,
        "avg_distance": avg_distance,
        "avg_drone_entropy": avg_drone_entropy,
        "avg_sensor_entropy": avg_sensor_entropy,
        "raw_execution_times": total_execution_times,
        "raw_fire_sizes": total_fire_sizes,
        "raw_fire_percentages": total_fire_percentages,
        "raw_map_explored": map_explored,
        "raw_distances": total_distances,
        "raw_drone_entropies": drone_entropies,
        "raw_sensor_entropies": sensor_entropies
    }
    
    # Still print the results for console feedback
    print(f"Avg time steps to fire detection: {avg_time_to_detection}")
    for device, percentage in device_percentages.items():
        print(f"Fire found {percentage}% of the time by {device}")
    print(f"Avg routing execution time: {avg_execution_time:.4f} sec")
    print(f"Avg fire size at detection (cells): {avg_fire_size:.2f}")
    print(f"Avg fire size at detection (% of map): {avg_fire_percentage:.2f}%")
    print(f"Avg percentage of map explored by drones: {avg_map_explored:.2f}%")
    print(f"Avg total distance traveled by drones: {avg_distance:.2f} units")
    print(f"Avg drone entropy per timestep: {avg_drone_entropy:.4f}")
    print(f"Avg sensor placement entropy: {avg_sensor_entropy:.4f}")
    
    return metrics

def run_benchmark_scenarii_sequential_precompute(input_dir, sensor_placement_strategy:SensorPlacementStrategy, drone_routing_strategy:DroneRoutingStrategy, custom_initialization_parameters_function:callable, custom_step_parameters_function:callable, starting_time:int=0, max_n_scenarii:int=None, file_format="npy"):
    """
    Run benchmarks on multiple scenarios sequentially, precomputing the sensor placement and drone routing strategy.

    Args:
        input_dir (str): Directory containing scenario files.
        sensor_placement_strategy (function): Strategy for placing ground sensors and charging stations.
        drone_routing_strategy (function): Strategy for controlling drone movements.
        custom_initialization_parameters (dict): Custom initialization parameters.
        custom_step_parameters_function (function): Function for custom step parameters.
        starting_time (int, optional): Time step at which the wildfire starts.
        max_n_scenarii (int, optional): Maximum number of scenarios to process. If None, processes all scenarios.
        file_format (str, optional): Format of the scenario files.
        
    Returns:
        dict: Metrics dictionary containing benchmark results.
    """
    if file_format not in ["npy", "jpg"]:
        raise ValueError("file_format must be 'npy' or 'jpg'")

    if not input_dir.endswith('/'):
        input_dir += '/'

    # choose the iterator and loader
    if file_format == "npy":
        iterable = listdir_npy_limited(input_dir, max_n_scenarii)
        load_scenario_fn = load_scenario_npy
    else:
        iterable = listdir_folder_limited(input_dir, max_n_scenarii)
        load_scenario_fn = load_scenario_jpg

    N_SCENARII = max_n_scenarii if max_n_scenarii else len(os.listdir(input_dir))
    # find the longest scenario to be used as canonical scenario
    max_scenario_length = 0
    for file in iterable:
        scenario = load_scenario_fn(file)
        if scenario.shape[0] > max_scenario_length:
            max_scenario_length = scenario.shape[0]
            canonical_scenario = scenario
    
    # precompute the sensor placement and drone routing strategy on canonical scenario
    #print("Precomputing sensor placement and drone routing strategy on canonical scenario...")
    run_drone_routing_strategy(drone_routing_strategy, sensor_placement_strategy, max_scenario_length, canonical_scenario, get_automatic_layout_parameters, custom_initialization_parameters_function, custom_step_parameters_function, input_dir) 
    #print("running on all scenarios...")
    return run_benchmark_scenarii_sequential(input_dir, sensor_placement_strategy, drone_routing_strategy, custom_initialization_parameters_function, custom_step_parameters_function, starting_time, max_n_scenarii, file_format)

def benchmark_on_sim2real_dataset_precompute(dataset_folder_name, ground_placement_strategy, drone_routing_strategy, custom_initialization_parameters_function, custom_step_parameters_function, max_n_scenarii=None, starting_time=0, max_n_layouts=None):
    """
    Run benchmarks on a simulation-to-real-world dataset structure.

    Args:
        dataset_folder_name (str): Root folder containing layout folders with scenario data.
        ground_placement_strategy (function): Strategy for placing ground sensors and charging stations.
        drone_routing_strategy (function): Strategy for controlling drone movements.
        ground_parameters (tuple): Parameters for ground placement strategy.
        routing_parameters (tuple): Parameters for routing strategy.
        max_n_scenarii (int, optional): Maximum number of scenarios to process per layout. If None, processes all scenarios.
        starting_time (int, optional): Time step at which the wildfire starts.
        
    Returns:
        dict: Dictionary mapping layout names to their respective metric dictionaries.
    """
    if not dataset_folder_name.endswith('/'):
        dataset_folder_name += '/'
    
    all_metrics = {}
    
    for layout_folder in listdir_folder_limited(dataset_folder_name, max_n_layouts):
        print(f"\n --- \n Processing layout {layout_folder}")
        layout_name = os.path.basename(layout_folder)
        
        if not os.path.exists(layout_folder + "/scenarii/"):
            print(f"No scenarii folder found in {layout_folder}, skipping...")
            continue
            
        metrics = run_benchmark_scenarii_sequential_precompute(
            layout_folder + "/scenarii/",
            ground_placement_strategy, 
            drone_routing_strategy, 
            custom_initialization_parameters_function, 
            custom_step_parameters_function, 
            starting_time=starting_time, 
            max_n_scenarii=max_n_scenarii
        )
        
        all_metrics[layout_name] = metrics
    
    return all_metrics

def run_benchmark_for_strategy(input_dir: str,
                               strategy_folder: str,
                               sensor_strategy_file: str,
                               sensor_class_name: str,
                               drone_strategy_file: str,
                               drone_class_name: str,
                               max_n_scenarii: int = None,
                               starting_time: int = 0,
                               file_format: str = "npy",
                               custom_init_params_fn= build_custom_init_params,
                               custom_step_params_fn= return_no_custom_parameters):
    """
    Runs benchmarks for the given sensor and drone strategies on all scenarios in input_dir.
    """

    if file_format not in ["npy", "jpg"]:
        raise ValueError("file_format must be 'npy' or 'jpg'")

    if not input_dir.endswith('/'):
        input_dir += '/'

    # choose the iterator and loader
    if file_format == "npy":
        iterable = listdir_npy_limited(input_dir, max_n_scenarii)
        load_scenario_fn = load_scenario_npy
    else:
        iterable = listdir_folder_limited(input_dir, max_n_scenarii)
        load_scenario_fn = load_scenario_jpg

    # === Load user strategies ===
    SensorPlacementStrategyClass = load_strategy(strategy_folder, sensor_strategy_file, sensor_class_name)
    DroneRoutingStrategyClass = load_strategy(strategy_folder, drone_strategy_file, drone_class_name)

    # === Load the first scenario to get parameters ===
    first_file = next(iter(iterable), None)
    if first_file is None:
        print(f"No scenarios found in {input_dir}")
        return

    # load the first scenario to get automatic parameters
    scenario = load_scenario_fn(first_file)
    automatic_init_params = get_automatic_layout_parameters(scenario)

    # === Create sensor placement strategy ===
    print("[run_benchmark_for_strategy] Running sensor placement strategy...")

    custom_init_params = custom_init_params_fn(input_dir, layout_name=os.path.basename(input_dir))

    sensor_placement_strategy_instance = SensorPlacementStrategyClass(automatic_init_params, custom_init_params)
    drone_routing_strategy_instance = DroneRoutingStrategyClass(automatic_init_params, custom_init_params)
    # === update automatic_init_params with sensor placements ===
    ground_sensor_locations, charging_station_locations = sensor_placement_strategy_instance.get_locations()
    automatic_init_params["ground_sensor_locations"] = ground_sensor_locations
    automatic_init_params["charging_stations_locations"] = charging_station_locations

    # === call the existing function ===
    run_benchmark_scenarii_sequential(
        input_dir=input_dir,
        sensor_placement_strategy=lambda *_: sensor_placement_strategy_instance,
        drone_routing_strategy=lambda *_: drone_routing_strategy_instance,
        custom_initialization_parameters_function=custom_init_params_fn,
        custom_step_parameters_function=custom_step_params_fn,
        starting_time=starting_time,
        max_n_scenarii=max_n_scenarii,
        file_format=file_format
    )

