from Drone import Drone
import time
import numpy as np
import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor

from dataset import load_scenario_npy
from Strategy import SensorPlacementStrategy, DroneRoutingStrategy

def get_automatic_layout_parameters(scenario:np.ndarray):
    return {
        "N": scenario.shape[1],
        "M": scenario.shape[2],
        "max_battery_distance": 100,
        "max_battery_time": 100,
        "n_drones": 1,
        "n_ground_stations": 1,
        "n_charging_stations": 1,
    }

def get_burnmap_parameters(input_dir:str):
    return {
        "burnmap_filename": f"{"/".join(input_dir.strip("/").split('/')[:-1])}/burn_map.npy"
    }


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
    if automatic_initialization_parameters_function  is None:
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

    # print(f"ground_sensor_locations: {ground_sensor_locations}")
    # print(f"charging_stations_locations: {charging_stations_locations}")

    # add computed positions to initialization parameters
    automatic_initialization_parameters["ground_sensor_locations"] = ground_sensor_locations
    automatic_initialization_parameters["charging_stations_locations"] = charging_stations_locations

    # 3. Initialize drones

    Routing_Strat = drone_routing_strategy(automatic_initialization_parameters, custom_initialization_parameters)
    drones = [Drone(x,y,charging_stations_locations,automatic_initialization_parameters["N"],automatic_initialization_parameters["M"], automatic_initialization_parameters["max_battery_distance"], automatic_initialization_parameters["max_battery_time"]) for (x,y) in Routing_Strat.get_initial_drone_locations()]
    drone_locations = [drone.get_position() for drone in drones]
    drone_batteries = [drone.get_battery() for drone in drones]
    drone_states = [drone.get_state() for drone in drones]
    drone_locations_history = None
    if return_history:
        drone_locations_history = [list(drone_locations)]

    # print(f"drone_locations: {drone_locations}")
    # print(f"drone_batteries: {drone_batteries}")

    t_found = 0
    device = 'undetected'
    grid = None

    for time_step in range(-starting_time,len(scenario)):
        if time_step >= 0: # The fire has started.
            # 1. Check if a fire is detected
            grid = scenario[time_step]
            
            if ground_sensor_locations:
                if (grid[rows_ground,cols_ground]==1).any():
                    device = 'ground sensor'
                    break

            if charging_stations_locations:
                if (grid[rows_charging,cols_charging]==1).any():
                    device = 'charging station'
                    break

            if drone_locations:
                rows, cols = zip(*drone_locations)
                if (grid[rows,cols]==1).any():
                    device = 'drone'
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
        actions = Routing_Strat.next_actions(automatic_step_parameters, custom_step_parameters)

        # 3. Move the drones
        for drone_index, (drone,action) in enumerate(zip(drones,actions)):
            new_x, new_y, new_distance_battery, new_time_battery, new_state = drone.route(action)
            drone_locations[drone_index] = (new_x,new_y)
            drone_batteries[drone_index] = (new_distance_battery,new_time_battery)
            drone_states[drone_index] = new_state
        if return_history:
            drone_locations_history.append(tuple(drone_locations))
        # print(f"drone_locations: {drone_locations}")

    delta_t = t_found-starting_time if t_found != len(scenario) else -1
    return delta_t, device, (drone_locations_history, ground_sensor_locations, charging_stations_locations) if return_history else ()


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
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.npy'):
                yield input_dir + entry.name
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

def run_benchmark_scenarii_sequential(input_dir, sensor_placement_strategy:SensorPlacementStrategy, drone_routing_strategy:DroneRoutingStrategy, custom_initialization_parameters_function:callable, custom_step_parameters_function:callable, starting_time:int=0, max_n_scenarii:int=None):
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
    Prints:
        Average time steps to fire detection and detection statistics by device type.
    """
    if not input_dir.endswith('/'):
        input_dir += '/'

    iterable = listdir_npy_limited(input_dir, max_n_scenarii)
    N_SCENARII = max_n_scenarii if max_n_scenarii else len(os.listdir(input_dir))
    automatic_initialization_parameters = None

    # Initialize counters
    delta_ts = 0
    fails = 0
    devices = {'ground sensor': 0, "charging station": 0, "drone": 0, 'undetected': 0}

    custom_initialization_parameters = custom_initialization_parameters_function(input_dir)
    
    for file in tqdm.tqdm(iterable, total = N_SCENARII):
        scenario = load_scenario_npy(file)
        if automatic_initialization_parameters is None:
            # Compute initialization parameters
            automatic_initialization_parameters = get_automatic_layout_parameters(scenario) #TODO compute them once only per layout rather than per scenario..
        delta_t, device, _ = run_benchmark_scenario(scenario, 
                                                    sensor_placement_strategy, 
                                                    drone_routing_strategy, 
                                                    custom_initialization_parameters,
                                                    custom_step_parameters_function, starting_time=starting_time)
  
        if delta_t == -1:
            fails += 1
            delta_t = 0
        delta_ts += delta_t
        devices[device] += 1
    
    print(f"This strategy took on average {delta_ts/max(1,(N_SCENARII-fails))} time steps to find the fire.")
    for device in devices.keys():
        print(f"Fire found {round(devices[device]/N_SCENARII*100,2)}% of the time by {device}")

def run_ground_log(input_dir, output_file, ground_placement_strategy, ground_parameters, max_n_scenarii=None):
    """
    Log ground sensor and charging station placements for multiple scenarios to a JSON file.

    Args:
        input_dir (str): Directory containing scenario files.
        output_file (str): Path to output JSON file.
        ground_placement_strategy (function): Strategy for placing ground sensors and charging stations.
        ground_parameters (tuple): Parameters for ground placement strategy.
        max_n_scenarii (int, optional): Maximum number of scenarios to process. If None, processes all scenarios.
    """
    if not input_dir.endswith('/'):
        input_dir += '/'

    iterable = listdir_txt_limited(input_dir, max_n_scenarii)
    all_placements = []
    for file in tqdm.tqdm(iterable, total = max_n_scenarii):
        ground_sensor_locations, charging_stations_locations =  ground_placement_strategy(*ground_parameters).get_locations()
        all_placements.append((ground_sensor_locations,charging_stations_locations))
    with open(output_file, "w") as outfile:
        json.dump(all_placements, outfile)

def benchmark_on_sim2real_dataset(dataset_folder_name, ground_placement_strategy, drone_routing_strategy, custom_initialization_parameters_function, custom_step_parameters_function, max_n_scenarii=None, starting_time=0):
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
    """
    if not dataset_folder_name.endswith('/'):
        dataset_folder_name += '/'
    
    for layout_folder in os.listdir(dataset_folder_name):
        if not os.path.exists(dataset_folder_name + layout_folder + "/scenarii/"):continue
        run_benchmark_scenarii_sequential(dataset_folder_name + layout_folder + "/scenarii/",
                                          ground_placement_strategy, 
                                          drone_routing_strategy, 
                                          custom_initialization_parameters_function, 
                                          custom_step_parameters_function, 
                                          starting_time=starting_time, 
                                          max_n_scenarii=max_n_scenarii)