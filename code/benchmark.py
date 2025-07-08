from Drone import Drone
import time
import numpy as np
import tqdm
import os
import json
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import entropy as scipy_entropy
from dataset import load_scenario_npy, load_scenario_jpg, listdir_limited, load_burn_map
from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
from new_clustering import get_wrapped_clustering_strategy
from Strategy import SensorPlacementStrategy, DroneRoutingStrategy
from displays import create_scenario_video
import tqdm
import math
import pandas as pd

def generate_coverage_area(coverage_radius_m, cell_size_m, x_center, y_center, N, M):
    coverage_width_cells = round(coverage_radius_m*2 / cell_size_m)//2
    covered_cells = set()
    for x_cov in range(x_center-coverage_width_cells//2, x_center+coverage_width_cells//2+1):
        for y_cov in range(y_center-coverage_width_cells//2, y_center+coverage_width_cells//2+1):
            if x_cov >= 0 and x_cov < N and y_cov >= 0 and y_cov < M:
                covered_cells.add((x_cov, y_cov))
    return covered_cells

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




def get_automatic_layout_parameters(scenario: np.ndarray, input_dir: str, simulation_parameters: dict, scenario_name: str = ""):
    #print("simulation_parameters", simulation_parameters)
    return {
        "N": scenario.shape[1],
        "M": scenario.shape[2],
        "max_battery_distance": simulation_parameters.get("max_battery_distance", DEFAULT_SIMULATION_PARAMETERS["max_battery_distance"]),
        "max_battery_time": simulation_parameters.get("max_battery_time", DEFAULT_SIMULATION_PARAMETERS["max_battery_time"]),
        "n_drones": simulation_parameters.get("n_drones", DEFAULT_SIMULATION_PARAMETERS["n_drones"]),
        "n_ground_stations": simulation_parameters.get("n_ground_stations", DEFAULT_SIMULATION_PARAMETERS["n_ground_stations"]),
        "n_charging_stations": simulation_parameters.get("n_charging_stations", DEFAULT_SIMULATION_PARAMETERS["n_charging_stations"]),
        "speed_m_per_min": simulation_parameters.get("drone_speed_m_per_min", DEFAULT_SIMULATION_PARAMETERS["drone_speed_m_per_min"]),
        "coverage_radius_m": simulation_parameters.get("coverage_radius_m", DEFAULT_SIMULATION_PARAMETERS["coverage_radius_m"]),
        "cell_size_m": simulation_parameters.get("cell_size_m", DEFAULT_SIMULATION_PARAMETERS["cell_size_m"]),
        "input_dir": input_dir,
        "transmission_range": simulation_parameters.get("transmission_range", DEFAULT_SIMULATION_PARAMETERS["transmission_range"]),
        "scenario_name": scenario_name or simulation_parameters.get("scenario_name", DEFAULT_SIMULATION_PARAMETERS["scenario_name"]),
    }


def return_no_custom_parameters():
    return {}

# update if you want default parameters for all the layouts
DEFAULT_SIMULATION_PARAMETERS = {
    "call_every_n_steps": -1,
    "optimization_horizon": -1,
    "reevaluation_step": -1,
    "max_battery_distance": -1,
    "max_battery_time": -1,
    "n_drones": -1,
    "n_ground_stations": -1,
    "n_charging_stations": -1,
    "drone_speed_m_per_min": -1,
    "coverage_radius_m": -1,
    "cell_size_m": -1,
    "transmission_range": -1,
    "scenario_name": "",
}

def build_custom_init_params(input_dir, layout_name):
    base_path = os.path.abspath(os.path.join(input_dir, ".."))
    params = {
        "burnmap_filename": os.path.join(base_path, "burn_map.npy"),
        "log_file": os.path.join(base_path, f"{layout_name}.json"),
    }

    # Inject all simulation parameters
    params.update(DEFAULT_SIMULATION_PARAMETERS)

    return params

def print_simulation_parameters(simulation_params: dict, title: str = "Simulation Parameters"):
    """
    Pretty print the simulation parameters at the beginning of the run.
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    for key, value in simulation_params.items():
        print(f"  {key:<25}: {value}")
    print("=" * 60 + "\n")

def get_burnmap_parameters(input_dir: str):
    return {
        "burnmap_filename": f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy"
    }
 

def compute_operational_substeps(data_cell_size_m, drone_speed_m_per_min, coverage_radius_m):
    """
    Estimate how many drone actions (substeps) should occur per one data timestep (60 minutes),
    based on the drone speed and its effective coverage range.

    Args:
        data_cell_size_m (float): Size of each data cell in meters.
        drone_speed_m_per_min (float): Speed of the drone in meters per minute.
        coverage_radius_m (float): Effective coverage radius of the drone in meters.
        
    Assumes:
        - Data timestep = 60 minutes

    Returns:
        int: Number of drone movement substeps per data timestep
    """

    coverage_width_m = 2 * coverage_radius_m
    coverage_width_cells = coverage_width_m / data_cell_size_m
    coverage_width_cells = max(1, round(coverage_width_cells))  # Ensure it's at least 1 and rounded

    if coverage_width_cells % 2 == 0:  # If even, make it odd
        coverage_width_cells -= 1
    
    drone_distance_m = 60 * drone_speed_m_per_min
    drone_distance_operational_cells_per_timestep = drone_distance_m // (coverage_width_cells*data_cell_size_m)
    
    return max(1, round(drone_distance_operational_cells_per_timestep))


def detect_fire_within_coverage(fire_grid, drone_pos, coverage_width_cells):
    """
    Returns True if any cell within the drone's square coverage area is on fire.
    """
    coverage_radius_cells = coverage_width_cells // 2
    x, y = drone_pos
    N, M = fire_grid.shape
    for dx in range(-coverage_radius_cells, coverage_radius_cells + 1):
        for dy in range(-coverage_radius_cells, coverage_radius_cells + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < M:
                if fire_grid[nx, ny] == 1:
                    return True
    return False

def operational_space_to_dataspace_coordinates(coordinate, coverage, datacell_size_m):
    """
    Convert operational space coordinates to data space coordinates.

    Args:
        coordinate (tuple): Coordinates in operational space (x, y).
        operational_space_m (float): Size of the operational area in meters.
        cell_size_m (float): Size of each data cell in meters.
        grid_size_m (tuple): Size of the grid in meters (width, height).

    Returns:
        tuple: Converted coordinates into middle of operational_cell in data space.
    """
    n_data_cells_in_coverage_area = 2 * coverage // datacell_size_m
    n_data_cells_in_coverage_area = max(1, round(n_data_cells_in_coverage_area))  # Ensure it's at least 1 and rounded
    
    if n_data_cells_in_coverage_area % 2 == 0:  # If even, make it odd
        n_data_cells_in_coverage_area -= 1

    x, y = coordinate
    sign = lambda x: 1 if x >= 0 else -1
    half_coverage = n_data_cells_in_coverage_area // 2
    new_x , newy = x * n_data_cells_in_coverage_area + half_coverage * sign(x), y * n_data_cells_in_coverage_area + half_coverage * sign(y)
    return (new_x, newy)

def run_drone_routing_strategy(drone_routing_strategy:DroneRoutingStrategy, sensor_placement_strategy:SensorPlacementStrategy, T:int, canonical_scenario:np.ndarray, automatic_initialization_parameters_function:callable, custom_initialization_parameters_function:callable, custom_step_parameters_function:callable, input_dir:str='', simulation_parameters:dict={}, file_format:str="npy", starting_time:int=1, scenario_name:str=""):
    """
    Runs a drone routing strategy on a wildfire scenario and collects performance metrics.
    
    This function:
    1. Initializes the scenario and strategies
    2. Places sensors and charging stations
    3. Routes drones according to the strategy
    4. Collects metrics about fire detection and drone performance
    
    Args:
        drone_routing_strategy (DroneRoutingStrategy): Strategy for drone movement
        sensor_placement_strategy (SensorPlacementStrategy): Strategy for sensor placement
        T (int): Number of time steps to simulate
        canonical_scenario (np.ndarray): The wildfire scenario to simulate
        automatic_initialization_parameters_function (callable): Function to get system parameters
        custom_initialization_parameters_function (callable): Function to get strategy parameters
        custom_step_parameters_function (callable): Function to get step-specific parameters
        input_dir (str, optional): Directory containing scenario data
        simulation_parameters (dict, optional): Additional simulation parameters
        file_format (str, optional): Format of scenario files ('npy' or 'jpg')
        starting_time (int, optional): Time step to start simulation
        
    Returns:
        float: Total execution time in seconds
        
    Notes:
        The function handles:
        - Rescaling of coordinates and battery levels
        - Drone movement and charging
        - Fire detection by drones and sensors
        - Collection of performance metrics
    """
    
    time_start = time.time()
    # 0. get layout parameters
    if automatic_initialization_parameters_function is None:
        automatic_initialization_parameters = get_automatic_layout_parameters(canonical_scenario, input_dir, simulation_parameters, scenario_name)
    else:
        automatic_initialization_parameters = automatic_initialization_parameters_function(canonical_scenario, input_dir, simulation_parameters, scenario_name)
    
    if custom_initialization_parameters_function is not None:
        custom_initialization_parameters = custom_initialization_parameters_function(input_dir)


    # 1. compute the rescaling factors
    cell_size_m = automatic_initialization_parameters.get("cell_size_m", 30)
    speed_m_per_min = automatic_initialization_parameters.get("speed_m_per_min", 9)
    coverage_radius_m = automatic_initialization_parameters.get("coverage_radius_m", 45)

    operational_substeps = compute_operational_substeps(cell_size_m, speed_m_per_min, coverage_radius_m)
    coverage_width_cells = round(coverage_radius_m*2 / cell_size_m)

    rescaled_N = automatic_initialization_parameters["N"] // coverage_width_cells
    rescaled_M = automatic_initialization_parameters["M"] // coverage_width_cells
    rescaled_max_battery_time = automatic_initialization_parameters["max_battery_time"] * operational_substeps
    # print("loading and preparing burnmap")
    rescaled_burnmap = load_burn_map(custom_initialization_parameters["burnmap_filename"])
    rescaled_burnmap = pool_burnmap_proba_at_least_one(rescaled_burnmap, coverage_width_cells)
    # duplicate the burn map for the operationnal time scale: each grid is duplicated operational_substeps times
    rescaled_burnmap = np.repeat(rescaled_burnmap, operational_substeps, axis=0)/operational_substeps # we also rescale the probabilities to time scale
    # print("saving the pooled burnmap")
    #save the pooled burnmap
    rescaled_burnmap_filename = custom_initialization_parameters["burnmap_filename"].replace(".npy", f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy")
    np.save(rescaled_burnmap_filename, rescaled_burnmap)

    rescaled_automatic_initialization_parameters = automatic_initialization_parameters.copy()
    rescaled_automatic_initialization_parameters["N"] = rescaled_N
    rescaled_automatic_initialization_parameters["M"] = rescaled_M
    rescaled_automatic_initialization_parameters["max_battery_time"] = rescaled_max_battery_time
    rescaled_automatic_initialization_parameters["burnmap_filename"] = rescaled_burnmap_filename

    rescaled_custom_initialization_parameters = custom_initialization_parameters.copy()
    rescaled_custom_initialization_parameters["burnmap_filename"] = rescaled_burnmap_filename

    # 2. Get ground sensor locations and convert them back to the original size
    # print("getting sensor locations")
    ground_sensor_locations_opt_scale, charging_stations_locations_opt_scale =  sensor_placement_strategy(rescaled_automatic_initialization_parameters, rescaled_custom_initialization_parameters).get_locations()

    ground_sensor_locations_data_scale = [(x*coverage_width_cells+coverage_width_cells//2, y*coverage_width_cells+coverage_width_cells//2) for x,y in ground_sensor_locations_opt_scale]
    charging_stations_locations_data_scale = [(x*coverage_width_cells+coverage_width_cells//2, y*coverage_width_cells+coverage_width_cells//2) for x,y in charging_stations_locations_opt_scale]
    rows_ground_data_scale, cols_ground_data_scale = zip(*ground_sensor_locations_data_scale) if ground_sensor_locations_data_scale else ((),())
    rows_charging_data_scale, cols_charging_data_scale = zip(*charging_stations_locations_data_scale) if charging_stations_locations_data_scale else ((),())

    # add computed positions to initialization parameters
    automatic_initialization_parameters["ground_sensor_locations"] = ground_sensor_locations_data_scale
    automatic_initialization_parameters["charging_stations_locations"] = charging_stations_locations_data_scale
    rescaled_automatic_initialization_parameters["ground_sensor_locations"] = ground_sensor_locations_opt_scale
    rescaled_automatic_initialization_parameters["charging_stations_locations"] = charging_stations_locations_opt_scale
    
    # 3. Initialize drones
    # print("initializing drones")
    Routing_Strat = drone_routing_strategy(rescaled_automatic_initialization_parameters, rescaled_custom_initialization_parameters)
    # print("getting initial drone locations")
    initial_drone_locations_and_state_opt_scale = Routing_Strat.get_initial_drone_locations()
    drones = [Drone(x*coverage_width_cells+coverage_width_cells//2,y*coverage_width_cells+coverage_width_cells//2,state,charging_stations_locations_data_scale,automatic_initialization_parameters["N"],automatic_initialization_parameters["M"], automatic_initialization_parameters["max_battery_distance"], automatic_initialization_parameters["max_battery_time"],automatic_initialization_parameters["max_battery_distance"]-1*(state=='fly'), automatic_initialization_parameters["max_battery_time"]-1*(state=='fly')) for (state,(x,y)) in initial_drone_locations_and_state_opt_scale]
    
    drone_locations_data_scale = [drone.get_position() for drone in drones]
    drone_locations_opt_scale = [(coord[0], coord[1]) for (status,coord) in initial_drone_locations_and_state_opt_scale]

    drone_batteries = [drone.get_battery() for drone in drones]
    drone_batteries_opt_scale = [rescaled_max_battery_time for drone in drones]
    drone_states = [drone.get_state() for drone in drones]

    # Initialize metrics
    execution_times = []
    drone_visited_cells = set(drone_locations_data_scale)
    total_distance_traveled = 0

    fire_size_cells = 0
    fire_size_percentage = 0
    # ========================

    t_found = 0
    device = 'undetected'

    # 4. OPTIONNAL: Load a custom DL burn map model
    # # Load trained burn map model (once)
    # burn_predictor = BurnMapPredictor(
    # model_path=custom_initialization_parameters["model_path"],     # You must add this param when calling this function
    # burn_map_path=custom_initialization_parameters["burnmap_filename"],
    # num_weather_timesteps=5
    # )


    # # Load real weather history
    # weather_file = custom_initialization_parameters["weather_file"]
    # num_timesteps = burn_predictor.num_weather_timesteps
    # concat_len = burn_predictor.concat_len
    # features_per_timestep = concat_len // num_timesteps

    fire_detected = False
    for time_step in tqdm.tqdm(range(-starting_time,len(canonical_scenario))):
        for substep in range(operational_substeps):
            # === Routing & movement ===
            custom_step_parameters = custom_step_parameters_function()
            automatic_step_parameters_opt_scale = {
                "drone_locations": drone_locations_opt_scale,
                "drone_batteries": drone_batteries_opt_scale,
                "drone_states": drone_states,
                "t": t_found
            }
            start_time = time.time()
            actions_opt_scale = Routing_Strat.next_actions(automatic_step_parameters_opt_scale, custom_step_parameters)
            new_position_opt_scale = [] # THIS IS TEMPORARY! WE NEED TO PROPERLY RESCALE BACK FROM DRONE.ROUTE BUT FOR THE INTEREST OF TIME WE FEED BACK THE OUTPUT /!\ TODO
            for drone_index, action in enumerate(actions_opt_scale):
                if action[0] in ['charge', 'fly']:
                    new_position_opt_scale.append(action[1])
                elif action[0] in ['move']:
                    new_position_opt_scale.append((max(0, min(automatic_initialization_parameters["N"]-1, drone_locations_opt_scale[drone_index][0] + action[1][0])), max(0, min(automatic_initialization_parameters["M"]-1, drone_locations_opt_scale[drone_index][1] + action[1][1]))))
            
            actions_data_scale = []
            for action in actions_opt_scale:
                action_type, coords_opt_scale = action

                if action_type in ['move', 'fly']:
                    
                    if action_type == 'fly':
                        converted = operational_space_to_dataspace_coordinates( 
                            coords_opt_scale,
                            coverage=coverage_radius_m,
                            datacell_size_m=cell_size_m,
                        )
                    else:
                        converted = (coverage_width_cells*coords_opt_scale[0], coverage_width_cells*coords_opt_scale[1])

                    actions_data_scale.append((action_type, converted))
                else:
                    action_type, coords_opt_scale = action
                    coords_data_scale = operational_space_to_dataspace_coordinates(coords_opt_scale, coverage=coverage_radius_m, datacell_size_m=cell_size_m)
                    actions_data_scale.append((action_type, coords_data_scale)) #TODO check that for charging mode
            
            execution_times.append(time.time() - start_time)

            # === Move drones and check detection ===
            for drone_index, (drone, action) in enumerate(zip(drones, actions_data_scale)):
                if not drone.is_alive():
                    # continue  # Skip dead drones #TODO figure out battery rescaling for dead drones
                    pass

                old_x_data_scale, old_y_data_scale = drone_locations_data_scale[drone_index]
                new_x_data_scale, new_y_data_scale, new_distance_battery, new_time_battery, new_state = drone.route(action)
                
                # new_x_opt_scale = math.ceil((new_x_data_scale-coverage_width_cells//2)/coverage_width_cells)
                # new_y_opt_scale = math.ceil((new_y_data_scale-coverage_width_cells//2)/coverage_width_cells)
                # ABOVE IS WHAT WE SHOULD USE BUT WE DONT. SEE COMMENTS ABOVE /!\ TODO

                drone_locations_data_scale[drone_index] = (new_x_data_scale, new_y_data_scale)
                drone_locations_opt_scale[drone_index] = new_position_opt_scale[drone_index]
                if action[0] in ['move', 'fly']:
                    drone_batteries_opt_scale[drone_index] -= 1
                elif action[0] in ['charge']:
                    drone_batteries_opt_scale[drone_index] = rescaled_max_battery_time

                drone_batteries[drone_index] = (new_distance_battery, new_time_battery)
                drone_states[drone_index] = new_state
                total_distance_traveled += abs(new_x_data_scale - old_x_data_scale) + abs(new_y_data_scale - old_y_data_scale)
                drone_visited_cells.update(generate_coverage_area(coverage_radius_m, cell_size_m, new_x_data_scale, new_y_data_scale, automatic_initialization_parameters["N"], automatic_initialization_parameters["M"]))
        t_found += 1

    return time.time() - time_start

    

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



def pool_burnmap_mean(burnmap, kernel_size):
    """
    Pool the burnmap to the new size by averaging the values in the new cells, with a kernel_size x kernel_size window and a stride of kernel_size.
    """
    N, M = burnmap.shape[1:]
    N_new = N // kernel_size
    M_new = M // kernel_size
    burnmap_pooled = np.zeros((burnmap.shape[0], N_new, M_new))
    for i in range(N_new):
        for j in range(M_new):
            burnmap_pooled[:, i, j] = np.mean(burnmap[:, i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size], axis=(1,2))
    return burnmap_pooled

def pool_burnmap_proba_at_least_one(burnmap, kernel_size):
    """
    Pool the burnmap to the new size by 1 - prod(1 - p) the values in the new cells, with a kernel_size x kernel_size window and a stride of kernel_size.
    """
    N, M = burnmap.shape[1:]
    N_new = N // kernel_size
    M_new = M // kernel_size
    burnmap_pooled = np.zeros((burnmap.shape[0], N_new, M_new))
    for i in range(N_new):
        for j in range(M_new):
            burnmap_pooled[:, i, j] = 1 - np.prod(1 - burnmap[:, i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size], axis=(1,2))
    return burnmap_pooled

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


# def run_benchmark_scenarii(input_dir, ground_placement_strategy, drone_routing_strategy, ground_parameters, routing_parameters, max_n_scenarii=None):
#     """
#     Run parallel benchmarks on multiple scenarios using thread pooling.

#     Args:
#         input_dir (str): Directory containing scenario files.
#         ground_placement_strategy (function): Strategy for placing ground sensors and charging stations.
#         drone_routing_strategy (function): Strategy for controlling drone movements.
#         ground_parameters (tuple): Parameters for ground placement strategy.
#         routing_parameters (tuple): Parameters for routing strategy.
#         max_n_scenarii (int, optional): Maximum number of scenarios to process. If None, processes all scenarios.

#     Prints:
#         Average time steps to fire detection and detection statistics by device type.
#     """
#     # TODO: add starting time
#     raise NotImplementedError("Starting time is not implemented yet")
#     if not input_dir.endswith('/'):
#         input_dir += '/'

#     iterable = listdir_txt_limited(input_dir, max_n_scenarii)

#     M = len(os.listdir(input_dir)) if max_n_scenarii is None else max_n_scenarii
    
#     def process_scenario(infile):
#         start = GroundPlacementOptimization(10,10,100,"burn_maps/burn_map_1")
#         return 0,'undetected'

#     # Initialize counters
#     delta_ts = 0
#     fails = 0
#     devices = {'ground sensor': 0, "charging station": 0, "drone": 0, 'undetected': 0}
    
#     # Use ThreadPoolExecutor to parallelize scenario processing
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         # Use tqdm to show progress bar for parallel execution
#         results = list(tqdm.tqdm(executor.map(process_scenario, iterable), total=M))
        
#     # Process results
#     for delta_t, device in results:
#         if delta_t == -1:
#             fails += 1
#             delta_t = 0
#         delta_ts += delta_t
#         devices[device] += 1
    
#     print(f"This strategy took on average {delta_ts/max(1,(M-fails))} time steps to find the fire.")
#     for device in devices.keys():
#         print(f"Fire found {round(devices[device]/M*100,2)}% of the time by {device}")

def run_benchmark_scenario(scenario: np.ndarray, sensor_placement_strategy:SensorPlacementStrategy, drone_routing_strategy:DroneRoutingStrategy, custom_initialization_parameters:dict, custom_step_parameters_function:callable, starting_time:int=0, return_history:bool=False, return_history_scale:str='data', custom_initialization_parameters_function:callable=None, automatic_initialization_parameters_function:callable=None, input_dir:str='', simulation_parameters:dict={}, progress_bar:bool=False, precomputing_time:float=0, scenario_name:str=""):
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
        return_history_scale (str, optional): Scale of the returned history. Must be either 'data' or 'operational' (or 'opt'). Defaults to 'data'.

    Returns:
        tuple: Contains:
            - delta_t (int): Time steps taken to detect fire, or -1 if undetected
            - device (str): Which device detected the fire ('ground sensor', 'charging station', 'drone', or 'undetected')
            - history (tuple): If return_history=True, returns (drone_locations_history, ground_sensor_locations, charging_stations_locations) at the specified scale
    """
    if return_history and return_history_scale not in ['data', 'operational', 'opt']:
        raise ValueError("return_history_scale must be either 'data' or 'operational' (or 'opt')")

    # 0. Get layout parameters
    if automatic_initialization_parameters_function is None:
        automatic_initialization_parameters = get_automatic_layout_parameters(scenario, input_dir, simulation_parameters, scenario_name)
    else:
        automatic_initialization_parameters = automatic_initialization_parameters_function(scenario, input_dir, simulation_parameters, scenario_name)
    
    if custom_initialization_parameters_function is not None:
        custom_initialization_parameters = custom_initialization_parameters_function(input_dir)

    # 1. compute the rescaling factors
    cell_size_m = automatic_initialization_parameters.get("cell_size_m", 30)
    speed_m_per_min = automatic_initialization_parameters.get("speed_m_per_min", 9)
    coverage_radius_m = automatic_initialization_parameters.get("coverage_radius_m", 45)

    operational_substeps = compute_operational_substeps(cell_size_m, speed_m_per_min, coverage_radius_m)
    coverage_width_cells = round(coverage_radius_m*2 / cell_size_m)

    rescaled_N = automatic_initialization_parameters["N"] // coverage_width_cells
    rescaled_M = automatic_initialization_parameters["M"] // coverage_width_cells
    rescaled_max_battery_time = automatic_initialization_parameters["max_battery_time"] * operational_substeps
    
    rescaled_burnmap = load_burn_map(custom_initialization_parameters["burnmap_filename"])
    
    rescaled_burnmap = pool_burnmap_mean(rescaled_burnmap, coverage_width_cells)
    rescaled_burnmap = np.repeat(rescaled_burnmap, operational_substeps, axis=0)/operational_substeps # we also rescale the probabilities to time scale
    #rescaled_burnmap = rescaled_burnmap.astype(np.float32)
    rescaled_burnmap_filename = custom_initialization_parameters["burnmap_filename"].replace(".npy", f"_rescaled_{rescaled_N}x{rescaled_M}_{operational_substeps}substeps.npy")
    np.save(rescaled_burnmap_filename, rescaled_burnmap)

    rescaled_automatic_initialization_parameters = automatic_initialization_parameters.copy()
    rescaled_automatic_initialization_parameters["N"] = rescaled_N
    rescaled_automatic_initialization_parameters["M"] = rescaled_M
    rescaled_automatic_initialization_parameters["max_battery_time"] = rescaled_max_battery_time
    rescaled_automatic_initialization_parameters["burnmap_filename"] = rescaled_burnmap_filename

    rescaled_custom_initialization_parameters = custom_initialization_parameters.copy()
    rescaled_custom_initialization_parameters["burnmap_filename"] = rescaled_burnmap_filename

    # 2. Get ground sensor locations and convert them back to the original size
    ground_sensor_locations_opt_scale, charging_stations_locations_opt_scale =  sensor_placement_strategy(rescaled_automatic_initialization_parameters, rescaled_custom_initialization_parameters).get_locations()

    ground_sensor_locations_data_scale = [(x*coverage_width_cells+coverage_width_cells//2, y*coverage_width_cells+coverage_width_cells//2) for x,y in ground_sensor_locations_opt_scale]
    charging_stations_locations_data_scale = [(x*coverage_width_cells+coverage_width_cells//2, y*coverage_width_cells+coverage_width_cells//2) for x,y in charging_stations_locations_opt_scale]
    rows_ground_data_scale, cols_ground_data_scale = zip(*ground_sensor_locations_data_scale) if ground_sensor_locations_data_scale else ((),())
    rows_charging_data_scale, cols_charging_data_scale = zip(*charging_stations_locations_data_scale) if charging_stations_locations_data_scale else ((),())

    # add computed positions to initialization parameters
    automatic_initialization_parameters["ground_sensor_locations"] = ground_sensor_locations_data_scale
    automatic_initialization_parameters["charging_stations_locations"] = charging_stations_locations_data_scale
    rescaled_automatic_initialization_parameters["ground_sensor_locations"] = ground_sensor_locations_opt_scale
    rescaled_automatic_initialization_parameters["charging_stations_locations"] = charging_stations_locations_opt_scale
    
    # 3. Initialize drones

    Routing_Strat = drone_routing_strategy(rescaled_automatic_initialization_parameters, rescaled_custom_initialization_parameters)
    
    initial_drone_locations_and_state_opt_scale = Routing_Strat.get_initial_drone_locations()
    drones = [Drone(x*coverage_width_cells+coverage_width_cells//2,y*coverage_width_cells+coverage_width_cells//2,state,charging_stations_locations_data_scale,automatic_initialization_parameters["N"],automatic_initialization_parameters["M"], automatic_initialization_parameters["max_battery_distance"], automatic_initialization_parameters["max_battery_time"],automatic_initialization_parameters["max_battery_distance"]-1*(state=='fly'), automatic_initialization_parameters["max_battery_time"]-1*(state=='fly')) for (state,(x,y)) in initial_drone_locations_and_state_opt_scale]
    
    drone_locations_data_scale = [drone.get_position() for drone in drones]
    drone_locations_opt_scale = [(coord[0], coord[1]) for (status,coord) in initial_drone_locations_and_state_opt_scale]

    drone_batteries = [drone.get_battery() for drone in drones]
    drone_batteries_opt_scale = [rescaled_max_battery_time for drone in drones]
    drone_states = [drone.get_state() for drone in drones]
    drone_locations_history = None
    drone_locations_history_opt = None
    if return_history:
        drone_locations_history = [list(drone_locations_data_scale)]
        drone_locations_history_opt = [list(drone_locations_opt_scale)]


    # Initialize metrics
    execution_times = []
    drone_visited_cells = set(drone_locations_data_scale)
    total_distance_traveled = 0
    fire_size_cells = 0
    fire_size_percentage = 0
    # ========================

    t_found = 0
    device = 'undetected'

    # # 4. Load the burn map model
    # # Load trained burn map model (once)
    # burn_predictor = BurnMapPredictor(
    # model_path=custom_initialization_parameters["model_path"],     # You must add this param when calling this function
    # burn_map_path=custom_initialization_parameters["burnmap_filename"],
    # num_weather_timesteps=5
    # )


    # # Load real weather history
    # weather_file = custom_initialization_parameters["weather_file"]
    # num_timesteps = burn_predictor.num_weather_timesteps
    # concat_len = burn_predictor.concat_len
    # features_per_timestep = concat_len // num_timesteps

    fire_detected = False
    if progress_bar:
        tqdm_iter = tqdm.tqdm(range(-starting_time,len(scenario)))
    else:
        tqdm_iter = range(-starting_time,len(scenario))
    for time_step in tqdm_iter:
        if time_step >= 0: # The fire has started.
            grid = scenario[time_step]
            
            if ground_sensor_locations_data_scale:
                if (grid[rows_ground_data_scale,cols_ground_data_scale]==1).any():
                    fire_detected = True
                    device = 'ground sensor'
                    fire_size_cells = np.sum(grid > 0.5)
                    fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                    break

            if charging_stations_locations_data_scale:
                if (grid[rows_charging_data_scale,cols_charging_data_scale]==1).any():
                    fire_detected = True
                    device = 'charging station'
                    fire_size_cells = np.sum(grid > 0.5)
                    fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                    break


        for substep in range(operational_substeps):

            # === Routing & movement ===
            custom_step_parameters = custom_step_parameters_function()
            automatic_step_parameters_opt_scale = {
                "drone_locations": drone_locations_opt_scale,
                "drone_batteries": drone_batteries_opt_scale,
                "drone_states": drone_states,
                "t": t_found
            }
            start_time = time.time()
            # print(f"substep {substep}, time_step {time_step}")
            actions_opt_scale = Routing_Strat.next_actions(automatic_step_parameters_opt_scale, custom_step_parameters)
            new_position_opt_scale = [] # THIS IS TEMPORARY! WE NEED TO PROPERLY RESCALE BACK FROM DRONE.ROUTE BUT FOR THE INTEREST OF TIME WE FEED BACK THE OUTPUT /!\ TODO
            for drone_index, action in enumerate(actions_opt_scale):
                if action[0] in ['charge', 'fly']:
                    new_position_opt_scale.append(action[1])
                elif action[0] in ['move']:
                    new_position_opt_scale.append((max(0, min(automatic_initialization_parameters["N"]-1, drone_locations_opt_scale[drone_index][0] + action[1][0])), max(0, min(automatic_initialization_parameters["M"]-1, drone_locations_opt_scale[drone_index][1] + action[1][1]))))
            
            actions_data_scale = []
            for action in actions_opt_scale:
                action_type, coords_opt_scale = action

                if action_type in ['move', 'fly']:
                    if action_type == 'fly':
                        converted = operational_space_to_dataspace_coordinates( 
                            coords_opt_scale,
                            coverage=coverage_radius_m,
                            datacell_size_m=cell_size_m,
                        )
                    else:
                        converted = (coverage_width_cells*coords_opt_scale[0], coverage_width_cells*coords_opt_scale[1])

                    actions_data_scale.append((action_type, converted))
                else:
                    action_type, coords_opt_scale = action
                    coords_data_scale = operational_space_to_dataspace_coordinates(coords_opt_scale, coverage=coverage_radius_m, datacell_size_m=cell_size_m)
                    actions_data_scale.append((action_type, coords_data_scale)) #TODO check that for charging mode
            
            execution_times.append(time.time() - start_time)

            # === Move drones and check detection ===
            for drone_index, (drone, action) in enumerate(zip(drones, actions_data_scale)):
                if not drone.is_alive():
                    #print(f"Drone {drone_index} is dead")
                    # continue  # Skip dead drones #TODO figure out battery rescaling for dead drones
                    pass

                old_x_data_scale, old_y_data_scale = drone_locations_data_scale[drone_index]
                new_x_data_scale, new_y_data_scale, new_distance_battery, new_time_battery, new_state = drone.route(action)

                # new_x_opt_scale = math.ceil((new_x_data_scale-coverage_width_cells//2)/coverage_width_cells)
                # new_y_opt_scale = math.ceil((new_y_data_scale-coverage_width_cells//2)/coverage_width_cells)
                # ABOVE IS WHAT WE SHOULD USE BUT WE DONT. SEE COMMENTS ABOVE /!\ TODO

                drone_locations_data_scale[drone_index] = (new_x_data_scale, new_y_data_scale)
                drone_locations_opt_scale[drone_index] = new_position_opt_scale[drone_index]
                if action[0] in ['move', 'fly']:
                    drone_batteries_opt_scale[drone_index] -= 1
                elif action[0] in ['charge']:
                    drone_batteries_opt_scale[drone_index] = rescaled_max_battery_time

                drone_batteries[drone_index] = (new_distance_battery, new_time_battery)
                drone_states[drone_index] = new_state
                total_distance_traveled += abs(new_x_data_scale - old_x_data_scale) + abs(new_y_data_scale - old_y_data_scale)
                drone_visited_cells.add((new_x_data_scale, new_y_data_scale))
            
            if return_history:
                drone_locations_history.append(tuple(drone_locations_data_scale))
                drone_locations_history_opt.append(tuple(drone_locations_opt_scale))

            
            # === Drone fire detection ===
            for drone_pos in drone_locations_data_scale:
                if time_step >= 0 and detect_fire_within_coverage(grid, drone_pos, coverage_width_cells):
                    device = 'drone'
                    fire_detected = True
                    fire_size_cells = np.sum(grid > 0.5)
                    fire_size_percentage = fire_size_cells / (grid.shape[0] * grid.shape[1]) * 100
                    break
       
            if fire_detected:
                break
        
        if fire_detected:
            break
            
        t_found += 1
        
    delta_t = t_found - starting_time
    for i in range(1, len(execution_times)):
        execution_times[i] += precomputing_time  # Add precomputing time to all execution times except the first one

    avg_execution_time = np.mean(execution_times)
    percentage_map_explored = len(drone_visited_cells) / (automatic_initialization_parameters["N"] * automatic_initialization_parameters["M"]) * 100

    if device == 'undetected':
        delta_t = len(scenario)
        final_grid = scenario[-1]
        fire_size_cells = np.sum(final_grid > 0.5)
        fire_size_percentage = fire_size_cells / (final_grid.shape[0] * final_grid.shape[1]) * 100

    results = {
        "substeps_per_timestep": operational_substeps, #just for creating scenario videos
        "delta_t": delta_t,
        "device": device,
        "avg_execution_time": avg_execution_time,
        "fire_size_cells": fire_size_cells,
        "fire_size_percentage": fire_size_percentage,
        "percentage_map_explored": percentage_map_explored,
        "total_distance_traveled": total_distance_traveled,
    }
    
    if return_history:
        if return_history_scale == 'data':
            return results, (drone_locations_history, ground_sensor_locations_data_scale, charging_stations_locations_data_scale)
        else:  # operational scale
            return results, (drone_locations_history_opt, ground_sensor_locations_opt_scale, charging_stations_locations_opt_scale)
    return results, ()


def run_benchmark_scenarii_sequential(input_dir, sensor_placement_strategy:SensorPlacementStrategy, drone_routing_strategy:DroneRoutingStrategy, custom_initialization_parameters_function:callable, custom_step_parameters_function:callable, starting_time:int=0, max_n_scenarii:int=None, file_format="npy", simulation_parameters:dict={}, config:dict={}, precomputing_time:float=0, experiment_name:str=''):
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

    per_scenario_results = []
    count = 0
    for file in tqdm.tqdm(iterable, total = N_SCENARII):
        scenario_name = file.split('/')[-1]
        starting_time = config.get(f"offset_{scenario_name}", 0)
        # print(f"Starting time: {starting_time}, file: {scenario_name}")
        #print(f"Starting time: {starting_time}, file: {scenario_name}")
        scenario = load_scenario_fn(file)
        if automatic_initialization_parameters is None:
            # Compute initialization parameters
            automatic_initialization_parameters = get_automatic_layout_parameters(scenario, input_dir, simulation_parameters, scenario_name) #TODO compute them once only per layout rather than per scenario..
        results, _ = run_benchmark_scenario(
            scenario,
            sensor_placement_strategy,
            drone_routing_strategy,
            custom_initialization_parameters,
            custom_step_parameters_function,
            starting_time=starting_time,
            input_dir=input_dir,
            simulation_parameters=simulation_parameters,
            return_history= count == 0,
            precomputing_time=precomputing_time,
            scenario_name=scenario_name,
        )

        delta_t = results["delta_t"]
        device = results["device"]

        if delta_t == -1:
            fails += 1
            delta_t = 0

        delta_ts += delta_t
        devices[device] += 1

        scenario_name = os.path.basename(file).split('_')[-1].split('.')[0]
        layout_name = os.path.basename(file).split('_')[0]

        sensor_strategy_name = sensor_placement_strategy.strategy_name

        drone_strategy_name = drone_routing_strategy.strategy_name
                
        # print("precomputing_time", precomputing_time)

        # if precomputing_time > 0:
        #     total_execution_times.append(results["avg_execution_time"] + precomputing_time)
        # else:
        #     total_execution_times.append(results["avg_execution_time"])

        per_scenario_results.append({
            "sensor_strategy": sensor_strategy_name,
            "drone_strategy": drone_strategy_name,
            "layout": layout_name,
            "scenario": scenario_name,
            "delta_t": delta_t,
            "device": device,
            "execution_time": results["avg_execution_time"],
            "fire_size_cells": results["fire_size_cells"],
            "fire_percentage": results["fire_size_percentage"],
            "map_explored": results["percentage_map_explored"],
            "total_distance": results["total_distance_traveled"],
        })
        total_fire_sizes.append(results["fire_size_cells"])
        total_fire_percentages.append(results["fire_size_percentage"])
        map_explored.append(results["percentage_map_explored"])
        total_distances.append(results["total_distance_traveled"])
   
    layout_dir ="/".join(input_dir.split('/')[:-2])
    csv_output_path = os.path.join(layout_dir, f"{layout_name}_benchmark_results{experiment_name}_{sensor_strategy_name}_{drone_strategy_name}.csv")
    df = pd.DataFrame(per_scenario_results)
    df.to_csv(csv_output_path, index=False)
    # print(f"Saved per-scenario results to {csv_output_path}")
    
    # Calculate metrics
    avg_time_to_detection = delta_ts / max(1, (N_SCENARII - fails))
    device_percentages = {device: round(count / N_SCENARII * 100, 2) for device, count in devices.items()}
    avg_execution_time = np.mean(total_execution_times)
    avg_fire_size = np.mean(total_fire_sizes)
    avg_fire_percentage = np.mean(total_fire_percentages)
    avg_map_explored = np.mean(map_explored)
    avg_distance = np.mean(total_distances)
    
    # Create metrics dictionary
    metrics = {
        "avg_time_to_detection": avg_time_to_detection,
        "device_percentages": device_percentages,
        "avg_execution_time": avg_execution_time,
        "avg_fire_size": avg_fire_size,
        "avg_fire_percentage": avg_fire_percentage,
        "avg_map_explored": avg_map_explored,
        "avg_distance": avg_distance,
        "raw_execution_times": total_execution_times,
        "raw_fire_sizes": total_fire_sizes,
        "raw_fire_percentages": total_fire_percentages,
        "raw_map_explored": map_explored,
        "raw_distances": total_distances,
    }
    
    return metrics

def run_benchmark_scenarii_parallel(input_dir, sensor_placement_strategy: SensorPlacementStrategy, drone_routing_strategy: DroneRoutingStrategy, custom_initialization_parameters_function: callable, custom_step_parameters_function: callable, starting_time: int = 0, max_n_scenarii: int = None, file_format="npy", simulation_parameters: dict = {}, config: dict = {}, precomputing_time: float = 0):
    """
    Run parallel benchmarks on multiple scenarios.

    Args:
        input_dir (str): Directory containing scenario files.
        sensor_placement_strategy (function): Strategy for placing ground sensors and charging stations.
        drone_routing_strategy (function): Strategy for controlling drone movements.
        custom_step_parameters_function (function): Function for custom step parameters.
        starting_time (int, optional): Time step at which the wildfire starts.
        max_n_scenarii (int, optional): Maximum number of scenarios to process. If None, processes all scenarios.

    Returns:
        dict: Metrics dictionary containing benchmark results.
    """
    if file_format not in ["npy", "jpg"]:
        raise ValueError("file_format must be 'npy' or 'jpg'")

    if not input_dir.endswith('/'):
        input_dir += '/'

    # Choose iterator and loader
    if file_format == "npy":
        iterable = listdir_npy_limited(input_dir, max_n_scenarii)
        load_scenario_fn = load_scenario_npy
    else:
        iterable = listdir_folder_limited(input_dir, max_n_scenarii)
        load_scenario_fn = load_scenario_jpg

    N_SCENARII = max_n_scenarii if max_n_scenarii else len(os.listdir(input_dir))

    # Initialize counters
    delta_ts = 0
    fails = 0
    devices = {'ground sensor': 0, 'charging station': 0, 'drone': 0, 'undetected': 0}
    total_execution_times = []
    total_fire_sizes = []
    total_fire_percentages = []
    map_explored = []
    total_distances = []

    # Extract layout name
    layout_name = os.path.basename(os.path.dirname(input_dir))

    # Determine how many args custom init fn accepts
    import inspect
    sig = inspect.signature(custom_initialization_parameters_function)
    param_count = len(sig.parameters)
    if param_count >= 2:
        custom_initialization_parameters = custom_initialization_parameters_function(input_dir, layout_name)
    else:
        custom_initialization_parameters = custom_initialization_parameters_function(input_dir)

    # Define single-scenario processor
    def process_scenario(file):
        scenario = load_scenario_fn(file)
        offset = config.get(f"offset_{os.path.basename(file)}", 0)
        results, _ = run_benchmark_scenario(
            scenario,
            sensor_placement_strategy,
            drone_routing_strategy,
            custom_initialization_parameters,
            custom_step_parameters_function,
            starting_time=offset,
            input_dir=input_dir,
            simulation_parameters=simulation_parameters,
            return_history=False
        )
        return results

    # Parallel execution
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_scenario, file): file for file in iterable}
        for fut in as_completed(futures):
            try:
                results = fut.result()
                delta_t = results['delta_t']
                device = results['device']
                if delta_t == -1:
                    fails += 1
                    delta_t = 0
                delta_ts += delta_t
                devices[device] += 1
                total_execution_times.append(results['avg_execution_time'] + precomputing_time)
                total_fire_sizes.append(results['fire_size_cells'])
                total_fire_percentages.append(results['fire_size_percentage'])
                map_explored.append(results['percentage_map_explored'])
                total_distances.append(results['total_distance_traveled'])
            except Exception as e:
                print(f"Error processing {futures[fut]}: {e}")

    # Compute metrics
    avg_time_to_detection = delta_ts / max(1, (N_SCENARII - fails))
    device_percentages = {d: round(c / N_SCENARII * 100, 2) for d, c in devices.items()}
    metrics = {
        'avg_time_to_detection': avg_time_to_detection,
        'device_percentages': device_percentages,
        'avg_execution_time': np.mean(total_execution_times),
        'avg_fire_size': np.mean(total_fire_sizes),
        'avg_fire_percentage': np.mean(total_fire_percentages),
        'avg_map_explored': np.mean(map_explored),
        'avg_distance': np.mean(total_distances),
        'raw_execution_times': total_execution_times,
        'raw_fire_sizes': total_fire_sizes,
        'raw_fire_percentages': total_fire_percentages,
        'raw_map_explored': map_explored,
        'raw_distances': total_distances,
    }
    return metrics



def run_benchmark_scenarii_sequential_precompute(input_dir, sensor_placement_strategy:SensorPlacementStrategy, drone_routing_strategy:DroneRoutingStrategy, custom_initialization_parameters_function:callable, custom_step_parameters_function:callable, starting_time:int=0, max_n_scenarii:int=None, file_format="npy", simulation_parameters:dict={}, config:dict={}):
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
    max_scenario_plus_offset_length = 0
    canonical_scenario = None
    canonical_offset = 0
    canonical_scenario_name = ""
    # print(f"finding the longest scenario in {input_dir}")
    for file in iterable:
        scenario_name = file.split('/')[-1].split('.')[0]
        # print(f"checking scenario {scenario_name}")
        scenario = load_scenario_fn(file)
        offset = config.get(f"offset_{scenario_name}", 0)
        # print(f"scenario {scenario_name} has length {scenario.shape[0]} + offset {offset} = {scenario.shape[0] + offset}")
        if scenario.shape[0] + offset > max_scenario_plus_offset_length:
            max_scenario_plus_offset_length = scenario.shape[0] + offset
            canonical_scenario = scenario
            canonical_offset = offset
            canonical_scenario_name = scenario_name
    if canonical_scenario is None:
        print(f"No scenario found in {input_dir}")
        return {}
    # find the biggest offset in config
    # print(f"Canonical offset: {canonical_offset}")
    print("precomputing")
    
    precomputing_time = run_drone_routing_strategy(drone_routing_strategy, sensor_placement_strategy, max_scenario_plus_offset_length, canonical_scenario, get_automatic_layout_parameters, custom_initialization_parameters_function, custom_step_parameters_function, input_dir, simulation_parameters, file_format, starting_time = canonical_offset, scenario_name=canonical_scenario_name)
    return run_benchmark_scenarii_sequential(input_dir, sensor_placement_strategy, drone_routing_strategy, custom_initialization_parameters_function, custom_step_parameters_function, starting_time, max_n_scenarii, file_format, simulation_parameters, config, precomputing_time, experiment_name)

def benchmark_on_sim2real_dataset_precompute(dataset_folder_name, ground_placement_strategy, drone_routing_strategy, custom_initialization_parameters_function, custom_step_parameters_function, max_n_scenarii=None, starting_time=0, max_n_layouts=None, simulation_parameters:dict={}, skip_folder_names:list=[], file_format="npy", config_file:str='', experiment_name:str=''):
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
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    if not dataset_folder_name.endswith('/'):
        dataset_folder_name += '/'
    
    all_metrics = {}
    count = 0
    for layout_folder in listdir_folder_limited(dataset_folder_name, max_n_layouts):
        print(f"\n --- \n Processing layout {layout_folder}, {count}")
        count += 1
        layout_name = os.path.basename(layout_folder)
        scenarios_folder = "/scenarii/" if file_format == "npy" else "/Satellite_Images_Mask/"
        if not os.path.exists(layout_folder + scenarios_folder):
            print(f"No {scenarios_folder} folder found in {layout_folder}, tryng other folder name")
            scenarios_folder = "/Satellite_Image_Mask/"
            if not os.path.exists(layout_folder + scenarios_folder):
                print(f"No {scenarios_folder} folder found in {layout_folder}, skipping...")
                continue

        if layout_folder in skip_folder_names:
            print(f"Skipping layout {layout_folder} because it is in the skip_folder_names list")
            continue

        # check if there is not too many discarded scenarios

        # load the selected_scenarios.txt file and check the failed percentage
        print("checking if there is not too many discarded scenarios")
        failed_percentage = 1
        selected_scenarios_file = layout_folder + "/selected_scenarios.txt"
        if os.path.exists(selected_scenarios_file):
            with open(selected_scenarios_file, "r") as f:
                lines = f.readlines()
            failed_percentage = float(lines[-1].split(" ")[-1])
        if failed_percentage > 0.2:
            print(f"Skipping layout {layout_folder} because it has more than 20% failed scenarios: {failed_percentage}")
            continue
        print("running benchmark")
        # try:
        metrics = run_benchmark_scenarii_sequential_precompute(
            layout_folder + scenarios_folder,
            ground_placement_strategy, 
            drone_routing_strategy, 
            custom_initialization_parameters_function, 
            custom_step_parameters_function, 
        starting_time=starting_time, 
        max_n_scenarii=max_n_scenarii,
        simulation_parameters=simulation_parameters,
        file_format=file_format,
        config=config,
        experiment_name=experiment_name,
        )
        all_metrics[layout_name] = metrics
        # except Exception as e:
        #     print(f"Error running benchmark on layout {layout_folder}: {e}")
        #     continue
        
    
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
                               custom_step_params_fn= return_no_custom_parameters,
                               simulation_parameters: dict = {},
                               config: dict = {}):
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

     # === Auto-wrap the strategies ===
    SensorPlacementStrategyClass = wrap_log_sensor_strategy(SensorPlacementStrategyClass)
    DroneRoutingStrategyClass = wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingStrategyClass))

    # === Load the first scenario to get parameters ===
    first_file = next(iter(iterable), None)
    if first_file is None:
        print(f"No scenarios found in {input_dir}")
        return

    # load the first scenario to get automatic parameters
    first_scenario_name = first_file.split('/')[-1]
    scenario = load_scenario_fn(first_file)
    automatic_init_params = get_automatic_layout_parameters(scenario, input_dir, simulation_parameters, first_scenario_name)

    # === Create sensor placement strategy ===
    print("[run_benchmark_for_strategy] Running sensor placement strategy...")

    custom_init_params = custom_init_params_fn(input_dir, layout_name=os.path.basename(input_dir))

    # === Create sensor placement strategy ===
    sensor_placement_strategy_instance = SensorPlacementStrategyClass(automatic_init_params, custom_init_params)

    # === Get ground and charging station locations ===
    ground_sensor_locations, charging_station_locations = sensor_placement_strategy_instance.get_locations()

    # === Update automatic_init_params with sensor locations ===
    automatic_init_params["ground_sensor_locations"] = ground_sensor_locations
    automatic_init_params["charging_stations_locations"] = charging_station_locations

    # === Now create the drone routing strategy ===
    drone_routing_strategy_instance = DroneRoutingStrategyClass(automatic_init_params, custom_init_params)

    # === call the existing function ===
    metrics = run_benchmark_scenarii_sequential(
        input_dir=input_dir,
        sensor_placement_strategy=lambda *_: sensor_placement_strategy_instance,
        drone_routing_strategy=lambda *_: drone_routing_strategy_instance,
        custom_initialization_parameters_function=custom_init_params_fn,
        custom_step_parameters_function=custom_step_params_fn,
        starting_time=starting_time,
        max_n_scenarii=max_n_scenarii,
        file_format=file_format,
        simulation_parameters=simulation_parameters,
        config=config,
        precomputing_time=0
    )
    
    return metrics
