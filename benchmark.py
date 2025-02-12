from Drone import Drone
import time
import numpy as np
import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor

from dataset import load_scenario_npy



def run_benchmark_scenario(scenario,starting_time,ground_placement,routing_strategy,ground_parameters, routing_static_parameters, return_history = False):

    ground_sensor_locations, charging_stations_locations =  ground_placement(*ground_parameters).get_locations()
    rows_ground, cols_ground = zip(*ground_sensor_locations)
    rows_charging, cols_charging = zip(*charging_stations_locations)

    Routing_Strat = routing_strategy(ground_sensor_locations, charging_stations_locations,*routing_static_parameters)

    drones = [Drone(x,y,charging_stations_locations,len(scenario[0])) for (x,y) in Routing_Strat.initialize()]
    drone_locations = [drone.get_position() for drone in drones]
    drone_batteries = [drone.get_battery() for drone in drones]
    drone_locations_history = [list(drone_locations)]

    t_found = 0
    device = 'undetected'

    for grid in scenario:
        if t_found >= starting_time:
            # check if a fire is detected
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
        # move the drones
        actions = Routing_Strat.next_actions(drone_locations,drone_batteries,t_found)
        for drone_index, (drone,action) in enumerate(zip(drones,actions)):
            new_x, new_y, new_distance_battery, new_time_battery = drone.route(action)
            drone_locations[drone_index] = (new_x,new_y)
            drone_batteries = (new_distance_battery,new_time_battery)
        if return_history:
            drone_locations_history.append(tuple(drone_locations))

    delta_t = t_found-starting_time if t_found != len(scenario) else -1
    return delta_t, device, (drone_locations_history, ground_sensor_locations, charging_stations_locations) if return_history else ()


def listdir_txt_limited(input_dir, max_n_scenarii=None):
    count = 0
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.txt'):
                yield input_dir + entry.name
                count += 1
                if max_n_scenarii is not None and count >= max_n_scenarii:
                    break

def listdir_npy_limited(input_dir, max_n_scenarii=None):
    count = 0
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.npy'):
                yield input_dir + entry.name
                count += 1
                if max_n_scenarii is not None and count >= max_n_scenarii:
                    break


from Strategy import GroundPlacementOptimization
def run_benchmark_scenarii(input_dir, ground_placement_strategy, drone_routing_strategy, ground_parameters, routing_parameters, max_n_scenarii=None):
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

from Strategy import GroundPlacementOptimization
def run_benchmark_scenarii_sequential(input_dir, ground_placement_strategy, drone_routing_strategy, ground_parameters, routing_parameters, max_n_scenarii=None):
    if not input_dir.endswith('/'):
        input_dir += '/'

    iterable = listdir_npy_limited(input_dir, max_n_scenarii)

    M = len(os.listdir(input_dir)) if max_n_scenarii is None else max_n_scenarii

    # Initialize counters
    delta_ts = 0
    fails = 0
    devices = {'ground sensor': 0, "charging station": 0, "drone": 0, 'undetected': 0}
    
    for file in tqdm.tqdm(iterable, total = M):
        
        scenario, start_time = load_scenario_npy(file)
        delta_t, device, _ = run_benchmark_scenario(scenario, start_time, ground_placement_strategy, drone_routing_strategy, ground_parameters, routing_parameters)
        
        if delta_t == -1:
            fails += 1
            delta_t = 0
        delta_ts += delta_t
        devices[device] += 1
    
    print(f"This strategy took on average {delta_ts/max(1,(M-fails))} time steps to find the fire.")
    for device in devices.keys():
        print(f"Fire found {round(devices[device]/M*100,2)}% of the time by {device}")

def run_ground_log(input_dir, output_file, ground_placement_strategy, ground_parameters, max_n_scenarii=None):
    if not input_dir.endswith('/'):
        input_dir += '/'

    iterable = listdir_txt_limited(input_dir, max_n_scenarii)
    all_placements = []
    for file in tqdm.tqdm(iterable, total = max_n_scenarii):
        ground_sensor_locations, charging_stations_locations =  ground_placement_strategy(*ground_parameters).get_locations()
        all_placements.append((ground_sensor_locations,charging_stations_locations))
    with open(output_file, "w") as outfile:
        json.dump(all_placements, outfile)

def benchmark_on_sim2real_dataset(dataset_folder_name, ground_placement_strategy, drone_routing_strategy, ground_parameters, routing_parameters, max_n_scenarii=None):
    if not dataset_folder_name.endswith('/'):
        dataset_folder_name += '/'
    
    for layout_folder in os.listdir(dataset_folder_name):
        if not os.path.exists(dataset_folder_name + layout_folder + "/scenarii/"):continue
        run_benchmark_scenarii_sequential(dataset_folder_name + layout_folder + "/scenarii/",ground_placement_strategy,drone_routing_strategy,ground_parameters,routing_parameters,max_n_scenarii)