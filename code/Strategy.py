import random
import os
import json
import numpy as np
from dataset import load_scenario, save_burn_map
import time
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Backend selection: WFDRONE_OPT_BACKEND = julia (default) | python
# ---------------------------------------------------------------------------
_OPT_BACKEND = os.environ.get("WFDRONE_OPT_BACKEND", "julia").lower()

if _OPT_BACKEND == "julia":
    from my_julia_caller import Main as jl
else:
    import opt as _pyopt


### Utilities ####

#### Gaussian Coverage Utilities ####
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import numpy as np

def count_paths_convolution(N, M, n, time_steps=1):
    N_prime, M_prime = min(2*N+1,2*n), min(2*M+1,2*n)
    dp = np.zeros((N_prime, M_prime), dtype=np.float64)
    origin = (N_prime//2, M_prime//2) # origin is the middle of the grid
    dp[origin[0], origin[1]] = 1

    kernel = np.ones((3, 3), dtype=np.float64)
    for _ in range(n):
        dp = convolve(dp, kernel, mode='constant', cval=0.0)
    
    origin_value = dp[origin[0], origin[1]]
    mapping = {(x - origin[0],y - origin[1]): min(dp[x,y]*time_steps/origin_value, 1.00) for x in range(N_prime) for y in range(M_prime)}

    return mapping

def return_no_custom_parameters():
    """
    Return an empty dictionary as no custom parameters are needed.
    """
    return {}


#### BASE CLASSES FOR DRONE AND SENSOR STRATEGIES ####

class SensorPlacementStrategy():
    """
    Base class for sensor (ground stations and charging stations) placement strategies.
    """
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using random placement.
        
        Args:
            automatic_initialization_parameters: dict with keys:
                "N": Grid height
                "M": Grid width
                "max_battery_distance": int
                "max_battery_time": int
                "n_drones": int
                "n_ground_stations": Target number of ground stations
                "n_charging_stations": Target number of charging stations
                "ground_sensor_locations": list of tuples (x,y)
            custom_initialization_parameters: dict
        Returns:
            ground_sensor_locations: list of tuples (x,y)
            charging_station_locations: list of tuples (x,y)
        """
        raise NotImplementedError("SensorPlacementStrategy is an abstract class and should not be instantiated directly.")
        # YOUR CODE HERE
        # Generate random positions
        self.ground_sensor_locations = [(random.randint(0, automatic_initialization_parameters["N"]-1), 
                                       random.randint(0, automatic_initialization_parameters["M"]-1)) 
                                      for _ in range(automatic_initialization_parameters["n_ground_stations"])]
        
        self.charging_station_locations = [(random.randint(0, automatic_initialization_parameters["N"]-1), 
                                          random.randint(0, automatic_initialization_parameters["M"]-1)) 
                                         for _ in range(automatic_initialization_parameters["n_charging_stations"])]

    def get_locations(self):
        """
        Returns the locations of the ground sensors and charging stations
        """
        # Do not overwrite this function
        return self.ground_sensor_locations, self.charging_station_locations

class DroneRoutingStrategy():
    """
    Base class for drone routing strategies.
    
    This class defines the interface that all drone routing strategies must implement.
    A drone routing strategy determines how drones move around the grid to detect fires
    while managing their battery levels and charging requirements.
    
    Args:
        automatic_initialization_parameters (dict): Parameters automatically provided by the system:
            - N (int): Grid height
            - M (int): Grid width
            - max_battery_distance (int): Maximum distance a drone can travel before recharging
            - max_battery_time (int): Maximum time a drone can fly before recharging
            - n_drones (int): Number of drones to control
            - n_ground_stations (int): Number of ground sensor stations
            - n_charging_stations (int): Number of charging stations
            - ground_sensor_locations (list): List of (x,y) tuples for ground sensors
            - charging_stations_locations (list): List of (x,y) tuples for charging stations
        custom_initialization_parameters (dict): Strategy-specific parameters
    """
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        automatic_initialization_parameters: dict with keys:
            "N": Grid height
            "M": Grid width
            "max_battery_distance": int
            "max_battery_time": int
            "n_drones": int
            "n_ground_stations": Target number of ground stations
            "n_charging_stations": Target number of charging stations
            "ground_sensor_locations": list of tuples (x,y)
            "charging_stations_locations": list of tuples (x,y)
        custom_initialization_parameters: dict
        """
        raise NotImplementedError("DroneRoutingStrategy is an abstract class and should not be instantiated directly.")
        # assign parameters
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters

        # Any intial computations
        # YOUR CODE HERE

    def get_initial_drone_locations(self):
        """
        Returns the initial locations and states for all drones.
        
        Returns:
            list: List of tuples (state, (x,y)) where:
                - state is either 'charge' or 'fly'
                - (x,y) are the initial coordinates
                All drones must start at charging stations (state='charge')
        """
        raise NotImplementedError("get_initial_drone_locations is an abstract method and should be implemented by subclasses.")
        
        n = len(self.automatic_initialization_parameters["charging_stations_locations"])
        q = self.automatic_initialization_parameters["n_drones"] // n
        r = self.automatic_initialization_parameters["n_drones"] % n
        
        # By default drones are spread uniformly aross charging stations
        return self.automatic_initialization_parameters["charging_stations_locations"]*q + self.automatic_initialization_parameters["charging_stations_locations"][:r]
    
    def next_actions(self, automatic_step_parameters:dict, custom_step_parameters:dict):
        """
        automatic_step_parameters: dict with keys:
            "drone_locations": list of tuples (x,y)
            "drone_batteries": list of tuples (distance,time)
            "t": int
        custom_step_parameters: dict
        Returns:
            list: List of tuples (action_type, action_parameters) where:
                - action_type is one of: 'move', 'fly', 'charge'
                - action_parameters are the coordinates or movement deltas
        """
        raise NotImplementedError("next_actions is an abstract method and should be implemented by subclasses.")

#### RANDOM STRATEGIES ####

# Random Sensor Placement Strategy in the paper
class RandomSensorPlacementStrategy(SensorPlacementStrategy):
    strategy_name = "RandomSensorPlacementStrategy"

    """
    Sensor placement strategy that places sensors randomly.
    """
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using random placement.
        
        Args:
            automatic_initialization_parameters: dict with keys:
                "N": Grid height
                "M": Grid width
                "max_battery_distance": int
                "max_battery_time": int
                "n_drones": int
                "n_ground_stations": Target number of ground stations
                "n_charging_stations": Target number of charging stations
                "ground_sensor_locations": list of tuples (x,y)
            custom_initialization_parameters: dict
        Returns:
            ground_sensor_locations: list of tuples (x,y)
            charging_station_locations: list of tuples (x,y)
        """
        # Generate random positions using list comprehensions
        # YOUR CODE HERE
        #print("RandomSensorPlacementStrategy")
        self.ground_sensor_locations = [(random.randint(0, automatic_initialization_parameters["N"]-1), 
                                       random.randint(0, automatic_initialization_parameters["M"]-1)) 
                                      for _ in range(automatic_initialization_parameters["n_ground_stations"])]
        
        self.charging_station_locations = [(random.randint(0, automatic_initialization_parameters["N"]-1), 
                                          random.randint(0, automatic_initialization_parameters["M"]-1)) 
                                         for _ in range(automatic_initialization_parameters["n_charging_stations"])]

    def get_locations(self):
        """
        Returns the locations of the ground sensors and charging stations
        """
        # Do not overwrite this function
        return self.ground_sensor_locations, self.charging_station_locations

# Brownian Strategy in the paper
class RandomDroneRoutingStrategy(DroneRoutingStrategy):
    strategy_name = "RandomDroneRoutingStrategy"
    """
    Drone routing strategy that moves drones randomly.
    """
    def __init__(self,automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        automatic_initialization_parameters: dict with keys:
            "N": Grid height
            "M": Grid width
            "max_battery_distance": int
            "max_battery_time": int
            "n_drones": int
            "n_ground_stations": Target number of ground stations
            "n_charging_stations": Target number of charging stations
            "ground_sensor_locations": list of tuples (x,y)
            "charging_stations_locations": list of tuples (x,y)
        custom_initialization_parameters: dict
        """
        # assign parameters
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters

        # Any intial computations
        # YOUR CODE HERE

    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones
        """
        
        n = len(self.automatic_initialization_parameters["charging_stations_locations"])
        q = self.automatic_initialization_parameters["n_drones"] // n
        r = self.automatic_initialization_parameters["n_drones"] % n
        
        # By default drones are spread uniformly aross charging stations
        positions = self.automatic_initialization_parameters["charging_stations_locations"]*q + self.automatic_initialization_parameters["charging_stations_locations"][:r]
        return [('charge',(x,y)) for x,y in positions]

    
    def sign(self,x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def next_actions(self, automatic_step_parameters:dict, custom_step_parameters:dict):
        """
        automatic_step_parameters: dict with keys:
            "drone_locations": list of tuples (x,y)
            "drone_batteries": list of int
            "t": int
        custom_step_parameters: dict
        Returns:
            actions: list of tuples (action_type, action_parameters)
        """
        moving_plan = []
        for i, (x,y) in enumerate(automatic_step_parameters["drone_locations"]):
            if automatic_step_parameters["drone_batteries"][i] == 0:
                moving_plan.append(('charge',(x,y)))
            else:
                # find the closest charging station in chebyshev distance
                closest_charging_station = min(self.automatic_initialization_parameters["charging_stations_locations"], key=lambda c: max(abs(x-c[0]),abs(y-c[1])))
                closest_distance = max(abs(x-closest_charging_station[0]),abs(y-closest_charging_station[1]))
                # if current distance to the charging station is equal to the remaiing battery time, move to the charging station
                if closest_distance == automatic_step_parameters["drone_batteries"][i]:
                    moving_plan.append(('move',(self.sign(closest_charging_station[0]-x),self.sign(closest_charging_station[1]-y))))
                    # otherwise, move randomly
                else:
                    moving_plan.append(('move',(random.randint(-1,1),random.randint(-1,1))))
        return moving_plan
        

#### STRATEGIES CALLING JULIA OPTIMIZATION MODELS ####

# Base class Max Coverage Sensor Placement Strategy (we use the gaussian coverage version in the paper)
class SensorPlacementOptimization(SensorPlacementStrategy):
    strategy_name = "SensorPlacementOptimization"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using optimization model.
        
        Args:
            automatic_initialization_parameters: dict with keys:
                "n_ground_stations": Target number of ground stations
                "n_charging_stations": Target number of charging stations
                "N": Grid height
                "M": Grid width
            custom_initialization_parameters: dict with keys:
                "burnmap_filename": burn map file name
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")

        print("calling optimization model (backend=%s)" % _OPT_BACKEND)
        if _OPT_BACKEND == "julia":
            x_vars, y_vars = jl.SENSOR_MAXCOV_STRATEGY(
                custom_initialization_parameters["burnmap_filename"],
                automatic_initialization_parameters["n_ground_stations"],
                automatic_initialization_parameters["n_charging_stations"],
            )
        else:
            x_vars, y_vars = _pyopt.sensor_maxcov_strategy(
                custom_initialization_parameters["burnmap_filename"],
                automatic_initialization_parameters["n_ground_stations"],
                automatic_initialization_parameters["n_charging_stations"],
            )
        print("optimization finished")
        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)

# Sensor Placement Gaussian Coverage Strategy in the paper
class SensorPlacementMaxCoverageGaussianTime(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTime"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using Julia's optimization model.
        
        Args:
            automatic_initialization_parameters: dict with keys:
                "n_ground_stations": Target number of ground stations
                "n_charging_stations": Target number of charging stations
                "n_drones": Target number of drones
                "N": Grid height
                "M": Grid width
            custom_initialization_parameters: dict with keys:
                "burnmap_filename": burn map file name
        """
        # Initialize empty lists (skip parent's random initialization)
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")

        # load the burnmap
        burnmap = load_scenario(custom_initialization_parameters["burnmap_filename"])
        T, N, M = burnmap.shape
        middle_point = (N//2, M//2)


        # create the kernel
        kernel =count_paths_convolution(N, M, 63, 10)
        kernel_size_x = 63
        kernel_size_y = 63
        # print("kernel=\n", kernel)


        if _OPT_BACKEND == "julia":
            x_vars, y_vars = jl.Max_Coverage_Kernel(
                custom_initialization_parameters["burnmap_filename"],
                automatic_initialization_parameters["n_ground_stations"],
                automatic_initialization_parameters["n_charging_stations"],
                automatic_initialization_parameters["n_drones"],
                kernel, kernel_size_x, kernel_size_y,
            )
        else:
            x_vars, y_vars = _pyopt.max_coverage_kernel(
                custom_initialization_parameters["burnmap_filename"],
                automatic_initialization_parameters["n_ground_stations"],
                automatic_initialization_parameters["n_charging_stations"],
                automatic_initialization_parameters["n_drones"],
                kernel, kernel_size_x, kernel_size_y,
            )
        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)


# Drone Uniform Coverage Routing Strategy in the paper
class DroneRoutingUniformCoverageResetStatic(DroneRoutingStrategy):
    strategy_name = "DroneRoutingUniformCoverageResetStatic"
    """
    Drone routing strategy that uses a max coverage approach and resets the burn map at every reevaluation step.
    """
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        automatic_initialization_parameters: dict with keys:
            "N": Grid height
            "M": Grid width
            "max_battery_distance": int
            "max_battery_time": int
            "n_drones": int
            "n_ground_stations": Target number of ground stations
            "n_charging_stations": Target number of charging stations
            "ground_sensor_locations": list of tuples (x,y)
            "charging_stations_locations": list of tuples (x,y)
        custom_initialization_parameters: dict with keys:
            "burnmap_filename": burn map file name
            "burnamap_type": static or dynamic
            "reevaluation_step": number of steps between calls to julia optimization model
            "optimization_horizon": number of steps to optimize for
            "regularization_param": regularization parameter for the objective
        """
        # Assign parameters
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters
        self.call_counter = 0  # Keeping track of how many times we call the function
        self.t = 0 # current timestep
        self.current_solution = None
        self.routing_model = None  # Will store the reusable JuMP model
        self.call_ID = random.randint(0, 1000000)
        self.burnmap_type = custom_initialization_parameters.get("burnmap_type", "static")
        # Validate required parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        self.initial_burnmap = load_scenario(self.custom_initialization_parameters["burnmap_filename"])
        self.current_burnmap = self.initial_burnmap.copy()
        
        # duplicate the data to go from shape (1,N,M) to shape (100,N,M)
        self.current_burnmap = np.ones((6000, self.initial_burnmap.shape[1], self.initial_burnmap.shape[2]))
        self.len_burnmap = self.initial_burnmap.shape[0]
        self.current_burnmap_filename = "./tmp_burnmaps/tmp_burnmap_" + str(self.call_ID) + ".npy"
        # create the tmp_burnmaps folder if it doesn't exist
        if not os.path.exists("./tmp_burnmaps"):
            os.makedirs("./tmp_burnmaps")
        self.automatic_initialization_parameters["burnmap_filename"] = self.current_burnmap_filename
        
        if "reevaluation_step" not in custom_initialization_parameters:
            raise ValueError("reevaluation_step is not defined")
        self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("optimization_horizon is not defined")
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2*63)
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Station locations – keep 0-based originals; Julia needs 1-based
        self.charging_stations_0 = list(self.automatic_initialization_parameters["charging_stations_locations"])
        self.ground_stations_0 = list(self.automatic_initialization_parameters["ground_sensor_locations"])
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.charging_stations_0]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.ground_stations_0]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        save_burn_map(self.current_burnmap, self.current_burnmap_filename)

        start_time = time.time()
        if _OPT_BACKEND == "julia":
            self.routing_model = jl.create_index_routing_model(
                self.current_burnmap_filename,
                self.automatic_initialization_parameters["n_drones"],
                self.julia_charging_stations_locations,
                self.julia_ground_sensor_locations,
                self.custom_initialization_parameters["optimization_horizon"],
                self.automatic_initialization_parameters["max_battery_time"],
            )
        else:
            self.routing_model = _pyopt.create_routing_model(
                self.current_burnmap_filename,
                self.automatic_initialization_parameters["n_drones"],
                self.charging_stations_0,
                self.ground_stations_0,
                self.custom_initialization_parameters["optimization_horizon"],
                self.automatic_initialization_parameters["max_battery_time"],
            )
        self.execution_time += time.time() - start_time

        start_time = time.time()
        if _OPT_BACKEND == "julia":
            self.current_solution = jl.solve_index_init_routing(
                self.routing_model, 
                self.custom_initialization_parameters["reevaluation_step"]
            )
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
        else:
            self.current_solution = _pyopt.solve_init_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
            )
        self.execution_time += time.time() - start_time

        initial_positions = self.current_solution[0]
        self.call_counter = 0
        
        print("Initial optimization finished")
        print(f"\nDEBUG: Available Charging Stations (after model creation): {self.charging_stations_locations}")

        return initial_positions

        
        
    def next_actions(self, automatic_step_parameters:dict, custom_step_parameters:dict):
        """
        automatic_step_parameters: dict with keys:
            "drone_locations": list of tuples (x,y)
            "drone_batteries": list of tuples (distance,time)
            "drone_states": list of strings "charge" or "fly"
            "t": int
        custom_step_parameters: dict 
        Returns:
            actions: list of tuples (action_type, action_parameters)
        """
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time

            start_time = time.time()
            if _OPT_BACKEND == "julia":
                julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
                self.current_solution = jl.solve_index_next_move_routing(
                    self.routing_model,
                    self.custom_initialization_parameters["reevaluation_step"],
                    julia_drone_locations,
                    automatic_step_parameters["drone_states"],
                    automatic_step_parameters["drone_batteries"],
                    self.t
                )
                self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                         for code, (x, y) in plan] for plan in self.current_solution]
            else:
                self.current_solution = _pyopt.solve_next_move_routing(
                    self.routing_model,
                    self.custom_initialization_parameters["reevaluation_step"],
                    automatic_step_parameters["drone_locations"],
                    automatic_step_parameters["drone_states"],
                    automatic_step_parameters["drone_batteries"],
                    self.t,
                )
            self.execution_time += time.time() - start_time

        self.call_counter += 1
        idx = min(self.call_counter, len(self.current_solution) - 1)
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                self.current_burnmap[self.t:self.t+self.reset_time,action[1][0], action[1][1]] = 0
        self.t += 1
        return self.current_solution[idx]

# Drone Max Coverage Routing Strategy in the paper
class DroneRoutingMaxCoverageResetStatic(DroneRoutingStrategy):
    strategy_name = "DroneRoutingMaxCoverageResetStatic"
    """
    Drone routing strategy that uses a max coverage approach and resets the burn map at every reevaluation step.
    """
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        automatic_initialization_parameters: dict with keys:
            "N": Grid height
            "M": Grid width
            "max_battery_distance": int
            "max_battery_time": int
            "n_drones": int
            "n_ground_stations": Target number of ground stations
            "n_charging_stations": Target number of charging stations
            "ground_sensor_locations": list of tuples (x,y)
            "charging_stations_locations": list of tuples (x,y)
        custom_initialization_parameters: dict with keys:
            "burnmap_filename": burn map file name
            "reevaluation_step": number of steps between calls to julia optimization model
            "optimization_horizon": number of steps to optimize for
        """
        # Assign parameters
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters
        self.call_counter = 0  # Keeping track of how many times we call the function
        self.t = 0 # current timestep
        self.current_solution = None
        self.routing_model = None  # Will store the reusable JuMP model
        self.call_ID = random.randint(0, 1000000)
        self.burnmap_type = custom_initialization_parameters.get("burnmap_type", "static")
        # Validate required parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        self.initial_burnmap = load_scenario(self.custom_initialization_parameters["burnmap_filename"])
        self.current_burnmap = self.initial_burnmap.copy()
        if self.burnmap_type == "static":
            # duplicate the data to go from shape (1,N,M) to shape (100,N,M)
            self.current_burnmap = np.tile(self.initial_burnmap, (90, 1, 1))
        else:
            print(f"careful: burnmap_type is not static, it is {self.burnmap_type}")
        

        self.len_burnmap = self.initial_burnmap.shape[0]
        self.current_burnmap_filename = "./tmp_burnmaps/tmp_burnmap_" + str(self.call_ID) + ".npy"
        # create the tmp_burnmaps folder if it doesn't exist
        if not os.path.exists("./tmp_burnmaps"):
            os.makedirs("./tmp_burnmaps")
        self.automatic_initialization_parameters["burnmap_filename"] = self.current_burnmap_filename

        self.current_burnmap+=1e-8
        save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        
        if "reevaluation_step" not in custom_initialization_parameters:
            raise ValueError("reevaluation_step is not defined")
        self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("optimization_horizon is not defined")
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2*63)
        
        # Station locations
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        self.charging_stations_0 = list(self.charging_stations_locations)
        self.ground_stations_0 = list(automatic_initialization_parameters["ground_sensor_locations"])
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.charging_stations_0]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.ground_stations_0]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        save_burn_map(self.current_burnmap, self.current_burnmap_filename)

        start_time = time.time()
        if _OPT_BACKEND == "julia":
            self.routing_model = jl.create_index_routing_model(
                self.current_burnmap_filename,
                self.automatic_initialization_parameters["n_drones"],
                self.julia_charging_stations_locations,
                self.julia_ground_sensor_locations,
                self.custom_initialization_parameters["optimization_horizon"],
                self.automatic_initialization_parameters["max_battery_time"],
            )
        else:
            self.routing_model = _pyopt.create_routing_model(
                self.current_burnmap_filename,
                self.automatic_initialization_parameters["n_drones"],
                self.charging_stations_0,
                self.ground_stations_0,
                self.custom_initialization_parameters["optimization_horizon"],
                self.automatic_initialization_parameters["max_battery_time"],
            )
        self.execution_time += time.time() - start_time

        start_time = time.time()
        if _OPT_BACKEND == "julia":
            self.current_solution = jl.solve_index_init_routing(
                self.routing_model, 
                self.custom_initialization_parameters["reevaluation_step"]
            )
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
        else:
            self.current_solution = _pyopt.solve_init_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
            )
        self.execution_time += time.time() - start_time

        initial_positions = self.current_solution[0]
        self.call_counter = 0
        
        print("Initial optimization finished")
        print(f"\nDEBUG: Available Charging Stations (after model creation): {self.charging_stations_locations}")

        return initial_positions

        
        
    def next_actions(self, automatic_step_parameters:dict, custom_step_parameters:dict):
        """
        automatic_step_parameters: dict with keys:
            "drone_locations": list of tuples (x,y)
            "drone_batteries": list of tuples (distance,time)
            "drone_states": list of strings "charge" or "fly"
            "t": int
        custom_step_parameters: dict 
        Returns:
            actions: list of tuples (action_type, action_parameters)
        """
        self.t += 1
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time

            start_time = time.time()
            if _OPT_BACKEND == "julia":
                julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
                self.current_solution = jl.solve_index_next_move_routing(
                    self.routing_model,
                    self.custom_initialization_parameters["reevaluation_step"],
                    julia_drone_locations,
                    automatic_step_parameters["drone_states"],
                    automatic_step_parameters["drone_batteries"],
                    self.t
                )
                self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                         for code, (x, y) in plan] for plan in self.current_solution]
            else:
                self.current_solution = _pyopt.solve_next_move_routing(
                    self.routing_model,
                    self.custom_initialization_parameters["reevaluation_step"],
                    automatic_step_parameters["drone_locations"],
                    automatic_step_parameters["drone_states"],
                    automatic_step_parameters["drone_batteries"],
                    self.t,
                )
            self.execution_time += time.time() - start_time

        self.call_counter += 1
        idx = self.call_counter
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
        return self.current_solution[idx]
