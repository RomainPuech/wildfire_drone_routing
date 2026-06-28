import random
import os
import uuid
from my_julia_caller import Main as jl
import json
import numpy as np
from dataset import load_scenario, save_burn_map
import time
from typing import List, Tuple


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

class RandomSensorPlacementStrategy(SensorPlacementStrategy):
    strategy_name = "RandomSensorPlacementStrategy"

    """
    Sensor placement strategy that places sensors randomly on unmasked cells.
    If a mask is provided (via mask_filename in automatic_initialization_parameters),
    sensors and charging stations are only placed on cells where mask == 1.
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
                "mask_filename": (optional) path to mask file; cells with mask==0 are blocked
            custom_initialization_parameters: dict
        Returns:
            ground_sensor_locations: list of tuples (x,y)
            charging_station_locations: list of tuples (x,y)
        """
        N = automatic_initialization_parameters["N"]
        M = automatic_initialization_parameters["M"]

        # Load mask and build list of valid (unmasked) cells
        mask_filename = automatic_initialization_parameters.get("mask_filename", None)
        if mask_filename is not None:
            mask = np.load(mask_filename)
            valid_cells = [(i, j) for i in range(N) for j in range(M) if mask[i, j] > 0]
        else:
            valid_cells = [(i, j) for i in range(N) for j in range(M)]

        if len(valid_cells) == 0:
            raise ValueError("No valid (unmasked) cells available for sensor placement.")

        n_ground = automatic_initialization_parameters["n_ground_stations"]
        n_charging = automatic_initialization_parameters["n_charging_stations"]

        # Sample from valid cells (with replacement if needed)
        self.ground_sensor_locations = [valid_cells[random.randint(0, len(valid_cells)-1)]
                                       for _ in range(n_ground)]
        self.charging_station_locations = [valid_cells[random.randint(0, len(valid_cells)-1)]
                                          for _ in range(n_charging)]

    def get_locations(self):
        """
        Returns the locations of the ground sensors and charging stations
        """
        # Do not overwrite this function
        return self.ground_sensor_locations, self.charging_station_locations

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
class SensorPlacementOptimization(SensorPlacementStrategy):
    strategy_name = "SensorPlacementOptimization"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using Julia's optimization model.
        
        Args:
            automatic_initialization_parameters: dict with keys:
                "n_ground_stations": Target number of ground stations
                "n_charging_stations": Target number of charging stations
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

     
        # Call the Julia optimization function
        print("calling julia optimization model")
        x_vars, y_vars = jl.NEW_SENSOR_STRATEGY(custom_initialization_parameters["burnmap_filename"], automatic_initialization_parameters["n_ground_stations"], automatic_initialization_parameters["n_charging_stations"])
        print("optimization finished")
        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)


class FixedPlacementStrategy(SensorPlacementStrategy):
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        self.charging_station_locations = [(35,14), (30,42)]
        self.ground_sensor_locations = []

class DroneRoutingOptimizationModelReuseIndex(DroneRoutingStrategy):
    strategy_name = "DroneRoutingOptimizationModelReuseIndex"

    """
    Drone routing strategy that uses the model reuse approach for improved performance.
    This class is functionally equivalent to DroneRoutingOptimizationSlow but uses model
    reuse to speed up computations by preserving the optimization model between calls.
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
        self.current_solution = None
        self.routing_model = None  # Will store the reusable JuMP model

        # Validate required parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        
        if "reevaluation_step" not in custom_initialization_parameters:
            raise ValueError("reevaluation_step is not defined")
        self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("optimization_horizon is not defined")
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        ground_sensors = self.automatic_initialization_parameters["ground_sensor_locations"]
        if ground_sensors:
            self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in ground_sensors]
        else:
            # Create empty list with explicit tuple type that PyCall can convert
            self.julia_ground_sensor_locations: List[Tuple[int, int]] = []
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        self.routing_model = jl.create_index_routing_model(
            self.custom_initialization_parameters["burnmap_filename"],
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"]
        )
        
        # Solve the initial routing problem with the model
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            #print("THE PB IS HERE : ", automatic_step_parameters["drone_locations"])
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )

            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            

        if self.current_solution is None:
            raise RuntimeError("Julia optimization did not return a solution.")
        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = min(self.call_counter, len(self.current_solution) - 1)
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        return self.current_solution[idx]

#### HEURISTIC STRATEGIES ####
class GREEDY_DRONE_STRATEGY(DroneRoutingStrategy):
    strategy_name = "GREEDY_DRONE_STRATEGY"
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
            "call_every_n_steps": number of steps between calls to julia optimization model
            "optimization_horizon": number of steps to optimize for
        """
        # assign parameters
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters
        self.call_counter = 0 # keeping track of how many time we call function to know when to call julia
        self.current_solution = None


        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        
        if "call_every_n_steps" not in custom_initialization_parameters:
            raise ValueError("call_every_n_steps is not defined")
        self.call_every_n_steps = custom_initialization_parameters["call_every_n_steps"]
        
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("optimization_horizon is not defined")
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones
        """
        # Uniform allocation of drones across charging stations (you can change this)
        
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
            actions: list of tuples (action_type, action_parameters)
        """
        # suggest actions
        if self.call_counter % self.call_every_n_steps == 0: # every `call_every_n_steps` calls, we call julia optimization model again
            # calling julia optimization model
            print("calling julia optimization model")
            # REPLACE HERE BY YOUR JULIA FUNCTION
            self.current_solution = jl.NEW_drone_routing_example(automatic_step_parameters["drone_locations"], automatic_step_parameters["drone_batteries"], self.custom_initialization_parameters["burnmap_filename"], self.custom_initialization_parameters["optimization_horizon"])
            print("optimization finished")

        return self.current_solution[self.call_counter % self.call_every_n_steps]


#### STRATEGIES THAT USE A LOG FILE (temporary #TODO use the wrapper instead) ####

class LoggedOptimizationSensorPlacementStrategy(SensorPlacementStrategy):
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using Julia's optimization model.
        
        Args:
            automatic_initialization_parameters: dict with keys:
                "n_ground_stations": Target number of ground stations
                "n_charging_stations": Target number of charging stations
                "N": Grid height
                "M": Grid width
            custom_initialization_parameters: dict with keys:
                "burnmap_filename": burn map file name
                "log_filename": Path to the log file
                "load_from_logfile": boolean
        """
        # Initialize empty lists (skip parent's random initialization)
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")

        if "log_filename" not in custom_initialization_parameters:
            custom_initialization_parameters["log_filename"] = "/".join(custom_initialization_parameters["burnmap_filename"].split("/")[:-1]) + f"/{automatic_initialization_parameters['n_ground_stations']}_{automatic_initialization_parameters['n_charging_stations']}_logged_sensor_placement.json"

        if "load_from_logfile" not in custom_initialization_parameters:
            custom_initialization_parameters["load_from_logfile"] = True


        if custom_initialization_parameters["load_from_logfile"] and os.path.exists(custom_initialization_parameters["log_filename"]):
            self.ground_sensor_locations, self.charging_station_locations = json.load(open(custom_initialization_parameters["log_filename"]))
        else:
            print("calling julia optimization model")
            x_vars, y_vars = jl.ground_charging_opt_model_grid(custom_initialization_parameters["burnmap_filename"], automatic_initialization_parameters["n_ground_stations"], automatic_initialization_parameters["n_charging_stations"])
            print("optimization finished")
            # save the result in a json file
            with open(custom_initialization_parameters["log_filename"], "w") as f:
                json.dump([list(x_vars), list(y_vars)], f)
            self.ground_sensor_locations = list(x_vars)
            self.charging_station_locations = list(y_vars)

class LoggedSensorPlacementStrategy(SensorPlacementStrategy):
        def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
            """
            Initialize the ground placement strategy using a log file. If no log is found,
            compute the sensor placement and log it for future runs.

            Args:
                automatic_initialization_parameters: dict 
                    Expected keys:
                        - n_ground_stations
                        - n_charging_stations
                        - N, M (grid size)
                custom_initialization_parameters: dict
                    Expected keys:
                        - log_file: Path to the log file
                        - burnmap_filename: Path to the burn map used by the Julia optimizer

            Returns:
                Initializes:
                    self.ground_sensor_locations: list of tuples (x, y)
                    self.charging_station_locations: list of tuples (x, y)
            """
            
            # Ensure required custom params exist
            if "log_file" not in custom_initialization_parameters:
                raise ValueError("custom_initialization_parameters must include 'log_file'")
            if "burnmap_filename" not in custom_initialization_parameters:
                raise ValueError("custom_initialization_parameters must include 'burnmap_filename'")

            
            # Extract the layout name from custom params (if available)
            layout_name = custom_initialization_parameters.get("layout_name", os.path.basename(custom_initialization_parameters["burnmap_filename"]))

            # Get n_ground_stations
            n_ground_stations = automatic_initialization_parameters.get("n_ground_stations", 0)

            # Get strategy name
            strategy_name = self.__class__.__name__

            # Build the log directory
            log_dir = os.path.dirname(custom_initialization_parameters["log_file"])
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Build the descriptive logfile name
            logfile = os.path.join(
                log_dir,
                f"{layout_name}_{strategy_name}_{n_ground_stations}_sensors.json"
            )

            burnmap_filename = custom_initialization_parameters["burnmap_filename"]

            self.ground_sensor_locations = []
            self.charging_station_locations = []

            # Check if the log file already exists
            if os.path.exists(logfile):
                print(f"[LoggedSensorPlacementStrategy] Loading placements from log file: {logfile}")
                with open(logfile, "r") as log:
                    data = json.load(log)
                    self.ground_sensor_locations = data["ground_sensor_locations"]
                    self.charging_station_locations = data["charging_station_locations"]

            else:
                print(f"[LoggedSensorPlacementStrategy] Log file not found at {logfile}. Running optimization...")
                print("calling julia optimization model")
            
                # Run Julia optimization function
                x_vars, y_vars = jl.ground_charging_opt_model_grid(
                    burnmap_filename,
                    automatic_initialization_parameters["n_ground_stations"],
                    automatic_initialization_parameters["n_charging_stations"]
                )
                print("optimization finished")
                # Save the locations
                self.ground_sensor_locations = list(x_vars)
                self.charging_station_locations = list(y_vars)

                # Write the results to the log file
                with open(logfile, "w") as log:
                    json.dump({
                        "ground_sensor_locations": self.ground_sensor_locations,
                        "charging_station_locations": self.charging_station_locations
                    }, log, indent=2)

                print(f"[LoggedSensorPlacementStrategy] Optimization done. Results saved to {logfile}")

                # print(f"[LoggedSensorPlacementStrategy] Log file not found at {logfile}. Running dummy optimization...")

                #     # MOCK: replace Julia optimization with dummy values
                #     # for example, just generate some random positions
    
                # n_ground_stations = automatic_initialization_parameters["n_ground_stations"]
                # n_charging_stations = automatic_initialization_parameters["n_charging_stations"]
                # N = automatic_initialization_parameters["N"]
                # M = automatic_initialization_parameters["M"]

                # # dummy lists of random locations
                # import random
                # x_vars = [(random.randint(0, N-1), random.randint(0, M-1)) for _ in range(n_ground_stations)]
                # y_vars = [(random.randint(0, N-1), random.randint(0, M-1)) for _ in range(n_charging_stations)]

                # # Save the locations
                # self.ground_sensor_locations = list(x_vars)
                # self.charging_station_locations = list(y_vars)
                
                # log_dir = os.path.dirname(logfile)
                # if not os.path.exists(log_dir):
                #     os.makedirs(log_dir, exist_ok=True)
                # # Write the results to the log file
                # with open(logfile, "w") as log:
                #     json.dump({
                #         "ground_sensor_locations": self.ground_sensor_locations,
                #         "charging_station_locations": self.charging_station_locations
                #     }, log, indent=2)

                # print(f"[LoggedSensorPlacementStrategy] Dummy optimization done. Results saved to {logfile}")


        def get_locations(self):
            return self.ground_sensor_locations, self.charging_station_locations

#### TEMPLATES FOR NEW STRATEGIES ####


class LogWrapperDrone(DroneRoutingStrategy):
    def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
        # We don't change the strategy name here, because we want to keep the same name as the strategy we are wrapping
        self.call_counter = 0
        self.strategy = custom_initialization_parameters["strategy_drone"](automatic_initialization_parameters, custom_initialization_parameters)
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters
        self.logfile = "/".join(custom_initialization_parameters["burnmap_filename"].split("/")[:-1]) + f"/{custom_initialization_parameters['strategy_drone'].__name__}_{automatic_initialization_parameters['n_drones']}_drones_{automatic_initialization_parameters['n_charging_stations']}_charging_stations_{automatic_initialization_parameters['n_ground_stations']}_ground_stations{'_'+custom_initialization_parameters['horizon'] if 'horizon' in custom_initialization_parameters else ''}_logged_drone_routing.json"
        # check if logfile exists and load from it if it does
        if "recompute_logfile" in custom_initialization_parameters and custom_initialization_parameters["recompute_logfile"]:
            self.loaded = False
        elif os.path.exists(self.logfile):
            self.loaded = True
            with open(self.logfile, "r") as log:
                data = json.load(log)
                self.current_solution = data
        else:
            self.loaded = False
            self.current_solution = []

    def get_initial_drone_locations(self):
        if self.loaded:
            return self.current_solution[0]
        else:
            initial_locations = self.strategy.get_initial_drone_locations()
            # log the result
            self.current_solution = [initial_locations]
            with open(self.logfile, "w") as log:
                json.dump(self.current_solution, log, indent=2)
            return initial_locations

    def next_actions(self, automatic_step_parameters, custom_step_parameters):
        self.call_counter += 1
        if self.loaded:
            return self.current_solution[self.call_counter]
        else:
            actions = self.strategy.next_actions(automatic_step_parameters, custom_step_parameters)
            self.current_solution.append(actions)
            with open(self.logfile, "w") as log:
                json.dump(self.current_solution, log, indent=2)
            return actions
        

class LogWrapperSensor(SensorPlacementStrategy):
    def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
        self.call_counter = 0
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters
        self.logfile = "/".join(custom_initialization_parameters["burnmap_filename"].split("/")[:-1]) + f"/{custom_initialization_parameters['strategy_sensor'].__name__}_{automatic_initialization_parameters['n_drones']}_drones_{automatic_initialization_parameters['n_charging_stations']}_charging_stations_{automatic_initialization_parameters['n_ground_stations']}_ground_stations_logged_sensor_placement.json"
        # check if logfile exists and load from it if it does
        if "recompute_logfile" in custom_initialization_parameters and custom_initialization_parameters["recompute_logfile"]:
            self.loaded = False
        elif os.path.exists(self.logfile):
            self.loaded = True
            with open(self.logfile, "r") as log:
                data = json.load(log)
                self.charging_station_locations = data["charging_station_locations"]
                self.ground_sensor_locations = data["ground_sensor_locations"]
            return
        self.loaded = False
        self.strategy = custom_initialization_parameters["strategy_sensor"](automatic_initialization_parameters, custom_initialization_parameters)
        self.charging_station_locations = []
        self.ground_sensor_locations = []

    def get_locations(self):
        if self.loaded:
            return self.charging_station_locations, self.ground_sensor_locations
        else:
            # run the strategy
            charging_station_locations, ground_sensor_locations = self.strategy.get_locations()
            # log the result
            self.charging_station_locations = charging_station_locations
            self.ground_sensor_locations = ground_sensor_locations
            with open(self.logfile, "w") as log:
                json.dump({
                    "charging_station_locations": self.charging_station_locations,
                    "ground_sensor_locations": self.ground_sensor_locations
                }, log, indent=2)
            return charging_station_locations, ground_sensor_locations

class LoggedDroneRoutingStrategy(DroneRoutingStrategy):
    """
    LoggedDroneRoutingStrategy logs drone routing actions and locations at every timestep.

    Args:
        automatic_initialization_parameters: dict
            Expected keys:
                - n_drones: Number of drones
                - N, M: Grid size
                - charging_stations_locations: list of tuples (x, y)
        custom_initialization_parameters: dict
            Expected keys:
                - burnmap_filename: Path to the burn map (not used in dummy version)
                - call_every_n_steps: Frequency to call the optimization (or dummy routing function)
                - optimization_horizon: Number of future steps to plan
                - log_file: (optional) Explicit path to save the drone routing log JSON file

    Returns:
        Initializes:
            - self.initial_drone_locations: list of tuples (x, y)
            - self.log_data: log structure with initial locations and step logs
    """
    def __init__(self, automatic_initialization_parameters, custom_initialization_parameters):
        # we don't change the strategy name here, because we want to keep the same name as the strategy we are wrapping
        # assign parameters from parent
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters

        # validate parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("Missing 'burnmap_filename' in custom_initialization_parameters")
        if "call_every_n_steps" not in custom_initialization_parameters:
            raise ValueError("Missing 'call_every_n_steps' in custom_initialization_parameters")
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("Missing 'optimization_horizon' in custom_initialization_parameters")

        # config values
        self.call_every_n_steps = custom_initialization_parameters["call_every_n_steps"]
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

        # initialize counters and memory
        self.call_counter = 0
        self.current_solution = []  # holds lists of actions between Julia calls

        # === LOG FILE SETUP (Optional log_file override) ===
        if "log_file" in custom_initialization_parameters:
            log_file_path = custom_initialization_parameters["log_file"]
            log_dir = os.path.dirname(log_file_path)
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = log_file_path
        else:
            # build log filename dynamically if log_file isn't provided
            N = self.automatic_initialization_parameters.get("N", "N")
            M = self.automatic_initialization_parameters.get("M", "M")

            n_drones = self.automatic_initialization_parameters.get("n_drones", 0)
            n_charging_stations = self.automatic_initialization_parameters.get("n_charging_stations", 0)

            log_filename = f"drone_strategy_{N}N_{M}M_{n_drones}drones_{n_charging_stations}charge.json"

            log_dir = custom_initialization_parameters.get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)

            self.log_file = os.path.join(log_dir, log_filename)

        # initialize logging structure
        self.log_data = {
            "initial_drone_locations": None,  # set in get_initial_drone_locations()
            "steps": []  # append timestep logs here
        }

        print(f"[LoggedDroneRoutingStrategy] Initialized with log file: {self.log_file}")

    def get_initial_drone_locations(self):
        charging_stations = self.automatic_initialization_parameters["charging_stations_locations"]
        n_drones = self.automatic_initialization_parameters["n_drones"]

        n_stations = len(charging_stations)
        q = n_drones // n_stations
        r = n_drones % n_stations

        initial_positions = charging_stations * q + charging_stations[:r]

        self.log_data["initial_drone_locations"] = initial_positions
        self._write_log_to_file()

        return initial_positions

    def next_actions(self, automatic_step_parameters: dict, custom_step_parameters: dict):
        """
        automatic_step_parameters: dict with keys:
            - "drone_locations": list of tuples (x,y)
            - "drone_batteries": list of tuples (distance,time)
            - "t": int
        Returns:
            actions: list of tuples (action_type, action_parameters)
        """
        

        if self.call_counter % self.call_every_n_steps == 0:
            print(f"[LoggedDroneRoutingStrategy] Calling dummy optimizer at timestep {self.call_counter}")
            _, self.current_solution = self.dummy_drone_routing_robust(
                automatic_step_parameters, custom_step_parameters
            )
            print("[LoggedDroneRoutingStrategy] Dummy optimization finished")

            # charging_stations = [tuple(x) for x in self.automatic_initialization_parameters["charging_stations_locations"]]
            # ground_stations = [tuple(x) for x in self.automatic_initialization_parameters["ground_sensor_locations"]]

            # _, self.current_solution = jl.NEW_ROUTING_STRATEGY_INIT(
            # self.custom_initialization_parameters["burnmap_filename"],
            # self.automatic_initialization_parameters["n_drones"],
            # charging_stations,
            # ground_stations,
            # self.custom_initialization_parameters["optimization_horizon"],
            # self.automatic_initialization_parameters["max_battery_time"],
            # self.custom_initialization_parameters["call_every_n_steps"]
            # )
            
        timestep_index = self.call_counter % self.call_every_n_steps
        actions = self.current_solution[timestep_index]

        # log actions and states
        self._log_timestep(
            timestep=automatic_step_parameters["t"],
            drone_locations=automatic_step_parameters["drone_locations"],
            drone_batteries=automatic_step_parameters["drone_batteries"],
            actions=actions
        )

        self.call_counter += 1
        return actions

    def dummy_drone_routing_robust(self, automatic_step_parameters, custom_step_parameters):
        print("[Dummy Function] Generating dummy routing solution...")

        n_drones = self.automatic_initialization_parameters.get("n_drones", 3)
        n_timesteps = self.optimization_horizon

        initial_locations = [(i * 5, i * 5) for i in range(n_drones)]

        actions_per_timestep = []
        for t in range(n_timesteps):
            actions = []
            for d in range(n_drones):
                if t % 2 == 0:
                    actions.append(('move', (1, 0)))
                else:
                    actions.append(('charge', None))
            actions_per_timestep.append(actions)

        return initial_locations, actions_per_timestep

    def _log_timestep(self, timestep, drone_locations, drone_batteries, actions):
        """
        Logs the state and actions at each timestep.
        """
        log_entry = {
            "timestep": timestep,
            "drone_locations": drone_locations,
            "drone_batteries": drone_batteries,
            "actions": actions
        }

        self.log_data["steps"].append(log_entry)

        # Write the log to file immediately after each timestep
        print(f"[LoggedDroneRoutingStrategy] Writing log to {self.log_file} at timestep {timestep}")
        self._write_log_to_file()

    def _write_log_to_file(self):
        """
        Writes the current log to the log_file.
        """
        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=2)

        print(f"[LoggedDroneRoutingStrategy] Log successfully written to {self.log_file}")




# ------------------ Regularized Index Routing ------------------

class DroneRoutingOptimizationModelReuseIndexRegularized(DroneRoutingStrategy):
    strategy_name = "DroneRoutingOptimizationModelReuseIndexRegularized"

    """
    Drone routing strategy that uses the model reuse approach for improved performance.
    This class is functionally equivalent to DroneRoutingOptimizationSlow but uses model
    reuse to speed up computations by preserving the optimization model between calls.
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
            "regularization_param": regularization parameter for the objective
        """
        # Assign parameters
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters
        self.call_counter = 0  # Keeping track of how many times we call the function
        self.current_solution = None
        self.routing_model = None  # Will store the reusable JuMP model

        # Validate required parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        
        if "reevaluation_step" not in custom_initialization_parameters:
            raise ValueError("reevaluation_step is not defined")
        self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("optimization_horizon is not defined")
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

        if "regularization_param" not in custom_initialization_parameters:
            raise ValueError("regularization_param is not defined")
        self.regularization_param = custom_initialization_parameters["regularization_param"]

        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        self.routing_model = jl.create_regularized_index_routing_model(
            self.custom_initialization_parameters["burnmap_filename"],
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
            self.regularization_param
        )
        
        # Solve the initial routing problem with the model
        self.current_solution = jl.solve_regularized_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            self.current_solution = jl.solve_regularized_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )

            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = min(self.call_counter, len(self.current_solution) - 1)
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        return self.current_solution[idx]



class DroneRoutingLinearMinTime(DroneRoutingStrategy):
    strategy_name = "DroneRoutingLinearMinTime"

    """
    Drone routing strategy that uses a linear programming approach to minimize the time taken by the drones to cover detect the fire in expected value.
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
        self.current_solution = None
        self.routing_model = None  # Will store the reusable JuMP model
        self.t = 0 # current timestep

        # Validate required parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        
        if "reevaluation_step" not in custom_initialization_parameters:
            raise ValueError("reevaluation_step is not defined")
        self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("optimization_horizon is not defined")
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]

        raw_burn = custom_initialization_parameters["burnmap_filename"]
        n_slices = custom_initialization_parameters.get("linear_risk_time_slices")
        tile_dir = custom_initialization_parameters.get("linear_tiled_burnmap_dir")
        self._linear_julia_burnmap = raw_burn
        if n_slices is not None:
            arr = np.load(raw_burn, mmap_mode="r")
            if arr.shape[0] == 1:
                d = int(n_slices)
                if d < 1:
                    d = 1
                tiled = np.tile(np.asarray(arr, dtype=np.float64), (d, 1, 1))
                os.makedirs(tile_dir or os.path.dirname(os.path.abspath(raw_burn)), exist_ok=True)
                out_dir = tile_dir or os.path.dirname(os.path.abspath(raw_burn))
                tiled_path = os.path.join(out_dir, f"linear_risk_tiled_{uuid.uuid4().hex}.npy")
                np.save(tiled_path, tiled)
                self._linear_julia_burnmap = tiled_path
        self._time_limit_seconds = float(custom_initialization_parameters.get("time_limit_seconds") or 0.0)

        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename (Julia): {self._linear_julia_burnmap}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        self.routing_model = jl.create_index_routing_model_linear(
            self._linear_julia_burnmap,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
            "min_cumulative_prob",
            self._time_limit_seconds,
        )
        
        # Solve the initial routing problem with the model
        self.current_solution = jl.solve_index_init_routing_linear(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        # "fly"/"charge" are grid positions → subtract 1; "move"/"stay" keep as-is
        self.current_solution = [[(code, (x-1, y-1)) if code not in ("move", "stay") else (code, (x, y))
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0

            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]

            # Solve next move with the existing model
            self.current_solution = jl.solve_index_next_move_routing_linear(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )

            # Convert to Python indexing
            self.current_solution = [[(code, (x-1, y-1)) if code not in ("move", "stay") else (code, (x, y))
                                     for code, (x, y) in plan] for plan in self.current_solution]

            #print("current solution (Python indexing)")
            #print(self.current_solution)
            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = min(self.call_counter, len(self.current_solution) - 1)
        self.t += 1
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        return self.current_solution[idx]



### final ones

# class DroneRoutingMaxCoverageResetStatic(DroneRoutingStrategy):
#     strategy_name = "DroneRoutingMaxCoverageResetStatic"
#     """
#     Drone routing strategy that uses a max coverage approach and resets the burn map at every reevaluation step.
#     """
#     def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
#         """
#         automatic_initialization_parameters: dict with keys:
#             "N": Grid height
#             "M": Grid width
#             "max_battery_distance": int
#             "max_battery_time": int
#             "n_drones": int
#             "n_ground_stations": Target number of ground stations
#             "n_charging_stations": Target number of charging stations
#             "ground_sensor_locations": list of tuples (x,y)
#             "charging_stations_locations": list of tuples (x,y)
#         custom_initialization_parameters: dict with keys:
#             "burnmap_filename": burn map file name
#             "burnamap_type": static or dynamic
#             "reevaluation_step": number of steps between calls to julia optimization model
#             "optimization_horizon": number of steps to optimize for
#         """
#         # Assign parameters
#         self.automatic_initialization_parameters = automatic_initialization_parameters
#         self.custom_initialization_parameters = custom_initialization_parameters
#         self.call_counter = 0  # Keeping track of how many times we call the function
#         self.t = 0 # current timestep
#         self.current_solution = None
#         self.routing_model = None  # Will store the reusable JuMP model
#         self.call_ID = random.randint(0, 1000000)
#         self.burnmap_type = custom_initialization_parameters.get("burnmap_type", "static")
#         # Validate required parameters
#         if "burnmap_filename" not in custom_initialization_parameters:
#             raise ValueError("burnmap_filename is not defined")
#         self.initial_burnmap = load_scenario(self.custom_initialization_parameters["burnmap_filename"])
#         self.current_burnmap = self.initial_burnmap.copy()
#         if self.burnmap_type == "static":
#             # duplicate the data to go from shape (1,N,M) to shape (100,N,M)
#             self.current_burnmap = np.tile(self.initial_burnmap, (5000, 1, 1))
#         self.len_burnmap = self.initial_burnmap.shape[0]
#         self.current_burnmap_filename = "./tmp_burnmaps/tmp_burnmap_" + str(self.call_ID) + ".npy"
#         # create the tmp_burnmaps folder if it doesn't exist
#         if not os.path.exists("./tmp_burnmaps"):
#             os.makedirs("./tmp_burnmaps")
#         self.automatic_initialization_parameters["burnmap_filename"] = self.current_burnmap_filename
        
#         if "reevaluation_step" not in custom_initialization_parameters:
#             raise ValueError("reevaluation_step is not defined")
#         self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
#         if "optimization_horizon" not in custom_initialization_parameters:
#             raise ValueError("optimization_horizon is not defined")
#         self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

       
#         self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
#         # Store original charging stations as class attribute
#         self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
#         # Convert to Julia indexing (Python 0-based → Julia 1-based)
#         self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
#         self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
#         self.execution_time = 0
#         self.saving_time = 0
        
#     def get_initial_drone_locations(self):
#         """
#         Returns the initial locations of the drones after creating the optimization model
#         and solving the initial routing problem.
#         """
#         print("Creating initial routing model (reusable)")
#         print("--- parameters for julia (Julia indexing) ---")
#         print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
#         print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
#         print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
#         print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
#         print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

#         save_burn_map(self.current_burnmap, self.current_burnmap_filename)

#         # Create the reusable routing model
#         start_time = time.time()
#         self.routing_model = jl.create_index_routing_model(
#             self.current_burnmap_filename,
#             self.automatic_initialization_parameters["n_drones"],
#             self.julia_charging_stations_locations,
#             self.julia_ground_sensor_locations,
#             self.custom_initialization_parameters["optimization_horizon"],
#             self.automatic_initialization_parameters["max_battery_time"],
#         )
#         self.execution_time += time.time() - start_time
#         # Solve the initial routing problem with the model
#         start_time = time.time()
#         self.current_solution = jl.solve_index_init_routing(
#             self.routing_model, 
#             self.custom_initialization_parameters["reevaluation_step"]
#         )
#         self.execution_time += time.time() - start_time
#         # print(f"current_solution (Julia indexing): {self.current_solution}")
        
#         # Convert to Python indexing (Julia 1-based → Python 0-based)
#         self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
#                                  for code, (x, y) in plan] for plan in self.current_solution]
        
#         # Extract initial positions from the first step of the solution
#         # Extract full action tuples from step 0
#         initial_plan = self.current_solution[0]  # list of (code, (x, y))

#         initial_positions = self.current_solution[0]
#         self.call_counter = 0
        
#         print("Initial optimization finished")
#         print(f"\nDEBUG: Available Charging Stations (after model creation): {self.charging_stations_locations}")


#         return initial_positions

        
        
#     def next_actions(self, automatic_step_parameters:dict, custom_step_parameters:dict):
#         """
#         automatic_step_parameters: dict with keys:
#             "drone_locations": list of tuples (x,y)
#             "drone_batteries": list of tuples (distance,time)
#             "drone_states": list of strings "charge" or "fly"
#             "t": int
#         custom_step_parameters: dict 
#         Returns:
#             actions: list of tuples (action_type, action_parameters)
#         """
#         # Every reevaluation_step calls, recompute the solution using the existing model
#         if self.call_counter == self.reevaluation_step-1:
#             self.call_counter = 0
#             # save the current burnmap
#             start_time = time.time()
#             save_burn_map(self.current_burnmap, self.current_burnmap_filename)
#             self.saving_time += time.time() - start_time
#             # print("Solving next move with model reuse (integer indexing)")
            
#             # Convert drone locations to Julia indexing
#             julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
#             # print("--- parameters for julia (Julia indexing) ---")
#             # print(f"drone_locations: {julia_drone_locations}")
#             # print(f"drone_states: {automatic_step_parameters['drone_states']}")
#             # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
#             # print("--- end of parameters ---")

#             # Solve next move with the existing model
#             start_time = time.time()
#             self.current_solution = jl.solve_index_next_move_routing(
#                 self.routing_model,
#                 self.custom_initialization_parameters["reevaluation_step"],
#                 julia_drone_locations,
#                 automatic_step_parameters["drone_states"],
#                 automatic_step_parameters["drone_batteries"],
#                 self.t
#             )
#             self.execution_time += time.time() - start_time
#             #print("Next move optimization finished")
#             # print("current solution (Julia indexing)")
#             # print(self.current_solution)

#             # Convert to Python indexing
#             self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
#                                      for code, (x, y) in plan] for plan in self.current_solution]
#             # uopdate the burnmap

            


#         # Return the appropriate step from the pre-computed plan
#         self.call_counter += 1
#         idx = min(self.call_counter, len(self.current_solution) - 1)
#         # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
#         # update the burnmap: set every visited cell to 0
#         for action in self.current_solution[idx]:
#             if action[0] == "fly":
#                 #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
#                 self.current_burnmap[self.t:self.t+self.reset_time,action[1][0], action[1][1]] = 0
#                 #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
#         self.t += 1
#         return self.current_solution[idx]







##



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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        save_burn_map(self.current_burnmap, self.current_burnmap_filename)

        # Create the reusable routing model
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        self.execution_time += time.time() - start_time
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = min(self.call_counter, len(self.current_solution) - 1)
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
                self.current_burnmap[self.t:self.t+self.reset_time,action[1][0], action[1][1]] = 0
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        self.t += 1
        return self.current_solution[idx]




######











##### NEW FORMULATION

class NewDroneRoutingOptimizationModelReuseIndex(DroneRoutingStrategy):
    strategy_name = "NewDroneRoutingOptimizationModelReuseIndex"

    """
    Drone routing strategy that uses the model reuse approach for improved performance.
    This class is functionally equivalent to DroneRoutingOptimizationSlow but uses model
    reuse to speed up computations by preserving the optimization model between calls.
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
        self.current_solution = None
        self.routing_model = None  # Will store the reusable JuMP model

        # Validate required parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        
        if "reevaluation_step" not in custom_initialization_parameters:
            raise ValueError("reevaluation_step is not defined")
        self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("optimization_horizon is not defined")
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        self.routing_model = jl.new_create_index_routing_model(
            self.custom_initialization_parameters["burnmap_filename"],
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"]
        )
        
        # Solve the initial routing problem with the model
        self.current_solution = jl.new_solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            #print("THE PB IS HERE : ", automatic_step_parameters["drone_locations"])
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            self.current_solution = jl.new_solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )

            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            

        if self.current_solution is None:
            raise RuntimeError("Julia optimization did not return a solution.")
        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = min(self.call_counter, len(self.current_solution) - 1)
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        return self.current_solution[idx]










class DroneRoutingMaxCoverageReset(DroneRoutingStrategy):
    strategy_name = "DroneRoutingMaxCoverageReset"
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
            # Tile to cover max simulation time of 24 data timesteps (hours) × substeps per hour
            self.current_burnmap = np.tile(self.initial_burnmap, (24 * self.automatic_initialization_parameters["max_battery_time"], 1, 1))
        

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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        self.execution_time += time.time() - start_time
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        return self.current_solution[idx]



class DroneRoutingExhaustiveSearch(DroneRoutingStrategy):
    strategy_name = "DroneRoutingExhaustiveSearch"
    """
    Drone routing strategy that uses a exhaustive search approach.
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
        

        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """

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
            "drone_states": list of strings "charge" or "fly"
            "t": int
        custom_step_parameters: dict 
        Returns:
            actions: list of tuples (action_type, action_parameters)
        """
        self.t += 1
        if self.current_solution is None:
            self.current_solution = []
            # TODO
        
        self.t += 1
        return self.current_solution[t]




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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        save_burn_map(self.current_burnmap, self.current_burnmap_filename)

        # Create the reusable routing model
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        self.execution_time += time.time() - start_time
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        #assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        return self.current_solution[idx]



class DroneRoutingMaxCoverageResetStaticGreedy(DroneRoutingMaxCoverageResetStatic):
    strategy_name = "DroneRoutingMaxCoverageResetStaticGreedy"





class DroneRoutingMaxCoverageResetUniform(DroneRoutingStrategy):
    strategy_name = "DroneRoutingMaxCoverageResetUniform"
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
        self.current_burnmap = np.zeros((6000, self.initial_burnmap.shape[1], self.initial_burnmap.shape[2]))

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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        self.execution_time += time.time() - start_time
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        #assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        return self.current_solution[idx]



class DroneRoutingMaxCoverageResetStaticGreedy(DroneRoutingMaxCoverageResetStatic):
    strategy_name = "DroneRoutingMaxCoverageResetStaticGreedy"





class DroneRoutingMaxCoverageResetUniform(DroneRoutingStrategy):
    strategy_name = "DroneRoutingMaxCoverageResetUniform"
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
        self.current_burnmap = np.zeros((6000, self.initial_burnmap.shape[1], self.initial_burnmap.shape[2]))

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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        self.execution_time += time.time() - start_time
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        return self.current_solution[idx]


####
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import numpy as np

def count_paths_convolution(N, M, n):
    """
    Uniform coverage kernel for coordinated drone patrol.

    With battery B and patrol range R = B // 2, the reachable zone is
    (2R+1) x (2R+1) = B x B cells.  Each drone visits B distinct cells
    per charge cycle, so B drones saturate the full zone.  Weight per
    cell = 1/B for all reachable cells.

    Parameters
    ----------
    N, M : int   Grid height / width (unused, kept for API compatibility).
    n    : int   Patrol range (max_battery_time // 2).

    Returns
    -------
    mapping : dict  (dx, dy) -> float
    """
    B = 2 * n + 1       # full battery = zone side length
    w = 1.0 / B
    mapping = {}
    for dx in range(-n, n + 1):
        for dy in range(-n, n + 1):
            mapping[(dx, dy)] = w
    return mapping


def count_hitting_probability_kernel(max_steps: int):
    """
    Return a translation-invariant kernel mapping (dx, dy) -> probability that a
    random walk starting at the origin visits the target offset within
    `max_steps` steps.

    The walk uses the same 9 moves as the legacy convolution kernel:
    8-neighborhood plus staying in place, all equiprobable.
    """
    size = 2 * max_steps + 1
    origin = max_steps
    offsets = [(dx, dy) for dx in range(-max_steps, max_steps + 1)
               for dy in range(-max_steps, max_steps + 1)]
    moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    move_prob = 1.0 / len(moves)
    mapping = {}

    for target_dx, target_dy in offsets:
        target_x = origin + target_dx
        target_y = origin + target_dy
        if target_x == origin and target_y == origin:
            mapping[(target_dx, target_dy)] = 1.0
            continue

        # DP over paths that have NOT hit the target yet.
        alive = np.zeros((size, size), dtype=np.float64)
        alive[origin, origin] = 1.0
        hit_prob = 0.0

        for _ in range(max_steps):
            nxt = np.zeros_like(alive)
            for x in range(size):
                for y in range(size):
                    p = alive[x, y]
                    if p == 0.0:
                        continue
                    for mx, my in moves:
                        nx = x + mx
                        ny = y + my
                        if 0 <= nx < size and 0 <= ny < size:
                            if nx == target_x and ny == target_y:
                                hit_prob += p * move_prob
                            else:
                                nxt[nx, ny] += p * move_prob
            alive = nxt

        mapping[(target_dx, target_dy)] = min(max(hit_prob, 0.0), 1.0)

    return mapping

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


        max_battery = automatic_initialization_parameters["max_battery_time"]
        patrol_range = max_battery // 2
        kernel = count_paths_convolution(N, M, patrol_range)
        kernel_size_x = patrol_range
        kernel_size_y = patrol_range

        x_vars, y_vars = jl.Max_Coverage_Kernel(custom_initialization_parameters["burnmap_filename"], automatic_initialization_parameters["n_ground_stations"], automatic_initialization_parameters["n_charging_stations"], automatic_initialization_parameters["n_drones"], kernel, kernel_size_x, kernel_size_y, custom_initialization_parameters["mask_filename"])
        # print("optimization finished")
        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)

    def get_drone_allocation(self):
        """
        Returns the number of drones allocated to each charging station.
        
        Returns:
            list[int]: List of integers, where each element corresponds to the number of drones
                      allocated to the charging station at the same index in charging_station_locations.
                      The list matches the order of self.charging_station_locations.
        """
        return self.drones_per_charging_station


class SensorPlacementMaxCoverageGaussianTimeMasked(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTimeMasked"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using Julia's optimization model with mask-aware coverage.
        
        Args:
            automatic_initialization_parameters: dict with keys:
                "n_ground_stations": Target number of ground stations
                "n_charging_stations": Target number of charging stations
                "n_drones": Target number of drones
                "N": Grid height
                "M": Grid width
            custom_initialization_parameters: dict with keys:
                "burnmap_filename": burn map file name
                "mask_filename": mask file name (cells with mask=0 are blocked)
                "recompute_kernel": If True, compute per-location kernels using masked DP (slower but more accurate)
                                    If False, use fixed kernel approximation (faster but less accurate near mask boundaries)
                "n_steps": Number of DP steps for kernel computation (default max_battery_time // 2, only used if recompute_kernel=True)
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")

        mask_filename = custom_initialization_parameters.get("mask_filename", None)
        recompute_kernel = custom_initialization_parameters.get("recompute_kernel", False)
        
        max_battery = automatic_initialization_parameters["max_battery_time"]
        patrol_range = max_battery // 2
        n_steps = custom_initialization_parameters.get("n_steps", patrol_range)

        burnmap = load_scenario(custom_initialization_parameters["burnmap_filename"])
        T, N, M = burnmap.shape

        kernel = count_paths_convolution(N, M, patrol_range)
        kernel_size_x = patrol_range
        kernel_size_y = patrol_range

        print(f"SensorPlacementMaxCoverageGaussianTimeMasked: recompute_kernel={recompute_kernel}, n_steps={n_steps}")

        # Call the Julia optimization function with masked kernel support
        x_vars, y_vars = jl.Max_Coverage_Kernel_Masked(
            custom_initialization_parameters["burnmap_filename"], 
            automatic_initialization_parameters["n_ground_stations"], 
            automatic_initialization_parameters["n_charging_stations"], 
            automatic_initialization_parameters["n_drones"], 
            kernel, 
            kernel_size_x, 
            kernel_size_y, 
            mask_filename,
            recompute_kernel,
            n_steps
        )
        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)


class SensorPlacementMaxCoverageGaussianTimeWithAllocation(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTimeWithAllocation"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using Julia's optimization model with drone allocation.
        
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


        max_battery = automatic_initialization_parameters["max_battery_time"]
        patrol_range = max_battery // 2
        kernel = count_paths_convolution(N, M, patrol_range)
        kernel_size_x = patrol_range
        kernel_size_y = patrol_range

        x_vars, y_vars, drone_allocations = jl.Max_Coverage_Kernel_WithAllocation(custom_initialization_parameters["burnmap_filename"], automatic_initialization_parameters["n_ground_stations"], automatic_initialization_parameters["n_charging_stations"], automatic_initialization_parameters["n_drones"], kernel, kernel_size_x, kernel_size_y, custom_initialization_parameters["mask_filename"])
        # print("optimization finished")
        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)
        self.drones_per_charging_station = list(drone_allocations)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)
        print("drones per charging station")
        print(self.drones_per_charging_station)

    def get_drone_allocation(self):
        """
        Returns the number of drones allocated to each charging station.
        
        Returns:
            list[int]: List of integers, where each element corresponds to the number of drones
                      allocated to the charging station at the same index in charging_station_locations.
                      The list matches the order of self.charging_station_locations.
        """
        return self.drones_per_charging_station


class SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Initialize the ground placement strategy using Julia's optimization model with mask-aware coverage and drone allocation.
        
        Args:
            automatic_initialization_parameters: dict with keys:
                "n_ground_stations": Target number of ground stations
                "n_charging_stations": Target number of charging stations
                "n_drones": Target number of drones
                "N": Grid height
                "M": Grid width
            custom_initialization_parameters: dict with keys:
                "burnmap_filename": burn map file name
                "mask_filename": mask file name (cells with mask=0 are blocked)
                "recompute_kernel": If True, compute per-location kernels using masked DP (slower but more accurate)
                                    If False, use fixed kernel approximation (faster but less accurate near mask boundaries)
                "n_steps": Number of DP steps for kernel computation (default max_battery_time // 2, only used if recompute_kernel=True)
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")

        mask_filename = custom_initialization_parameters.get("mask_filename", None)
        recompute_kernel = custom_initialization_parameters.get("recompute_kernel", False)
        
        max_battery = automatic_initialization_parameters["max_battery_time"]
        patrol_range = max_battery // 2
        n_steps = custom_initialization_parameters.get("n_steps", patrol_range)

        burnmap = load_scenario(custom_initialization_parameters["burnmap_filename"])
        T, N, M = burnmap.shape

        kernel = count_paths_convolution(N, M, patrol_range)
        kernel_size_x = patrol_range
        kernel_size_y = patrol_range

        print(f"SensorPlacementMaxCoverageGaussianTimeMaskedWithAllocation: recompute_kernel={recompute_kernel}, n_steps={n_steps}")

        # Call the Julia optimization function with masked kernel support and allocation
        time_limit = custom_initialization_parameters.get("time_limit_seconds", 600.0)
        x_vars, y_vars, drone_allocations = jl.Max_Coverage_Kernel_Masked_WithAllocation(
            custom_initialization_parameters["burnmap_filename"],
            automatic_initialization_parameters["n_ground_stations"],
            automatic_initialization_parameters["n_charging_stations"],
            automatic_initialization_parameters["n_drones"],
            kernel,
            kernel_size_x,
            kernel_size_y,
            mask_filename,
            recompute_kernel,
            n_steps,
            time_limit,
        )
        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)
        self.drones_per_charging_station = list(drone_allocations)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)
        print("drones per charging station")
        print(self.drones_per_charging_station)

    def get_drone_allocation(self):
        """
        Returns the number of drones allocated to each charging station.
        
        Returns:
            list[int]: List of integers, where each element corresponds to the number of drones
                      allocated to the charging station at the same index in charging_station_locations.
                      The list matches the order of self.charging_station_locations.
        """
        return self.drones_per_charging_station


class SensorPlacementMaxCoverageGaussianTimeMaskedBudget(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTimeMaskedBudget"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Budget-constrained sensor and drone placement.  The optimiser determines
        how many ground sensors, charging stations and drones to deploy given a
        total budget, then places and allocates them to maximise coverage.

        Args:
            automatic_initialization_parameters: dict with keys:
                "max_battery_time": drone battery time (for kernel computation)
                "N": Grid height (operational)
                "M": Grid width  (operational)
            custom_initialization_parameters: dict with keys:
                "burnmap_filename": burn map file name
                "mask_filename":    mask file name (cells with mask=0 are blocked)
                "budget_millions":  total budget in millions of dollars
                "cost_sensor":      cost per ground sensor  in millions (default 0.1  = $100k)
                "cost_station":     cost per charging station in millions (default 0.15 = $150k)
                "cost_drone":       cost per drone in millions (default 0.05 = $50k)
                "recompute_kernel": bool (default False)
                "n_steps":          int  (default max_battery_time // 2)
                "time_limit_seconds": float (default 600)
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        if "budget_millions" not in custom_initialization_parameters:
            raise ValueError("budget_millions is not defined")

        mask_filename    = custom_initialization_parameters.get("mask_filename", None)
        recompute_kernel = custom_initialization_parameters.get("recompute_kernel", False)
        max_battery      = automatic_initialization_parameters["max_battery_time"]
        patrol_range     = max_battery // 2
        n_steps          = custom_initialization_parameters.get("n_steps", patrol_range)

        budget_millions  = custom_initialization_parameters["budget_millions"]
        cost_sensor      = custom_initialization_parameters.get("cost_sensor", 0.1)
        cost_station     = custom_initialization_parameters.get("cost_station", 0.15)
        cost_drone       = custom_initialization_parameters.get("cost_drone", 0.05)

        burnmap = load_scenario(custom_initialization_parameters["burnmap_filename"])
        T, N, M = burnmap.shape

        kernel = count_paths_convolution(N, M, patrol_range)
        kernel_size_x = patrol_range
        kernel_size_y = patrol_range

        print(f"SensorPlacementMaxCoverageGaussianTimeMaskedBudget: "
              f"budget={budget_millions}M, costs=sensor:{cost_sensor}M station:{cost_station}M drone:{cost_drone}M, "
              f"recompute_kernel={recompute_kernel}, n_steps={n_steps}")

        time_limit = custom_initialization_parameters.get("time_limit_seconds", 600.0)
        x_vars, y_vars, drone_allocations = jl.Max_Coverage_Kernel_Masked_Budget(
            custom_initialization_parameters["burnmap_filename"],
            budget_millions,
            cost_sensor,
            cost_station,
            cost_drone,
            kernel,
            kernel_size_x,
            kernel_size_y,
            mask_filename,
            recompute_kernel,
            n_steps,
            time_limit,
        )

        self.ground_sensor_locations    = list(x_vars)
        self.charging_station_locations = list(y_vars)
        self.drones_per_charging_station = [int(x) for x in drone_allocations]

        self.n_ground_sensors   = len(self.ground_sensor_locations)
        self.n_charging_stations = len(self.charging_station_locations)
        self.n_drones            = int(sum(self.drones_per_charging_station)) if self.drones_per_charging_station else 0
        self.budget_millions     = budget_millions

        budget_used = (self.n_ground_sensors * cost_sensor
                       + self.n_charging_stations * cost_station
                       + self.n_drones * cost_drone)
        print(f"Budget allocation: {self.n_ground_sensors} sensors, "
              f"{self.n_charging_stations} stations, {self.n_drones} drones")
        print(f"Budget used: {budget_used:.2f}M / {budget_millions}M")
        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)
        print("drones per charging station")
        print(self.drones_per_charging_station)

    def get_drone_allocation(self):
        """Returns the number of drones allocated to each charging station."""
        return self.drones_per_charging_station

    def get_device_counts(self):
        """Returns the number of each device type chosen by the optimiser."""
        return {
            "n_ground_sensors": self.n_ground_sensors,
            "n_charging_stations": self.n_charging_stations,
            "n_drones": self.n_drones,
        }


class SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMax(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMax"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Budget-constrained placement variant where each cell only receives drone
        coverage from its best charging station. Same-station multi-drone
        contributions are still precomputed as capped constants.
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        if "budget_millions" not in custom_initialization_parameters:
            raise ValueError("budget_millions is not defined")

        mask_filename = custom_initialization_parameters.get("mask_filename", None)
        max_battery = automatic_initialization_parameters["max_battery_time"]
        one_way_reach = max_battery // 2
        n_steps = custom_initialization_parameters.get("n_steps", one_way_reach)
        recompute_kernel = False
        if custom_initialization_parameters.get("recompute_kernel", False):
            print("Warning: recompute_kernel is ignored for StationMax; using calibrated open-grid hitting kernel.", flush=True)

        budget_millions = custom_initialization_parameters["budget_millions"]
        cost_sensor = custom_initialization_parameters.get("cost_sensor", 0.1)
        cost_station = custom_initialization_parameters.get("cost_station", 0.15)
        cost_drone = custom_initialization_parameters.get("cost_drone", 0.05)

        burnmap = load_scenario(custom_initialization_parameters["burnmap_filename"])
        T, N, M = burnmap.shape

        kernel = count_hitting_probability_kernel(one_way_reach)
        kernel_size_x = max_battery
        kernel_size_y = max_battery

        max_drones_per_station = custom_initialization_parameters.get(
            "max_drones_per_station", max_battery
        )
        candidate_percentile = custom_initialization_parameters.get(
            "candidate_percentile",
            0.50 if abs(budget_millions - 100.0) < 1e-9 else 0.80,
        )
        print(f"SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMax: "
              f"budget={budget_millions}M, costs=sensor:{cost_sensor}M station:{cost_station}M drone:{cost_drone}M, "
              f"one_way_reach={one_way_reach}, max_drones_per_station={max_drones_per_station}, "
              f"candidate_percentile={candidate_percentile}, "
              f"recompute_kernel={recompute_kernel}, n_steps={n_steps}")
        time_limit = custom_initialization_parameters.get("time_limit_seconds", 600.0)
        x_vars, y_vars, drone_allocations = jl.Max_Coverage_Kernel_Masked_Budget_StationMax(
            custom_initialization_parameters["burnmap_filename"],
            budget_millions,
            cost_sensor,
            cost_station,
            cost_drone,
            kernel,
            kernel_size_x,
            kernel_size_y,
            mask_filename,
            recompute_kernel,
            n_steps,
            time_limit,
            max_drones_per_station,
            candidate_percentile,
        )

        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)
        self.drones_per_charging_station = [int(x) for x in drone_allocations]

        self.n_ground_sensors = len(self.ground_sensor_locations)
        self.n_charging_stations = len(self.charging_station_locations)
        self.n_drones = int(sum(self.drones_per_charging_station)) if self.drones_per_charging_station else 0
        self.budget_millions = budget_millions

        budget_used = (self.n_ground_sensors * cost_sensor
                       + self.n_charging_stations * cost_station
                       + self.n_drones * cost_drone)
        print(f"Budget allocation: {self.n_ground_sensors} sensors, "
              f"{self.n_charging_stations} stations, {self.n_drones} drones")
        print(f"Budget used: {budget_used:.2f}M / {budget_millions}M")
        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)
        print("drones per charging station")
        print(self.drones_per_charging_station)

    def get_drone_allocation(self):
        """Returns the number of drones allocated to each charging station."""
        return self.drones_per_charging_station

    def get_device_counts(self):
        """Returns the number of each device type chosen by the optimiser."""
        return {
            "n_ground_sensors": self.n_ground_sensors,
            "n_charging_stations": self.n_charging_stations,
            "n_drones": self.n_drones,
        }


class SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniform(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniform"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Budget-constrained StationMax placement where different stations do not
        add coverage, but same-station multi-drone coverage follows a mask-aware
        uniform kernel over the station's reachable feasible zone.
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        if "budget_millions" not in custom_initialization_parameters:
            raise ValueError("budget_millions is not defined")

        mask_filename = custom_initialization_parameters.get("mask_filename", None)
        max_battery = automatic_initialization_parameters["max_battery_time"]
        one_way_reach = max_battery // 2
        n_steps = custom_initialization_parameters.get("n_steps", one_way_reach)
        recompute_kernel = False
        if custom_initialization_parameters.get("recompute_kernel", False):
            print("Warning: recompute_kernel is ignored for StationMaxUniform; using mask-aware uniform station kernels.", flush=True)

        budget_millions = custom_initialization_parameters["budget_millions"]
        cost_sensor = custom_initialization_parameters.get("cost_sensor", 0.1)
        cost_station = custom_initialization_parameters.get("cost_station", 0.15)
        cost_drone = custom_initialization_parameters.get("cost_drone", 0.05)

        burnmap = load_scenario(custom_initialization_parameters["burnmap_filename"])
        T, N, M = burnmap.shape

        kernel = count_paths_convolution(N, M, one_way_reach)
        kernel_size_x = max_battery
        kernel_size_y = max_battery

        max_drones_per_station = custom_initialization_parameters.get(
            "max_drones_per_station", max_battery
        )
        candidate_percentile = custom_initialization_parameters.get(
            "candidate_percentile",
            0.50 if abs(budget_millions - 100.0) < 1e-9 else 0.80,
        )
        print(f"SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniform: "
              f"budget={budget_millions}M, costs=sensor:{cost_sensor}M station:{cost_station}M drone:{cost_drone}M, "
              f"one_way_reach={one_way_reach}, max_drones_per_station={max_drones_per_station}, "
              f"candidate_percentile={candidate_percentile}, "
              f"recompute_kernel={recompute_kernel}, n_steps={n_steps}")
        time_limit = custom_initialization_parameters.get("time_limit_seconds", 600.0)
        x_vars, y_vars, drone_allocations = jl.Max_Coverage_Kernel_Masked_Budget_StationMax_Uniform(
            custom_initialization_parameters["burnmap_filename"],
            budget_millions,
            cost_sensor,
            cost_station,
            cost_drone,
            kernel,
            kernel_size_x,
            kernel_size_y,
            mask_filename,
            recompute_kernel,
            n_steps,
            time_limit,
            max_drones_per_station,
            candidate_percentile,
        )

        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)
        self.drones_per_charging_station = [int(x) for x in drone_allocations]

        self.n_ground_sensors = len(self.ground_sensor_locations)
        self.n_charging_stations = len(self.charging_station_locations)
        self.n_drones = int(sum(self.drones_per_charging_station)) if self.drones_per_charging_station else 0
        self.budget_millions = budget_millions

        budget_used = (self.n_ground_sensors * cost_sensor
                       + self.n_charging_stations * cost_station
                       + self.n_drones * cost_drone)
        print(f"Budget allocation: {self.n_ground_sensors} sensors, "
              f"{self.n_charging_stations} stations, {self.n_drones} drones")
        print(f"Budget used: {budget_used:.2f}M / {budget_millions}M")
        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)
        print("drones per charging station")
        print(self.drones_per_charging_station)

    def get_drone_allocation(self):
        """Returns the number of drones allocated to each charging station."""
        return self.drones_per_charging_station

    def get_device_counts(self):
        """Returns the number of each device type chosen by the optimiser."""
        return {
            "n_ground_sensors": self.n_ground_sensors,
            "n_charging_stations": self.n_charging_stations,
            "n_drones": self.n_drones,
        }


class SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxGreedyUniform(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxGreedyUniform"
    def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
        """
        Budget-constrained StationMax placement with greedy-uniform coverage.

        The greedy set-cover heuristic determines what fraction of a station's
        zone risk k drones can cover.  That aggregate fraction is then applied
        uniformly to every reachable cell in the zone, instead of being limited
        to the specific greedy paths.

        Args:
            automatic_initialization_parameters: dict with keys:
                "max_battery_time": drone battery time (for kernel / reach)
                "N": Grid height
                "M": Grid width
            custom_initialization_parameters: dict with keys:
                "burnmap_filename": burn map file name
                "mask_filename":    mask file name (cells with mask=0 are blocked)
                "budget_millions":  total budget in millions of dollars
                "cost_sensor":      cost per ground sensor  in millions (default 0.1)
                "cost_station":     cost per charging station in millions (default 0.15)
                "cost_drone":       cost per drone in millions (default 0.05)
                "max_drones_per_station": int (default max_battery_time)
                "candidate_percentile":  float (default 0.50 for 100M, 0.80 otherwise)
                "n_steps":          int  (default max_battery_time // 2)
                "time_limit_seconds": float (default 600)
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        if "budget_millions" not in custom_initialization_parameters:
            raise ValueError("budget_millions is not defined")

        mask_filename = custom_initialization_parameters.get("mask_filename", None)
        max_battery = automatic_initialization_parameters["max_battery_time"]
        one_way_reach = max_battery // 2
        n_steps = custom_initialization_parameters.get("n_steps", one_way_reach)

        budget_millions = custom_initialization_parameters["budget_millions"]
        cost_sensor = custom_initialization_parameters.get("cost_sensor", 0.1)
        cost_station = custom_initialization_parameters.get("cost_station", 0.15)
        cost_drone = custom_initialization_parameters.get("cost_drone", 0.05)

        burnmap = load_scenario(custom_initialization_parameters["burnmap_filename"])
        T, N, M = burnmap.shape

        kernel = count_hitting_probability_kernel(one_way_reach)
        kernel_size_x = max_battery
        kernel_size_y = max_battery

        max_drones_per_station = custom_initialization_parameters.get(
            "max_drones_per_station", max_battery
        )
        candidate_percentile = custom_initialization_parameters.get(
            "candidate_percentile",
            0.50 if abs(budget_millions - 100.0) < 1e-9 else 0.80,
        )
        print(f"SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxGreedyUniform: "
              f"budget={budget_millions}M, costs=sensor:{cost_sensor}M station:{cost_station}M drone:{cost_drone}M, "
              f"one_way_reach={one_way_reach}, max_drones_per_station={max_drones_per_station}, "
              f"candidate_percentile={candidate_percentile}")
        time_limit = custom_initialization_parameters.get("time_limit_seconds", 600.0)
        warm_start_file = custom_initialization_parameters.get("warm_start_file", "")
        fixed_drones = int(custom_initialization_parameters.get("fixed_drones_per_station", 0) or 0)
        x_vars, y_vars, drone_allocations = jl.Max_Coverage_Kernel_Masked_Budget_StationMax_GreedyUniform(
            custom_initialization_parameters["burnmap_filename"],
            budget_millions,
            cost_sensor,
            cost_station,
            cost_drone,
            kernel,
            kernel_size_x,
            kernel_size_y,
            mask_filename,
            False,
            n_steps,
            time_limit,
            max_drones_per_station,
            candidate_percentile,
            warm_start_file,
            fixed_drones,
        )

        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)
        self.drones_per_charging_station = [int(x) for x in drone_allocations]

        self.n_ground_sensors = len(self.ground_sensor_locations)
        self.n_charging_stations = len(self.charging_station_locations)
        self.n_drones = int(sum(self.drones_per_charging_station)) if self.drones_per_charging_station else 0
        self.budget_millions = budget_millions

        budget_used = (self.n_ground_sensors * cost_sensor
                       + self.n_charging_stations * cost_station
                       + self.n_drones * cost_drone)
        print(f"Budget allocation: {self.n_ground_sensors} sensors, "
              f"{self.n_charging_stations} stations, {self.n_drones} drones")
        print(f"Budget used: {budget_used:.2f}M / {budget_millions}M")
        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)
        print("drones per charging station")
        print(self.drones_per_charging_station)

    def get_drone_allocation(self):
        """Returns the number of drones allocated to each charging station."""
        return self.drones_per_charging_station

    def get_device_counts(self):
        """Returns the number of each device type chosen by the optimiser."""
        return {
            "n_ground_sensors": self.n_ground_sensors,
            "n_charging_stations": self.n_charging_stations,
            "n_drones": self.n_drones,
        }


class SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniformFixedDrones(SensorPlacementStrategy):
    strategy_name = "SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniformFixedDrones"
    def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
        """
        Budget StationMax placement with exactly ``fixed_drones_per_station`` drones
        on every open charging station (default 7).  Drone coverage per assigned
        cell is 1 (no fractional uniform kernel / no w variables).

        ``warm_start_file`` may point to a greedy-uniform ``sensor_alloc_*.json``
        (e.g. 100M run): sets MIP start values for ground sensors and stations.
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        if "budget_millions" not in custom_initialization_parameters:
            raise ValueError("budget_millions is not defined")

        mask_filename = custom_initialization_parameters.get("mask_filename", None)
        max_battery = automatic_initialization_parameters["max_battery_time"]
        one_way_reach = max_battery // 2
        n_steps = custom_initialization_parameters.get("n_steps", one_way_reach)

        budget_millions = custom_initialization_parameters["budget_millions"]
        cost_sensor = custom_initialization_parameters.get("cost_sensor", 0.1)
        cost_station = custom_initialization_parameters.get("cost_station", 0.15)
        cost_drone = custom_initialization_parameters.get("cost_drone", 0.05)

        burnmap = load_scenario(custom_initialization_parameters["burnmap_filename"])
        T, N, M = burnmap.shape

        kernel = count_paths_convolution(N, M, one_way_reach)
        kernel_size_x = max_battery
        kernel_size_y = max_battery

        fixed_drones = int(custom_initialization_parameters.get("fixed_drones_per_station", 7) or 7)
        candidate_percentile = custom_initialization_parameters.get(
            "candidate_percentile",
            0.50 if abs(budget_millions - 100.0) < 1e-9 else 0.80,
        )
        print(
            f"SensorPlacementMaxCoverageGaussianTimeMaskedBudgetStationMaxUniformFixedDrones: "
            f"budget={budget_millions}M, costs=sensor:{cost_sensor}M station:{cost_station}M drone:{cost_drone}M, "
            f"fixed_drones_per_station={fixed_drones}, candidate_percentile={candidate_percentile}",
            flush=True,
        )
        time_limit = custom_initialization_parameters.get("time_limit_seconds", 600.0)
        warm_start_file = custom_initialization_parameters.get("warm_start_file", "")
        budget_regularization_epsilon = float(
            custom_initialization_parameters.get("budget_regularization_epsilon", -1.0)
        )

        x_vars, y_vars, drone_allocations = jl.Max_Coverage_Kernel_Masked_Budget_StationMax_UniformFixedDrones(
            custom_initialization_parameters["burnmap_filename"],
            budget_millions,
            cost_sensor,
            cost_station,
            cost_drone,
            kernel,
            kernel_size_x,
            kernel_size_y,
            mask_filename,
            False,
            n_steps,
            time_limit,
            fixed_drones,
            candidate_percentile,
            warm_start_file,
            budget_regularization_epsilon,
        )

        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)
        self.drones_per_charging_station = [int(x) for x in drone_allocations]

        self.n_ground_sensors = len(self.ground_sensor_locations)
        self.n_charging_stations = len(self.charging_station_locations)
        self.n_drones = int(sum(self.drones_per_charging_station)) if self.drones_per_charging_station else 0
        self.budget_millions = budget_millions

        budget_used = (
            self.n_ground_sensors * cost_sensor
            + self.n_charging_stations * cost_station
            + self.n_drones * cost_drone
        )
        print(
            f"Budget allocation: {self.n_ground_sensors} sensors, "
            f"{self.n_charging_stations} stations, {self.n_drones} drones",
            flush=True,
        )
        print(f"Budget used: {budget_used:.2f}M / {budget_millions}M", flush=True)
        print("ground sensor locations", flush=True)
        print(self.ground_sensor_locations, flush=True)
        print("charging station locations", flush=True)
        print(self.charging_station_locations, flush=True)
        print("drones per charging station", flush=True)
        print(self.drones_per_charging_station, flush=True)

    def get_drone_allocation(self):
        """Returns the number of drones allocated to each charging station (fixed count per open station)."""
        return self.drones_per_charging_station

    def get_device_counts(self):
        """Returns the number of each device type chosen by the optimiser."""
        return {
            "n_ground_sensors": self.n_ground_sensors,
            "n_charging_stations": self.n_charging_stations,
            "n_drones": self.n_drones,
        }















######################################################################################## Growing 



class DroneRoutingMaxCoverageGrowingStatic(DroneRoutingStrategy):
    strategy_name = "DroneRoutingMaxCoverageGrowingStatic"
    """
    Drone routing strategy that uses a max coverage approach and grows the burn map at every reevaluation step.
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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        print("solving initial routing problem")
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        print("initial routing problem solved")
        print(f"current_solution (Julia indexing): {self.current_solution}")
        self.execution_time += time.time() - start_time
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        # if t is a multiple of the data time resolution, we update the whole burn map
        if self.t % self.data_time_resolution == 0:
            # Check if we have enough time steps in the initial burnmap # THIS IS A PB!! IF THE INIT IS STATIC!! TODO
            if self.t < self.initial_burnmap.shape[0]:
                self.current_burnmap[self.t:] += self.initial_burnmap[self.t]

        return self.current_solution[idx]



class DroneRoutingUniformCoverageGrowingStatic(DroneRoutingStrategy):
    strategy_name = "DroneRoutingUniformCoverageGrowingStatic"
    """
    Drone routing strategy that uses a uniform coverage approach and grows the burn map at every reevaluation step.
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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        # Create the reusable routing model
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        self.execution_time += time.time() - start_time
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        # if t is a multiple of the data time resolution, we update the whole burn map
        if self.t % self.data_time_resolution == 0:
            self.current_burnmap[self.t:] += self.initial_burnmap[self.t]


        return self.current_solution[idx]






##### TOP STRATEGIES #####

class DroneRoutingTOP(DroneRoutingStrategy):
    strategy_name = "DroneRoutingTOP"
    burnmap_handeling_type = "fixed_reset"
    """
    Drone routing strategy that uses a Team Orienteering Problem (TOP) approach.
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
        self.use_linf_cost = custom_initialization_parameters.get("use_linf_cost", False)
    
        # Validate required parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        self.initial_burnmap = load_scenario(self.custom_initialization_parameters["burnmap_filename"])
        self.current_burnmap = self.initial_burnmap.copy()
        if self.burnmap_type == "static":
            # duplicate the data to go from shape (1,N,M) to shape (100,N,M)
            self.current_burnmap = np.tile(self.current_burnmap, (200, 1, 1))
            self.initial_burnmap = np.tile(self.initial_burnmap, (200, 1, 1))
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

        # reevaluation step is equal to the max battery time
        self.reevaluation_step = self.automatic_initialization_parameters["max_battery_time"]

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        self.reset_time_periods = self.reset_time // self.reevaluation_step
        print(f"reset_time_periods: {self.reset_time_periods}")
        self.data_time_resolution = automatic_initialization_parameters.get("data_time_resolution", 1)
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial TOP plan via Julia CPA solver")
        print("--- parameters for Julia (1-based indexing) ---")
        print(f"burnmap_filename: {self.current_burnmap_filename}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"use_linf_cost: {self.use_linf_cost}")

        start_time = time.time()
        self.current_solution = jl.compute_TOP_plan_multiple_depots(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.automatic_initialization_parameters["max_battery_time"],
            0, # t = 0 for the initial plan
            False,  # verbose=False to disable Julia plots
            [], # initial_drone_positions
            None, # mask_filename (no mask for base DroneRoutingTOP)
            self.use_linf_cost
        )
        #print(f"current solution: {self.current_solution}")
        self.execution_time += time.time() - start_time
        print(f"execution time for initial plan: {self.execution_time}")
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0

        initial_positions = self.current_solution[0]
        self.call_counter = 0
        
        print("Initial optimisation finished")
        print(f"initial_positions: {initial_positions}")
        #print(f"current solution: {self.current_solution}")

        print(f"\nDEBUG: Available Charging Stations (after model creation): {self.charging_stations_locations}")


        return initial_positions#, self.current_burnmap_filename
    

        
        
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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            #julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            print("current drone locations in julia indexing are:", julia_drone_locations)
            # if drone are not on charging stations, we raise an error # TODO, put this in julia instead of here
            for drone_location in julia_drone_locations:
                if drone_location not in self.julia_charging_stations_locations:
                    raise ValueError(f"Drone is not on a charging station: {drone_location}")
            
            start_time = time.time()
            self.current_solution = jl.compute_TOP_plan_multiple_depots(
                self.current_burnmap_filename,
                self.automatic_initialization_parameters["n_drones"],
                self.julia_charging_stations_locations,
                self.julia_ground_sensor_locations,
                self.automatic_initialization_parameters["max_battery_time"],
                self.t,
                False,  # verbose=False to disable Julia plots
                julia_drone_locations,
                None, # mask_filename (no mask for base DroneRoutingTOP)
                self.use_linf_cost
            )
            print(f"execution time for next move: {time.time() - start_time}")
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}. reevaluation_step={self.reevaluation_step}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        _, bm_N, bm_M = self.current_burnmap.shape
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                ax, ay = action[1][0], action[1][1]
                # Skip out-of-bounds cells (Julia solver may route beyond grid edges)
                if ax < 0 or ax >= bm_N or ay < 0 or ay >= bm_M:
                    continue
                # What we do here with the burn map depends on `burnmap_handeling_type`. If it is "fixed_reset", we reset the burn map to 0 for the next reset_time steps. If it is "growing", we set it to 0 forever and add the initial burnmap to the current burnmap.
                if self.burnmap_handeling_type == "fixed_reset":
                    # in the case of TOP, we don't reset for the next reset_time steps, but rather reset for the time left until the next reevaluation (as the only time steps of the burn map actually used are the ones on the re-optimization times)
                    time_left_until_next_reevaluation = self.reevaluation_step - self.t % self.reevaluation_step
                    self.current_burnmap[self.t:min(self.t+time_left_until_next_reevaluation + (self.reset_time_periods-1)*self.reevaluation_step,self.current_burnmap.shape[0]), ax, ay] = 0
                elif self.burnmap_handeling_type == "growing" or self.burnmap_handeling_type == "growing_proba":
                    self.current_burnmap[self.t:, ax, ay] = 0
                else:
                    raise ValueError(f"Invalid burnmap_handeling_type: {self.burnmap_handeling_type}")
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        # if t is a multiple of the data time resolution, we update the whole burn map
        if self.t % self.data_time_resolution == 0:
            if self.burnmap_handeling_type == "growing":
                self.current_burnmap[self.t:] += self.initial_burnmap[self.t] #TODO adapt to dynamic map
            elif self.burnmap_handeling_type == "growing_proba": # this is assuming independence though
                self.current_burnmap[self.t:] = 1 - (1 - self.current_burnmap[self.t]) * (1 - self.initial_burnmap[self.t])
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)


        return self.current_solution[idx]

class DroneRoutingTOPwarm(DroneRoutingTOP):
    strategy_name = "DroneRoutingTOPwarm"
    
class DroneRoutingTOPGrowing(DroneRoutingTOP):
    strategy_name = "DroneRoutingTOPGrowing"
    burnmap_handeling_type = "growing"

class DroneRoutingTOPGrowingProba(DroneRoutingTOP):
    strategy_name = "DroneRoutingTOPGrowingProba"
    burnmap_handeling_type = "growing_proba"


class DroneRoutingTOPMasked(DroneRoutingStrategy):
    strategy_name = "DroneRoutingTOPMasked"
    burnmap_handeling_type = "fixed_reset"
    """
    Drone routing strategy that uses a Team Orienteering Problem (TOP) approach with mask support.
    Blocked cells (where mask == 0) are avoided during routing.
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
            "mask_filename": mask file name (optional, points where mask == 0 are blocked)
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
        self.mask_filename = custom_initialization_parameters.get("mask_filename", None)
        self.call_ID = random.randint(0, 1000000)
        self.burnmap_type = custom_initialization_parameters.get("burnmap_type", "static")
        self.use_linf_cost = custom_initialization_parameters.get("use_linf_cost", False)
    
        # Validate required parameters
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        self.initial_burnmap = load_scenario(self.custom_initialization_parameters["burnmap_filename"])
        self.current_burnmap = self.initial_burnmap.copy()
        if self.burnmap_type == "static":
            # duplicate the data to go from shape (1,N,M) to shape (100,N,M)
            self.current_burnmap = np.tile(self.current_burnmap, (200, 1, 1))
            self.initial_burnmap = np.tile(self.initial_burnmap, (200, 1, 1))
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

        # reevaluation step is equal to the max battery time
        self.reevaluation_step = self.automatic_initialization_parameters["max_battery_time"]

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        self.reset_time_periods = self.reset_time // self.reevaluation_step
        print(f"reset_time_periods: {self.reset_time_periods}")
        self.data_time_resolution = automatic_initialization_parameters.get("data_time_resolution", 1)
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial TOP plan via Julia CPA solver (with mask support)")
        print("--- parameters for Julia (1-based indexing) ---")
        print(f"burnmap_filename: {self.current_burnmap_filename}")
        print(f"mask_filename: {self.mask_filename}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"use_linf_cost: {self.use_linf_cost}")

        start_time = time.time()
        self.current_solution = jl.compute_TOP_plan_multiple_depots(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.automatic_initialization_parameters["max_battery_time"],
            0, # t = 0 for the initial plan
            False,  # verbose=False to disable Julia plots
            [], # initial_drone_positions
            self.mask_filename,  # Pass mask filename to Julia
            self.use_linf_cost
        )
        #print(f"current solution: {self.current_solution}")
        self.execution_time += time.time() - start_time
        print(f"execution time for initial plan: {self.execution_time}")
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0

        initial_positions = self.current_solution[0]
        self.call_counter = 0
        
        print("Initial optimisation finished")
        print(f"initial_positions: {initial_positions}")
        #print(f"current solution: {self.current_solution}")

        print(f"\nDEBUG: Available Charging Stations (after model creation): {self.charging_stations_locations}")


        return initial_positions#, self.current_burnmap_filename
    

        
        
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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            #julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            print("current drone locations in julia indexing are:", julia_drone_locations)
            # if drone are not on charging stations, we raise an error # TODO, put this in julia instead of here
            for drone_location in julia_drone_locations:
                if drone_location not in self.julia_charging_stations_locations:
                    raise ValueError(f"Drone is not on a charging station: {drone_location}")
            
            start_time = time.time()
            self.current_solution = jl.compute_TOP_plan_multiple_depots(
                self.current_burnmap_filename,
                self.automatic_initialization_parameters["n_drones"],
                self.julia_charging_stations_locations,
                self.julia_ground_sensor_locations,
                self.automatic_initialization_parameters["max_battery_time"],
                self.t,
                False,  # verbose=False to disable Julia plots
                julia_drone_locations,
                self.mask_filename,  # Pass mask filename to Julia
                self.use_linf_cost
            )
            print(f"execution time for next move: {time.time() - start_time}")
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}. reevaluation_step={self.reevaluation_step}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        _, bm_N, bm_M = self.current_burnmap.shape
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                ax, ay = action[1][0], action[1][1]
                # Skip out-of-bounds cells (Julia solver may route beyond grid edges)
                if ax < 0 or ax >= bm_N or ay < 0 or ay >= bm_M:
                    continue
                # What we do here with the burn map depends on `burnmap_handeling_type`. If it is "fixed_reset", we reset the burn map to 0 for the next reset_time steps. If it is "growing", we set it to 0 forever and add the initial burnmap to the current burnmap.
                if self.burnmap_handeling_type == "fixed_reset":
                    # in the case of TOP, we don't reset for the next reset_time steps, but rather reset for the time left until the next reevaluation (as the only time steps of the burn map actually used are the ones on the re-optimization times)
                    time_left_until_next_reevaluation = self.reevaluation_step - self.t % self.reevaluation_step
                    self.current_burnmap[self.t:min(self.t+time_left_until_next_reevaluation + (self.reset_time_periods-1)*self.reevaluation_step,self.current_burnmap.shape[0]), ax, ay] = 0
                elif self.burnmap_handeling_type == "growing" or self.burnmap_handeling_type == "growing_proba":
                    self.current_burnmap[self.t:, ax, ay] = 0
                else:
                    raise ValueError(f"Invalid burnmap_handeling_type: {self.burnmap_handeling_type}")
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        # if t is a multiple of the data time resolution, we update the whole burn map
        if self.t % self.data_time_resolution == 0:
            if self.burnmap_handeling_type == "growing":
                self.current_burnmap[self.t:] += self.initial_burnmap[self.t] #TODO adapt to dynamic map
            elif self.burnmap_handeling_type == "growing_proba": # this is assuming independence though
                self.current_burnmap[self.t:] = 1 - (1 - self.current_burnmap[self.t]) * (1 - self.initial_burnmap[self.t])
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)


        return self.current_solution[idx]


class DroneRoutingTOPMaskedGrowingProba(DroneRoutingTOPMasked):
    strategy_name = "DroneRoutingTOPMaskedGrowingProba"
    burnmap_handeling_type = "growing_proba"
    """
    Drone routing strategy that combines:
    - Mask support (blocked cells avoided during routing)
    - Probabilistic growing burn map (fire risk accumulates probabilistically over time)
    """


class DroneRoutingMaxCoverageGrowingMasked(DroneRoutingStrategy):
    strategy_name = "DroneRoutingMaxCoverageGrowingMasked"
    """
    Drone routing strategy that uses a max coverage approach with mask support and grows the burn map at every reevaluation step.
    Combines mask support (from DroneRoutingMaxCoverageResetStaticMasked) with growing burn map (from DroneRoutingMaxCoverageGrowingStatic).
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
            "mask_filename": mask file name (optional, points where mask == 0 are blocked)
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
        self.mask_filename = custom_initialization_parameters.get("mask_filename", None)
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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        self.data_time_resolution = automatic_initialization_parameters.get("data_time_resolution", 1)
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable, masked, growing)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"mask_filename: {self.mask_filename}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        save_burn_map(self.current_burnmap, self.current_burnmap_filename)

        # Create the reusable routing model (with mask support)
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model_masked(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
            self.mask_filename,
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        print("solving initial routing problem")
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        print("initial routing problem solved")
        print(f"current_solution (Julia indexing): {self.current_solution}")
        self.execution_time += time.time() - start_time
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]

        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}"
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
        # if t is a multiple of the data time resolution, we update the whole burn map (GROWING)
        if self.t % self.data_time_resolution == 0:
            # Check if we have enough time steps in the initial burnmap
            if self.t < self.initial_burnmap.shape[0]:
                self.current_burnmap[self.t:] += self.initial_burnmap[self.t]

        return self.current_solution[idx]


class TestStrategy(RandomDroneRoutingStrategy):
    strategy_name = "TestStrategy"


#### Startegies with mask


class DroneRoutingMaxCoverageResetStaticMasked(DroneRoutingStrategy):
    strategy_name = "DroneRoutingMaxCoverageResetStaticMasked"
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
        self.mask_filename = custom_initialization_parameters.get("mask_filename", None)
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

       
        self.reset_time = custom_initialization_parameters.get("reset_time", 2 * self.automatic_initialization_parameters["max_battery_time"])
        
        # Store original charging stations as class attribute
        self.charging_stations_locations = automatic_initialization_parameters["charging_stations_locations"]
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        self.execution_time = 0
        self.saving_time = 0
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        print("Creating initial routing model (reusable)")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")

        save_burn_map(self.current_burnmap, self.current_burnmap_filename)

        # Create the reusable routing model
        start_time = time.time()
        self.routing_model = jl.create_index_routing_model_masked(
            self.current_burnmap_filename,
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
            self.mask_filename,
        )
        self.execution_time += time.time() - start_time
        # Solve the initial routing problem with the model
        start_time = time.time()
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        self.execution_time += time.time() - start_time
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        # Extract full action tuples from step 0
        initial_plan = self.current_solution[0]  # list of (code, (x, y))

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
        # Every reevaluation_step calls, recompute the solution using the existing model
        if self.call_counter == self.reevaluation_step-1:
            self.call_counter = 0
            # save the current burnmap
            start_time = time.time()
            save_burn_map(self.current_burnmap, self.current_burnmap_filename)
            self.saving_time += time.time() - start_time
            # print("Solving next move with model reuse (integer indexing)")
            
            # Convert drone locations to Julia indexing
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
            # print("--- parameters for julia (Julia indexing) ---")
            # print(f"drone_locations: {julia_drone_locations}")
            # print(f"drone_states: {automatic_step_parameters['drone_states']}")
            # print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            # print("--- end of parameters ---")

            # Solve next move with the existing model
            start_time = time.time()
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"],
                self.t
            )
            self.execution_time += time.time() - start_time
            #print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            # uopdate the burnmap

            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        idx = self.call_counter
        #assert idx < len(self.current_solution), f"idx={idx} is greater than the number of steps in the solution={len(self.current_solution)}"
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        # update the burnmap: set every visited cell to 0
        for action in self.current_solution[idx]:
            if action[0] == "fly":
                #print(f"setting burnmap at {action[1]} to 0 at time {self.t}")
                self.current_burnmap[self.t:min(self.t+self.reset_time,self.current_burnmap.shape[0]),action[1][0], action[1][1]] = 0
                #save_burn_map(self.current_burnmap, self.current_burnmap_filename)
        return self.current_solution[idx]
