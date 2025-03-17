import random
import os
from my_julia_caller import jl # DEACTIVATE IT TO RUN THINGS IN PARALLEL 
import json
import numpy as np


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
        # Generate random positions using list comprehensions
        # YOUR CODE HERE
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
    """
    #def __init__(self,ground_sensor_locations,charging_stations_locations,n_drones=None, max_battery_distance = 100, max_battery_time = 100):
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
        raise NotImplementedError("DroneRoutingStrategy is an abstract class and should not be instantiated directly.")
        # assign parameters
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters

        # Any intial computations
        # YOUR CODE HERE

    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones
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
            actions: list of tuples (action_type, action_parameters)
        """
        # suggest actions
        raise NotImplementedError("next_actions is an abstract method and should be implemented by subclasses.")


#### RANDOM STRATEGIES ####

class RandomSensorPlacementStrategy(SensorPlacementStrategy):
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
        print("RandomSensorPlacementStrategy")
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

class RandomDroneRoutingStrategy(DroneRoutingStrategy):
    """
    Drone routing strategy that moves drones randomly.
    """

    #def __init__(self,ground_sensor_locations,charging_stations_locations,n_drones=None, max_battery_distance = 100, max_battery_time = 100):
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
        return [('move',(random.randint(-5,5),random.randint(-5,5))) for _ in range(self.automatic_initialization_parameters["n_drones"])]
    

#### STRATEGIES CALLING JULIA OPTIMIZATION MODELS ####
class SensorPlacementOptimization(SensorPlacementStrategy):
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
        x_vars, y_vars = jl.ground_charging_opt_model_grid(custom_initialization_parameters["burnmap_filename"], automatic_initialization_parameters["n_ground_stations"], automatic_initialization_parameters["n_charging_stations"])
        print("optimization finished")

        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)



class DroneRoutingOptimizationSlow(DroneRoutingStrategy):
    # remember to use julia indexing!
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
        # assign parameters
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters
        self.call_counter = 0 # keeping track of how many time we call function to know when to call julia
        self.current_solution = None


        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("burnmap_filename is not defined")
        
        if "reevaluation_step" not in custom_initialization_parameters:
            raise ValueError("reevaluation_step is not defined")
        self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("optimization_horizon is not defined")
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones
        """
        # Uniform allocation of drones across charging stations (you can change this)
        
        # n = len(self.automatic_initialization_parameters["charging_stations_locations"])
        # q = self.automatic_initialization_parameters["n_drones"] // n
        # r = self.automatic_initialization_parameters["n_drones"] % n
        
        # # By default drones are spread uniformly aross charging stations
        # return self.automatic_initialization_parameters["charging_stations_locations"]*q + self.automatic_initialization_parameters["charging_stations_locations"][:r]
        
        
        # use julia indexing 
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
            
        print("calling julia optimization model")
        print("--- parameters for julia (Julia indexing) ---")
        print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        print(f"charging_stations_locations: {self.automatic_initialization_parameters['charging_stations_locations']}")
        print(f"ground_sensor_locations: {self.automatic_initialization_parameters['ground_sensor_locations']}")
        print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")


            # REPLACE HERE BY YOUR JULIA FUNCTION
        self.current_solution = jl.NEW_ROUTING_STRATEGY_INIT_INTEGER_BATTERY(self.custom_initialization_parameters["burnmap_filename"], self.automatic_initialization_parameters["n_drones"], self.julia_charging_stations_locations, self.julia_ground_sensor_locations, self.custom_initialization_parameters["optimization_horizon"], self.automatic_initialization_parameters["max_battery_time"], self.custom_initialization_parameters["reevaluation_step"])
        initial_positions_only = type(self.current_solution) == tuple # otherwise it is a list
        print(f"initial_positions_only (Julia indexing): {initial_positions_only}")
        print(f"current_solution (Julia indexing): {self.current_solution}")
        # convert to python indexing
        if initial_positions_only:
            initial_positions = [(x-1, y-1) for x, y in self.current_solution]
            self.current_solution = []
        else:
            # we got a full plan
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) for code, (x, y) in plan] for plan in self.current_solution]
            initial_positions = [(x,y) for code, (x, y) in self.current_solution[0]]
            self.call_counter = 1
        print("initial optimization finished")
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
        # suggest actions
        if self.call_counter % self.reevaluation_step == 0: # every `reevaluation_step` calls, we call julia optimization model again
            # calling julia optimization model
            print("calling julia optimization model")
            # REPLACE HERE BY YOUR JULIA FUNCTION
            # go from 0 index to 1 index in automatic_step_parameters["drone_locations"]
            julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            print("--- parameters for julia (Julia indexing) ---")
            print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
            print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
            print(f"drone_locations: {julia_drone_locations}")
            print(f"drone_states: {automatic_step_parameters['drone_states']}")
            print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
            print("--- end of parameters ---")

            self.current_solution = jl.NEW_ROUTING_STRATEGY_NEXTMOVE_INTEGER_BATTERY(self.custom_initialization_parameters["burnmap_filename"], self.automatic_initialization_parameters["n_drones"], self.julia_charging_stations_locations, self.julia_ground_sensor_locations, self.custom_initialization_parameters["optimization_horizon"], self.automatic_initialization_parameters["max_battery_time"], self.custom_initialization_parameters["reevaluation_step"], julia_drone_locations, automatic_step_parameters["drone_states"], automatic_step_parameters["drone_batteries"])
            # convert to python indexing
            print("optimization finished")
            print("current solution (Julia indexing)")
            print(self.current_solution)
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) for code, (x, y) in plan] for plan in self.current_solution]

        self.call_counter += 1
        return self.current_solution[(self.call_counter - 1) % self.reevaluation_step]
           

class DroneRoutingOptimizationModelReuse(DroneRoutingStrategy):
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
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        # print("Creating initial routing model (reusable)")
        # print("--- parameters for julia (Julia indexing) ---")
        # print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        # print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        # print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        # print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        # print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")
        
        # Create the reusable routing model
        self.routing_model = jl.create_routing_model(
            self.custom_initialization_parameters["burnmap_filename"],
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"]
        )
        
        # Solve the initial routing problem with the model
        self.current_solution = jl.solve_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        
        # print(f"current_solution (Julia indexing): {self.current_solution}")
        
        # Convert to Python indexing (Julia 1-based → Python 0-based)
        self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                 for code, (x, y) in plan] for plan in self.current_solution]
        
        # Extract initial positions from the first step of the solution
        initial_positions = self.current_solution[0]
        self.call_counter = 0
        
        print("Initial optimization finished")
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
            self.current_solution = jl.solve_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"]
            )

            print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        return self.current_solution[self.call_counter]

# class DroneRoutingOptimizationModelReuseIndex(DroneRoutingStrategy):
#     """
#     Drone routing strategy that uses the model reuse approach with integer indexing.
#     This class combines both model reuse and integer-based indexing for maximum performance.
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
#             "reevaluation_step": number of steps between calls to julia optimization model
#             "optimization_horizon": number of steps to optimize for
#         """
#         # Assign parameters
#         self.automatic_initialization_parameters = automatic_initialization_parameters
#         self.custom_initialization_parameters = custom_initialization_parameters
#         self.call_counter = 0  # Keeping track of how many times we call the function
#         self.current_solution = None
#         self.routing_model = None  # Will store the reusable JuMP model

#         # Validate required parameters
#         if "burnmap_filename" not in custom_initialization_parameters:
#             raise ValueError("burnmap_filename is not defined")
        
#         if "reevaluation_step" not in custom_initialization_parameters:
#             raise ValueError("reevaluation_step is not defined")
#         self.reevaluation_step = custom_initialization_parameters["reevaluation_step"]
        
#         if "optimization_horizon" not in custom_initialization_parameters:
#             raise ValueError("optimization_horizon is not defined")
#         self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]
        
#         # Convert to Julia indexing (Python 0-based → Julia 1-based)
#         self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
#         self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        
#     def get_initial_drone_locations(self):
#         """
#         Returns the initial locations of the drones after creating the optimization model
#         and solving the initial routing problem using integer indexing.
#         """
#         print("Creating initial routing model with integer indexing (reusable)")
#         print("--- parameters for julia (Julia indexing) ---")
#         print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
#         print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
#         print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
#         print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
#         print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")
        
#         # Create the reusable routing model using integer indexing
#         self.routing_model = jl.create_index_routing_model(
#             self.custom_initialization_parameters["burnmap_filename"],
#             self.automatic_initialization_parameters["n_drones"],
#             self.julia_charging_stations_locations,
#             self.julia_ground_sensor_locations,
#             self.custom_initialization_parameters["optimization_horizon"],
#             self.automatic_initialization_parameters["max_battery_time"]
#         )
        
#         # Solve the initial routing problem with the model
#         self.current_solution = jl.solve_index_init_routing(
#             self.routing_model, 
#             self.custom_initialization_parameters["reevaluation_step"]
#         )
        
#         print(f"current_solution (Julia indexing): {self.current_solution}")
        
#         # Convert to Python indexing (Julia 1-based → Python 0-based)
#         self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
#                                  for code, (x, y) in plan] for plan in self.current_solution]
        
#         # Extract initial positions from the first step of the solution
#         initial_positions = [(x,y) for code, (x, y) in self.current_solution[0]]
#         self.call_counter = 1
        
#         print("Initial optimization finished")
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
#         if self.call_counter % self.reevaluation_step == 0:
#             print("Solving next move with integer-indexed model reuse")
            
#             # Convert drone locations to Julia indexing
#             julia_drone_locations = [(x+1, y+1) for x, y in automatic_step_parameters["drone_locations"]]
            
#             print("--- parameters for julia (Julia indexing) ---")
#             print(f"drone_locations: {julia_drone_locations}")
#             print(f"drone_states: {automatic_step_parameters['drone_states']}")
#             print(f"drone_batteries: {automatic_step_parameters['drone_batteries']}")
#             print("--- end of parameters ---")

#             # Solve next move with the existing model using integer indexing
#             self.current_solution = jl.solve_index_next_move_routing(
#                 self.routing_model,
#                 self.custom_initialization_parameters["reevaluation_step"],
#                 julia_drone_locations,
#                 automatic_step_parameters["drone_states"],
#                 automatic_step_parameters["drone_batteries"]
#             )

#             print("Next move optimization finished")
#             print("current solution (Julia indexing)")
#             print(self.current_solution)

#             # Convert to Python indexing
#             self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
#                                      for code, (x, y) in plan] for plan in self.current_solution]
            
            

#         # Return the appropriate step from the pre-computed plan
#         self.call_counter += 1
#         return self.current_solution[(self.call_counter - 1) % self.reevaluation_step]

class DroneRoutingOptimizationModelReuseIndex(DroneRoutingStrategy):
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
        
        # Convert to Julia indexing (Python 0-based → Julia 1-based)
        self.julia_charging_stations_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["charging_stations_locations"]]
        self.julia_ground_sensor_locations = [(x+1, y+1) for x, y in self.automatic_initialization_parameters["ground_sensor_locations"]]
        
    def get_initial_drone_locations(self):
        """
        Returns the initial locations of the drones after creating the optimization model
        and solving the initial routing problem.
        """
        # print("Creating initial routing model (reusable)")
        # print("--- parameters for julia (Julia indexing) ---")
        # print(f"burnmap_filename: {self.custom_initialization_parameters['burnmap_filename']}")
        # print(f"n_drones: {self.automatic_initialization_parameters['n_drones']}")
        # print(f"charging_stations_locations: {self.julia_charging_stations_locations}")
        # print(f"ground_sensor_locations: {self.julia_ground_sensor_locations}")
        # print(f"optimization_horizon: {self.custom_initialization_parameters['optimization_horizon']}")
        
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
        initial_positions = self.current_solution[0]
        self.call_counter = 0
        
        print("Initial optimization finished")
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
            self.current_solution = jl.solve_index_next_move_routing(
                self.routing_model,
                self.custom_initialization_parameters["reevaluation_step"],
                julia_drone_locations,
                automatic_step_parameters["drone_states"],
                automatic_step_parameters["drone_batteries"]
            )

            print("Next move optimization finished")
            # print("current solution (Julia indexing)")
            # print(self.current_solution)

            # Convert to Python indexing
            self.current_solution = [[(code,(x-1, y-1)) if code != "move" else (code, (x, y)) 
                                     for code, (x, y) in plan] for plan in self.current_solution]
            


        # Return the appropriate step from the pre-computed plan
        self.call_counter += 1
        return self.current_solution[self.call_counter]

#### HEURISTIC STRATEGIES ####
class GREEDY_DRONE_STRATEGY(DroneRoutingStrategy):
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
                        - logfile: Path to the log file
                        - load_from_logfile: boolean if we want to load the logfile or recompute and save a new one
                        - burnmap_filename: Path to the burn map used by the Julia optimizer

            Returns:
                Initializes:
                    self.ground_sensor_locations: list of tuples (x, y)
                    self.charging_station_locations: list of tuples (x, y)
            """
            
            # Ensure required custom params exist
            if "logfile" not in custom_initialization_parameters:
                raise ValueError("custom_initialization_parameters must include 'logfile'")
            if "burnmap_filename" not in custom_initialization_parameters:
                raise ValueError("custom_initialization_parameters must include 'burnmap_filename'")

            
            # Extract the layout name from custom params (if available)
            layout_name = custom_initialization_parameters.get("layout_name", "layout")

            # Get n_ground_stations
            n_ground_stations = automatic_initialization_parameters.get("n_ground_stations", 0)

            # Get strategy name
            strategy_name = self.__class__.__name__

            # Build the log directory
            log_dir = os.path.dirname(custom_initialization_parameters["logfile"])
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
                
                # Run Julia optimization function
                x_vars, y_vars = jl.ground_charging_opt_model_grid(
                    burnmap_filename,
                    automatic_initialization_parameters["n_ground_stations"],
                    automatic_initialization_parameters["n_charging_stations"]
                )

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

class LoggedDroneRoutingStrategy(DroneRoutingStrategy):
    def __init__(self, automatic_initialization_parameters, custom_initialization_parameters):
        # check for logfile
        # if it exists:
        #   load initial locations and actions
        # else:
        #   run optimization, save to logfile
        
        # Initialize parameters from parent class (if applicable)
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters

        # Ensure required custom params exist            
        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("custom_initialization_parameters must include 'burnmap_filename'")
        if "logfile" not in custom_initialization_parameters:
            # extract layout name from burnmap filename
            layout_name = custom_initialization_parameters["burnmap_filename"].split("/")[-1].split(".")[0]
            custom_initialization_parameters["layout_name"] = layout_name
            print(f"[LoggedDroneRoutingStrategy] Layout name: {layout_name}")
            raise ValueError("logfile is not defined")

        # === Extract dynamic params ===
        layout_name = custom_initialization_parameters.get("layout_name", "layout")
        n_drones = automatic_initialization_parameters.get("n_drones", 0)
        strategy_name = self.__class__.__name__

        # make them filename safe
        safe_layout_name = re.sub(r'\W+', '_', layout_name)
        safe_strategy_name = re.sub(r'\W+', '_', strategy_name)

        # build log directory
        log_dir = os.path.dirname(custom_initialization_parameters["logfile"])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        
        # build logfile name
        logfile = os.path.join(
            log_dir,
            f"{safe_layout_name}_{safe_strategy_name}_{n_drones}_drones.json"
        )

        print(f"[LoggedDroneRoutingStrategy] Using log file: {logfile}")

        burnmap_filename = custom_initialization_parameters["burnmap_filename"]

        self.initial_drone_locations = []
        self.actions_per_timestep = []

        # Check if the log file already exists
        if os.path.exists(logfile):
            print(f"[LoggedDroneRoutingStrategy] Loading drone routing from log file: {logfile}")

            with open(logfile, "r") as log:
                data = json.load(log)

                # Expect JSON structure with initial locations + actions
                self.initial_drone_locations = data["initial_drone_locations"]
                self.actions_per_timestep = data["actions_per_timestep"]

        else:
            print(f"[LoggedDroneRoutingStrategy] Log file not found at {logfile}. Running optimization...")

            # Call Julia optimizer or your drone routing model
            # You can replace `drone_routing_opt_model` with your specific function
            initial_locations, actions = jl.drone_routing_opt_model(
                burnmap_filename,
                automatic_initialization_parameters["ground_sensor_locations"],
                automatic_initialization_parameters["charging_stations_locations"],
                automatic_initialization_parameters["n_drones"],
                automatic_initialization_parameters["N"],
                automatic_initialization_parameters["M"]
                # add more if needed
            )

            # Store the results
            self.initial_drone_locations = list(initial_locations)
            self.actions_per_timestep = list(actions)

            # Save the routing plan to the logfile
            with open(logfile, "w") as log:
                json.dump({
                    "initial_drone_locations": self.initial_drone_locations,
                    "actions_per_timestep": self.actions_per_timestep
                }, log, indent=2)

            print(f"[LoggedDroneRoutingStrategy] Optimization complete. Routing saved to {logfile}")

            # print(f"[LoggedDroneRoutingStrategy] Log file not found at {logfile}. Running dummy optimization...")

            # # MOCK: replace Julia optimization with dummy initial locations and actions
            # n_drones = automatic_initialization_parameters["n_drones"]
            # charging_stations = automatic_initialization_parameters["charging_stations_locations"]

            # # dummy: place all drones on the first charging station (or spread them if you want)
            # if len(charging_stations) > 0:
            #     initial_locations = [charging_stations[0] for _ in range(n_drones)]
            # else:
            #     initial_locations = [(0, 0) for _ in range(n_drones)]

            # # dummy: create simple actions per timestep (just hover in place)
            # # structure: actions_per_timestep[timestep][drone_index] = action
            # # actions should be lists of lists of tuples ('move'/'recharge', (dx, dy)/None)
            # n_timesteps = 10  # arbitrary number of timesteps for testing
            # actions_per_timestep = [
            #     [('move', (0, 0)) for _ in range(n_drones)]  # each drone does nothing at each timestep
            #     for _ in range(n_timesteps)
            # ]

            # # store results
            # self.initial_drone_locations = list(initial_locations)
            # self.actions_per_timestep = list(actions_per_timestep)

            # # write dummy data to logfile so it loads next time
            # with open(logfile, "w") as log:
            #     json.dump({
            #         "initial_drone_locations": self.initial_drone_locations,
            #         "actions_per_timestep": self.actions_per_timestep
            #     }, log, indent=2)

            # print(f"[LoggedDroneRoutingStrategy] Dummy optimization complete. Routing saved to {logfile}")

    def get_initial_drone_locations(self):
        # return loaded or computed initial locations
        pass
    def next_actions(self, automatic_step_parameters, custom_step_parameters):
        # return the next precomputed actions for this timestep

        # going to call the last timestep drone orientation and then run next actions
        pass


#### TEMPLATES FOR NEW STRATEGIES ####

