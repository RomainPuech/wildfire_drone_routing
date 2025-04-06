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
            "drone_batteries": list of tuples (distance,time)
            "t": int
        custom_step_parameters: dict
        Returns:
            actions: list of tuples (action_type, action_parameters)
        """
        moving_plan = []
        for i, (x,y) in enumerate(automatic_step_parameters["drone_locations"]):
            if automatic_step_parameters["drone_batteries"][i][1] == 0:
                moving_plan.append(('charge',(x,y)))
            else:
                # find the closest charging station in chebyshev distance
                closest_charging_station = min(self.automatic_initialization_parameters["charging_stations_locations"], key=lambda c: max(abs(x-c[0]),abs(y-c[1])))
                closest_distance = max(abs(x-closest_charging_station[0]),abs(y-closest_charging_station[1]))
                # if current distance to the charging station is equal to the remaiing battery time, move to the charging station
                if closest_distance == automatic_step_parameters["drone_batteries"][i][1]:
                    moving_plan.append(('move',(self.sign(closest_charging_station[0]-x),self.sign(closest_charging_station[1]-y))))
                    # otherwise, move randomly
                else:
                    moving_plan.append(('move',(random.randint(-1,1),random.randint(-1,1))))
        return moving_plan
        
    

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
            layout_name = custom_initialization_parameters.get("layout_name", "layout")

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
            # print(f"[LoggedDroneRoutingStrategy] Calling dummy optimizer at timestep {self.call_counter}")
            # _, self.current_solution = self.dummy_drone_routing_robust(
            #     automatic_step_parameters, custom_step_parameters
            # )
            # print("[LoggedDroneRoutingStrategy] Dummy optimization finished")

            charging_stations = [tuple(x) for x in self.automatic_initialization_parameters["charging_stations_locations"]]
            ground_stations = [tuple(x) for x in self.automatic_initialization_parameters["ground_sensor_locations"]]

            _, self.current_solution = jl.NEW_ROUTING_STRATEGY_INIT(
            self.custom_initialization_parameters["burnmap_filename"],
            self.automatic_initialization_parameters["n_drones"],
            charging_stations,
            ground_stations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"],
            self.custom_initialization_parameters["call_every_n_steps"]
            )
            
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