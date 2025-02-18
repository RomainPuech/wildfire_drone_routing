import random
import os
# from my_julia_caller import jl # DEACTIVATED TO RUN THINGS IN PARALLEL 
import json
import numpy as np


def get_automatic_layout_parameters(scenario:np.ndarray):
    return {
        "N": scenario.shape[0],
        "M": scenario.shape[1],
        "max_battery_distance": 100,
        "max_battery_time": 100,
        "n_drones": 10,
        "n_ground_stations": 10,
        "n_charging_stations": 10
    }

def get_burnmap_dir(scenario:np.ndarray, filename:str):
    return {
        "risk_pertime_dir": filename
    }

class DroneRoutingStrategy():

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
        
        n = len(self.automatic_parameters["charging_stations_locations"])
        q = self.user_parameters["n_drones"] // n
        r = self.user_parameters["n_drones"] % n
        
        # By default drones are spread uniformly aross charging stations
        return self.automatic_parameters["charging_stations_locations"]*q + self.automatic_parameters["charging_stations_locations"][:r]

    
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
        return [('move',(random.randint(-5,5),random.randint(-5,5))) for _ in range(self.n_drones)]
    

class SensorPlacementStrategy():
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
        self.ground_sensor_locations = [(random.randint(0, automatic_initialization_parameters["M"]-1), 
                                       random.randint(0, automatic_initialization_parameters["N"]-1)) 
                                      for _ in range(automatic_initialization_parameters["n_ground_stations"])]
        
        self.charging_station_locations = [(random.randint(0, automatic_initialization_parameters["M"]-1), 
                                          random.randint(0, automatic_initialization_parameters["N"]-1)) 
                                         for _ in range(automatic_initialization_parameters["n_charging_stations"])]

    def get_locations(self):
        """
        Returns the locations of the ground sensors and charging stations
        """
        # Do not overwrite this function
        return self.ground_sensor_locations, self.charging_station_locations

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
                "risk_pertime_dir": Directory containing burn map files
        """
        # Initialize empty lists (skip parent's random initialization)
        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if not custom_initialization_parameters["risk_pertime_dir"].endswith("/"):
            custom_initialization_parameters["risk_pertime_dir"] += "/"
        # Load the Julia module and function
     #    jl.include("julia/ground_charging_opt.jl") # Done in julia_caller
        
        # Call the Julia optimization function
        x_vars, y_vars = jl.ground_charging_opt_model_grid(custom_initialization_parameters["risk_pertime_dir"], automatic_initialization_parameters["n_ground_stations"], automatic_initialization_parameters["n_charging_stations"])
        # save the result in a json file
        with open(custom_initialization_parameters["risk_pertime_dir"][-1] + "_ground_sensor_locations.json", "w") as f:
            json.dump(x_vars, f)
        with open(custom_initialization_parameters["risk_pertime_dir"][-1] + "_charging_station_locations.json", "w") as f:
            json.dump(y_vars, f)
        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

class LoggedSensorPlacementStrategy(SensorPlacementStrategy):
        def __init__(self, automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
            """
            Initialize the ground placement strategy using a logged strategy.
            
            Args:
            automatic_initialization_parameters: dict 
            custom_initialization_parameters: dict with keys:
                "logfile": Path to the log file
        Returns:
            ground_sensor_locations: list of tuples (x,y)
            charging_station_locations: list of tuples (x,y)
            """
            with open(custom_initialization_parameters["logfile"]) as log:
                 self.ground_sensor_locations, self.charging_station_locations = json.load(log)
           
