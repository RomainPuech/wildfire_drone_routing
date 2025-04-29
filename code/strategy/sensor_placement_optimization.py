import random
import os
from my_julia_caller import jl # DEACTIVATED TO RUN THINGS IN PARALLEL 
import json
import re
import numpy as np
from Strategy import SensorPlacementStrategy

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
        x_vars, y_vars = jl.NEW_SENSOR_STRATEGY_3(custom_initialization_parameters["burnmap_filename"], automatic_initialization_parameters["n_ground_stations"], automatic_initialization_parameters["n_charging_stations"])
        print("optimization finished")
        print("Charging Station Locations from Julia Optimization Model: ", self.charging_station_locations)


        
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

        print("ground sensor locations")
        print(self.ground_sensor_locations)
        print("charging station locations")
        print(self.charging_station_locations)