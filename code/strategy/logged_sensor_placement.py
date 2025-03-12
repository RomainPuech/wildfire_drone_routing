import random
import os
from my_julia_caller import jl # DEACTIVATED TO RUN THINGS IN PARALLEL 
import json
import numpy as np


class LoggedSensorPlacementStrategy:
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
