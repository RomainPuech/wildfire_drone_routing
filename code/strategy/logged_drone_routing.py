import random
import os
# from my_julia_caller import jl # DEACTIVATED TO RUN THINGS IN PARALLEL 
import json
import re
import numpy as np


class LoggedDroneRoutingStrategy:
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
            if "logfile" not in custom_initialization_parameters:
                raise ValueError("custom_initialization_parameters must include 'logfile'")
            if "burnmap_filename" not in custom_initialization_parameters:
                raise ValueError("custom_initialization_parameters must include 'burnmap_filename'")

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
            return self.initial_drone_locations
        
        def next_actions(self, automatic_step_parameters, custom_step_parameters):
            # return the next precomputed actions for this timestep
            return self.actions_per_timestep
