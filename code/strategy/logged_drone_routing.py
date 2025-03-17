# import random
# import os
# # from my_julia_caller import jl # DEACTIVATED TO RUN THINGS IN PARALLEL 
# import json
# import re
# import numpy as np


# class LoggedDroneRoutingStrategy:
#     """
#     LoggedDroneRoutingStrategy logs drone routing actions and locations at every timestep.
    
#     Args:
#         automatic_initialization_parameters: dict
#             Expected keys:
#                 - n_drones: Number of drones
#                 - N, M: Grid size
#                 - charging_stations_locations: list of tuples (x, y)
#         custom_initialization_parameters: dict
#             Expected keys:
#                 - burnmap_filename: Path to the burn map (not used in dummy version)
#                 - logfile: Path to save the drone routing log JSON file
#                 - call_every_n_steps: Frequency to call the optimization (or dummy routing function)
#                 - optimization_horizon: Number of future steps to plan

#     Returns:
#         Initializes:
#             - self.initial_drone_locations: list of tuples (x, y)
#             - self.log_data: log structure with initial locations and step logs
#     """
#     def __init__(self, automatic_initialization_parameters, custom_initialization_parameters):
#         # assign parameters from parent
#         self.automatic_initialization_parameters = automatic_initialization_parameters
#         self.custom_initialization_parameters = custom_initialization_parameters

#         # validate parameters
#         if "burnmap_filename" not in custom_initialization_parameters:
#             raise ValueError("Missing 'burnmap_filename' in custom_initialization_parameters")
#         if "log_file" not in custom_initialization_parameters:
#             raise ValueError("Missing 'logfile' in custom_initialization_parameters")
#         if "call_every_n_steps" not in custom_initialization_parameters:
#             raise ValueError("Missing 'call_every_n_steps' in custom_initialization_parameters")
#         if "optimization_horizon" not in custom_initialization_parameters:
#             raise ValueError("Missing 'optimization_horizon' in custom_initialization_parameters")

#         # config values
#         self.call_every_n_steps = custom_initialization_parameters["call_every_n_steps"]
#         self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

#         # initialize counters and memory
#         self.call_counter = 0
#         self.current_solution = []  # holds lists of actions between Julia calls

#         # logging
#         self.logfile = custom_initialization_parameters["logfile"]
#         self.log_data = {
#             "initial_drone_locations": None,  # set in get_initial_drone_locations()
#             "steps": []  # append timestep logs here
#         }

#         print(f"[LoggedDroneRoutingStrategy] Initialized with log file: {self.logfile}")

#     def get_initial_drone_locations(self):
#         charging_stations = self.automatic_initialization_parameters["charging_stations_locations"]
#         n_drones = self.automatic_initialization_parameters["n_drones"]

#         n_stations = len(charging_stations)
#         q = n_drones // n_stations
#         r = n_drones % n_stations

#         initial_positions = charging_stations * q + charging_stations[:r]

#         self.log_data["initial_drone_locations"] = initial_positions
#         self._write_log_to_file()

#         return initial_positions

#     def next_actions(self, automatic_step_parameters: dict, custom_step_parameters: dict):
#         """
#         automatic_step_parameters: dict with keys:
#             - "drone_locations": list of tuples (x,y)
#             - "drone_batteries": list of tuples (distance,time)
#             - "t": int
#         Returns:
#             actions: list of tuples (action_type, action_parameters)
#         """
#         if self.call_counter % self.call_every_n_steps == 0:
#             print(f"[LoggedDroneRoutingStrategy] Calling dummy optimizer at timestep {self.call_counter}")
#             _, self.current_solution = self.dummy_drone_routing_robust(
#                 automatic_step_parameters, custom_step_parameters
#             )
#             print("[LoggedDroneRoutingStrategy] Dummy optimization finished")

#         timestep_index = self.call_counter % self.call_every_n_steps
#         actions = self.current_solution[timestep_index]

#         # log actions and states
#         self._log_timestep(
#             timestep=automatic_step_parameters["t"],
#             drone_locations=automatic_step_parameters["drone_locations"],
#             drone_batteries=automatic_step_parameters["drone_batteries"],
#             actions=actions
#         )

#         self.call_counter += 1
#         return actions
    
#     def dummy_drone_routing_robust(self, automatic_step_parameters, custom_step_parameters):
#         print("[Dummy Function] Generating dummy routing solution...")

#         n_drones = self.automatic_initialization_parameters.get("n_drones", 3)
#         n_timesteps = self.optimization_horizon

#         initial_locations = [(i * 5, i * 5) for i in range(n_drones)]

#         actions_per_timestep = []
#         for t in range(n_timesteps):
#             actions = []
#             for d in range(n_drones):
#                 if t % 2 == 0:
#                     actions.append(('move', (1, 0)))
#                 else:
#                     actions.append(('charge', None))
#             actions_per_timestep.append(actions)

#         return initial_locations, actions_per_timestep
    
#     def _log_timestep(self, timestep, drone_locations, drone_batteries, actions):
#         """
#         Logs the state and actions at each timestep.
#         """
#         log_entry = {
#             "timestep": timestep,
#             "drone_locations": drone_locations,
#             "drone_batteries": drone_batteries,
#             "actions": actions
#         }

#         self.log_data["steps"].append(log_entry)

#         # === Write the log to file immediately after each timestep ===
#         print(f"[LoggedDroneRoutingStrategy] Writing log to {self.logfile} at timestep {timestep}")
#         self._write_log_to_file()

#     def _write_log_to_file(self):
#         """
#         Writes the current log to the logfile.
#         """
#         log_dir = os.path.dirname(self.logfile)
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir, exist_ok=True)

#         with open(self.logfile, "w") as f:
#             json.dump(self.log_data, f, indent=2)

#         print(f"[LoggedDroneRoutingStrategy] Log successfully written to {self.logfile}")
