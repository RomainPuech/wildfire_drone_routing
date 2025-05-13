import random
import os
from my_julia_caller import jl # DEACTIVATED TO RUN THINGS IN PARALLEL 
import json
import re
import numpy as np
from Strategy import DroneRoutingStrategy

class DroneRoutingOptimizationModelReuseIndex(DroneRoutingStrategy):
    strategy_name = "DroneRoutingOptimizationModelReuseIndex"
    # print("=== TEST PRINT: Entered DroneRoutingOptimizationModelReuseIndex class definition ===")

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
        self.routing_model = jl.create_index_routing_model(
            self.custom_initialization_parameters["burnmap_filename"],
            self.automatic_initialization_parameters["n_drones"],
            self.julia_charging_stations_locations,
            self.julia_ground_sensor_locations,
            self.custom_initialization_parameters["optimization_horizon"],
            self.automatic_initialization_parameters["max_battery_time"]
        )
        
        # Solve the initial routing problem with the model
        print("[DEBUG] Running solve_index_init_routing()...")
        self.current_solution = jl.solve_index_init_routing(
            self.routing_model, 
            self.custom_initialization_parameters["reevaluation_step"]
        )
        print("[DEBUG] Finished solve_index_init_routing()...")
        
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
        idx = min(self.call_counter, len(self.current_solution) - 1)
        # print(f"[debug] returning plan step {self.call_counter} of {len(self.current_solution)}")
        return self.current_solution[idx]