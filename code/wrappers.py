
import importlib.util
import os
import tqdm
import json
from dataset import load_scenario_npy
from benchmark import run_benchmark_for_strategy, return_no_custom_parameters
from Strategy import RandomSensorPlacementStrategy, SensorPlacementOptimization

### For SensorPlacement Strategies
def wrap_log_strategy(input_strat_cls):
    """
    Wraps a SensorPlacementStrategy to log and reuse previous placements.

    Args:
        input_strat_cls (SensorPlacementStrategy): The input sensor placement strategy class.

    Returns:
        WrappedStrategy (SensorPlacementStrategy): A wrapped version that logs and reuses results.
    """

    class WrappedStrategy(input_strat_cls):
        def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
            """
            Initialize the wrapped strategy, logging results or loading if already logged.

            automatic_initialization_parameters: dict 
                    Expected keys:
                        - n_ground_stations
                        - n_charging_stations
                        - N, M (grid size)
                custom_initialization_parameters: dict
                    Expected keys:
                        - log_file: Path to the log file
                        - burnmap_filename: Path to the burn map used by the Julia optimizer
            """

            
            layout_name = custom_initialization_parameters.get("log_file", "layout")
            n_ground = automatic_initialization_parameters.get("n_ground_stations", 0)
            n_charging = automatic_initialization_parameters.get("n_charging_stations", 0)
            N = automatic_initialization_parameters.get("N", 0)
            M = automatic_initialization_parameters.get("M", 0)
            strategy_name = input_strat_cls.__name__

            log_path = f"{layout_name}_{strategy_name}_{N}N_{M}M_{n_ground}ground_{n_charging}charge.json"
            

            self.ground_sensor_locations = []
            self.charging_station_locations = []

            if os.path.exists(log_path):
                print(f"[wrap_log_strategy] Loading placement from: {log_path}")
                with open(log_path, "r") as log_file:
                    data = json.load(log_file)

                    # Convert list to tuple
                    self.ground_sensor_locations = [tuple(loc) for loc in data["ground_sensor_locations"]]
                    self.charging_station_locations = [tuple(loc) for loc in data["charging_station_locations"]]
            else:
                print(f"[wrap_log_strategy] Log not found, running {strategy_name}...")
                # call the parent strategy to compute placements
                super().__init__(automatic_initialization_parameters, custom_initialization_parameters)
                # save the computed locations
                self.ground_sensor_locations, self.charging_station_locations = super().get_locations()

                # log to file
                with open(log_path, "w") as log_file:
                    json.dump({
                        "ground_sensor_locations": self.ground_sensor_locations,
                        "charging_station_locations": self.charging_station_locations
                    }, log_file, indent=2)
                print(f"[wrap_log_strategy] Placements saved to: {log_path}")

        def get_locations(self):
            return self.ground_sensor_locations, self.charging_station_locations

    return WrappedStrategy


def wrap_log_drone_strategy(input_drone_cls):
    """
    Wraps a DroneRoutingStrategy to log and reuse routing decisions.

    Args:
        input_drone_cls (DroneRoutingStrategy): The drone routing strategy class to wrap.

        automatic_initialization_parameters: dict
            Expected keys:
                - n_drones: Number of drones
                - N, M: Grid size
                - charging_stations_locations: list of tuples (x, y)
        custom_initialization_parameters: dict
            Expected keys:
                - burnmap_filename: Path to the burn map (not used in dummy version)
                - log_file: Path to save the drone routing log JSON file
                - call_every_n_steps: Frequency to call the optimization (or dummy routing function)
                - optimization_horizon: Number of future steps to plan
        

    Returns:
        WrappedDroneRoutingStrategy (DroneRoutingStrategy): A wrapped version that logs and reuses routing decisions.
    """
    
    class WrappedDroneRoutingStrategy(input_drone_cls):
        def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
            """
            Initialize the wrapped drone strategy, logging results or loading if already logged.
            """
            layout_name = custom_initialization_parameters.get("log_file", "layout")
            n_drones = automatic_initialization_parameters.get("n_drones", 0)
            N = automatic_initialization_parameters.get("N", 0)
            M = automatic_initialization_parameters.get("M", 0)
            strategy_name = input_drone_cls.__name__

            # Log path construction (adjust as needed)
            self.log_path = f"{layout_name}_{strategy_name}_{N}N_{M}M_{n_drones}drones.json"

            self.initial_drone_locations = []
            self.actions_per_timestep = []
            self.call_counter = 0

            if os.path.exists(self.log_path):
                print(f"[wrap_log_drone_strategy] Loading routing from: {self.log_path}")
                with open(self.log_path, "r") as log_file:
                    data = json.load(log_file)
                    self.initial_drone_locations = [tuple(loc) for loc in data["initial_drone_locations"]]
                    self.actions_per_timestep = data["actions_per_timestep"]
            else:
                print(f"[wrap_log_drone_strategy] Log not found, running {strategy_name} initialization...")
                super().__init__(automatic_initialization_parameters, custom_initialization_parameters)
                # Get initial drone locations
                self.initial_drone_locations = super().get_initial_drone_locations()

                # Precompute the routing actions (simulate running through a full optimization)
                # Here we run `next_actions` for `optimization_horizon` timesteps
                self.actions_per_timestep = []
                for t in range(custom_initialization_parameters.get("optimization_horizon", 10)):
                    # Dummy params for testing; replace with your actual inputs as needed
                    automatic_step_params = {
                        "drone_locations": self.initial_drone_locations,
                        "drone_batteries": [(100, 100)] * n_drones,
                        "t": t
                    }
                    custom_step_params = {}

                    actions = super().next_actions(automatic_step_params, custom_step_params)
                    self.actions_per_timestep.append(actions)

                # Save to log file
                with open(self.log_path, "w") as log_file:
                    json.dump({
                        "initial_drone_locations": self.initial_drone_locations,
                        "actions_per_timestep": self.actions_per_timestep
                    }, log_file, indent=2)

                print(f"[wrap_log_drone_strategy] Routing saved to: {self.log_path}")

        def get_initial_drone_locations(self):
            return self.initial_drone_locations

        def next_actions(self, automatic_step_parameters: dict, custom_step_parameters: dict):
            pass #do we need a next_actions if super() already has one?

    return WrappedDroneRoutingStrategy
