
import importlib.util
import os
import tqdm
import json
from dataset import load_scenario_npy
from benchmark import run_benchmark_for_strategy, return_no_custom_parameters, my_custom_init_params, load_strategy
from Strategy import RandomSensorPlacementStrategy, SensorPlacementOptimization


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





# if __name__ == "__main__":
    # Wrapped = wrap_log_strategy(RandomSensorPlacementStrategy)

    # # === parameters ===
    # automatic_params = {"N": 10, "M": 10, "n_ground_stations": 4, "n_charging_stations": 4}
    # custom_params = {"logfile": "./MinimalDataset/0001/scenarii", "layout_name": "layout_B"}

    # # === run wrapped strategy ===
    # strategy = Wrapped(automatic_params, custom_params)
    # ground, charge = strategy.get_locations()

    # print("Ground sensors:", ground)
    # print("Charging stations:", charge)

    # # === run again (no computation) ===
    # strategy2 = Wrapped(automatic_params, custom_params)
    # ground2, charge2 = strategy2.get_locations()

    # print("Ground sensors (second run):", ground2)
    # print("Charging stations (second run):", charge2)
    # ===================================================#
   
    # Wrapped = wrap_log_strategy(SensorPlacementOptimization)

    # # === parameters ===
    # automatic_params = {"N": 10, "M": 10, "n_ground_stations": 4, "n_charging_stations": 4}
    # custom_params = {"logfile": "./MinimalDataset/0001/scenarii", "layout_name": "layout_B", "burnmap_filename": "./MinimalDataset/0001/burn_map.npy"}

    # # === run wrapped strategy ===
    # strategy = Wrapped(automatic_params, custom_params)
    # ground, charge = strategy.get_locations()

    # print("Ground sensors:", ground)
    # print("Charging stations:", charge)

    # # === run again (no computation) ===
    # strategy2 = Wrapped(automatic_params, custom_params)
    # ground2, charge2 = strategy2.get_locations()

    # print("Ground sensors (second run):", ground2)
    # print("Charging stations (second run):", charge2)


    # === wrap it ===

    # run_benchmark_for_strategy(
    #     input_dir="MinimalDataset/0001/scenarii",
    #     strategy_folder="code/strategy",
    #     sensor_strategy_file="logged_sensor_placement.py",
    #     sensor_class_name="LoggedSensorPlacementStrategy",
    #     drone_strategy_file="logged_drone_routing.py",
    #     drone_class_name="LoggedDroneRoutingStrategy",
    #     max_n_scenarii=4,
    #     starting_time=0,
    #     file_format= "npy",
    #     custom_init_params_fn= my_custom_init_params,  # user-defined initialization params function
    #     custom_step_params_fn= return_no_custom_parameters
    # )