import os
import time

simulation_parameters = {
    "max_battery_distance": -1,
    "max_battery_time": 1,
    "n_drones": 3,
    "n_ground_stations": 4,
    "n_charging_stations": 2,
    "drone_speed_m_per_min": 600,
    "coverage_radius_m": 300,
    "cell_size_m": 30,
    "transmission_range": 50000,
}

def custom_init_fn(input_dir):
    layout_dir = os.path.abspath(os.path.join(input_dir, ".."))
    return {
        "burnmap_filename": f"{layout_dir}/static_risk.npy",
        "reevaluation_step": 2,
        "optimization_horizon": 2,
        "regularization_param": 1,
    }

def step_fn():
    return {}

# === Safe Main Execution ===

if __name__ == "__main__":
    from multiprocessing import freeze_support
    from wrappers import RandomSensorPlacementStrategyLogged, DroneRoutingUniformCoverageResetStaticLogged
    from benchmark import benchmark_on_sim2real_dataset_precompute_parallel
    from Strategy import RandomSensorPlacementStrategy, DroneRoutingUniformCoverageResetStatic
    from new_clustering import get_wrapped_clustering_strategy  # if needed

    freeze_support()

    SensorStrategy = RandomSensorPlacementStrategyLogged
    DroneStrategy = DroneRoutingUniformCoverageResetStaticLogged

    print("\n=== Running SIM2REAL PARALLEL PRECOMPUTE ===")
    
    start_par = time.time()
    all_metrics = benchmark_on_sim2real_dataset_precompute_parallel(
        dataset_folder_name="../DroneBench/",
        ground_placement_strategy=SensorStrategy,
        drone_routing_strategy=DroneStrategy,
        custom_initialization_parameters_function=custom_init_fn,
        custom_step_parameters_function=step_fn,
        max_n_scenarii=50,
        starting_time=0,
        max_n_layouts=12,
        simulation_parameters=simulation_parameters,
        skip_folder_names=[],
        selected_layout_names=["0004_01191", "0016_03070"],
        file_format="jpg",
        config_file='',
        experiment_name='sim2real_test'

    )
    end_par = time.time()

    print(f"\nSIM2REAL parallel precompute runtime: {end_par - start_par:.2f} seconds")

    output_file = "sim2real_metrics_log.txt"
    with open(output_file, "w") as f:
        f.write("[SIM2REAL PARALLEL] Aggregated Metrics per Layout:\n")
        for layout, metrics in all_metrics.items():
            f.write(f"\nLayout: {layout}\n")
            for k, v in metrics.items():
                if isinstance(v, list):
                    f.write(f"  {k}: [list of length {len(v)}]\n")
                else:
                    f.write(f"  {k}: {v}\n")

    print(f"\nMetrics have been written to {output_file}")

    

# if __name__ == "__main__":

#     from wrappers import RandomSensorPlacementStrategyLogged, DroneRoutingUniformCoverageResetStaticLogged
#     from benchmark import benchmark_on_sim2real_dataset_precompute
#     SensorStrategy = RandomSensorPlacementStrategyLogged
#     DroneStrategy = DroneRoutingUniformCoverageResetStaticLogged

#     start_seq = time.time()
#     all_metrics = benchmark_on_sim2real_dataset_precompute(
#         dataset_folder_name="../DroneBench/",
#         ground_placement_strategy=SensorStrategy,
#         drone_routing_strategy=DroneStrategy,
#         custom_initialization_parameters_function=custom_init_fn,
#         custom_step_parameters_function=step_fn,
#         max_n_scenarii=50,
#         starting_time=0,
#         max_n_layouts=12,  # Keep small for sequential test
#         simulation_parameters=simulation_parameters,
#         skip_folder_names=[],
#         selected_layout_names=["0004_01191", "0016_03070", "0019_01316", "0023_00995", "0024_02655", "0025_02019", "0037_01578", "0041_02386", "0069_03539", "0068_04211", "0058_03866", "0089_00984"],
#         file_format="jpg",
#         config_file='',
#         experiment_name='sim2real_sequential_test'
#     )
#     end_seq = time.time()

#     print(f"\nSIM2REAL sequential runtime: {end_seq - start_seq:.2f} seconds")

#     print("\n[SIM2REAL SEQUENTIAL] Aggregated Metrics per Layout:")
#     output_file = "sim2real_metrics_log_sequential.txt"
#     with open(output_file, "w") as f:
#         f.write("[SIM2REAL SEQUENTIAL] Aggregated Metrics per Layout:\n")
#         for layout, metrics in all_metrics.items():
#             f.write(f"\nLayout: {layout}\n")
#             for k, v in metrics.items():
#                 if isinstance(v, list):
#                     f.write(f"  {k}: [list of length {len(v)}]\n")
#                 else:
#                     f.write(f"  {k}: {v}\n")

#     print(f"\nMetrics have been written to {output_file}")
