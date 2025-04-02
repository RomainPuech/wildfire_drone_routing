import ray
ray.init()  # Or ray.init(num_cpus=N) to control parallelism
@ray.remote

def parallel_run_scenario(scenario_file, 
                          sensor_placement_strategy_cls, 
                          drone_routing_strategy_cls, 
                          custom_init_params_fn, 
                          custom_step_params_fn, 
                          starting_time):
    # 1. Import what you need inside the task!
    from juliacall import Main as jl
    from dataset import load_scenario_npy
    from benchmark import run_benchmark_scenario, get_automatic_layout_parameters

    # 2. Initialize Julia here to avoid session sharing issues
    jl.seval("""
    using Logging
    global_logger(SimpleLogger(stderr, Logging.Error))  # Silence info logs
    """)
    print("Initializing Julia inside Ray task.")
    jl.include("julia/ground_charging_opt.jl")
    jl.include("julia/drone_routing_opt.jl")
    print("Julia session initialized inside Ray task.")

    # 3. Load scenario
    scenario = load_scenario_npy(scenario_file)

    # 4. Get initialization params
    automatic_init_params = get_automatic_layout_parameters(scenario)
    custom_init_params = custom_init_params_fn(scenario_file, layout_name="layout_A")

    # 5. Create strategy instances
    sensor_strategy = sensor_placement_strategy_cls(automatic_init_params, custom_init_params)
    drone_strategy = drone_routing_strategy_cls(automatic_init_params, custom_init_params)

    # 6. Run the benchmark
    results, _ = run_benchmark_scenario(
        scenario,
        sensor_strategy,
        drone_strategy,
        custom_init_params,
        custom_step_params_fn,
        starting_time=starting_time
    )

    return results


def run_parallel_benchmarks(input_dir, 
                            sensor_placement_strategy_cls, 
                            drone_routing_strategy_cls, 
                            custom_init_params_fn, 
                            custom_step_params_fn, 
                            starting_time=0, 
                            max_n_scenarii=None):
    from benchmark import listdir_npy_limited
    import tqdm

    # Get scenario files
    scenario_files = list(listdir_npy_limited(input_dir, max_n_scenarii))
    print(f"Found {len(scenario_files)} scenarios.")

    # Launch Ray tasks
    futures = []
    for scenario_file in scenario_files:
        future = parallel_run_scenario.remote(
            scenario_file,
            sensor_placement_strategy_cls,
            drone_routing_strategy_cls,
            custom_init_params_fn,
            custom_step_params_fn,
            starting_time
        )
        futures.append(future)

    # Collect results
    results = []
    for result in tqdm.tqdm(ray.get(futures), total=len(futures)):
        results.append(result)

    return results



if __name__ == "__main__":
    # ray.init()  # Or ray.init(num_cpus=N)

    # Import strategies
    from Strategy import RandomSensorPlacementStrategy, RandomDroneRoutingStrategy
    from benchmark import build_custom_init_params, return_no_custom_parameters

    results = run_parallel_benchmarks(
        input_dir="./MinimalDataset/0001/scenarii",
        sensor_placement_strategy_cls=RandomSensorPlacementStrategy,
        drone_routing_strategy_cls=RandomDroneRoutingStrategy,
        custom_init_params_fn=build_custom_init_params,
        custom_step_params_fn=return_no_custom_parameters,
        starting_time=0,
        max_n_scenarii=10
    )

    # Do something with results
    print("All parallel benchmarks completed!")
    print(results)

    ray.shutdown()
