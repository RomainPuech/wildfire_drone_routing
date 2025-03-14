#!/usr/bin/env python3

import os
import time
import sys
import json
import numpy as np
import gc
from pprint import pprint
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import our strategy classes
from Strategy import (
    DroneRoutingOptimizationSlow, 
    DroneRoutingOptimizationModelReuse,
    DroneRoutingOptimizationModelReuseIndex
)

def setup_benchmark_environment(n_drones=2, horizon=15, reevaluation_step=5):
    """Set up benchmark parameters with configurable values."""
    # Basic simulation parameters
    burnmap_filename = "./WideDataset/0001/burn_map.npy"
    charging_stations_locations = [(100, 98)]  # Python 0-indexed
    ground_sensor_locations = [(0, 0)]  # Python 0-indexed
    max_battery_time = 10
    
    # Initialize parameters dictionaries
    automatic_initialization_parameters = {
        "N": 100,
        "M": 100,
        "max_battery_distance": 100,
        "max_battery_time": max_battery_time,
        "n_drones": n_drones,
        "n_ground_stations": len(ground_sensor_locations),
        "n_charging_stations": len(charging_stations_locations),
        "ground_sensor_locations": ground_sensor_locations,
        "charging_stations_locations": charging_stations_locations
    }
    
    custom_initialization_parameters = {
        "burnmap_filename": burnmap_filename,
        "reevaluation_step": reevaluation_step,
        "optimization_horizon": horizon
    }
    
    return automatic_initialization_parameters, custom_initialization_parameters

def benchmark_strategy(strategy_class, auto_params, custom_params, num_steps=5):
    """Benchmark a specific strategy class."""
    print(f"\n--- Benchmarking {strategy_class.__name__} ---")
    
    # Initialize strategy and measure initialization + get_initial_drone_locations time
    init_start = time.time()
    strategy = strategy_class(auto_params, custom_params)
    initial_locations = strategy.get_initial_drone_locations()
    init_end = time.time()
    init_time = init_end - init_start
    print(f"Initialization time: {init_time:.4f}s")
    
    # Measure next_actions calls
    step_times = []
    
    # Set up parameters for next_actions
    auto_step_params = {
        "drone_locations": initial_locations,
        "drone_batteries": [(100, 10) for _ in range(len(initial_locations))],
        "drone_states": ["fly" for _ in range(len(initial_locations))],
        "t": 0
    }
    custom_step_params = {}
    
    # Set reevaluation_step to 1 to force model creation/reuse on every call
    strategy.reevaluation_step = 1
    
    for step in range(num_steps):
        auto_step_params["t"] = step
        
        # Force gc to ensure fair comparison
        gc.collect()
        
        start_time = time.time()
        strategy.next_actions(auto_step_params, custom_step_params)
        end_time = time.time()
        
        step_time = end_time - start_time
        step_times.append(step_time)
        print(f"Step {step + 1} time: {step_time:.4f}s")
    
    # Calculate total time
    total_time = init_time + sum(step_times)
    print(f"Total time: {total_time:.4f}s")
    
    return {
        "init_time": init_time,
        "step_times": step_times,
        "total_time": total_time,
        "initial_locations": initial_locations,
        "solution": strategy.current_solution
    }

def run_indexing_comparison_benchmark():
    """Compare the performance of different indexing approaches."""
    print("\n=== Comparing Performance of Different Drone Routing Approaches ===\n")
    
    # Benchmark parameters
    n_drones = 2
    horizon = 15
    reevaluation_step = 1  # Force model creation/reuse on every call
    num_steps = 5  # Number of next_actions calls to make
    
    # Set up environment
    auto_params, custom_params = setup_benchmark_environment(
        n_drones=n_drones, 
        horizon=horizon, 
        reevaluation_step=reevaluation_step
    )
    
    # Benchmark each strategy
    strategies = [
        DroneRoutingOptimizationSlow,
        DroneRoutingOptimizationModelReuse,
        DroneRoutingOptimizationModelReuseIndex
    ]
    
    results = {}
    
    for strategy_class in strategies:
        gc.collect()  # Force garbage collection between benchmarks
        results[strategy_class.__name__] = benchmark_strategy(
            strategy_class, 
            auto_params, 
            custom_params, 
            num_steps
        )
    
    # Compare solutions
    print("\n--- Solution Comparison ---")
    solutions_match = True
    base_solution = results["DroneRoutingOptimizationSlow"]["solution"]
    
    for strategy_name, strategy_results in results.items():
        if strategy_name == "DroneRoutingOptimizationSlow":
            continue
            
        solution_match = strategy_results["solution"] == base_solution
        if not solution_match:
            solutions_match = False
            print(f"{strategy_name} solution differs from base solution")
    
    if solutions_match:
        print("All strategies produced identical solutions")
    
    # Calculate speedups
    print("\n--- Performance Comparison ---")
    base_time = results["DroneRoutingOptimizationSlow"]["total_time"]
    
    for strategy_name, strategy_results in results.items():
        if strategy_name == "DroneRoutingOptimizationSlow":
            continue
            
        strategy_time = strategy_results["total_time"]
        speedup = base_time / strategy_time
        time_saved = base_time - strategy_time
        percent_faster = (time_saved / base_time) * 100
        
        print(f"{strategy_name} vs. DroneRoutingOptimizationSlow:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time saved: {time_saved:.4f}s ({percent_faster:.1f}% faster)")
    
    # Compare model reuse approaches
    if "DroneRoutingOptimizationModelReuse" in results and "DroneRoutingOptimizationModelReuseIndex" in results:
        tuple_time = results["DroneRoutingOptimizationModelReuse"]["total_time"]
        index_time = results["DroneRoutingOptimizationModelReuseIndex"]["total_time"]
        index_vs_tuple_speedup = tuple_time / index_time
        
        print("\nInteger-based vs. Tuple-based model reuse:")
        print(f"  Speedup: {index_vs_tuple_speedup:.2f}x")
    
    # Create bar chart comparing total times
    plt.figure(figsize=(10, 6))
    
    strategy_names = [s.__name__ for s in strategies]
    total_times = [results[s.__name__]["total_time"] for s in strategies]
    init_times = [results[s.__name__]["init_time"] for s in strategies]
    step_times = [sum(results[s.__name__]["step_times"]) for s in strategies]
    
    # Plot stacked bar chart
    bar_width = 0.6
    plt.bar(strategy_names, init_times, bar_width, label='Initialization')
    plt.bar(strategy_names, step_times, bar_width, bottom=init_times, label='Next Actions')
    
    # Add total times as text
    for i, total in enumerate(total_times):
        plt.text(i, total + 0.5, f"{total:.2f}s", ha='center')
    
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison of Drone Routing Strategies')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('indexing_comparison_benchmark.png')
    print("\nPlot saved to 'indexing_comparison_benchmark.png'")
    
    # Save detailed results to JSON
    detailed_results = {
        strategy_name: {
            "init_time": data["init_time"],
            "step_times": data["step_times"],
            "total_time": data["total_time"]
        } for strategy_name, data in results.items()
    }
    
    detailed_results["comparisons"] = {
        "model_reuse_vs_slow": base_time / results["DroneRoutingOptimizationModelReuse"]["total_time"],
        "model_reuse_index_vs_slow": base_time / results["DroneRoutingOptimizationModelReuseIndex"]["total_time"],
        "model_reuse_index_vs_model_reuse": results["DroneRoutingOptimizationModelReuse"]["total_time"] / results["DroneRoutingOptimizationModelReuseIndex"]["total_time"],
        "solutions_match": solutions_match
    }
    
    with open("indexing_comparison_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print("Results saved to 'indexing_comparison_results.json'")
    print("\n=== Benchmark Complete ===\n")
    
    return detailed_results

if __name__ == "__main__":
    run_indexing_comparison_benchmark() 