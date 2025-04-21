import unittest
import numpy as np
import os
import time
import sys
import shutil

# Add code directory to Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "code"))
if module_path not in sys.path:
    sys.path.append(module_path)

from Strategy import (
    DroneRoutingOptimizationModelReuseIndex, 
    RandomSensorPlacementStrategy,
    SensorPlacementOptimization,
    RandomDroneRoutingStrategy
)
from wrappers import wrap_log_drone_strategy, wrap_log_sensor_strategy
from dataset import preprocess_sim2real_dataset, load_scenario_npy, compute_and_save_burn_maps_sim2real_dataset
from benchmark import (
    run_benchmark_scenario,
    run_benchmark_scenarii_sequential,
    benchmark_on_sim2real_dataset,
    get_burnmap_parameters,
    run_benchmark_scenarii_sequential_precompute,
    benchmark_on_sim2real_dataset_precompute
)

class TestDroneRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests."""
        # Check if MinimalDataset exists
        cls.original_dataset = "MinimalDataset"
        if not os.path.exists(cls.original_dataset):
            raise unittest.SkipTest("MinimalDataset not found in current directory. Please ensure it exists.")
        print("Found MinimalDataset, proceeding with tests...")

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Load actual scenario and burn map paths
        self.scenario_path = "MinimalDataset/0001/scenarii/0001_00002.npy"
        self.burn_map_path = "MinimalDataset/0001/burn_map.npy"
        
        # Load actual scenario to get dimensions
        self.scenario = load_scenario_npy(self.scenario_path)
        self.N = self.scenario.shape[1]
        self.M = self.scenario.shape[2]

        # Create wrapped strategy classes
        self.WrappedSensorStrategy = wrap_log_sensor_strategy(RandomSensorPlacementStrategy)
        self.WrappedDroneStrategy = wrap_log_drone_strategy(DroneRoutingOptimizationModelReuseIndex)

    def my_automatic_layout_parameters(self, scenario):
        """Define automatic layout parameters as in the notebook."""
        return {
            "N": scenario.shape[1],
            "M": scenario.shape[2],
            "max_battery_distance": -10,
            "max_battery_time": 20,
            "n_drones": 10,
            "n_ground_stations": 1,
            "n_charging_stations": 5,
        }

    def custom_initialization_parameters_function(self, input_dir, layout_name=None):
        """Define custom initialization parameters as in the notebook."""
        print(f"input_dir: {input_dir}, layout_name: {layout_name}")
        
        # Determine path to burnmap
        burnmap_path = f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy"
        
        # Create logs directory if it doesn't exist
        log_dir = f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file path
        log_file = f"{log_dir}/{layout_name or 'default'}_log.json"
        
        return {
            "burnmap_filename": burnmap_path,
            "reevaluation_step": 5,
            "optimization_horizon": 5,
            "strategy_drone": DroneRoutingOptimizationModelReuseIndex,
            "strategy_sensor": RandomSensorPlacementStrategy,
            "log_file": log_file
        }

    def test_single_scenario_benchmark(self):
        """Test running a benchmark on a single scenario."""
        print("Testing single scenario benchmark...")
        time_start = time.time()
        
        # Load scenario
        scenario = load_scenario_npy(self.scenario_path)
        
        # Run benchmark
        results, (position_history, ground, charging) = run_benchmark_scenario(
            scenario,
            SensorPlacementOptimization,
            DroneRoutingOptimizationModelReuseIndex,
            custom_initialization_parameters={
                "burnmap_filename": self.burn_map_path,
                "load_from_logfile": False,
                "reevaluation_step": 5,
                "optimization_horizon": 5,
                "strategy_drone": DroneRoutingOptimizationModelReuseIndex,
                "strategy_sensor": RandomSensorPlacementStrategy
            },
            custom_step_parameters_function=lambda: None,
            automatic_initialization_parameters_function=self.my_automatic_layout_parameters,
            return_history=True
        )
        
        # Basic assertions
        self.assertIsInstance(results, dict)
        self.assertIn('delta_t', results)
        self.assertIn('device', results)
        
        print(f"Time taken to run benchmark: {time.time() - time_start} seconds")

    def test_dataset_benchmark(self):
        """Test running a benchmark on the dataset."""
        print("Testing dataset benchmark...")
        
        def custom_initialization_parameters_function(input_dir: str, layout_name: str = None):
            print(f"input_dir: {input_dir}, layout_name: {layout_name}")
            burnmap_path = f"{'/'.join(input_dir.strip('/').split('/')[:-1])}/burn_map.npy"
            return {
                "burnmap_filename": burnmap_path,
                "load_from_logfile": False,
                "reevaluation_step": 5,
                "optimization_horizon": 5,
                "strategy_drone": DroneRoutingOptimizationModelReuseIndex,
                "strategy_sensor": RandomSensorPlacementStrategy
            }

        input_dir = "MinimalDataset/0002/scenarii/"
        print(f"Using input directory: {input_dir}")
        
        # First try sequential benchmark
        run_benchmark_scenarii_sequential_precompute(
            input_dir=input_dir,
            sensor_placement_strategy=SensorPlacementOptimization,
            drone_routing_strategy=DroneRoutingOptimizationModelReuseIndex,
            custom_initialization_parameters_function=custom_initialization_parameters_function,
            custom_step_parameters_function=lambda: None,
            starting_time=0,
            max_n_scenarii=2
        )
        print(f"dataset-wide benchmark sim2real precompute")
        # Then try dataset-wide benchmark
        benchmark_on_sim2real_dataset_precompute(
            "MinimalDataset/",
            SensorPlacementOptimization,
            DroneRoutingOptimizationModelReuseIndex,
            custom_initialization_parameters_function,
            lambda: None,
            max_n_scenarii=2,
            starting_time=0
        )
        
        # Check that log files were created
        log_dir = "MinimalDataset/0001/logs"
        self.assertTrue(os.path.exists(log_dir), f"Log directory {log_dir} was not created")

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests."""
        # No cleanup needed since we're using the original dataset directly
        pass

if __name__ == '__main__':
    unittest.main() 