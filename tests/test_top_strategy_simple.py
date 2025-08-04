import os
import sys
import numpy as np
import tempfile

# Add code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from Strategy import DroneRoutingTOP


def test_top_strategy_basic():
    """Basic sanity-check for DroneRoutingTOP.

    1. Build a tiny 6×6 static burn-map with uniform low risk.
    2. Instantiate the strategy with 2 drones and 1 charging station.
    3. Call get_initial_drone_locations() and then run for 2×battery_time
       steps, collecting the returned actions.
    4. Assert that:
       • no exception is raised
       • each step returns exactly n_drones actions
    """

    # --- 1. Create toy burn-map ------------------------------------
    burnmap = np.full((1, 6, 6), 0.1, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        burnmap_file = os.path.join(tmpdir, "burn.npy")
        np.save(burnmap_file, burnmap)

        # --- 2. Build parameter dictionaries ------------------------
        auto_params = {
            "N": 6,
            "M": 6,
            "max_battery_distance": -1,
            "max_battery_time": 4,
            "n_drones": 2,
            "n_ground_stations": 0,
            "n_charging_stations": 1,
            "ground_sensor_locations": [],
            "charging_stations_locations": [(0, 0)],  # Python 0-based
        }

        custom_params = {
            "burnmap_filename": burnmap_file,
            "reevaluation_step": auto_params["max_battery_time"],
            "optimization_horizon": auto_params["max_battery_time"],
        }

        print("Creating DroneRoutingTOP strategy...")
        # --- 3. Instantiate and get initial positions ---------------
        strat = DroneRoutingTOP(auto_params, custom_params)
        print("Getting initial drone locations...")
        init_actions = strat.get_initial_drone_locations()
        print(f"Initial actions: {init_actions}")
        assert len(init_actions) == auto_params["n_drones"]

        # --- 4. Step through 2×battery_time timesteps --------------
        step_params = {
            "drone_locations": [a[1] for a in init_actions],
            "drone_batteries": [(auto_params["max_battery_distance"], auto_params["max_battery_time"]) for _ in range(auto_params["n_drones"])],
            "drone_states": [a[0] for a in init_actions],
            "t": 0,
        }
        
        print("Running simulation steps...")
        for t in range(2 * auto_params["max_battery_time"]):
            actions = strat.next_actions(step_params, {})
            print(f"Step {t}: actions = {actions}")
            assert len(actions) == auto_params["n_drones"], f"Invalid action length at t={t}"
            
            # Update step_params for next iteration 
            step_params["drone_locations"] = [a[1] for a in actions]
            step_params["drone_states"] = [a[0] for a in actions]
            step_params["t"] = t + 1


if __name__ == "__main__":
    try:
        test_top_strategy_basic()
        print("✅ Test passed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 