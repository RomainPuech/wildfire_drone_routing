import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from Strategy import DroneRoutingTOP
from displays import create_video_scenario_burnmap
from dataset import load_burn_map

def plot_drone_trajectories(trajectories, grid_size, charging_stations, max_battery_time, save_path=None):
    """Plot the trajectories of all drones on separate grids for each journey.
    
    Args:
        trajectories: dict with drone_id as key and list of (x, y) positions as value
        grid_size: tuple (N, M) representing grid dimensions
        charging_stations: list of (x, y) charging station locations
        max_battery_time: int, used to determine journey boundaries
        save_path: optional path to save the plot
    """
    # Determine the number of journeys by looking at the longest trajectory
    max_trajectory_length = max(len(traj) for traj in trajectories.values()) if trajectories else 0
    num_journeys = ((max_trajectory_length - 1) // max_battery_time) + 1
    
    print(f"Creating {num_journeys} separate plots for each journey")
    
    # Create subplots - arrange in a row for better comparison
    fig, axes = plt.subplots(1, num_journeys, figsize=(6 * num_journeys, 6))
    if num_journeys == 1:
        axes = [axes]  # Make it iterable for single subplot
    
    # Define colors for different drones
    drone_colors = ['blue', 'darkgreen', 'orange', 'purple', 'brown', 'deeppink']
    
    N, M = grid_size
    
    # Plot each journey in a separate subplot
    for journey_idx in range(num_journeys):
        ax = axes[journey_idx]
        
        # Set up the grid for this journey
        ax.set_xlim(-0.5, M-0.5)
        ax.set_ylim(-0.5, N-0.5)
        ax.set_aspect('equal')
        
        # Draw grid lines
        for i in range(N+1):
            ax.axhline(i-0.5, color='lightgray', linewidth=0.5)
        for j in range(M+1):
            ax.axvline(j-0.5, color='lightgray', linewidth=0.5)
        
        # Plot charging stations
        if charging_stations:
            cs_x, cs_y = zip(*charging_stations)
            ax.scatter(cs_y, cs_x, marker='s', s=300, c='red', 
                       label='Charging Stations', zorder=5, edgecolor='black', linewidth=2)
        
        # Plot trajectories for each drone in this journey
        journey_has_data = False
        for drone_id, trajectory in trajectories.items():
            if len(trajectory) > 0:
                # Calculate journey boundaries for this drone
                journey_boundaries = [0]
                for i in range(max_battery_time, len(trajectory), max_battery_time):
                    journey_boundaries.append(i)
                journey_boundaries.append(len(trajectory))
                
                # Check if this journey exists for this drone
                if journey_idx < len(journey_boundaries) - 1:
                    start_idx = journey_boundaries[journey_idx]
                    end_idx = journey_boundaries[journey_idx + 1]
                    journey_trajectory = trajectory[start_idx:end_idx]
                    
                    if len(journey_trajectory) > 1:
                        journey_has_data = True
                        
                        # Extract coordinates for this journey
                        x_coords = [pos[1] for pos in journey_trajectory]  # y coordinate -> x axis
                        y_coords = [pos[0] for pos in journey_trajectory]  # x coordinate -> y axis
                        
                        drone_color = drone_colors[drone_id % len(drone_colors)]
                        
                        # Plot trajectory line for this journey
                        ax.plot(x_coords, y_coords, color=drone_color, linewidth=3, 
                                alpha=0.8, marker='o', markersize=6, 
                                label=f'Drone {drone_id}', markerfacecolor='white', 
                                markeredgecolor=drone_color, markeredgewidth=2)
                        
                        # Mark journey start position with larger marker
                        ax.scatter(x_coords[0], y_coords[0], marker='o', s=200, 
                                   c=drone_color, alpha=1.0, edgecolor='black', 
                                   linewidth=3, zorder=6, label=f'Drone {drone_id} Start')
                        
                        # Mark journey end position
                        ax.scatter(x_coords[-1], y_coords[-1], marker='X', s=200, 
                                   c=drone_color, alpha=1.0, edgecolor='black', 
                                   linewidth=3, zorder=6)
                        
                        # Add drone ID annotation at the start
                        ax.annotate(f'D{drone_id}', 
                                   (x_coords[0], y_coords[0]), 
                                   xytext=(15, 15), textcoords='offset points',
                                   fontsize=12, color=drone_color, weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
                        
                        # Add step numbers along the path
                        for step, (x, y) in enumerate(zip(x_coords[1:], y_coords[1:]), 1):
                            ax.annotate(f'{step}', (x, y), 
                                       xytext=(2, 2), textcoords='offset points',
                                       fontsize=8, color=drone_color, alpha=0.7)
        
        # Set title and labels for this journey
        ax.set_title(f'Journey {journey_idx + 1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Grid Y Coordinate', fontsize=10)
        if journey_idx == 0:
            ax.set_ylabel('Grid X Coordinate', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first subplot to avoid clutter
        if journey_idx == 0 and journey_has_data:
            # Create a custom legend with unique entries
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)
        
        # If no data for this journey, add a note
        if not journey_has_data:
            ax.text(0.5, 0.5, 'No trajectory data\nfor this journey', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, alpha=0.5, style='italic')
    
    # Add overall title
    fig.suptitle('Drone Trajectories - Separate Journeys', fontsize=16, fontweight='bold')
    
    # Ensure tight layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Journey plots saved to: {save_path}")
    
    plt.show()


def test_top_strategy_basic():
    """Basic sanity-check for DroneRoutingTOP.

    1. Build a tiny 15×15 static burn-map with uniform low risk.
    2. Instantiate the strategy with 2 drones and 1 charging station.
    3. Call get_initial_drone_locations() and then run for 2×battery_time
       steps, collecting the returned actions.
    4. Track and plot drone trajectories, showing different journeys.
    5. Assert that:
       • no exception is raised
       • each step returns exactly n_drones actions
    """

    # --- 1. Create toy burn-map ------------------------------------
    # exponential distribution with mean 0.1
    burnmap = np.random.pareto(1, size=(1, 15, 15)) * 0.001  # Pareto with shape parameter 1 gives very fat tails
    #burnmap = burnmap / np.max(burnmap)
    
    burnmap_file = os.path.join("tmp_burnmap.npy")
    np.save(burnmap_file, burnmap)

    # --- 2. Build parameter dictionaries ------------------------
    L = 10
    auto_params = {
        "N": 15,
        "M": 15,
        "max_battery_distance": -1,
        "max_battery_time": L,
        "n_drones": 2,
        "n_ground_stations": 0,
        "n_charging_stations": 1,
        "ground_sensor_locations": [],
        "charging_stations_locations": [(6, 5)],  # Python 0-based
        "data_time_resolution": L, # TODO for the momemnt battery = data res. THINK ABT IMPLICATIONS IF BATTERY IS LESS
    }

    custom_params = {
        "burnmap_filename": burnmap_file,
        "reevaluation_step": auto_params["max_battery_time"],
        "optimization_horizon": auto_params["max_battery_time"],
        "burnmap_type": "static",
        "reset_time": 2*auto_params["max_battery_time"],
    }

    print("Creating DroneRoutingTOP strategy...")
    # --- 3. Instantiate and get initial positions ---------------
    strat = DroneRoutingTOP(auto_params, custom_params)
    print("Getting initial drone locations...")
    init_actions, current_burnmap_filename = strat.get_initial_drone_locations()
    print(f"Initial actions: {init_actions}")
    assert len(init_actions) == auto_params["n_drones"]

    # --- 4. Initialize trajectory tracking ----------------------
    trajectories = {i: [] for i in range(auto_params["n_drones"])}
    
    # Add initial positions to trajectories
    for i, action in enumerate(init_actions):
        trajectories[i].append(action[1])  # action[1] is the (x, y) position

    # --- 5. Step through 2×battery_time timesteps --------------
    step_params = {
        "drone_locations": [a[1] for a in init_actions],
        "drone_batteries": [(auto_params["max_battery_distance"], auto_params["max_battery_time"]) for _ in range(auto_params["n_drones"])],
        "drone_states": [a[0] for a in init_actions],
        "t": 0,
    }
    
    print("Running simulation steps...")
    total_steps = 2 * auto_params["max_battery_time"]
    for t in range(total_steps):
        actions = strat.next_actions(step_params, {})
        # Print progress every 10 steps instead of every step to avoid blocking IO
        if t % 10 == 0 or t == total_steps - 1:
            print(f"Step {t}/{total_steps}: actions = {actions}")
        assert len(actions) == auto_params["n_drones"], f"Invalid action length at t={t}"
        
        # Track trajectories
        for i, action in enumerate(actions):
            trajectories[i].append(action[1])  # action[1] is the (x, y) position
        
        # Update step_params for next iteration 
        step_params["drone_locations"] = [a[1] for a in actions]
        step_params["drone_states"] = [a[0] for a in actions]
        step_params["t"] = t + 1

    # --- 6. Plot trajectories ------------------------------------
    print("\nPlotting drone trajectories...")
    plot_drone_trajectories(
        trajectories=trajectories,
        grid_size=(auto_params["N"], auto_params["M"]),
        charging_stations=auto_params["charging_stations_locations"],
        max_battery_time=auto_params["max_battery_time"],
        save_path="drone_trajectories_separate_journeys.png"
    )
    
    # --- 7. Print trajectory summary -----------------------------
    print("\nTrajectory Summary:")
    for drone_id, trajectory in trajectories.items():
        print(f"Drone {drone_id}: {len(trajectory)} total positions")
        print(f"  Start: {trajectory[0]}, End: {trajectory[-1]}")
        
        # Show journey breakdown
        journey_boundaries = [0]
        for i in range(auto_params["max_battery_time"], len(trajectory), auto_params["max_battery_time"]):
            journey_boundaries.append(i)
        journey_boundaries.append(len(trajectory))
        
        for journey_idx in range(len(journey_boundaries) - 1):
            start_idx = journey_boundaries[journey_idx]
            end_idx = journey_boundaries[journey_idx + 1]
            journey_length = end_idx - start_idx
            print(f"    Journey {journey_idx + 1}: {journey_length} steps")


    # --- 8. Generate video of trajectories overlaid on burnmap ----
    print("\nGenerating video of trajectories overlaid on burnmap...")
    
    
    # Format drone trajectories for the video function
    # The function expects a list where each element is the drone positions at that timestep
    drone_locations_history = []
    for t in range(len(trajectories[0])):  # Use length of first drone's trajectory
        timestep_positions = []
        for drone_id in trajectories.keys():
            if t < len(trajectories[drone_id]):
                timestep_positions.append(trajectories[drone_id][t])
        drone_locations_history.append(timestep_positions)
    
    # Create the video
    current_burnmap = load_burn_map(current_burnmap_filename)
    create_video_scenario_burnmap(
        burn_map=current_burnmap,
        drone_locations_history=drone_locations_history,
        out_filename="drone_trajectories_burnmap",
        ground_sensor_locations=auto_params.get("ground_sensor_locations", []),
        charging_stations_locations=auto_params["charging_stations_locations"],
        frames_per_image=2,
        maxframes=2 * auto_params["max_battery_time"] + 1
    )
    
    print("Video saved as: display_drone_trajectories_burnmap/drone_trajectories_burnmap.mp4")
    
    # Print trajectory summary instead of full data to avoid blocking IO
    for drone_id in trajectories:
        traj_length = len(trajectories[drone_id])
        if traj_length > 0:
            start_pos = trajectories[drone_id][0]
            end_pos = trajectories[drone_id][-1]
            print(f"Drone {drone_id}: {traj_length} positions, Start: ({start_pos[0]+1}, {start_pos[1]+1}), End: ({end_pos[0]+1}, {end_pos[1]+1})")
        
    # Clean up temporary file
    if os.path.exists(burnmap_file):
        os.remove(burnmap_file)


if __name__ == "__main__":
    try:
        test_top_strategy_basic()
        print("✅ Test passed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 