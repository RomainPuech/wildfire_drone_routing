# Romain Puech, 2024
# Displays
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shutil
import time

from dataset import load_scenario

def display_grid(grid, smoke_grid, drones, display):
    """
    Print the grid displaying fire, smoke, and/or drones.

    Parameters:
        grid: NxN numpy array for wildfire states (0: not burning, 1: burning, 2: burnt).
        smoke_grid: NxN numpy array for smoke concentrations.
        drones: List of Drone objects with positions.
        display: Set of options ('fire', 'smoke', 'drones') to decide what to display.
    """
    N = grid.shape[0]
    display_char = [[" " for _ in range(N)] for _ in range(N)]

    # Fire display
    if 'fire' in display:
        for i in range(N):
            for j in range(N):
                if grid[i, j] == 1:
                    display_char[i][j] = "#"  # Burning
                elif grid[i, j] == 2:
                    display_char[i][j] = "X"  # Burnt
                elif grid[i, j] == 0 and display_char[i][j] == " ":
                    display_char[i][j] = "."  # Not burning

    # Smoke display (if enabled)
    if 'smoke' in display:
        for i in range(N):
            for j in range(N):
                if smoke_grid[i, j] > 0:
                    display_char[i][j] = str(max(round(smoke_grid[i, j]),9))  # Simplified smoke concentration

    # Drones display
    if 'drones' in display:
        for drone in drones:
            x, y = drone.get_position()
            if x >= 0 and x < N and y >= 0 and y < N:
                display_char[x][y] = "D"

    # Print the grid
    for row in display_char:
        print("".join(row))
    print()

# deprecated? uses smoke_grid, which is not used anymore
def save_grid_image(grid, smoke_grid, drones, display, timestep, output_dir="images", ground_sensors_locations = [], charging_stations_locations = [], coverage_cell_width = 3):
    """
    Save a PNG image of the grid with overlays for fire, smoke, and drones, including a smoke scale.

    Parameters:
        grid: MxN numpy array for wildfire states (0: not burning, 1: burning, 2: burnt).
        smoke_grid: MxN numpy array for smoke concentrations.
        drones: List of drone locations (x,y).
        display: Set of options ('fire', 'smoke', 'drones') to decide what to overlay.
        timestep: Time step (for naming the file).
        output_dir: Directory to save the images.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use grid dimensions directly for figure size
    M, N = grid.shape
    figsize = (N/10, M/10)  # Divide by 10 to convert pixels to inches (standard dpi is 100)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)  # Set dpi explicitly to 100

    # Base grid: Smoke color or white background
    base_grid = np.ones((M, N, 3))  # Initialize as white background (R=1, G=1, B=1)
    
    if 'smoke' in display:
        # Cap smoke values at 10
        capped_smoke_grid = np.clip(smoke_grid, 0, 10)

        # Map smoke values to grayscale (custom colormap)
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_greyscale", [(0, "white"), (1, "black")], N=256
        )
        norm = mcolors.Normalize(vmin=0, vmax=10)
        
        # Convert smoke values to RGB using the custom colormap
        smoke_rgb = custom_cmap(norm(capped_smoke_grid))[:, :, :3]
        base_grid = smoke_rgb  # Default to smoke grid colors
    
    # Fire Overlay supersedes smoke if both are displayed
    if 'fire' in display:
        for i in range(M):
            for j in range(N):
                if grid[i, j] == 1:  # Burning cells
                    base_grid[i, j] = [1, 0, 0]  # Red (fire)
                elif grid[i, j] == 2 and 'smoke' not in display:  # Burnt cells (only when smoke is not displayed)
                    base_grid[i, j] = [0,0,0] # Black
    
    # Plot the combined grid
    ax.imshow(base_grid, interpolation="nearest", aspect='equal')
    
    # Drone Overlay (unaffected by fire/smoke logic)
    if 'drones' in display:
        for (y,x) in drones:
            if x >= 0 and x < N and y >= 0 and y < M:
                transformed_y = y
                ax.scatter(x, transformed_y, c="black", s=5, marker="D", label="Drone")
                for x_cov in range(x-coverage_cell_width//2, x+coverage_cell_width//2+1):
                    for y_cov in range(y-coverage_cell_width//2, y+coverage_cell_width//2+1):
                        if x_cov >= 0 and x_cov < N and y_cov >= 0 and y_cov < M:
                                transformed_y = y_cov
                                ax.scatter(x_cov, transformed_y, c="gray", alpha=0.3, s=5, marker="s")

    # add ground sensors and charging stations
    for (y,x) in ground_sensors_locations:
        if x >= 0 and x < N and y >= 0 and y < M:
            transformed_y = y
            ax.scatter(x, transformed_y, c="green", s=10, marker="s", label="Ground Sensor")
            for x_cov in range(x-coverage_cell_width//2, x+coverage_cell_width//2+1):
                for y_cov in range(y-coverage_cell_width//2, y+coverage_cell_width//2+1):
                    if x_cov >= 0 and x_cov < N and y_cov >= 0 and y_cov < M:
                        transformed_y = y_cov
                        ax.scatter(x_cov, transformed_y, c="gray", alpha=0.3, s=5, marker="s")
    
    for (y,x) in charging_stations_locations:
        if x >= 0 and x < N and y >= 0 and y < M:
            transformed_y = y
            ax.scatter(x, transformed_y, c="blue", s=10, marker="*", label="Charging Station")
            for x_cov in range(x-coverage_cell_width//2, x+coverage_cell_width//2+1):
                for y_cov in range(y-coverage_cell_width//2, y+coverage_cell_width//2+1):
                    if x_cov >= 0 and x_cov < N and y_cov >= 0 and y_cov < M:
                        transformed_y = y_cov
                        ax.scatter(x_cov, transformed_y, c="gray", alpha=0.3, s=5, marker="s")

    # Add smoke colorbar only if smoke is displayed
    if 'smoke' in display:
        # Add color bar for smoke concentration
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Smoke Concentration")
        cbar.set_ticks([0, 2, 4, 6, 8, 10])
        cbar.set_ticklabels(["0", "2", "4", "6", "8", "10+"])

    # Finalize and save the plot
    ax.axis("off")
    ax.set_title(f"Grid Visualization - Time Step {timestep}")

    image_path = os.path.join(output_dir, f"grid_timestep_{timestep:03d}.png")
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()


def save_ignition_map_image(ignition_map, timestep, output_dir="images", burn_map=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    import matplotlib.pyplot as plt
    
    N = ignition_map.shape[0]
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap from white to yellow to red
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]  # White to yellow to red
    n_bins = 100  # Number of color gradients
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Plot the heatmap
    im = plt.imshow(ignition_map, cmap=cmap, vmin=0, vmax=np.max(ignition_map))
    
    # Add colorbar with formatted labels
    label = 'Ignition Probability' if not burn_map else f"Burn Probability"
    cbar = plt.colorbar(im, label=label)
    max_val = np.max(ignition_map)
    tick_count = 5
    ticks = np.linspace(0, max_val, tick_count)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{v:.4f}' for v in ticks])
    
    # Add title and labels
    image_title = label = 'Ignition Probability Map' if not burn_map else f"Burn Probability Map at t={timestep}"
    plt.title(image_title)
    plt.xlabel('Y coordinate')
    plt.ylabel('X coordinate')
    
    

    image_path = os.path.join(output_dir, f"grid_timestep_{timestep:03d}.png")
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()



def create_video_from_images(image_dir="images", output_filename="simulation.mp4", frames_per_image=1):
    """
    Combine all images in the directory into an MP4 video.

    Parameters:
        image_dir: Directory containing the images.
        output_filename: Name of the output video file.
        frames_per_image: Number of frames to display each image (controls speed).
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    
    if not image_files:
        print("No images found to compile into a video.")
        return

    # Load the first image to determine frame size
    first_image_path = os.path.join(image_dir, image_files[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape

    # Define codec and create VideoWriter
    video_path = os.path.join(image_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30 // frames_per_image  # Frames per second adjustment
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Add each image to the video
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
       
        frame = cv2.imread(image_path)
        for _ in range(frames_per_image):
            video_writer.write(frame)

    video_writer.release()
    print(f"Video saved at: {video_path}")



def create_scenario_video(scenario_or_filename, drone_locations_history = None, burn_map = False, out_filename = "simulation", starting_time = 0, ground_sensor_locations = [], charging_stations_locations = [], substeps_per_timestep = 1, coverage_cell_width = 3, maxframes = np.inf):
    """
    Create a video visualization of a saved scenario or burn_map
    
    Args:
        scenario_or_filename: Either a filename (str) or a scenario array (numpy.ndarray)
        drone_locations_history: List of drone locations for each timestep
        burn_map: Boolean indicating if this is a burn probability map
        out_filename: Name for the output file (without extension)
        starting_time: Initial timestep
        ground_sensor_locations: List of ground sensor coordinates
        charging_stations_locations: List of charging station coordinates
        substeps_per_timestep: Number of substeps per timestep
        coverage_cell_width: Width of the coverage cell
    """
    substeps_per_timestep = 266 #TODO remove
    coverage_cell_width = 10
    # Remove .txt extension if present
    scenario = None
    if isinstance(scenario_or_filename, str):  # Using isinstance instead of type()
        # the input is a file name
        base_filename = scenario_or_filename.replace('.txt', '')  # Fixed variable name
        filename = scenario_or_filename  # Fixed variable name
    else:
        base_filename = out_filename
        scenario = scenario_or_filename
        starting_time = starting_time
    
    # Create output directory with same name as scenario file
    output_dir = 'display_' + base_filename
    if os.path.exists(output_dir):
    # Create a backup subdirectory with a timestamp
        backup_dir = os.path.join(output_dir, f"backup_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_dir, exist_ok=True)

        # Move existing files to the backup directory
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            backup_path = os.path.join(backup_dir, file)
            try:
                if os.path.isfile(file_path):
                    shutil.move(file_path, backup_path)
            except Exception as e:
                print(f"Error moving {file}: {e}")
    else:
        os.makedirs(output_dir)
    
    # Load the scenario
    if scenario is None:
        scenario, starting_time = load_scenario(filename)
    T, height, width = scenario.shape  # Using height and width instead of N
    # print("scenario.shape = ", scenario.shape)
    
    if not burn_map:
        # Create an empty smoke grid (not used but required by display function)
        smoke_grid = np.zeros((height, width))
        
        # Create images for each time step
        if drone_locations_history is not None:
            total_substeps = len(drone_locations_history)

            print("total_substeps = ", total_substeps)
            print("substeps_per_timestep = ", substeps_per_timestep)
            print("T = ", T)
            
            for t in range(min(total_substeps, maxframes)):
                scenario_index = min(t // substeps_per_timestep, T - 1)  
                save_grid_image(
                    grid=scenario[scenario_index],
                    smoke_grid=smoke_grid,
                    drones=drone_locations_history[t],
                    display={'fire', 'drones'},
                    ground_sensors_locations=ground_sensor_locations,
                    charging_stations_locations=charging_stations_locations,
                    timestep=t,
                    output_dir=output_dir,
                    coverage_cell_width=coverage_cell_width
                )
        else:
            for t in range(min(T, maxframes)):
                save_grid_image(
                    grid=scenario[t],
                    smoke_grid=smoke_grid,
                    drones=None,
                    display={'fire'},
                    ground_sensors_locations=ground_sensor_locations,
                    charging_stations_locations=charging_stations_locations,
                    timestep=t,
                    output_dir=output_dir,
                    coverage_cell_width=coverage_cell_width
                )
    else:
        # Create images for each time step
        for t in range(min(T, maxframes)):
            save_ignition_map_image(
                ignition_map=scenario[t],
                timestep=t,
                output_dir=output_dir,
                burn_map=True
            )
    
    # Create video from saved images
    create_video_from_images(
        image_dir=output_dir,
        output_filename=f"{base_filename}.mp4",
        frames_per_image=3
    )
    
    # print(f"Video saved as {base_filename}.mp4")
if __name__ == "__main__":
    # Example usage
    from benchmark import run_benchmark_scenario
    from Strategy import RandomDroneRoutingStrategy, return_no_custom_parameters, SensorPlacementOptimization, RandomSensorPlacementStrategy, LoggedOptimizationSensorPlacementStrategy,DroneRoutingOptimizationSlow, DroneRoutingOptimizationModelReuse, DroneRoutingOptimizationModelReuseIndex, LoggedDroneRoutingStrategy, LogWrapperDrone, LogWrapperSensor, DroneRoutingOptimizationModelReuseIndexRegularized
    # change values here to change benchmarking parameters
    from dataset import load_scenario_npy
    from wrappers import wrap_log_sensor_strategy, wrap_log_drone_strategy
    from new_clustering import get_wrapped_clustering_strategy

    
    scenario = load_scenario_npy("MinimalDataset/0001/scenarii/0001_00058.npy")
    def my_automatic_layout_parameters(scenario:np.ndarray,b,c):
        return {
            "N": scenario.shape[1],
            "M": scenario.shape[2],
            "max_battery_distance": -1,
            "max_battery_time": 20,
            "n_drones": 10,
            "n_ground_stations": 12,
            "n_charging_stations": 10,
            "speed_m_per_min": 10,
            "coverage_radius_m": 10,
            "cell_size_m": 40,
            "transmission_range": 100,
    }
    results, (position_history, ground, charging)  = run_benchmark_scenario(scenario, wrap_log_sensor_strategy(SensorPlacementOptimization), 
                                                                            wrap_log_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingOptimizationModelReuseIndex)), 
                                                                            custom_initialization_parameters = {"burnmap_filename": "./MinimalDataset/0001/burn_map.npy", 
                                                                                                                "load_from_logfile": False, "reevaluation_step": 6, 
                                                                                                                "optimization_horizon":6, "regularization_param": 0.0001}, 
                                                                                                                custom_step_parameters_function = return_no_custom_parameters, 
                                                                                                                automatic_initialization_parameters_function=my_automatic_layout_parameters, 
                                                                                                                return_history=True)
    # print("POSITION HISTORY", position_history, len(position_history))
    print("RESULTS", results)
    print("Ground sensors", ground)
    print("Charging stations", charging)
    
    create_scenario_video(scenario[:len(position_history)],
                          drone_locations_history=position_history,starting_time=0,
                          out_filename='test_simulation', ground_sensor_locations = ground, 
                          charging_stations_locations = charging, 
                          substeps_per_timestep= results["substeps_per_timestep"])