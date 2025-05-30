# Romain Puech, 2024
# Displays
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import shutil
import time
import imageio

from dataset import load_scenario

def display_grid(grid, smoke_grid, drones, display):
    """
    Print the grid displaying fire, and/or drones.

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

def save_grid_image(grid, smoke_grid, drones, display, timestep, output_dir="images", ground_sensors_locations = [], charging_stations_locations = [], coverage_cell_width = 3, burn_map_background = None):
    """
    Save a PNG image of the grid with overlays for fire, smoke, and drones, including a smoke scale.

    Parameters:
        grid: MxN numpy array for wildfire states (0: not burning, 1: burning, 2: burnt).
        smoke_grid: MxN numpy array for smoke concentrations.
        drones: List of drone locations (x,y).
        display: Set of options ('fire', 'smoke', 'drones') to decide what to overlay.
        timestep: Time step (for naming the file).
        output_dir: Directory to save the images.
        burn_map_background: Optional burn probability map to use as background.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use grid dimensions directly for figure size
    M, N = grid.shape
    figsize = (N/10, M/10)  # Divide by 10 to convert pixels to inches (standard dpi is 100)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=100)  # Set dpi explicitly to 100

    # Base grid: Smoke color or white background
    base_grid = np.ones((M, N, 3))  # Initialize as white background (R=1, G=1, B=1)
    
    # If burn map is provided, use it as background
    if burn_map_background is not None:
        # Create custom colormap from white to yellow to red
        colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]  # White to yellow to red
        n_bins = 100  # Number of color gradients
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        
        # Get the burn map for the current timestep
        current_burn_map = burn_map_background[timestep] if len(burn_map_background.shape) == 3 else burn_map_background
        
        # Plot the burn map background
        im = ax.imshow(current_burn_map, cmap=cmap, vmin=0, vmax=np.max(current_burn_map), alpha=0.7)
        
        # Add colorbar with formatted labels
        cbar = plt.colorbar(im, label="Burn Probability")
        max_val = np.max(current_burn_map)
        tick_count = 5
        ticks = np.linspace(0, max_val, tick_count)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{v:.4f}' for v in ticks])
    
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

    # Finalize and save the plot
    ax.axis("off")
    ax.set_title(f"Grid Visualization - Time Step {timestep}")

    image_path = os.path.join(output_dir, f"grid_timestep_{timestep:03d}.png")
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()


def save_ignition_map_image(ignition_map, timestep, output_dir="images", is_burn_map=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    N = ignition_map.shape[0]
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap from white to yellow to red
    colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]  # White to yellow to red
    n_bins = 100  # Number of color gradients
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Plot the heatmap
    im = plt.imshow(ignition_map, cmap=cmap, vmin=0, vmax=np.max(ignition_map))
    
    # Add colorbar with formatted labels
    label = 'Ignition Probability' if not is_burn_map else f"Burn Probability"
    cbar = plt.colorbar(im, label=label)
    max_val = np.max(ignition_map)
    tick_count = 5
    ticks = np.linspace(0, max_val, tick_count)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{v:.4f}' for v in ticks])
    
    # Add title and labels
    image_title = label = 'Ignition Probability Map' if not is_burn_map else f"Burn Probability Map at t={timestep}"
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



def create_scenario_video(scenario_or_filename, drone_locations_history = None, is_burn_map = False, out_filename = "simulation", starting_time = 0, ground_sensor_locations = [], charging_stations_locations = [], substeps_per_timestep = 1, coverage_cell_width = 3, maxframes = np.inf, burn_map_background = None):
    """
    Create a video visualization of a saved scenario or burn_map
    
    Args:
        scenario_or_filename: Either a filename (str) or a scenario array (numpy.ndarray)
        drone_locations_history: List of drone locations for each timestep
        is_burn_map: Boolean indicating if this is a burn probability map
        out_filename: Name for the output file (without extension)
        starting_time: Initial timestep
        ground_sensor_locations: List of ground sensor coordinates
        charging_stations_locations: List of charging station coordinates
        substeps_per_timestep: Number of substeps per timestep
        coverage_cell_width: Width of the coverage cell
        burn_map_background: Optional burn probability map to use as background
    """
    # Remove .txt extension if present
    scenario = None
    if isinstance(scenario_or_filename, str):  # Using isinstance instead of type()
        # the input is a file name
        base_filename = scenario_or_filename.replace('.txt', '')  # Fixed variable name
        filename = scenario_or_filename  # Fixed variable name
    else:
        base_filename = out_filename
        scenario = scenario_or_filename
    
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
        scenario = load_scenario(filename)
    T, height, width = scenario.shape
    # print("scenario.shape = ", scenario.shape)

    # if starting_time not zero, prepend empty grids to the scenario
    if starting_time != 0:
        scenario = np.concatenate([np.zeros((starting_time, height, width)), scenario], axis=0)
        T = scenario.shape[0]
    
    if not is_burn_map:
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
                    coverage_cell_width=coverage_cell_width,
                    burn_map_background=burn_map_background
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
                    coverage_cell_width=coverage_cell_width,
                    burn_map_background=burn_map_background
                )
    else:
        # Create images for each time step
        for t in range(min(T, maxframes)):
            save_ignition_map_image(
                ignition_map=scenario[t],
                timestep=t,
                output_dir=output_dir,
                is_burn_map=True
            )
    
    # Create video from saved images
    create_video_from_images(
        image_dir=output_dir,
        output_filename=f"{base_filename}.mp4",
        frames_per_image=3
    )
    
def create_video_scenario_burnmap(
    burn_map,
    drone_locations_history=None,
    out_filename="simulation_burnmap",
    ground_sensor_locations=[],
    charging_stations_locations=[],
    frames_per_image=3,
    maxframes=np.inf,
    cmap=None,
    vmin=None,
    vmax=None
):
    """
    Create a video visualization of a burn map with drones, sensors, and charging stations overlaid.

    Args:
        burn_map: TxNxM numpy array representing the burn probability map
        drone_locations_history: List of drone locations for each timestep
        out_filename: Name for the output file (without extension)
        ground_sensor_locations: List of ground sensor coordinates
        charging_stations_locations: List of charging station coordinates
        frames_per_image: Number of frames to display each image (controls speed)
        maxframes: Maximum number of frames to render
        cmap: Colormap to use (optional)
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)
    """
    import matplotlib.pyplot as plt
    import os
    from matplotlib.colors import LinearSegmentedColormap
    import cv2

    T, N, M = burn_map.shape
    output_dir = f"display_{out_filename}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cmap is None:
        colors = [(1, 1, 1), (1, 1, 0), (1, 0, 0)]  # White to yellow to red
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    if vmin is None:
        vmin = max(1e-9, np.min(burn_map))
    if vmax is None:
        vmax = np.max(burn_map)
        if vmax <= 1e-9:
            vmax = 1e-5

    T = min(T, len(drone_locations_history))

    for t in range(min(T, maxframes)):
        fig, ax = plt.subplots(figsize=(M/10, N/10), dpi=100)
        im = ax.imshow(burn_map[t].T, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), alpha=1.0, origin='lower')

        # Drones
        if drone_locations_history is not None and t < len(drone_locations_history):
            for drone in drone_locations_history[t]:
                ax.scatter(drone[0], drone[1], c="black", s=30, marker="D", label="Drone" if t == 0 else None)

        # Ground sensors
        if ground_sensor_locations:
            ax.scatter([xy[0] for xy in ground_sensor_locations], [xy[1] for xy in ground_sensor_locations],
                       c="green", s=30, marker="s", label="Ground Sensor" if t == 0 else None)
        # Charging stations
        if charging_stations_locations:
            ax.scatter([xy[0] for xy in charging_stations_locations], [xy[1] for xy in charging_stations_locations],
                       c="blue", s=30, marker="*", label="Charging Station" if t == 0 else None)

        ax.set_title(f"Burn Probability Map at t={t}")
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        plt.colorbar(im, ax=ax, label="Burn Probability")
        ax.axis("on")
        image_path = os.path.join(output_dir, f"grid_timestep_{t:03d}.png")
        plt.savefig(image_path, bbox_inches=None)
        plt.close()

    # Create video from saved images
    image_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
    print(len(image_files))
    if not image_files:
        print("No images found to compile into a video.")
        return
    output_path = os.path.join(output_dir, f"{out_filename}.mp4")
    writer = imageio.get_writer(output_path, fps=10)
    for image_file in image_files:
        image_path = os.path.join(output_dir, image_file)
        img = imageio.imread(image_path)
        for _ in range(frames_per_image):
            writer.append_data(img)
    writer.close()
    print(f"Video saved at: {output_path}")