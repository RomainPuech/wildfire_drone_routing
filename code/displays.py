# Romain Puech, 2024
# Displays
from pathlib import Path

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
    
def plot_fire_locations(
    background_map,
    fire_rows,
    fire_cols,
    out_path,
    title="Fire ignition points",
    marker="o",
    color="black",
    marker_size=15,
):
    """
    Save a PNG showing fire ignition points overlaid on a background map.

    Parameters
    ----------
    background_map : 2-D float array (H × W), NaN outside the mask.
    fire_rows : array-like of int, row coordinates in data-space.
    fire_cols : array-like of int, col coordinates in data-space.
    out_path : str or Path, destination PNG file.
    title : str, figure title.
    marker : matplotlib marker string.
    color : marker colour.
    marker_size : scatter marker size (s parameter).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = background_map.shape
    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))

    im = ax.imshow(
        background_map,
        cmap="YlOrRd",
        origin="upper",
        interpolation="nearest",
        vmin=0, vmax=255,
        extent=[0, W, H, 0],
    )
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Avg WFPI (0–255)")

    ax.scatter(fire_cols, fire_rows, marker=marker, s=marker_size,
               color=color, edgecolors="none", alpha=0.8, zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Column (~1 km/cell)")
    ax.set_ylabel("Row (~1 km/cell)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_overview(
    background_map,
    mask,
    clusters,
    sensors_data,
    stations_data,
    detected_fires,
    missed_disc_fires,
    non_disc_fires,
    one_way_reach_opt,
    coverage_w,
    out_path,
    title="Benchmark overview",
):
    """
    Save a static overview PNG showing sensor placement, cluster zones, and
    benchmark fire outcomes on a WFPI background map.

    Parameters
    ----------
    background_map : 2-D float array (H × W), NaN outside the mask.
    mask : 2-D int/bool array (H × W).
    clusters : list of dicts with key "stations_opt" → list of (row, col) opt-coords.
    sensors_data : list of (row, col) data-space sensor centres.
    stations_data : list of (row, col) data-space station centres.
    detected_fires : DataFrame with columns 'row' and 'col' (data-space).
    missed_disc_fires : same format — fires in range but not detected.
    non_disc_fires : same format — fires outside drone range.
    one_way_reach_opt : int, one-way drone reach in opt-space cells (floor(max_battery/2)).
    coverage_w : int, opt-cell width in data cells.
    out_path : str or Path, destination PNG file.
    title : str, figure title.

    Cluster zones are drawn as the union of each station's L∞ reachable square
    (±one_way_reach_opt opt-cells).  If two stations in the same cluster have
    overlapping zones the union is shown as a single connected region.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.collections as mc
    import matplotlib.colors as mcolors

    _CLUSTER_COLOURS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]

    H, W = background_map.shape
    zone_half = one_way_reach_opt * coverage_w  # data-cell radius

    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))

    # Background map
    im = ax.imshow(
        background_map,
        cmap="YlOrRd",
        origin="upper",
        interpolation="nearest",
        vmin=0, vmax=255,
        extent=[0, W, H, 0],
    )
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Avg WFPI (0–255)")

    # ── Cluster zones: union of per-station reachable squares ─────────────────
    # Build a single fill overlay (all clusters combined) and per-cluster borders.
    fill_overlay = np.zeros((H, W, 4), dtype=float)

    for i, cl in enumerate(clusters):
        colour = _CLUSTER_COLOURS[i % len(_CLUSTER_COLOURS)]
        r_val, g_val, b_val = mcolors.to_rgb(colour)

        # Union mask for this cluster in data-space
        cl_mask = np.zeros((H, W), dtype=bool)
        for r_opt, c_opt in cl["stations_opt"]:
            r_d = r_opt * coverage_w + coverage_w // 2
            c_d = c_opt * coverage_w + coverage_w // 2
            r0 = max(0, r_d - zone_half);  r1 = min(H, r_d + zone_half + 1)
            c0 = max(0, c_d - zone_half);  c1 = min(W, c_d + zone_half + 1)
            cl_mask[r0:r1, c0:c1] = True

        # Fill: write into shared overlay (clusters are spatially disjoint)
        fill_overlay[cl_mask, 0] = r_val
        fill_overlay[cl_mask, 1] = g_val
        fill_overlay[cl_mask, 2] = b_val
        fill_overlay[cl_mask, 3] = 0.25

        # Border: find pixel-aligned edges of the union mask via diff
        m = cl_mask.astype(np.int8)

        # Horizontal edges (between row i-1 and row i → drawn at y = i)
        m_v = np.zeros((H + 2, W), dtype=np.int8)
        m_v[1:H + 1, :] = m
        diff_v = m_v[:-1, :] ^ m_v[1:, :]          # shape (H+1, W)
        ry, cx = np.where(diff_v)
        if len(ry):
            x0 = cx.astype(float)
            y0 = ry.astype(float)
            segs = np.stack(
                [np.column_stack([x0, y0]), np.column_stack([x0 + 1, y0])], axis=1)
            ax.add_collection(mc.LineCollection(
                segs, colors=colour, linewidths=1.5, zorder=4))

        # Vertical edges (between col j-1 and col j → drawn at x = j)
        m_h = np.zeros((H, W + 2), dtype=np.int8)
        m_h[:, 1:W + 1] = m
        diff_h = m_h[:, :-1] ^ m_h[:, 1:]          # shape (H, W+1)
        ry2, cx2 = np.where(diff_h)
        if len(ry2):
            x0 = cx2.astype(float)
            y0 = ry2.astype(float)
            segs = np.stack(
                [np.column_stack([x0, y0]), np.column_stack([x0, y0 + 1])], axis=1)
            ax.add_collection(mc.LineCollection(
                segs, colors=colour, linewidths=1.5, zorder=4))

    ax.imshow(fill_overlay, origin="upper", extent=[0, W, H, 0],
              zorder=3, interpolation="nearest")

    # ── Fire markers ──────────────────────────────────────────────────────────
    def get_rc(obj):
        """Extract (rows, cols) from a DataFrame or list of (row, col) tuples."""
        if hasattr(obj, "iterrows"):
            return obj["row"].tolist(), obj["col"].tolist()
        return [r for r, _ in obj], [c for _, c in obj]

    nd_r, nd_c = get_rc(non_disc_fires)
    md_r, md_c = get_rc(missed_disc_fires)
    dt_r, dt_c = get_rc(detected_fires)

    if nd_r:
        ax.scatter(nd_c, nd_r, marker=".", s=20, color="gray",
                   alpha=0.7, zorder=5,
                   label=f"Non-discoverable (n={len(nd_r)})")
    if md_r:
        ax.scatter(md_c, md_r, marker="x", s=50, color="black",
                   linewidths=1.5, alpha=0.9, zorder=6,
                   label=f"Missed discoverable (n={len(md_r)})")
    if dt_r:
        ax.scatter(dt_c, dt_r, marker="o", s=80, color="limegreen",
                   edgecolors="darkgreen", linewidths=0.8, zorder=8,
                   label=f"Detected (n={len(dt_r)}, {len(dt_r)}%)")

    # ── Sensors (white stars) and stations (blue triangles) ───────────────────
    if sensors_data:
        sr, sc = zip(*sensors_data)
        ax.scatter(sc, sr, marker="*", s=40, color="white",
                   edgecolors="black", linewidths=0.5, zorder=7,
                   label=f"Ground sensor (n={len(sensors_data)})")
    if stations_data:
        tr, tc = zip(*stations_data)
        ax.scatter(tc, tr, marker="^", s=50, color="blue",
                   edgecolors="white", linewidths=0.8, zorder=7,
                   label=f"Charging station (n={len(stations_data)})")

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.85)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Column (~1 km/cell)")
    ax.set_ylabel("Row (~1 km/cell)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    vmax=None,
    display_zones=False,
    fire_scenario=None,
    substeps_per_timestep=1,
    mask=None,
    coverage_width_cells=None,
    mask_pooling_mode="min"
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
        display_zones: If True, display squares of length 60 centered on charging stations (default: False)
        fire_scenario: Optional T_data x N_data x M_data scenario array (data scale). Fire cells (>0.5) are shown as purple crosses.
        substeps_per_timestep: Number of operational substeps per data timestep (used to map video frames to scenario time)
        mask: Optional N_data x M_data mask array (data scale). Cells with mask==0 are shown as gray.
        coverage_width_cells: Kernel size used in the simulation for pooling data→operational scale.
            Must match the value used in the benchmark (round(2*coverage_radius_m/cell_size_m)).
            If None, falls back to N_data//N_op (which may differ and cause visual discrepancies).
        mask_pooling_mode: "min" (masked if any data cell masked) or "max" (masked only if all data cells masked).
            Must match the mode used in the benchmark simulation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
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

    # Pre-compute pooled mask (operational scale) if provided
    # Use the same kernel size and pooling mode as the simulation to avoid discrepancies
    mask_ops = None
    if mask is not None:
        cwc = coverage_width_cells if coverage_width_cells is not None else max(1, mask.shape[0] // N)
        pool_fn = np.min if mask_pooling_mode == "min" else np.max
        mask_ops = np.ones((N, M))
        for mi in range(N):
            for mj in range(M):
                mask_ops[mi, mj] = pool_fn(mask[mi*cwc:(mi+1)*cwc, mj*cwc:(mj+1)*cwc])

    T = min(T, len(drone_locations_history))

    for t in range(min(T, maxframes)):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        im = ax.imshow(burn_map[t].T, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), alpha=1.0, origin='lower')

        # Gray overlay on masked (inaccessible) cells
        if mask_ops is not None:
            gray_overlay = np.full((N, M, 4), 0.0)  # RGBA
            masked_cells = mask_ops == 0
            gray_overlay[masked_cells] = [0.5, 0.5, 0.5, 0.8]  # gray with high opacity
            ax.imshow(gray_overlay.transpose(1, 0, 2), origin='lower')

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
            
            # Display zones (squares of length 60 centered on charging stations)
            if display_zones:
                zone_length = 60
                half_length = zone_length / 2
                for i, (x, y) in enumerate(charging_stations_locations):
                    # Calculate square corners (centered on charging station)
                    x_min = x - half_length
                    y_min = y - half_length
                    rect = Rectangle(
                        (x_min, y_min), 
                        zone_length, 
                        zone_length,
                        linewidth=2,
                        edgecolor='cyan',
                        facecolor='none',
                        linestyle='--',
                        label="Drone Zone" if t == 0 and i == 0 else None
                    )
                    ax.add_patch(rect)

        # Fire cells from scenario (purple crosses)
        if fire_scenario is not None:
            scenario_index = min(t // substeps_per_timestep, len(fire_scenario) - 1)
            if scenario_index >= 0:
                fire_grid = fire_scenario[scenario_index]
                # Pool fire grid from data scale to operational scale (max pooling)
                # Use the same kernel size as the simulation
                N_data, M_data = fire_grid.shape
                cwc_fire = coverage_width_cells if coverage_width_cells is not None else max(1, N_data // N)
                fire_ops = np.zeros((N, M))
                for fi in range(N):
                    for fj in range(M):
                        fire_ops[fi, fj] = np.max(fire_grid[fi*cwc_fire:(fi+1)*cwc_fire, fj*cwc_fire:(fj+1)*cwc_fire])
                fire_cells = np.argwhere(fire_ops > 0.5)
                if len(fire_cells) > 0:
                    ax.scatter(fire_cells[:, 0], fire_cells[:, 1],
                               c="purple", s=200, marker="x", linewidths=3,
                               label="Fire" if t == 0 else None)

        ax.set_title(f"Burn Probability Map at t={t}")
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        plt.colorbar(im, ax=ax, label="Burn Probability")
        ax.axis("on")
        
        # Set consistent limits to ensure same image size
        # burn_map[t].T has shape (M, N): M rows (y-axis) and N cols (x-axis)
        ax.set_xlim(-0.5, N-0.5)
        ax.set_ylim(-0.5, M-0.5)
        
        image_path = os.path.join(output_dir, f"grid_timestep_{t:03d}.png")
        plt.savefig(image_path, bbox_inches='tight', dpi=100)
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


def _pyrologix_imshow_01_bottom_left(ax, z_hw, valid_mask_hw, invalid_rgba=(1.0, 1.0, 1.0, 1.0)):
    """
    Pyrologix heatmap: values scaled to [0, 1], y-axis upward (origin bottom-left
    visually; array row 0 = north remains at top of map via flipud).
    Returns (im, H, W).
    """
    import matplotlib.pyplot as plt

    z = np.asarray(z_hw, dtype=np.float32)
    H, W = z.shape
    m = np.asarray(valid_mask_hw) == 1
    z01 = np.clip(z / 255.0, 0.0, 1.0)
    display = np.where(m, z01, np.nan)
    display = np.flipud(display)

    try:
        cmap = plt.colormaps["YlOrRd"].copy()
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("YlOrRd").copy()
    cmap.set_bad(color=invalid_rgba)

    im = ax.imshow(
        display,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        extent=[0, W, 0, H],
        aspect="equal",
    )
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks([])
    ax.set_yticks([])
    for _sp in ax.spines.values():
        _sp.set_visible(False)
    return im, H, W


def _scatter_fire_layers_bottom_left(ax, fire_layers, H):
    """Scatter using original grid (row from top); maps to y-up coordinates."""
    for layer in fire_layers or []:
        r = np.asarray(layer["rows"], dtype=int)
        c = np.asarray(layer["cols"], dtype=int)
        if r.size == 0:
            continue
        y = (H - 1) - r
        if layer.get("include_in_legend", True):
            label = layer.get("label", "")
        else:
            label = "_nolegend_"
        kw = {
            "c": layer.get("color", "black"),
            "marker": layer.get("marker", "o"),
            "s": layer.get("s", 25),
            "alpha": layer.get("alpha", 0.9),
            "zorder": layer.get("zorder", 5),
            "label": label,
        }
        if "edgecolors" in layer:
            kw["edgecolors"] = layer["edgecolors"]
            kw["linewidths"] = layer.get("linewidths", 0.4)
        ax.scatter(c, y, **kw)


def make_usfs_fire_legend_handles(
    n_urb,
    n_off,
    n_ds,
    *,
    include_off_mask=True,
    benchmark_label=None,
    urban_color="#0d9488",
):
    """
    Proxy legend entries matching USFS explainer fire symbology (fig05 style).

    Parameters
    ----------
    include_off_mask : bool
        If False, omit the "Fire in unburnable area" row (e.g. before WFPI invalid cells are applied).
    benchmark_label : str or None
        If set, append a final row (black disk) for the benchmark subsample.
    urban_color : str
        Face/edge color for urban triangle markers (matches scatter layers).
    """
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="^",
            color=urban_color,
            markersize=9,
            label=f"Urban fire (n={n_urb})",
        ),
    ]
    if include_off_mask:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                color="#888888",
                markersize=6,
                alpha=0.75,
                label=f"Fire in unburnable area (n={n_off})",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            color="#0d0d0d",
            markersize=7,
            label=f"Fire in dataset (n={n_ds})",
        )
    )
    if benchmark_label:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                color="#0d0d0d",
                markersize=9,
                label=benchmark_label,
            )
        )
    return handles


_LM_OTF_CANDIDATES = (
    "/Library/TeX/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    str(Path.home() / "texmf/fonts/opentype/public/lm/lmroman10-regular.otf"),
)

_LM_REGISTERED = False


def _register_latin_modern_for_matplotlib() -> None:
    """
    Register Latin Modern Roman OTF if present (same paths as
    ``visualize_sensor_placement_2021.py`` / Nature Figure~4 cluster maps).
    """
    global _LM_REGISTERED
    if _LM_REGISTERED:
        return
    import matplotlib.font_manager as fm

    for path in _LM_OTF_CANDIDATES:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            fm.fontManager.addfont(str(p))
        except (OSError, ValueError, RuntimeError):
            continue
        _LM_REGISTERED = True
        return
    _LM_REGISTERED = True


def _pyrologix_publication_rc():
    """
    Match ``visualize_sensor_placement_2021.py`` (Nature Figure~4 placement maps):
    Latin Modern when TeX LM is installed, else CMU/DejaVu serif; Computer Modern math text.
    """
    _register_latin_modern_for_matplotlib()
    return {
        "font.family": "serif",
        "font.serif": [
            "Latin Modern Roman",
            "Latin Modern",
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }


def _colorbar_inset_top_right(
    fig,
    ax,
    im,
    label="Ignition probability (0–1)",
    *,
    tick_fontsize=13,
    label_fontsize=14,
):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # borderpad in points: keep colorbar + label inside axes (away from figure edge)
    cax = inset_axes(
        ax,
        width="6.0%",
        height="36%",
        loc="upper right",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=3.2,
    )
    cb = fig.colorbar(im, cax=cax)
    cax.set_facecolor("white")
    for spine in cax.spines.values():
        spine.set_visible(False)
    cb.ax.tick_params(labelsize=tick_fontsize, pad=3)
    cb.set_label(label, fontsize=label_fontsize, labelpad=10)
    cb.outline.set_visible(False)
    return cb


def _ca_boundary_segments_plot_xy(ca_gdf_wfpi, cropped_affine, H, W):
    """
    California boundary as polylines in Pyrologix panel coordinates (x=col, y=imshow-up).
    """
    import rasterio.transform
    from shapely.geometry import box as shapely_box

    if ca_gdf_wfpi is None or cropped_affine is None:
        return []
    left, bottom, right, top = rasterio.transform.array_bounds(H, W, cropped_affine)
    crop_box = shapely_box(left, bottom, right, top)
    geom = ca_gdf_wfpi.geometry.iloc[0]
    if geom is None or geom.is_empty:
        return []
    clipped = geom.boundary.intersection(crop_box)
    if clipped.is_empty:
        return []

    segs = []

    def append_linestring(line):
        coords = np.asarray(line.coords, dtype=np.float64)
        if coords.shape[0] < 2:
            return
        cols, yps = [], []
        for x, y in coords:
            r, c = rasterio.transform.rowcol(cropped_affine, float(x), float(y))
            cols.append(float(c))
            yps.append(float((H - 1) - r))
        segs.append(np.column_stack([cols, yps]))

    g = clipped
    if g.geom_type == "LineString":
        append_linestring(g)
    elif g.geom_type == "MultiLineString":
        for line in g.geoms:
            append_linestring(line)
    elif g.geom_type == "GeometryCollection":
        for sub in g.geoms:
            if sub.geom_type == "LineString":
                append_linestring(sub)
            elif sub.geom_type == "MultiLineString":
                for line in sub.geoms:
                    append_linestring(line)
    return segs


def _legend_entries_to_handles_labels(entries):
    """
    Convert legend entries to (handles, labels, handler_map).

    Each entry is either a matplotlib Artist (label from get_label()) or a
    3-tuple (line_dot, line_x, label) for a single combined row (HandlerTuple).
    """
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    handles = []
    labels = []
    need_tuple_handler = False
    for e in entries:
        if (
            isinstance(e, tuple)
            and len(e) == 3
            and isinstance(e[0], Line2D)
            and isinstance(e[1], Line2D)
            and isinstance(e[2], str)
        ):
            handles.append((e[0], e[1]))
            labels.append(e[2])
            need_tuple_handler = True
        else:
            handles.append(e)
            labels.append(e.get_label())
    handler_map = {tuple: HandlerTuple(ndivide=None)} if need_tuple_handler else None
    return handles, labels, handler_map


def _ncol_nrows_for_below_legend(nh: int) -> tuple[int, int]:
    """Prefer one row (up to 4 items); wider figures get an extra row if needed."""
    if nh <= 1:
        return 1, nh
    if nh <= 4:
        return nh, 1
    ncol = min(4, nh)
    nrows = (nh + ncol - 1) // ncol
    return ncol, nrows


def _pyrologix_legend_below_map(
    fig,
    ax,
    entries,
    *,
    legend_fontsize: float,
    framed: bool,
) -> None:
    """Place legend under the map (axes coords). ``entries`` are Artists or (h1, h2, label) tuples."""
    handles, labels, handler_map = _legend_entries_to_handles_labels(entries)
    nh = len(handles)
    ncol, nrows = _ncol_nrows_for_below_legend(nh)
    bottom = min(0.30, 0.13 + 0.020 * max(0, nrows - 1))
    fig.subplots_adjust(bottom=bottom)
    common = dict(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.078),
        bbox_transform=ax.transAxes,
        ncol=ncol,
        columnspacing=0.85,
        borderaxespad=0.0,
        labelspacing=0.32,
        handletextpad=0.42,
        handlelength=1.35,
        prop={"size": legend_fontsize, "weight": "bold"},
    )
    if handler_map is not None:
        common["handler_map"] = handler_map
    if framed:
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="0.75",
            framealpha=1.0,
            **common,
        )
    else:
        ax.legend(frameon=False, **common)


def plot_pyrologix_valid_region(
    pyrologix_hw,
    valid_mask_hw,
    out_path,
    title=None,
    fire_layers=None,
    legend_handles=None,
    legend_loc="lower right",  # unused: legend is placed below the map (Nature Figure 2)
    invalid_rgb=(1.0, 1.0, 1.0),
    colorbar_label="Ignition probability (0–1)",
    dpi=150,
    *,
    show_colorbar=True,
    ca_boundary_gdf_wfpi=None,
    cropped_affine=None,
    legend_fontsize=11,
    title_fontsize=11,
):
    """
    Static risk map (Pyrologix on the WFPI grid): colormap only where valid_mask == 1;
    other cells are white. y increases upward; risk in [0, 1]. Optional colorbar inset.

    fire_layers: optional list of scatter dicts (same as plot_pyrologix_fire_categories).
    legend_handles: optional list of Artist handles (e.g. from make_usfs_fire_legend_handles).
    ca_boundary_gdf_wfpi + cropped_affine: if both set, draw CA state outline (WFPI CRS).
    ``legend_loc`` is accepted for backward compatibility; the legend is placed below the map.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    invalid_rgba = invalid_rgb + (1.0,) if len(invalid_rgb) == 3 else invalid_rgb

    H, W = np.asarray(pyrologix_hw).shape
    aspect = W / H
    with plt.rc_context(_pyrologix_publication_rc()):
        fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        im, H, W = _pyrologix_imshow_01_bottom_left(
            ax, pyrologix_hw, valid_mask_hw, invalid_rgba
        )
        segs = _ca_boundary_segments_plot_xy(
            ca_boundary_gdf_wfpi, cropped_affine, H, W
        )
        if segs:
            lc = LineCollection(
                segs,
                colors="#444444",
                linewidths=1.0,
                zorder=6,
                capstyle="round",
            )
            ax.add_collection(lc)
        _scatter_fire_layers_bottom_left(ax, fire_layers, H)
        if show_colorbar:
            _colorbar_inset_top_right(fig, ax, im, label=colorbar_label)

        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
        if legend_handles:
            _pyrologix_legend_below_map(
                fig,
                ax,
                legend_handles,
                legend_fontsize=legend_fontsize,
                framed=False,
            )

        fig.savefig(
            str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.2, facecolor="white"
        )
        plt.close(fig)


def plot_pyrologix_fire_categories(
    pyrologix_hw,
    valid_mask_hw,
    fire_layers,
    out_path,
    title=None,
    colorbar_label="Ignition probability (0–1)",
    dpi=150,
    legend_loc="lower right",  # unused: legend is placed below the map
    legend_handles=None,
    *,
    legend_fontsize=11,
):
    """
    Pyrologix background (masked to valid_mask) with multiple fire scatter groups.
    y increases upward; risk [0–1]; colorbar inset top-right; no axis labels by default.

    fire_layers: list of dicts with keys:
      rows, cols (arrays), color, marker, s, optional include_in_legend, label, edgecolors, linewidths, alpha, zorder
    legend_handles: if provided, use these for ax.legend instead of scatter labels.
    Legend is drawn below the map to avoid overlapping markers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = np.asarray(pyrologix_hw).shape
    aspect = W / H
    fig, ax = plt.subplots(figsize=(aspect * 11 + 2, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im, H, W = _pyrologix_imshow_01_bottom_left(
        ax, pyrologix_hw, valid_mask_hw, invalid_rgba=(1.0, 1.0, 1.0, 1.0)
    )
    _scatter_fire_layers_bottom_left(ax, fire_layers, H)
    _colorbar_inset_top_right(fig, ax, im, label=colorbar_label)

    if legend_handles is not None:
        _pyrologix_legend_below_map(
            fig,
            ax,
            legend_handles,
            legend_fontsize=legend_fontsize,
            framed=True,
        )
    else:
        handles, labels = ax.get_legend_handles_labels()
        if labels and any(lab for lab in labels):
            _pyrologix_legend_below_map(
                fig,
                ax,
                handles,
                legend_fontsize=legend_fontsize,
                framed=True,
            )

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.savefig(
        str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.2, facecolor="white"
    )
    plt.close(fig)