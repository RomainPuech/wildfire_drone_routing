# Romain Puech, 2024
# Displays
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


def save_grid_image(grid, smoke_grid, drones, display, timestep, output_dir="images"):
    """
    Save a PNG image of the grid with overlays for fire, smoke, and drones, including a smoke scale.

    Parameters:
        grid: NxN numpy array for wildfire states (0: not burning, 1: burning, 2: burnt).
        smoke_grid: NxN numpy array for smoke concentrations.
        drones: List of Drone objects with positions.
        display: Set of options ('fire', 'smoke', 'drones') to decide what to overlay.
        timestep: Time step (for naming the file).
        output_dir: Directory to save the images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(6, 6))
    N = grid.shape[0]

    # Base grid: Smoke color or white background
    base_grid = np.ones((N, N, 3))  # Initialize as white background (R=1, G=1, B=1)
    
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
        for i in range(N):
            for j in range(N):
                if grid[i, j] == 1:  # Burning cells
                    base_grid[i, j] = [1, 0, 0]  # Red (fire)
                elif grid[i, j] == 2 and 'smoke' not in display:  # Burnt cells (only when smoke is not displayed)
                    base_grid[i, j] = [0.5, 0.5, 0.5]  # Gray
    
    # Plot the combined grid
    ax.imshow(base_grid, interpolation="nearest")

    # Drone Overlay (unaffected by fire/smoke logic)
    if 'drones' in display:
        for drone in drones:
            x, y = drone.get_position()
            if x >= 0 and x < N and y >= 0 and y < N:
                if drone.fire_alert:
                    ax.scatter(y, x, c="green", s=30, marker="D", label="Drone")
                elif drone.smoke_alert:
                    ax.scatter(y, x, c="blue", s=30, marker="D", label="Drone")
                else:
                    ax.scatter(y, x, c="black", s=30, marker="D", label="Drone")


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


