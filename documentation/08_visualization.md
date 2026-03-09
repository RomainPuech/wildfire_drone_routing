# Visualization

This document describes the visualization tools for inspecting simulation outputs: static grid images, animated scenario videos, burn-map videos, and helper utilities to replay logged runs.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Text-Based Grid Display](#2-text-based-grid-display)
3. [Static Grid Images](#3-static-grid-images)
4. [Ignition / Burn Map Images](#4-ignition--burn-map-images)
5. [Scenario Video Generation](#5-scenario-video-generation)
6. [Burn-Map Video Generation](#6-burn-map-video-generation)
7. [Video Compilation](#7-video-compilation)
8. [Video Helpers — Replaying Logged Runs](#8-video-helpers--replaying-logged-runs)
9. [Visual Encoding Reference](#9-visual-encoding-reference)

---

## 1. Overview

Two modules handle visualization:

| Module | Purpose |
|--------|---------|
| `displays.py` | Core rendering: images, videos, overlays |
| `video_helpers.py` | Replaying logged runs from JSON files |

The typical workflow is:

```
Simulation completes
  └── run_benchmark_scenario(return_history=True)
        ├── drone_locations_history  ─┐
        ├── ground_sensor_locations   │
        └── charging_station_locs     │
                                      ▼
                          create_scenario_video()      ← fire + drones
                          create_video_scenario_burnmap() ← burn map + drones
```

For post-hoc replays (from cached logs):

```
Logged JSON files (from wrappers)
  └── generate_video_from_latest_run()
        ├── load_drone_positions_from_log()
        ├── load burn map from tmp_burnmaps/
        └── create_video_scenario_burnmap()
```

---

## 2. Text-Based Grid Display

### `display_grid(grid, smoke_grid, drones, display)`

A lightweight ASCII renderer for terminal-based debugging. Renders the grid as text characters:

| Symbol | Meaning |
|--------|---------|
| `#` | Burning cell |
| `X` | Burnt cell |
| `.` | Unburnt cell |
| `D` | Drone position |
| ` ` | Empty / not displayed |

**Parameters**:
- `grid`: `N×N` NumPy array with fire states (0 = unburnt, 1 = burning, 2 = burnt)
- `smoke_grid`: Smoke concentration array (currently unused in the character rendering)
- `drones`: List of `Drone` objects
- `display`: Set of options — `{'fire', 'smoke', 'drones'}`

This function prints directly to stdout and is useful for quick interactive debugging without matplotlib.

---

## 3. Static Grid Images

### `save_grid_image(grid, smoke_grid, drones, display, timestep, ...)`

Renders a single simulation frame as a high-resolution PNG image using matplotlib.

```
save_grid_image(
    grid,                          # MxN fire state array
    smoke_grid,                    # MxN smoke array (background shading)
    drones,                        # list of (row, col) tuples
    display,                       # set: {'fire', 'smoke', 'drones'}
    timestep,                      # int — used for file naming
    output_dir="images",           # output folder
    ground_sensors_locations=[],   # list of (row, col) for sensors
    charging_stations_locations=[], # list of (row, col) for charging stations
    coverage_cell_width=3,         # coverage square side length
    burn_map_background=None       # optional TxMxN or MxN burn probability
)
```

### Rendering Pipeline

```
┌──────────────────────────────────────────────────────┐
│ 1. Base Layer                                        │
│    • White background (MxN RGB array, all ones)      │
│    • If burn_map_background provided:                │
│      imshow() with white→yellow→red colormap + cbar  │
├──────────────────────────────────────────────────────┤
│ 2. Fire Overlay                                      │
│    • Burning cells (state == 1) → Red [1,0,0]        │
│    • Burnt cells (state == 2)   → Black [0,0,0]      │
│      (only if 'smoke' not in display)                │
├──────────────────────────────────────────────────────┤
│ 3. imshow() of the combined RGB grid                 │
├──────────────────────────────────────────────────────┤
│ 4. Drone Overlay (scatter)                           │
│    • Black diamond marker at drone position          │
│    • Gray coverage squares around drone              │
├──────────────────────────────────────────────────────┤
│ 5. Sensor Overlay (scatter)                          │
│    • Green squares for ground sensors                │
│    • Blue stars for charging stations                │
│    • Gray coverage squares around each               │
├──────────────────────────────────────────────────────┤
│ 6. Save as grid_timestep_XXX.png                     │
└──────────────────────────────────────────────────────┘
```

### Image Sizing

The figure size is derived directly from the grid dimensions:

```python
M, N = grid.shape
figsize = (N / 10, M / 10)  # 100 DPI → 1 pixel per cell
```

This ensures a 1:1 pixel-to-cell mapping at 100 DPI, keeping images crisp regardless of grid resolution.

### Coverage Visualization

For each drone, ground sensor, and charging station, a semi-transparent gray square of `coverage_cell_width × coverage_cell_width` is drawn around the device location. This visually communicates the detection area.

---

## 4. Ignition / Burn Map Images

### `save_ignition_map_image(ignition_map, timestep, output_dir, is_burn_map)`

Renders a heatmap of ignition or burn probability for a single timestep.

**Colormap**: White → Yellow → Red (100-bin `LinearSegmentedColormap`)

**Colorbar**: 5 evenly-spaced ticks from 0 to `max(ignition_map)`, formatted to 4 decimal places.

**Title**: Adapts based on `is_burn_map`:
- `True` → "Burn Probability Map at t={timestep}"
- `False` → "Ignition Probability Map"

Output follows the same naming convention: `grid_timestep_XXX.png`.

---

## 5. Scenario Video Generation

### `create_scenario_video(scenario_or_filename, ...)`

The main video creation function for fire scenario visualizations. Accepts either a filename or a NumPy array directly.

```python
create_scenario_video(
    scenario_or_filename,          # str (filename) or ndarray (TxMxN)
    drone_locations_history=None,  # list of lists of (row,col) per substep
    is_burn_map=False,             # use burn map renderer instead
    out_filename="simulation",     # output name prefix
    starting_time=0,               # pre-fire patrol timesteps
    ground_sensor_locations=[],
    charging_stations_locations=[],
    substeps_per_timestep=1,       # operational substeps per data step
    coverage_cell_width=3,
    maxframes=np.inf,              # cap on number of frames
    burn_map_background=None       # optional background layer
)
```

### Process

```
1. Resolve input
   ├── String → load_scenario(filename) → TxMxN array
   └── ndarray → use directly

2. Pre-fire padding
   └── if starting_time > 0:
         prepend starting_time × MxN zero grids

3. Create output directory (display_{name}/)
   └── backup existing files with timestamp

4. Generate frames
   ├── if is_burn_map → save_ignition_map_image() per timestep
   └── else → save_grid_image() per substep
         └── scenario_index = t // substeps_per_timestep
             (maps substep index to correct fire grid frame)

5. Compile video
   └── create_video_from_images(output_dir, ..., frames_per_image=3)
```

### Substep-to-Timestep Mapping

When drones move at operational substeps (multiple moves per data timestep), the fire grid only advances at data timestep boundaries:

```
substep t    → scenario_index = min(t // substeps_per_timestep, T-1)
```

For example, with `substeps_per_timestep = 5`:
- Substeps 0–4 → fire grid at t=0
- Substeps 5–9 → fire grid at t=1
- etc.

This keeps the fire visually static between substeps while drones move smoothly.

---

## 6. Burn-Map Video Generation

### `create_video_scenario_burnmap(burn_map, ...)`

A specialized video renderer for burn probability maps, using a **log-scale colormap** for better visibility of low-probability regions.

```python
create_video_scenario_burnmap(
    burn_map,                      # TxNxM burn probability array
    drone_locations_history=None,
    out_filename="simulation_burnmap",
    ground_sensor_locations=[],
    charging_stations_locations=[],
    frames_per_image=3,
    maxframes=np.inf,
    cmap=None,                     # custom colormap (optional)
    vmin=None,                     # min value for LogNorm
    vmax=None,                     # max value for LogNorm
    display_zones=False            # show drone zones around charging stations
)
```

### Key Differences from `create_scenario_video`

| Feature | `create_scenario_video` | `create_video_scenario_burnmap` |
|---------|------------------------|-------------------------------|
| Color scale | Linear | Logarithmic (`LogNorm`) |
| Background | White + fire overlay | Full burn probability heatmap |
| Axis origin | Upper-left (default) | Lower-left (`origin='lower'`) |
| Video library | OpenCV (`cv2`) | `imageio` |
| Drone zones | Not supported | Optional cyan rectangles |

### Drone Zone Display

When `display_zones=True`, a dashed cyan square of side length 60 cells is drawn centered on each charging station. This visualizes the operational coverage zone each drone is responsible for.

```python
rect = Rectangle(
    (x - 30, y - 30), 60, 60,
    linewidth=2, edgecolor='cyan',
    facecolor='none', linestyle='--'
)
```

### Video Assembly (imageio)

This function assembles the video internally using `imageio` rather than the shared `create_video_from_images()` helper:

```python
writer = imageio.get_writer(output_path, fps=10)
for image_file in image_files:
    img = imageio.imread(image_path)
    for _ in range(frames_per_image):
        writer.append_data(img)
writer.close()
```

---

## 7. Video Compilation

### `create_video_from_images(image_dir, output_filename, frames_per_image)`

A utility function that compiles a folder of sequentially-named PNG images into an MP4 video using OpenCV.

```python
# Collect and sort images
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

# Determine frame size from first image
first_frame = cv2.imread(os.path.join(image_dir, image_files[0]))
height, width, layers = first_frame.shape

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30 // frames_per_image
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# Write frames (each image repeated frames_per_image times)
for image_file in image_files:
    frame = cv2.imread(os.path.join(image_dir, image_file))
    for _ in range(frames_per_image):
        video_writer.write(frame)
```

**Speed control**: Higher `frames_per_image` → slower playback. Default is 3, which produces a 10 FPS video from the 30 FPS base.

---

## 8. Video Helpers — Replaying Logged Runs

The `video_helpers.py` module provides utilities to reconstruct and visualize simulation runs from cached log files, without re-running the simulation.

### `load_drone_positions_from_log(log_path)`

Parses a logged drone routing JSON file and extracts the position history:

```python
def load_drone_positions_from_log(log_path):
    with open(log_path, "r") as f:
        log = json.load(f)

    drone_locations_history = []
    for t, timestep_actions in enumerate(log["actions_history"]):
        positions = [
            tuple(action[1])
            for action in timestep_actions
            if isinstance(action, list) and len(action) > 1
        ]
        drone_locations_history.append(positions)

    return drone_locations_history
```

The JSON structure (from `LoggableDroneStrategyWrapper`) stores each timestep's actions as a list of `[action_type, [x, y]]` pairs. This function extracts just the `(x, y)` positions.

### `get_latest_layout_name_from_logs(dataset_folder_name)`

Scans all layout folders in a dataset for the most recently modified log file:

```
dataset_folder/
  ├── layout_0058_03866/
  │     └── logs/
  │           └── *logged_drone_routing.json  ← check creation time
  ├── layout_0059_01234/
  │     └── logs/
  │           └── *logged_drone_routing.json
  └── ...
```

Returns the layout name with the **most recent** log file. This is useful for quickly visualizing the last run without manually specifying the layout.

### `generate_video_from_latest_run(experiment_name, layout_name, dataset_folder_name)`

An end-to-end function that reconstructs and renders a video from the most recent experiment run:

```
1. Locate latest burn map
   └── tmp_burnmaps/tmp_burnmap_*.npy → most recently created

2. Load drone positions
   └── {layout}/logs/*logged_drone_routing.json → load_drone_positions_from_log()

3. Load static device locations
   └── {layout}/logs/*charge.json → ground_sensor_locations, charging_station_locations

4. Render video
   └── create_video_scenario_burnmap(burn_map, drone_history, ...)
```

### Typical Usage

```python
from video_helpers import get_latest_layout_name_from_logs, generate_video_from_latest_run

layout = get_latest_layout_name_from_logs("MiniTractDataset/")
generate_video_from_latest_run(
    experiment_name="SMwhp_parallel",
    layout_name=layout,
    dataset_folder_name="MiniTractDataset/"
)
```

---

## 9. Visual Encoding Reference

### Color Legend

| Element | Color | Marker | Layer |
|---------|-------|--------|-------|
| Burning cell | Red `[1,0,0]` | — | Grid overlay |
| Burnt cell | Black `[0,0,0]` | — | Grid overlay |
| Unburnt cell | White `[1,1,1]` | — | Base layer |
| Drone | Black | Diamond `D` | Scatter |
| Ground sensor | Green | Square `s` | Scatter |
| Charging station | Blue | Star `*` | Scatter |
| Coverage area | Gray (α=0.3) | Square `s` | Scatter |
| Drone zone | Cyan | Dashed rectangle | Patch |
| Burn probability | White→Yellow→Red | — | Heatmap (imshow) |

### File Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Frame image | `grid_timestep_XXX.png` | `grid_timestep_042.png` |
| Scenario video | `{name}.mp4` (in `display_{name}/`) | `display_simulation/simulation.mp4` |
| Burn map video | `{name}.mp4` (in `display_{name}/`) | `display_simulation_burnmap/simulation_burnmap.mp4` |

### Output Directories

Each video generation function creates a `display_{name}/` folder containing:
1. All individual frame PNGs
2. The compiled MP4 video

If the folder already exists, existing files are moved to a timestamped `backup_YYYYMMDD_HHMMSS/` subdirectory before new frames are generated.

---

*Previous: [07 — Benchmarking Pipeline](07_benchmarking_pipeline.md)*
