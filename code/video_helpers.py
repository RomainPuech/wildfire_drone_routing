import os
import json
import glob
from dataset import load_scenario_npy
from displays import create_video_scenario_burnmap

def load_drone_positions_from_log(log_path):
    with open(log_path, "r") as f:
        log = json.load(f)

    if "actions_history" not in log:
        raise KeyError("'actions_history' key not found in drone log.")

    drone_locations_history = []
    for t, timestep_actions in enumerate(log["actions_history"]):
        try:
            positions = [tuple(action[1]) for action in timestep_actions if isinstance(action, list) and len(action) > 1]
            drone_locations_history.append(positions)
        except Exception as e:
            print(f"[!] Error at timestep {t}: {timestep_actions}")
            raise e

    return drone_locations_history

def get_latest_layout_name_from_logs(dataset_folder_name):
    layout_root = dataset_folder_name
    layout_candidates = []

    for layout_dir in os.listdir(layout_root):
        logs_dir = os.path.join(layout_root, layout_dir, "logs")
        if not os.path.isdir(logs_dir):
            continue

        full_paths = [
            os.path.join(logs_dir, f)
            for f in os.listdir(logs_dir)
            if f.endswith("logged_drone_routing.json")
        ]

        for path in full_paths:
            layout_candidates.append((layout_dir, os.path.getctime(path)))

    if not layout_candidates:
        raise FileNotFoundError("No layout with a log file found.")

    # Return layout name with most recent log file
    layout_name = max(layout_candidates, key=lambda x: x[1])[0]
    print(f"[🧠] Dynamically selected layout: {layout_name}")
    return layout_name

def generate_video_from_latest_run(experiment_name, layout_name, dataset_folder_name):
    print("🎞️ Starting video generation...")

    layout_log_dir = os.path.join(dataset_folder_name, layout_name, "logs")
    burnmap_files = glob.glob("tmp_burnmaps/tmp_burnmap_*.npy")
    if not burnmap_files:
        print(f"[!] No burnmap found in tmp_burnmaps/. Skipping video.")
        return

    latest_burnmap = max(burnmap_files, key=os.path.getctime)
    print(f"📄 Using burnmap: {latest_burnmap}")
    burn_map = load_scenario_npy(latest_burnmap)

    matching_logs = [f for f in os.listdir(layout_log_dir) if f.endswith("logged_drone_routing.json")]
    if not matching_logs:
        print(f"[!] No 'logged_drone_routing.json' found in {layout_log_dir}")
        return
    log_path = os.path.join(layout_log_dir, matching_logs[0])
    print(f"📄 Found drone log: {log_path}")
    drone_locations_history = load_drone_positions_from_log(log_path)

    ground_sensor_locations = []
    charging_stations_locations = []
    static_json_candidates = [f for f in os.listdir(layout_log_dir) if f.endswith("charge.json")]
    if static_json_candidates:
        static_json_path = os.path.join(layout_log_dir, static_json_candidates[0])
        print(f"📄 Found static location file: {static_json_path}")
        with open(static_json_path, "r") as f:
            static_data = json.load(f)
            ground_sensor_locations = static_data.get("ground_sensor_locations", [])
            charging_stations_locations = static_data.get("charging_station_locations", [])
        print(f"📍 Ground sensors: {ground_sensor_locations}")
        print(f"🔌 Charging stations: {charging_stations_locations}")
    else:
        print(f"[!] No static location JSON file found in {layout_log_dir}. Proceeding without it.")

    os.makedirs("videos", exist_ok=True)
    output_path = f"{experiment_name}_burnmap_video"

    create_video_scenario_burnmap(
        burn_map=burn_map,
        drone_locations_history=drone_locations_history,
        ground_sensor_locations=ground_sensor_locations,
        charging_stations_locations=charging_stations_locations,
        out_filename=output_path,
    )