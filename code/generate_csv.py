import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import re

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'

def compute_fire_radius(scenario):
    fire_cells = np.argwhere(scenario == 1)
    if len(fire_cells) == 0:
        return 0
    center = fire_cells.mean(axis=0)
    dists = np.linalg.norm(fire_cells - center, axis=1)
    return dists.max()

def get_scenario_season(layout_path, layout_number, scenario_number):
    """
    Get the season for a specific scenario using the corresponding weather file.
    """
    weather_folder = os.path.join(layout_path, "Weather_Data")
    filename = f"{layout_number}_{scenario_number}.txt"
    file_path = os.path.join(weather_folder, filename)

    if not os.path.exists(file_path):
        return None, None

    with open(file_path, "r") as f:
        first_line = f.readline()
        if len(first_line) < 15:
            return None, None
        try:
            first_date = datetime.strptime(first_line[:15], "%Y %m %d %H%M")
            season = month_to_season(first_date.month)
            return season, first_date
        except Exception:
            return None, None

def load_scenario_jpg(folder_path, binary=False):
    """
    Load a wildfire scenario from a sequence of grayscale JPG images.
    """
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    if not folder_path.endswith("/"):
        folder_path += "/"

    jpg_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')],
                       key=natural_sort_key)
    
    if not jpg_files:
        raise FileNotFoundError(f"No JPG files found in folder: {folder_path}")
    
    first_image = Image.open(os.path.join(folder_path, jpg_files[0])).convert('L')
    height, width = first_image.size[1], first_image.size[0]
    T = len(jpg_files)
    scenario = np.zeros((T, height, width))
    
    for t, jpg_file in enumerate(jpg_files):
        img_path = os.path.join(folder_path, jpg_file)
        img = Image.open(img_path).convert('L')
        if img.size != (width, height):
            raise ValueError(f"Image {jpg_file} has different dimensions than the first image")
        img_array = np.array(img).astype(float) / 255.0
        if binary:
            img_array = (img_array >= 0.5).astype(float)
        scenario[t] = img_array
    
    return scenario

def load_selected_scenarios(selected_file_path):
    """
    Load selected scenario IDs from a file, filtering lines like '0047_03634, <some_number>'.
    """
    selected = set()
    with open(selected_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2 and "_" in parts[0]:
                selected.add(parts[0].strip())
    print(f"Selected scenarios loaded: {len(selected)}")
    return selected

def summarize_selected_scenarios_jpg(root_folder, selected_file_name="selected_scenarios.txt", km_per_cell=1):
    records = []
    for layout_name in os.listdir(root_folder):
        # print(f"Processing layout: {layout_name}")
        layout_path = os.path.join(root_folder, layout_name)
        if not os.path.isdir(layout_path):
            continue

        selected_file_path = os.path.join(layout_path, selected_file_name)
        print(f"Selected file path: {selected_file_path}")
        if not os.path.exists(selected_file_path):
            # print("HERE!")
            continue
        
        selected_ids = load_selected_scenarios(selected_file_path)
        # Build set of used weather filenames like '0047_03634.txt'
        used_weather_files = {f"{sid}.txt" for sid in selected_ids}

        weather_folder = os.path.join(layout_path, "Weather_Data")
        for fname in os.listdir(weather_folder):
            if not fname.endswith(".txt"):
                continue
            if fname not in used_weather_files:
                trash_dir = os.path.abspath(os.path.join(root_folder, "..", "trash"))
                os.makedirs(trash_dir, exist_ok=True)

                src = os.path.join(weather_folder, fname)
                dst = os.path.join(trash_dir, fname)

                print(f"Moving unused weather file: {fname} → {trash_dir}")
                os.rename(src, dst)
        
        print(f"Processing layout: {layout_name}, selected scenarios: {len(selected_ids)}")
        seasonal_match = os.path.exists(os.path.join(layout_path, "selected_scenarios_seasonal.txt"))
        historical_match = os.path.exists(os.path.join(layout_path, "selected_scenarios_historical.txt"))
        scenarios_folder = os.path.join(layout_path, "Satellite_Images_Mask")

        for scenario_folder_name in os.listdir(scenarios_folder):
            if scenario_folder_name not in selected_ids:
                trash_dir = os.path.abspath(os.path.join(root_folder, "..", "trash"))
                os.makedirs(trash_dir, exist_ok=True)

                src = os.path.join(scenarios_folder, scenario_folder_name)
                dst = os.path.join(trash_dir, scenario_folder_name)

                print(f"Moving unused scenario: {scenario_folder_name} → {trash_dir}")
                os.rename(src, dst)
                continue

            scenario_folder = os.path.join(scenarios_folder, scenario_folder_name)
            if not os.path.isdir(scenario_folder):
                continue

            try:
                layout_number, scenario_number = scenario_folder_name.split("_")
            except ValueError:
                print(f"Invalid scenario folder name: {scenario_folder_name}")
                continue

            scenario = load_scenario_jpg(scenario_folder, binary=True)

            season, first_date = get_scenario_season(layout_path, layout_number, scenario_number)
            season_number = ['Winter', 'Spring', 'Summer', 'Autumn'].index(season) + 1 if season else None

            final_t = scenario[-1]
            final_fire_size = (final_t == 1).sum()
            final_radius_km = compute_fire_radius(final_t) * km_per_cell

            t10_index = min(10, len(scenario)-1)
            t10_size = (scenario[t10_index] == 1).sum()
            fast_fire = t10_size >= 0.5 * final_fire_size
            slow_fire = not fast_fire

            record = {
                "layout_number": layout_number,
                "scenario_number": scenario_number,
                "season_number": season_number,
                "seasonal_match": seasonal_match,
                "historical_match": historical_match,
                "big_fire": final_radius_km >= 20,
                "small_fire": final_radius_km < 20,
                "fast_fire": fast_fire,
                "slow_fire": slow_fire
            }

            records.append(record)

    df = pd.DataFrame(records)
    csv_path = os.path.join(root_folder, "scenario_summary.csv")
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    summarize_selected_scenarios_jpg("./MinimalDataset")