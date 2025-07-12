import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import re
import geopandas as gpd
from datetime import datetime
def find_earliest_latest_dates(layout_path):
    """
    Find the earliest and latest dates in a layout folder.
    """
    earliest_date = None
    latest_date = None
    for filename in os.listdir(os.path.join(layout_path, "Weather_Data")):
        if filename.endswith(".txt"):
            first_line, last_line = read_first_and_last_lines(os.path.join(layout_path, "Weather_Data", filename))
            first_date = " ".join(first_line.split(" ")[:4])
            last_date = " ".join(last_line.split(" ")[:4])
            first_date = datetime.strptime(first_date, "%Y %m %d %H%M")
            last_date = datetime.strptime(last_date, "%Y %m %d %H%M")
            if earliest_date is None or first_date < earliest_date:
                earliest_date = first_date
            if latest_date is None or last_date > latest_date:
                latest_date = last_date
    return earliest_date, latest_date


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
        print(f"\n\nFile does not exist: {file_path}\n\n")
        return None, None

    with open(file_path, "r") as f:
        first_line = f.readline()
        if len(first_line) < 15:
            print(f"\n\nFirst line is too short: {file_path}\n\n")
            return None, None
        try:
            y,m,d,h = first_line.split(" ")[:4]
            first_date = datetime(int(y), int(m), int(d), int(h))
            season = month_to_season(first_date.month)
            return season, first_date
        except Exception as e:
            print(f"\n\nError parsing first line: {file_path},    {e}\n\n")
            print(first_line)
            return None, datetime(1900, 1, 1, 0, 0)

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
    Load selected scenario IDs from a file, creating dictionary scenario_id:fire_id.
    """
    selected = {}
    with open(selected_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2 and "_" in parts[0]:
                selected[parts[0].strip()] = parts[1].strip()
    print(f"Selected scenarios loaded: {len(selected)}")
    return selected

def summarize_selected_scenarios_jpg(root_folder, selected_file_name="selected_scenarios.txt", km_per_cell=1):
    sm = 0
    # load the fires dataset
    new_fires_gdf = gpd.read_file("./newfires.gpkg") # 6th edition # gpd.read_file("./FPA_FOD_20210617.gpkg") 5th edition  
    # replace illegal dates by jan first 1900
    new_fires_gdf['DISCOVERYDATETIME'] = new_fires_gdf['DISCOVERYDATETIME'].replace('1001/01/01 00:00:00+00', '1900/01/01 00:00:00+00')
    print(1)
    new_fires_gdf['DISCOVERY_DATE'] = pd.to_datetime(new_fires_gdf['DISCOVERYDATETIME'])
    new_fires_gdf['DISCOVERY_DATE'].dt.date
    print(2)
    new_fires_gdf['LONGITUDE'] = new_fires_gdf['LONGDD83']
    new_fires_gdf['LATITUDE'] = new_fires_gdf['LATDD83']
    new_fires_gdf = new_fires_gdf.to_crs("EPSG:4326")
    print(len(new_fires_gdf)) # 

    old_fires_gdf = gpd.read_file("FPA_FOD_20221014.gpkg")
    old_fires_gdf['DISCOVERY_DATE'] = pd.to_datetime(old_fires_gdf['DISCOVERY_DATE'])
    old_fires_gdf['OBJECTID'] = old_fires_gdf['FOD_ID']
    old_fires_gdf = old_fires_gdf.to_crs("EPSG:4326")
    print(len(old_fires_gdf)) # 2 303 566

    # merge the two gdfs
    fires_gdf = pd.concat([new_fires_gdf, old_fires_gdf])
    print(len(fires_gdf)) 

    # drop duplicates as defined as same lat, long and same date
    fires_gdf = fires_gdf.drop_duplicates(subset=['LATITUDE', 'LONGITUDE', 'DISCOVERY_DATE'])
    print(len(fires_gdf)) 

    records = []
    errors = []
    for layout_name in os.listdir(root_folder):
        try:
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
            try:
                selected_ids_historical = load_selected_scenarios(os.path.join(layout_path, "selected_scenarios_historical.txt"))
            except FileNotFoundError:
                selected_ids_historical = {}
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

                    # print(f"Moving unused weather file: {fname} → {trash_dir}")
                    os.rename(src, dst)
            
            print(f"Processing layout: {layout_name}, selected scenarios: {len(selected_ids)}")
            scenarios_folder = os.path.join(layout_path, "Satellite_Images_Mask")
            if not os.path.exists(scenarios_folder):
                scenarios_folder = os.path.join(layout_path, "Satellite_Image_Mask")
                if not os.path.exists(scenarios_folder):
                    print(f"No scenarios folder found in {layout_path}, skipping...")
                    continue

            for scenario_folder_name in os.listdir(scenarios_folder):
                if scenario_folder_name not in selected_ids:
                    trash_dir = os.path.abspath(os.path.join(root_folder, "..", "trash"))
                    os.makedirs(trash_dir, exist_ok=True)

                    src = os.path.join(scenarios_folder, scenario_folder_name)
                    dst = os.path.join(trash_dir, scenario_folder_name)

                    #print(f"Moving unused scenario: {scenario_folder_name} → {trash_dir}")
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

####
                corresponding_fire_id = int(selected_ids[scenario_folder_name])
                corresponding_fire = fires_gdf[fires_gdf['OBJECTID'] == corresponding_fire_id]
                assert len(corresponding_fire) == 1, f"Expected 1 fire, got {len(corresponding_fire)}, {corresponding_fire_id}"
                # seasonal natch if the corresponding fire has the same day of the year as the scenario
                seasonal_match = corresponding_fire['DISCOVERY_DATE'].iloc[0].strftime('%m-%d') == first_date.strftime('%m-%d')
                if seasonal_match:
                    sm+=1
                    print(f"Seasonal match: {scenario_folder_name}, {corresponding_fire['DISCOVERY_DATE'].iloc[0].strftime('%m-%d')} == {first_date.strftime('%m-%d')}")
                # historical match if the scenario is also in the historical dataset
                historical_match = scenario_folder_name in selected_ids_historical
####
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
        except Exception as e:
            print(f"Error processing layout {layout_name}: {e}")
            errors.append(layout_name)
            continue

    df = pd.DataFrame(records)
    csv_path = os.path.join(root_folder, "scenario_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Errors: {errors}")
    return df


if __name__ == "__main__":
    summarize_selected_scenarios_jpg("./WideDataset")