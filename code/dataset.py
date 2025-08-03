# By Romain Puech

import os
import tqdm
import shutil
import numpy as np
import random
import time
import os
from PIL import Image
import re
import rasterio
import pandas as pd
import json

def convert_tif_to_npy(input_folder, output_folder):
    """
    Convert all .tif or .tiff files in a folder to .npy format.

    Args:
        input_folder (str): Path to the directory containing .tif/.tiff files.
        output_folder (str): Path to the directory where .npy files will be saved.
        max_files (int, optional): Maximum number of .tif files to process. If None, process all files.

    Yields:
        str: Path to each .npy file created.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            tif_path = os.path.join(input_folder, file)
            with rasterio.open(tif_path) as src:
                data = src.read(1)  # Read the first band

            npy_filename = os.path.splitext(file)[0] + '.npy'
            np.save(os.path.join(output_folder, npy_filename), data)
            print(f"Converted {file} to {npy_filename}")

def listdir_limited(input_dir, max_n_scenarii=None):
    """
    Iterate through directory contents with an optional limit on number of items.

    Args:
        input_dir (str): Path to the directory to scan
        max_n_scenarii (int, optional): Maximum number of items to yield. If None, yields all items.

    Yields:
        str: Path to each directory entry
    """
    if not input_dir.endswith('/'):
        input_dir += '/'

    if max_n_scenarii is None:
        yield from os.listdir(input_dir)
        return
    
    count = 0
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.name != ".DS_Store":
                yield  entry.name
                count += 1
                if not max_n_scenarii is None and count >= max_n_scenarii:
                    break

def delete_logs_folder(folder_path):
    """
    Deletes the 'logs' subfolder in the given folder_path.
    
    Args:
        folder_path (str): The path to the directory containing the 'logs' folder.
    """
    logs_path = os.path.join(folder_path, 'logs')
    if os.path.exists(logs_path) and os.path.isdir(logs_path):
        shutil.rmtree(logs_path)
        print(f"Deleted 'logs' folder at: {logs_path}")
    else:
        print(f"No 'logs' folder found at: {logs_path}")





####### Functions to load data #######   

def load_scenario_jpg(folder_path, binary=True, first_frame_only=False):
    """
    Load a wildfire scenario from a sequence of grayscale JPG images.

    Args:
        folder_path (str): Path to the folder containing the JPG image sequence
        binary (bool, optional): If True, threshold images at 0.5 to create binary values. Defaults to False.

    Returns:
        numpy.ndarray: TxNxN array representing the fire progression

    Raises:
        FileNotFoundError: If no JPG files found in folder
        ValueError: If images have inconsistent dimensions

    Example:
        >>> scenario = load_scenario_jpg("fire_sequence/")
        >>> print(scenario.shape)
        (10, 100, 100)
    """
    def natural_sort_key(s):
        """
        Helper function to sort strings with numbers in natural order.
        Converts 'im1', 'im2', 'im10' to proper numerical order.
        """
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    if not folder_path.endswith("/"):
        folder_path += "/"

    # Get list of jpg files in the folder and sort naturally
    jpg_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')],
                      key=natural_sort_key)
    
    if not jpg_files:
        raise FileNotFoundError(f"No JPG files found in folder: {folder_path}")
    
    # Read first image to get dimensions
    first_image = Image.open(os.path.join(folder_path, jpg_files[0])).convert('L')
    if first_frame_only:
        return np.array(first_image).astype(float) / 255.0
    height, width = first_image.size[1], first_image.size[0]  # Get both dimensions
    T = len(jpg_files)
    
    # Initialize scenario array
    scenario = np.zeros((T, height, width))
    
    # Load each image
    for t, jpg_file in enumerate(jpg_files):
        img_path = os.path.join(folder_path, jpg_file)
        img = Image.open(img_path).convert('L')
        
        # Verify image dimensions
        if img.size != (width, height):
            raise ValueError(f"Image {jpg_file} has different dimensions than the first image")
        
        # Convert image to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(float) / 255.0
        
        # Apply binary threshold if requested
        if binary:
            img_array = (img_array >= 0.5).astype(float)
        
        scenario[t] = img_array
    
    # Starting time is always 0
    return scenario

def load_scenario_npy(filename):
    """
    Load a scenario from a NumPy binary file.
    
    Args:
        filename (str): Name of the file to load (with or without .npy extension)
    
    Returns:
        numpy.ndarray: TxNxN array representing the fire progression
    """
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    try:
        loaded_data = np.load(filename, allow_pickle=True)
        if loaded_data.ndim > 0:  # If it's a regular array (new format)
            return loaded_data
        else:  # If it's a 0-dim array containing a dictionary (old format)
            return loaded_data.item()['scenario']
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except Exception as e:
        raise Exception(f"Error loading scenario: {str(e)}")



    
####### Functions to save data #######

def save_scenario_npy(scenario, out_filename="scenario"):
    """
    Save a wildfire scenario to a NumPy binary file.

    Args:
        scenario (numpy.ndarray): TxNxM array representing the fire progression over time
        out_filename (str, optional): Output filename. Defaults to "scenario". 
            ".npy" extension will be added if not present.

    Notes:
        The scenario data is converted to float32 type for storage efficiency.
    """
    if not out_filename.endswith('.npy'):
        out_filename += '.npy'
    
    # Save scenario data directly as array
    np.save(out_filename, scenario.astype(np.float32))

def save_scenario_jpg(scenario, out_folder_name):
    """
    Save a scenario as a folder of JPG images.
    Args:
        scenario (numpy.ndarray): TxNxM array representing the scenario
        out_folder_name (str): Path to the folder to save the JPG images
    """
    if not out_folder_name.endswith("/"):
        out_folder_name += "/"
    for t in range(scenario.shape[0]):
        img = Image.fromarray(scenario[t].astype(np.uint8))
        img.save(out_folder_name + f"{t}.jpg")

def save_scenario(scenario, filename, extension = ".npy"):
    """
    Save a scenario to a npy file or a folder of jpg images.
    Args:
        scenario (numpy.ndarray): TxNxM array representing the scenario
        filename (str): Path to the file to save the scenario
        extension (str): .npy if filename is a npy file, .jpg if filename is a folder of jpg images
    """
    if not extension.startswith("."):
        extension = "." + extension
    if extension == ".npy":
        save_scenario_npy(scenario, filename)
    else:
        save_scenario_jpg(scenario, filename)

def save_burn_map(burn_map, filename, extension = ".npy"):
    """
    Save a burn map to a npy file or a folder of jpg images.
    
    Args:
        burn_map (numpy.ndarray): TxNxM array representing the burn map
        filename (str): Name of the npy file or folder of jpg images to save the burn map
        extension (str): .npy if filename is a npy file, .jpg if filename is a folder of jpg images
    """
    save_scenario(burn_map, filename, extension)

####### Functions to preprocess data #######

def jpg_scenario_to_npy(jpg_folder_name, npy_folder_name = None, npy_filename = None):
    """
    Convert a scenario stored as a folder of JPG images to a NPY file.
    Args:
        jpg_folder_name (str): Path to the folder containing the JPG images
        npy_folder_name (str): Path to the folder to save the NPY file
        npy_filename (str): Name of the NPY file (without extension)
    """
    if npy_folder_name is None:
        npy_folder_name = jpg_folder_name
    if npy_filename is None:
        npy_filename = "scenario"
    if npy_filename.endswith(".npy"):
        npy_filename = npy_filename[:-4]
    if not jpg_folder_name.endswith("/"):
        jpg_folder_name += "/"
    if not npy_folder_name.endswith("/"):
        npy_folder_name += "/"
    
    scenario = load_scenario_jpg(jpg_folder_name)
    save_scenario_npy(scenario, npy_folder_name + npy_filename + ".npy")

def sim2real_scenario_jpg_folders_to_npy(dataset_folder_name, npy_folder_name = None, n_max_scenarii_per_layout = None, verbose = False, n_max_layouts = None, mismatch_threshold = None):
    """
    Convert all JPG scenarios in the sim2real dataset to NPY files for faster processing.
    Args:
        dataset_folder_name (str): Path to the dataset folder
        npy_folder_name (str): Path to the folder to save the NPY files
        n_max_scenarii_per_layout (int): Maximum number of scenarii per layout to process
        mismatch_threshold (float): Threshold for percentage of historical fires that could not be martched with the dataset's scenarios. If the percentage is higher than the threshold, the layout is not converted.
    """
    print(f"Converting JPG scenarios to NPY for {dataset_folder_name}")
    if npy_folder_name is None:
        npy_folder_name = dataset_folder_name
    if not dataset_folder_name.endswith("/"):
        dataset_folder_name += "/"
    if not npy_folder_name.endswith("/"):
        npy_folder_name += "/"

    n_layout = 0
    for layout_folder in os.listdir(dataset_folder_name):
        print(layout_folder)
        if n_max_layouts is not None and n_layout >= n_max_layouts:
            break

        if mismatch_threshold is not None:
            # check the percentage of failed matches
            if not os.path.exists(os.path.join("WideDataset/", layout_folder, "selected_scenarios.txt")):
                print(f"Skipping folder: {layout_folder} because it does not contain selected_scenarios.txt")
                continue
            with open(os.path.join("WideDataset/", layout_folder, "selected_scenarios.txt"), "r") as f:
                lines = f.readlines()
                failed_percentage = float(lines[-1].split(" ")[-1])
                if failed_percentage > mismatch_threshold:
                    print(f"Skipping folder: {layout_folder} because it has a mismatch percentage of {failed_percentage}")
                    continue
            
        n_layout += 1
        if verbose: print(f"Converting JPG scenarios to NPY for {dataset_folder_name + layout_folder}")
        #if os.path.exists(dataset_folder_name + layout_folder + "/scenarii/"):continue
        if not os.path.exists(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/"):continue
        if not os.path.exists(dataset_folder_name + layout_folder + "/scenarii/") :os.makedirs(dataset_folder_name + layout_folder + "/scenarii/", exist_ok=True)
        for scenario_folder in tqdm.tqdm(listdir_limited(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/", n_max_scenarii_per_layout)):
            # has this scenario already been converted?
            if os.path.exists(dataset_folder_name + layout_folder + "/scenarii/" + scenario_folder.strip("/") + ".npy"):
                print(f"Scenario {scenario_folder} already converted")
                continue
            try:
                jpg_scenario_to_npy(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/" + scenario_folder, npy_folder_name + layout_folder + "/scenarii/", scenario_folder.strip("/"))
            except Exception as e:
                print(f"Error converting {dataset_folder_name + layout_folder + '/Satellite_Images_Mask/' + scenario_folder} to NPY: {e}")

def load_scenario(file_or_folder_name, extension = ".npy", first_frame_only=False):
    """
    Load a scenario from a file or a folder.
    Args:
        file_or_folder_name (str): Path to the npy file or jpg folder containing the scenario
        extension (str): .npy if file_or_folder_name is a npy file, .jpg if file_or_folder_name is a folder of jpg images
    """
    if not extension.startswith("."):
        extension = "." + extension
    if extension == ".npy":
        return load_scenario_npy(file_or_folder_name)
    else:
        return load_scenario_jpg(file_or_folder_name, first_frame_only=first_frame_only)
    

def compute_burn_map(folder_name, extension = ".npy", noncumulative = False, config=None):
    """
    Compute the burn map for a layout.
    Args:
        folder_name (str): Path to the folder containing the scenario files / folders
    """
    if not extension.startswith("."):
        extension = "." + extension

    print(f"Computing burn map for {folder_name} and files with extension {extension}")
    if not folder_name.endswith("/"):
        folder_name += "/"
    
    burn_map = None
    counts = None
    N = M = None
    
    # Process all scenarios in a single pass
    for filename in tqdm.tqdm(os.listdir(folder_name)):
        if (extension == ".npy" and filename.endswith(extension)) or (extension == ".jpg" and os.path.isdir(folder_name + filename)):
            scenario = load_scenario(folder_name + filename, extension)
            previous_frame = np.zeros_like(scenario[0]) if noncumulative else None
            T, curr_N, curr_M = scenario.shape
            
            # Initialize arrays on first file
            if burn_map is None:
                N, M = curr_N, curr_M
                if config is not None:
                    starting_time = config.get(f"offset_{filename.split('/')[-1].split('.')[0]}", 0)
                    print(f"Starting time: {starting_time}, file: {filename.split('/')[-1].split('.')[0]}")
                    if starting_time > 0:
                        #print(f"Starting time: {starting_time}, file: {filename.split('/')[-1]}")
                        # prepend empty grids to the scenario
                        scenario = np.concatenate([np.zeros((starting_time, N, M)), scenario], axis=0)
                T = scenario.shape[0]
                burn_map = np.zeros((T, N, M))
                counts = np.zeros(T, dtype=int)
            else:
                # Verify grid dimensions
                if (curr_N, curr_M) != (N, M):
                    raise ValueError(f"Inconsistent grid dimensions in {filename}")

                if config is not None:
                    starting_time = config.get(f"offset_{filename.split('/')[-1].split('.')[0]}", 0)
                    print(f"Starting time: {starting_time}, file: {filename.split('/')[-1].split('.')[0]}")
                    if starting_time > 0:
                        #print(f"Starting time: {starting_time}, file: {filename.split('/')[-1]}")
                        # prepend empty grids to the scenario
                        scenario = np.concatenate([np.zeros((starting_time, N, M)), scenario], axis=0)
                        T = scenario.shape[0]

                # Extend arrays if needed
                if T > burn_map.shape[0]:
                    burn_map = np.pad(burn_map, ((0, T - burn_map.shape[0]), (0, 0), (0, 0)))
                    counts = np.pad(counts, (0, T - counts.shape[0]))
            
            # Add scenario data
            for t in range(T):
                if noncumulative:
                    burn_map[t] += (scenario[t] - previous_frame)
                    previous_frame = scenario[t]
                else:
                    burn_map[t] += scenario[t]
                counts[t] += 1
    
    # Calculate mean for each timestep
    for t in range(burn_map.shape[0]):
        if counts[t] > 0:
            burn_map[t] /= counts[t]
    
    return burn_map

def load_burn_map(filename, extension = ".npy"):
    """
    Load a burn map from a npy file.
    Args:
        filename (str): Path to the npy file containing the burn map
    """
    if not extension.startswith("."):
        extension = "." + extension
    if extension == ".npy":
        return load_scenario(filename, ".npy")
    else:
        return load_scenario(filename, ".jpg")

####### Prepocess the sim2real dataset #######

# def compute_and_save_burn_maps_sim2real_dataset(dataset_folder_name, extension = ".npy", n_max_layouts = None):
#     """
#     Compute the burn map for all scenarios in the sim2real dataset and save them as NPY files.
#     Args:
#         dataset_folder_name (str): Path to the dataset folder
#     """
#     if not dataset_folder_name.endswith("/"):
#         dataset_folder_name += "/"
#     n_layout = 0
#     for layout_folder in os.listdir(dataset_folder_name):
#         if n_max_layouts is not None and n_layout >= n_max_layouts:
#             break
#         if not os.path.exists(dataset_folder_name + layout_folder + "/scenarii/"):continue
#         burn_map = compute_burn_map(dataset_folder_name + layout_folder + "/scenarii/", extension)
#         save_burn_map(burn_map, dataset_folder_name + layout_folder + "/burn_map.npy")
#         n_layout += 1
def preprocess_sim2real_dataset(dataset_folder_name, n_max_scenarii_per_layout = None, n_max_layouts = None, mismatch_threshold = None, config_file = None):
    """
    Preprocess the sim2real dataset by converting JPG scenarios to NPY files and computing burn maps.
    Args:
        dataset_folder_name (str): Path to the dataset folder
        n_max_scenarii_per_layout (int): Maximum number of scenarii per layout to process
    """
    sim2real_scenario_jpg_folders_to_npy(dataset_folder_name, n_max_scenarii_per_layout = n_max_scenarii_per_layout, n_max_layouts = n_max_layouts, mismatch_threshold = mismatch_threshold)
    print("Computing burn maps...")
    config = json.load(open(config_file)) if config_file is not None else None
    compute_and_save_burn_maps_sim2real_dataset(dataset_folder_name, n_max_layouts = n_max_layouts, mismatch_threshold = mismatch_threshold, config = config)

def compute_and_save_burn_maps_sim2real_dataset(dataset_folder_name, n_max_layouts = None, extension = ".npy", noncumulative = False, config=None, mismatch_threshold = None):
    """
    Compute the burn maps for all scenarios in the sim2real dataset.
    Args:
        dataset_folder_name (str): Path to the dataset folder
        n_max_layouts (int): Maximum number of layouts to process
        n_max_scenarii_per_layout (int): Maximum number of scenarii per layout to process
    """
    if not dataset_folder_name.endswith("/"):
        dataset_folder_name += "/"
    if not extension[0] == '.':
        extension = '.' + extension
    n_layout = 0
    for layout_folder in os.listdir(dataset_folder_name):
        if n_max_layouts is not None and n_layout >= n_max_layouts:
            break
        extension_folder = "/scenarii/" if extension == ".npy" else "/Satellite_Images_Mask/"
        if extension != ".npy" and not os.path.exists(dataset_folder_name + layout_folder + extension_folder):
            extension_folder = "/Satellite_Image_Mask/"
        if extension != ".npy" and not os.path.exists(dataset_folder_name + layout_folder + extension_folder):
            continue
        if mismatch_threshold is not None:
            if not os.path.exists(os.path.join("WideDataset/", layout_folder, "selected_scenarios.txt")):
                print(f"Skipping folder: {layout_folder} because it does not contain selected_scenarios.txt")
                continue
            with open(os.path.join("WideDataset/", layout_folder, "selected_scenarios.txt"), "r") as f:
                lines = f.readlines()
                failed_percentage = float(lines[-1].split(" ")[-1])
                if failed_percentage > mismatch_threshold:
                    print(f"Skipping folder: {layout_folder} because it has a mismatch percentage of {failed_percentage}")
                    continue
        try:
            bm = compute_burn_map(dataset_folder_name + layout_folder + extension_folder, extension, noncumulative, config=config)
            ns = "_noncumulative" if noncumulative else ""
            save_burn_map(bm, dataset_folder_name + layout_folder + f"/burn_map{ns}.npy")
        except Exception as e:
            print(f"Error computing burn map for {dataset_folder_name + layout_folder + extension_folder}: {e}")
        n_layout += 1


def combine_all_benchmark_results(dataset_folder: str, strategy_name = "RandomSensorPlacementStrategy_DroneRoutingMaxCoverageResetStatic", experiment_name = ""):
    """
    Combines all per-layout benchmark CSVs from Satellite_Images_Mask folders into one file.
    Preserves layout/scenario formatting (e.g., 0001, 00002).

    Args:
        dataset_folder (str): Root folder containing layout subfolders.
        output_filename (str): Name of the combined CSV file to write.

    Returns:
        pd.DataFrame: Combined DataFrame of all results.
    """
    if not dataset_folder.endswith('/'):
        dataset_folder += '/'

    all_dfs = []

    for layout in os.listdir(dataset_folder):
        layout_path = os.path.join(dataset_folder, layout)
        if not os.path.isdir(layout_path):
            continue    

        layout_shortened_name = layout.split("_")[0]

        csv_path = os.path.join(layout_path, f"{layout_shortened_name}_benchmark_results{experiment_name}_{strategy_name}.csv")
        if os.path.exists(csv_path):
            print(f"✔ Found: {csv_path}")
            df = pd.read_csv(csv_path, dtype={"layout": str, "scenario": str})
            all_dfs.append(df)
        else:
            print(f"⚠ No benchmark CSV found at: {csv_path}")

    if not all_dfs:
        print("❌ No CSV files found. Nothing to combine.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    if experiment_name is None:
        experiment_name = strategy_name
    combined_path = os.path.join("results", "combined_benchmark_results"+experiment_name+".csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"\n✅ Combined results saved to: {combined_path}")

    return combined_df


def clean_layout_folders(root_folder):
    """
    Cleans each layout folder inside the root_folder by keeping only specified files and folders.
    If 'Satellite_Image_Mask' is present instead of 'Satellite_Images_Mask', it is renamed.
    """
    # Set of allowed names
    allowed_items = {
        "Fuel_Map",
        "Satellite_Images_Mask",
        "satellite_image.png",
        "static_risk.npy",
        "Topography_Map",
        "Vegetation_Map",
        "Weather_Data"
    }

    for layout_name in os.listdir(root_folder):
        layout_path = os.path.join(root_folder, layout_name)
        if not os.path.isdir(layout_path):
            continue  # Skip non-directories

        # Handle renaming if needed
        mask_path = os.path.join(layout_path, "Satellite_Image_Mask")
        if os.path.exists(mask_path) and not os.path.exists(os.path.join(layout_path, "Satellite_Images_Mask")):
            os.rename(mask_path, os.path.join(layout_path, "Satellite_Images_Mask"))

        # Remove any file/folder that isn't in the allowed list
        for item in os.listdir(layout_path):
            item_path = os.path.join(layout_path, item)
            if item not in allowed_items:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

        print(f"Cleaned: {layout_name}")

def clean_logs_folder(root_folder):
    """
    Cleans the logs folder inside the root_folder by keeping only the files that are in the allowed list.
    """
    for layout_name in os.listdir(root_folder):
        layout_path = os.path.join(root_folder, layout_name)
        if not os.path.isdir(layout_path):
            continue
        logs_path = os.path.join(layout_path, "logs")
        if not os.path.exists(logs_path):
            continue
        for file in os.listdir(logs_path):
            #print(f"Removing: {os.path.join(logs_path, file)}")
            os.remove(os.path.join(logs_path, file))
        

if __name__ == "__main__":
    clean_logs_folder("WideDataset/")
    # combine_all_benchmark_results("WideDataset/", suffix = "SensorPlacementOptimization_DroneRoutingMaxCoverageResetStaticGreedy")
    #0058_benchmark_resultsRandomSensorPlacementStrategy_DroneRoutingMaxCoverageResetStatic
    #WideDataset/0058_03866/Satellite_Images_Mask/0058_03866_benchmark_resultsRandomSensorPlacementStrategy_DroneRoutingMaxCoverageResetStatic.csv
