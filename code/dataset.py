# By Romain Puech

import os
import tqdm
import numpy as np
import numpy as np
import random
import time
import os
from PIL import Image
import re


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

####### Functions to load data #######

# DEPRECATED
# def load_ignition_map(filename):
#     """
#     Load an ignition map from a text file.
    
#     Args:
#         filename (str): Name of the file to load (with or without .txt extension)
    
#     Returns:
#         numpy.ndarray: NxN array of probabilities loaded from the file
#     """
#     # Add .txt extension if not present
#     if not filename.endswith('.txt'):
#         filename += '.txt'
    
#     try:
#         # Load the map using numpy's loadtxt function with comma delimiter
#         ignition_map = np.loadtxt(filename, delimiter=',')
        
#         # Verify the map is square
#         if ignition_map.shape[0] != ignition_map.shape[1]:
#             raise ValueError("Loaded ignition map is not square")
            
#         # print(f"Successfully loaded ignition map from {filename}")
#         return ignition_map
        
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Could not find file: {filename}")
#     except Exception as e:
#         raise Exception(f"Error loading ignition map: {str(e)}")
#     
# DEPRECATED
# def load_scenario(filename):
#     """
#     Load a scenario and its starting time from a text file.
    
#     Args:
#         filename (str): Name of the file to load (with or without .txt extension)
    
#     Returns:
#         tuple: (scenario, starting_time) where scenario is a TxNxN array and starting_time is an integer
#     """
#     if not filename.endswith('.txt'):
#         filename += '.txt'
    
#     try:
#         # Read the first two lines separately for metadata
#         with open(filename, 'r') as f:
#             starting_time = int(f.readline().strip())
#             T, N, M = map(int, f.readline().strip().split(','))
        
#         # Use numpy's faster mmap_mode to load the data
#         # Skip the first two lines (header) and reshape directly
#         data = np.loadtxt(filename, delimiter=',', skiprows=2, dtype=np.float32)
#         data = data.reshape(T, N, M)
        
#         return data, starting_time
    
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Could not find file: {filename}")
#     except Exception as e:
#         raise Exception(f"Error loading scenario: {str(e)}")
    

def load_scenario_jpg(folder_path, binary=False):
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

    # Get list of jpg files in the folder and sort naturally
    jpg_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')],
                      key=natural_sort_key)
    
    if not jpg_files:
        raise FileNotFoundError(f"No JPG files found in folder: {folder_path}")
    
    # Read first image to get dimensions
    first_image = Image.open(os.path.join(folder_path, jpg_files[0])).convert('L')
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

# DEPRECATED
# def save_ignition_map(ignition_map, filename):
#     """
#     Save the ignition map to a text file.
    
#     Args:
#         ignition_map (numpy.ndarray): NxN array of probabilities
#         filename (str): Name of the file to save (with .txt extension)
#     """
#     # Add .txt extension if not present
#     if not filename.endswith('.txt'):
#         filename += '.txt'
    
#     # Save with high precision (8 decimal places) and scientific notation
#     np.savetxt(filename, ignition_map, fmt='%.8e', delimiter=',')
    

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

def sim2real_scenario_jpg_folders_to_npy(dataset_folder_name, npy_folder_name = None, n_max_scenarii_per_layout = None, verbose = False):
    """
    Convert all JPG scenarios in the sim2real dataset to NPY files for faster processing.
    Args:
        dataset_folder_name (str): Path to the dataset folder
        npy_folder_name (str): Path to the folder to save the NPY files
        n_max_scenarii_per_layout (int): Maximum number of scenarii per layout to process
    """
    print(f"Converting JPG scenarios to NPY for {dataset_folder_name}")
    if npy_folder_name is None:
        npy_folder_name = dataset_folder_name
    if not dataset_folder_name.endswith("/"):
        dataset_folder_name += "/"
    if not npy_folder_name.endswith("/"):
        npy_folder_name += "/"

    for layout_folder in os.listdir(dataset_folder_name):
        if verbose: print(f"Converting JPG scenarios to NPY for {dataset_folder_name + layout_folder}")
        if os.path.exists(dataset_folder_name + layout_folder + "/scenarii/"):continue
        if not os.path.exists(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/"):continue
        if not os.path.exists(dataset_folder_name + layout_folder + "/scenarii/") :os.makedirs(dataset_folder_name + layout_folder + "/scenarii/", exist_ok=True)
        for scenario_folder in tqdm.tqdm(listdir_limited(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/", n_max_scenarii_per_layout)):
            try:
                jpg_scenario_to_npy(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/" + scenario_folder, npy_folder_name + layout_folder + "/scenarii/", scenario_folder.strip("/"))
            except Exception as e:
                print(f"Error converting {dataset_folder_name + layout_folder + "/Satellite_Images_Mask/" + scenario_folder} to NPY: {e}")

def load_scenario(file_or_folder_name, extension = ".npy"):
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
        return load_scenario_jpg(file_or_folder_name)
    

def compute_burn_map(folder_name, extension = ".npy", output_extension = ".npy"):
    """
    Compute the burn map for a scenario stored as a folder of NPY files.
    Args:
        folder_name (str): Path to the folder containing the NPY files
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
        if filename.endswith(extension):
            scenario = load_scenario(folder_name + filename, extension)
            T, curr_N, curr_M = scenario.shape
            
            # Initialize arrays on first file
            if burn_map is None:
                N, M = curr_N, curr_M
                burn_map = np.zeros((T, N, M))
                counts = np.zeros(T, dtype=int)
            else:
                # Verify grid dimensions
                if (curr_N, curr_M) != (N, M):
                    raise ValueError(f"Inconsistent grid dimensions in {filename}")
                # Extend arrays if needed
                if T > burn_map.shape[0]:
                    burn_map = np.pad(burn_map, ((0, T - burn_map.shape[0]), (0, 0), (0, 0)))
                    counts = np.pad(counts, (0, T - counts.shape[0]))
            
            # Add scenario data
            for t in range(T):
                burn_map[t] += scenario[t]
                counts[t] += 1
    
    # Calculate mean for each timestep
    for t in range(burn_map.shape[0]):
        if counts[t] > 0:
            burn_map[t] /= counts[t]
    
    return burn_map

####### Prepocess the sim2real dataset #######

def compute_and_save_burn_maps_sim2real_dataset(dataset_folder_name, extension = ".npy"):
    """
    Compute the burn map for all scenarios in the sim2real dataset and save them as NPY files.
    Args:
        dataset_folder_name (str): Path to the dataset folder
    """
    if not dataset_folder_name.endswith("/"):
        dataset_folder_name += "/"
    
    for layout_folder in os.listdir(dataset_folder_name):
        if not os.path.exists(dataset_folder_name + layout_folder + "/scenarii/"):continue
        burn_map = compute_burn_map(dataset_folder_name + layout_folder + "/scenarii/", extension)
        save_burn_map(burn_map, dataset_folder_name + layout_folder + "/burn_map.npy")

def preprocess_sim2real_dataset(dataset_folder_name, n_max_scenarii_per_layout = None):
    """
    Preprocess the sim2real dataset by converting JPG scenarios to NPY files and computing burn maps.
    Args:
        dataset_folder_name (str): Path to the dataset folder
        n_max_scenarii_per_layout (int): Maximum number of scenarii per layout to process
    """
    sim2real_scenario_jpg_folders_to_npy(dataset_folder_name, n_max_scenarii_per_layout = n_max_scenarii_per_layout)
    print("Computing burn maps...")
    compute_and_save_burn_maps_sim2real_dataset(dataset_folder_name)
