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


####### Functions to load data #######

def load_ignition_map(filename):
    """
    Load an ignition map from a text file.
    
    Args:
        filename (str): Name of the file to load (with or without .txt extension)
    
    Returns:
        numpy.ndarray: NxN array of probabilities loaded from the file
    """
    # Add .txt extension if not present
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    try:
        # Load the map using numpy's loadtxt function with comma delimiter
        ignition_map = np.loadtxt(filename, delimiter=',')
        
        # Verify the map is square
        if ignition_map.shape[0] != ignition_map.shape[1]:
            raise ValueError("Loaded ignition map is not square")
            
        # print(f"Successfully loaded ignition map from {filename}")
        return ignition_map
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except Exception as e:
        raise Exception(f"Error loading ignition map: {str(e)}")
    

def load_scenario(filename):
    """
    Load a scenario and its starting time from a text file.
    
    Args:
        filename (str): Name of the file to load (with or without .txt extension)
    
    Returns:
        tuple: (scenario, starting_time) where scenario is a TxNxN array and starting_time is an integer
    """
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    try:
        # Read the first two lines separately for metadata
        with open(filename, 'r') as f:
            starting_time = int(f.readline().strip())
            T, N, M = map(int, f.readline().strip().split(','))
        
        # Use numpy's faster mmap_mode to load the data
        # Skip the first two lines (header) and reshape directly
        data = np.loadtxt(filename, delimiter=',', skiprows=2, dtype=np.float32)
        data = data.reshape(T, N, M)
        
        return data, starting_time
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except Exception as e:
        raise Exception(f"Error loading scenario: {str(e)}")
    
def load_scenario_jpg(folder_path, binary=False):
    """
    Load a scenario from a folder containing grayscale JPG images.
    
    Args:
        folder_path (str): Path to the folder containing the JPG images
        binary (bool): If True, threshold the images at 0.5 to create binary values
    
    Returns:
        tuple: (scenario, starting_time) where scenario is a TxNxN array and starting_time is 0
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
    starting_time = 0
    
    return scenario, starting_time

def load_scenario_npy(filename):
    """
    Load a scenario and its starting time from a binary numpy file.
    
    Args:
        filename (str): Name of the file to load (with or without .npy extension)
    
    Returns:
        tuple: (scenario, starting_time) where scenario is a TxNxN array and starting_time is an integer
    """
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    try:
        data = np.load(filename, allow_pickle=True).item()
        return data['scenario'], data['starting_time']
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except Exception as e:
        raise Exception(f"Error loading scenario: {str(e)}")
    
####### Functions to save data #######

def save_ignition_map(ignition_map, filename):
    """
    Save the ignition map to a text file.
    
    Args:
        ignition_map (numpy.ndarray): NxN array of probabilities
        filename (str): Name of the file to save (with .txt extension)
    """
    # Add .txt extension if not present
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    # Save with high precision (8 decimal places) and scientific notation
    np.savetxt(filename, ignition_map, fmt='%.8e', delimiter=',')
    



def save_scenario(scenario, starting_time, out_filename):
    """
    Save a scenario and its starting time to a text file.
    
    Args:
        scenario (numpy.ndarray): TxNxM array representing the wildfire evolution
        starting_time (int): The time step when the fire starts
        filename (str): Name of the file to save (with or without .txt extension)
    """
    if not out_filename.endswith('.txt'):
        out_filename += '.txt'
    
    # Save starting_time on first line, then scenario data
    with open(out_filename, 'w') as f:
        # Write starting time
        f.write(f"{starting_time}\n")
        
        # Write scenario dimensions on second line
        T, N, M = scenario.shape
        f.write(f"{T},{N},{M}\n")
        
        # Write scenario data
        for t in range(T):
            for i in range(N):
                row = ','.join(map(str, scenario[t, i, :]))
                f.write(row + '\n')
    

def save_scenario_npy(scenario, starting_time, out_filename="scenario"):
    """Save scenario to binary file for faster loading"""
    if not out_filename.endswith('.npy'):
        out_filename += '.npy'
    
    # Save metadata and scenario data together as a dictionary
    np.save(out_filename, {
        'starting_time': starting_time,
        'scenario': scenario.astype(np.float32)
    })

def save_burn_map(burn_map, filename):
    """
    Save a burn map to a text file.
    
    Args:
        burn_map (numpy.ndarray): TxNxM array representing the burn map
        filename (str): Name of the file to save (with .txt extension)
    """
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    save_scenario_npy(burn_map, 0, filename)

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
    
    scenario, _ = load_scenario_jpg(jpg_folder_name)
    save_scenario_npy(scenario, 0, npy_folder_name + npy_filename + ".npy")

def sim2real_scenario_jpg_folders_to_npy(dataset_folder_name, npy_folder_name = None):
    """
    Convert all JPG scenarios in the sim2real dataset to NPY files for faster processing.
    Args:
        dataset_folder_name (str): Path to the dataset folder
        npy_folder_name (str): Path to the folder to save the NPY files
    """
    print(f"Converting JPG scenarios to NPY for {dataset_folder_name}")
    if npy_folder_name is None:
        npy_folder_name = dataset_folder_name
    if not dataset_folder_name.endswith("/"):
        dataset_folder_name += "/"
    if not npy_folder_name.endswith("/"):
        npy_folder_name += "/"

    for layout_folder in os.listdir(dataset_folder_name):
        print(f"Converting JPG scenarios to NPY for {dataset_folder_name + layout_folder}")
        if not os.path.exists(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/"):continue
        if not os.path.exists(dataset_folder_name + layout_folder + "/scenarii/") :os.makedirs(dataset_folder_name + layout_folder + "/scenarii/", exist_ok=True)
        for scenario_folder in tqdm.tqdm(os.listdir(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/")):
            if scenario_folder == ".DS_Store":continue
            jpg_scenario_to_npy(dataset_folder_name + layout_folder + "/Satellite_Images_Mask/" + scenario_folder, npy_folder_name + layout_folder + "/scenarii/", scenario_folder.strip("/"))


def compute_burn_map(folder_name):
    """
    Compute the burn map for a scenario stored as a folder of NPY files.
    Args:
        folder_name (str): Path to the folder containing the NPY files
    """
    print(f"Computing burn map for {folder_name}")
    if not folder_name.endswith("/"):
        folder_name += "/"
    
    burn_map = None
    counts = None
    N = M = None
    
    # Process all scenarios in a single pass
    for filename in tqdm.tqdm(os.listdir(folder_name)):
        if filename.endswith(".npy"):
            scenario, _ = load_scenario_npy(folder_name + filename)
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

def compute_and_save_burn_maps_sim2real_dataset(dataset_folder_name):
    """
    Compute the burn map for all scenarios in the sim2real dataset and save them as NPY files.
    Args:
        dataset_folder_name (str): Path to the dataset folder
    """
    if not dataset_folder_name.endswith("/"):
        dataset_folder_name += "/"
    
    for layout_folder in os.listdir(dataset_folder_name):
        if not os.path.exists(dataset_folder_name + layout_folder + "/scenarii/"):continue
        burn_map = compute_burn_map(dataset_folder_name + layout_folder + "/scenarii/")
        save_burn_map(burn_map, dataset_folder_name + layout_folder + "/burn_map.npy")

def preprocess_sim2real_dataset(dataset_folder_name):
    """
    Preprocess the sim2real dataset by converting JPG scenarios to NPY files and computing burn maps.
    Args:
        dataset_folder_name (str): Path to the dataset folder
    """
    print("Converting JPG scenarios to NPY...")
    sim2real_scenario_jpg_folders_to_npy(dataset_folder_name)
    print("Computing burn maps...")
    compute_and_save_burn_maps_sim2real_dataset(dataset_folder_name)
