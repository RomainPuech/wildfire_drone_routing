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
        with open(filename, 'r') as f:
            # Read starting time
            starting_time = int(f.readline().strip())
            
            # Read dimensions
            T, N = map(int, f.readline().strip().split(','))
            
            # Initialize scenario array
            scenario = np.zeros((T, N, N))
            
            # Read scenario data
            for t in range(T):
                for i in range(N):
                    row = f.readline().strip().split(',')
                    scenario[t, i, :] = list(map(float, row))
        
        # print(f"Successfully loaded scenario from {filename}")
        return scenario, starting_time
    
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
    



def save_scenario(scenario, starting_time, filename):
    """
    Save a scenario and its starting time to a text file.
    
    Args:
        scenario (numpy.ndarray): TxNxM array representing the wildfire evolution
        starting_time (int): The time step when the fire starts
        filename (str): Name of the file to save (with or without .txt extension)
    """
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    # Save starting_time on first line, then scenario data
    with open(filename, 'w') as f:
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

def save_burn_map(burn_map, filename):
    """
    Save a burn map to a text file.
    
    Args:
        burn_map (numpy.ndarray): TxNxM array representing the burn map
        filename (str): Name of the file to save (with .txt extension)
    """
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    save_scenario(burn_map, 0, filename)
    


####### Utils to be extension agnostic #######

extension_to_function = {
    ".txt": load_scenario,
    ".jpg": load_scenario_jpg
}

####### Functions to preprocess data #######

def compute_burn_map(folder_name, extension = ".txt"):
    extension = ". " + extension if not extension.startswith(".") else extension
    load_scenario_extension = extension_to_function[extension]
    if not folder_name.endswith("/"):
        folder_name += "/"
    
    # Load first scenario to get dimensions
    first_scenario, _ = load_scenario_extension(folder_name + os.listdir(folder_name)[0])
    T, N, M = first_scenario.shape
    
    # Initialize accumulator array
    burn_map = np.zeros((T, N, M))
    count = 0
    
    # Sum up all scenarios
    for filename in tqdm.tqdm(os.listdir(folder_name)):
        if filename.endswith(extension):
            scenario, _ = load_scenario_extension(folder_name + filename)
            burn_map += scenario
            count += 1
    
    # Calculate mean
    burn_map = burn_map / count
    return burn_map

