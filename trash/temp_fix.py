import os
import numpy as np
import matplotlib.pyplot as plt
from benchmark import listdir_folder_limited
from dataset import load_scenario_jpg

def plot_ignition_point(dataset_folder_name, max_n_layouts=50, max_n_scenarii=100):
    """
    Plot the ignition points of fires in each layout using JPEG folders instead of NPY files.
    For each layout, collects ignition points from all scenarios and plots them.
    """
    i = 0
    for layout_folder in listdir_folder_limited(dataset_folder_name, max_n_layouts):
        # Check if the folder has a Satellite_Images_Mask directory
        if not os.path.exists(layout_folder + "/Satellite_Images_Mask/"):
            continue
            
        print(f"Processing layout {i}: {layout_folder}")
        i += 1
        
        # Lists to store x and y coordinates
        x_coords = []
        y_coords = []
        
        # Process each scenario folder (containing JPG images)
        for scenario_folder in listdir_folder_limited(layout_folder + "/Satellite_Images_Mask/", max_n_scenarii):
            try:
                # Load the scenario from JPG images
                scenario = load_scenario_jpg(scenario_folder)
                
                # Find the ignition point(s) in the first frame
                fire_points = np.argwhere(scenario[0, :, :])
                
                if len(fire_points) > 0:
                    # Take the first ignition point if there are multiple
                    point = fire_points[0]
                    # Store coordinates (x is column, y is row in the image)
                    y_coords.append(point[0])  # Row coordinate
                    x_coords.append(point[1])  # Column coordinate
            except Exception as e:
                print(f"Error processing {scenario_folder}: {e}")
                continue
        
        if len(x_coords) > 0:
            # Plot the ignition points for this layout
            plt.figure(figsize=(10, 8))
            plt.scatter(x_coords, y_coords, alpha=0.7)
            plt.title(f'Ignition Points for Layout {os.path.basename(layout_folder)}')
            plt.xlabel('X coordinate (column)')
            plt.ylabel('Y coordinate (row)')
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            print(f"No valid ignition points found for layout {layout_folder}") 