import os
import numpy as np
import glob
from tqdm import tqdm
import shutil

def process_weather_data(foldername):
    """
    Preprocess weather data files in the given dataset folder.
    
    Args:
        foldername (str): Path to the dataset folder (e.g., 'MLDataset/0002')
        
    The function:
    1. Creates a new Weather_Data_Processed folder with the same structure
    2. Processes each weather file to:
       - Drop the year (first column)
       - Use dummy variables for the month (second column)
       - Drop the day (third column)
       - Normalize all other weather variables
    """
    # Define paths
    weather_data_path = os.path.join(foldername, 'Weather_Data')
    processed_data_path = os.path.join(foldername, 'Weather_Data_Processed')
    
    # Check if Weather_Data folder exists
    if not os.path.exists(weather_data_path):
        print(f"Weather_Data folder not found in {foldername}")
        return
    
    # Create the processed data folder if it doesn't exist
    if os.path.exists(processed_data_path):
        print(f"Weather_Data_Processed folder already exists in {foldername}. Removing it...")
        shutil.rmtree(processed_data_path)
    
    os.makedirs(processed_data_path, exist_ok=True)
    print(f"Created folder: {processed_data_path}")
    
    # Get all weather files
    weather_files = glob.glob(os.path.join(weather_data_path, '*.txt'))
    if not weather_files:
        print(f"No weather files found in {weather_data_path}")
        return
    
    print(f"Found {len(weather_files)} weather files")
    
    # First pass: Calculate statistics for normalization
    # We'll collect all values for each column to calculate mean and std
    column_values = {}
    
    print("First pass: Collecting statistics for normalization...")
    for weather_file in tqdm(weather_files):
        with open(weather_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                # Skip year (0), month (1), and day (2)
                for i, val in enumerate(data[3:], start=3):
                    if i not in column_values:
                        column_values[i] = []
                    column_values[i].append(float(val))
    
    # Calculate mean and std for each column
    column_stats = {}
    for col_idx, values in column_values.items():
        mean = np.mean(values)
        std = np.std(values)
        # Prevent division by zero in normalization
        if std == 0:
            std = 1
        column_stats[col_idx] = {'mean': mean, 'std': std}
    
    print("Statistics calculated for normalization")
    
    # Second pass: Process files and save to the new location
    print("Second pass: Processing weather files...")
    for weather_file in tqdm(weather_files):
        filename = os.path.basename(weather_file)
        output_file = os.path.join(processed_data_path, filename)
        
        processed_lines = []
        with open(weather_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                
                # Keep only necessary data: drop year and day, use dummy vars for month
                month = int(data[1])
                
                # Create an empty list for the processed line
                processed_data = []
                
                # Add month dummy variables (1-12)
                for m in range(1, 13):
                    processed_data.append('1' if month == m else '0')
                
                # Process time and other variables (normalize)
                for i, val in enumerate(data[3:], start=3):
                    # Normalize the value
                    normalized_val = (float(val) - column_stats[i]['mean']) / column_stats[i]['std']
                    processed_data.append(str(normalized_val))
                
                processed_lines.append(' '.join(processed_data))
        
        # Write processed data to the new file
        with open(output_file, 'w') as f:
            f.write('\n'.join(processed_lines))
    
    print(f"Processing complete. Processed files saved to {processed_data_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process weather data files')
    parser.add_argument('foldername', type=str, help='Path to the dataset folder')
    
    args = parser.parse_args()
    process_weather_data(args.foldername)
