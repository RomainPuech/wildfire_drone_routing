# move the .npy files from the risk_layouts folder to the appropriate folder
import shutil
import os
from glob import glob
import tifffile
import numpy as np

# Path to the folder containing .tif files
folder_path = "/home/gridsan/rpuech/ZWF/risk_layouts/whp" #/bp

# Get a list of all .tif files in the folder
tif_files = glob(os.path.join(folder_path, "*.tif"))
print(tif_files)

layout_list = []
missing_layouts = []

# Open each .tif file
for tif_file in tif_files:
    print(tif_file)
    id = tif_file.split('/')[-1].split('cnt_')[1].split('.tif')[0]
    print("------",id)

    # open tiff and convert to npy
    try:
        tif = tifffile.imread(tif_file)
        npy_file = os.path.join("/home/gridsan/rpuech/ZWF/WideDataset", f"{id}/static_risk_whp.npy")
        np.save(npy_file, tif)
    except Exception as e:
        print(f"Error processing {tif_file} : {e}")
    
print(len(missing_layouts), "layouts skipped, copied", len(tif_files) - len(missing_layouts), "layouts")
