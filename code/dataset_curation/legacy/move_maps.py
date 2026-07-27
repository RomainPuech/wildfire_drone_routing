# move the .npy files from the risk_layouts folder to the appropriate folder
import shutil
import os
from glob import glob
import tifffile
import numpy as np
from matplotlib import pyplot as plt

# Path to the folder containing .tif files
folder_path = "/Users/puech/Desktop/Climate/wildfire_drone_routing/risk_layouts/whp" #/bp

# Get a list of all .tif files in the folder
tif_files = glob(os.path.join(folder_path, "*.tif"))
print(tif_files)

layout_list = []
missing_layouts = []
failed_files = []

# Open each .tif file
for tif_file in tif_files:
    print(tif_file)
    id = tif_file.split('/')[-1].split('cnt_')[1].split('.tif')[0]
    print("------",id)

    # open tiff and convert to npy
    try:
        # Try standard tifffile reading first
        tif = tifffile.imread(tif_file)
    except ValueError as e:
        if "is not castable to int16" in str(e):
            print(f"GDAL casting error for {tif_file}, trying alternative method...")
            try:
                # Try reading without GDAL tags
                with tifffile.TiffFile(tif_file) as tf:
                    tif = tf.asarray()
            except Exception as e2:
                print(f"Alternative method also failed for {tif_file}: {e2}")
                failed_files.append(id)
                continue
        else:
            print(f"Error processing {tif_file} : {e}")
            failed_files.append(id)
            continue
    except Exception as e:
        print(f"Error processing {tif_file} : {e}")
        failed_files.append(id)
        continue
    
    try:
        npy_file = os.path.join("/Users/puech/Desktop/Climate/wildfire_drone_routing/WideDataset", f"{id}/static_risk_whp.npy")
        # reshape from 2D (N,M) to 3D (1,N,M)
        tif = tif[np.newaxis, :, :]
        # save it if the layout folder exists
        if os.path.exists(os.path.join("/Users/puech/Desktop/Climate/wildfire_drone_routing/WideDataset", f"{id}")):
            np.save(npy_file, tif)
            # # display the map using imshow
            # bm = tif[0, :, :]
            # plt.imshow(bm)
            # plt.savefig("test.png")
        else:
            missing_layouts.append(id)
    except Exception as e:
        print(f"Error saving {tif_file} : {e}")
        failed_files.append(id)
    
print(f"{len(missing_layouts)} layouts skipped (missing folder)")
print(f"{len(failed_files)} files failed to process")
print(f"{len(tif_files) - len(missing_layouts) - len(failed_files)} layouts successfully processed")
