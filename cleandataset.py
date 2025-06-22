import os
import shutil
empty = 0
tot = 0

##### remove folders with a lot of skipped scenarios (i.e a lot of mismatches)

# for layout_folder in os.listdir("WideDataset/"):
#     if not os.path.isdir(os.path.join("WideDataset/", layout_folder)):
#         print(f"Skipping non-directory folder: {layout_folder}")
#         continue
#     # look for the 'selected_scenarios.txt' file
#     if not os.path.exists(os.path.join("WideDataset/", layout_folder, "selected_scenarios.txt")):
#         print(f"Skipping folder: {layout_folder} because it does not contain selected_scenarios.txt")
#         continue
#     # read the 'selected_scenarios.txt' file
#     print(os.path.join("WideDataset/", layout_folder, "selected_scenarios.txt"))
#     with open(os.path.join("WideDataset/", layout_folder, "selected_scenarios.txt"), "r") as f:
#         lines = f.readlines()
#         failed_percentage = float(lines[-1].split(" ")[-1])
#         if failed_percentage > 0.1:
#             bad +=1
#             # move the folder to the bad_folders folder
#             print(os.path.join("WideDataset/", layout_folder))
#             print(os.path.join("WideDataset/", "bad_folders", layout_folder))
#             #shutil.move(os.path.join("WideDataset/", layout_folder), os.path.join("WideDataset/", "bad_folders", layout_folder))
#     tot += 1
# print(f"Total folders: {tot}")
# print(f"Bad folders: {bad}")
# print(f"Percentage of bad folders: {round(bad/tot*100, 2)}%")

#### remove folders with empty folders

# for layout_folder in os.listdir("WideDataset/"):
#     if not os.path.isdir(os.path.join("WideDataset/", layout_folder)):
#         print(f"Skipping non-directory folder: {layout_folder}")
#         continue
#     # look for the 'Sattelite_Images_Mask' folder and check if it's empty
#     folder = os.path.join("WideDataset/", layout_folder, "Satellite_Images_Mask")
#     if not os.path.exists(folder):
#         print(f"{folder} doesn't exist")
#         folder = os.path.join("WideDataset/", layout_folder, "Satellite_Image_Mask")
#         if not os.path.exists(folder):
#             print(f"{folder} doesn't exist")
#             folder = os.path.join("WideDataset/", layout_folder, "Satellite_lmages_Mask")
#             if not os.path.exists(folder):
#                 print(f"{folder} doesn't exist")
#                 continue

#     # rename the folder to Satellite_Images_Mask
#     taget_folder = os.path.join("WideDataset/", layout_folder, "Satellite_Images_Mask")
#     shutil.move(folder, taget_folder)
    

#     # check if folder is empty
#     if not os.listdir(taget_folder):
#         print(f"{taget_folder} is empty")
#         empty +=1
#         # move the folder to the empty_folders folder
#         print(f"moving {os.path.join('WideDataset/', layout_folder)} to {os.path.join('WideDataset/', 'empty_folders', layout_folder)}")
#         shutil.move(os.path.join("WideDataset/", layout_folder), os.path.join("WideDataset/", "empty_folders", layout_folder))
#         continue
#     print("\n")
#     tot += 1

# print(f"Total folders: {tot}")
# print(f"Empty folders: {empty}")
# print(f"Percentage of empty folders: {round(empty/tot*100, 2)}%")


# empty all logs folders
removed = 0
print("Emptying all logs folders")
for folder in os.listdir("WideDataset/"):
    if os.path.isdir(os.path.join("WideDataset/", folder)):
        logs_folder = os.path.join("WideDataset/", folder, "logs")
        if os.path.isdir(logs_folder):
            for file in os.listdir(logs_folder):
                print(os.path.join(logs_folder, file))
                os.remove(os.path.join(logs_folder, file))
                removed +=1
        else:
            print(f"Skipping {folder} because it is not a logs folder")
print(f"Removed {removed} files")


# remove all previous csv results
removed = 0
print("Deleting all csv results")
for folder in os.listdir("WideDataset/"):
    if os.path.isdir(os.path.join("WideDataset/", folder)):
        for file in os.listdir(os.path.join("WideDataset/", folder)):
            if file.endswith(".csv"):
                print(os.path.join("WideDataset/", folder, file))
                os.remove(os.path.join("WideDataset/", folder, file))
                removed +=1
        if os.path.isdir(os.path.join("WideDataset/", folder, "Satellite_Images_Mask")):
            for file in os.listdir(os.path.join("WideDataset/", folder, "Satellite_Images_Mask")):
                if file.endswith(".csv"):
                    print(os.path.join("WideDataset/", folder, "Satellite_Images_Mask", file))
                    os.remove(os.path.join("WideDataset/", folder, "Satellite_Images_Mask", file))
                    removed +=1
        if os.path.isdir(os.path.join("WideDataset/", folder, "scenarii")):
            for file in os.listdir(os.path.join("WideDataset/", folder, "scenarii")):
                if file.endswith(".csv"):
                    print(os.path.join("WideDataset/", folder, "scenarii", file))
                    os.remove(os.path.join("WideDataset/", folder, "scenarii", file))
                    removed +=1
print(f"Removed {removed} files")

    