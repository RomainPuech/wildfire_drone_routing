import numpy as np
from burn_map_model import BurnMapPredictor, load_weather_data
import matplotlib.pyplot as plt

# === 1. Paths to model and burn map ===
model_path = "/Users/josephye/Desktop/wildfire_drone_routing/code/ML_burn_map/models/ok_residualbatch128_normfirst_auc2_best_ap_improvement.pt"
burn_map_path = "/Users/josephye/Desktop/wildfire_drone_routing/code/ML_burn_map/ML/MLDatasets/0002/burn_map.npy"
weather_file_path = "/Users/josephye/Desktop/wildfire_drone_routing/code/ML_burn_map/ML/MLDatasets/0002/Weather_Data_Processed/0002_00010.txt"
output_path = "/Users/josephye/Desktop/wildfire_drone_routing/code/ML_burn_map/ML/MLDatasets/0002/predicted_burnmaps/burnmap_0002_00010.npy"

# === 2. Create the predictor ===
predictor = BurnMapPredictor(model_path, burn_map_path, num_weather_timesteps=5)

# === 3. Setup ===
num_timesteps   = 5
feat_len        = predictor.concat_len // num_timesteps
burn_map_full   = np.load(burn_map_path)
T_total         = burn_map_full.shape[0]  # Number of total timesteps

# === 4. Predict for each timestep ===
predicted_burnmaps = []

for timestep in range(T_total):
    history = []
    for k in range(num_timesteps):
        t = timestep - k
        vec = load_weather_data(weather_file_path, t) if t >= 0 else np.zeros(feat_len)
        history.append(vec)
    weather_history = np.concatenate(history).astype(np.float32)

    burn_prob_map = predictor.predict(timestep=timestep, weather_history=weather_history)
    predicted_burnmaps.append(burn_prob_map)

# === 5. Stack predictions ===
predicted_burnmaps = np.stack(predicted_burnmaps, axis=0)  # Shape (T, N, M)


# === 7. Save final burnmap ===
np.save(output_path, predicted_burnmaps)
print(f"Saved predicted burn map with shape {predicted_burnmaps.shape} to {output_path}")

import os

# === 7a. Create a directory to save the images ===
images_output_dir = "/Users/josephye/Desktop/wildfire_drone_routing/code/ML_burn_map/ML/MLDatasets/0002/predicted_burnmaps/images"
os.makedirs(images_output_dir, exist_ok=True)

# === 7b. Save an image for each timestep ===
for t in range(predicted_burnmaps.shape[0]):
    plt.figure()
    plt.imshow(predicted_burnmaps[t], cmap="hot", vmin=0, vmax=1)
    plt.colorbar(label="Fire probability")
    plt.title(f"Predicted Fire Risk at Timestep {t}")
    plt.axis('off')  # Hide axes if you prefer clean images

    img_path = os.path.join(images_output_dir, f"burnmap_t{t:04d}.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()  # Close the figure after saving to avoid memory issues

print(f"Saved all burnmap images to {images_output_dir}")
