import numpy as np
from burn_map_model import BurnMapPredictor, load_weather_data

# === 1. Paths to model and burn map ===
model_path = "/Users/josephye/Desktop/wildfire_drone_routing/code/ML_burn_map/models/ok_residualbatch128_normfirst_auc2_best_ap_improvement.pt"
burn_map_path = "/Users/josephye/Desktop/wildfire_drone_routing/code/ML_burn_map/ML/MLDatasets/0002/burn_map.npy"  # <- adjust if needed
weather_file_path = "/Users/josephye/Desktop/wildfire_drone_routing/code/ML_burn_map/ML/MLDatasets/0002/Weather_Data_Processed/0002_00003.txt"

# === 2. Create the predictor ===
predictor = BurnMapPredictor(model_path, burn_map_path, num_weather_timesteps=5)


# === 3. Create weather history vector ===
num_timesteps   = 5               # must match the model you trained
feat_len = predictor.concat_len // num_timesteps   # e.g. 95 // 5 → 19
timestep = 3

history = []
for k in range(num_timesteps):
    t  = timestep - k
    vec = load_weather_data(weather_file_path, t) if t >= 0 else np.zeros(feat_len)
    print(vec)
    history.append(vec)

weather_history = np.concatenate(history).astype(np.float32)

# === 4. Create predictor and run prediction ===
predictor = BurnMapPredictor(model_path, burn_map_path, num_weather_timesteps=num_timesteps)
burn_prob_map = predictor.predict(timestep=timestep, weather_history=weather_history)


# === 5. Show results ===
print("Prediction shape:", burn_prob_map.shape)
print("Prediction (mean value):", np.mean(burn_prob_map))

import matplotlib.pyplot as plt

plt.imshow(burn_prob_map, cmap="hot", vmin=0, vmax=1)
plt.colorbar(label="Fire probability")
plt.title(f"Predicted Fire Risk at Timestep {timestep}")
plt.show()
