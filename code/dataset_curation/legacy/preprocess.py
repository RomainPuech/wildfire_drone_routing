import os
import sys
import json
module_path = os.path.abspath(".") + "/code"
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset import preprocess_sim2real_dataset, compute_and_save_burn_maps_sim2real_dataset
# preprocess_sim2real_dataset("./WideDataset", mismatch_threshold = 0.2)
config = json.load(open("./config_s2r.json"))
compute_and_save_burn_maps_sim2real_dataset("./WideDataset", mismatch_threshold = 0.2, config = config, noncumulative=True)