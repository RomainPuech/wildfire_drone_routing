import os
import sys
module_path = os.path.abspath(".") + "/code"
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset import preprocess_sim2real_dataset
preprocess_sim2real_dataset("./WideDataset", mismatch_threshold = 0.2)