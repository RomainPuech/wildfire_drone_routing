import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import glob
import json
import argparse
import time
import traceback
import warnings
import platform
import logging
from tqdm import tqdm
import torch.serialization  # For safe globals support
import torch.nn.functional as F  # Make sure this is at the top of imports if not already there

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger().setLevel(logging.ERROR)


def find_weather_file(weather_folder, scenario_id):
    """Find the weather file corresponding to the scenario ID"""
    for filename in os.listdir(weather_folder):
        if scenario_id in filename:
            return os.path.join(weather_folder, filename)
    return None

def load_weather_data(weather_file, time_step):
    """Load weather data for a specific time step"""
    with open(weather_file, 'r') as f:
        lines = f.readlines()
        if time_step < len(lines):
            weather_data = lines[time_step].strip().split()
            return np.array([float(x) for x in weather_data])
        else:
            return np.zeros(10)  # Default to 10 weather features

# New function for batch loading of weather data
def load_all_weather_data(weather_file):
    """
    Load all weather data from a file at once
    
    Args:
        weather_file: Path to weather file
        
    Returns:
        List of numpy arrays containing weather data for each timestep
    """
    try:
        with open(weather_file, 'r') as f:
            lines = f.readlines()
            weather_data = [
                np.array([float(x) for x in line.strip().split()]) 
                for line in lines
            ]
            return weather_data
    except Exception as e:
        print(f"Error loading weather data from {weather_file}: {e}")
        return []

def configure_device():
    """
    Configure device for PyTorch, with specific optimizations for Apple Silicon
    
    Returns:
        device: torch.device
        using_gpu: bool
        gpu_name: str
    """
    using_gpu = False
    gpu_name = "None"
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        using_gpu = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA GPU: {gpu_name}")
    # Check for Apple Silicon and MPS (Metal Performance Shaders)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        using_gpu = True
        gpu_name = "Apple Silicon MPS"
        print("Using Apple Silicon GPU via MPS")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    
    return device, using_gpu, gpu_name

# New function to check for and use processed weather data
def get_weather_folder(folder_path):
    """
    Check if processed weather data exists and return the appropriate folder
    
    Args:
        folder_path: Path to the dataset folder
        
    Returns:
        Path to the weather data folder to use
    """
    processed_weather_folder = os.path.join(folder_path, 'Weather_Data_Processed')
    if os.path.exists(processed_weather_folder):
        print(f"Using processed weather data from {processed_weather_folder}")
        return processed_weather_folder
    
    # Fall back to original weather data
    weather_folder = os.path.join(folder_path, 'Weather_Data')
    print(f"Using original weather data from {weather_folder}")
    return weather_folder

# Custom PyTorch Dataset for wildfire data
class WildfireDataset(Dataset):
    def __init__(self, burn_map, scenario_files, weather_folder, timestep_sample_rate=1, num_weather_timesteps=5):
        """
        Dataset for wildfire data
        
        Args:
            burn_map: Baseline burn map array
            scenario_files: List of scenario file paths
            weather_folder: Path to weather data folder
            timestep_sample_rate: Sample every nth timestep
            num_weather_timesteps: Number of past weather timesteps to include (including current t)
        """
        self.burn_map = burn_map
        self.scenario_files = scenario_files
        self.weather_folder = weather_folder
        self.timestep_sample_rate = timestep_sample_rate
        self.num_weather_timesteps = num_weather_timesteps  # New parameter
        
        # Storage for preloaded data
        self.preloaded_scenarios = {}
        self.preloaded_weather = {}  # Store weather data lists keyed by scenario_id
        self.original_weather_feature_count = None  # Store the original feature count
        
        # Pre-calculate tensors to avoid repeated operations
        self.cached_tensors = {}
        
        # Pre-index all examples and preload scenario data
        self.examples = []
        self._preload_data_and_index_examples()
        
        # Pre-compile common tensor operations
        self._prepare_tensor_cache()
    
    @property
    def concatenated_weather_feature_count(self):
        """Returns the total feature count after concatenating weather history."""
        if self.original_weather_feature_count is None:
            return None 
        return self.num_weather_timesteps * self.original_weather_feature_count
    
    def _preload_data_and_index_examples(self):
        """Preload all scenario data and create an index of all valid examples for faster retrieval"""
        print(f"Preloading {len(self.scenario_files)} scenario files...")
        start_time = time.time()
        
        for scenario_idx, scenario_file in enumerate(self.scenario_files):
            scenario_id = os.path.basename(scenario_file).split('.')[0]
            
            # Find corresponding weather file
            weather_file = find_weather_file(self.weather_folder, scenario_id)
            if not weather_file:
                continue
            
            # Load all weather data at once for this scenario
            all_weather_data = load_all_weather_data(weather_file)
            if not all_weather_data:
                continue

            # Store preloaded weather data for this scenario
            self.preloaded_weather[scenario_id] = all_weather_data

            # Determine original weather feature count if not already set
            if self.original_weather_feature_count is None and all_weather_data:
                self.original_weather_feature_count = len(all_weather_data[0])
            
            try:
                # Load scenario data 
                scenario_data = np.load(scenario_file, allow_pickle=True, mmap_mode='r')  # Use memory mapping for large files
                # Handle different scenario data formats
                if isinstance(scenario_data, np.ndarray) and scenario_data.dtype == np.dtype('O'):
                    scenario_data = np.load(scenario_file, allow_pickle=True)
                    scenario_data = scenario_data.item().get('scenario', None)
                    if scenario_data is None:
                        continue
                
                # Store the loaded scenario data (force C-contiguous for faster access)
                self.preloaded_scenarios[scenario_id] = np.ascontiguousarray(scenario_data)
                
                # Determine timesteps to use for training
                num_timesteps = min(scenario_data.shape[0], self.burn_map.shape[0])
                # Adjust start timestep based on required weather history
                # We need data from t-(num_weather_timesteps-1)
                start_timestep = self.num_weather_timesteps - 1 
                
                # Process each timestep using timestep_sample_rate
                for t in range(start_timestep, num_timesteps, self.timestep_sample_rate):
                    # Store index data for this example (only scenario_id and timestep)
                    self.examples.append({
                        'scenario_id': scenario_id,
                        'timestep': t
                    })
                
                # Print progress periodically
                if scenario_idx > 0 and scenario_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Preloaded {scenario_idx}/{len(self.scenario_files)} scenarios ({elapsed:.2f}s)")
                    
            except Exception as e:
                print(f"Error processing scenario {scenario_id}: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"Preloaded {len(self.preloaded_scenarios)} scenarios with {len(self.examples)} examples in {total_time:.2f}s")
        # Print the determined original weather feature count
        if self.original_weather_feature_count is not None:
            print(f"Determined original weather features: {self.original_weather_feature_count}")
            print(f"Using {self.num_weather_timesteps} timesteps, concatenated features: {self.concatenated_weather_feature_count}")
        else:
            print("Warning: Could not determine original weather feature count.")
    
    def _prepare_tensor_cache(self):
        """Pre-compute burn map tensors for faster retrieval"""
        # Cache burn map tensors for each timestep used in the dataset
        print("Building tensor cache for faster data loading...")
        start_time = time.time()
        
        # Get all unique timesteps
        timesteps = set(example['timestep'] for example in self.examples)
        
        # Pre-compute burn map tensors
        for ts in timesteps:
            baseline_burn = self.burn_map[ts]
            # Convert to torch tensor and add channel dimension
            baseline_tensor = torch.from_numpy(np.ascontiguousarray(baseline_burn, dtype=np.float32)).float().unsqueeze(0)
            self.cached_tensors[ts] = baseline_tensor
        
        print(f"Built tensor cache for {len(timesteps)} timesteps in {time.time() - start_time:.2f}s")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a single training example"""
        example = self.examples[idx]
        timestep = example['timestep']
        scenario_id = example['scenario_id']
        
        # Retrieve preloaded weather data list for the scenario
        all_weather_data = self.preloaded_weather[scenario_id]
        
        # --- Collect weather history ---
        weather_history = []
        # Pad with zeros if history goes beyond available data (e.g., t=0)
        zero_padding_weather = np.zeros(self.original_weather_feature_count, dtype=np.float32) 
        
        for k in range(self.num_weather_timesteps):
            current_t = timestep - k
            if current_t >= 0 and current_t < len(all_weather_data):
                weather_history.append(all_weather_data[current_t].astype(np.float32))
            else:
                # Append zero padding if timestep is out of bounds
                weather_history.append(zero_padding_weather)
        
        # Concatenate history (newest timestep t first, oldest t-4 last)
        weather_input_np = np.concatenate(weather_history)
        weather_input = torch.from_numpy(weather_input_np) # Shape: [num_weather_timesteps * original_features]
        # --- End weather history collection ---

        # Get scenario data from preloaded dictionary
        scenario_data = self.preloaded_scenarios[scenario_id]
        
        # Get ground truth for timestep
        ground_truth = scenario_data[timestep]
        
        # Get baseline from cache
        baseline_input = self.cached_tensors[timestep]
        
        # Create target tensor (ground truth)
        target = torch.from_numpy(np.ascontiguousarray(ground_truth)).float().unsqueeze(0)
        
        # Calculate residual (ground truth - baseline)
        baseline_np = baseline_input.squeeze(0).numpy()
        residual_np = ground_truth - baseline_np
        residual = torch.from_numpy(np.ascontiguousarray(residual_np)).float().unsqueeze(0)
        
        return {
            'baseline_input': baseline_input,
            'weather_input': weather_input,  # Now contains concatenated history
            'target': target,
            'residual': residual  # Include the residual in the output
        }

# Super simple model to start with and make sure we can fit the data
class SimpleResidualCNN(nn.Module):
    def __init__(self, input_shape, weather_features_concatenated):
        """
        Simple Residual CNN model for wildfire prediction
        
        Args:
            input_shape: The shape of the input (height, width)
            weather_features_concatenated: Total number of weather features after concatenating history
        """
        super(SimpleResidualCNN, self).__init__()
        
        self.height, self.width = input_shape
        
        # Larger weather branch to process concatenated weather history
        self.weather_dense = nn.Sequential(
            nn.Linear(weather_features_concatenated, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # CNN for processing baseline burn map
        self.baseline_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
        )
        
        # Process combined features
        # Input channels: 32 (from baseline_cnn) + 128 (from weather_dense)
        self.combined_cnn = nn.Sequential(
            nn.Conv2d(32 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            
            # Residual block 1
            self._make_residual_block(64),
            
            # Residual block 2
            self._make_residual_block(64),
            
            # Output a single channel (fire probability)
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )
    
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            
        )
    
    def forward(self, baseline_input, weather_input):
        """
        Forward pass
        
        Args:
            baseline_input: Baseline burn map [B, 1, H, W]
            weather_input: Concatenated weather features history [B, F_concatenated]
            
        Returns:
            Predicted fire probability [B, 1, H, W]
        """
        # Process baseline map
        x = self.baseline_cnn(baseline_input)
        
        # Process weather data
        batch_size = baseline_input.size(0)
        h, w = self.height, self.width
        
        w_features = self.weather_dense(weather_input)  # [B, 128]
        
        # Reshape weather to match spatial dimensions [B, 128, H, W]
        w_repeated = w_features.unsqueeze(2).unsqueeze(3).expand(batch_size, 128, h, w)
        
        # Concatenate features along channel dimension
        combined = torch.cat([x, w_repeated], dim=1)
        
        # Process combined features
        output_raw = self.combined_cnn(combined)
        
        # Ensure output is between 0 and 1 (fire probability)
        output = (torch.tanh(output_raw) + 1) / 2
        
        # Calculate residual as the difference from baseline
        residual = output - baseline_input
        
        # Return both the output probability and the residual
        return output, residual

class HubertLoss(nn.Module):
    def __init__(self, delta=1.0, balance_weight=None, reduction='mean', log_interval=100):
        """
        Hubert Loss for wildfire prediction
        
        Args:
            delta: Threshold at which to change from quadratic to linear (default: 1.0)
            balance_weight: Weight for positive class (fire pixels)
                         If None, it will be calculated from the batch
            reduction: 'mean' or 'sum'
            log_interval: How often to log statistics (in batches)
        """
        super(HubertLoss, self).__init__()
        self.delta = delta
        self.balance_weight = balance_weight
        self.reduction = reduction
        self.log_interval = log_interval
        self.batch_count = 0
    
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs: Predicted probabilities [B, 1, H, W]
            targets: Ground truth labels [B, 1, H, W]
            
        Returns:
            loss: Hubert loss value
        """
        # Calculate class balance statistics for logging
        if self.batch_count % self.log_interval == 0:
            pos_count = torch.sum(targets > 0.5)
            neg_count = torch.sum(targets <= 0.5)
            
            if pos_count == 0:
                pos_count = 1
            if neg_count == 0:
                neg_count = 1
                
            pos_ratio = pos_count.float() / (pos_count.float() + neg_count.float())
            print(f"  Class balance - Positive pixels: {pos_count.item():.0f} ({pos_ratio.item()*100:.2f}%), "
                  f"Negative pixels: {neg_count.item():.0f}, Hubert δ: {self.delta:.1f}")
        
        self.batch_count += 1
        
        # Calculate weight for balancing positive/negative examples
        weight = None
        if self.balance_weight is None:
            # Count positive and negative pixels
            pos_count = torch.sum(targets > 0.5).float()
            neg_count = torch.sum(targets <= 0.5).float()
            
            # Avoid division by zero
            if pos_count == 0:
                pos_count = 1.0
            if neg_count == 0:
                neg_count = 1.0
            
            # Create a weight tensor for class balance
            weight = torch.ones_like(targets)
            pos_weight = neg_count / pos_count
            weight[targets > 0.5] = pos_weight.clamp(min=1.0, max=10.0)
        else:
            # Use provided weight
            weight = torch.ones_like(targets)
            weight[targets > 0.5] = self.balance_weight
        
        # Calculate the absolute error
        abs_error = torch.abs(inputs - targets)
        
        # Apply the Huber Loss formula
        # For small errors: 0.5 * error^2
        # For large errors: delta * (error - 0.5 * delta)
        huber_loss = torch.where(
            abs_error <= self.delta,
            0.5 * abs_error * abs_error,
            self.delta * (abs_error - 0.5 * self.delta)
        )
        
        # Apply weighting for class balance
        if weight is not None:
            huber_loss = huber_loss * weight
        
        # Apply reduction
        if self.reduction == 'mean':
            return huber_loss.mean()
        elif self.reduction == 'sum':
            return huber_loss.sum()
        else:
            return huber_loss

class AUCLoss(nn.Module):
    """
    AUC Loss: Directly optimize the Area Under the ROC Curve
    
    This loss uses a smooth approximation of the AUC metric as a loss function
    to directly optimize for ranking performance.
    """
    def __init__(self, epsilon=0.01, sample_size=1000, reduction='mean', log_interval=100):
        """
        Initialize AUC Loss
        
        Args:
            epsilon: Smoothing parameter for the sigmoid approximation (default: 0.01)
            sample_size: Number of pairs to sample for approximating AUC (default: 1000)
            reduction: 'mean' or 'sum'
            log_interval: How often to log statistics (in batches)
        """
        super(AUCLoss, self).__init__()
        self.epsilon = epsilon
        self.sample_size = sample_size
        self.reduction = reduction
        self.log_interval = log_interval
        self.batch_count = 0
        
    def _smooth_sigmoid(self, x):
        """
        Smooth approximation of step function using sigmoid with numerical stability improvements
        
        Args:
            x: Input tensor
            
        Returns:
            Sigmoid of x/epsilon with clipping to prevent numerical issues
        """
        # Clip values to avoid extreme values that could cause overflow
        x_safe = torch.clamp(x / self.epsilon, min=-15.0, max=15.0)
        return 1.0 / (1.0 + torch.exp(-x_safe))
    
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs: Predicted probabilities [B, 1, H, W]
            targets: Ground truth labels [B, 1, H, W]
            
        Returns:
            loss: 1 - AUC approximation (for minimization)
        """
        # Log statistics occasionally
        if self.batch_count % self.log_interval == 0:
            pos_count = torch.sum(targets > 0.5)
            neg_count = torch.sum(targets <= 0.5)
            
            if pos_count == 0:
                pos_count = 1
            if neg_count == 0:
                neg_count = 1
                
            pos_ratio = pos_count.float() / (pos_count.float() + neg_count.float())
            print(f"  Class balance - Positive pixels: {pos_count.item():.0f} ({pos_ratio.item()*100:.2f}%), "
                  f"Negative pixels: {neg_count.item():.0f}, AUC Loss ε: {self.epsilon:.4f}")
        
        self.batch_count += 1
        
        # Flatten inputs and targets
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Get positive and negative samples
        pos_indices = (targets_flat > 0.5).nonzero(as_tuple=True)[0]
        neg_indices = (targets_flat <= 0.5).nonzero(as_tuple=True)[0]
        
        # Skip if we don't have both positive and negative samples
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            # Return small constant loss that can still backprop
            return torch.tensor(0.1, device=inputs.device, requires_grad=True)
        
        # Sample pairs for efficiency if there are too many
        num_pos = len(pos_indices)
        num_neg = len(neg_indices)
        
        # Determine how many pairs to sample
        num_pairs = min(self.sample_size, num_pos * num_neg)
        
        try:
            if num_pairs < num_pos * num_neg:
                # Random sampling of positive indices
                pos_sample_indices = torch.randint(0, num_pos, (num_pairs,), device=inputs.device)
                sampled_pos_indices = pos_indices[pos_sample_indices]
                
                # Random sampling of negative indices
                neg_sample_indices = torch.randint(0, num_neg, (num_pairs,), device=inputs.device)
                sampled_neg_indices = neg_indices[neg_sample_indices]
                
                # Get predictions for sampled pairs
                pos_preds = inputs_flat[sampled_pos_indices]
                neg_preds = inputs_flat[sampled_neg_indices]
            else:
                # For smaller datasets, use careful batch processing to avoid OOM
                # We'll sample a subset of pairs instead of all combinations
                max_safe_pairs = min(1000000, num_pos * num_neg)  # Limit total pairs to avoid OOM
                
                if max_safe_pairs < num_pos * num_neg:
                    pos_sample_indices = torch.randint(0, num_pos, (max_safe_pairs,), device=inputs.device)
                    neg_sample_indices = torch.randint(0, num_neg, (max_safe_pairs,), device=inputs.device)
                    
                    pos_preds = inputs_flat[pos_indices[pos_sample_indices]]
                    neg_preds = inputs_flat[neg_indices[neg_sample_indices]]
                else:
                    # Safe to use all pairs for small datasets
                    pos_preds = inputs_flat[pos_indices].repeat_interleave(num_neg)
                    neg_preds = inputs_flat[neg_indices].repeat(num_pos)
            
            # Calculate pairwise differences between positive and negative predictions
            # Clamp the differences to avoid extreme values
            diff = torch.clamp(pos_preds - neg_preds, min=-10.0, max=10.0)
            
            # Apply sigmoid approximation of step function
            # Adding a small epsilon to avoid exact zeros
            auc_approx = self._smooth_sigmoid(diff).mean()
            
            # Ensure the loss is within valid range and doesn't become NaN
            if torch.isnan(auc_approx) or torch.isinf(auc_approx):
                # Return a reasonable default loss value if we hit NaN
                return torch.tensor(0.5, device=inputs.device, requires_grad=True)
            
            # Return 1 - AUC for minimization (clamped for stability)
            loss = 1.0 - auc_approx.clamp(min=0.0, max=1.0)
            
            return loss
            
        except Exception as e:
            # Handle any unforeseen errors by returning a default loss value
            print(f"Error in AUC loss calculation: {e}")
            return torch.tensor(0.5, device=inputs.device, requires_grad=True)
            

def train_simple_residual_model(folder_path, model_output_path, epochs=10, batch_size=32, 
                               start_scenario=0, max_scenarios=100, timestep_sample_rate=1,
                               patience=3, min_delta=0.001, reduce_lr=True, tensorboard=False,
                               hubert_delta=1.0, AUC_fit=False, auc_epsilon=0.01, num_weather_timesteps=5):
    """
    Train a Simple Residual CNN model
    
    Args:
        folder_path: Path to data folder
        model_output_path: Path to save trained model
        epochs: Number of epochs to train
        batch_size: Batch size for training
        start_scenario: Starting scenario index for training set
        max_scenarios: Maximum number of scenarios to use for training
        timestep_sample_rate: Sample every nth timestep 
        patience: Number of epochs to wait before early stopping
        min_delta: Minimum improvement required for early stopping
        reduce_lr: Whether to use learning rate reduction on plateau
        tensorboard: Whether to use TensorBoard logging
        hubert_delta: Delta parameter for Hubert loss
        AUC_fit: Whether to use AUC loss
        auc_epsilon: Epsilon parameter for AUC loss
        num_weather_timesteps: Number of past weather timesteps to include
    
    Returns:
        Trained model and history of training metrics
    """
    print(f"Training simple residual CNN model with PyTorch for {folder_path}")

    start_time = time.time()
    
    # Configure device
    device, using_gpu, gpu_name = configure_device()
    
    if using_gpu:
        print(f"✅ GPU acceleration is ENABLED using: {gpu_name}")
    else:
        print("⚠️ GPU acceleration is DISABLED - training will use CPU only (slower)")

    print(f"Setting up GPU took {time.time() - start_time:.2f} seconds")
    
    # Load burn map (baseline)
    burn_map_path = os.path.join(folder_path, 'burn_map.npy')
    burn_map = np.load(burn_map_path)
    time_burn_map = time.time()
    print(f"Loaded burn map from {burn_map_path}, shape: {burn_map.shape}\n Took {time_burn_map-start_time:.2f} seconds")
    
    # Get grid shape from burn map
    grid_shape = burn_map.shape[1:3]  # (height, width)
    
    # Get scenario files
    scenarii_folder = os.path.join(folder_path, 'scenarii')
    scenario_files = sorted(glob.glob(os.path.join(scenarii_folder, '*.npy')))
    
    # Check if we have enough scenarios
    if len(scenario_files) < (start_scenario + max_scenarios):
        print(f"Warning: Requested {max_scenarios} scenarios starting from {start_scenario}, "
              f"but only found {len(scenario_files) - start_scenario} available")
        max_scenarios = min(max_scenarios, len(scenario_files) - start_scenario)
    
    # Select scenarios based on start_scenario and max_scenarios
    selected_scenario_files = scenario_files[start_scenario:start_scenario + max_scenarios]
    time_scenario_files = time.time()
    print(f"Selected {len(selected_scenario_files)} scenarios for training\n Took {time_scenario_files-time_burn_map:.2f} seconds")
    
    weather_folder = get_weather_folder(folder_path)
    
    # Create dataset
    dataset = WildfireDataset(
        burn_map=burn_map,
        scenario_files=selected_scenario_files,
        weather_folder=weather_folder,
        timestep_sample_rate=timestep_sample_rate,
        num_weather_timesteps=num_weather_timesteps
    )
    time_dataset = time.time()
    print(f"Created dataset\n Took {time_dataset-time_scenario_files:.2f} seconds")
    
    # Check if we have enough data
    if len(dataset) == 0:
        print("No training data generated")
        return None, None
    
    # Split into training and validation sets
    val_size = int(len(dataset) * 0.1)  # 10% for validation
    train_size = len(dataset) - val_size
    
    # Ensure we have at least one sample in validation set if dataset is very small
    if val_size == 0 and len(dataset) > 0:
        val_size = 1
        train_size = len(dataset) - 1
        
    if train_size <= 0:
        print("Error: Not enough data to create a training set after validation split.")
        return None, None
        
    # Use generator for deterministic split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    time_split = time.time()
    print(f"Split dataset into training ({len(train_dataset)}) and validation ({len(val_dataset)}) sets\n Took {time_split-time_dataset:.2f} seconds")
    
    # Determine optimal number of workers based on CPU cores
    num_workers = min(8, os.cpu_count() or 4)
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        num_workers = min(6, num_workers)  # Limit to 6 for Apple Silicon
    
    print(f"Using {num_workers} workers for data loading")
    
    # Create data loader settings based on hardware
    loader_settings = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': using_gpu,
        'pin_memory_device': str(device) if using_gpu and hasattr(torch, 'pin_memory_device') else None,
        'prefetch_factor': 2 if num_workers > 0 else None,
        'persistent_workers': True if num_workers > 0 else False,
    }
    
    # Remove None values which are not accepted by DataLoader
    loader_settings = {k: v for k, v in loader_settings.items() if v is not None}
    
    # Create training data loader
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True,
        drop_last=True,  # Drop last to avoid batch size of 1 during training
        **loader_settings
    )
    time_train_loader = time.time()
    print(f"Created train loader\n Took {time_train_loader-time_split:.2f} seconds")
    
    # Create validation data loader
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        drop_last=False,  # Don't drop last for validation
        **loader_settings
    )
    time_val_loader = time.time()
    print(f"Created val loader\n Took {time_val_loader-time_train_loader:.2f} seconds")
    
    print(f"Training dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    
    # Create model
    model = SimpleResidualCNN(
        grid_shape, 
        weather_features_concatenated=dataset.concatenated_weather_feature_count
    )
    
    # --- Multi-GPU Handling --- 
    if using_gpu and torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using DataParallel for multi-GPU training.")
        model = nn.DataParallel(model)
    # --- End Multi-GPU Handling ---
        
    model.to(device)
    time_model = time.time()
    print(f"Created model and moved to device: {device}\n Took {time_model-time_val_loader:.2f} seconds")
    
    # Define loss functions
    if AUC_fit:
        loss_fn = AUCLoss(epsilon=auc_epsilon)
        print(f"Using AUC Loss (epsilon={auc_epsilon}) to directly optimize AUC")
    else:
        loss_fn = HubertLoss(delta=hubert_delta)
        print(f"Using Hubert Loss (delta={hubert_delta}) to handle prediction errors")
    
    mse_loss = nn.MSELoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler if requested
    scheduler = None
    if reduce_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=patience//2,
            min_lr=0.00001,
            verbose=True
        )
    
    writer = None
    
    # Create directories for outputs
    os.makedirs('models/logs', exist_ok=True)
    os.makedirs('models/visualizations', exist_ok=True)
    
    # Early stopping setup
    best_val_loss = float('inf')  # Track best validation loss instead of AP
    best_ap_improvement = float('-inf')  # Track the best AP improvement
    early_stop_counter = 0
    best_model_state = None
    best_ap_improvement_model_state = None  # Store the model with best AP improvement
    best_ap_improvement_model_path = None  # Initialize the path variable
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_output_loss': [],
        'val_output_loss': [],
        'train_residual_loss': [],
        'val_residual_loss': [],
        'train_ap': [],
        'val_ap': [],
        'val_baseline_ap': [],
        'val_improvement': [],
        'learning_rate': []
    }
    
    # Training loop
    print("\nStarting model training...")
    print(f"Training for up to {epochs} epochs with early stopping (patience={patience})")
    
    # Record start time
    start_time_training = time.time()
    combined_fwd_time = 0
    combined_bwd_time = 0
    combined_load_time = 0
    combined_cpu_to_gpu_time = 0
    combined_gpu_to_cpu_time = 0
    combined_store_time = 0
    combined_calculate_metrics_time = 0
    
    # Enable asynchronous data copying (good for MPS and CUDA)
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    
    # Use non-blocking transfers for better GPU utilization
    non_blocking = using_gpu
    
    # For pixel-level metrics (now kept consistent with evaluation)
    max_samples_for_metrics = 10000  # Maximum samples to use for AP calculation
    
    for epoch in range(epochs):
        #
        # Training phase
        #
        model.train()
        train_loss = 0.0
        train_output_loss = 0.0
        train_residual_loss = 0.0
        
        # Track per-example metrics (for backwards compatibility)
        train_metrics_tracker = {
            'model_ap_sum': 0.0,
            'model_ap_min': 0.0,
            'model_ap_max': 0.0,
            'model_ap_sum_squared': 0.0,
            'baseline_ap_sum': 0.0,
            'baseline_ap_min': 0.0,
            'baseline_ap_max': 0.0,
            'baseline_ap_sum_squared': 0.0,
            'count': 0,
            'improvement_sum': 0.0
        }
        
        # For timestep metrics
        train_timestep_metrics = {}
        
        # For pixel-level metrics using reservoir sampling - PRIMARY metrics for consistent reporting
        train_sample_predictions = np.zeros(max_samples_for_metrics, dtype=np.float32)
        train_sample_labels = np.zeros(max_samples_for_metrics, dtype=np.float32)
        train_sample_baselines = np.zeros(max_samples_for_metrics, dtype=np.float32)
        train_samples_seen = 0
        
        time_load_start = time.time()
        # Use a progress bar that doesn't require frequent updates (better performance)
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} Training", 
                          bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                          mininterval=0.5)
        
        # Iterate through batches
        for batch_idx, batch in enumerate(train_loader):
            time_load_end = time.time()
            combined_load_time += time_load_end - time_load_start
            
            # Transfer data to device
            start_cpu_to_gpu = time.time()
            baseline_input = batch['baseline_input'].to(device, non_blocking=non_blocking)
            weather_input = batch['weather_input'].to(device, non_blocking=non_blocking)
            target = batch['target'].to(device, non_blocking=non_blocking)
            residual_target = batch['residual'].to(device, non_blocking=non_blocking)
            time_cpu_to_gpu = time.time()
            combined_cpu_to_gpu_time += time_cpu_to_gpu - start_cpu_to_gpu
            
            # Get batch size
            batch_size = target.size(0)
            
            # Zero the gradients (use optimizer.zero_grad(set_to_none=True) for better performance)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            time_fwd_start = time.time()
            output, residual = model(baseline_input, weather_input)
            
            # Calculate losses
            # Calculate MSE between predicted residuals and actual residuals (for monitoring only)
            residual_loss = mse_loss(residual, residual_target)
            # Calculate loss between predicted fire probabilities and ground truth
            output_loss = loss_fn(output, target)  # Using either Hubert or AUC loss
            loss = output_loss #residual_loss # output_loss
            time_fwd_end = time.time()
            combined_fwd_time += time_fwd_end - time_fwd_start
            
            # Backward pass and optimize
            time_bwd_start = time.time()
            loss.backward()
            optimizer.step()
            time_bwd_end = time.time()
            combined_bwd_time += time_bwd_end - time_bwd_start
            
            # Update metrics
            train_loss += loss.item()
            train_output_loss += output_loss.item()
            train_residual_loss += residual_loss.item()
            
            # Collect data for metrics calculation
            time_store_start = time.time()
            with torch.no_grad():
                # Efficient data movement to CPU as numpy array
                output_np = output.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                baseline_np = baseline_input.detach().cpu().numpy()
                time_gpu_to_cpu = time.time()
                combined_gpu_to_cpu_time += time_gpu_to_cpu - time_store_start
                
                # Process each example in the batch
                for i in range(output_np.shape[0]):
                    # Get data for this example
                    pred_squeezed = output_np[i].squeeze(0) if output_np[i].ndim == 3 and output_np[i].shape[0] == 1 else output_np[i]
                    gt_squeezed = target_np[i].squeeze(0) if target_np[i].ndim == 3 and target_np[i].shape[0] == 1 else target_np[i]
                    baseline_squeezed = baseline_np[i].squeeze(0) if baseline_np[i].ndim == 3 and baseline_np[i].shape[0] == 1 else baseline_np[i]
                    
                    # Get example index for timestep tracking
                    example_idx = batch_idx * train_loader.batch_size + i
                    timestep = None
                    if example_idx < len(train_dataset):
                        # Try to get the timestep - handle potential dataset wrapping 
                        try:
                            # Get the original index in the dataset (accounting for random split)
                            train_indices = train_dataset.indices
                            original_idx = train_indices[example_idx] if hasattr(train_dataset, 'indices') else example_idx
                            timestep = dataset.examples[original_idx]['timestep']
                        except (AttributeError, IndexError):
                            # If we can't get the timestep, continue without it
                            pass
                    
                    # Calculate metrics for this example
                    time_calculate_metrics_start = time.time()
                    metrics = calculate_metrics(
                        predictions=pred_squeezed,
                        ground_truth=gt_squeezed,
                        baseline=baseline_squeezed
                    )
                    time_calculate_metrics_end = time.time()
                    combined_calculate_metrics_time += time_calculate_metrics_end - time_calculate_metrics_start
                    
                    # Update metrics tracker
                    train_metrics_tracker['model_ap_sum'] += metrics['model']['ap']
                    train_metrics_tracker['baseline_ap_sum'] += metrics['baseline']['ap']
                    train_metrics_tracker['improvement_sum'] += metrics['model']['ap'] - metrics['baseline']['ap']
                    train_metrics_tracker['count'] += 1
                    train_metrics_tracker['model_ap_sum_squared'] += metrics['model']['ap'] ** 2
                    train_metrics_tracker['baseline_ap_sum_squared'] += metrics['baseline']['ap'] ** 2
                    train_metrics_tracker['model_ap_min'] = min(train_metrics_tracker['model_ap_min'], metrics['model']['ap'])
                    train_metrics_tracker['baseline_ap_min'] = min(train_metrics_tracker['baseline_ap_min'], metrics['baseline']['ap'])
                    train_metrics_tracker['model_ap_max'] = max(train_metrics_tracker['model_ap_max'], metrics['model']['ap'])
                    train_metrics_tracker['baseline_ap_max'] = max(train_metrics_tracker['baseline_ap_max'], metrics['baseline']['ap'])
                    
                    # Update timestep metrics if available
                    if timestep is not None:
                        if timestep not in train_timestep_metrics:
                            train_timestep_metrics[timestep] = {
                                'model_ap_sum': 0.0,
                                'baseline_ap_sum': 0.0,
                                'count': 0
                            }
                        train_timestep_metrics[timestep]['model_ap_sum'] += metrics['model']['ap']
                        train_timestep_metrics[timestep]['baseline_ap_sum'] += metrics['baseline']['ap']
                        train_timestep_metrics[timestep]['count'] += 1
                    
                    # Also sample data for overall metrics (pixel-level reservoir sampling)
                    output_flat = output_np[i].flatten()
                    target_flat = target_np[i].flatten()
                    baseline_flat = baseline_np[i].flatten()
                    
                    # Sample pixels from this example for overall metrics
                    for j in range(min(100, len(output_flat))):
                        idx = np.random.randint(0, len(output_flat))
                        
                        if train_samples_seen < max_samples_for_metrics:
                            # Fill reservoir until max samples reached
                            train_sample_predictions[train_samples_seen] = output_flat[idx]
                            train_sample_baselines[train_samples_seen] = baseline_flat[idx]
                            train_sample_labels[train_samples_seen] = 1 if target_flat[idx] > 0.5 else 0
                        else:
                            # Use reservoir sampling once we exceed max samples
                            pos = np.random.randint(0, train_samples_seen + 1)
                            if pos < max_samples_for_metrics:
                                train_sample_predictions[pos] = output_flat[idx]
                                train_sample_baselines[pos] = baseline_flat[idx]
                                train_sample_labels[pos] = 1 if target_flat[idx] > 0.5 else 0
                        
                        train_samples_seen += 1
            
            time_store_end = time.time()
            combined_store_time += time_store_end - time_store_start
            
            # Update progress bar
            progress_bar.update(1)
            
            # Start loading next batch
            time_load_start = time.time()
        
        # Close progress bar
        progress_bar.close()
        
        # Timing for out-of-loop operations
        time_out_of_loop = time.time()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        train_output_loss /= len(train_loader)
        train_residual_loss /= len(train_loader)
        
        # --- IMPORTANT CHANGE: Always prioritize pixel-level metrics ---
        
        # Initialize fallback values (only used if pixel sampling fails)
        train_pixel_model_ap = 0.5
        train_pixel_baseline_ap = 0.5
        train_pixel_improvement = 0.0
        
        # Calculate pixel-level metrics from sampled data - PRIMARY metrics
        if train_samples_seen > 0:
            # Filter out any NaN values
            mask = ~np.isnan(train_sample_predictions[:min(train_samples_seen, max_samples_for_metrics)]) & \
                   ~np.isnan(train_sample_labels[:min(train_samples_seen, max_samples_for_metrics)])
            
            filtered_preds = train_sample_predictions[:min(train_samples_seen, max_samples_for_metrics)][mask]
            filtered_labels = train_sample_labels[:min(train_samples_seen, max_samples_for_metrics)][mask]
            filtered_baselines = train_sample_baselines[:min(train_samples_seen, max_samples_for_metrics)][mask]
            
            # Ensure binary labels for AP calculation - CRITICAL for consistency with evaluation
            binary_labels = (filtered_labels > 0.5).astype(np.int32)
            
            # Calculate overall metrics
            if len(filtered_preds) > 10 and np.sum(binary_labels > 0) > 0 and np.sum(binary_labels == 0) > 0:
                train_overall_metrics = calculate_metrics(
                    predictions=filtered_preds,
                    ground_truth=binary_labels,
                    baseline=filtered_baselines
                )
                train_pixel_model_ap = train_overall_metrics['model']['ap']
                train_pixel_baseline_ap = train_overall_metrics['baseline']['ap']
                train_pixel_improvement = train_pixel_model_ap - train_pixel_baseline_ap
            elif train_metrics_tracker['count'] > 0:
                # Fall back to per-example metrics only if pixel sampling fails and we have example metrics
                train_pixel_model_ap = train_metrics_tracker['model_ap_sum'] / train_metrics_tracker['count']
                train_pixel_baseline_ap = train_metrics_tracker['baseline_ap_sum'] / train_metrics_tracker['count']
                train_pixel_improvement = train_metrics_tracker['improvement_sum'] / train_metrics_tracker['count']
        elif train_metrics_tracker['count'] > 0:
            # Fall back to per-example metrics only if no pixel samples
            train_pixel_model_ap = train_metrics_tracker['model_ap_sum'] / train_metrics_tracker['count']
            train_pixel_baseline_ap = train_metrics_tracker['baseline_ap_sum'] / train_metrics_tracker['count']
            train_pixel_improvement = train_metrics_tracker['improvement_sum'] / train_metrics_tracker['count']
            
        # Always use pixel-level metrics for reporting and history (for consistency with evaluation)
        train_ap = train_pixel_model_ap
        train_baseline_ap = train_pixel_baseline_ap
        train_improvement = train_pixel_improvement
        
        # Validation phase with the same approach
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_output_loss = 0.0
        val_residual_loss = 0.0
        
        # Track per-example metrics for validation
        val_metrics_tracker = {
            'model_ap_sum': 0.0,
            'model_ap_min': 0.0,
            'model_ap_max': 0.0,
            'model_ap_sum_squared': 0.0,
            'baseline_ap_sum': 0.0,
            'baseline_ap_min': 0.0,
            'baseline_ap_max': 0.0,
            'baseline_ap_sum_squared': 0.0,
            'count': 0,
            'improvement_sum': 0.0
        }
        
        # For timestep metrics in validation
        val_timestep_metrics = {}
        
        # For pixel-level metrics using reservoir sampling in validation
        val_sample_predictions = np.zeros(max_samples_for_metrics, dtype=np.float32)
        val_sample_labels = np.zeros(max_samples_for_metrics, dtype=np.float32)
        val_sample_baselines = np.zeros(max_samples_for_metrics, dtype=np.float32)
        val_samples_seen = 0
        
        # Time tracking for validation
        time_val_start = time.time()
        
        # Validation loop - no gradient updates
        with torch.no_grad():
            # Use progress bar for validation
            val_progress_bar = tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} Validation", 
                                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                                 mininterval=0.5)
            
            # Process validation batches
            for batch_idx, batch in enumerate(val_loader):
                # Transfer data to device
                baseline_input = batch['baseline_input'].to(device, non_blocking=non_blocking)
                weather_input = batch['weather_input'].to(device, non_blocking=non_blocking)
                target = batch['target'].to(device, non_blocking=non_blocking)
                residual_target = batch['residual'].to(device, non_blocking=non_blocking)
                
                # Forward pass only (no backprop in validation)
                output, residual = model(baseline_input, weather_input)
                
                # Calculate losses for monitoring
                residual_loss = mse_loss(residual, residual_target)
                output_loss = loss_fn(output, target)
                loss = output_loss
                
                # Update metrics
                val_loss += loss.item()
                val_output_loss += output_loss.item()
                val_residual_loss += residual_loss.item()
                
                # Process validation data for metrics
                output_np = output.cpu().numpy()
                target_np = target.cpu().numpy()
                baseline_np = baseline_input.cpu().numpy()
                
                # Process each example in the batch
                for i in range(output_np.shape[0]):
                    # Get data for this example
                    pred_squeezed = output_np[i].squeeze(0) if output_np[i].ndim == 3 and output_np[i].shape[0] == 1 else output_np[i]
                    gt_squeezed = target_np[i].squeeze(0) if target_np[i].ndim == 3 and target_np[i].shape[0] == 1 else target_np[i]
                    baseline_squeezed = baseline_np[i].squeeze(0) if baseline_np[i].ndim == 3 and baseline_np[i].shape[0] == 1 else baseline_np[i]
                    
                    # Get example index for timestep tracking
                    example_idx = batch_idx * val_loader.batch_size + i
                    timestep = None
                    if example_idx < len(val_dataset):
                        try:
                            # Get the original index in the dataset (accounting for random split)
                            val_indices = val_dataset.indices
                            original_idx = val_indices[example_idx] if hasattr(val_dataset, 'indices') else example_idx
                            timestep = dataset.examples[original_idx]['timestep']
                        except (AttributeError, IndexError):
                            pass
                    
                    # Calculate metrics for this validation example
                    metrics = calculate_metrics(
                        predictions=pred_squeezed,
                        ground_truth=gt_squeezed,
                        baseline=baseline_squeezed
                    )
                    
                    # Update metrics tracker
                    val_metrics_tracker['model_ap_sum'] += metrics['model']['ap']
                    val_metrics_tracker['baseline_ap_sum'] += metrics['baseline']['ap']
                    val_metrics_tracker['improvement_sum'] += metrics['model']['ap'] - metrics['baseline']['ap']
                    val_metrics_tracker['count'] += 1
                    val_metrics_tracker['model_ap_sum_squared'] += metrics['model']['ap'] ** 2
                    val_metrics_tracker['baseline_ap_sum_squared'] += metrics['baseline']['ap'] ** 2
                    val_metrics_tracker['model_ap_min'] = min(val_metrics_tracker['model_ap_min'], metrics['model']['ap'])
                    val_metrics_tracker['baseline_ap_min'] = min(val_metrics_tracker['baseline_ap_min'], metrics['baseline']['ap'])
                    val_metrics_tracker['model_ap_max'] = max(val_metrics_tracker['model_ap_max'], metrics['model']['ap'])
                    val_metrics_tracker['baseline_ap_max'] = max(val_metrics_tracker['baseline_ap_max'], metrics['baseline']['ap'])
                    
                    # Update timestep metrics if available
                    if timestep is not None:
                        if timestep not in val_timestep_metrics:
                            val_timestep_metrics[timestep] = {
                                'model_ap_sum': 0.0,
                                'baseline_ap_sum': 0.0,
                                'count': 0
                            }
                        val_timestep_metrics[timestep]['model_ap_sum'] += metrics['model']['ap']
                        val_timestep_metrics[timestep]['baseline_ap_sum'] += metrics['baseline']['ap']
                        val_timestep_metrics[timestep]['count'] += 1
                    
                    # Sample pixels from this example for overall metrics
                    output_flat = output_np[i].flatten()
                    target_flat = target_np[i].flatten()
                    baseline_flat = baseline_np[i].flatten()
                    
                    # Sample pixels from validation data
                    for j in range(min(100, len(output_flat))):
                        idx = np.random.randint(0, len(output_flat))
                        
                        if val_samples_seen < max_samples_for_metrics:
                            # Fill reservoir until max samples reached
                            val_sample_predictions[val_samples_seen] = output_flat[idx]
                            val_sample_baselines[val_samples_seen] = baseline_flat[idx]
                            val_sample_labels[val_samples_seen] = 1 if target_flat[idx] > 0.5 else 0
                        else:
                            # Use reservoir sampling once we exceed max samples
                            pos = np.random.randint(0, val_samples_seen + 1)
                            if pos < max_samples_for_metrics:
                                val_sample_predictions[pos] = output_flat[idx]
                                val_sample_baselines[pos] = baseline_flat[idx]
                                val_sample_labels[pos] = 1 if target_flat[idx] > 0.5 else 0
                        
                        val_samples_seen += 1
                
                # Update progress bar
                val_progress_bar.update(1)
            
            # Close progress bar
            val_progress_bar.close()
        
        # Time tracking for validation end
        time_val_end = time.time()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_output_loss /= len(val_loader)
        val_residual_loss /= len(val_loader)
        
        # Initialize fallback values (only used if pixel sampling fails)
        val_pixel_model_ap = 0.5
        val_pixel_baseline_ap = 0.5
        val_pixel_improvement = 0.0
        
        # Calculate pixel-level metrics from sampled data - PRIMARY metrics for consistency
        if val_samples_seen > 0:
            # Filter out any NaN values
            mask = ~np.isnan(val_sample_predictions[:min(val_samples_seen, max_samples_for_metrics)]) & \
                   ~np.isnan(val_sample_labels[:min(val_samples_seen, max_samples_for_metrics)])
            
            filtered_preds = val_sample_predictions[:min(val_samples_seen, max_samples_for_metrics)][mask]
            filtered_labels = val_sample_labels[:min(val_samples_seen, max_samples_for_metrics)][mask]
            filtered_baselines = val_sample_baselines[:min(val_samples_seen, max_samples_for_metrics)][mask]
            
            # Ensure binary labels for AP calculation - CRITICAL for consistency with evaluation
            binary_labels = (filtered_labels > 0.5).astype(np.int32)
            
            # Calculate overall metrics
            if len(filtered_preds) > 10 and np.sum(binary_labels > 0) > 0 and np.sum(binary_labels == 0) > 0:
                val_overall_metrics = calculate_metrics(
                    predictions=filtered_preds,
                    ground_truth=binary_labels,
                    baseline=filtered_baselines
                )
                val_pixel_model_ap = val_overall_metrics['model']['ap']
                val_pixel_baseline_ap = val_overall_metrics['baseline']['ap']
                val_pixel_improvement = val_pixel_model_ap - val_pixel_baseline_ap
            elif val_metrics_tracker['count'] > 0:
                # Fall back to per-example metrics only if pixel sampling fails and we have example metrics
                val_pixel_model_ap = val_metrics_tracker['model_ap_sum'] / val_metrics_tracker['count']
                val_pixel_baseline_ap = val_metrics_tracker['baseline_ap_sum'] / val_metrics_tracker['count']
                val_pixel_improvement = val_metrics_tracker['improvement_sum'] / val_metrics_tracker['count']
        elif val_metrics_tracker['count'] > 0:
            # Fall back to per-example metrics only if no pixel samples
            val_pixel_model_ap = val_metrics_tracker['model_ap_sum'] / val_metrics_tracker['count']
            val_pixel_baseline_ap = val_metrics_tracker['baseline_ap_sum'] / val_metrics_tracker['count'] 
            val_pixel_improvement = val_metrics_tracker['improvement_sum'] / val_metrics_tracker['count']
        
        # Always use pixel-level metrics for consistency with evaluation
        val_model_ap = val_pixel_model_ap
        val_baseline_ap = val_pixel_baseline_ap
        val_improvement = val_pixel_improvement
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} summary:")
        print(f"  Training   - Loss: {train_loss:.4f}, Residual Loss: {train_residual_loss:.4f}, Output Loss: {train_output_loss:.4f}, AP: {train_ap:.4f}")
        print(f"  Validation - Loss: {val_loss:.4f}, Residual Loss: {val_residual_loss:.4f}, Output Loss: {val_output_loss:.4f}, AP: {val_model_ap:.4f}")
        print(f"  Baseline AP: {val_baseline_ap:.4f}, AUPRC Improvement over baseline: {val_improvement:.4f}")
        print(f"  Load: {combined_load_time:.2f}s, Fwd: {combined_fwd_time:.2f}s, Bwd: {combined_bwd_time:.2f}s, Store: {combined_store_time:.2f}s, Cpu to Gpu: {combined_cpu_to_gpu_time:.2f}s, Gpu to Cpu: {combined_gpu_to_cpu_time:.2f}s, compute metrics: {combined_calculate_metrics_time:.2f}s, Train: {time_out_of_loop-start_time_training:.2f}s, Val: {time_val_end-time_val_start:.2f}s")
        
        # Update training history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_output_loss'].append(train_output_loss)
        history['val_output_loss'].append(val_output_loss)
        history['train_residual_loss'].append(train_residual_loss)
        history['val_residual_loss'].append(val_residual_loss)
        history['train_ap'].append(train_ap)
        history['val_ap'].append(val_model_ap)
        history['val_baseline_ap'].append(val_baseline_ap)
        history['val_improvement'].append(val_improvement)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Check if we should update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
            
        # Save current epoch model
        current_model_path = f"{model_output_path}_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), current_model_path)
        print(f"  Saved model checkpoint to {current_model_path}")
        
        # Check if this is the best model so far (based on validation loss)
        if val_loss < best_val_loss - min_delta:
            print(f"  Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # Save best model state
            best_model_state = model.state_dict().copy()
            
            # Save best model to disk
            best_model_path = f"{model_output_path}_best.pt"
            torch.save(best_model_state, best_model_path)
            print(f"  Saved best model to {best_model_path}")
        else:
            early_stop_counter += 1
            print(f"  Validation loss did not improve. Early stopping counter: {early_stop_counter}/{patience}")
        
        # Check if this is the best model in terms of AP improvement
        if val_improvement > best_ap_improvement:
            print(f"  AP improvement increased from {best_ap_improvement:.4f} to {val_improvement:.4f}")
            best_ap_improvement = val_improvement
            
            # Save best AP improvement model state
            best_ap_improvement_model_state = model.state_dict().copy()
            
            # Save best AP improvement model to disk
            best_ap_improvement_model_path = f"{model_output_path}_best_ap_improvement.pt"
            torch.save(best_ap_improvement_model_state, best_ap_improvement_model_path)
            print(f"  Saved best AP improvement model to {best_ap_improvement_model_path}")
        
        # Check for early stopping
        if early_stop_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Reset timers for next epoch
        combined_load_time = 0
        combined_fwd_time = 0
        combined_bwd_time = 0
        combined_cpu_to_gpu_time = 0
        combined_store_time = 0
        combined_gpu_to_cpu_time = 0
        combined_calculate_metrics_time = 0
    # Training complete, load best model if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (validation loss: {best_val_loss:.4f})")
    
    # Save final model
    final_model_path = f"{model_output_path}.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Note about model evaluation
    print("\nNote: The evaluation will use the model with the best AP improvement")
    print(f"Best AP improvement: {best_ap_improvement:.4f}")
    if best_ap_improvement_model_state is not None:
        print("Using model with best AP improvement for evaluation")
    else:
        print("Warning: No model with AP improvement was saved during training")
    
    # Save training history
    history_path = f"{model_output_path}_history.json"
    with open(history_path, 'w') as f:
        # Convert history values to Python native types for JSON
        serializable_history = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(serializable_history, f, indent=2)
    print(f"Saved training history to {history_path}")
    
    # Total training time
    total_training_time = time.time() - start_time_training
    print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    
    # Ensure we have a path for the best AP improvement model
    if best_ap_improvement_model_path is None:
        print("\nWarning: No AP improvement was recorded during training.")
        # Use the best validation loss model as a fallback
        if os.path.exists(f"{model_output_path}_best.pt"):
            best_ap_improvement_model_path = f"{model_output_path}_best.pt"
            print(f"Using best validation loss model as fallback: {best_ap_improvement_model_path}")
        else:
            # Use the final model as a last resort
            best_ap_improvement_model_path = final_model_path
            print(f"Using final model as fallback: {best_ap_improvement_model_path}")
    
    return model, history, best_ap_improvement_model_path, best_ap_improvement  # Return the best AP improvement value

def predict_with_simple_residual_model(model, burn_map, weather_features, timestep, device=None):
    """
    Predict fire risk using the simple residual CNN model
    
    Args:
        model: Trained PyTorch model
        burn_map: Burn map array
        weather_features: Weather features for this timestep
        timestep: Current timestep
        device: torch device
        
    Returns:
        Predicted fire risk map and residual correction
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get baseline burn map for this timestep
    baseline_burn = burn_map[timestep]
    
    # Convert to tensors and add batch and channel dimensions
    baseline_input = torch.from_numpy(baseline_burn).float().unsqueeze(0).unsqueeze(0).to(device)
    weather_input = torch.from_numpy(weather_features).float().unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        try:
            # Forward pass
            prediction, residual = model(baseline_input, weather_input)
            
            # Convert to numpy
            prediction = prediction.squeeze().cpu().numpy()
            residual = residual.squeeze().cpu().numpy()
            
            # Important: return the unmodified baseline for proper comparison
            return prediction, residual, baseline_burn
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback to returning baseline
            print("Returning baseline as prediction and zero residual")
            return baseline_burn, np.zeros_like(baseline_burn), baseline_burn

def calculate_metrics(predictions, ground_truth, baseline, threshold=0.5):
    """
    Calculate performance metrics for wildfire predictions
    
    Args:
        predictions: Predicted fire risk maps (binary or probability)
        ground_truth: Ground truth fire maps (binary)
        baseline: Baseline burn maps (binary or probability)
        threshold: Threshold for converting probability to binary (default: 0.5)
        
    Returns:
        Dictionary of metrics for both the model and baseline
    """
    # Ensure inputs are numpy arrays
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    baseline = np.asarray(baseline)
    
    # Convert to binary if needed
    pred_binary = predictions > threshold
    baseline_binary = baseline > threshold
    
    # Flatten arrays for metric calculation
    gt_flat = ground_truth.flatten()
    pred_flat = predictions.flatten()
    pred_binary_flat = pred_binary.flatten()
    baseline_flat = baseline.flatten()
    baseline_binary_flat = baseline_binary.flatten()
    
    # Ensure ground truth is binary for AP calculation
    gt_binary_flat = (gt_flat > threshold).astype(np.int32)
    
    # Calculate metrics for model predictions
    model_metrics = {}
    
    # AP
    try:
        # Filter out any NaN values
        mask = ~np.isnan(pred_flat) & ~np.isnan(gt_binary_flat)
        filtered_preds = pred_flat[mask]
        filtered_gt = gt_binary_flat[mask]
        
        # Check if we have enough samples for reliable metrics
        if len(filtered_preds) > 10 and np.sum(filtered_gt > 0) > 0 and np.sum(filtered_gt == 0) > 0:
            model_metrics['ap'] = average_precision_score(filtered_gt, filtered_preds)
        else:
            model_metrics['ap'] = np.mean(filtered_gt) if len(filtered_gt) > 0 else 0.5
            print(f"Warning: Not enough data for reliable AP calculation. Using defaults.")
    except Exception as e:
        print(f"Error calculating AP: {e}")
        model_metrics['ap'] = 0.5
    
    # Calculate metrics for baseline
    baseline_metrics = {}
    
    # AP for baseline
    try:
        # Filter out any NaN values
        mask = ~np.isnan(baseline_flat) & ~np.isnan(gt_binary_flat)
        filtered_baseline = baseline_flat[mask]
        filtered_gt = gt_binary_flat[mask]
        
        # Check if we have enough samples for reliable metrics
        if len(filtered_baseline) > 10 and np.sum(filtered_gt > 0) > 0 and np.sum(filtered_gt == 0) > 0:
            baseline_metrics['ap'] = average_precision_score(filtered_gt, filtered_baseline)
        else:
            baseline_metrics['ap'] = np.mean(filtered_gt) if len(filtered_gt) > 0 else 0.5
    except Exception as e:
        print(f"Error calculating baseline AP: {e}")
        baseline_metrics['ap'] = 0.5
    
    # Calculate improvement over baseline
    model_metrics['ap_improvement'] = model_metrics['ap'] - baseline_metrics['ap']
    
    # Return both model and baseline metrics
    return {
        'model': model_metrics,
        'baseline': baseline_metrics
    }

def visualize_prediction(prediction, ground_truth, baseline, residual=None, timestep=None, save_path=None):
    """
    Visualize model predictions compared to ground truth and baseline
    
    Args:
        prediction: Predicted fire risk map
        ground_truth: Ground truth fire map
        baseline: Baseline burn map
        residual: Residual correction (optional)
        timestep: Current timestep (optional)
        save_path: Path to save visualization (optional)
        
    Returns:
        Matplotlib figure
    """
    # Ensure inputs are numpy arrays
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    baseline = np.asarray(baseline)
    if residual is not None:
        residual = np.asarray(residual)
    
    # Remove channel dimension if present
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)
    if ground_truth.ndim == 3 and ground_truth.shape[0] == 1:
        ground_truth = ground_truth.squeeze(0)
    if baseline.ndim == 3 and baseline.shape[0] == 1:
        baseline = baseline.squeeze(0)
    if residual is not None and residual.ndim == 3 and residual.shape[0] == 1:
        residual = residual.squeeze(0)
    
    # Create figure
    if residual is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten()
    
    # Set title
    if timestep is not None:
        fig.suptitle(f"Fire Prediction - Timestep {timestep}", fontsize=16)
    else:
        fig.suptitle("Fire Prediction", fontsize=16)
    
    # Plot ground truth
    if residual is not None:
        im0 = axes[0, 0].imshow(ground_truth, cmap='hot', vmin=0, vmax=1)
        axes[0, 0].set_title("Ground Truth")
        axes[0, 0].axis('off')
        
        # Plot baseline
        im1 = axes[0, 1].imshow(baseline, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title("Baseline")
        axes[0, 1].axis('off')
        
        # Plot prediction
        im2 = axes[1, 0].imshow(prediction, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title("Model Prediction")
        axes[1, 0].axis('off')
        
        # Plot residual
        im3 = axes[1, 1].imshow(residual, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title("Residual Correction")
        axes[1, 1].axis('off')
        
        # Add colorbars
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    else:
        # Plot ground truth
        im0 = axes[0].imshow(ground_truth, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
        
        # Plot baseline
        im1 = axes[1].imshow(baseline, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title("Baseline")
        axes[1].axis('off')
        
        # Plot prediction
        im2 = axes[2].imshow(prediction, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title("Model Prediction")
        axes[2].axis('off')
        
        # Add colorbars
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def evaluate_simple_residual_model(folder_path, model_path, start_scenario=100, max_scenarios=5, 
                                 timestep_sample_rate=1, visualize=True, num_visualizations=3,
                                 num_weather_timesteps=5):
    """
    Evaluate a trained Simple Residual CNN model
    
    Args:
        folder_path: Path to dataset folder
        model_path: Path to model file (.pt)
        start_scenario: Starting index for scenarios (for test set)
        max_scenarios: Maximum number of scenarios to use (for test set)
        timestep_sample_rate: Sample every nth timestep
        visualize: Whether to generate visualizations
        num_visualizations: Number of examples to visualize
        num_weather_timesteps: Number of past weather timesteps to include
    
    Returns:
        Evaluation results (dictionary) or None if evaluation fails
    """
    print("\n=== Evaluating Simple Residual CNN model on test set ===")
    print(f"Using model weights from: {model_path}")
    print(f"Test scenarios {start_scenario} to {start_scenario + max_scenarios - 1}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    # Configure device
    device, using_gpu, gpu_name = configure_device()
    print(f"{'✅' if using_gpu else '❌'} GPU acceleration is {'ENABLED' if using_gpu else 'DISABLED'} using: {gpu_name}")
    start_time = time.time()
    
    # Load burn map
    try:
        burn_map_path = os.path.join(folder_path, 'burn_map.npy')
        burn_map = np.load(burn_map_path)
        print(f"Loaded burn map from {burn_map_path}, shape: {burn_map.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find burn map at {burn_map_path}")
        return None
    except Exception as e:
        print(f"Error loading burn map: {e}")
        return None
    
    # Determine grid shape from burn map
    grid_shape = (burn_map.shape[1], burn_map.shape[2])
    
    # Get all scenario files
    scenario_folder = os.path.join(folder_path, 'scenarii')
    all_scenario_files = sorted(glob.glob(os.path.join(scenario_folder, '*.npy')))
    
    # Select scenarios for evaluation
    if start_scenario >= len(all_scenario_files):
         print(f"Error: start_scenario ({start_scenario}) is out of bounds. Found {len(all_scenario_files)} scenarios.")
         return None
    end_scenario = min(start_scenario + max_scenarios, len(all_scenario_files))
    selected_scenario_files = all_scenario_files[start_scenario:end_scenario]

    if not selected_scenario_files:
         print("Error: No scenario files selected for evaluation.")
         return None
    print(f"Selected {len(selected_scenario_files)} scenarios for evaluation (indices {start_scenario} to {end_scenario-1})")

    
    weather_folder = get_weather_folder(folder_path)
    
    # Create dataset
    try:
        dataset = WildfireDataset(
            burn_map=burn_map,
            scenario_files=selected_scenario_files,
            weather_folder=weather_folder,
            timestep_sample_rate=timestep_sample_rate,
            num_weather_timesteps=num_weather_timesteps
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None

    # Check if we have enough data
    if len(dataset) == 0:
        print("No evaluation data generated from selected scenarios.")
        return None
    
    print(f"Evaluation dataset: {len(dataset)} examples")
    
    # Create data loader
    num_workers = min(8, os.cpu_count() or 4)
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        num_workers = min(6, num_workers)  # Limit for Apple Silicon

    eval_loader = DataLoader(
        dataset,
        batch_size=32, # Can use larger batch size for evaluation
        shuffle=False,
        drop_last=False,  # Keep all data for evaluation
        num_workers=num_workers,
        pin_memory=using_gpu
    )
    
    # Create model instance
    model = SimpleResidualCNN(
        input_shape=grid_shape,
        weather_features_concatenated=dataset.concatenated_weather_feature_count
    )

    # --- Multi-GPU Handling --- 
    if using_gpu and torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using DataParallel for multi-GPU evaluation.")
        model = nn.DataParallel(model)
    # --- End Multi-GPU Handling ---
    
    try:
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        return None
    
    # Set model to evaluation mode
    model.eval()
    
    # Use non-blocking transfers if using GPU
    non_blocking = using_gpu
    
    # Setup for visualization
    vis_save_dir = 'models/visualizations/test_predictions'
    if visualize and num_visualizations > 0:
        os.makedirs(vis_save_dir, exist_ok=True)
        if len(dataset) > 0:
            # Select random indices ensuring they are within dataset bounds
            valid_indices = np.arange(len(dataset))
            num_to_visualize = min(num_visualizations, len(dataset))
            visualize_indices = np.random.choice(valid_indices, size=num_to_visualize, replace=False)
            print(f"Will visualize {len(visualize_indices)} examples.")
        else:
            print("Cannot generate visualizations as dataset is empty.")
    
    # For aggregate AP calculation using reservoir sampling
    max_samples_for_metrics = 20000  # Increased sample size for evaluation
    sample_predictions = np.zeros(max_samples_for_metrics, dtype=np.float32)  # Model predictions
    sample_labels = np.zeros(max_samples_for_metrics, dtype=np.float32)       # Ground truth 
    sample_baselines = np.zeros(max_samples_for_metrics, dtype=np.float32)    # Baseline values
    samples_seen = 0
    
    # Initialize dictionary for timestep metrics
    timestep_data = {}  # {ts: {'preds': [], 'gt': [], 'baseline': []}}
    
    # Evaluate model
    print("Evaluating model...")
    with torch.no_grad():
        # Process all examples
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            # Skip batches with only 1 sample if BatchNorm is used
            if batch['baseline_input'].size(0) <= 1 and any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):
                continue
            
            # Transfer data to device
            baseline_input = batch['baseline_input'].to(device, non_blocking=non_blocking)
            weather_input = batch['weather_input'].to(device, non_blocking=non_blocking)
            target = batch['target'].to(device, non_blocking=non_blocking)
            
            # Forward pass
            prediction, residual = model(baseline_input, weather_input)
            
            # Move data to CPU for metrics calculation and visualization
            prediction_np = prediction.cpu().numpy()
            target_np = target.cpu().numpy()
            baseline_np = baseline_input.cpu().numpy()
            residual_np = residual.cpu().numpy()
            
            # Use reservoir sampling for overall metrics
            for i in range(prediction_np.shape[0]):
                # Flatten data for sampling
                pred_flat = prediction_np[i].flatten()
                target_flat = target_np[i].flatten()
                baseline_flat = baseline_np[i].flatten()
                
                # Sample pixels for overall AP metric calculation
                for _ in range(min(100, len(pred_flat))):  # Sample up to 100 pixels/example
                    idx = np.random.randint(0, len(pred_flat))
                    
                    if samples_seen < max_samples_for_metrics:
                        # Fill reservoir
                        sample_predictions[samples_seen] = pred_flat[idx]
                        sample_baselines[samples_seen] = baseline_flat[idx]
                        sample_labels[samples_seen] = target_flat[idx]
                        samples_seen += 1
                    else:
                        pos = np.random.randint(0, samples_seen + 1)
                        if pos < max_samples_for_metrics:
                            sample_predictions[pos] = pred_flat[idx]
                            sample_baselines[pos] = baseline_flat[idx]
                            sample_labels[pos] = target_flat[idx]
                        samples_seen += 1  # Increment total seen count regardless
                
                # Collect data for timestep metrics
                global_idx_approx = batch_idx * eval_loader.batch_size + i  # Approximation needed as we might skip batches
                timestep = None
                try:
                    # Use approximate index to get timestep, might fail if batches skipped
                    if global_idx_approx < len(dataset):
                        timestep = dataset.examples[global_idx_approx]['timestep']
                except (IndexError, KeyError):
                    pass  # Ignore if index is out of bounds or timestep not available
                
                if timestep is not None:
                    if timestep not in timestep_data:
                        timestep_data[timestep] = {'preds': [], 'gt': [], 'baseline': []}
                    # Store flattened data
                    timestep_data[timestep]['preds'].append(prediction_np[i].flatten())
                    timestep_data[timestep]['gt'].append(target_np[i].flatten())
                    timestep_data[timestep]['baseline'].append(baseline_np[i].flatten())
                
                # Check if we should visualize this example
                # Calculate global index (approximate)
                current_batch_size = prediction_np.shape[0]
                start_idx_in_batch = batch_idx * eval_loader.batch_size 
                global_idx = start_idx_in_batch + i  # This is an approximation if batches were skipped
                
                if visualize and global_idx in visualize_indices:
                    vis_save_path = os.path.join(vis_save_dir, f'prediction_idx_{global_idx}.png')
                    # Get timestep if available
                    timestep = None
                    try:
                        timestep = dataset.examples[global_idx]['timestep']  # Use approximate index
                    except IndexError:
                        pass  # Ignore if index is out of bounds
                    
                    # Calculate residual for visualization
                    vis_residual = prediction_np[i, 0] - target_np[i, 0]
                    
                    fig = visualize_prediction(
                        prediction=prediction_np[i, 0],     # Remove batch and channel dimensions
                        ground_truth=target_np[i, 0],       # Remove batch and channel dimensions 
                        baseline=baseline_np[i, 0],         # Remove batch and channel dimensions
                        residual=vis_residual,              # Use calculated residual
                        timestep=timestep,
                        save_path=vis_save_path
                    )
                    plt.close(fig)  # Close figure to free memory
    
    # Calculate overall metrics using sampled data
    print("\nCalculating overall metrics from sampled pixels...")
    overall_model_ap = 0.5  # Default
    overall_baseline_ap = 0.5  # Default
    overall_improvement = 0.0  # Default
    
    if samples_seen > 0:
        current_sample_count = min(samples_seen, max_samples_for_metrics)
        # Filter out NaNs
        mask = ~np.isnan(sample_predictions[:current_sample_count]) & \
               ~np.isnan(sample_labels[:current_sample_count]) & \
               ~np.isnan(sample_baselines[:current_sample_count])
        
        filtered_preds = sample_predictions[:current_sample_count][mask]
        filtered_labels = sample_labels[:current_sample_count][mask]
        filtered_baselines = sample_baselines[:current_sample_count][mask]
        
        # Convert to binary labels for AP score
        binary_labels = (filtered_labels > 0.5).astype(np.int32)
        
        print(f"Calculating metrics using {len(filtered_preds)} valid sampled pixels.")
        
        # Calculate overall metrics
        if len(filtered_preds) > 10 and np.sum(binary_labels > 0) > 0 and np.sum(binary_labels == 0) > 0:
            overall_metrics_dict = calculate_metrics(
                predictions=filtered_preds,
                ground_truth=filtered_labels,
                baseline=filtered_baselines
            )
            overall_model_ap = overall_metrics_dict['model']['ap']
            overall_baseline_ap = overall_metrics_dict['baseline']['ap']
            overall_improvement = overall_metrics_dict['model']['ap_improvement']
        else:
            print("Warning: Not enough diverse samples for reliable overall AP calculation. Using defaults.")
    else:
        print("Warning: No samples collected for overall metrics calculation.")
    
    # Print results
    print("\nEvaluation Results (Based on Sampled Pixels):")
    print(f"Overall Model AP (PR AUC):      {overall_model_ap:.4f}")
    print(f"Overall Baseline AP (PR AUC):   {overall_baseline_ap:.4f}")
    print(f"AP Improvement over Baseline:   {overall_improvement:.4f}")
    
    # Calculate and report timestep metrics
    print("\nCalculating timestep-level metrics...")
    timestep_results = {}  # Store calculated APs per timestep
    timestep_results_for_json = {}  # For saving to JSON
    
    sorted_timesteps = sorted(timestep_data.keys())
    
    for ts in sorted_timesteps:
        if ts not in timestep_data or not timestep_data[ts]['preds']:
            continue  # Skip if no data for this timestep
        
        try:
            ts_preds = np.concatenate(timestep_data[ts]['preds'])
            ts_gt = np.concatenate(timestep_data[ts]['gt'])
            ts_baselines = np.concatenate(timestep_data[ts]['baseline'])
            
            # Convert ground truth to binary labels for AP calculation
            ts_gt_binary = (ts_gt > 0.5).astype(np.int32)
            
            # Filter NaNs
            mask = ~np.isnan(ts_preds) & ~np.isnan(ts_gt_binary) & ~np.isnan(ts_baselines)
            filtered_preds = ts_preds[mask]
            filtered_gt_binary = ts_gt_binary[mask]
            filtered_baselines = ts_baselines[mask]
            
            # Calculate metrics for this timestep
            model_ap_ts = 0.5
            baseline_ap_ts = 0.5
            count_ts = len(filtered_preds)
            
            if count_ts > 10 and np.sum(filtered_gt_binary > 0) > 0 and np.sum(filtered_gt_binary == 0) > 0:
                # Pass binary labels to calculate_metrics
                ts_metrics_dict = calculate_metrics(
                    predictions=filtered_preds,
                    ground_truth=filtered_gt_binary,
                    baseline=filtered_baselines
                )
                model_ap_ts = ts_metrics_dict['model']['ap']
                baseline_ap_ts = ts_metrics_dict['baseline']['ap']
            else:
                # Use fallback if not enough data or only one class
                if count_ts > 0:
                    baseline_ap_ts = np.mean(filtered_gt_binary)  # Simple baseline if needed
                    model_ap_ts = np.mean(filtered_gt_binary)     # Simple fallback if needed
            
            improvement_ts = model_ap_ts - baseline_ap_ts
            timestep_results[ts] = {
                'model_ap': model_ap_ts,
                'baseline_ap': baseline_ap_ts,
                'improvement': improvement_ts,
                'count': count_ts
            }
            # Prepare for JSON (convert to float/int)
            timestep_results_for_json[str(ts)] = {
                'model_ap': float(model_ap_ts) if not np.isnan(model_ap_ts) else None,
                'baseline_ap': float(baseline_ap_ts) if not np.isnan(baseline_ap_ts) else None,
                'improvement': float(improvement_ts) if not np.isnan(improvement_ts) else None,
                'count': int(count_ts)
            }
        except Exception as e:
            print(f"Warning: Error calculating metrics for timestep {ts}: {e}")
            timestep_results_for_json[str(ts)] = {'error': str(e)}
    
    # Print timestep summary table
    print("\nTime Step Analysis - Mean AUPRC (AP):")
    print("-------------------------------------------------------")
    print(" Time Step | Model AP   | Baseline AP | Improvement ")
    print("-------------------------------------------------------")
    for ts in sorted_timesteps:
        if ts in timestep_results:
            res = timestep_results[ts]
            print(f" {ts:<9d} | {res['model_ap']:.6f} | {res['baseline_ap']:.6f}  | {res['improvement']:.6f}  ")
        elif str(ts) in timestep_results_for_json and 'error' in timestep_results_for_json[str(ts)]:
            print(f" {ts:<9d} | {'ERROR':<10} | {'ERROR':<11} | {'ERROR':<11} ")
    print("-------------------------------------------------------")
    
    # Calculate time taken
    total_time = time.time() - start_time
    
    # Prepare results dictionary
    results = {
        'model_path': model_path,
        'dataset_folder': folder_path,
        'evaluation_scenarios': f"{start_scenario} to {end_scenario-1}",
        'num_examples_processed': len(dataset),  # Or count processed examples if different
        'metrics_basis': 'Sampled Pixels',
        'num_sampled_pixels': samples_seen,
        'max_samples_for_metrics': max_samples_for_metrics,
        'overall_metrics': {
            'model_ap': float(overall_model_ap),
            'baseline_ap': float(overall_baseline_ap),
            'improvement_ap': float(overall_improvement)
        },
        'time_taken_seconds': total_time,
        'timestep_metrics': timestep_results_for_json
    }
    
    # Save results to JSON
    results_path = model_path.replace('.pt', '_evaluation.json')
    if results_path == model_path:  # Handle edge case where model_path has no extension
        results_path = model_path + '_evaluation.json'
    
    try:
        with open(results_path, 'w') as f:
            # Use default=str for any types not directly serializable (like numpy types sometimes)
            json.dump(results, f, indent=2, default=str)
        print(f"Evaluation results saved to {results_path}")
    except Exception as e:
        print(f"Error saving evaluation results to {results_path}: {e}")
    
    return results

def test_processed_weather_data(folder_path):
    """
    Test the processed weather data by creating a model and verifying that it can process
    the data without errors.
    
    Args:
        folder_path: Path to the dataset folder
    """
    print("\n=== Testing processed weather data ===")
    print(f"Testing with dataset: {folder_path}")
    
    # Configure device
    device, using_gpu, gpu_name = configure_device()
    print(f"{'✅' if using_gpu else '❌'} GPU acceleration is {'ENABLED' if using_gpu else 'DISABLED'} using: {gpu_name}")
    
    # Load burn map
    try:
        burn_map_path = os.path.join(folder_path, 'burn_map.npy')
        burn_map = np.load(burn_map_path)
        print(f"Loaded burn map from {burn_map_path}, shape: {burn_map.shape}")
    except FileNotFoundError:
        print(f"Could not find burn map at {burn_map_path}")
        return None
    
    # Determine grid shape from burn map
    grid_shape = (burn_map.shape[1], burn_map.shape[2])
    
    # Get all scenario files
    scenario_folder = os.path.join(folder_path, 'scenarii')
    scenario_files = glob.glob(os.path.join(scenario_folder, '*.npy'))
    
    # Sort scenario files to ensure consistent order
    scenario_files.sort()
    
    # Select a few scenarios for testing
    test_scenario_count = 2
    selected_scenario_files = scenario_files[0:test_scenario_count]
    print(f"Selected {len(selected_scenario_files)} scenarios for testing")
    
    # Get the appropriate weather folder (processed if available)
    weather_folder = get_weather_folder(folder_path)
    
    # Create dataset with a subset of data
    dataset = WildfireDataset(
        burn_map=burn_map,
        scenario_files=selected_scenario_files,
        weather_folder=weather_folder,
        timestep_sample_rate=5  # Sample every 5th timestep to keep it small
    )
    
    # Check if we have enough data
    if len(dataset) == 0:
        print("No test data generated")
        return None
    
    print(f"Test dataset: {len(dataset)} examples")
    
    # Create a small data loader
    test_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )
    
    # Get weather feature count from dataset
    weather_feature_count = dataset.examples[0]['weather_features'].shape[0]
    print(f"Detected {weather_feature_count} weather features")
    
    # Create a new model with the appropriate feature count
    print(f"Creating a test model with {weather_feature_count} weather features")
    model = SimpleResidualCNN(
        input_shape=grid_shape,
        weather_features=weather_feature_count
    )
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process one batch to verify everything works
    print("Processing a test batch...")
    with torch.no_grad():
        for batch in test_loader:
            # Transfer data to device
            baseline_input = batch['baseline_input'].to(device, non_blocking=True)
            weather_input = batch['weather_input'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            
            # Forward pass - this is the test to see if everything is compatible
            output, residual = model(baseline_input, weather_input)
            
            # Print the input and output shapes
            print(f"\nInput shapes:")
            print(f"  Baseline input: {baseline_input.shape}")
            print(f"  Weather input: {weather_input.shape}")
            print(f"\nOutput shapes:")
            print(f"  Prediction output: {output.shape}")
            print(f"  Residual output: {residual.shape}")
            
            # Test successful if we get here without errors
            print("\n✅ Test successful! The model can process the processed weather data.")
            
            # Only process one batch
            break
    
    return True

class BurnMapPredictor:
    def __init__(self, model_path, burn_map_path, num_weather_timesteps=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- load baseline burn map ------------------------------------------------
        self.burn_map = np.load(burn_map_path)

        # --- inspect checkpoint to discover expected weather-feature length --------
        ckpt = torch.load(model_path, map_location="cpu")          # 1) load raw state-dict
        concat_len = ckpt["weather_dense.0.weight"].shape[1]       # 2) width of first Linear

        # --- build model with the discovered size ---------------------------------
        self.model = SimpleResidualCNN(
            input_shape=self.burn_map.shape[1:],                   # (H, W)
            weather_features_concatenated=concat_len
        )
        self.model.load_state_dict(ckpt, strict=True)              # weights now fit
        self.model.to(self.device).eval()

        # keep for sanity-checks later
        self.num_weather_timesteps = num_weather_timesteps
        self.concat_len             = concat_len

    def predict(self, timestep: int, weather_history: np.ndarray):
        """
        Args
        ----
        timestep         : index t into burn_map (0 ≤ t < T)
        weather_history  : 1-D vector length == concat_len
                           newest timestep first, oldest last

        Returns:
            burn_probability_map (np.ndarray): shape (H, W)
        """
        if weather_history.shape[-1] != self.concat_len:
            raise ValueError(
                f"weather_history length {weather_history.size} does not match "
                f"model expectation {self.concat_len}"
            )

        baseline_tensor = (
            torch.from_numpy(self.burn_map[timestep])
                 .float()
                 .unsqueeze(0).unsqueeze(0)              # [1,1,H,W]
                 .to(self.device)
        )
        weather_tensor  = (
            torch.from_numpy(weather_history)
                 .float()
                 .unsqueeze(0)                           # [1, F]
                 .to(self.device)
        )

        with torch.no_grad():
            prob_map, _ = self.model(baseline_tensor, weather_tensor)
        return prob_map.squeeze().cpu().numpy()
    
    # def __init__(self, model_path, burn_map_path, num_weather_timesteps=5):
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     # Load baseline burn map
    #     self.burn_map = np.load(burn_map_path)
    #     self.model = SimpleResidualCNN(
    #         input_shape=self.burn_map.shape[1:], 
    #         weather_features_concatenated=num_weather_timesteps * 19  # Adjust if not 19 features
    #     )
    #     self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    #     self.model.to(self.device)
    #     self.model.eval()

    #     self.num_weather_timesteps = num_weather_timesteps

    # def predict(self, timestep, weather_history):
    #     """
    #     Predict the fire probability map at a specific timestep.

    #     Args:
    #         timestep (int): current time step in the scenario
    #         weather_history (np.ndarray): shape (num_timesteps * num_features, )

    #     Returns:
    #         burn_probability_map (np.ndarray): shape (H, W)
    #     """
    #     baseline = self.burn_map[timestep]
    #     baseline_tensor = torch.tensor(baseline).float().unsqueeze(0).unsqueeze(0).to(self.device)
    #     weather_tensor = torch.tensor(weather_history).float().unsqueeze(0).to(self.device)

    #     with torch.no_grad():
    #         output, _ = self.model(baseline_tensor, weather_tensor)
    #         return output.squeeze().cpu().numpy()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate a Simple Residual CNN model with PyTorch for wildfire prediction')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset folder path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--train_start', type=int, default=0, help='Starting index for training scenarios')
    parser.add_argument('--train_max', type=int, default=10, help='Maximum number of scenarios for training')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum delta for early stopping')
    parser.add_argument('--reduce_lr', action='store_true', default=True, help='Reduce learning rate on plateau')
    parser.add_argument('--no_reduce_lr', action='store_false', dest='reduce_lr', help='Disable learning rate reduction')
    parser.add_argument('--tensorboard', action='store_true', help='Use TensorBoard for logging')
    parser.add_argument('--hubert_delta', type=float, default=1.0, help='Delta parameter for Hubert loss')
    parser.add_argument('--auc_fit', action='store_true', help='Use AUC loss instead of Hubert loss')
    parser.add_argument('--auc_epsilon', type=float, default=0.01, help='Epsilon parameter for AUC loss')
    parser.add_argument('--test_start', type=int, default=100, help='Starting index for test scenarios')
    parser.add_argument('--test_max', type=int, default=5, help='Maximum number of scenarios for testing')
    parser.add_argument('--timestep_sample_rate', type=int, default=1, help='Sample every nth timestep')
    parser.add_argument('--model_path', type=str, default='models/simple_residual_cnn', help='Base path to save/load model')
    parser.add_argument('--evaluate_only', action='store_true', help='Skip training and only evaluate model')
    parser.add_argument('--visualize', action='store_true', default=True, help='Generate visualizations during evaluation')
    parser.add_argument('--no_visualize', action='store_false', dest='visualize', help='Disable visualizations during evaluation')
    parser.add_argument('--num_visualizations', type=int, default=5, help='Number of examples to visualize')
    parser.add_argument('--num_weather_timesteps', type=int, default=5, help='Number of past weather timesteps to include')
    
    args = parser.parse_args()
    
    # Construct full model path with .pt extension for loading if not specified
    model_file_path = args.model_path if args.model_path.endswith('.pt') else f"{args.model_path}.pt"
    
    if args.evaluate_only:
        # Evaluate model
        evaluate_simple_residual_model(
            folder_path=args.dataset,
            model_path=model_file_path,
            start_scenario=args.test_start,
            max_scenarios=args.test_max,
            timestep_sample_rate=args.timestep_sample_rate,
            visualize=args.visualize,
            num_visualizations=args.num_visualizations,
            num_weather_timesteps=args.num_weather_timesteps
        )
    else:
        # Train model
        model, history, best_ap_improvement_model_path, best_ap_improvement = train_simple_residual_model(
            folder_path=args.dataset,
            model_output_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            start_scenario=args.train_start,
            max_scenarios=args.train_max,
            timestep_sample_rate=args.timestep_sample_rate,
            patience=args.patience,
            min_delta=args.min_delta,
            reduce_lr=args.reduce_lr,
            tensorboard=args.tensorboard,
            hubert_delta=args.hubert_delta,
            AUC_fit=args.auc_fit,
            auc_epsilon=args.auc_epsilon,
            num_weather_timesteps=args.num_weather_timesteps
        )
        
        # Optionally, evaluate after training if it was successful
        if model is not None:
            print("\n--- Evaluating model with best AP improvement ---")
            print(f"Best AP improvement achieved: {best_ap_improvement:.4f}")
            print(f"Using model: {best_ap_improvement_model_path}")
            
            # Try to determine the epoch where best AP improvement was achieved
            if 'val_improvement' in history and len(history['val_improvement']) > 0:
                try:
                    best_epoch_index = history['val_improvement'].index(best_ap_improvement)
                    print(f"Best AP improvement was achieved at epoch {best_epoch_index + 1}")
                except ValueError:
                    # In case the exact value isn't found due to floating point issues
                    print("Could not determine the exact epoch for best AP improvement")
            
            evaluate_simple_residual_model(
                folder_path=args.dataset,
                model_path=best_ap_improvement_model_path,  # Use the best AP improvement model
                start_scenario=args.test_start,
                max_scenarios=args.test_max,
                timestep_sample_rate=args.timestep_sample_rate,
                visualize=args.visualize,
                num_visualizations=args.num_visualizations,
                num_weather_timesteps=args.num_weather_timesteps
            )
    
    print("Done!")