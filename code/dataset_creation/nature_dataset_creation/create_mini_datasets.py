#!/usr/bin/env python3
"""
Create mini versions of all California 2020 datasets for benchmarking.

This script creates mini datasets by randomly selecting 10 fires from each
full dataset. All mini datasets are placed in a common folder structure.
"""

import os
import sys
import json
import shutil
import random
import numpy as np
from pathlib import Path

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "MiniCalifornia2020Datasets")

# List of all datasets to create mini versions of
DATASETS = [
    {
        'name': 'California2020Dataset',
        'source_dir': os.path.join(PROJECT_ROOT, 'California2020Dataset'),
        'risk_map_pattern': 'wfpi_*.npy',
        'config_file': 'config_california_2020.json',
        'summary_file': 'dataset_summary.json'
    },
    {
        'name': 'California2020Dataset_Day1',
        'source_dir': os.path.join(PROJECT_ROOT, 'California2020Dataset_Day1'),
        'risk_map_pattern': 'wfpi_*.npy',
        'config_file': 'config_california_2020_day1.json',
        'summary_file': 'dataset_summary.json'
    },
    {
        'name': 'California2020Dataset_LargeFires',
        'source_dir': os.path.join(PROJECT_ROOT, 'California2020Dataset_LargeFires'),
        'risk_map_pattern': 'wfpi_*.npy',
        'config_file': 'config_california_2020_large_fires.json',
        'summary_file': 'dataset_summary.json'
    },
    {
        'name': 'California2020Dataset_Day1_LargeFires',
        'source_dir': os.path.join(PROJECT_ROOT, 'California2020Dataset_Day1_LargeFires'),
        'risk_map_pattern': 'wfpi_*.npy',
        'config_file': 'config_california_2020_day1_large_fires.json',
        'summary_file': 'dataset_summary.json'
    },
    {
        'name': 'California2020Dataset_IgnitionProb',
        'source_dir': os.path.join(PROJECT_ROOT, 'California2020Dataset_IgnitionProb'),
        'risk_map_pattern': 'static_risk_ignition_prob.npy',
        'config_file': 'config_california_2020_ignition_prob.json',
        'summary_file': 'dataset_summary.json'
    },
    {
        'name': 'California2020Dataset_IgnitionProb_LargeFires',
        'source_dir': os.path.join(PROJECT_ROOT, 'California2020Dataset_IgnitionProb_LargeFires'),
        'risk_map_pattern': 'static_risk_ignition_prob.npy',
        'config_file': 'config_california_2020_ignition_prob_large_fires.json',
        'summary_file': 'dataset_summary.json'
    },
    {
        'name': 'California2020Dataset_BurnProb',
        'source_dir': os.path.join(PROJECT_ROOT, 'California2020Dataset_BurnProb'),
        'risk_map_pattern': 'static_risk_burn_prob.npy',
        'config_file': 'config_california_2020_burn_prob.json',
        'summary_file': 'dataset_summary.json'
    },
    {
        'name': 'California2020Dataset_BurnProb_LargeFires',
        'source_dir': os.path.join(PROJECT_ROOT, 'California2020Dataset_BurnProb_LargeFires'),
        'risk_map_pattern': 'static_risk_burn_prob.npy',
        'config_file': 'config_california_2020_burn_prob_large_fires.json',
        'summary_file': 'dataset_summary.json'
    },
]

NUM_FIRES = 10  # Number of fires to select per dataset


def create_mini_dataset(dataset_info):
    """Create a mini version of a dataset by selecting random fires."""
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_info['name']}")
    print(f"{'='*80}")
    
    source_dir = dataset_info['source_dir']
    dataset_name = dataset_info['name']
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"  WARNING: Source directory does not exist: {source_dir}")
        print(f"  Skipping...")
        return False
    
    # Create output directory for this mini dataset
    mini_dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(mini_dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(mini_dataset_dir, 'scenarii'), exist_ok=True)
    
    # Get all scenario files
    scenarii_dir = os.path.join(source_dir, 'scenarii')
    if not os.path.exists(scenarii_dir):
        print(f"  ERROR: Scenarii directory does not exist: {scenarii_dir}")
        return False
    
    all_scenarios = list(Path(scenarii_dir).glob("*.npy"))
    print(f"  Total scenarios in source: {len(all_scenarios)}")
    
    if len(all_scenarios) < NUM_FIRES:
        print(f"  WARNING: Only {len(all_scenarios)} scenarios available, using all of them")
        selected_scenarios = all_scenarios
    else:
        # Randomly select NUM_FIRES scenarios
        selected_scenarios = random.sample(all_scenarios, NUM_FIRES)
    
    print(f"  Selected {len(selected_scenarios)} scenarios")
    
    # Copy mask.npy
    mask_source = os.path.join(source_dir, 'mask.npy')
    if os.path.exists(mask_source):
        mask_dest = os.path.join(mini_dataset_dir, 'mask.npy')
        shutil.copy2(mask_source, mask_dest)
        print(f"  Copied mask.npy")
    else:
        print(f"  WARNING: mask.npy not found in source")
    
    # Copy risk map(s)
    risk_map_pattern = dataset_info['risk_map_pattern']
    if risk_map_pattern == 'wfpi_*.npy':
        # Copy all WFPI maps (they might be needed for the selected fires)
        wfpi_files = list(Path(source_dir).glob('wfpi_*.npy'))
        for wfpi_file in wfpi_files:
            shutil.copy2(wfpi_file, os.path.join(mini_dataset_dir, wfpi_file.name))
        print(f"  Copied {len(wfpi_files)} WFPI map(s)")
    else:
        # Copy static risk map
        risk_map_source = os.path.join(source_dir, risk_map_pattern)
        if os.path.exists(risk_map_source):
            risk_map_dest = os.path.join(mini_dataset_dir, risk_map_pattern)
            shutil.copy2(risk_map_source, risk_map_dest)
            print(f"  Copied {risk_map_pattern}")
        else:
            print(f"  WARNING: {risk_map_pattern} not found in source")
    
    # Copy selected scenarios
    selected_fire_keys = []
    for scenario_file in selected_scenarios:
        scenario_dest = os.path.join(mini_dataset_dir, 'scenarii', scenario_file.name)
        shutil.copy2(scenario_file, scenario_dest)
        
        # Extract fire key for config (remove _scenario1 suffix if present)
        fire_key = scenario_file.stem.replace('_scenario1', '')
        selected_fire_keys.append(fire_key)
    
    print(f"  Copied {len(selected_scenarios)} scenario files")
    
    # Load and update config file
    config_source = os.path.join(source_dir, dataset_info['config_file'])
    if os.path.exists(config_source):
        with open(config_source, 'r') as f:
            full_config = json.load(f)
        
        # Create mini config with only selected fires
        mini_config = {}
        for fire_key in selected_fire_keys:
            config_key = f"offset_{fire_key}"
            if config_key in full_config:
                mini_config[config_key] = full_config[config_key]
            else:
                # If not found, use a random offset (1-12)
                mini_config[config_key] = random.randint(1, 12)
        
        config_dest = os.path.join(mini_dataset_dir, dataset_info['config_file'])
        with open(config_dest, 'w') as f:
            json.dump(mini_config, f, indent=2)
        print(f"  Created mini config with {len(mini_config)} entries")
    else:
        print(f"  WARNING: Config file not found: {config_source}")
    
    # Load and update summary file
    summary_source = os.path.join(source_dir, dataset_info['summary_file'])
    if os.path.exists(summary_source):
        with open(summary_source, 'r') as f:
            full_summary = json.load(f)
        
        # Update summary for mini dataset
        mini_summary = full_summary.copy()
        mini_summary['dataset_name'] = f"{full_summary.get('dataset_name', dataset_name)}_Mini"
        mini_summary['total_fires'] = len(selected_scenarios)
        mini_summary['successful_fires'] = len(selected_scenarios)
        mini_summary['failed_fires'] = 0
        mini_summary['is_mini_dataset'] = True
        mini_summary['source_dataset'] = dataset_name
        mini_summary['selected_fires'] = [key for key in selected_fire_keys]
        
        summary_dest = os.path.join(mini_dataset_dir, dataset_info['summary_file'])
        with open(summary_dest, 'w') as f:
            json.dump(mini_summary, f, indent=2)
        print(f"  Created mini summary")
    else:
        print(f"  WARNING: Summary file not found: {summary_source}")
    
    print(f"  ✓ Mini dataset created: {mini_dataset_dir}")
    return True


def main():
    """Create mini versions of all datasets."""
    print("="*80)
    print("Creating Mini California 2020 Datasets")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Fires per dataset: {NUM_FIRES}")
    print(f"Total datasets: {len(DATASETS)}")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each dataset
    successful = 0
    failed = 0
    
    for dataset_info in DATASETS:
        try:
            if create_mini_dataset(dataset_info):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: Failed to create mini dataset: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(DATASETS)}")
    print(f"\nMini datasets created in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
