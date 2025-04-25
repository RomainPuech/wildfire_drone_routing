#!/bin/bash

#SBATCH --job-name="burnmap_predictor_test"
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -n 4                      # 4 CPU cores (adjust as needed)
#SBATCH --time=00:30:00           # 30 minutes runtime
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=justdoye@mit.edu
#SBATCH --gres=gpu:1              # Request 1 GPU

# Loading the required module
source /etc/profile
module load anaconda

# Run the script
# python residual_cnn_normfirst.py --dataset ML/MLDataset/0002 --epochs 20 --batch_size 128 --train_max 500 --model_path models/ok_residualbatch128_normfirst_auc2_best_ap_improvement.pt --test_max 500 --test_start 500 --num_weather_timesteps 5 --evaluate_only
python ML/burn_map_model.py --dataset ML/MLDataset/0002 --epochs 20 --batch_size 128 --train_max 500 --model_path models/ok_residualbatch128_normfirst_auc2_best_ap_improvement.pt --test_max 500 --test_start 500 --num_weather_timesteps 5 --evaluate_only
# python ML/displays.py --model models/ok_residualbatch128_normfirst_auc2_best_ap_improvement.pt --scenario ML/MLDataset/0002/scenarii/0002_00001.npy --patch --comparison
# python helloworld.py