#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a40:2
#SBATCH --account=shrikann_35

eval "$(conda shell.bash hook)"
conda activate /project/shrikann_35/dbose/envs/lavis

cd /project/shrikann_35/dbose/codes/ads_codes/SAIM-ADS/caption_extraction

CUDA_VISIBLE_DEVICES=0 python shot_sample_caption_extraction.py --caption_folder /scratch1/dbose/ads_repo/captions/csv_files --image_folder /scratch1/dbose/ads_repo/captions/images --model_name blip_caption --model_type large_coco --shot_folder /scratch1/dbose/ads_repo/shot_folder/PySceneDetect --shard_file /project/shrikann_35/dbose/codes/ads_codes/caption_split_file/incomplete_files_caption_extraction.txt


