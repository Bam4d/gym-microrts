#!/bin/sh
#$ -cwd
#$ -pe smp 8
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=volta
#$ -l h_rt=6:0:0


# Use the requested GPU
export WANDB_API_KEY=$(cat key.txt)

module load anaconda3
module load cuda
module load java/11.0.2
module load use.dev ffmpeg/4.1.6-singularity

conda create --name microrts python=3.7
conda activate microrts

pip install -r requirements.txt
cd ../../ && pip install -e .


python ../paper/ppo_gridnet_diverse_impala_decoder.py  --num-bot-envs 24 --cuda True --wandb-project-name gridnet  --wandb-entity chrisbam4d --prod-mode --total-timesteps 1000000