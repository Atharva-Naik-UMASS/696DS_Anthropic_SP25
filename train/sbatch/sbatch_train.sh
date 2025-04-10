#!/bin/bash
#SBATCH --partition=gpu-preempt

#SBATCH --gpus=a100:1
#SBATCH --time=10:00:00
#SBATCH --mem=40g
#SBATCH --job-name=flipkarflipkart_nums_v1_3B     #IMPORTANT Set the job name to "JobName"
#SBATCH --ntasks-per-node=2      #Request 4 tasks/cores per node

if [ $# -eq 0 ]; then
    echo "Usage: sbatch $0 <config_file_path>"
    exit 1
fi

# Print GPU info
echo "### GPU Information ###"
nvidia-smi
echo "#######################"

module load conda/latest
conda activate /work/pi_wenlongzhao_umass_edu/6/envs/anthro_finetune

python main.py yaml_configs/$1
# torchrun --standalone --nproc_per_node=2 main.py yaml_configs/$1

## RUN: 
## cd 696DS_Anthropic_SP25/train
## sbatch sbatch/sbatch_train.sh <yaml_config_file_name>.yaml 

## try accelerate launch --multi_gpu --num_processes=4 main2.py yaml_configs/$1