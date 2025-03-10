#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=08:00:00

if [ $# -eq 0 ]; then
    echo "Usage: sbatch $0 <config_file_path>"
    exit 1
fi

# Print GPU info
echo "### GPU Information ###"
nvidia-smi
echo "#######################"

module load conda/latest
conda activate anthro_finetune

cd /work/pi_wenlongzhao_umass_edu/6/atharva/src/train
python main.py $1