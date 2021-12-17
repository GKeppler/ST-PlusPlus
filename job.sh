#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uwdus@student.kit.edu
#SBATCH --error=%j_error.txt
#SBATCH --output=%j_output.txt
#SBATCH --job-name=STplus
#SBATCH --constraint=LSDF

# remove all modules
module purge

# activate python env
module load devel/miniconda/4.9.2

# conda init + activate
source ~/.bashrc
eval "$(conda shell.bash hook)"

conda activate ssl-lin

# activate cuda
module load devel/cuda/10.2
module load devel/cudnn/10.2

# move to target dir
cd /home/kit/stud/uwdus/Masterthesis/git/ST-PlusPlus/


python main.py --epochs 20 --plus
