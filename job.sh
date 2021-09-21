#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uwdus@kit.edu
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
export semi_setting='pascal/1_8/split_0'

CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
--dataset pascal --data-root /lsdf/kit/iai/projects/iai-aida/Daten_Keppler/Pascal \
--batch-size 16 --backbone resnet50 --model deeplabv3plus \
--labeled-id-path dataset/splits/$semi_setting/labeled.txt \
--unlabeled-id-path dataset/splits/$semi_setting/unlabeled.txt \
--pseudo-mask-path outdir/pseudo_masks/$semi_setting \
--save-path outdir/models/$semi_setting --plus --reliable-id-path outdir/reliable_ids/$semi_setting1~
