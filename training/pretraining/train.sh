#!/bin/bash
#SBATCH -N 2
#SBATCH -c 8
#SBATCH -t 1-00:15:00
#SBATCH --mem=128G
#SBATCH -p general
#SBATCH -q private
#SBATCH --gres=gpu:a30:3
#-C a100_80
#SBATCH -o logs/slurm.%j.out
#SBATCH -e logs/slurm.%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=npattab1@asu.edu

module load mamba/latest
source activate torch


MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=29400


echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

srun torchrun --nnodes=2 \
  --nproc-per-node=3 \
  --node_rank=$SLURM_PROCID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --rdzv_backend=c10d \
  train.py

