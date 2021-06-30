#!/bin/bash
#SBATCH --time=0:30:0
#SBATCH --job-name=pred_via_constraints
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=test.out


TF_CONTAINER=~/tensorflow-2.3.0-gpu.simg
EXP_PATH=/home/mohamadol/fusion/final_cnns
DATASET=/home/mohamadol/tensorflow_datasets

P1=/usr/lib/python36.zip
P2=/usr/lib/python3.6
P3=/usr/lib/python3.6/lib-dynload
P4=/home/mohamadol/.local/lib/python3.6/site-packages
P5=/usr/local/lib/python3.6/dist-packages

TRAINING=False
PRE_TRAINED=True
GPUS=2

singularity exec --nv $TF_CONTAINER python3 $EXP_PATH/cifar10/mnasnet/modified/vanilla/mnasnet.py $DATASET $TRAINING $PRE_TRAINED $GPUS $P1 $P2 $P3 $P4 $P5
