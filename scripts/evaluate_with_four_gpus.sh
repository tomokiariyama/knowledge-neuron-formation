#!/bin/sh
USAGE="bash $0 [.sif_file] [dataset_name]"

#$ -l rt_G.large=1
#$ -l h_rt=23:00:00
#$ -j y
#$ -cwd
#$ -m a
#$ -m e

SIF_FILE=$1
DATASET=$2

### Singularity ###
source /etc/profile.d/modules.sh
module load singularitypro openmpi/4.1.3

### CUDA ###
module load cuda/11.3/11.3.1
module load cudnn/8.2/8.2.4
module load nccl/2.9/2.9.9-1

DATE=`date +%Y%m%d-%H%M`
echo $DATE


### pretrained-steps of models ###
STEPS=(1700 1800 1900 2000)
### other parameters ###
DATASET_PATH="/home/acd13293hw/sftp_sync/master_research/concept-neurons_by_time/data/GenericsKB"
NT=4
AT=0.2
MW=10


### actual jobs ###
xargs -P4 -I% bash -c "%" << EOF
singularity exec --nv $SIF_FILE python evaluate.py \
    --model_name "google/multiberts-seed_0-step_${STEPS[0]}k" \
    --dataset_type $DATASET \
    --dataset_path $DATASET_PATH \
    --number_of_templates $NT \
    --adaptive_threshold $AT \
    --max_words $MW \
    --logfile_name "multiberts-seed_0-step_${STEPS[0]}k_${DATE}" \
    --local_rank 0
singularity exec --nv $SIF_FILE python evaluate.py \
    --model_name "google/multiberts-seed_0-step_${STEPS[1]}k" \
    --dataset_type $DATASET \
    --dataset_path $DATASET_PATH \
    --number_of_templates $NT \
    --adaptive_threshold $AT \
    --max_words $MW \
    --logfile_name "multiberts-seed_0-step_${STEPS[1]}k_${DATE}" \
    --local_rank 1
singularity exec --nv $SIF_FILE python evaluate.py \
    --model_name "google/multiberts-seed_0-step_${STEPS[2]}k" \
    --dataset_type $DATASET \
    --dataset_path $DATASET_PATH \
    --number_of_templates $NT \
    --adaptive_threshold $AT \
    --max_words $MW \
    --logfile_name "multiberts-seed_0-step_${STEPS[2]}k_${DATE}" \
    --local_rank 2
singularity exec --nv $SIF_FILE python evaluate.py \
    --model_name "google/multiberts-seed_0-step_${STEPS[3]}k" \
    --dataset_type $DATASET \
    --dataset_path $DATASET_PATH \
    --number_of_templates $NT \
    --adaptive_threshold $AT \
    --max_words $MW \
    --logfile_name "multiberts-seed_0-step_${STEPS[3]}k_${DATE}" \
    --local_rank 3
EOF


DATE=`date +%Y%m%d-%H%M`
echo $DATE
