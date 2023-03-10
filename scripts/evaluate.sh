#!/bin/bash
USAGE="bash $0 [model_name] [dataset_type]"

MODEL=$1
DATASET_TYPE=$2

if [ -n "$3" ] ; then
  echo "[ERROR] The arguemnt of $0 takes 2 positional arguments but 3 (or more) was given"
  echo "$USAGE"
  exit 1
fi

ARR=(${MODEL//// })

DATE=$(date +%Y%m%d-%H%M)
echo "$DATE"

### actual jobs ###
python evaluate.py \
    --model_name "$MODEL" \
    --dataset_type "$DATASET_TYPE" \
    --dataset_path "data" \
    --number_of_templates 4 \
    --adaptive_threshold 0.2 \
    --max_words 10 \
    --logfile_name "${ARR[-1]}_${DATE}"

DATE=$(date +%Y%m%d-%H%M)
echo "$DATE"
