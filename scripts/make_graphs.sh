#!/bin/bash
USAGE="bash $0 [dataset_type]"

DATASET=$1

if [ -n "$2" ] ; then
  echo "[ERROR] The arguemnt of $0 takes 1 positional arguments but 2 (or more) was given"
  echo "$USAGE"
  exit 1
fi

DATE=$(date +%Y%m%d-%H%M)
echo "$DATE"

### ACTUAL JOBS ###
python violinplots.py \
    --result_path "work/result/${DATASET}"

DATE=$(date +%Y%m%d-%H%M)
echo "$DATE"
