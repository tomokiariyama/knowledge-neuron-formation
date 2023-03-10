#!/bin/bash

./scripts/setup.sh

STEPS=(0 20 40 60 80 100 200 300 400 500 1000 1500 2000)
for STEP in "${STEPS[@]}"
do
  ./scripts/evaluate.sh google/multiberts-seed_0-step_"$STEP"k generics_kb_best
done

./scripts/make_graphs.sh generics_kb_best
