#!/bin/bash

# Load aliases
# alias omni_python="/home/paolo/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh"

# Number of runs
num_runs=100

for i in $(seq 1 $num_runs)
do
    echo "Running script - Iteration $i"
    /home/paolo/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh build_dataset.py
done

