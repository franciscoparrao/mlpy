#!/bin/bash
# MLPY CLI Demo Script
# This script demonstrates various MLPY CLI commands

echo "==================================="
echo "MLPY CLI Demo"
echo "==================================="

# Show MLPY info
echo -e "\n1. MLPY Installation Info:"
python -m mlpy info

# Inspect dataset
echo -e "\n2. Dataset Information:"
python -m mlpy task info iris_sample.csv -y species

# Train a single model
echo -e "\n3. Training Random Forest:"
python -m mlpy train iris_sample.csv -t classif -y species -l rf -k 3 -m acc -m f1

# Benchmark multiple models
echo -e "\n4. Benchmarking Multiple Models:"
python -m mlpy benchmark iris_sample.csv -t classif -y species -l rf -l lr -l dt -k 3

# Create experiment template
echo -e "\n5. Creating Experiment Template:"
python -m mlpy experiment iris_experiment.yaml

echo -e "\nDemo complete! Check iris_experiment.yaml for the experiment template."