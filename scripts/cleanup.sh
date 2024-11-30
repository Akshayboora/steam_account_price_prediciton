#!/bin/bash

# Cleanup script for ml_price_predictor
# Removes generated data, cache files, and temporary files

echo "Cleaning up generated files and directories..."

# Remove model files
rm -rf models/trained/*
rm -rf models/onnx/*

# Remove plots
rm -rf plots/*

# Remove cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".coverage" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name "*.egg" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name ".eggs" -exec rm -rf {} +

# Remove temporary files
rm -rf tmp/*
rm -f .DS_Store
rm -rf .ipynb_checkpoints

# Create necessary directories
mkdir -p models/{trained,onnx}
mkdir -p plots
mkdir -p tmp

echo "Cleanup complete!"
