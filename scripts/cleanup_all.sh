#!/bin/bash

# Script to clean up all generated files

echo "Starting comprehensive cleanup..."

# Clean Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".coverage" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name "*.egg" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name ".eggs" -exec rm -rf {} +

# Clean generated data directories
find data/raw -type f ! -name '.gitkeep' -delete
find data/processed -type f ! -name '.gitkeep' -delete

# Clean model artifacts
find models/trained -type f ! -name '.gitkeep' -delete
find models/onnx -type f ! -name '.gitkeep' -delete

# Clean CatBoost files
rm -rf catboost_info/

# Clean temporary files
rm -rf .pytest_cache/
rm -rf .coverage
rm -rf htmlcov/
rm -rf dist/
rm -rf build/

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/trained
mkdir -p models/onnx

echo "Cleanup complete!"
