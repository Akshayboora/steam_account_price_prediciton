#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from src.models.single_cat_model import SingleCategoryModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train a category-specific price prediction model')
    parser.add_argument('--category-id', type=int, required=True, help='Category ID to train model for')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data JSON file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save trained model')
    parser.add_argument('--test-size', type=float, default=0.078, help='Test split size (default: 0.078)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed (default: 42)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load and prepare data
        print(f"Loading data from {args.data_path}")
        df = pd.read_json(args.data_path)

        # Filter data for specific category
        category_df = df[df['category_id'] == args.category_id].copy()
        if len(category_df) == 0:
            raise ValueError(f"No data found for category_id {args.category_id}")

        print(f"Training model for category {args.category_id}")
        print(f"Total samples for category: {len(category_df)}")

        # Initialize and train model
        model = SingleCategoryModel(category_number=args.category_id)
        model.train(category_df)

        # Save model
        output_path = os.path.join(args.output_dir, f'category_{args.category_id}_model.onnx')
        model.export(model_path=output_path)
        print(f"Model saved to {output_path}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
