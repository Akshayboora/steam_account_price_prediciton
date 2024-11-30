#!/usr/bin/env python3

import argparse
import os
import pandas as pd
from src.models.single_cat_model import SingleCategoryModel

def parse_args():
    parser = argparse.ArgumentParser(description='Validate a trained price prediction model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to validation data JSON file')
    parser.add_argument('--category-id', type=int, required=True, help='Category ID for validation')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save validation results')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load and filter data
        df = pd.read_json(args.data_path)
        valid_df = df[df['category_id'] == args.category_id].copy()

        if len(valid_df) == 0:
            raise ValueError(f"No validation data found for category_id {args.category_id}")

        # Initialize, load model and validate
        model = SingleCategoryModel(category_number=args.category_id)
        model.load_model(args.model_path)
        metrics = model.validate(df=valid_df, save_dir=args.output_dir)

        # Save validation results
        results_path = os.path.join(args.output_dir, f'validation_results_cat_{args.category_id}.txt')
        with open(results_path, 'w') as f:
            f.write("Validation Results:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise

if __name__ == '__main__':
    main()
