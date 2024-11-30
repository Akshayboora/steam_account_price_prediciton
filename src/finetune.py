#!/usr/bin/env python3
"""
Fine-tune an existing ONNX model on new data.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from src.models.single_cat_model import SingleCategoryModel

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune an existing ONNX model on new data')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the existing ONNX model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to new training data (JSON format)')
    parser.add_argument('--category-id', type=int, required=True,
                       help='Category ID for the model')
    parser.add_argument('--output-dir', type=str, default='models/trained',
                       help='Directory to save fine-tuned model')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of iterations for fine-tuning')
    parser.add_argument('--learning-rate', type=float, default=0.03,
                       help='Learning rate for fine-tuning')
    return parser.parse_args()

def load_and_preprocess_data(data_path: str, category_id: int) -> pd.DataFrame:
    """Load and preprocess the fine-tuning data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df[df['category_id'] == category_id].copy()

def finetune_model(model: SingleCategoryModel,
                  data: pd.DataFrame,
                  iterations: int,
                  learning_rate: float) -> SingleCategoryModel:
    """Fine-tune the model on new data."""
    finetune_params = {
        'iterations': iterations,
        'learning_rate': learning_rate
    }
    model.train(data, is_finetuning=True, **finetune_params)
    return model

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and finetune model
    model = SingleCategoryModel.load(args.model_path)
    data = load_and_preprocess_data(args.data_path, args.category_id)
    model = finetune_model(
        model=model,
        data=data,
        iterations=args.iterations,
        learning_rate=args.learning_rate
    )

    # Save fine-tuned model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(
        args.output_dir,
        f'category_{args.category_id}_finetuned_{timestamp}.onnx'
    )
    model.export(output_path)

    # Save metadata
    metadata = {
        'original_model': args.model_path,
        'finetuning_data': args.data_path,
        'timestamp': timestamp,
        'iterations': args.iterations,
        'learning_rate': args.learning_rate
    }
    metadata_path = output_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == '__main__':
    main()
