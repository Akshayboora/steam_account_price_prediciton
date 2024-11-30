#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.models.single_cat_model import SingleCategoryModel
import onnxruntime as ort

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using a trained ONNX model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to input data JSON file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save predictions CSV')
    parser.add_argument('--category-id', type=int, required=True, help='Category ID for preprocessing')
    return parser.parse_args()

def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        print(f"Failed to load ONNX model from {model_path}: {e}")
        raise

def predict_with_onnx(session, X):
    if isinstance(X, pd.DataFrame):
        X = X.drop(columns=['category_id', 'sold_price'], errors='ignore')
        X = X.to_numpy()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    predictions = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    return predictions

def main():
    args = parse_args()

    try:
        # Load data
        df = pd.read_json(args.data_path)

        # Filter data for specific category
        category_df = df[df['category_id'] == args.category_id].copy()
        if len(category_df) == 0:
            raise ValueError(f"No data found for category_id {args.category_id}")

        # Initialize model and preprocess data
        model = SingleCategoryModel(category_number=args.category_id)
        processed_data = model.preprocess_data(category_df)

        # Load ONNX model and make predictions
        onnx_session = load_onnx_model(args.model_path)
        predictions = predict_with_onnx(onnx_session, processed_data)

        # Save predictions
        results_df = pd.DataFrame({
            'id': category_df.index,
            'predicted_price': predictions.flatten()
        })
        results_df.to_csv(args.output_path, index=False)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == '__main__':
    main()
