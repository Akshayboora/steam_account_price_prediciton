#!/usr/bin/env python3
"""Model evaluation script."""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

from src.models.single_cat_model import SingleCategoryModel
from sklearn.metrics import r2_score

def evaluate_model(model_path, data_path, output_dir, save_predictions=None):
    """Run model evaluation."""
    model = SingleCategoryModel.load(model_path)
    df = pd.read_json(data_path, orient='records')
    predictions = model.predict(df)

    # Try different possible target column names
    target_columns = ['target', 'sold_price', 'price']
    actuals = None
    for col in target_columns:
        if col in df.columns:
            actuals = df[col]
            break

    if actuals is None:
        raise ValueError(f"No target column found. Expected one of: {target_columns}")

    metrics = {
        'pearson': pearsonr(actuals, predictions)[0],
        'r2': r2_score(actuals, predictions)
    }

    if save_predictions:
        pred_df = pd.DataFrame({
            'prediction': predictions,
            'actual': actuals
        })
        pred_df.to_csv(save_predictions, index=False)

    # Create output directory and save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics['num_samples'] = len(actuals)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate scatter plot with Pearson correlation in title
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actuals, y=predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Prediction vs Actual Values\nPearson Correlation: {metrics["pearson"]:.3f}')
    plt.savefig(output_dir / 'scatter_plot.png')
    plt.close()

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--data-path", required=True, help="Path to evaluation data")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--save-predictions", help="Path to save predictions CSV")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_path, args.output_dir, args.save_predictions)
