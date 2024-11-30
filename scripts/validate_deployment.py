#!/usr/bin/env python3
"""Validate model performance against deployment criteria."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

def generate_pearson_vs_samples_plot(model, X, y, n_points=10, output_dir=None):
    """Generate plot showing Pearson correlation vs number of samples."""
    sample_sizes = np.linspace(1000, len(X), n_points, dtype=int)
    correlations = []

    for size in sample_sizes:
        indices = np.random.choice(len(X), size=size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        y_pred = model.predict(X_sample)
        corr = np.corrcoef(y_pred, y_sample)[0, 1]
        correlations.append(corr)

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, correlations, 'b-', marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
    plt.fill_between(sample_sizes, correlations, 0.95, where=(np.array(correlations) >= 0.95),
                     color='green', alpha=0.3, label='Above Target')
    plt.fill_between(sample_sizes, correlations, 0.95, where=(np.array(correlations) < 0.95),
                     color='red', alpha=0.3, label='Below Target')

    plt.xlabel('Number of Samples')
    plt.ylabel('Pearson Correlation')
    plt.title('Pearson Correlation vs Number of Samples')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if output_dir:
        plt.savefig(Path(output_dir) / 'pearson_vs_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_learning_curves(model, X, y, output_dir=None):
    """Generate learning curves plot."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='r2', n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation Score', color='green')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')

    plt.xlabel('Training Examples')
    plt.ylabel('R² Score')
    plt.title('Learning Curves')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    if output_dir:
        plt.savefig(Path(output_dir) / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def validate_model_performance(metrics_file, criteria_file=None):
    """Validate model metrics against deployment criteria."""
    # Default criteria
    criteria = {
        'pearson_threshold': 0.95,
        'r2_threshold': 0.7
    }

    # Load custom criteria if provided
    if criteria_file:
        with open(criteria_file) as f:
            criteria.update(json.load(f))

    # Ensure directory exists
    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)

    # Load metrics
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Validate metrics
    validation_results = {
        'pearson_passed': metrics['pearson'] > criteria['pearson_threshold'],
        'r2_passed': metrics['r2'] > criteria['r2_threshold'],
        'metrics': metrics,
        'criteria': criteria
    }

    return validation_results

def main():
    parser = argparse.ArgumentParser(description="Validate model for deployment")
    parser.add_argument("--metrics-file", required=True, help="Path to metrics JSON file")
    parser.add_argument("--criteria-file", help="Path to custom criteria JSON file")
    parser.add_argument("--output-file", help="Path to save validation results")
    parser.add_argument("--model-path", help="Path to trained model for generating plots")
    parser.add_argument("--data-path", help="Path to validation data for generating plots")
    parser.add_argument("--plots-dir", help="Directory to save plots")
    args = parser.parse_args()

    results = validate_model_performance(args.metrics_file, args.criteria_file)

    # Generate plots if model and data are provided
    if args.model_path and args.data_path and args.plots_dir:
        from src.models.single_cat_model import SingleCategoryModel
        import pandas as pd

        # Load model and data
        model = SingleCategoryModel.load(args.model_path)
        data = pd.read_csv(args.data_path)
        X = data.drop('target', axis=1)
        y = data['target']

        # Generate plots
        Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
        generate_pearson_vs_samples_plot(model, X.values, y.values, output_dir=args.plots_dir)
        generate_learning_curves(model, X.values, y.values, output_dir=args.plots_dir)
        print(f"\nPlots saved to {args.plots_dir}")

    # Print results
    print("\nModel Validation Results:")
    print("========================")
    print(f"Pearson Correlation: {results['metrics']['pearson']:.3f} (threshold: {results['criteria']['pearson_threshold']}) - {'✓' if results['pearson_passed'] else '✗'}")
    print(f"R² Score: {results['metrics']['r2']:.3f} (threshold: {results['criteria']['r2_threshold']}) - {'✓' if results['r2_passed'] else '✗'}")

    all_passed = all([results['pearson_passed'], results['r2_passed']])
    print(f"\nOverall Status: {'PASSED' if all_passed else 'FAILED'}")

    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    # Exit with status code
    exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
