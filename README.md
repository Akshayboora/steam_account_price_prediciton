# ML Price Predictor

A machine learning system for predicting gaming account prices.

### Best model path
```
models/trained/category_1_model.onnx
```
## Project Structure
```
ml_price_predictor/
├── src/                    # Source code
│   ├── models/            # ML model implementations
│   ├── data_processing/   # Data processing utilities
│   └── inference/         # Training and inference scripts
├── tests/                 # Test suite
└── models/               # Saved models
    ├── trained/         # CatBoost models
    └── onnx/            # ONNX exported models
```

## Quick Start

### Installation
```bash
pip install -e .
```

### Usage

1. Training a model (trains a model on given data and saves it already in ONNX format):
```bash
python src/train.py \
    --category-id CATEGORY_ID \
    --data-path PATH_TO_YOUR_TRAINING_DATA \
    --output-dir models/trained
```

2. Validating a model:
```bash
python src/inference/validate.py \
    --model-path models/trained/category_CATEGORY_ID_model.onnx \
    --data-path YOUR_TEST_DATA_PATH \
    --category-id CATEGORY_ID \
    --output-dir DIR_TO_SAVE_VAL_RESULTS
```

3. Making predictions:
```bash
python src/inference/predict.py \
    --model-path models/onnx/category_CATEGORY_ID_model.onnx \
    --data-path PATH_TO_YOUR_TEST_DATA \
    --output-path predictions.csv \
    --category-id CATEGORY_ID
```

4. Fine-tuning an existing model:
```bash
python src/finetune.py \
    --model-path models/onnx/category_CATEGORY_ID_model.onnx \
    --data-path PATH_TO_YOUR_FINETUNING_DATA \
    --category-id CATEGORY_ID \
    --output-dir models/trained \
    --iterations 1000 \
    --learning-rate 0.03
```

## Model Architecture
- Uses CatBoost for robust price prediction with early stopping
- Supports category-specific model training with automatic feature preprocessing
- Handles both 'target' and 'sold_price' columns for flexible data input
- ONNX export for efficient deployment
- Supports fine-tuning of existing models on new data:
  - Preserves model knowledge while adapting to new patterns
  - Configurable learning rate and iterations
  - Automatic early stopping to prevent overfitting

## Validation Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Pearson Correlation (target > 0.95)

### Feature Engineering
The model implements preprocessing:
- Automatic handling of timestamp features
- One-hot encoding for categorical variables
- Missing value imputation
- Feature aggregation (sum, mean, std)
- Price-related feature engineering
- Steam-specific feature processing
