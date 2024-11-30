# Model Configuration Documentation

## Overview
This document describes the configuration options for the ML Price Predictor model using CatBoost.

## Model Parameters
- `iterations` (int): Maximum number of trees to build (60000)
- `l2_leaf_reg` (float): L2 regularization coefficient (2.7)
- `early_stopping_rounds` (int): Stop training if metric doesn't improve for N rounds (300)
- `posterior_sampling` (bool): Enable posterior sampling for better uncertainty estimates
- `grow_policy` (str): Tree construction policy ("SymmetricTree")
- `bootstrap_type` (str): Bootstrap type for bagging ("Bernoulli")
- `random_state` (int): Random seed for reproducibility (42)
- `leaf_estimation_method` (str): Method for leaf value calculation ("Newton")
- `score_function` (str): Score type for leaf estimation ("Cosine")
- `colsample_bylevel` (float): Subsample ratio of columns for each split (0.94)
- `thread_count` (int): Number of threads for training (4)

## Data Parameters
- `train_test_split` (float): Fraction of data to use for testing (0.2)
- `validation_split` (float): Fraction of training data to use for validation (0.1)
- `random_seed` (int): Random seed for data splitting (42)

### Feature Engineering
#### Categorical Features:
- `item_origin`: Origin of the item
- `steam_country`: User's Steam country
- `steam_community_ban`: Community ban status
- `steam_is_limited`: Limited account status
- `steam_cs2_wingman_rank_id`: CS2 Wingman rank
- `steam_cs2_rank_id`: CS2 competitive rank
- `steam_cs2_ban_type`: CS2 ban type
- `steam_currency`: Steam wallet currency

### Model Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- Pearson Correlation

## Usage
1. Copy the default configuration
2. Modify parameters as needed
3. Validate changes through cross-validation
4. Review training metrics for performance

## Best Practices
1. Start with default parameters
2. Adjust `iterations` and `early_stopping_rounds` based on convergence
3. Tune `l2_leaf_reg` for regularization
4. Modify `thread_count` based on available CPU cores
5. Keep `random_state` constant for reproducibility
