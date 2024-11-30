# Validation Criteria Documentation

## Overview
This document describes the validation criteria used to ensure model quality and reliability before deployment.

## Criteria Definitions

### Performance Metrics
- `rmse_threshold` (100.0): Maximum acceptable Root Mean Square Error
  - Higher values indicate larger prediction errors
  - Conservative threshold suitable for price prediction

- `mae_threshold` (50.0): Maximum acceptable Mean Absolute Error
  - More interpretable metric for stakeholders
  - Represents average prediction deviation in currency units

- `r2_threshold` (0.7): Minimum required R² score
  - Measures explained variance
  - 0.7 indicates strong predictive performance

- `pearson_threshold` (0.95): Minimum required Pearson correlation
  - Measures prediction accuracy
  - Critical threshold for model acceptance

## Usage
1. These criteria are automatically checked during:
   - Model validation
   - Model training
   - Fine-tuning process

2. Validation failures trigger:
   - Detailed error messages
   - Performance reports

## Modification Guidelines
1. Adjust thresholds based on:
   - Business requirements
   - Data characteristics
   - Performance needs

2. Document changes in:
   - Version control
   - Model validation reports
