# 2StageSystemEweParturition
A 2-stage system which predicts ewe parturition by combining binary and multi-class classification approaches. The system is designed to process sensor data, optimize model hyperparameters, and evaluate performance using various metrics.

## Overview

This project implements a two-stage machine learning pipeline to predict ewe parturition. The first stage performs binary classification to distinguish between partum and non-partum states. The second stage performs multi-class classification to predict the number of hours until parturition.

The system leverages lightweight models (e.g., Decision Trees, LogReg, KNN) for binary classification and ensemble models (e.g., Random Forest, Extra Trees, Bagging) for multi-class classification. It includes functionality for hyperparameter optimization, data processing, and result aggregation.


## Key Files

- **`eweLabelling/scripts/`**: Scripts for preprocessing and aggregating ewe data (DataSet Labelling).
- **`twoStageSystem/lightweightDT/test.py`**: Main script for training, evaluating, and optimizing models.
- **`twoStageSystem/lightweightDT/3dVisualization.py`**: Script for visualizing model performance.
- **`twoStageSystem/lightweightDT/experiment_results/`**: Directory for storing results, models, and plots.
- **`twoStageSystem/RFBinary/aggregate_metrics.csv`**: csv file that contains the aggregated metrics for the binary, multiclass and overall system performance for a non-lightweight binary model.

## Features
- Binary Classification: Predicts whether a ewe is in a partum or non-partum state.
- Multi-class Classification: Predicts the number of hours until parturition.
- Hyperparameter Optimization: Uses Optuna to find the best hyperparameters for each model.
- Data Processing: Handles large datasets by processing data in manageable chunks.
- Visualization: Generates 3D surface plots for performance metrics.