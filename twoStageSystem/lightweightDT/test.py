from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend, Parallel, delayed
import tensorflow as tf
import pickle
from datetime import datetime
import json
import gc
import seaborn as sns
import io
import os
import psutil
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import optuna
import time

# Fix TensorFlow GPU initialization issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Use only the first GPU

# Configure GPU memory growth - more carefully
print("TensorFlow version:", tf.__version__)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU devices available: {len(gpus)}")
        for gpu in gpus:
            # Limit memory growth
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for GPU: {gpu}")
            except Exception as e:
                print(f"Error setting memory growth: {e}")
        print("GPU memory growth enabled")
    else:
        print("No GPU found, falling back to CPU")
except Exception as e:
    print(f"Error configuring GPUs: {e}")

# Chunk size for data processing - adjust based on sample rate
CHUNK_SIZE = 500000  # Adjust based on your system's memory capacity

SAMPLE_RATES = [
    #('0.5Hz', '2s'),     # 0.5 Hz = 1 sample every 2 seconds
    #('1Hz', '1s'),       # 1 Hz = 1 sample per second
    ('2Hz', '500ms'),    # 2 Hz = 2 samples per second
    #('3Hz', '333ms'),    # 3 Hz = 3 samples per second
]

# Define window sizes to test
WINDOW_SIZES_MINUTES = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
#WINDOW_SIZES_MINUTES = [165, 180]
WINDOW_SIZES = [(min, min * 60) for min in WINDOW_SIZES_MINUTES]  # (minutes, seconds)

# Create output directories
os.makedirs('experiment_results', exist_ok=True)
os.makedirs('experiment_results/plots', exist_ok=True)
os.makedirs('experiment_results/models', exist_ok=True)
os.makedirs('experiment_results/models/binary', exist_ok=True)
os.makedirs('experiment_results/models/multiclass', exist_ok=True)

def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def get_adaptive_chunk_size(sample_rate_hz):
    """Get appropriate chunk size based on sample rate"""
    base_chunk_size = 500000
    rate_factor = float(sample_rate_hz.replace('Hz', ''))
    adaptive_size = int(base_chunk_size / rate_factor)
    print(f"Using adaptive chunk size of {adaptive_size} for {sample_rate_hz}")
    return adaptive_size

def dataframe_shift(df, columns, window_seconds, sample_rate_hz):
    """Create windowed features with memory optimization"""
    steps = int(window_seconds * float(sample_rate_hz.replace('Hz', '')))
    print(f"Window would have {steps} steps for {window_seconds}s window at {sample_rate_hz}")
    
    # Use fewer points for higher sampling rates
    rate_factor = float(sample_rate_hz.replace('Hz', ''))
    points_per_feature = max(8, min(16, int(16 / rate_factor)))
    
    if steps <= points_per_feature * len(columns):
        print(f"Using all {steps} points")
        # Add columns in batches to reduce memory pressure
        for i in range(1, steps, max(1, steps // 10)):
            batch_columns = []
            for col in columns:
                batch_columns.append(pl.col(col).shift(i).alias(f'prev_{i}_{col}'))
            df = df.with_columns(batch_columns)
            # Force garbage collection after each batch
            gc.collect()
    else:
        stride = max(1, steps // points_per_feature)
        print(f"Reducing {steps} points to ~{points_per_feature} points per feature with stride {stride}")
        
        for col in columns:
            sample_points = list(range(1, steps, stride))[:points_per_feature]
            batch_size = max(1, len(sample_points) // 4)  # Process in 4 batches
            
            for i in range(0, len(sample_points), batch_size):
                batch_points = sample_points[i:i+batch_size]
                batch_columns = []
                for point in batch_points:
                    batch_columns.append(pl.col(col).shift(point).alias(f'prev_{point}_{col}'))
                df = df.with_columns(batch_columns)
                # Force garbage collection after each batch
                gc.collect()
    
    return df.drop_nulls()

def optimize_threshold_for_class_balance(y_true, y_proba):
    # Target the correct proportion of partum samples
    target_proportion = sum(y_true) / len(y_true)
    
    # Find threshold that gives similar proportion in predictions
    best_threshold = 0.5
    best_diff = float('inf')
    
    for threshold in np.linspace(0.1, 0.9, 41):
        y_pred = (y_proba >= threshold).astype(int)
        pred_proportion = sum(y_pred) / len(y_pred)
        diff = abs(pred_proportion - target_proportion)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
            
    return best_threshold

def optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type, is_binary=True, n_trials=15):
    """Optimize hyperparameters using Optuna for a specific model type"""
    def objective(trial):
        if model_type == 'DecisionTreeClassifier':
            params = {
                'max_depth': trial.suggest_int('max_depth', 200, 300),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 600, 1000),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'class_weight': 'balanced',
                'random_state': 42
            }
            model = DecisionTreeClassifier(**params)
        
        elif model_type == 'RandomForestClassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 150),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'class_weight': 'balanced',
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
        
        elif model_type == 'ExtraTreesClassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 150),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'random_state': 42
            }
            model = ExtraTreesClassifier(**params)
        
        # elif model_type == 'Bagging':
        #     params = {
        #         'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        #         'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        #         'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        #         'random_state': 42
        #     }
        #     model = BaggingClassifier(**params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train and evaluate
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba") and is_binary:
            val_proba = model.predict_proba(X_val)[:, 1]
            best_threshold = optimize_threshold_for_class_balance(y_val, val_proba)
            val_pred = (val_proba >= best_threshold).astype(int)
        else:
            val_pred = model.predict(X_val)
        
        return matthews_corrcoef(y_val, val_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def generate_models(n_input, n_output, optimized_hyperparams=None, model_type='binary', lightweight=None):
    """Generate models with optimized hyperparameters
    
    Args:
        n_input: Number of input features
        n_output: Number of output classes
        optimized_hyperparams: Dictionary of hyperparameters (already filtered for the model type)
        model_type: 'binary' or 'multiclass'
        lightweight: If True, only use lightweight models (Decision Tree), else use ensemble models
    """
    # Set default lightweight value based on model_type if not specified
    if lightweight is None:
        lightweight = (model_type == 'binary')
        
    print(f"Generating models for {model_type}, lightweight={lightweight}")
    print(f"Hyperparams: {optimized_hyperparams}")
    
    if optimized_hyperparams is None:
        # No optimized parameters, use defaults
        if lightweight:
            # Lightweight models (primarily for binary classification)
            models = [
                ('DecisionTreeClassifier', DecisionTreeClassifier(
                max_depth=244, max_features='sqrt', max_leaf_nodes=850, 
                random_state=42, class_weight='balanced')),
            ]
        else:
            # More complex models (primarily for multiclass)
            models = [
                ('RandomForestClassifier', RandomForestClassifier(random_state=42, class_weight='balanced')),
                ('ExtraTreesClassifier', ExtraTreesClassifier(random_state=42, class_weight='balanced')),
            ]
        return models
    
    # Determine which model types to use based on lightweight flag
    if lightweight:
        model_types = ['DecisionTreeClassifier']
    else:
        model_types = ['RandomForestClassifier', 'ExtraTreesClassifier']
    
    # Create models using the optimized hyperparameters
    models = []
    for model_name in model_types:
        params = optimized_hyperparams.get(model_name, {})
        if params:
            print(f"Creating {model_name} with params: {params}")
            if model_name == 'DecisionTreeClassifier':
                params['random_state'] = 42
                params['class_weight'] = 'balanced'
                models.append((model_name, DecisionTreeClassifier(**params)))
            elif model_name == 'RandomForestClassifier':
                params['random_state'] = 42
                params['class_weight'] = 'balanced'
                models.append((model_name, RandomForestClassifier(**params)))
            elif model_name == 'ExtraTreesClassifier':
                params['random_state'] = 42
                params['class_weight'] = 'balanced'
                models.append((model_name, ExtraTreesClassifier(**params)))
        else:
            print(f"No params found for {model_name}, skipping")
    
    if not models:
        print("WARNING: No models were created! Falling back to defaults.")
        return generate_models(n_input, n_output, None, model_type, lightweight)
    
    print(f"Created {len(models)} models: {[m[0] for m in models]}")
    return models

def train_and_evaluate_model(name, clf, X_train, y_train, X_val, y_val, X_test, y_test, is_binary=True):
    """Train and evaluate a single model with validation set for model selection"""
    print(f"Training {name} with {X_train.shape[0]} samples...")
    
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training {name}: {e}")
        return None

    # Optimize threshold on validation set for binary
    if hasattr(clf, "predict_proba") and is_binary:
        # Use validation set for threshold optimization
        val_proba = clf.predict_proba(X_val)[:, 1]
        best_threshold = optimize_threshold_for_class_balance(y_val, val_proba)
        print(f"Best threshold for {name}: {best_threshold:.2f}")
        
        # Apply threshold to test set
        test_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (test_proba >= best_threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)
        best_threshold = None
    
    # Calculate sample weights for test set
    if is_binary:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)
        sample_weights = np.array([class_weights[yi] for yi in y_test])
    else:
        sample_weights = None
    
    # Calculate metrics on test set
    acc = accuracy_score(y_test, y_pred, sample_weight=sample_weights)
    mcc = matthews_corrcoef(y_test, y_pred, sample_weight=sample_weights)
    average = 'weighted' if not is_binary else 'binary'
    f1 = f1_score(y_test, y_pred, average=average, sample_weight=sample_weights)
    recall = recall_score(y_test, y_pred, average=average, sample_weight=sample_weights)
    precision = precision_score(y_test, y_pred, average=average, sample_weight=sample_weights)

    # Print test metrics
    print(f'{name:<22} {acc:>8.2f} {precision:>9.2f} {recall:>6.2f} {f1:>8.2f} {mcc:>5.2f}')
    
    # Get model size
    with io.BytesIO() as buffer:
        pickle.dump(clf, buffer)
        model_size_kb = buffer.getbuffer().nbytes / 1024
    
    return {
        'model': clf,
        'name': name,
        'predictions': y_pred,
        'threshold': best_threshold,
        'metrics': (acc, precision, recall, f1, mcc),
        'size_kb': model_size_kb,
        'val_predictions': clf.predict(X_val) if not is_binary else (clf.predict_proba(X_val)[:, 1] >= best_threshold).astype(int)
    }

def process_data_in_chunks(df_windowed, sample_rate, window_size, unique_labels, experiment_results, aggregate_data, optimized_hyperparams=None):
    """Process windowed data in manageable chunks with train/val/test splits"""
    if optimized_hyperparams is None:
        optimized_hyperparams = {}
    
    print(f'--- Processing data - Sample Rate: {sample_rate} - Window: {window_size[0]}min ---', flush=True)
    
    # Create binary labels
    df_windowed = df_windowed.with_columns(
        pl.when(pl.col('Class') < 13).then(1).otherwise(0).alias('Binary_Class')
    )
    
    # Analyze class distribution
    binary_distribution = df_windowed['Binary_Class'].value_counts()
    print(f"Original binary class distribution: {binary_distribution}")
    
    # First split into train+val/test
    train_val_indices, test_indices = train_test_split(
        np.arange(len(df_windowed)), test_size=0.2, random_state=42, 
        stratify=df_windowed['Binary_Class'].to_numpy()
    )
    
    # Then split train+val into train/val (0.75/0.25 of remaining data = 0.6/0.2 of total)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.25, random_state=42,
        stratify=df_windowed['Binary_Class'].to_numpy()[train_val_indices]
    )
    
    # Extract feature columns
    X_columns = [col for col in df_windowed.columns if col not in ['Class', 'Binary_Class', 'Time']]
    
    # Extract test set
    X_test_full = df_windowed.drop(['Class', 'Binary_Class', 'Time']).to_numpy()[test_indices]
    y_test_full = df_windowed['Class'].to_numpy()[test_indices]
    y_binary_test_full = df_windowed['Binary_Class'].to_numpy()[test_indices]

    # Extract validation set
    X_val_full = df_windowed.drop(['Class', 'Binary_Class', 'Time']).to_numpy()[val_indices]
    y_val_full = df_windowed['Class'].to_numpy()[val_indices]
    y_binary_val_full = df_windowed['Binary_Class'].to_numpy()[val_indices]

    # Get partum and non-partum indices for test and validation
    test_partum_indices = np.where(y_binary_test_full == 1)[0]
    test_non_partum_indices = np.where(y_binary_test_full == 0)[0]
    
    val_partum_indices = np.where(y_binary_val_full == 1)[0]
    val_non_partum_indices = np.where(y_binary_val_full == 0)[0]

    # Use original test and validation datasets without balancing
    test_balanced_indices = np.arange(len(y_binary_test_full))
    val_balanced_indices = np.arange(len(y_binary_val_full))

    X_test = X_test_full[test_balanced_indices]
    y_test = y_test_full[test_balanced_indices]
    y_binary_test = y_binary_test_full[test_balanced_indices]
    
    X_val = X_val_full[val_balanced_indices]
    y_val = y_val_full[val_balanced_indices]
    y_binary_val = y_binary_val_full[val_balanced_indices]

    print("\nDataset split sizes:", flush=True)
    print(f"Test dataset: {len(y_binary_test)} samples", flush=True)
    print(f"  - Partum: {len(test_partum_indices)}, Non-partum: {len(test_non_partum_indices)}", flush=True)
    print(f"Validation dataset: {len(y_binary_val)} samples", flush=True)
    print(f"  - Partum: {len(val_partum_indices)}, Non-partum: {len(val_non_partum_indices)}", flush=True)

    # Get partum samples for multiclass (from full test/val sets)
    partum_mask_test = y_binary_test_full == 1
    X_test_partum = X_test_full[partum_mask_test]
    y_test_partum = y_test_full[partum_mask_test]
    
    partum_mask_val = y_binary_val_full == 1
    X_val_partum = X_val_full[partum_mask_val]
    y_val_partum = y_val_full[partum_mask_val]
    
    # Get training indices for partum/non-partum
    train_partum_indices = train_indices[df_windowed['Binary_Class'].to_numpy()[train_indices] == 1]
    train_non_partum_indices = train_indices[df_windowed['Binary_Class'].to_numpy()[train_indices] == 0]

    # Combine indices for balanced training
    train_balanced_indices = np.concatenate([train_partum_indices, train_non_partum_indices])
    np.random.shuffle(train_balanced_indices)

    print(f"Training dataset: {len(train_balanced_indices)} samples", flush=True)
    print(f"  - Partum: {len(train_partum_indices)}, Non-partum: {len(train_non_partum_indices)}", flush=True)
    
    # Process binary classification
    print("\n=== STAGE 1: Binary Classification (Partum vs Non-Partum) ===", flush=True)
    print(f'{"":<22} Accuracy Precision Recall F1-score   MCC', flush=True)
    
    # Load training data in chunks
    chunk_size = min(get_adaptive_chunk_size(sample_rate), len(train_balanced_indices))
    X_train_binary = []
    y_binary_train = []
    
    for i in range(0, len(train_balanced_indices), chunk_size):
        chunk_indices = train_balanced_indices[i:i+chunk_size]
        X_chunk = df_windowed.select(X_columns).to_numpy()[chunk_indices]
        y_chunk = df_windowed['Binary_Class'].to_numpy()[chunk_indices]
        
        X_train_binary.append(X_chunk)
        y_binary_train.append(y_chunk)
    
    X_train_binary = np.vstack(X_train_binary)
    y_binary_train = np.hstack(y_binary_train)

    # Train binary models
    n_input_binary = X_train_binary.shape[1]
    n_output_binary = 2
    print("Binary HyperParams: ", optimized_hyperparams.get('binary', {}))
    binary_models = generate_models(
        n_input_binary, n_output_binary, 
        optimized_hyperparams=(optimized_hyperparams or {}).get('binary', None),
        model_type='binary',
        lightweight=True
    )
    
    binary_model_results = []
    for name, clf in binary_models:
        with parallel_backend('loky', n_jobs=4):
            # Modified to include validation set
            result = train_and_evaluate_model(
                name, clf, 
                X_train_binary, y_binary_train,
                X_val, y_binary_val,  # Validation data
                X_test, y_binary_test,
                is_binary=True
            )
            binary_model_results.append(result)
        
        # Create confusion matrix using test set
        cm = confusion_matrix(y_binary_test, result['predictions'])
        cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Binary Confusion Matrix - {name} - MCC: {result['metrics'][4]:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0.5, 1.5], ['Non-Partum', 'Partum'])
        plt.yticks([0.5, 1.5], ['Non-Partum', 'Partum'])
        plt.close()
        
        # Save binary model
        model_path = f"experiment_results/models/binary/{name}_{sample_rate}_{window_size[0]}min.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f, protocol=pickle.HIGHEST_PROTOCOL)
        
        gc.collect()
    
    # Select best binary model
    if binary_model_results:
        best_binary_result = max(binary_model_results, key=lambda x: x['metrics'][4])  # Sort by MCC
    else:
        print("No binary models were successfully trained.")
        return experiment_results  # Exit the function early if no results
    
    # Add binary predictions to aggregate data
    aggregate_data['binary_true'].extend(y_binary_test.tolist())
    aggregate_data['binary_pred'].extend(best_binary_result['predictions'].tolist())
    
    # Store the best binary model if it's better than what we have or if we don't have one yet
    if ('best_binary_model' not in aggregate_data or 
        'best_binary_mcc' not in aggregate_data or 
        best_binary_result['metrics'][4] > aggregate_data['best_binary_mcc']):
        
        aggregate_data['best_binary_model'] = best_binary_result['model']
        aggregate_data['best_binary_mcc'] = best_binary_result['metrics'][4]
        aggregate_data['best_binary_name'] = best_binary_result['name']
        print(f"New best binary model: {best_binary_result['name']} with MCC: {best_binary_result['metrics'][4]:.3f}")
    
    # Load partum training data for multiclass in chunks
    X_train_partum = []
    y_train_partum = []
    
    print("\nLoading partum training data for multiclass...", flush=True)
    
    # Extract partum training data in chunks
    chunk_size = min(get_adaptive_chunk_size(sample_rate), len(train_partum_indices))
    for i in range(0, len(train_partum_indices), chunk_size):
        chunk_indices = train_partum_indices[i:i+chunk_size]
        X_chunk = df_windowed.select(X_columns).to_numpy()[chunk_indices]
        y_chunk = df_windowed['Class'].to_numpy()[chunk_indices]
        
        X_train_partum.append(X_chunk)
        y_train_partum.append(y_chunk)
    
    # Combine chunks for multiclass training
    X_train_partum = np.vstack(X_train_partum)
    y_train_partum = np.hstack(y_train_partum)
    
    # Initialize multiclass variables
    multiclass_result = None
    
    # Free up memory before multiclass training
    del X_train_binary, y_binary_train
    print_memory_usage()
    gc.collect()
    print_memory_usage()
    
    # Train multiclass models if we have partum samples
    if len(X_train_partum) > 0 and len(X_test_partum) > 0:
        print("\n=== STAGE 2: Multiclass Classification (Hours until Partum) ===", flush=True)
        print(f'{"":<22} Accuracy Precision Recall F1-score   MCC', flush=True)
        
        n_input_multi = X_train_partum.shape[1]
        n_output_multi = len(np.unique(y_train_partum))
        print("Multiclass HyperParams: ", optimized_hyperparams.get('multiclass', {}))
        multiclass_models = generate_models(
            n_input_multi, n_output_multi, 
            optimized_hyperparams=(optimized_hyperparams or {}).get('multiclass', None),
            model_type='multiclass',
            lightweight=False
        )
        
        multiclass_model_results = []
        
        # Train each multiclass model
        for mc_name, mc_clf in multiclass_models:
            try:
                # Reduce n_jobs from 4 to 1 to prevent memory issues
                with parallel_backend('loky', n_jobs=2):  # Changed from 4 to 1
                    result = train_and_evaluate_model(
                        mc_name, mc_clf,
                        X_train_partum, y_train_partum,
                        X_val_partum, y_val_partum,  
                        X_test_partum, y_test_partum,
                        is_binary=False
                    )
                    
                    # Check if result is None before trying to use it
                    if result is not None:
                        multiclass_model_results.append(result)
                        
                        # Save multiclass model with fold identifier
                        model_path = f"experiment_results/models/multiclass/{mc_name}_{sample_rate}_{window_size}min.pkl"
                        with open(model_path, 'wb') as f:
                            pickle.dump(result['model'], f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        print(f"Training {mc_name} failed, skipping this model.")
                        
                    # Clean up immediately
                    gc.collect()
                    
            except (MemoryError, RuntimeError) as e:
                print(f"Memory error training {mc_name}: {e}")
                continue

        # Select best multiclass model if any were trained successfully
        if multiclass_model_results:
            multiclass_result = max(multiclass_model_results, key=lambda x: x['metrics'][4])  # Sort by MCC
            
            # Add multiclass predictions to aggregate data
            aggregate_data['multiclass_true'].extend(y_test_partum.tolist())
            aggregate_data['multiclass_pred'].extend(multiclass_result['predictions'].tolist())
            
            # Store the best multiclass model if it's better than what we have or if we don't have one yet
            if ('best_multiclass_model' not in aggregate_data or 
                'best_multiclass_mcc' not in aggregate_data or 
                multiclass_result['metrics'][4] > aggregate_data['best_multiclass_mcc']):
                
                aggregate_data['best_multiclass_model'] = multiclass_result['model']
                aggregate_data['best_multiclass_mcc'] = multiclass_result['metrics'][4]
                aggregate_data['best_multiclass_name'] = multiclass_result['name']
                print(f"New best multiclass model: {multiclass_result['name']} with MCC: {multiclass_result['metrics'][4]:.3f}")

        # plot confusion matrix for the best multiclass model
        if multiclass_result:
            cm = confusion_matrix(y_test_partum, multiclass_result['predictions'])
            cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
            plt.title(f"Multiclass Confusion Matrix - {multiclass_result['name']} - MCC: {multiclass_result['metrics'][4]:.2f}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(len(unique_labels)), unique_labels, rotation=45)
            plt.yticks(range(len(unique_labels)), unique_labels)
            plt.close()
    
    # Evaluate combined system if we have both models
    combined_metrics = None
    if best_binary_result and multiclass_result:
        print("\n=== COMBINED SYSTEM EVALUATION ===", flush=True)
        
        # Initialize predictions array with default non-partum class (13)
        y_combined_pred = np.ones_like(y_test) * 13
        
        # Get indices where best binary model predicts partum
        partum_pred_indices = np.where(best_binary_result['predictions'] == 1)[0]
        
        # For those indices, use the best multiclass model
        if len(partum_pred_indices) > 0:
            mc_predictions = multiclass_result['model'].predict(X_test[partum_pred_indices])
            y_combined_pred[partum_pred_indices] = mc_predictions
        
        # Modify combined system evaluation
        # Add combined predictions to aggregate data
        aggregate_data['all_true'].extend(y_test.tolist())
        aggregate_data['combined_pred'].extend(y_combined_pred.tolist())

        # Calculate separate class weights for partum and non-partum samples
        partum_mask = y_test < 13
        class_weights = np.ones_like(y_test, dtype=float)
        class_weights[partum_mask] = len(y_test) / (2 * np.sum(partum_mask))
        class_weights[~partum_mask] = len(y_test) / (2 * np.sum(~partum_mask))

        # Calculate combined metrics with balanced weights
        combined_acc = accuracy_score(y_test, y_combined_pred, sample_weight=class_weights)
        combined_precision = precision_score(y_test, y_combined_pred, average='weighted', sample_weight=class_weights)
        combined_recall = recall_score(y_test, y_combined_pred, average='weighted', sample_weight=class_weights)
        combined_f1 = f1_score(y_test, y_combined_pred, average='weighted', sample_weight=class_weights)
        combined_mcc = matthews_corrcoef(y_test, y_combined_pred, sample_weight=class_weights)

        combined_metrics = (combined_acc, combined_precision, combined_recall, combined_f1, combined_mcc)
        
        print(f'Combined System     {combined_acc:>8.2f} {combined_precision:>9.2f} {combined_recall:>6.2f} {combined_f1:>8.2f} {combined_mcc:>5.2f}')

        # Store the best combined system if it's better than what we have or if we don't have one yet
        if ('best_combined_mcc' not in aggregate_data or combined_mcc > aggregate_data['best_combined_mcc']):
            aggregate_data['best_combined_mcc'] = combined_mcc
            aggregate_data['best_combined_binary_model'] = best_binary_result['model']
            aggregate_data['best_combined_multiclass_model'] = multiclass_result['model']
            aggregate_data['best_combined_binary_name'] = best_binary_result['name']
            aggregate_data['best_combined_multiclass_name'] = multiclass_result['name']
            print(f"New best combined system: {best_binary_result['name']} + {multiclass_result['name']} with MCC: {combined_mcc:.3f}")

        # plot combined confusion matrix
        cm = confusion_matrix(y_test, y_combined_pred)
        cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Combined Confusion Matrix - MCC: {combined_mcc:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(unique_labels)), unique_labels, rotation=45)
        plt.yticks(range(len(unique_labels)), unique_labels)
        plt.close()
    
    # Write results to CSV
    csv_path = 'experiment_results/model_performance.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('sample_rate,window_minutes,binary_model,binary_accuracy,binary_precision,binary_recall,' +
                   'binary_f1,binary_mcc,binary_threshold,binary_size_kb,multiclass_model,multiclass_accuracy,' +
                   'multiclass_precision,multiclass_recall,multiclass_f1,multiclass_mcc,combined_accuracy,' +
                   'combined_precision,combined_recall,combined_f1,combined_mcc\n')
    
    # Format values for CSV
    def format_value(value):
        if value is None:
            return ""
        return f"{float(value):.6f}"
    
    # Get metrics from results
    binary_metrics = best_binary_result['metrics']
    mc_metrics = multiclass_result['metrics'] if multiclass_result else (None, None, None, None, None)
    
    # Add row to CSV
    with open(csv_path, 'a') as f:
        row = [
            sample_rate,
            window_size[0],
            best_binary_result['name'],
            format_value(binary_metrics[0]),  # accuracy
            format_value(binary_metrics[1]),  # precision
            format_value(binary_metrics[2]),  # recall
            format_value(binary_metrics[3]),  # f1
            format_value(binary_metrics[4]),  # mcc
            format_value(best_binary_result['threshold']),
            format_value(best_binary_result['size_kb']),
            multiclass_result['name'] if multiclass_result else "",
            format_value(mc_metrics[0]),  # accuracy
            format_value(mc_metrics[1]),  # precision
            format_value(mc_metrics[2]),  # recall
            format_value(mc_metrics[3]),  # f1
            format_value(mc_metrics[4]),  # mcc
            format_value(combined_metrics[0] if combined_metrics else None),  # combined accuracy
            format_value(combined_metrics[1] if combined_metrics else None),  # combined precision
            format_value(combined_metrics[2] if combined_metrics else None),  # combined recall
            format_value(combined_metrics[3] if combined_metrics else None),  # combined f1
            format_value(combined_metrics[4] if combined_metrics else None)   # combined mcc
        ]
        f.write(",".join(map(str, row)) + "\n")
    
    # Save experimental result for JSON output
    result_summary = {
        'sample_rate': sample_rate,
        'window_minutes': window_size[0],
        'binary_model': best_binary_result['name'],
        'binary_mcc': float(binary_metrics[4]),
    }
    
    if multiclass_result:
        result_summary.update({
            'multiclass_model': multiclass_result['name'],
            'multiclass_mcc': float(mc_metrics[4]),
        })
    
    if combined_metrics:
        result_summary['combined_mcc'] = float(combined_metrics[4])
        result_summary['binary_model_for_combined'] = best_binary_result['name']
        result_summary['multiclass_model_for_combined'] = multiclass_result['name']
    
    experiment_results.append(result_summary)
    
    print(f"Results written to CSV: {csv_path}")
    gc.collect()
    return experiment_results


def process_files_for_config(rate_hz, rate_interval, window_min, window_sec, unique_labels, experiment_results, optimized_hyperparams=None):   
    """Process all files for one configuration (sample rate and window size)"""
    print(f"\n--- Testing Sample Rate: {rate_hz}, Window Size: {window_min} minutes ---", flush=True)
    
    all_files = os.listdir('../data/train2')    
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    
    # Create combined results dataframe after windowing
    all_results = None
    
    # Initialize aggregate data storage
    aggregate_data = {
        'binary_true': [],
        'binary_pred': [],
        'multiclass_true': [],
        'multiclass_pred': [],
        'all_true': [],
        'combined_pred': []
    }
    
    # Process each file separately and completely
    for dataset in tqdm(csv_files, desc="Processing files"):
        print(f"Processing {dataset}...", flush=True)
        
        # Load and resample file
        df = pl.read_csv(f'../data/train2/{dataset}', separator=';')
        
        df = df.with_columns(
            pl.col('Time').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S.%3f')
        )
        
        # Resample at current rate
        df_resampled = df.set_sorted('Time').group_by_dynamic('Time', every=rate_interval).agg(
            pl.col('Acc_X (mg)').median(),
            pl.col('Acc_Y (mg)').median(),
            pl.col('Acc_Z (mg)').median(),
            pl.col('Temperature (C)').median(),
            pl.col('Class').mode().first()
        )
        
        # Scale
        df_resampled = df_resampled.with_columns(
            pl.col('Acc_X (mg)').map_batches(lambda x: pl.Series(minmax_scale(x))),
            pl.col('Acc_Y (mg)').map_batches(lambda x: pl.Series(minmax_scale(x))),
            pl.col('Acc_Z (mg)').map_batches(lambda x: pl.Series(minmax_scale(x)))
        )
        
        # Create windowed features
        df_windowed = dataframe_shift(
            df_resampled, 
            columns=['Acc_X (mg)', 'Acc_Y (mg)', 'Acc_Z (mg)', 'Temperature (C)'], 
            window_seconds=window_sec,
            sample_rate_hz=rate_hz
        )
        
        # Add to combined results or process immediately
        if all_results is None:
            all_results = df_windowed
        else:
            all_results = pl.concat([all_results, df_windowed])
        
        # Process in chunks if the combined dataset is getting too large
        chunk_size = get_adaptive_chunk_size(rate_hz)
        if all_results.shape[0] > chunk_size * 2:
            print(f"Processing accumulated data batch (rows: {all_results.shape[0]})...", flush=True)
            experiment_results = process_data_in_chunks(
                all_results, 
                sample_rate=rate_hz,
                window_size=(window_min, window_sec),
                unique_labels=unique_labels,
                experiment_results=experiment_results,
                aggregate_data=aggregate_data,
                optimized_hyperparams=optimized_hyperparams
            )
            # Reset accumulated results
            all_results = None
            gc.collect()
        
        # Free memory
        del df, df_resampled, df_windowed
        gc.collect()
    
    # Process any remaining data
    if all_results is not None and all_results.shape[0] > 0:
        print(f"Processing final data batch (rows: {all_results.shape[0]})...", flush=True)
        experiment_results = process_data_in_chunks(
            all_results, 
            sample_rate=rate_hz,
            window_size=(window_min, window_sec),
            unique_labels=unique_labels,
            experiment_results=experiment_results,
            aggregate_data=aggregate_data,
            optimized_hyperparams=optimized_hyperparams
        )
    
    # Create aggregate confusion matrices and plots
    create_aggregate_confusion_matrices(aggregate_data, rate_hz, window_min, unique_labels)
    
    return experiment_results


def create_aggregate_confusion_matrices(aggregate_data, rate_hz, window_min, unique_labels):
    """Create confusion matrices for aggregate predictions across all files"""
    print("\n=== Creating aggregate confusion matrices ===")

   # Binary confusion matrix
    if aggregate_data['binary_true'] and aggregate_data['binary_pred']:

       # Calculate class weights for binary metrics
        binary_classes = np.unique(aggregate_data['binary_true'])
        binary_class_weights = compute_class_weight('balanced',
                                                classes=binary_classes,
                                                y=aggregate_data['binary_true'])
        binary_sample_weights = np.array([binary_class_weights[yi] for yi in aggregate_data['binary_true']])

        # Calculate overall binary metrics with weights
        binary_acc = accuracy_score(
            aggregate_data['binary_true'],
            aggregate_data['binary_pred'],
            sample_weight=binary_sample_weights
        )
        binary_mcc = matthews_corrcoef(
            aggregate_data['binary_true'],
            aggregate_data['binary_pred'],
            sample_weight=binary_sample_weights
        )
        binary_f1 = f1_score(
            aggregate_data['binary_true'],
            aggregate_data['binary_pred'],
            average='binary',  # Changed from 'weighted' to 'binary'
            sample_weight=binary_sample_weights
        )
        binary_cm = confusion_matrix(
            aggregate_data['binary_true'],
            aggregate_data['binary_pred'],
            labels=[0, 1]
        )

        # Normalize confusion matrix by rows (row normalization)
        row_sums = binary_cm.sum(axis=1, keepdims=True)
        binary_cm_percent = (binary_cm / row_sums * 100) if row_sums.any() else np.zeros_like(binary_cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(binary_cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Aggregate Binary Confusion Matrix - {rate_hz} - {window_min}min -> MCC: {binary_mcc:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0.5, 1.5], ['Non-Partum', 'Partum'])
        plt.yticks([0.5, 1.5], ['Non-Partum', 'Partum'])
        plt.savefig(f"experiment_results/plots/aggregate_binary_cm_{rate_hz}_{window_min}min.png")
        plt.close()

        print(f"Aggregate Binary Results: Acc={binary_acc:.3f}, MCC={binary_mcc:.3f}, F1={binary_f1:.3f}")

        # Save aggregate metrics to a JSON file
        aggregate_metrics = {
            'binary': {
                'accuracy': float(binary_acc),
                'mcc': float(binary_mcc),
                'f1': float(binary_f1),
                'confusion_matrix': binary_cm.tolist()
            }
        }

        # If we have a 'best_binary_model' in aggregate_data, save it
        if 'best_binary_model' in aggregate_data:
            # Save the best binary model
            model_path = f"experiment_results/models/binary/best_aggregate_{rate_hz}_{window_min}min.pkl"
            print(f"Saving best binary model to {model_path}")
            with open(model_path, 'wb') as f:
                pickle.dump(aggregate_data['best_binary_model'], f, protocol=pickle.HIGHEST_PROTOCOL)

    # Multiclass confusion matrix
    if aggregate_data['multiclass_true'] and aggregate_data['multiclass_pred']:
        # Get unique classes in the multiclass data
        unique_mc_classes = sorted(list(set(aggregate_data['multiclass_true'] + aggregate_data['multiclass_pred'])))

        multi_cm = confusion_matrix(
            aggregate_data['multiclass_true'],
            aggregate_data['multiclass_pred'],
            labels=unique_mc_classes
        )


        # Calculate class weights for multiclass metrics
        multi_classes = np.unique(aggregate_data['multiclass_true'])
        multi_class_weights = compute_class_weight('balanced',
                                                classes=multi_classes,
                                                y=aggregate_data['multiclass_true'])
        multi_sample_weights = np.array([multi_class_weights[yi] for yi in aggregate_data['multiclass_true']])

        # Calculate overall multiclass metrics with weights
        multi_acc = accuracy_score(
            aggregate_data['multiclass_true'],
            aggregate_data['multiclass_pred'],
            sample_weight=multi_sample_weights
        )
        multi_mcc = matthews_corrcoef(
            aggregate_data['multiclass_true'],
            aggregate_data['multiclass_pred'],
            sample_weight=multi_sample_weights
        )
        multi_f1 = f1_score(
            aggregate_data['multiclass_true'],
            aggregate_data['multiclass_pred'],
            average='weighted',
            sample_weight=multi_sample_weights
        )

         # Handle potential class imbalance
        row_sums = multi_cm.sum(axis=1)
        multi_cm_percent = np.zeros_like(multi_cm, dtype=float)
        for i, row_sum in enumerate(row_sums):
            if row_sum > 0:
                multi_cm_percent[i] = multi_cm[i] / row_sum * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(multi_cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Aggregate Multiclass Confusion Matrix - {rate_hz} - {window_min}min")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(np.arange(len(unique_mc_classes)) + 0.5, unique_mc_classes, rotation=45)
        plt.yticks(np.arange(len(unique_mc_classes)) + 0.5, unique_mc_classes)
        plt.savefig(f"experiment_results/plots/aggregate_multiclass_cm_{rate_hz}_{window_min}min.png")
        plt.close()

        print(f"Aggregate Multiclass Results: Acc={multi_acc:.3f}, MCC={multi_mcc:.3f}, F1={multi_f1:.3f}")

        # Add multiclass metrics to aggregate_metrics
        if 'aggregate_metrics' not in locals():
            aggregate_metrics = {}

        aggregate_metrics['multiclass'] = {
            'accuracy': float(multi_acc),
            'mcc': float(multi_mcc),
            'f1': float(multi_f1),
            'confusion_matrix': multi_cm.tolist()
        }

        # If we have a 'best_multiclass_model' in aggregate_data, save it
        if 'best_multiclass_model' in aggregate_data:
            # Save the best multiclass model
            model_path = f"experiment_results/models/multiclass/best_aggregate_{rate_hz}_{window_min}min.pkl"
            print(f"Saving best multiclass model to {model_path}")
            with open(model_path, 'wb') as f:
                pickle.dump(aggregate_data['best_multiclass_model'], f, protocol=pickle.HIGHEST_PROTOCOL)

    # Combined system confusion matrix
    if aggregate_data['all_true'] and aggregate_data['combined_pred']:
        # Get unique classes in the combined data
        unique_combined_classes = sorted(list(set(aggregate_data['all_true'] + aggregate_data['combined_pred'])))

        combined_cm = confusion_matrix(
            aggregate_data['all_true'],
            aggregate_data['combined_pred'],
            labels=unique_combined_classes
        )

       # Calculate class weights for combined metrics
        partum_mask = np.array(aggregate_data['all_true']) < 13
        class_weights = np.ones_like(aggregate_data['all_true'], dtype=float)

        # Calculate weights for two groups: partum (0-12) and non-partum (13)
        class_weights[partum_mask] = len(aggregate_data['all_true']) / (2 * np.sum(partum_mask))
        class_weights[~partum_mask] = len(aggregate_data['all_true']) / (2 * np.sum(~partum_mask))

        # Calculate overall combined metrics with binary-style weights
        combined_acc = accuracy_score(
            aggregate_data['all_true'],
            aggregate_data['combined_pred'],
            sample_weight=class_weights
        )
        combined_mcc = matthews_corrcoef(
            aggregate_data['all_true'],
            aggregate_data['combined_pred'],
            sample_weight=class_weights
        )
        combined_f1 = f1_score(
            aggregate_data['all_true'],
            aggregate_data['combined_pred'],
            average='weighted',
            sample_weight=class_weights
        )

          # Handle potential class imbalance
        row_sums = combined_cm.sum(axis=1)
        combined_cm_percent = np.zeros_like(combined_cm, dtype=float)
        for i, row_sum in enumerate(row_sums):
            if row_sum > 0:
                combined_cm_percent[i] = combined_cm[i] / row_sum * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(combined_cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Aggregate Combined System Matrix - {rate_hz} - {window_min}min -> MCC: {combined_mcc:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(np.arange(len(unique_combined_classes)) + 0.5, unique_combined_classes, rotation=45)
        plt.yticks(np.arange(len(unique_combined_classes)) + 0.5, unique_combined_classes)
        plt.savefig(f"experiment_results/plots/aggregate_combined_cm_{rate_hz}_{window_min}min.png")
        plt.close()

        print(f"Aggregate Combined Results: Acc={combined_acc:.3f}, MCC={combined_mcc:.3f}, F1={combined_f1:.3f}")

                # Add combined metrics to aggregate_metrics
        if 'aggregate_metrics' not in locals():
            aggregate_metrics = {}

        aggregate_metrics['combined'] = {
            'accuracy': float(combined_acc),
            'mcc': float(combined_mcc),
            'f1': float(combined_f1),
            'confusion_matrix': combined_cm.tolist()
        }

    # Save aggregate metrics to a JSON file
    if 'aggregate_metrics' in locals():
        metrics_path = f"experiment_results/metrics_{rate_hz}_{window_min}min.json"
        with open(metrics_path, 'w') as f:
            json.dump(aggregate_metrics, f)

        # Also save to comprehensive CSV
        csv_path = 'experiment_results/aggregate_metrics.csv'
        is_new_csv = not os.path.exists(csv_path)

        # Calculate all required metrics for each model type
        binary_precision = precision_score(aggregate_data['binary_true'], aggregate_data['binary_pred'], average='weighted') if 'binary' in aggregate_metrics else None
        binary_recall = recall_score(aggregate_data['binary_true'], aggregate_data['binary_pred'], average='weighted') if 'binary' in aggregate_metrics else None

        multiclass_precision = precision_score(aggregate_data['multiclass_true'], aggregate_data['multiclass_pred'], average='weighted') if 'multiclass' in aggregate_metrics else None
        multiclass_recall = recall_score(aggregate_data['multiclass_true'], aggregate_data['multiclass_pred'], average='weighted') if 'multiclass' in aggregate_metrics else None

        combined_precision = precision_score(aggregate_data['all_true'], aggregate_data['combined_pred'], average='weighted') if 'combined' in aggregate_metrics else None
        combined_recall = recall_score(aggregate_data['all_true'], aggregate_data['combined_pred'], average='weighted') if 'combined' in aggregate_metrics else None

        # Get model names from aggregate_data
        binary_model = aggregate_data.get('best_binary_name', '') if 'binary' in aggregate_metrics else ''
        multiclass_model = aggregate_data.get('best_multiclass_name', '') if 'multiclass' in aggregate_metrics else ''

        # Get model size if available
        binary_size_kb = 0
        if 'best_binary_model' in aggregate_data:
            with io.BytesIO() as buffer:
                pickle.dump(aggregate_data['best_binary_model'], buffer)
                binary_size_kb = buffer.getbuffer().nbytes / 1024

        # Format values for CSV
        def format_value(value):
            if value is None:
                return ""
            return f"{float(value):.6f}"

        # Write to CSV
        with open(csv_path, 'a') as f:
            # Write header if new file
            if is_new_csv:
                f.write('sample_rate,window_minutes,binary_model,binary_accuracy,binary_precision,binary_recall,' +
                    'binary_f1,binary_mcc,binary_threshold,binary_size_kb,multiclass_model,multiclass_accuracy,' +
                    'multiclass_precision,multiclass_recall,multiclass_f1,multiclass_mcc,combined_accuracy,' +
                    'combined_precision,combined_recall,combined_f1,combined_mcc\n')

            # Prepare row data
            row = [
                rate_hz,
                window_min,
                binary_model,
                format_value(aggregate_metrics.get('binary', {}).get('accuracy')),
                format_value(binary_precision),
                format_value(binary_recall),
                format_value(aggregate_metrics.get('binary', {}).get('f1')),
                format_value(aggregate_metrics.get('binary', {}).get('mcc')),
                format_value(aggregate_data.get('best_binary_threshold', None)),
                format_value(binary_size_kb),
                multiclass_model,
                format_value(aggregate_metrics.get('multiclass', {}).get('accuracy')),
                format_value(multiclass_precision),
                format_value(multiclass_recall),
                format_value(aggregate_metrics.get('multiclass', {}).get('f1')),
                format_value(aggregate_metrics.get('multiclass', {}).get('mcc')),
                format_value(aggregate_metrics.get('combined', {}).get('accuracy')),
                format_value(combined_precision),
                format_value(combined_recall),
                format_value(aggregate_metrics.get('combined', {}).get('f1')),
                format_value(aggregate_metrics.get('combined', {}).get('mcc'))
            ]

            # Write row to CSV
            f.write(",".join(map(str, row)) + "\n")


def sample_random_chunks(csv_files, num_chunks=5):
    """Sample random chunks from the dataset"""
    import random
    return random.sample(csv_files, min(num_chunks, len(csv_files)))

def optimize_on_random_chunks(rate_hz, rate_interval, window_min, window_sec, unique_labels, num_chunks=5):
    """Run hyperparameter optimization on randomly selected chunks"""
    print(f"\n--- Optimizing Hyperparameters for Sample Rate: {rate_hz}, Window Size: {window_min} minutes ---", flush=True)
    
    # Get all files
    all_files = os.listdir('../data/train2')    
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    
    # Sample random chunks
    import random
    random_chunks = random.sample(csv_files, min(num_chunks, len(csv_files)))
    print(f"Selected {len(random_chunks)} random chunks for optimization: {random_chunks}", flush=True)
    
    # Process and combine these chunks
    combined_data = None
    
    for dataset in random_chunks:
        print(f"Processing {dataset} for optimization...", flush=True)
        
        # Load and process file (same as in process_files_for_config)
        df = pl.read_csv(f'../data/train2/{dataset}', separator=';')
        df = df.with_columns(pl.col('Time').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S.%3f'))
        
        # Resample and window (same as in your existing code)
        df_resampled = df.set_sorted('Time').group_by_dynamic('Time', every=rate_interval).agg(
            pl.col('Acc_X (mg)').median(),
            pl.col('Acc_Y (mg)').median(),
            pl.col('Acc_Z (mg)').median(),
            pl.col('Temperature (C)').median(),
            pl.col('Class').mode().first()
        )
        
        # Scale and window
        df_resampled = df_resampled.with_columns(
            pl.col('Acc_X (mg)').map_batches(lambda x: pl.Series(minmax_scale(x))),
            pl.col('Acc_Y (mg)').map_batches(lambda x: pl.Series(minmax_scale(x))),
            pl.col('Acc_Z (mg)').map_batches(lambda x: pl.Series(minmax_scale(x)))
        )
        
        df_windowed = dataframe_shift(df_resampled, 
            columns=['Acc_X (mg)', 'Acc_Y (mg)', 'Acc_Z (mg)', 'Temperature (C)'], 
            window_seconds=window_sec, sample_rate_hz=rate_hz)
        
        # Add to combined data
        if combined_data is None:
            combined_data = df_windowed
        else:
            combined_data = pl.concat([combined_data, df_windowed])
        
        del df, df_resampled, df_windowed
        gc.collect()
    
    # Create binary labels and split data
    combined_data = combined_data.with_columns(
        pl.when(pl.col('Class') < 13).then(1).otherwise(0).alias('Binary_Class')
    )
    
    # Split for training/validation
    X_columns = [col for col in combined_data.columns if col not in ['Class', 'Binary_Class', 'Time']]
    train_indices, val_indices = train_test_split(
        np.arange(len(combined_data)), test_size=0.25, random_state=42, 
        stratify=combined_data['Binary_Class'].to_numpy()
    )
    
    # Extract data
    X_train = combined_data.select(X_columns).to_numpy()[train_indices]
    y_binary_train = combined_data['Binary_Class'].to_numpy()[train_indices]
    y_multi_train = combined_data['Class'].to_numpy()[train_indices]
    
    X_val = combined_data.select(X_columns).to_numpy()[val_indices]
    y_binary_val = combined_data['Binary_Class'].to_numpy()[val_indices]
    y_multi_val = combined_data['Class'].to_numpy()[val_indices]
    
    # Optimize hyperparameters
    binary_hyperparams = {}
    for model_type in ['DecisionTreeClassifier']:
        binary_hyperparams[model_type] = optimize_hyperparameters(
            X_train, y_binary_train, X_val, y_binary_val, model_type, is_binary=True)
    
    # Optimize multiclass if we have partum samples
    multiclass_hyperparams = {}
    partum_train = y_binary_train == 1
    partum_val = y_binary_val == 1
    if np.sum(partum_train) > 0 and np.sum(partum_val) > 0:
        for model_type in ['RandomForestClassifier', 'ExtraTreesClassifier']:
            multiclass_hyperparams[model_type] = optimize_hyperparameters(
                X_train[partum_train], y_multi_train[partum_train], 
                X_val[partum_val], y_multi_val[partum_val], 
                model_type, is_binary=False)

    # Save and return
    hyperparams = {'binary': binary_hyperparams, 'multiclass': multiclass_hyperparams}
    with open(f"experiment_results/hyperparams_{rate_hz}_{window_min}min.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)
        
    return hyperparams

def run_experiment():
    experiment_results = []

    
    # Process each combination of sample rate and window size
    for rate_hz, rate_interval in SAMPLE_RATES:
        # Get unique labels (do this once outside the loops)
        sample_df = pl.read_csv(f'../data/train2/{os.listdir("../data/train2")[0]}', separator=';')
        unique = sample_df.unique(subset=['Class'], maintain_order=True)
        unique_labels = sorted(unique['Class'].to_list())
        print("Unique classes found:", unique_labels, flush=True)
        
        for window_min, window_sec in WINDOW_SIZES:
            # First run hyperparameter optimization on random chunks
            optimized_hyperparams = optimize_on_random_chunks(
                rate_hz, rate_interval, window_min, window_sec, unique_labels)
            
            # Process one file at a time for this combination
            experiment_results = process_files_for_config(
                rate_hz, 
                rate_interval, 
                window_min, 
                window_sec, 
                unique_labels, 
                experiment_results, 
                optimized_hyperparams
            )
            
            # Save intermediate results after each window size
            print("Saving intermediate results...", flush=True)
            with open(f'experiment_results/results_{rate_hz}_{window_min}min.json', 'w') as f:
                json.dump(experiment_results, f)
    
    # Save final results
    print("Saving final results...", flush=True)
    with open('experiment_results/results.json', 'w') as f:
        json.dump(experiment_results, f)
    
    try:
        results_df = pd.read_csv('experiment_results/model_performance.csv')
        
        # Find best configuration for each model type
        best_binary = results_df.loc[results_df['binary_mcc'].idxmax()]
        best_multiclass = results_df.loc[results_df['multiclass_mcc'].idxmax()] if 'multiclass_mcc' in results_df.columns else None
        best_combined = results_df.loc[results_df['combined_mcc'].idxmax()] if 'combined_mcc' in results_df.columns else None
        
        best_configs = {
            'binary': {
                'sample_rate': best_binary['sample_rate'],
                'window_minutes': int(best_binary['window_minutes']),
                'model': best_binary['binary_model'],
                'mcc': float(best_binary['binary_mcc']),
                'accuracy': float(best_binary['binary_accuracy']),
                'model_path': f"experiment_results/models/binary/best_aggregate_{best_binary['sample_rate']}_{int(best_binary['window_minutes'])}min.pkl"
            }
        }
        
        if best_multiclass is not None:
            best_configs['multiclass'] = {
                'sample_rate': best_multiclass['sample_rate'],
                'window_minutes': int(best_multiclass['window_minutes']),
                'model': best_multiclass['multiclass_model'],
                'mcc': float(best_multiclass['multiclass_mcc']),
                'accuracy': float(best_multiclass['multiclass_accuracy']),
                'model_path': f"experiment_results/models/multiclass/best_aggregate_{best_multiclass['sample_rate']}_{int(best_multiclass['window_minutes'])}min.pkl"
            }
        
        if best_combined is not None:
            best_configs['combined'] = {
                'sample_rate': best_combined['sample_rate'],
                'window_minutes': int(best_combined['window_minutes']),
                'binary_model': best_combined['binary_model'],
                'multiclass_model': best_combined['multiclass_model'],
                'mcc': float(best_combined['combined_mcc']),
                'accuracy': float(best_combined['combined_accuracy'])
            }
        
        # Save best configurations
        with open('experiment_results/best_configurations.json', 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        print("Best configurations saved to experiment_results/best_configurations.json", flush=True)
        
    except Exception as e:
        print(f"Error creating best configurations summary: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    print("Done!", flush=True)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Experiment started at: {start_time}", flush=True)
    
    try:
        run_experiment()
    except Exception as e:
        print(f"Error in experiment: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Experiment completed at: {end_time}", flush=True)
    print(f"Total duration: {duration}", flush=True)


