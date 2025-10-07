"""
Validators for MLPY data and tasks.

This module provides validation functions for data quality,
task compatibility, and model requirements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings


def validate_task_data(
    data: Union[pd.DataFrame, np.ndarray],
    target: Optional[str] = None,
    task_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate data for ML tasks.
    
    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input data to validate
    target : str, optional
        Target column name if DataFrame
    task_type : str, optional
        Type of task ('classification', 'regression', 'clustering')
    
    Returns
    -------
    dict
        Validation results with keys:
        - valid: bool, whether data is valid
        - errors: list of error messages
        - warnings: list of warning messages
        - suggestions: list of suggestions
        - stats: dict of data statistics
    """
    errors = []
    warnings = []
    suggestions = []
    stats = {}
    
    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # Basic data checks
    if data is None or (hasattr(data, 'empty') and data.empty):
        errors.append("Data is empty or None")
        return {
            'valid': False,
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'stats': stats
        }
    
    # Check data shape
    n_samples, n_features = data.shape
    stats['n_samples'] = n_samples
    stats['n_features'] = n_features
    
    # Minimum samples check
    if n_samples < 10:
        errors.append(f"Insufficient samples: {n_samples} < 10")
        suggestions.append("Collect more data samples (minimum 10 recommended)")
    elif n_samples < 50:
        warnings.append(f"Low sample size: {n_samples} samples may lead to poor generalization")
        suggestions.append("Consider collecting more data for better model performance")
    
    # Check for missing values
    missing_counts = data.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        missing_ratio = total_missing / (n_samples * n_features)
        stats['missing_ratio'] = missing_ratio
        stats['missing_columns'] = missing_counts[missing_counts > 0].to_dict()
        
        if missing_ratio > 0.5:
            errors.append(f"Too many missing values: {missing_ratio:.1%} of data is missing")
        elif missing_ratio > 0.2:
            warnings.append(f"High missing values: {missing_ratio:.1%} of data is missing")
            suggestions.append("Consider imputation strategies or removing features with many missing values")
        else:
            warnings.append(f"Missing values detected in {len(missing_counts[missing_counts > 0])} columns")
            suggestions.append("Use imputation or handle missing values before training")
    
    # Check for constant features
    if n_samples > 1:
        constant_features = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            warnings.append(f"Constant features detected: {constant_features}")
            suggestions.append("Remove constant features as they provide no information")
            stats['constant_features'] = constant_features
    
    # Check for duplicates
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        duplicate_ratio = duplicates / n_samples
        stats['duplicate_ratio'] = duplicate_ratio
        
        if duplicate_ratio > 0.5:
            warnings.append(f"High number of duplicate rows: {duplicates} ({duplicate_ratio:.1%})")
            suggestions.append("Remove duplicate rows or verify data collection process")
        elif duplicate_ratio > 0.1:
            warnings.append(f"Duplicate rows detected: {duplicates} ({duplicate_ratio:.1%})")
    
    # Target column validation
    if target:
        if target not in data.columns:
            errors.append(f"Target column '{target}' not found in data")
        else:
            target_data = data[target]
            
            # Check target missing values
            target_missing = target_data.isnull().sum()
            if target_missing > 0:
                errors.append(f"Target column has {target_missing} missing values")
                suggestions.append("Remove rows with missing target values")
            
            # Task-specific validation
            if task_type == 'classification':
                n_classes = target_data.nunique()
                stats['n_classes'] = n_classes
                
                if n_classes < 2:
                    errors.append("Classification requires at least 2 classes")
                elif n_classes > n_samples / 2:
                    warnings.append(f"Too many classes ({n_classes}) for {n_samples} samples")
                    suggestions.append("Consider reducing number of classes or collecting more data")
                
                # Check class balance
                class_counts = target_data.value_counts()
                min_class_count = class_counts.min()
                max_class_count = class_counts.max()
                
                if min_class_count < 5:
                    warnings.append(f"Some classes have very few samples (min: {min_class_count})")
                    suggestions.append("Consider collecting more samples for minority classes")
                
                imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
                if imbalance_ratio > 10:
                    warnings.append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
                    suggestions.append("Consider using class weights or resampling techniques")
                
            elif task_type == 'regression':
                # Check for outliers in target
                if pd.api.types.is_numeric_dtype(target_data):
                    q1 = target_data.quantile(0.25)
                    q3 = target_data.quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((target_data < q1 - 3*iqr) | (target_data > q3 + 3*iqr)).sum()
                    
                    if outliers > 0:
                        outlier_ratio = outliers / n_samples
                        stats['target_outliers'] = outliers
                        warnings.append(f"Target has {outliers} potential outliers ({outlier_ratio:.1%})")
                        suggestions.append("Consider outlier removal or robust regression methods")
    
    # Feature type analysis
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    stats['numeric_features'] = len(numeric_features)
    stats['categorical_features'] = len(categorical_features)
    
    # High cardinality check for categorical features
    for col in categorical_features:
        if col != target:
            cardinality = data[col].nunique()
            if cardinality > n_samples / 10:
                warnings.append(f"High cardinality in '{col}': {cardinality} unique values")
                suggestions.append(f"Consider encoding or grouping values in '{col}'")
    
    # Check for infinite values in numeric features
    for col in numeric_features:
        if col != target:
            if np.isinf(data[col].values).any():
                errors.append(f"Infinite values found in column '{col}'")
                suggestions.append(f"Remove or replace infinite values in '{col}'")
    
    # Determine if data is valid
    valid = len(errors) == 0
    
    return {
        'valid': valid,
        'errors': errors,
        'warnings': warnings,
        'suggestions': suggestions,
        'stats': stats
    }


def validate_model_params(
    model_class: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate model parameters.
    
    Parameters
    ----------
    model_class : str
        Name of the model class
    params : dict
        Parameters to validate
    
    Returns
    -------
    dict
        Validation results
    """
    errors = []
    warnings = []
    validated_params = params.copy()
    
    # Common parameter validations
    if 'n_estimators' in params:
        if params['n_estimators'] < 1:
            errors.append("n_estimators must be >= 1")
        elif params['n_estimators'] < 10:
            warnings.append("Low n_estimators may lead to poor performance")
    
    if 'max_depth' in params:
        if params['max_depth'] is not None and params['max_depth'] < 1:
            errors.append("max_depth must be >= 1 or None")
    
    if 'learning_rate' in params:
        if params['learning_rate'] <= 0 or params['learning_rate'] > 1:
            errors.append("learning_rate must be in (0, 1]")
    
    if 'n_neighbors' in params:
        if params['n_neighbors'] < 1:
            errors.append("n_neighbors must be >= 1")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'validated_params': validated_params
    }


def validate_prediction_format(
    predictions: Any,
    expected_length: int,
    task_type: str = 'classification'
) -> Dict[str, Any]:
    """
    Validate prediction format and consistency.
    
    Parameters
    ----------
    predictions : Any
        Predictions to validate
    expected_length : int
        Expected number of predictions
    task_type : str
        Type of task
    
    Returns
    -------
    dict
        Validation results
    """
    errors = []
    warnings = []
    
    # Check prediction length
    if hasattr(predictions, '__len__'):
        pred_length = len(predictions)
        if pred_length != expected_length:
            errors.append(f"Prediction length mismatch: got {pred_length}, expected {expected_length}")
    else:
        errors.append("Predictions must be iterable")
    
    # Task-specific validation
    if task_type == 'classification':
        # Check if all predictions are valid classes
        if hasattr(predictions, '__iter__'):
            unique_preds = set(predictions)
            if None in unique_preds:
                errors.append("Predictions contain None values")
    
    elif task_type == 'regression':
        # Check for NaN or infinite values
        if isinstance(predictions, (list, np.ndarray)):
            pred_array = np.array(predictions)
            if np.isnan(pred_array).any():
                errors.append("Predictions contain NaN values")
            if np.isinf(pred_array).any():
                errors.append("Predictions contain infinite values")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def validate_train_test_split(
    train_indices: List[int],
    test_indices: List[int],
    total_samples: int
) -> Dict[str, Any]:
    """
    Validate train/test split indices.
    
    Parameters
    ----------
    train_indices : list
        Training set indices
    test_indices : list
        Test set indices
    total_samples : int
        Total number of samples
    
    Returns
    -------
    dict
        Validation results
    """
    errors = []
    warnings = []
    
    # Check for overlap
    train_set = set(train_indices)
    test_set = set(test_indices)
    overlap = train_set.intersection(test_set)
    
    if overlap:
        errors.append(f"Train and test sets overlap: {len(overlap)} samples")
    
    # Check coverage
    all_indices = train_set.union(test_set)
    if len(all_indices) != total_samples:
        missing = total_samples - len(all_indices)
        warnings.append(f"Not all samples used: {missing} samples missing")
    
    # Check for valid indices
    if train_indices:
        if min(train_indices) < 0 or max(train_indices) >= total_samples:
            errors.append("Train indices out of bounds")
    
    if test_indices:
        if min(test_indices) < 0 or max(test_indices) >= total_samples:
            errors.append("Test indices out of bounds")
    
    # Check split ratio
    if train_indices and test_indices:
        train_ratio = len(train_indices) / (len(train_indices) + len(test_indices))
        if train_ratio < 0.5:
            warnings.append(f"Small training set: {train_ratio:.1%}")
        elif train_ratio > 0.95:
            warnings.append(f"Small test set: {1-train_ratio:.1%}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'total_size': total_samples
    }