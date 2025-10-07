# Creating Wrappers for External ML Libraries in MLPY

This guide explains how to create wrappers for external machine learning libraries to integrate them with MLPY.

## Overview

MLPY's architecture allows easy integration of any ML library through its wrapper system. The key is to implement the MLPY interface (`train`, `predict`, etc.) while delegating the actual computation to the external library.

## Basic Structure

A wrapper needs to:
1. Inherit from appropriate MLPY base classes
2. Convert between MLPY tasks and the library's data format
3. Implement the core methods: `train()` and `predict()`
4. Handle library-specific features and properties

## Example Wrappers

### 1. H2O Integration

```python
from mlpy.learners.base import Learner
import h2o
from h2o.estimators.estimator_base import H2OEstimator

class LearnerH2O(Learner):
    def __init__(self, estimator: H2OEstimator, **kwargs):
        # Initialize H2O if needed
        h2o.init()
        
        # Store the H2O estimator
        self.estimator = estimator
        
        # Detect properties (prob, importance, etc.)
        properties = self._detect_properties(estimator)
        
        super().__init__(properties=properties, packages={"h2o"}, **kwargs)
    
    def train(self, task, row_ids=None):
        # Convert MLPY task to H2O Frame
        h2o_frame = self._task_to_h2o_frame(task, row_ids)
        
        # Train H2O model
        self.estimator.train(
            x=task.feature_names,
            y=task.target_names[0],
            training_frame=h2o_frame
        )
        
        return self
    
    def predict(self, task, row_ids=None):
        # Convert task data to H2O Frame
        h2o_frame = self._task_to_h2o_frame(task, row_ids, features_only=True)
        
        # Get predictions
        h2o_preds = self.estimator.predict(h2o_frame)
        
        # Convert back to MLPY Prediction object
        return self._h2o_to_prediction(h2o_preds, task, row_ids)
```

### 2. XGBoost Native Integration

```python
import xgboost as xgb
from mlpy.learners.base import Learner

class LearnerXGBoost(Learner):
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.booster = None
        
        super().__init__(
            properties={"importance", "prob"},
            packages={"xgboost"},
            **kwargs
        )
    
    def train(self, task, row_ids=None):
        # Get data as numpy arrays
        X = task.data(rows=row_ids, cols=task.feature_names, data_format="array")
        y = task.truth(rows=row_ids)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train model
        self.booster = xgb.train(self.xgb_params, dtrain)
        
        return self
    
    def predict(self, task, row_ids=None):
        X = task.data(rows=row_ids, cols=task.feature_names, data_format="array")
        dtest = xgb.DMatrix(X)
        
        # Get predictions
        preds = self.booster.predict(dtest)
        
        # Return appropriate Prediction object
        return self._create_prediction(preds, task, row_ids)
```

### 3. LightGBM Integration

```python
import lightgbm as lgb
from mlpy.learners.classification import LearnerClassif

class LearnerLightGBM(LearnerClassif):
    def __init__(self, **lgb_params):
        self.lgb_params = lgb_params
        self.model = None
        
        super().__init__(
            properties={"importance", "prob"},
            packages={"lightgbm"},
            predict_type="prob",
            **kwargs
        )
    
    def train(self, task, row_ids=None):
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            task.data(rows=row_ids, cols=task.feature_names),
            label=task.truth(rows=row_ids),
            feature_name=task.feature_names
        )
        
        # Train model
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            valid_sets=[train_data]
        )
        
        return self
```

### 4. CatBoost Integration

```python
from catboost import CatBoostClassifier, CatBoostRegressor
from mlpy.learners.base import Learner

class LearnerCatBoost(Learner):
    def __init__(self, task_type="classif", **catboost_params):
        if task_type == "classif":
            self.model = CatBoostClassifier(**catboost_params)
        else:
            self.model = CatBoostRegressor(**catboost_params)
            
        super().__init__(
            properties={"importance", "prob", "shap"},
            packages={"catboost"},
            **kwargs
        )
    
    def train(self, task, row_ids=None):
        X = task.data(rows=row_ids, cols=task.feature_names)
        y = task.truth(rows=row_ids)
        
        # Handle categorical features
        cat_features = [i for i, ft in enumerate(task.feature_types.values()) 
                       if ft == "character"]
        
        self.model.fit(
            X, y,
            cat_features=cat_features if cat_features else None,
            verbose=False
        )
        
        return self
```

### 5. PyTorch Integration

```python
import torch
import torch.nn as nn
from mlpy.learners.base import Learner

class LearnerPyTorch(Learner):
    def __init__(self, model: nn.Module, optimizer_class=torch.optim.Adam, 
                 epochs=100, batch_size=32, **kwargs):
        self.model = model
        self.optimizer_class = optimizer_class
        self.epochs = epochs
        self.batch_size = batch_size
        
        super().__init__(
            properties={"deep_learning"},
            packages={"torch"},
            **kwargs
        )
    
    def train(self, task, row_ids=None):
        # Convert to PyTorch tensors
        X = torch.FloatTensor(
            task.data(rows=row_ids, cols=task.feature_names, data_format="array")
        )
        y = torch.LongTensor(task.truth(rows=row_ids))
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        optimizer = self.optimizer_class(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
```

## Best Practices

### 1. Data Conversion
```python
def _task_to_library_format(self, task, row_ids=None):
    """Convert MLPY task to library-specific format."""
    # Get data in appropriate format
    if self.library_prefers == "dataframe":
        return task.data(rows=row_ids, data_format="dataframe")
    elif self.library_prefers == "numpy":
        return task.data(rows=row_ids, data_format="array")
    elif self.library_prefers == "sparse":
        # Convert to sparse matrix if needed
        pass
```

### 2. Property Detection
```python
def _detect_properties(self, estimator):
    """Auto-detect what the model can do."""
    properties = set()
    
    # Check for probability predictions
    if hasattr(estimator, 'predict_proba'):
        properties.add('prob')
    
    # Check for feature importance
    if hasattr(estimator, 'feature_importances_'):
        properties.add('importance')
    
    # Check for interpretability features
    if hasattr(estimator, 'get_shap_values'):
        properties.add('shap')
        
    return properties
```

### 3. Error Handling
```python
def train(self, task, row_ids=None):
    try:
        # Attempt training
        self._train_internal(task, row_ids)
    except LibrarySpecificError as e:
        # Convert to MLPY-friendly error
        raise RuntimeError(f"Training failed: {e}") from e
    except OutOfMemoryError:
        # Suggest solutions
        raise RuntimeError(
            "Out of memory. Try: "
            "1) Reducing batch size, "
            "2) Using fewer features, "
            "3) Sampling the data"
        )
```

### 4. State Management
```python
class LearnerExternal(Learner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._is_trained = False
        
    @property
    def is_trained(self):
        return self._is_trained and self._model is not None
    
    def reset(self):
        """Reset to untrained state."""
        self._model = None
        self._is_trained = False
        # Clean up any library-specific resources
        return self
```

## Creating a Wrapper Package

To distribute your wrapper:

### 1. Project Structure
```
mlpy-contrib-library/
├── mlpy_contrib_library/
│   ├── __init__.py
│   ├── learners.py
│   ├── utils.py
│   └── tests/
├── setup.py
├── README.md
└── requirements.txt
```

### 2. Setup.py
```python
from setuptools import setup, find_packages

setup(
    name="mlpy-contrib-library",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mlpy>=0.1.0",
        "library>=1.0.0",  # The ML library you're wrapping
    ],
    python_requires=">=3.8",
)
```

### 3. Registration with MLPY
```python
# In your __init__.py
from mlpy.utils.registry import mlpy_learners

# Register your learners
mlpy_learners.register("library.classifier", LearnerLibraryClassifier)
mlpy_learners.register("library.regressor", LearnerLibraryRegressor)

# Make them easily importable
__all__ = ["LearnerLibraryClassifier", "LearnerLibraryRegressor", "learner_library"]
```

## Testing Your Wrapper

```python
import pytest
from mlpy.tasks import TaskClassif
from mlpy.measures import MeasureClassifAccuracy
from mlpy import resample

def test_library_wrapper():
    # Create task
    task = create_test_task()
    
    # Create learner
    learner = LearnerLibrary()
    
    # Test training
    learner.train(task)
    assert learner.is_trained
    
    # Test prediction
    pred = learner.predict(task)
    assert len(pred.response) == task.nrow
    
    # Test with resampling
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=3),
        measures=MeasureClassifAccuracy()
    )
    
    assert result.aggregate()['classif.acc'].mean() > 0.5
```

## Contributing Your Wrapper

1. Ensure compatibility with MLPY's core interface
2. Add comprehensive tests
3. Document all parameters and methods
4. Submit as a separate package or PR to MLPY-contrib
5. Consider adding examples and benchmarks

## Common Patterns

### Handling Different Task Types
```python
def learner_auto(estimator, **kwargs):
    """Automatically detect classifier vs regressor."""
    if hasattr(estimator, 'predict_proba') or is_classifier(estimator):
        return LearnerLibraryClassif(estimator, **kwargs)
    else:
        return LearnerLibraryRegr(estimator, **kwargs)
```

### Handling Special Features
```python
class LearnerWithExplainability(Learner):
    def explain(self, task, row_ids=None):
        """Get feature explanations."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
            
        # Get SHAP values or other explanations
        shap_values = self._get_shap_values(task, row_ids)
        
        return Explanation(
            values=shap_values,
            feature_names=task.feature_names,
            baseline=self._get_baseline()
        )
```

## Conclusion

Creating wrappers for external ML libraries in MLPY is straightforward:

1. Inherit from appropriate base classes
2. Implement data conversion methods
3. Implement `train()` and `predict()`
4. Handle library-specific features
5. Add proper error handling and state management

This allows any ML library to integrate seamlessly with MLPY's ecosystem of resampling, benchmarking, pipelines, and measures.