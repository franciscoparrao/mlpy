# MLPY Model Persistence Guide

This guide explains how to save, load, and manage trained models in MLPY.

## Overview

MLPY provides a comprehensive persistence system that allows you to:

- ✅ Save trained models to disk in various formats
- ✅ Load models for prediction or further training
- ✅ Attach metadata to saved models
- ✅ Organize models with a registry system
- ✅ Export models as self-contained packages
- ✅ Support multiple serialization backends

## Quick Start

### Basic Save and Load

```python
from mlpy import save_model, load_model
from mlpy.learners.sklearn import learner_sklearn
from sklearn.ensemble import RandomForestClassifier

# Train a model
rf = RandomForestClassifier(n_estimators=100)
learner = learner_sklearn(rf)
learner.train(task)

# Save the model
save_model(learner, "my_model.pkl")

# Load the model
loaded_learner = load_model("my_model.pkl")

# Use for predictions
predictions = loaded_learner.predict(new_task)
```

### Save with Metadata

```python
# Save with additional information
save_model(
    learner,
    "model_with_metadata.pkl",
    metadata={
        "experiment_id": "exp_001",
        "accuracy": 0.95,
        "training_date": "2025-08-04",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }
)

# Load and access metadata
bundle = load_model("model_with_metadata.pkl", return_bundle=True)
print(bundle.metadata)  # {'experiment_id': 'exp_001', ...}
model = bundle.model
```

## Serialization Formats

### Pickle (Default)

The most general format that can handle any Python object:

```python
from mlpy.persistence import PickleSerializer

# Using pickle explicitly
save_model(learner, "model.pkl", serializer="pickle")

# Or with custom protocol
serializer = PickleSerializer(protocol=4)  # Python 3.4+ compatible
save_model(learner, "model.pkl", serializer=serializer)
```

**Pros:**
- Works with any Python object
- Preserves all model attributes
- Default format, always available

**Cons:**
- Not portable across Python versions
- Security risks with untrusted files
- Can be large for numerical data

### Joblib (Recommended for scikit-learn)

Optimized for scientific computing with compression support:

```python
# Save with joblib
save_model(learner, "model.joblib", serializer="joblib")

# With compression (0-9)
save_model(learner, "model.joblib", serializer="joblib", compression=3)

# Maximum compression
from mlpy.persistence import JoblibSerializer
serializer = JoblibSerializer(compression=9)
save_model(learner, "model_compressed.joblib", serializer=serializer)
```

**Pros:**
- Efficient for numpy arrays
- Built-in compression
- Standard for scikit-learn

**Cons:**
- Requires joblib installation
- Still Python-specific

### JSON (Metadata only)

For saving model metadata or simple configurations:

```python
# Save metadata
metadata = {
    "model_type": "RandomForest",
    "features": ["age", "income", "score"],
    "performance": {"auc": 0.89, "accuracy": 0.92}
}

save_model(metadata, "metadata.json", serializer="json")

# Load metadata
data = load_model("metadata.json")
```

### ONNX (Cross-platform)

For deploying models across different platforms:

```python
# Convert and save as ONNX (if supported)
save_model(learner, "model.onnx", serializer="onnx")

# Or use the conversion function
from mlpy.persistence.onnx_serializer import convert_to_onnx
convert_to_onnx(learner, "model.onnx")
```

**Pros:**
- Cross-platform deployment
- Optimized inference
- Industry standard

**Cons:**
- Limited model support
- Requires onnx/skl2onnx
- May lose some features

## Model Registry

The ModelRegistry provides organized model management:

### Setting Up a Registry

```python
from mlpy.persistence import ModelRegistry

# Create registry
registry = ModelRegistry("./model_registry")

# Register a trained model
registry.register_model(
    model=learner,
    name="customer_churn",
    version="v1.0",
    tags=["production", "classification"],
    metadata={
        "accuracy": 0.94,
        "false_positive_rate": 0.05,
        "training_samples": 10000
    }
)
```

### Working with Versions

```python
# Register improved version
registry.register_model(
    model=improved_learner,
    name="customer_churn",
    version="v2.0",
    tags=["production", "classification", "improved"],
    metadata={
        "accuracy": 0.96,
        "improvements": "Added feature engineering"
    }
)

# List all versions
models = registry.list_models()
print(models)  # {'customer_churn': ['v1.0', 'v2.0']}

# Load specific version
model_v1 = registry.load_model("customer_churn", version="v1.0")

# Load latest version
model_latest = registry.load_model("customer_churn")  # Returns v2.0

# Get metadata without loading
metadata = registry.get_metadata("customer_churn", version="v2.0")
print(f"Accuracy: {metadata['accuracy']}")
```

### Registry Management

```python
# Delete specific version
registry.delete_model("customer_churn", version="v1.0")

# Delete all versions
registry.delete_model("customer_churn")

# List all models in registry
all_models = registry.list_models()
for name, versions in all_models.items():
    print(f"{name}: {', '.join(versions)}")
```

## Saving Pipelines

MLPY pipelines are fully supported:

```python
from mlpy.pipelines import linear_pipeline, PipeOpScale, PipeOpLearner

# Create and train pipeline
pipeline = linear_pipeline(
    PipeOpScale(id="scaler"),
    PipeOpLearner(learner, id="model")
)
pipeline.train(task)

# Save entire pipeline
save_model(
    pipeline,
    "preprocessing_pipeline.pkl",
    metadata={
        "pipeline_steps": ["scaling", "random_forest"],
        "input_features": task.feature_names
    }
)

# Load and use
loaded_pipeline = load_model("preprocessing_pipeline.pkl")
predictions = loaded_pipeline.predict(new_task)
```

## Model Packages

Export models as self-contained packages for distribution:

```python
from mlpy.persistence import export_model_package

# Create distributable package
export_model_package(
    model=learner,
    output_path="model_package.zip",
    name="SentimentAnalyzer",
    include_dependencies=True,
    include_examples=True,
    metadata={
        "version": "1.0.0",
        "author": "ML Team",
        "description": "Production sentiment analysis model",
        "performance": {
            "accuracy": 0.92,
            "f1_score": 0.89
        }
    }
)
```

The package includes:
- `model.pkl`: The serialized model
- `requirements.txt`: Python dependencies
- `example.py`: Usage examples
- `README.md`: Documentation

## Best Practices

### 1. Choose the Right Serializer

```python
# For scikit-learn models
save_model(sklearn_learner, "model.joblib", serializer="joblib", compression=3)

# For pure MLPY models
save_model(mlpy_learner, "model.pkl", serializer="pickle")

# For deployment
save_model(compatible_learner, "model.onnx", serializer="onnx")
```

### 2. Always Include Metadata

```python
import datetime

metadata = {
    "model_id": "prod_model_001",
    "created_at": datetime.datetime.now().isoformat(),
    "git_commit": "abc123",
    "dataset_version": "2.1",
    "performance_metrics": {
        "train_accuracy": 0.98,
        "val_accuracy": 0.94,
        "test_accuracy": 0.93
    },
    "hyperparameters": learner.param_set
}

save_model(learner, "model.pkl", metadata=metadata)
```

### 3. Version Control Models

```python
# Use semantic versioning
registry.register_model(
    model=learner,
    name="recommender",
    version="2.1.0",  # major.minor.patch
    metadata={
        "changes": "Fixed cold start problem",
        "previous_version": "2.0.3"
    }
)
```

### 4. Validate Loaded Models

```python
# Load and validate
loaded_model = load_model("model.pkl")

# Check it's trained
assert loaded_model.is_trained, "Model not trained!"

# Verify on test data
test_predictions = loaded_model.predict(test_task)
assert len(test_predictions.response) == test_task.nrow

# Check reproducibility
original_pred = learner.predict(test_task)
loaded_pred = loaded_model.predict(test_task)
assert np.allclose(original_pred.response, loaded_pred.response)
```

### 5. Security Considerations

```python
# Only load trusted models
trusted_sources = ["/approved/models/", "/production/models/"]

model_path = "path/to/model.pkl"
if any(model_path.startswith(src) for src in trusted_sources):
    model = load_model(model_path)
else:
    raise ValueError("Model from untrusted source")

# Compute checksums
from mlpy.persistence.utils import compute_model_hash

# After saving
model_hash = compute_model_hash("model.pkl")
print(f"Model checksum: {model_hash}")

# Before loading
expected_hash = "abc123..."  # Store this securely
actual_hash = compute_model_hash("model.pkl")
if actual_hash != expected_hash:
    raise ValueError("Model file has been modified!")
```

## Advanced Usage

### Custom Serialization

Create custom serializers for specific needs:

```python
from mlpy.persistence import ModelSerializer, SERIALIZERS

@SERIALIZERS.register("custom")
class CustomSerializer(ModelSerializer):
    def can_serialize(self, obj):
        # Define what objects this handles
        return hasattr(obj, 'custom_attribute')
    
    def serialize(self, obj, path):
        # Custom serialization logic
        with open(path, 'w') as f:
            f.write(obj.to_custom_format())
        return {"format": "custom"}
    
    def deserialize(self, path):
        # Custom deserialization
        with open(path, 'r') as f:
            return MyClass.from_custom_format(f.read())
    
    @property
    def file_extension(self):
        return ".custom"

# Use custom serializer
save_model(special_model, "model.custom", serializer="custom")
```

### Lazy Loading

For large models, implement lazy loading:

```python
class LazyModelLoader:
    def __init__(self, path):
        self.path = path
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            print("Loading model...")
            self._model = load_model(self.path)
        return self._model
    
    def predict(self, task):
        return self.model.predict(task)

# Usage
lazy_model = LazyModelLoader("large_model.pkl")
# Model only loaded when needed
predictions = lazy_model.predict(task)
```

### Model Serving

Integrate with model serving frameworks:

```python
# FastAPI example
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load model once at startup
model = load_model("production_model.pkl")

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert to MLPY task
    data = pd.DataFrame([request.features])
    task = TaskRegr(data=data)
    
    # Make prediction
    prediction = model.predict(task)
    
    return {
        "prediction": float(prediction.response[0]),
        "model_version": model.metadata.get("version", "unknown")
    }
```

## Troubleshooting

### Import Errors After Loading

```python
# Ensure all dependencies are available
try:
    model = load_model("model.pkl")
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install required packages and try again")
```

### Version Compatibility

```python
# Check compatibility before loading
import pickle
import sys

# Peek at pickle protocol
with open("model.pkl", "rb") as f:
    protocol = pickle.load(f, encoding='bytes')
    
if protocol > pickle.HIGHEST_PROTOCOL:
    print(f"Model requires newer Python version")
```

### Large File Handling

```python
# For very large models, use compression
save_model(
    large_model,
    "large_model.joblib",
    serializer=JoblibSerializer(compression=9)
)

# Or save in chunks (custom implementation needed)
```

## Summary

MLPY's persistence system provides:

1. **Flexibility**: Multiple serialization formats for different needs
2. **Organization**: Model registry for version management
3. **Portability**: Export packages for easy distribution
4. **Metadata**: Rich metadata support for tracking
5. **Integration**: Works seamlessly with all MLPY components

Whether you're experimenting locally or deploying to production, MLPY's persistence features ensure your models are saved, versioned, and ready for use!