"""
Neural Architecture Search (NAS)
=================================

Automated neural network architecture design.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available. NAS features will be limited.")

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available. NAS features will be limited.")


class ArchitectureSpace:
    """Define the search space for neural architectures."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: int,
        task_type: str = "classification",
        max_layers: int = 10,
        max_units: int = 512
    ):
        """
        Initialize architecture space.
        
        Args:
            input_shape: Input data shape
            output_shape: Output shape (num classes or 1 for regression)
            task_type: "classification" or "regression"
            max_layers: Maximum number of layers
            max_units: Maximum units per layer
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.task_type = task_type
        self.max_layers = max_layers
        self.max_units = max_units
        
        # Define search space
        self.layer_types = ["dense", "dropout", "batch_norm"]
        self.activation_functions = ["relu", "tanh", "sigmoid", "elu", "selu"]
        self.optimizers = ["adam", "sgd", "rmsprop", "adamax"]
        self.learning_rates = [0.001, 0.01, 0.1, 0.0001]
        self.dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.layer_sizes = [32, 64, 128, 256, 512]
    
    def sample_architecture(self, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Sample a random architecture."""
        rng = np.random.RandomState(random_state)
        
        # Sample number of layers
        n_layers = rng.randint(2, self.max_layers + 1)
        
        # Sample architecture
        architecture = {
            "layers": [],
            "optimizer": rng.choice(self.optimizers),
            "learning_rate": rng.choice(self.learning_rates),
            "batch_size": rng.choice([16, 32, 64, 128]),
            "epochs": rng.choice([10, 20, 50, 100])
        }
        
        # Build layers
        for i in range(n_layers):
            if i == n_layers - 1:
                # Output layer
                if self.task_type == "classification":
                    architecture["layers"].append({
                        "type": "dense",
                        "units": self.output_shape,
                        "activation": "softmax" if self.output_shape > 1 else "sigmoid"
                    })
                else:
                    architecture["layers"].append({
                        "type": "dense",
                        "units": self.output_shape,
                        "activation": "linear"
                    })
            else:
                # Hidden layer
                layer_type = rng.choice(self.layer_types)
                
                if layer_type == "dense":
                    architecture["layers"].append({
                        "type": "dense",
                        "units": rng.choice(self.layer_sizes),
                        "activation": rng.choice(self.activation_functions)
                    })
                elif layer_type == "dropout":
                    architecture["layers"].append({
                        "type": "dropout",
                        "rate": rng.choice(self.dropout_rates)
                    })
                elif layer_type == "batch_norm":
                    architecture["layers"].append({
                        "type": "batch_norm"
                    })
        
        return architecture


class NetworkBuilder:
    """Build neural networks from architecture definitions."""
    
    def __init__(self, backend: str = "tensorflow"):
        """
        Initialize network builder.
        
        Args:
            backend: "tensorflow" or "pytorch"
        """
        self.backend = backend
        
        if backend == "tensorflow" and not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")
        elif backend == "pytorch" and not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
    
    def build_tensorflow_model(
        self,
        architecture: Dict[str, Any],
        input_shape: Tuple[int, ...]
    ) -> Any:
        """Build TensorFlow/Keras model."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")
        
        model = keras.Sequential()
        
        # Add input layer
        model.add(layers.InputLayer(input_shape=input_shape))
        
        # Add layers
        for layer_config in architecture["layers"]:
            layer_type = layer_config["type"]
            
            if layer_type == "dense":
                model.add(layers.Dense(
                    units=layer_config["units"],
                    activation=layer_config.get("activation", "relu")
                ))
            elif layer_type == "dropout":
                model.add(layers.Dropout(rate=layer_config["rate"]))
            elif layer_type == "batch_norm":
                model.add(layers.BatchNormalization())
        
        # Compile model
        optimizer = self._get_tensorflow_optimizer(
            architecture["optimizer"],
            architecture["learning_rate"]
        )
        
        if architecture.get("task_type") == "classification":
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
        else:
            loss = "mse"
            metrics = ["mae"]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def build_pytorch_model(
        self,
        architecture: Dict[str, Any],
        input_shape: Tuple[int, ...]
    ) -> Any:
        """Build PyTorch model."""
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")
        
        class DynamicNet(nn.Module):
            def __init__(self, architecture, input_shape):
                super(DynamicNet, self).__init__()
                self.layers = nn.ModuleList()
                
                # Calculate input size
                input_size = np.prod(input_shape)
                prev_size = input_size
                
                # Build layers
                for layer_config in architecture["layers"]:
                    layer_type = layer_config["type"]
                    
                    if layer_type == "dense":
                        units = layer_config["units"]
                        self.layers.append(nn.Linear(prev_size, units))
                        
                        # Add activation
                        activation = layer_config.get("activation", "relu")
                        if activation == "relu":
                            self.layers.append(nn.ReLU())
                        elif activation == "tanh":
                            self.layers.append(nn.Tanh())
                        elif activation == "sigmoid":
                            self.layers.append(nn.Sigmoid())
                        elif activation == "elu":
                            self.layers.append(nn.ELU())
                        
                        prev_size = units
                        
                    elif layer_type == "dropout":
                        self.layers.append(nn.Dropout(p=layer_config["rate"]))
                    elif layer_type == "batch_norm":
                        self.layers.append(nn.BatchNorm1d(prev_size))
            
            def forward(self, x):
                x = x.view(x.size(0), -1)  # Flatten
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return DynamicNet(architecture, input_shape)
    
    def _get_tensorflow_optimizer(self, name: str, learning_rate: float):
        """Get TensorFlow optimizer."""
        if name == "adam":
            return keras.optimizers.Adam(learning_rate=learning_rate)
        elif name == "sgd":
            return keras.optimizers.SGD(learning_rate=learning_rate)
        elif name == "rmsprop":
            return keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif name == "adamax":
            return keras.optimizers.Adamax(learning_rate=learning_rate)
        else:
            return keras.optimizers.Adam(learning_rate=learning_rate)


class NASSearcher:
    """Neural Architecture Search engine."""
    
    def __init__(
        self,
        search_space: ArchitectureSpace,
        backend: str = "tensorflow",
        search_strategy: str = "random",
        n_trials: int = 20,
        time_budget: int = 3600
    ):
        """
        Initialize NAS searcher.
        
        Args:
            search_space: Architecture search space
            backend: Deep learning backend
            search_strategy: Search strategy ("random", "evolutionary", "bayesian")
            n_trials: Number of architectures to try
            time_budget: Time budget in seconds
        """
        self.search_space = search_space
        self.backend = backend
        self.search_strategy = search_strategy
        self.n_trials = n_trials
        self.time_budget = time_budget
        
        self.builder = NetworkBuilder(backend)
        self.evaluated_architectures = []
        self.best_architecture = None
        self.best_score = -np.inf
    
    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Search for best architecture.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Best architecture found
        """
        import time
        start_time = time.time()
        
        for trial in range(self.n_trials):
            if time.time() - start_time > self.time_budget:
                logger.info("Time budget exceeded")
                break
            
            # Sample architecture
            if self.search_strategy == "random":
                architecture = self.search_space.sample_architecture()
            elif self.search_strategy == "evolutionary":
                architecture = self._evolutionary_sample(trial)
            else:
                architecture = self.search_space.sample_architecture()
            
            # Add task type to architecture
            architecture["task_type"] = self.search_space.task_type
            
            # Evaluate architecture
            try:
                score = self._evaluate_architecture(
                    architecture,
                    X_train, y_train,
                    X_val, y_val
                )
                
                self.evaluated_architectures.append({
                    "architecture": architecture,
                    "score": score,
                    "trial": trial
                })
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_architecture = architecture
                    logger.info(f"New best architecture found: score={score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Architecture evaluation failed: {e}")
                continue
        
        return self.best_architecture
    
    def _evaluate_architecture(
        self,
        architecture: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Evaluate a single architecture."""
        if self.backend == "tensorflow":
            return self._evaluate_tensorflow(
                architecture,
                X_train, y_train,
                X_val, y_val
            )
        else:
            return self._evaluate_pytorch(
                architecture,
                X_train, y_train,
                X_val, y_val
            )
    
    def _evaluate_tensorflow(
        self,
        architecture: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Evaluate TensorFlow model."""
        if not HAS_TENSORFLOW:
            return 0.0
        
        # Build model
        model = self.builder.build_tensorflow_model(
            architecture,
            X_train.shape[1:]
        )
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=architecture.get("epochs", 20),
            batch_size=architecture.get("batch_size", 32),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        if architecture["task_type"] == "classification":
            _, accuracy = model.evaluate(X_val, y_val, verbose=0)
            return accuracy
        else:
            _, mae = model.evaluate(X_val, y_val, verbose=0)
            return -mae  # Negative MAE so higher is better
    
    def _evaluate_pytorch(
        self,
        architecture: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Evaluate PyTorch model."""
        if not HAS_PYTORCH:
            return 0.0
        
        # Build model
        model = self.builder.build_pytorch_model(
            architecture,
            X_train.shape[1:]
        )
        
        # Convert data to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        # Setup training
        criterion = nn.CrossEntropyLoss() if architecture["task_type"] == "classification" else nn.MSELoss()
        optimizer = self._get_pytorch_optimizer(
            model,
            architecture["optimizer"],
            architecture["learning_rate"]
        )
        
        # Train
        epochs = architecture.get("epochs", 20)
        batch_size = architecture.get("batch_size", 32)
        
        for epoch in range(epochs):
            model.train()
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_train_t[batch_indices]
                batch_y = y_train_t[batch_indices]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_t)
            
            if architecture["task_type"] == "classification":
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_val_t).sum().item() / len(y_val_t)
                return accuracy
            else:
                mse = criterion(outputs, y_val_t).item()
                return -mse  # Negative MSE so higher is better
    
    def _get_pytorch_optimizer(
        self,
        model: nn.Module,
        name: str,
        learning_rate: float
    ):
        """Get PyTorch optimizer."""
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif name == "adamax":
            return torch.optim.Adamax(model.parameters(), lr=learning_rate)
        else:
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def _evolutionary_sample(self, generation: int) -> Dict[str, Any]:
        """Sample using evolutionary strategy."""
        if generation < 5 or not self.evaluated_architectures:
            # Random for first generations
            return self.search_space.sample_architecture()
        
        # Select parent from top architectures
        sorted_archs = sorted(
            self.evaluated_architectures,
            key=lambda x: x["score"],
            reverse=True
        )
        
        parent = sorted_archs[0]["architecture"]
        
        # Mutate parent
        child = parent.copy()
        
        # Random mutation
        if np.random.random() < 0.5:
            # Mutate a layer
            if child["layers"]:
                idx = np.random.randint(len(child["layers"]))
                layer = child["layers"][idx]
                
                if layer["type"] == "dense":
                    layer["units"] = np.random.choice(self.search_space.layer_sizes)
                elif layer["type"] == "dropout":
                    layer["rate"] = np.random.choice(self.search_space.dropout_rates)
        
        if np.random.random() < 0.3:
            # Mutate optimizer settings
            child["learning_rate"] = np.random.choice(self.search_space.learning_rates)
        
        return child