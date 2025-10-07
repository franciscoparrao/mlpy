"""
Learners base para PyTorch.

Implementa la interfaz de MLPY para modelos PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
import logging
import warnings
from tqdm import tqdm

from ...learners.base import Learner
from ...tasks import Task, TaskClassif, TaskRegr
from ...predictions import PredictionClassif, PredictionRegr
from .datasets import MLPYDataset, create_data_loaders
from .callbacks import PyTorchCallback
from .utils import get_device

logger = logging.getLogger(__name__)


class LearnerPyTorch(Learner):
    """Learner base para modelos PyTorch.
    
    Parameters
    ----------
    model : nn.Module
        Modelo PyTorch.
    loss_fn : Optional[nn.Module]
        Función de pérdida. Si None, se selecciona automáticamente.
    optimizer_class : type
        Clase del optimizador (default: Adam).
    optimizer_params : Dict[str, Any]
        Parámetros para el optimizador.
    epochs : int
        Número de épocas de entrenamiento.
    batch_size : int
        Tamaño del batch.
    learning_rate : float
        Tasa de aprendizaje.
    device : Optional[str]
        Dispositivo ('cuda', 'cpu', 'mps' o None para auto).
    callbacks : List[PyTorchCallback]
        Callbacks para el entrenamiento.
    verbose : int
        Nivel de verbosidad (0=silencio, 1=progreso, 2=detalle).
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer_class: type = optim.Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        callbacks: Optional[List[PyTorchCallback]] = None,
        verbose: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params or {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = get_device(device)
        self.callbacks = callbacks or []
        self.verbose = verbose
        
        # Estado del entrenamiento
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_model_state = None
        self.current_epoch = 0
        
        # Mover modelo al dispositivo si ya está definido
        if self.model is not None:
            self.model = self.model.to(self.device)
    
    def _create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Crea el modelo si no se proporcionó uno.
        
        Parameters
        ----------
        input_dim : int
            Dimensión de entrada.
        output_dim : int
            Dimensión de salida.
            
        Returns
        -------
        nn.Module
            Modelo creado.
        """
        # Modelo MLP simple por defecto
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                hidden_dim = max(64, (input_dim + output_dim) // 2)
                
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return SimpleMLP(input_dim, output_dim)
    
    def _prepare_data(self, task: Task, validation_split: float = 0.2) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepara los dataloaders para entrenamiento.
        
        Parameters
        ----------
        task : Task
            Tarea de MLPY.
        validation_split : float
            Proporción de datos para validación.
            
        Returns
        -------
        Tuple[DataLoader, Optional[DataLoader]]
            DataLoaders de entrenamiento y validación.
        """
        train_loader, val_loader = create_data_loaders(
            task,
            batch_size=self.batch_size,
            validation_split=validation_split,
            shuffle=True
        )
        
        return train_loader, val_loader
    
    def train(self, task: Task, row_ids: Optional[List[int]] = None):
        """Entrena el modelo.
        
        Parameters
        ----------
        task : Task
            Tarea de entrenamiento.
        row_ids : Optional[List[int]]
            IDs de filas a usar.
        """
        # Filtrar datos si se especifican row_ids
        if row_ids is not None:
            task = task.filter(row_ids)
        
        # Obtener dimensiones
        X_sample = task.X()
        n_features = X_sample.shape[1]
        
        # Crear modelo si no existe
        if self.model is None:
            if isinstance(task, TaskClassif):
                n_classes = len(task.class_labels)
                self.model = self._create_model(n_features, n_classes)
            else:
                self.model = self._create_model(n_features, 1)
            
            self.model = self.model.to(self.device)
        
        # Configurar función de pérdida si no existe
        if self.loss_fn is None:
            if isinstance(task, TaskClassif):
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.MSELoss()
        
        # Configurar optimizador
        self.optimizer = self.optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
            **self.optimizer_params
        )
        
        # Preparar datos
        train_loader, val_loader = self._prepare_data(task)
        
        # Callbacks - on_train_begin
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        # Bucle de entrenamiento
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Callbacks - on_epoch_begin
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            
            # Entrenamiento
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validación
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
            
            # Log de progreso
            if self.verbose > 0:
                msg = f"Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.4f}"
                logger.info(msg)
                if self.verbose == 1:
                    print(msg)
            
            # Callbacks - on_epoch_end
            stop_training = False
            for callback in self.callbacks:
                if callback.on_epoch_end(self, epoch, {'train_loss': train_loss, 'val_loss': val_loss}):
                    stop_training = True
                    break
            
            if stop_training:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Callbacks - on_train_end
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        # Restaurar mejor modelo si existe
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model weights")
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Entrena una época.
        
        Parameters
        ----------
        train_loader : DataLoader
            DataLoader de entrenamiento.
            
        Returns
        -------
        float
            Pérdida promedio de la época.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Usar tqdm si verbose > 1
        iterator = tqdm(train_loader, desc="Training") if self.verbose > 1 else train_loader
        
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            # Ajustar forma de salida si es necesario
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            loss = self.loss_fn(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if self.verbose > 1 and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': loss.item()})
        
        return total_loss / max(n_batches, 1)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Valida una época.
        
        Parameters
        ----------
        val_loader : DataLoader
            DataLoader de validación.
            
        Returns
        -------
        float
            Pérdida promedio de validación.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> Union[PredictionClassif, PredictionRegr]:
        """Realiza predicciones.
        
        Parameters
        ----------
        task : Task
            Tarea de predicción.
        row_ids : Optional[List[int]]
            IDs de filas a predecir.
            
        Returns
        -------
        Union[PredictionClassif, PredictionRegr]
            Predicciones.
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Filtrar datos si es necesario
        if row_ids is not None:
            task = task.filter(row_ids)
        else:
            row_ids = list(range(len(task.data)))
        
        # Crear dataset
        dataset = MLPYDataset(task, training=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Realizar predicciones
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_x in loader:
                if isinstance(batch_x, tuple):
                    batch_x = batch_x[0]
                
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                if isinstance(task, TaskClassif):
                    # Clasificación
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_predictions.append(preds.cpu().numpy())
                    all_probabilities.append(probs.cpu().numpy())
                else:
                    # Regresión
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze(1)
                    all_predictions.append(outputs.cpu().numpy())
        
        # Concatenar resultados
        predictions = np.concatenate(all_predictions)
        
        if isinstance(task, TaskClassif):
            probabilities = np.concatenate(all_probabilities)
            
            # Obtener valores verdaderos si están disponibles
            truth = None
            if task.has_target():
                truth = task.truth(row_ids)
            
            return PredictionClassif(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                truth=truth,
                response=predictions,
                prob=probabilities
            )
        else:
            truth = None
            if task.has_target():
                truth = task.truth(row_ids)
            
            return PredictionRegr(
                task=task,
                learner_id=self.id,
                row_ids=row_ids,
                truth=truth,
                response=predictions
            )
    
    def save_model(self, path: Union[str, Path]):
        """Guarda el modelo.
        
        Parameters
        ----------
        path : Union[str, Path]
            Ruta donde guardar el modelo.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
            'current_epoch': self.current_epoch,
            'hyperparameters': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Carga el modelo.
        
        Parameters
        ----------
        path : Union[str, Path]
            Ruta del modelo guardado.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is None:
            raise ValueError("Model architecture must be defined before loading weights")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.history = checkpoint.get('history', {})
        self.current_epoch = checkpoint.get('current_epoch', 0)
        
        logger.info(f"Model loaded from {path}")


class LearnerPyTorchClassif(LearnerPyTorch):
    """Learner PyTorch para clasificación."""
    
    @property
    def task_type(self) -> str:
        return "classification"
    
    def __init__(self, **kwargs):
        """Inicializa learner de clasificación.
        
        Parameters
        ----------
        **kwargs
            Argumentos para LearnerPyTorch.
        """
        super().__init__(**kwargs)
        
        # Configurar pérdida por defecto para clasificación
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()


class LearnerPyTorchRegr(LearnerPyTorch):
    """Learner PyTorch para regresión."""
    
    @property
    def task_type(self) -> str:
        return "regression"
    
    def __init__(self, **kwargs):
        """Inicializa learner de regresión.
        
        Parameters
        ----------
        **kwargs
            Argumentos para LearnerPyTorch.
        """
        super().__init__(**kwargs)
        
        # Configurar pérdida por defecto para regresión
        if self.loss_fn is None:
            self.loss_fn = nn.MSELoss()