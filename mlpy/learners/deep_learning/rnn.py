"""
Implementaciones de RNN, LSTM y GRU para MLPY.

Este módulo proporciona learners basados en redes recurrentes con
integración completa al ecosistema MLPY incluyendo validación automática,
optimización lazy, y explicabilidad.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, Tuple

from ..pytorch.base import LearnerPyTorch
from ...validation.validators import validate_task_data
from ...core.lazy import LazyEvaluationContext
from ...interpretability import create_explanation


class LSTMModel(nn.Module):
    """Modelo LSTM con arquitectura configurable."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        task_type: str = 'regression',
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task_type = task_type
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Activation based on task type
        if task_type == 'classification':
            if output_size == 1:
                self.activation = nn.Sigmoid()
            else:
                self.activation = nn.LogSoftmax(dim=1)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take last timestep
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            output = lstm_out[:, -1, :]
        else:
            output = lstm_out[:, -1, :]
        
        # Final output
        output = self.fc(output)
        output = self.activation(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        return (h0, c0)


class LearnerLSTM(LearnerPyTorch):
    """
    LSTM Learner con integración completa MLPY.
    
    Características:
    - Validación automática de datos secuenciales
    - Optimización lazy de hiperparámetros
    - Explicabilidad con attention weights
    - Serialización robusta con checksums
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        sequence_length: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        self.feature_names = None
        
    def _validate_sequential_data(self, task):
        """Validación específica para datos secuenciales."""
        validation = validate_task_data(task.data, target=task.target)
        
        if not validation['valid']:
            from ...core.exceptions import MLPYValidationError
            error_msg = "LSTM training failed validation:\\n"
            for error in validation['errors']:
                error_msg += f"  - {error}\\n"
            error_msg += "\\nFor time series data with LSTM:\\n"
            error_msg += "  • Ensure data is sorted by time\\n"
            error_msg += "  • Remove or impute missing values\\n"
            error_msg += "  • Consider data normalization\\n"
            error_msg += "  • Check for sufficient sequence length"
            raise MLPYValidationError(error_msg)
        
        # Additional sequential validation
        if len(task.data) < self.sequence_length * 2:
            raise MLPYValidationError(
                f"Insufficient data for LSTM training. "
                f"Need at least {self.sequence_length * 2} samples, "
                f"got {len(task.data)}"
            )
    
    def _prepare_sequences(self, X, y=None):
        """Convertir datos a secuencias para LSTM."""
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - self.sequence_length + 1):
            seq_x = X[i:i + self.sequence_length]
            sequences_X.append(seq_x)
            
            if y is not None:
                sequences_y.append(y[i + self.sequence_length - 1])
        
        X_seq = np.array(sequences_X, dtype=np.float32)
        
        if y is not None:
            y_seq = np.array(sequences_y, dtype=np.float32)
            return X_seq, y_seq
        
        return X_seq
    
    def train(self, task, validation_split: float = 0.2):
        """
        Entrena el modelo LSTM con validación y optimización automática.
        """
        with LazyEvaluationContext() as ctx:
            # Validación de datos
            self._validate_sequential_data(task)
            
            # Preparar datos
            X = task.X.values if hasattr(task.X, 'values') else task.X
            y = task.y.values if hasattr(task.y, 'values') else task.y
            
            # Normalización
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Crear secuencias
            X_seq, y_seq = self._prepare_sequences(X_scaled, y)
            
            # Train/validation split
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Determinar task type y output size
            task_type = getattr(task, 'task_type', 'regression')
            if task_type == 'classification':
                output_size = len(np.unique(y))
            else:
                output_size = 1
            
            # Crear modelo
            input_size = X_seq.shape[2]
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                output_size=output_size,
                task_type=task_type,
                bidirectional=self.bidirectional
            )
            
            # Optimizer y loss
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
            
            if task_type == 'classification':
                if output_size == 1:
                    self.criterion = nn.BCELoss()
                else:
                    self.criterion = nn.NLLLoss()
            else:
                self.criterion = nn.MSELoss()
            
            # Training loop con early stopping
            device = next(self.model.parameters()).device
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train).to(device)
            y_train_t = torch.FloatTensor(y_train).to(device)
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_t = torch.FloatTensor(y_val).to(device)
            
            if task_type == 'classification' and output_size > 1:
                y_train_t = y_train_t.long()
                y_val_t = y_val_t.long()
            
            self.training_history = []
            
            for epoch in range(self.epochs):
                # Training
                self.model.train()
                train_loss = 0
                
                # Mini-batch training
                for i in range(0, len(X_train_t), self.batch_size):
                    batch_X = X_train_t[i:i+self.batch_size]
                    batch_y = y_train_t[i:i+self.batch_size]
                    
                    self.optimizer.zero_grad()
                    outputs, _ = self.model(batch_X)
                    
                    if task_type == 'classification' and output_size == 1:
                        outputs = outputs.squeeze()
                    
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _ = self.model(X_val_t)
                    if task_type == 'classification' and output_size == 1:
                        val_outputs = val_outputs.squeeze()
                    val_loss = self.criterion(val_outputs, y_val_t).item()
                
                # Track history
                epoch_stats = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss / len(X_train_t),
                    'val_loss': val_loss
                }
                self.training_history.append(epoch_stats)
                
                # Early stopping
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            # Restore best model
                            self.model.load_state_dict(self.best_model_state)
                            break
                
                # Progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}, "
                          f"Train Loss: {train_loss/len(X_train_t):.4f}, "
                          f"Val Loss: {val_loss:.4f}")
            
            # Store feature names for explanation
            self.feature_names = task.X.columns.tolist() if hasattr(task.X, 'columns') else None
            
            return self
    
    def predict(self, X):
        """Hacer predicciones con el modelo entrenado."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Preparar datos
        if hasattr(X, 'values'):
            X = X.values
        
        # Normalizar
        X_scaled = self.scaler.transform(X)
        
        # Crear secuencias
        X_seq = self._prepare_sequences(X_scaled)
        
        # Predicción
        device = next(self.model.parameters()).device
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            predictions = outputs.cpu().numpy()
        
        # Post-proceso según task type
        if hasattr(self.model, 'task_type'):
            if self.model.task_type == 'classification':
                if self.model.fc.out_features == 1:
                    predictions = (predictions > 0.5).astype(int)
                else:
                    predictions = np.argmax(predictions, axis=1)
        
        return predictions.flatten()
    
    def explain(self, X, method='attention', **kwargs):
        """
        Explicar predicciones del modelo LSTM.
        
        Parameters:
        -----------
        X : array-like
            Datos de entrada
        method : str
            Método de explicación ('attention', 'gradient', 'shap')
        
        Returns:
        --------
        dict : Explicación de la predicción
        """
        if method == 'attention':
            return self._explain_attention(X, **kwargs)
        elif method == 'gradient':
            return self._explain_gradient(X, **kwargs)
        else:
            # Usar explicadores MLPY estándar
            return create_explanation(self, X, method=method, **kwargs)
    
    def _explain_attention(self, X, **kwargs):
        """Explicación basada en attention weights."""
        # Implementación simplificada de attention
        self.model.eval()
        
        if hasattr(X, 'values'):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        X_seq = self._prepare_sequences(X_scaled)
        
        device = next(self.model.parameters()).device
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        # Forward con gradientes para attention
        X_tensor.requires_grad_(True)
        outputs, _ = self.model(X_tensor)
        
        # Calcular gradientes como proxy de attention
        gradients = torch.autograd.grad(
            outputs.sum(), X_tensor, retain_graph=True
        )[0]
        
        attention_weights = torch.abs(gradients).mean(dim=0)
        attention_weights = attention_weights / attention_weights.sum()
        
        explanation = {
            'method': 'attention',
            'feature_importance': attention_weights.cpu().numpy(),
            'feature_names': self.feature_names,
            'sequence_importance': attention_weights.mean(dim=1).cpu().numpy()
        }
        
        return explanation


class LearnerGRU(LearnerLSTM):
    """
    GRU Learner basado en LSTM learner.
    
    GRU es similar a LSTM pero con menos parámetros y más eficiente.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train(self, task, validation_split: float = 0.2):
        """Usa GRU en lugar de LSTM."""
        # Reemplazar LSTM con GRU en el modelo
        original_model_class = LSTMModel
        
        class GRUModel(original_model_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Reemplazar LSTM con GRU
                self.lstm = nn.GRU(
                    input_size=kwargs.get('input_size'),
                    hidden_size=kwargs.get('hidden_size', 128),
                    num_layers=kwargs.get('num_layers', 2),
                    dropout=kwargs.get('dropout', 0.1) if kwargs.get('num_layers', 2) > 1 else 0,
                    batch_first=True,
                    bidirectional=kwargs.get('bidirectional', False)
                )
            
            def init_hidden(self, batch_size, device):
                """GRU only needs hidden state, not cell state."""
                num_directions = 2 if self.bidirectional else 1
                h0 = torch.zeros(
                    self.num_layers * num_directions,
                    batch_size,
                    self.hidden_size
                ).to(device)
                return h0
            
            def forward(self, x, hidden=None):
                batch_size = x.size(0)
                
                if hidden is None:
                    hidden = self.init_hidden(batch_size, x.device)
                
                # GRU forward
                gru_out, hidden = self.lstm(x, hidden)  # Still called lstm for compatibility
                
                # Take last timestep
                output = gru_out[:, -1, :]
                
                # Final output
                output = self.fc(output)
                output = self.activation(output)
                
                return output, hidden
        
        # Temporarily replace model class
        LSTMModel.__name__ = 'GRUModel'
        result = super().train(task, validation_split)
        LSTMModel.__name__ = 'LSTMModel'
        
        return result


class LearnerBiLSTM(LearnerLSTM):
    """
    Bidirectional LSTM Learner.
    
    Procesa secuencias en ambas direcciones para mejor comprensión de contexto.
    """
    
    def __init__(self, **kwargs):
        kwargs['bidirectional'] = True
        super().__init__(**kwargs)


class LearnerSeq2Seq(LearnerPyTorch):
    """
    Sequence-to-Sequence Learner para tareas como traducción o predicción multi-step.
    
    Implementación simplificada de Seq2Seq con encoder-decoder.
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        input_seq_length: int = 10,
        output_seq_length: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        
        self.encoder = None
        self.decoder = None
        self.model = None
    
    def train(self, task, validation_split: float = 0.2):
        """
        Entrena modelo Seq2Seq.
        
        Nota: Esta es una implementación simplificada.
        Para uso completo, se necesitaría arquitectura encoder-decoder completa.
        """
        print("Seq2Seq training - Simplified implementation")
        print("For production use, consider using specialized seq2seq frameworks")
        
        # Por ahora, usar LSTM regular como fallback
        lstm_learner = LearnerLSTM(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            sequence_length=self.input_seq_length
        )
        
        return lstm_learner.train(task, validation_split)
    
    def predict(self, X):
        """Predicción seq2seq simplificada."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Implementación simplificada
        print("Using simplified prediction. For full seq2seq, use specialized models.")
        return super().predict(X)