"""
Sistema de callbacks para PyTorch en MLPY.

Callbacks para monitoreo y control del entrenamiento.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
import numpy as np
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PyTorchCallback:
    """Callback base para PyTorch.
    
    Los callbacks permiten ejecutar código en momentos específicos
    del entrenamiento sin modificar el loop principal.
    """
    
    def on_train_begin(self, learner):
        """Llamado al inicio del entrenamiento.
        
        Parameters
        ----------
        learner : LearnerPyTorch
            Learner que está entrenando.
        """
        pass
    
    def on_train_end(self, learner):
        """Llamado al final del entrenamiento.
        
        Parameters
        ----------
        learner : LearnerPyTorch
            Learner que está entrenando.
        """
        pass
    
    def on_epoch_begin(self, learner, epoch: int):
        """Llamado al inicio de cada época.
        
        Parameters
        ----------
        learner : LearnerPyTorch
            Learner que está entrenando.
        epoch : int
            Número de época actual.
        """
        pass
    
    def on_epoch_end(self, learner, epoch: int, logs: Dict[str, float]) -> bool:
        """Llamado al final de cada época.
        
        Parameters
        ----------
        learner : LearnerPyTorch
            Learner que está entrenando.
        epoch : int
            Número de época actual.
        logs : Dict[str, float]
            Métricas de la época.
            
        Returns
        -------
        bool
            True si se debe detener el entrenamiento.
        """
        return False
    
    def on_batch_begin(self, learner, batch: int):
        """Llamado al inicio de cada batch.
        
        Parameters
        ----------
        learner : LearnerPyTorch
            Learner que está entrenando.
        batch : int
            Número de batch actual.
        """
        pass
    
    def on_batch_end(self, learner, batch: int, logs: Dict[str, float]):
        """Llamado al final de cada batch.
        
        Parameters
        ----------
        learner : LearnerPyTorch
            Learner que está entrenando.
        batch : int
            Número de batch actual.
        logs : Dict[str, float]
            Métricas del batch.
        """
        pass


class EarlyStopping(PyTorchCallback):
    """Early stopping para evitar overfitting.
    
    Parameters
    ----------
    monitor : str
        Métrica a monitorear (e.g., 'val_loss').
    patience : int
        Épocas sin mejora antes de parar.
    min_delta : float
        Cambio mínimo para considerar mejora.
    mode : str
        'min' para minimizar, 'max' para maximizar.
    restore_best : bool
        Si restaurar los mejores pesos al final.
    verbose : bool
        Si imprimir mensajes.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best: bool = True,
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose
        
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.best_value = None
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def on_train_begin(self, learner):
        """Inicializa el early stopping."""
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.best_value = np.Inf if self.mode == 'min' else -np.Inf
        self.best_weights = None
    
    def on_epoch_end(self, learner, epoch: int, logs: Dict[str, float]) -> bool:
        """Verifica si debe parar el entrenamiento."""
        current = logs.get(self.monitor)
        
        if current is None:
            logger.warning(f"Early stopping: metric '{self.monitor}' not found in logs")
            return False
        
        # Verificar si hay mejora
        if self.monitor_op(current - self.min_delta, self.best_value):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Guardar mejores pesos
            if self.restore_best:
                self.best_weights = learner.model.state_dict().copy()
            
            if self.verbose:
                logger.info(f"Early stopping: improvement in {self.monitor} to {current:.6f}")
        else:
            self.wait += 1
            if self.verbose and self.wait > 0:
                logger.info(f"Early stopping: no improvement for {self.wait} epochs")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                
                # Restaurar mejores pesos
                if self.restore_best and self.best_weights is not None:
                    learner.model.load_state_dict(self.best_weights)
                    learner.best_model_state = self.best_weights
                    if self.verbose:
                        logger.info(f"Restored best weights from epoch {self.best_epoch}")
                
                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                
                return True  # Detener entrenamiento
        
        return False


class ModelCheckpoint(PyTorchCallback):
    """Guarda el modelo durante el entrenamiento.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Ruta donde guardar el modelo.
    monitor : str
        Métrica a monitorear.
    save_best_only : bool
        Si guardar solo el mejor modelo.
    mode : str
        'min' para minimizar, 'max' para maximizar.
    save_freq : int
        Frecuencia de guardado en épocas.
    verbose : bool
        Si imprimir mensajes.
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        save_freq: int = 1,
        verbose: bool = True
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.verbose = verbose
        
        self.best_value = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
        
        # Crear directorio si no existe
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, learner):
        """Inicializa el checkpoint."""
        self.best_value = np.Inf if self.mode == 'min' else -np.Inf
    
    def on_epoch_end(self, learner, epoch: int, logs: Dict[str, float]) -> bool:
        """Guarda el modelo si corresponde."""
        # Verificar frecuencia
        if (epoch + 1) % self.save_freq != 0:
            return False
        
        # Si save_best_only, verificar si hay mejora
        if self.save_best_only:
            current = logs.get(self.monitor)
            
            if current is None:
                logger.warning(f"ModelCheckpoint: metric '{self.monitor}' not found")
                return False
            
            if not self.monitor_op(current, self.best_value):
                return False
            
            self.best_value = current
        
        # Preparar filepath con formato
        filepath = str(self.filepath)
        filepath = filepath.format(epoch=epoch, **logs)
        filepath = Path(filepath)
        
        # Guardar checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': learner.model.state_dict(),
            'optimizer_state_dict': learner.optimizer.state_dict() if learner.optimizer else None,
            'metrics': logs,
            'best_value': self.best_value if self.save_best_only else None
        }
        
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            if self.save_best_only:
                logger.info(f"Model checkpoint: saved best model to {filepath} ({self.monitor}={self.best_value:.6f})")
            else:
                logger.info(f"Model checkpoint: saved to {filepath}")
        
        return False


class LearningRateScheduler(PyTorchCallback):
    """Ajusta el learning rate durante el entrenamiento.
    
    Parameters
    ----------
    schedule : Union[Callable, str]
        Función o nombre de estrategia ('step', 'exponential', 'cosine').
    step_size : int
        Tamaño del paso para 'step'.
    gamma : float
        Factor de decaimiento.
    min_lr : float
        Learning rate mínimo.
    verbose : bool
        Si imprimir mensajes.
    """
    
    def __init__(
        self,
        schedule: Union[Callable, str] = 'step',
        step_size: int = 10,
        gamma: float = 0.1,
        min_lr: float = 1e-7,
        verbose: bool = True
    ):
        self.schedule = schedule
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.initial_lr = None
    
    def on_train_begin(self, learner):
        """Guarda el learning rate inicial."""
        self.initial_lr = learner.learning_rate
    
    def on_epoch_end(self, learner, epoch: int, logs: Dict[str, float]) -> bool:
        """Ajusta el learning rate."""
        if isinstance(self.schedule, Callable):
            # Función personalizada
            new_lr = self.schedule(epoch)
        elif self.schedule == 'step':
            # Step decay
            new_lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
        elif self.schedule == 'exponential':
            # Exponential decay
            new_lr = self.initial_lr * (self.gamma ** epoch)
        elif self.schedule == 'cosine':
            # Cosine annealing
            new_lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                     (1 + np.cos(np.pi * epoch / learner.epochs)) / 2
        else:
            return False
        
        # Aplicar límite mínimo
        new_lr = max(new_lr, self.min_lr)
        
        # Actualizar learning rate
        for param_group in learner.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if self.verbose:
            logger.info(f"Learning rate adjusted to {new_lr:.2e}")
        
        return False


class TensorBoardLogger(PyTorchCallback):
    """Logging para TensorBoard.
    
    Parameters
    ----------
    log_dir : Union[str, Path]
        Directorio para logs de TensorBoard.
    comment : str
        Comentario para el run.
    log_graph : bool
        Si loggear el grafo del modelo.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path] = './runs',
        comment: str = '',
        log_graph: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.comment = comment
        self.log_graph = log_graph
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_available = True
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.tensorboard_available = False
    
    def on_train_begin(self, learner):
        """Inicializa el writer de TensorBoard."""
        if not self.tensorboard_available:
            return
        
        from torch.utils.tensorboard import SummaryWriter
        
        # Crear nombre único para el run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{self.comment}" if self.comment else timestamp
        log_dir = self.log_dir / run_name
        
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logging to {log_dir}")
        
        # Loggear grafo del modelo si está disponible
        if self.log_graph and learner.model is not None:
            try:
                # Crear input dummy
                dummy_input = torch.randn(1, learner.model.layers[0].in_features)
                self.writer.add_graph(learner.model, dummy_input)
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")
    
    def on_epoch_end(self, learner, epoch: int, logs: Dict[str, float]) -> bool:
        """Loggea métricas de la época."""
        if not self.tensorboard_available or self.writer is None:
            return False
        
        # Loggear métricas
        for key, value in logs.items():
            if value is not None:
                self.writer.add_scalar(key, value, epoch)
        
        # Loggear learning rate
        if learner.optimizer is not None:
            lr = learner.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('learning_rate', lr, epoch)
        
        # Loggear histogramas de pesos
        if epoch % 10 == 0:  # Cada 10 épocas para no sobrecargar
            for name, param in learner.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"weights/{name}", param, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)
        
        return False
    
    def on_train_end(self, learner):
        """Cierra el writer de TensorBoard."""
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")


class ProgressBar(PyTorchCallback):
    """Muestra una barra de progreso durante el entrenamiento.
    
    Parameters
    ----------
    show_metrics : bool
        Si mostrar métricas en la barra.
    leave : bool
        Si dejar la barra al terminar.
    """
    
    def __init__(self, show_metrics: bool = True, leave: bool = True):
        self.show_metrics = show_metrics
        self.leave = leave
        self.pbar = None
        
        try:
            from tqdm import tqdm
            self.tqdm_available = True
        except ImportError:
            self.tqdm_available = False
    
    def on_train_begin(self, learner):
        """Crea la barra de progreso."""
        if not self.tqdm_available:
            return
        
        from tqdm import tqdm
        self.pbar = tqdm(total=learner.epochs, desc="Training", leave=self.leave)
    
    def on_epoch_end(self, learner, epoch: int, logs: Dict[str, float]) -> bool:
        """Actualiza la barra de progreso."""
        if self.pbar is None:
            return False
        
        # Actualizar descripción con métricas
        if self.show_metrics and logs:
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items() if v is not None])
            self.pbar.set_description(f"Epoch {epoch+1} - {metrics_str}")
        
        self.pbar.update(1)
        return False
    
    def on_train_end(self, learner):
        """Cierra la barra de progreso."""
        if self.pbar is not None:
            self.pbar.close()


class GradientClipping(PyTorchCallback):
    """Aplica gradient clipping durante el entrenamiento.
    
    Parameters
    ----------
    max_norm : float
        Norma máxima para los gradientes.
    verbose : bool
        Si imprimir mensajes cuando se aplica clipping.
    """
    
    def __init__(self, max_norm: float = 1.0, verbose: bool = False):
        self.max_norm = max_norm
        self.verbose = verbose
    
    def on_batch_end(self, learner, batch: int, logs: Dict[str, float]):
        """Aplica gradient clipping después de cada batch."""
        if learner.model is None:
            return
        
        # Aplicar clipping
        total_norm = nn.utils.clip_grad_norm_(learner.model.parameters(), self.max_norm)
        
        if self.verbose and total_norm > self.max_norm:
            logger.info(f"Gradient clipping: reduced norm from {total_norm:.4f} to {self.max_norm}")


class MetricsLogger(PyTorchCallback):
    """Guarda las métricas en un archivo JSON.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Ruta del archivo de métricas.
    append : bool
        Si agregar a archivo existente.
    """
    
    def __init__(self, filepath: Union[str, Path], append: bool = False):
        self.filepath = Path(filepath)
        self.append = append
        self.metrics_history = []
    
    def on_train_begin(self, learner):
        """Inicializa el logger."""
        if self.append and self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.metrics_history = json.load(f)
        else:
            self.metrics_history = []
        
        # Crear directorio si no existe
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, learner, epoch: int, logs: Dict[str, float]) -> bool:
        """Guarda las métricas de la época."""
        metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **logs
        }
        
        self.metrics_history.append(metrics)
        
        # Guardar a archivo
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        return False