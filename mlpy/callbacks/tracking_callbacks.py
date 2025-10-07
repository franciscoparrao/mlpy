"""
Callbacks para integración con tracking de experimentos.
"""

from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path

from .base import Callback
from ..tracking import ExperimentTracker, create_tracker, get_tracker

logger = logging.getLogger(__name__)


class TrackerCallback(Callback):
    """Callback para tracking de experimentos con MLflow/WandB.
    
    Parameters
    ----------
    tracker : Union[str, ExperimentTracker]
        Tracker o nombre del tracker a usar.
    log_params : bool
        Si loggear parámetros del learner.
    log_metrics : bool
        Si loggear métricas durante el entrenamiento.
    log_model : bool
        Si loggear el modelo al final.
    log_artifacts : bool
        Si loggear artefactos adicionales.
    log_frequency : int
        Frecuencia de logging de métricas (en épocas/iteraciones).
    run_name : Optional[str]
        Nombre del run.
    tags : Optional[Dict[str, str]]
        Tags para el run.
    """
    
    def __init__(
        self,
        tracker: Union[str, ExperimentTracker],
        log_params: bool = True,
        log_metrics: bool = True,
        log_model: bool = True,
        log_artifacts: bool = True,
        log_frequency: int = 1,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        
        # Obtener tracker
        if isinstance(tracker, str):
            self.tracker = get_tracker(tracker)
            if self.tracker is None:
                raise ValueError(f"Tracker '{tracker}' not found")
        else:
            self.tracker = tracker
        
        self.log_params = log_params
        self.log_metrics = log_metrics
        self.log_model = log_model
        self.log_artifacts = log_artifacts
        self.log_frequency = log_frequency
        self.run_name = run_name
        self.tags = tags
        
        self.iteration = 0
    
    def on_train_begin(self, learner, **kwargs):
        """Inicia run de tracking."""
        if not self.tracker.is_active:
            self.tracker.start_run(
                run_name=self.run_name,
                tags=self.tags
            )
        
        if self.log_params:
            self._log_learner_params(learner)
    
    def on_train_end(self, learner, **kwargs):
        """Finaliza run y loggea modelo."""
        if self.log_model and hasattr(learner, 'model'):
            try:
                self.tracker.log_model(
                    learner.model,
                    model_name=f"{learner.__class__.__name__}_model",
                    metadata={
                        'learner_type': learner.__class__.__name__,
                        'learner_id': getattr(learner, 'id', 'unknown')
                    }
                )
            except Exception as e:
                logger.error(f"Error logging model: {e}")
        
        # No cerrar el run aquí por si hay más callbacks
    
    def on_iteration_end(self, learner, iteration: int, metrics: Dict[str, float], **kwargs):
        """Loggea métricas de iteración."""
        self.iteration = iteration
        
        if self.log_metrics and iteration % self.log_frequency == 0:
            self.tracker.log_metrics(metrics, step=iteration)
    
    def on_epoch_end(self, learner, epoch: int, metrics: Dict[str, float], **kwargs):
        """Loggea métricas de época."""
        if self.log_metrics and epoch % self.log_frequency == 0:
            # Añadir prefijo para distinguir de métricas de iteración
            epoch_metrics = {f"epoch_{k}": v for k, v in metrics.items()}
            epoch_metrics['epoch'] = epoch
            
            self.tracker.log_metrics(epoch_metrics, step=epoch)
    
    def on_evaluation_end(self, learner, metrics: Dict[str, float], **kwargs):
        """Loggea métricas de evaluación."""
        if self.log_metrics:
            # Añadir prefijo para métricas de evaluación
            eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
            self.tracker.log_metrics(eval_metrics)
    
    def _log_learner_params(self, learner):
        """Loggea parámetros del learner."""
        params = {}
        
        # Parámetros básicos
        params['learner_type'] = learner.__class__.__name__
        params['learner_id'] = getattr(learner, 'id', 'unknown')
        
        # Parámetros específicos según tipo de learner
        param_attrs = [
            'learning_rate', 'epochs', 'batch_size', 'optimizer',
            'loss', 'regularization', 'dropout', 'activation',
            'n_estimators', 'max_depth', 'min_samples_split',
            'C', 'kernel', 'gamma', 'alpha', 'penalty',
            'n_neighbors', 'weights', 'metric'
        ]
        
        for attr in param_attrs:
            if hasattr(learner, attr):
                value = getattr(learner, attr)
                if value is not None:
                    params[attr] = value
        
        if params:
            self.tracker.log_params(params)


class MLFlowCallback(TrackerCallback):
    """Callback específico para MLflow.
    
    Parameters
    ----------
    experiment_name : str
        Nombre del experimento.
    tracking_uri : Optional[str]
        URI del servidor MLflow.
    **kwargs
        Argumentos adicionales para TrackerCallback.
    """
    
    def __init__(
        self,
        experiment_name: str = "mlpy-experiment",
        tracking_uri: Optional[str] = None,
        **kwargs
    ):
        # Crear tracker MLflow
        from ..tracking import create_tracker, register_tracker
        
        tracker = create_tracker(
            'mlflow',
            experiment_name=experiment_name,
            tracking_uri=tracking_uri or "file:./mlruns"
        )
        
        # Registrar globalmente
        register_tracker(f"mlflow_{experiment_name}", tracker)
        
        super().__init__(tracker=tracker, **kwargs)


class WandBCallback(TrackerCallback):
    """Callback específico para Weights & Biases.
    
    Parameters
    ----------
    project : str
        Nombre del proyecto en WandB.
    entity : Optional[str]
        Entidad/equipo en WandB.
    group : Optional[str]
        Grupo de runs.
    **kwargs
        Argumentos adicionales para TrackerCallback.
    """
    
    def __init__(
        self,
        project: str = "mlpy-experiments",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        **kwargs
    ):
        # Crear tracker WandB
        from ..tracking import create_tracker, register_tracker
        
        tracker = create_tracker(
            'wandb',
            project=project,
            entity=entity,
            group=group
        )
        
        # Registrar globalmente
        register_tracker(f"wandb_{project}", tracker)
        
        super().__init__(tracker=tracker, **kwargs)
    
    def on_train_begin(self, learner, **kwargs):
        """Además del inicio normal, configura watch para modelos PyTorch."""
        super().on_train_begin(learner, **kwargs)
        
        # Si es un modelo PyTorch, usar watch
        if hasattr(learner, 'model'):
            model = getattr(learner, 'model')
            if 'torch' in str(type(model)):
                self.tracker.watch(model)


class CompareRunsCallback(Callback):
    """Callback para comparar múltiples runs al final.
    
    Parameters
    ----------
    tracker : Union[str, ExperimentTracker]
        Tracker a usar.
    compare_metrics : List[str]
        Métricas a comparar.
    save_comparison : bool
        Si guardar la comparación como artefacto.
    """
    
    def __init__(
        self,
        tracker: Union[str, ExperimentTracker],
        compare_metrics: Optional[List[str]] = None,
        save_comparison: bool = True
    ):
        super().__init__()
        
        if isinstance(tracker, str):
            self.tracker = get_tracker(tracker)
        else:
            self.tracker = tracker
        
        self.compare_metrics = compare_metrics
        self.save_comparison = save_comparison
        self.run_ids = []
    
    def on_train_begin(self, learner, **kwargs):
        """Registra el run actual."""
        if self.tracker.is_active and self.tracker.current_run:
            self.run_ids.append(self.tracker.current_run.run_id)
    
    def on_train_end(self, learner, **kwargs):
        """Compara runs al final."""
        if len(self.run_ids) > 1:
            comparison = self.tracker.compare_runs(
                self.run_ids,
                metrics=self.compare_metrics
            )
            
            if self.save_comparison and comparison:
                # Guardar comparación como artefacto
                import json
                import tempfile
                
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='_comparison.json',
                    delete=False
                ) as f:
                    json.dump(comparison, f, indent=2, default=str)
                    self.tracker.log_artifact(f.name, artifact_type='comparison')
                
                logger.info(f"Saved comparison of {len(self.run_ids)} runs")


class AutoLogCallback(Callback):
    """Callback para auto-logging de métricas y artefactos.
    
    Detecta automáticamente qué loggear basándose en el contexto.
    
    Parameters
    ----------
    tracker : Union[str, ExperimentTracker]
        Tracker a usar.
    log_gradients : bool
        Si loggear gradientes (para redes neuronales).
    log_weights : bool
        Si loggear pesos del modelo.
    log_plots : bool
        Si loggear plots generados.
    log_confusion_matrix : bool
        Si loggear matriz de confusión.
    log_feature_importance : bool
        Si loggear importancia de features.
    """
    
    def __init__(
        self,
        tracker: Union[str, ExperimentTracker],
        log_gradients: bool = False,
        log_weights: bool = False,
        log_plots: bool = True,
        log_confusion_matrix: bool = True,
        log_feature_importance: bool = True
    ):
        super().__init__()
        
        if isinstance(tracker, str):
            self.tracker = get_tracker(tracker)
        else:
            self.tracker = tracker
        
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        self.log_plots = log_plots
        self.log_confusion_matrix = log_confusion_matrix
        self.log_feature_importance = log_feature_importance
    
    def on_train_begin(self, learner, **kwargs):
        """Auto-detecta y loggea información inicial."""
        if not self.tracker.is_active:
            self.tracker.start_run()
        
        # Auto-detectar y loggear info del learner
        self._auto_log_learner_info(learner)
    
    def on_iteration_end(self, learner, iteration: int, metrics: Dict[str, float], **kwargs):
        """Auto-loggea durante iteraciones."""
        # Loggear gradientes si está habilitado y disponible
        if self.log_gradients and hasattr(learner, 'model'):
            self._log_gradients(learner.model, iteration)
        
        # Loggear pesos si está habilitado
        if self.log_weights and iteration % 10 == 0:  # Cada 10 iteraciones
            self._log_weights(learner.model, iteration)
    
    def on_evaluation_end(self, learner, metrics: Dict[str, float], predictions=None, **kwargs):
        """Auto-loggea resultados de evaluación."""
        # Loggear matriz de confusión si es clasificación
        if self.log_confusion_matrix and predictions is not None:
            self._log_confusion_matrix(predictions)
        
        # Loggear importancia de features si está disponible
        if self.log_feature_importance:
            self._log_feature_importance(learner)
    
    def _auto_log_learner_info(self, learner):
        """Auto-detecta y loggea información del learner."""
        info = {
            'learner_class': learner.__class__.__name__,
            'learner_module': learner.__class__.__module__
        }
        
        # Detectar framework
        if 'sklearn' in info['learner_module']:
            info['framework'] = 'scikit-learn'
        elif 'torch' in info['learner_module']:
            info['framework'] = 'pytorch'
        elif 'tensorflow' in info['learner_module']:
            info['framework'] = 'tensorflow'
        else:
            info['framework'] = 'custom'
        
        # Detectar tipo de tarea
        if hasattr(learner, 'task_type'):
            info['task_type'] = learner.task_type
        elif 'classif' in info['learner_class'].lower():
            info['task_type'] = 'classification'
        elif 'regr' in info['learner_class'].lower():
            info['task_type'] = 'regression'
        elif 'cluster' in info['learner_class'].lower():
            info['task_type'] = 'clustering'
        
        self.tracker.log_params(info)
    
    def _log_gradients(self, model, step: int):
        """Loggea gradientes del modelo."""
        try:
            if 'torch' in str(type(model)):
                import torch
                
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[f"grad_{name}_mean"] = param.grad.mean().item()
                        gradients[f"grad_{name}_std"] = param.grad.std().item()
                
                if gradients:
                    self.tracker.log_metrics(gradients, step=step)
        except Exception as e:
            logger.debug(f"Could not log gradients: {e}")
    
    def _log_weights(self, model, step: int):
        """Loggea pesos del modelo."""
        try:
            if 'torch' in str(type(model)):
                import torch
                
                weights = {}
                for name, param in model.named_parameters():
                    weights[f"weight_{name}_mean"] = param.data.mean().item()
                    weights[f"weight_{name}_std"] = param.data.std().item()
                
                if weights:
                    self.tracker.log_metrics(weights, step=step)
        except Exception as e:
            logger.debug(f"Could not log weights: {e}")
    
    def _log_confusion_matrix(self, predictions):
        """Loggea matriz de confusión."""
        try:
            if hasattr(predictions, 'confusion_matrix'):
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                cm = predictions.confusion_matrix()
                
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                self.tracker.log_figure(fig, 'confusion_matrix')
                plt.close(fig)
        except Exception as e:
            logger.debug(f"Could not log confusion matrix: {e}")
    
    def _log_feature_importance(self, learner):
        """Loggea importancia de features."""
        try:
            if hasattr(learner, 'feature_importances_'):
                import matplotlib.pyplot as plt
                import numpy as np
                
                importances = learner.feature_importances_
                indices = np.argsort(importances)[::-1][:20]  # Top 20
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(indices)), importances[indices])
                ax.set_title('Feature Importances')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Importance')
                
                self.tracker.log_figure(fig, 'feature_importances')
                plt.close(fig)
        except Exception as e:
            logger.debug(f"Could not log feature importance: {e}")